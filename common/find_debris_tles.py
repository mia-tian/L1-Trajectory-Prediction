#!/usr/bin/env python3
"""
Filter TLE records from one or more TLE catalog files by estimated altitude
and eccentricity, and group them by NORAD ID into per-NORAD output files.

This script parses TLE fields directly (no SGP4/propagation). It converts
mean motion (revs/day) to semi-major axis via Kepler's third law to
approximate mean altitude.
"""
import os
import csv
import math
import datetime
import argparse
from collections import defaultdict

START_FULL = datetime.datetime(2022, 1, 4)
END_FULL = datetime.datetime(2025, 12, 28)
MIN_TLES_FULL = 3000


def is_passive_name(name: str) -> bool:
    """Heuristic check whether a TLE name indicates debris/rocket-body/fragment.

    Normalizes common variants and checks against an expanded keyword list.
    """
    if not name:
        return False
    s = name.upper().strip()
    # normalize punctuation/spacing
    for ch in ('-', '/', '_', '\\'):
        s = s.replace(ch, ' ')
    s = ' '.join(s.split())
    kws = (
        'DEB', 'DEBRIS', 'FRAGMENT', 'FRAG', 'FRAGM',
        'R B', 'RB', 'R/B', 'ROCKET', 'ROCKET BODY', 'ROCKET-BODY',
        'RKT', 'RBODY', 'PAYLOAD FRAGMENT', 'PAYLOAD',
    )
    for kw in kws:
        if kw in s:
            return True
    return False


def parse_tle_stream(file_obj):
    """Generator yielding (name_or_none, line1, line2) from an open file object.
    Handles 2-line and 3-line (name) records in a streaming fashion.
    """
    it = iter(file_obj)
    for raw in it:
        l = raw.rstrip('\n')
        if not l.strip():
            continue

        # Case: name line followed by line1 and line2
        if l and not l.lstrip().startswith('1 '):
            try:
                nxt1 = next(it)
            except StopIteration:
                break
            try:
                nxt2 = next(it)
            except StopIteration:
                break

            if nxt1.lstrip().startswith('1 ') and nxt2.lstrip().startswith('2 '):
                name = l.strip()
                line1 = nxt1.rstrip('\n')
                line2 = nxt2.rstrip('\n')
                yield name, line1, line2
                continue
            else:
                if nxt1.lstrip().startswith('1 '):
                    if nxt2.lstrip().startswith('2 '):
                        yield None, nxt1.rstrip('\n'), nxt2.rstrip('\n')
                        continue
                    else:
                        continue
                else:
                    continue

        # Case: line1 followed by line2 (no name)
        if l.lstrip().startswith('1 '):
            try:
                nxt = next(it)
            except StopIteration:
                break
            if nxt.lstrip().startswith('2 '):
                yield None, l.rstrip('\n'), nxt.rstrip('\n')
                continue
            else:
                continue

        continue


def parse_fields_from_tle(line1, line2):
    """Extract NORAD (str), eccentricity (float), mean_motion (rev/day), and epoch (datetime) from TLE lines.

    Uses fixed-column slicing per TLE format.
    """
    # Ensure lines are long enough by padding
    l1 = line1.ljust(80)
    l2 = line2.ljust(80)

    norad = l1[2:7].strip()
    # Normalize NORAD: convert digit IDs to non-padded integer string
    if norad.isdigit():
        try:
            norad = str(int(norad))
        except Exception:
            norad = norad.strip()
    else:
        norad = norad.strip()

    # line2 fixed columns (1-indexed in spec):
    # inclination: cols 9-16 -> [8:16]
    # raan: cols 18-25 -> [17:25]
    # eccentricity (no leading decimal): cols 27-33 -> [26:33]
    # argp: cols 35-42 -> [34:42]
    # mean anomaly: cols 44-51 -> [43:51]
    # mean motion: cols 53-63 -> [52:63]
    try:
        ecc_str = l2[26:33].strip()
        mm_str = l2[52:63].strip()
    except Exception:
        return norad, None, None, None

    try:
        ecc = float('0.' + ecc_str) if ecc_str else 0.0
    except Exception:
        try:
            ecc = float(ecc_str)
        except Exception:
            ecc = None

    try:
        mm = float(mm_str) if mm_str else None
    except Exception:
        mm = None

    # parse epoch from line1 (cols 19-32 -> index [18:32])
    epoch = None
    try:
        epoch_str = l1[18:32].strip()
        # format: YYDDD.DDDDD... (two-digit year + day-of-year with fraction)
        if epoch_str:
            yy = int(epoch_str[0:2])
            doy = float(epoch_str[2:])
            year = 1900 + yy if yy >= 57 else 2000 + yy
            # convert day-of-year (1-based) to datetime
            day_int = int(math.floor(doy))
            frac = doy - day_int
            epoch = datetime.datetime(year, 1, 1) + datetime.timedelta(days=day_int - 1, seconds=round(frac * 86400))
    except Exception:
        epoch = None

    return norad, ecc, mm, epoch


def mean_motion_to_altitude_km(mm_rev_per_day):
    """Convert mean motion (rev/day) to approximate mean altitude in km.

    Uses mu = 398600.4418 km^3/s^2 and Earth radius 6378.137 km.
    """
    if mm_rev_per_day is None:
        return None
    # convert to rad/s
    n_rad = mm_rev_per_day * 2.0 * math.pi / 86400.0
    if n_rad <= 0:
        return None
    mu = 398600.4418
    a_km = (mu / (n_rad ** 2)) ** (1.0 / 3.0)
    r_earth = 6378.137
    alt_km = a_km - r_earth
    return alt_km


def load_satcat_names(satcat_path='satcat.csv'):
    """Load a {norad: object_name} mapping from a Space-Track satcat CSV export.

    Expects columns OBJECT_NAME and NORAD_CAT_ID (as in a satcat.csv download).
    Returns an empty dict if the file is missing or unreadable.
    """
    names = {}
    if not satcat_path or not os.path.exists(satcat_path):
        return names
    with open(satcat_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            norad = (row.get('NORAD_CAT_ID') or '').strip()
            if not norad:
                continue
            if norad.isdigit():
                norad = str(int(norad))
            names[norad] = (row.get('OBJECT_NAME') or '').strip()
    return names


def scan_metadata(input_paths):
    """Scan one or more TLE files and aggregate per-NORAD stats.

    Returns stats dict keyed by NORAD with ntles, min_epoch, max_epoch,
    sum_ecc, sum_alt, example_name.
    """
    stats = {}
    for input_path in input_paths:
        with open(input_path, 'r') as f:
            for name, l1, l2 in parse_tle_stream(f):
                norad, ecc, mm, epoch = parse_fields_from_tle(l1, l2)
                if not norad or ecc is None or mm is None or epoch is None:
                    continue
                alt = mean_motion_to_altitude_km(mm)
                if alt is None:
                    continue

                st = stats.get(norad)
                if st is None:
                    st = {
                        'ntles': 0,
                        'min_epoch': epoch,
                        'max_epoch': epoch,
                        'sum_ecc': 0.0,
                        'sum_alt': 0.0,
                        'example_name': name if name else '',
                    }
                    stats[norad] = st

                st['ntles'] += 1
                if epoch < st['min_epoch']:
                    st['min_epoch'] = epoch
                if epoch > st['max_epoch']:
                    st['max_epoch'] = epoch
                st['sum_ecc'] += ecc
                st['sum_alt'] += alt

    return stats


def decide_accepted(stats, ecc_threshold, max_alt_km, require_passive, require_full):
    """Decide which NORADs pass the altitude/eccentricity/passive/full-coverage criteria.

    Fills 'mean_ecc' and 'mean_alt_km' into each stats entry as a side effect.
    Returns the accepted set of NORAD ids.
    """
    accepted = set()
    for norad, st in stats.items():
        mean_ecc = st['sum_ecc'] / st['ntles']
        mean_alt = st['sum_alt'] / st['ntles']
        st['mean_ecc'] = mean_ecc
        st['mean_alt_km'] = mean_alt

        if mean_alt >= max_alt_km or mean_ecc >= ecc_threshold:
            continue

        if require_full:
            # strict: require passive name, minimum TLE count, and full coverage span
            if not is_passive_name(st.get('example_name')):
                continue
            if st['ntles'] < MIN_TLES_FULL or st['min_epoch'] > START_FULL or st['max_epoch'] < END_FULL:
                continue
            accepted.add(norad)
        else:
            if require_passive and not is_passive_name(st.get('example_name')):
                continue
            accepted.add(norad)

    return accepted


def write_matches(input_paths, outdir, accepted_set, overwrite=True):
    """Stream through the input files and append matching TLEs to per-NORAD files.

    Returns list of (norad, ntles_written).
    """
    os.makedirs(outdir, exist_ok=True)
    counts = {}
    for input_path in input_paths:
        with open(input_path, 'r') as f:
            for name, l1, l2 in parse_tle_stream(f):
                norad, ecc, mm, epoch = parse_fields_from_tle(l1, l2)
                if not norad or norad not in accepted_set:
                    continue

                path = os.path.join(outdir, f"{norad}.txt")
                # If overwrite==True and file doesn't exist, create new; if overwrite==False and exists, skip writing
                if not overwrite and os.path.exists(path):
                    counts[norad] = counts.get(norad, 0) + 1
                    continue

                with open(path, 'a') as out:
                    if name:
                        out.write(name.rstrip('\n') + '\n')
                    out.write(l1.rstrip('\n') + '\n')
                    out.write(l2.rstrip('\n') + '\n')

                counts[norad] = counts.get(norad, 0) + 1

    written = sorted([(n, c) for n, c in counts.items()], key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    return written


def main():
    p = argparse.ArgumentParser(description='Filter TLEs and group by NORAD (no propagation).')
    p.add_argument('--input', '-i', nargs='+', default=['tle2022.txt', 'tle2023.txt', 'tle2024.txt', 'tle2025.txt'],
                    help='Input TLE file(s)')
    p.add_argument('--outdir', '-o', default='deb_tles_all', help='Output directory')
    p.add_argument('--ecc-threshold', type=float, default=0.001, help='Max eccentricity (default 0.001)')
    p.add_argument('--max-alt-km', type=float, default=1000.0, help='Max mean altitude in km (default 1000)')
    p.add_argument('--require-passive', action='store_true', help='Require passive keyword in name')
    p.add_argument('--require-full', action='store_true',
                    help=f'Require >= {MIN_TLES_FULL} TLEs spanning {START_FULL.date()} through {END_FULL.date()}')
    p.add_argument('--satcat', default='satcat.csv', help='Path to satcat CSV for NORAD-to-name lookup')
    p.add_argument('--no-clobber', action='store_true', help="Don't overwrite existing files")
    args = p.parse_args()

    missing = [path for path in args.input if not os.path.exists(path)]
    if missing:
        print(f"Input file(s) not found: {', '.join(missing)}")
        return

    # Pass 1: scan all input files for lightweight per-NORAD stats
    stats = scan_metadata(args.input)

    # Fill in names missing from the TLE data itself using the satcat CSV lookup
    satcat_names = load_satcat_names(args.satcat)
    for norad, st in stats.items():
        if not st.get('example_name') and satcat_names.get(norad):
            st['example_name'] = satcat_names[norad]

    accepted = decide_accepted(stats, args.ecc_threshold, args.max_alt_km,
                                args.require_passive, args.require_full)

    # Pass 2: stream through the input files again and write matching TLEs
    written = write_matches(args.input, args.outdir, accepted, overwrite=not args.no_clobber)

    # Print summary
    print(f"Saved {len(written)} objects to '{args.outdir}':")
    for norad, ntles in sorted(written, key=lambda x: int(x[0]) if x[0].isdigit() else x[0]):
        st = stats.get(norad, {})
        name = st.get('example_name') or ''
        alt = st.get('mean_alt_km')
        ecc = st.get('mean_ecc')
        alt_str = f"{alt:.1f}" if alt is not None else "N/A"
        ecc_str = f"{ecc:.5f}" if ecc is not None else "N/A"
        print(f"- {norad}: {ntles} TLEs, name='{name}', mean_alt_km={alt_str}, mean_ecc={ecc_str}")


if __name__ == '__main__':
    main()
