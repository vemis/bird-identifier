"""Download bird recordings and their metadata from the xeno-canto API v3.

Usage:
    .venv/bin/python experiments/xeno_canto_download.py "Turdus merula" --max 50
    .venv/bin/python experiments/xeno_canto_download.py --species-file experiments/species_central_europe_50.txt

Reads the API key from XC_API_KEY (environment or the gitignored .env file).
Audio goes to <out>/audio/XC<id>.<ext>, metadata is appended to <out>/metadata.csv.
Recordings already present are skipped, so an interrupted run can be restarted.

The API lists newest uploads first, so taking the first N results would let one
recent uploader dominate a species. Instead all matching recordings are fetched,
shuffled (with a fixed seed), and at most --per-recordist are taken from each
recordist; quality A is preferred, B fills species with too few A recordings.
The cap is only relaxed when a species has too few recordists to reach --max.
"""

import argparse
import csv
import os
import random
import time
from collections import Counter
from pathlib import Path

import requests

API_URL = "https://xeno-canto.org/api/3/recordings"
REPO_ROOT = Path(__file__).resolve().parent.parent
PAUSE_S = 1.0  # be polite: one request at a time with a short pause

FIELDS = ["id", "gen", "sp", "ssp", "en", "type", "sex", "stage", "q", "length",
          "date", "uploaded", "rec", "cnt", "loc", "lat", "lon", "also", "lic",
          "url", "file_name"]


def api_key():
    if "XC_API_KEY" in os.environ:
        return os.environ["XC_API_KEY"]
    env = REPO_ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("XC_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise SystemExit("XC_API_KEY not set (environment or .env)")


def get(session, url, params, timeout, attempts=5):
    """GET with retries: the server is sometimes very slow or briefly fails."""
    for attempt in range(1, attempts + 1):
        try:
            r = session.get(url, params=params, timeout=timeout)
            r.raise_for_status()
            return r
        except requests.RequestException as e:
            if attempt == attempts:
                raise
            wait = 30 * attempt
            print(f"  {e.__class__.__name__}: {e}; retry {attempt}/{attempts - 1} in {wait} s")
            time.sleep(wait)


def search(session, query, key):
    """Return all recording dicts matching the query, following pagination."""
    recs, page = [], 1
    while True:
        r = get(session, API_URL, {"query": query, "key": key, "page": page, "per_page": 500}, timeout=120)
        data = r.json()
        recs.extend(data["recordings"])
        if page >= int(data["numPages"]):
            return recs
        page += 1
        time.sleep(PAUSE_S)


def select(candidates, n, per_recordist, already):
    """Pick up to n recordings, at most per_recordist per recordist where possible.

    candidates are in preference order; already maps recordist -> count of
    recordings of this species we already have.
    """
    counts = Counter(already)
    chosen, rest = [], []
    for rec in candidates:
        if len(chosen) >= n:
            break
        if counts[rec["rec"]] < per_recordist:
            chosen.append(rec)
            counts[rec["rec"]] += 1
        else:
            rest.append(rec)
    # Too few distinct recordists: fill up with the least-represented ones first.
    rest.sort(key=lambda rec: counts[rec["rec"]])
    for rec in rest:
        if len(chosen) >= n:
            break
        chosen.append(rec)
        counts[rec["rec"]] += 1
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("species", nargs="*", help='scientific names, e.g. "Turdus merula"')
    p.add_argument("--species-file", type=Path, help="file with one scientific name per line (# comments)")
    p.add_argument("--max", type=int, default=50, help="recordings per species (default: %(default)s)")
    p.add_argument("--per-recordist", type=int, default=3, help="soft cap per recordist and species (default: %(default)s)")
    p.add_argument("--filters", default="area:europe len:5-60",
                   help="extra xeno-canto search tags, quality is added per tier (default: %(default)s)")
    p.add_argument("--qualities", default="A,B", help="quality tiers in order of preference (default: %(default)s)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=REPO_ROOT / "datasets" / "xeno-canto")
    args = p.parse_args()

    species = list(args.species)
    if args.species_file:
        species += [s.strip() for s in args.species_file.read_text().splitlines()
                    if s.strip() and not s.startswith("#")]
    if not species:
        p.error("give species names or --species-file")

    key = api_key()
    audio_dir = args.out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    meta_path = args.out / "metadata.csv"
    existing = []
    if meta_path.exists():
        with meta_path.open(newline="") as f:
            existing = list(csv.DictReader(f))
    known = {row["id"] for row in existing}

    session = requests.Session()
    rng = random.Random(args.seed)
    with meta_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if not existing:
            writer.writeheader()

        for name in species:
            gen, sp = name.split()[:2]
            have = [row for row in existing if row["gen"] == gen and row["sp"] == sp]
            missing = args.max - len(have)
            if missing <= 0:
                print(f"{name}: already {len(have)}, skipping")
                continue

            candidates = []
            for q in args.qualities.split(","):
                # Restricted species have "file" redacted and cannot be downloaded.
                tier = [rec for rec in search(session, f'sp:"{name}" {args.filters} q:{q}', key)
                        if rec["id"] not in known and rec.get("file", "").startswith("http")]
                rng.shuffle(tier)
                candidates += tier
                time.sleep(PAUSE_S)
            chosen = select(candidates, missing, args.per_recordist, Counter(row["rec"] for row in have))
            print(f"{name}: {len(candidates)} candidates, downloading {len(chosen)}"
                  f" from {len({rec['rec'] for rec in chosen})} recordists")

            for rec in chosen:
                ext = Path(rec["file-name"]).suffix.lower() or ".mp3"
                target = audio_dir / f"XC{rec['id']}{ext}"
                if not target.exists():
                    time.sleep(PAUSE_S)
                    r = get(session, rec["file"], {"key": key}, timeout=180)
                    tmp = target.with_suffix(target.suffix + ".part")
                    tmp.write_bytes(r.content)
                    tmp.rename(target)
                row = {k: rec.get(k, "") for k in FIELDS}
                row["also"] = ";".join(rec.get("also") or [])
                row["file_name"] = target.name
                writer.writerow(row)
                f.flush()
                known.add(rec["id"])
                existing.append(row)

    print(f"done -> {args.out}")


if __name__ == "__main__":
    main()
