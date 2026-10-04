import argparse
import glob
import os
import sys
import time


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Check Leonardo adapter availability and the frozen split plan before a batch."
    )
    ap.add_argument("disciplines", help="comma-separated FoR codes")
    ap.add_argument("--strict", action="store_true",
                    help="fail if every requested discipline/split has the requested episode count")
    ap.add_argument("--splits", default="src,val,id,ood",
                    help="comma-separated splits required in --strict mode")
    ap.add_argument("--n", type=int, default=None,
                    help="episodes required per requested split in --strict mode")
    args = ap.parse_args()
    args.disciplines = [x.strip() for x in args.disciplines.split(",") if x.strip()]
    args.splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    valid = {"src", "val", "id", "ood"}
    bad = sorted(set(args.splits) - valid)
    if not args.disciplines:
        ap.error("disciplines must not be empty")
    if not args.splits:
        ap.error("--splits must not be empty")
    if bad:
        ap.error(f"unknown split(s): {', '.join(bad)}")
    if args.n is not None and args.n < 1:
        ap.error("--n must be >= 1")
    if args.n is not None and not args.strict:
        ap.error("--n is only valid with --strict")
    return args


args = _args()

# The login node has no system ffmpeg.  Keep the preflight check identical to
# run_batch_leo.sh by selecting the pinned static binary shipped with imageio-ffmpeg.
if not os.environ.get("SCIENCECLAW_FFMPEG"):
    _ffmpeg = glob.glob(
        "/leonardo_scratch/fast/AIFAC_F02_774/rqian000/envs/sc-run/"
        "lib64/python3.11/site-packages/imageio_ffmpeg/binaries/ffmpeg-linux-x86_64-*"
    )
    if _ffmpeg:
        os.environ["SCIENCECLAW_FFMPEG"] = _ffmpeg[0]

sys.path.insert(0, "/leonardo_scratch/fast/AIFAC_F02_774/rqian000/scienceclaw")
from scienceclaw.config import load_config
from scienceclaw.bench.splits import load_adapters, SplitPlan
cfg = load_config("/leonardo_scratch/fast/AIFAC_F02_774/rqian000/scienceclaw/configs/toolon_leo.yaml")
cfg.bench.disciplines = args.disciplines
t = time.time()
ad = load_adapters(cfg.bench)
plan = SplitPlan.build(cfg.bench, ad)
for w in plan.warnings: print("WARN", w)
print("loaded", sorted(ad), f"{time.time()-t:.0f}s")
for sp in ("src", "val", "id", "ood"):
    print(sp, {c: len(v) for c, v in plan.episodes.get(sp, {}).items()})

if args.strict:
    errors = []
    plan_counts = {"src": int(cfg.bench.rounds), "val": int(cfg.bench.n_val),
                   "id": int(cfg.bench.n_id), "ood": int(cfg.bench.n_ood)}
    for code in args.disciplines:
        for split in args.splits:
            need = args.n if args.n is not None else plan_counts[split]
            got = len(plan.episodes.get(split, {}).get(code, []))
            if got < need:
                errors.append(f"{code}/{split}: have {got}, need {need}")
    if errors:
        for error in errors:
            print("ERROR", error, file=sys.stderr)
        raise SystemExit(2)
    print("strict preflight OK", {"disciplines": args.disciplines, "splits": args.splits,
                                  "episodes": args.n if args.n is not None else plan_counts})
