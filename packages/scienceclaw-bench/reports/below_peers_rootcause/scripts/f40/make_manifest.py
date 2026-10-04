"""Manifest of every clip of the FoR40 delivery (probe clips of src/val/id + the 168 fit-support clips) with the byte range of its zip member
in the public Zenodo dev archives (record 10902294), so a GPU host can fetch exactly these members instead of the 2.2 GB of archives.

usage: python make_manifest.py <reconstructed_v3 dir> <out.json>
"""
import json
import sys
from pathlib import Path

root, out = Path(sys.argv[1]), sys.argv[2]
cat = {}
for p in (root / "archives/range_fragments").glob("*.complete-member-catalog.json"):
    cat[p.name.split(".")[0]] = {m["name"]: m for m in json.load(open(p))}
rec = json.load(open(root.parent / "official_record_10902294.json"))
url = {f["key"][:-4]: f["links"]["self"] for f in rec["files"]}
rows = []
for role in ("source", "val", "id"):
    for c in json.load(open(root / f"roles/{role}.json")):
        for r in c["probe_records"]:
            rows.append((role, r))
for r in json.load(open(root / "source-normal-support.json")):
    rows.append(("support", r))
man = []
for role, r in rows:
    arch = "dev_" + r["machine"]
    m = cat[arch][r["id"]]
    assert m["bytes"] == r["bytes"], (r["id"], m["bytes"], r["bytes"])
    man.append({"role": role, "id": r["id"], "public_id": r["public_id"], "machine": r["machine"], "domain": r.get("domain"), "label": r.get("label"),
                "url": url[arch], "offset": m["header_offset"], "csize": m["compressed_bytes"], "usize": m["bytes"], "crc32": m["crc32"],
                "method": m["compression"], "sha256": r["sha256"], "pcm_sha256": r["pcm_sha256"], "attributes": r.get("attributes")})
json.dump(man, open(out, "w"))
print(len(man), "clips,", len({m["id"] for m in man}), "distinct,", sum(m["csize"] for m in man) / 1e6, "MB compressed")
