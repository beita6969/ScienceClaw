import json, sys
p = sys.argv[1]
rows = [json.loads(l) for l in open(p)]
print("FILE", p, "steps", len(rows))
for r in rows:
    a = r["action"]
    pl = a.get("payload", {})
    t = a.get("type")
    node = pl.get("node") or {}
    code = node.get("code")
    fb = r.get("feedback") or {}
    recs = fb.get("records") or {}
    print(f"--- step {r['step']} type={t} wall={r.get('wall_s'):.1f} usage={r.get('policy_usage')} parse_error={r.get('parse_error')}")
    print("  thought:", (a.get("thought") or r.get("thought") or "")[:400])
    if node:
        print("  node:", node.get("id"), node.get("kind"), node.get("ref"), "wire:", json.dumps(pl.get("wire"))[:200])
    if code:
        print("  code:\n" + "\n".join("    " + x for x in code.splitlines()[:40]))
    if t not in ("add_node",):
        print("  payload:", json.dumps(pl)[:400])
    for nid, rec in recs.items():
        if rec.get("status") != "ok" or rec.get("error") or rec.get("stdout_tail"):
            print("  REC", nid, rec.get("status"), rec.get("wall_s"), (rec.get("error") or "")[:300], (rec.get("stdout_tail") or "")[-300:])
        else:
            out = rec.get("outputs_summary") or {}
            small = {k: v for k, v in out.items() if k in ("dev_las", "dev_uas", "dev_reference_las")}
            if small:
                print("  REC", nid, "wall", rec.get("wall_s"), json.dumps(small)[:300])
            else:
                print("  REC", nid, "ok wall", round(rec.get("wall_s") or 0, 2))
    print("  action_error:", fb.get("action_error"), "valid_err:", fb.get("validation_errors"))
