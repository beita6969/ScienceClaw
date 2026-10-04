import glob, json, os, re, sys
root = sys.argv[1]
pat = {"pretrained": re.compile(r"pretrained\s*=\s*(?!0\.0\b|0\b|False|None)[0-9.TtRrUuEe\[\"']"), "chronos": re.compile(r"chronos|tsfm"), "embed": re.compile(r"embed\s*=\s*[\[\"']|audioenc")}
for d in sorted(glob.glob(root + "/*/")):
    n = os.path.basename(d.rstrip("/"))
    if not os.path.exists(d + "final_graph.json"): print(n, "no graph"); continue
    src = json.dumps(json.load(open(d + "final_graph.json"))).replace("\\n", "\n").replace('\\"', '"')
    r = json.load(open(d + "result.json")) if os.path.exists(d + "result.json") else {}
    hits = {k: bool(p.search(src)) for k, p in pat.items()}
    print(n, "z=%s" % r.get("z"), "steps=%s" % r.get("n_steps"), hits)
