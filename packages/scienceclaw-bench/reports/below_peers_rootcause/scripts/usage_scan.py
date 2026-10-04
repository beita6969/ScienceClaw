import glob, json, os, re, sys, collections
roots = sys.argv[1:]
mods = collections.defaultdict(lambda: collections.defaultdict(int))   # code -> module -> n episodes using it
n_ep = collections.Counter(); z = collections.Counter()
for root in roots:
    for d in glob.glob(root + "/*/"):
        code = os.path.basename(d.rstrip("/")).split("-")[0]
        if not os.path.exists(d + "final_graph.json"): continue
        src = json.dumps(json.load(open(d + "final_graph.json"))).replace("\\n", "\n")
        used = set(re.findall(r"scilib(?:\.|\s+import\s+)([a-z_0-9]+)", src))
        used |= set(re.findall(r"from scilib import ([a-z_0-9, ]+)", src) and [m.strip() for m in ",".join(re.findall(r"from scilib import ([a-z_0-9, ]+)", src)).split(",")])
        n_ep[code] += 1
        for m in used: mods[code][m] += 1
for code in sorted(n_ep):
    print(code, n_ep[code], "episodes:", dict(sorted(mods[code].items())))
