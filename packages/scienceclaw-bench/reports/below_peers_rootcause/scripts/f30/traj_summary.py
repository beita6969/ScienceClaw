"""Summarise an agent trajectory (steps, action type, thought, code head, phenoseg calls, stdout tail, tokens)."""
import json, sys, re
p = sys.argv[1]
maxcode = int(sys.argv[2]) if len(sys.argv) > 2 else 600
tot = 0
for l in open(p):
    r = json.loads(l)
    a = r.get('action') or {}
    pl = a.get('payload') or {}
    pu = r.get('policy_usage') or {}
    fb = r.get('feedback') or {}
    print(f"=== step {r['step']} {a.get('type')} ok={fb.get('action_ok')} err={str(fb.get('action_error'))[:160]} wall={r.get('wall_s'):.0f}s tok={pu.get('total_tokens') or pu.get('prompt_tokens')}/{pu.get('completion_tokens')}")
    print("thought:", (a.get('thought') or '')[:400])
    node = pl.get('node') or {}
    if node.get('kind') == 'code':
        print("CODE:", node.get('code', '')[:maxcode])
    ce = (pl.get('patch') or {}).get('code_edit')
    if ce:
        print("EDIT find:", ce.get('find', '')[:200], "\n     repl:", ce.get('replace', '')[:maxcode])
    if (pl.get('patch') or {}).get('code'):
        print("CODE(patch):", pl['patch']['code'][:maxcode])
    recs = fb.get('records') or {}
    for nid, rec in recs.items():
        if rec.get('cached'): continue
        print(f"  node {nid} status={rec.get('status')} wall={rec.get('wall_s'):.1f}s err={str(rec.get('error'))[:300]} stdout={str(rec.get('stdout_tail'))[-400:]!r}")
    for k in ('evaluation','eval','dev','evidence'):
        if k in fb: print("  ", k, str(fb[k])[:400])
