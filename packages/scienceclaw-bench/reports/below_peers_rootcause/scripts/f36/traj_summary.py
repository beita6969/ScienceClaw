import json,sys
D=sys.argv[1]
rows=[json.loads(l) for l in open(D)]
for r in rows:
    a=r['action']; fb=r['feedback'] or {}
    t=a.get('type'); p=a.get('payload') or {}
    desc=fb.get('action_desc')
    dev=fb.get('dev')
    u=r.get('policy_usage') or {}
    print(f"--- step {r['step']} {t} | {desc} | ok={fb.get('action_ok')} err={str(fb.get('action_error'))[:200]}")
    print("   thought:", str(a.get('thought'))[:400])
    if t in('add_node','replace_node','set_code','update_node','patch_node') or 'node' in p:
        n=p.get('node') or p
        cfg=(n.get('config') or {}) if isinstance(n,dict) else {}
        code=cfg.get('code') if isinstance(cfg,dict) else None
        if code: print("   CODE:\n"+"\n".join("      "+x for x in code.splitlines()[:80]))
        else: print("   payload:", str(p)[:600])
    else:
        print("   payload:", str(p)[:600])
    print("   dev:", dev, "| wall", round(r.get('wall_s') or 0,1), "| tokens", u.get('prompt_tokens'), u.get('completion_tokens'))
    txt=fb.get('text') or ''
    print("   FB:", txt[-900:].replace("\n","\n      "))
