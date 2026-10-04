import json,sys,re
for D in sys.argv[1:]:
    rows=[json.loads(l) for l in open(D)]
    print("=====",D,len(rows),"steps")
    for r in rows:
        a=r['action']; fb=r['feedback'] or {}
        p=a.get('payload') or {}
        n=p.get('node') or {}
        code=n.get('code') or (p.get('patch') or {}).get('code')
        line=f"s{r['step']} {a.get('type')} {fb.get('action_desc')}"
        m=re.findall(r'dev_sdr: float unit=dB value=([-\d.]+)',fb.get('text') or '')
        if m: line+=f" | dev_sdr={m[-1]}"
        if not fb.get('action_ok'): line+=f" | ERR {str(fb.get('action_error'))[:150]}"
        # node errors
        for nid,rec in (fb.get('records') or {}).items():
            if rec.get('status') not in ('ok',) : line+=f" | node {nid} {rec.get('status')} {str(rec.get('error'))[:120]}"
        print(line)
        if code:
            body=[x for x in code.splitlines() if x.strip()]
            print("     code(%d lines): "%len(body)+" || ".join(x.strip() for x in body[:25])[:700])
        th=a.get('thought')
        if th: print("     thought:",th[:260])
