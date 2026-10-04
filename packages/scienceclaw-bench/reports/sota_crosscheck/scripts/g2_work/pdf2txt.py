import sys
from pypdf import PdfReader
r=PdfReader(sys.argv[1]); out=[]
for i,p in enumerate(r.pages):
    try: t=p.extract_text()
    except Exception as e: t=f"ERR {e}"
    out.append(f"\n=====PAGE {i+1}=====\n"+t)
open(sys.argv[2],'w').write(''.join(out)); print(len(r.pages),'pages')
