import os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer as HTTPServer
ROOT = "/root/sc-mig/in"
class H(BaseHTTPRequestHandler):
    def do_PUT(self):
        p = os.path.normpath(os.path.join(ROOT, self.path.lstrip("/")))
        if not p.startswith(ROOT + "/"):
            self.send_response(400); self.end_headers(); return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        n = int(self.headers.get("Content-Length", 0)); got = 0
        with open(p, "wb") as f:
            while got < n:
                b = self.rfile.read(min(1 << 20, n - got))
                if not b: break
                f.write(b); got += len(b)
        self.send_response(201 if got == n else 500); self.end_headers()
    def do_GET(self):
        p = os.path.normpath(os.path.join(ROOT, self.path.lstrip("/")))
        if not p.startswith(ROOT) or not os.path.isfile(p):
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("Content-Length", str(os.path.getsize(p))); self.end_headers()
        with open(p, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b: break
                self.wfile.write(b)
    def log_message(self, *a): pass
HTTPServer(("0.0.0.0", int(sys.argv[1])), H).serve_forever()
