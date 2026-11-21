"""테스트용 가짜 llama-server: /health가 잠시 503을 돌려준 뒤 200이 된다."""
import argparse
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="127.0.0.1")
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--fail", default="")
ap.add_argument("--load-seconds", type=float, default=0.8)
a, _ = ap.parse_known_args()

print("main: loading model", flush=True)
if a.fail:
    print(a.fail, flush=True)
    sys.exit(1)

started = time.time()


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        ready = time.time() - started > a.load_seconds
        self.send_response(200 if ready else 503)
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}' if ready else b'{"status":"loading"}')

    def log_message(self, *args):
        pass


srv = HTTPServer((a.host, a.port), H)
print(f"main: server is listening on http://{a.host}:{a.port}", flush=True)


def speed():
    time.sleep(a.load_seconds + 0.2)
    print("       eval time =     100.00 ms /    10 tokens (   10.00 ms per token,    42.50 tokens per second)", flush=True)


threading.Thread(target=speed, daemon=True).start()
srv.serve_forever()
