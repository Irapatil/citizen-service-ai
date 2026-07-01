"""
Simple HTTP server for the CitizenAI frontend.
Serves index.html on http://localhost:3000

Usage:
    python frontend/serve.py
"""
import http.server
import os
import sys

PORT = 3000
DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def log_message(self, fmt, *args):
        print(f"  [frontend] {fmt % args}")


if __name__ == "__main__":
    os.chdir(DIR)
    print(f"  CitizenAI Frontend -> http://localhost:{PORT}")
    print(f"  Serving from: {DIR}")
    print("  Press Ctrl+C to stop.\n")
    with http.server.HTTPServer(("", PORT), Handler) as srv:
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\n  Frontend server stopped.")
            sys.exit(0)
