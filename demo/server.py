"""
RSNA Knee MRI CADx - Interactive Presentation Demo Server
Zero external dependencies (uses standard library http.server).
Automatically opens the workstation UI in the default web browser.
"""

import os
import sys
import socket
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler

DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PORT = 8501

class PACSHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DEMO_DIR, **kwargs)

    def log_message(self, format, *args):
        # Clean custom access logger
        sys.stderr.write(f"[PACS-UI] {args[0]} -> {args[1]}\n")

def find_open_port(starting_port=8501):
    port = starting_port
    while port < starting_port + 50:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
            port += 1
    return starting_port

def run_server():
    port = find_open_port(DEFAULT_PORT)
    server_address = ('localhost', port)
    httpd = HTTPServer(server_address, PACSHandler)
    url = f"http://localhost:{port}/index.html"

    print("=" * 70)
    print("  RSNA KNEE MRI CADx - CLINICAL DIAGNOSTIC WORKSTATION")
    print("=" * 70)
    print(f"  [+] Workstation Demo Root : {DEMO_DIR}")
    print(f"  [+] Serving live UI at    : {url}")
    print(f"  [+] Pre-loaded Cases      : 10 Studies (PNG, Zero Disk Extraction)")
    print(f"  [+] Status                : ONLINE & READY FOR JUDGES PRESENTATION")
    print("=" * 70)
    print("  Launching default browser... (Press Ctrl+C to stop server)\n")

    # Open browser automatically
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"  [!] Note: Please open {url} manually in your browser: {e}")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  [x] Workstation server stopped by user.")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
