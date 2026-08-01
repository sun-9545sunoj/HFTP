"""
Dashboard Web Server Launcher.

Launches a lightweight local web server serving the human-friendly High-Frequency
Microstructure & Execution Backtesting Dashboard at http://localhost:8000.
"""

import http.server
import socketserver
import webbrowser
from pathlib import Path

PORT = 8000
DASHBOARD_DIR = Path(__file__).parent / "src" / "dashboard"


class DashboardHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)


def main():
    print("=========================================================")
    print("  High-Frequency Microstructure Web Dashboard Launcher   ")
    print("=========================================================\n")

    print(f"-> Serving Dashboard files from: {DASHBOARD_DIR.resolve()}")
    print(f"-> Dashboard URL: http://localhost:{PORT}\n")

    with socketserver.TCPServer(("", PORT), DashboardHTTPRequestHandler) as httpd:
        print(f"Server started cleanly! Open http://localhost:{PORT} in your web browser.")
        print("Press Ctrl+C to stop the dashboard server.\n")
        
        # Try to automatically open in default web browser
        try:
            webbrowser.open(f"http://localhost:{PORT}")
        except Exception:
            pass

        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nDashboard web server stopped.")


if __name__ == "__main__":
    main()
