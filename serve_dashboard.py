#!/usr/bin/env python3
"""
Simple HTTP server to serve the bridge criticality dashboard.
This solves CORS issues when opening HTML files directly.

Usage:
    python serve_dashboard.py
    
The dashboard opens in your browser and is only reachable from this computer.
"""

import errno
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 8000

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add headers to allow local file access
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate')
        return super().end_headers()

if __name__ == '__main__':
    os.chdir(Path(__file__).parent)
    
    try:
        # Bind to this computer only: the served folder can contain your raw data.
        with socketserver.TCPServer(("127.0.0.1", PORT), MyHTTPRequestHandler) as httpd:
            url = f"http://localhost:{PORT}/bridge_criticality_interactive_dashboard.html"
            print(f"Bridge Health Intelligence Dashboard")
            print(f"Server : http://localhost:{PORT}")
            print(f"Open   : {url}")
            print(f"Data   : {Path(__file__).parent / 'dashboard_data.json'}")
            print(f"Run    : python prepare_dashboard_data.py  (to regenerate data)")
            print(f"Stop   : Ctrl+C")
            print()
            
            # Try to open in browser
            try:
                webbrowser.open(url)
                print("Opening browser...")
            except:
                print(f"Please open browser and go to: {url}")
            
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n✓ Server stopped")
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, 10048):  # 10048 is the Windows socket error
            print(f"❌ Port {PORT} is already in use")
            print(f"   Stop the other server, or change PORT in this file")
        else:
            print(f"❌ Error: {e}")
