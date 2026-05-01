#!/usr/bin/env python3
"""
Simple HTTP server to serve the bridge criticality dashboard.
This solves CORS issues when opening HTML files directly.

Usage:
    python serve_dashboard.py
    
Then open http://localhost:8000 in your browser.
"""

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
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
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
        if e.errno == 48:  # Port already in use
            print(f"❌ Port {PORT} is already in use")
            print(f"   Try: python serve_dashboard.py  (or change PORT variable)")
        else:
            print(f"❌ Error: {e}")
