import http.server
import os
import sys
import socketserver

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
DIR = os.path.dirname(os.path.abspath(__file__))

class SmartHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIR, **kwargs)

    def do_GET(self):
        path = self.translate_path(self.path)
        if os.path.exists(path):
            return super().do_GET()
        if '.' not in self.path.split('/')[-1]:
            html_path = self.path.rstrip('/') + '.html'
            html_full = self.translate_path(html_path)
            if os.path.exists(html_full):
                self.path = html_path
                return super().do_GET()
        return super().do_GET()

    # Disable keep-alive to avoid CLOSE_WAIT pileup
    def close_connection(self):
        return True  # force Connection: close

class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

if __name__ == '__main__':
    server = ThreadingServer(('0.0.0.0', PORT), SmartHandler)
    print(f'Serving on http://localhost:{PORT}')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
