
import os
import sys
import json
import tempfile
import threading
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import cgi

# Import your existing pipeline
from parser import convert_to_markdown
from extractor import extract_form_data
from mapper import transform_to_client_dict

# Auto-reload on file changes
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("  Install 'watchdog' for auto-reload: pip install watchdog")


class FormConverterHandler(SimpleHTTPRequestHandler):
    """HTTP handler with API endpoints for document conversion."""
    
    def __init__(self, *args, **kwargs):
        # Serve from frontend directory
        super().__init__(*args, directory="frontend", **kwargs)
    
    def do_POST(self):
        """Handle POST requests for file upload."""
        parsed_url = urlparse(self.path)
        if parsed_url.path == '/api/convert':
            # Parse query params for format selection
            query_params = parse_qs(parsed_url.query)
            # Default to client format (UI-ready)
            output_format = query_params.get('format', ['client'])[0]
            self.handle_convert(output_format=output_format)
        else:
            self.send_error(404, "Not Found")
    
    def handle_convert(self, output_format: str = "internal"):
        """
        Convert uploaded document to JSON.
        
        Args:
            output_format: 'internal' (default) or 'client' for UI-ready format
        """
        try:
            # Parse multipart form data
            content_type = self.headers.get('Content-Type', '')
            if 'multipart/form-data' not in content_type:
                self.send_error(400, "Expected multipart/form-data")
                return
            
            # Parse the form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    'REQUEST_METHOD': 'POST',
                    'CONTENT_TYPE': content_type
                }
            )
            
            # Get uploaded file
            file_item = form['file']
            if not file_item.file:
                self.send_error(400, "No file uploaded")
                return
            
            filename = file_item.filename
            file_data = file_item.file.read()
            
            # Save to temp file
            suffix = Path(filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_data)
                tmp_path = tmp.name
            
            try:
                # Run your existing pipeline
                print(f"Converting: {filename}")
                
                # Step 1: Parse to markdown
                markdown = convert_to_markdown(tmp_path)
                
                # Step 2: Extract structured data
                form_data = extract_form_data(markdown)
                
                # Convert to dict based on requested format
                if output_format == "client":
                    result = transform_to_client_dict(form_data)
                    print(f"  → Using client format (UI-ready)")
                else:
                    result = form_data.model_dump()
                    print(f"  → Using internal format")
                
                # Send response
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result, indent=2).encode())
                
                print(f"✓ Converted successfully: {filename}")
                
            finally:
                # Clean up temp file
                os.unlink(tmp_path)
                
        except Exception as e:
            print(f"✗ Error: {e}")
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
    
    def log_message(self, format, *args):
        """Custom logging."""
        if args and isinstance(args[0], str) and '/api/' in args[0]:
            print(f"API: {args[0]}")


class ReloadHandler(FileSystemEventHandler):
    """Watch for Python file changes and trigger reload."""
    
    def __init__(self, restart_callback):
        self.restart_callback = restart_callback
        self.last_modified = {}
        
    def on_modified(self, event):
        if event.src_path.endswith('.py'):
            # Debounce: only reload if file hasn't been modified in last second
            import time
            now = time.time()
            if event.src_path not in self.last_modified or now - self.last_modified[event.src_path] > 1:
                self.last_modified[event.src_path] = now
                print(f"\n Detected change in {Path(event.src_path).name}")
                print(" Restarting server...")
                self.restart_callback()


def run_server(port=8000, auto_reload=True):
    """Start the development server with optional auto-reload."""
    server = HTTPServer(('localhost', port), FormConverterHandler)
    
    reload_msg = "Auto-reload: ENABLED" if auto_reload and WATCHDOG_AVAILABLE else "Auto-reload: DISABLED"
    
    print(f"""
╔══════════════════════════════════════════════════╗
║         Form Converter - Dev Server              ║
╠══════════════════════════════════════════════════╣
║  http://localhost:{port}                          ║
║  {reload_msg:<48} ║
║                                                  ║
║  • Upload PDF/DOCX to convert                    ║
║  • Or paste JSON directly to preview             ║
║                                                  ║
║  Press Ctrl+C to stop                            ║
╚══════════════════════════════════════════════════╝
    """)
    
    observer = None
    
    if auto_reload and WATCHDOG_AVAILABLE:
        # Set up file watcher
        def restart():
            print("⏹  Stopping server...")
            server.shutdown()
            observer.stop()
            observer.join()
            # Restart process
            print(" Restarting...")
            os.execv(sys.executable, [sys.executable] + sys.argv)
        
        observer = Observer()
        event_handler = ReloadHandler(restart)
        # Watch current directory for .py file changes
        observer.schedule(event_handler, path='.', recursive=False)
        observer.start()
        print(" Watching for file changes...")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Server stopped.")
        if observer:
            observer.stop()
            observer.join()
        server.shutdown()


if __name__ == '__main__':
    run_server()

