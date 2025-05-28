import http.server
import socketserver
import json
import asyncio
import cgi
from ..analysis.analysis_orchestrator import StockAnalysisOrchestrator
from urllib.parse import parse_qs, urlparse, unquote_plus
import logging

# Configure logging to be more readable
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize the orchestrator
orchestrator = StockAnalysisOrchestrator()

class StockAnalysisHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        """Override to provide cleaner logging"""
        if args and len(args) > 2 and "code 400" in args[0]:
            return  # Skip logging bad request binary data
        logger.info("%s - - %s" % (self.address_string(), format%args))

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            
            with open('src/web/templates/index.html', 'rb') as f:
                self.wfile.write(f.read())
        else:
            return super().do_GET()
            
    def do_POST(self):
        """Handle POST requests"""
        if self.path == '/analyze':
            # Parse multipart form data
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={'REQUEST_METHOD': 'POST',
                        'CONTENT_TYPE': self.headers['Content-Type']}
            )
            
            # Get stock code from form data
            stock_code = form.getvalue('stock_code', '').strip()
            logger.debug(f"Received stock code: '{stock_code}'")
            
            if not stock_code:
                logger.warning("Empty stock code received")
                self._send_json_response({'error': 'Stock code is required'}, 400)
                return
                
            try:
                # Log analysis request
                logger.info(f"Starting analysis for stock: {stock_code}")
                
                # Run analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(orchestrator.analyze_stock(stock_code, vix=15.5))
                
                # Format response
                analysis_result = {
                    'symbol': result['symbol'],
                    'overall_signal': result['overall_signal'],
                    'overall_confidence': f"{result['overall_confidence']:.1f}%",
                    'analysis': {}
                }
                
                # Format each strategy's results
                for strategy_name, strategy_result in result['strategy_results'].items():
                    analysis_result['analysis'][strategy_name] = {
                        'signal': strategy_result['signal'],
                        'confidence': f"{strategy_result['confidence']:.1f}%",
                        'reasoning': strategy_result['reasoning']
                    }
                    
                logger.info(f"Analysis completed for {stock_code}")
                self._send_json_response(analysis_result)
                
            except Exception as e:
                error_msg = f"Error analyzing stock {stock_code}: {str(e)}"
                logger.error(error_msg)
                self._send_json_response({'error': error_msg}, 500)
                
        else:
            self.send_error(404, "Path not found")
            
    def _send_json_response(self, data, status=200):
        """Helper method to send JSON responses"""
        self.send_response(status)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

def run_server(port=8000):
    """Run the web server"""
    with socketserver.TCPServer(("", port), StockAnalysisHandler) as httpd:
        print(f"Server running at http://localhost:{port}")
        httpd.serve_forever()

if __name__ == '__main__':
    run_server()
