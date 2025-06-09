import os
import sys
import asyncio
from flask import Flask, render_template, request, jsonify

# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.analysis.analysis_orchestrator import StockAnalysisOrchestrator
from src.agents.technical_agent import technical_analyst_agent
from src.agents.peter_lynch_agent import PeterLynchAgent
from src.agents.warren_buffett_agent import warren_buffett_agent
import asyncio
import logging


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the orchestrator
orchestrator = StockAnalysisOrchestrator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
async def analyze():
    try:
        stock_code = request.form.get('stock_code')
        if not stock_code:
            return jsonify({'error': 'Stock code is required'}), 400
            
        # Default VIX value (can be made dynamic later)
        vix = 15.5
        
        # Run analysis
        result = await orchestrator.analyze_stock(stock_code, vix)
        
        # Extract the relevant information for display
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
            
        return jsonify(analysis_result)
        
    except Exception as e:
        logger.error(f"Error analyzing stock {stock_code}: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)  # 添加host参数
