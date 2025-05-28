# AI Investment Agent

A modular, agent-based stock analysis tool that collects data from AKShare and provides multi-strategy investment opinions using a local LLM (Qwen3 14B via Ollama on macOS).

## Features
- **Data Collection:** Fetches real-time and historical stock data from AKShare.
- **AKShare Data Fields:**
  - **Real-time Market Data:**
    - 最新价 (Latest Price)
    - 成交量 (Volume)
    - 最高 (High)
    - 最低 (Low)
    - 今开 (Open)
    - 昨收 (Previous Close)
    - 涨跌幅 (Change Percentage)
    - 总市值 (Market Cap)
  - **Historical Price Data:**
    - 收盘价 (Close Prices)
    - 成交量 (Volumes)
    - SMA 50/200 (Technical Indicators)
    - RSI 14 (Relative Strength Index)
    - MACD (Moving Average Convergence Divergence)
    - Volume MA (Volume Moving Average)
  - **Fundamental Metrics:**
    - 市盈率 (P/E Ratio)
    - 市净率 (P/B Ratio)
    - 营业收入同比增长率 (Revenue Growth)
    - 基本每股收益同比增长率 (EPS Growth)
    - 净资产收益率 (ROE)
    - 销售毛利率 (Profit Margin)
    - 资产负债率 (Debt Ratio)
- **Missing Data Fields (TODO):**
  - **Warren Buffett Strategy Metrics:**
    - 所有者收益 (Owner Earnings) - Not available from AKShare
    - 投资回报率 (ROIC) - Not available from AKShare
    - 现金转换周期 (Cash Conversion Cycle) - Not available from AKShare
    - 资本支出比率 (Capex Ratio) - Not available from AKShare
    - 行业集中度 (Market Concentration) - Not available from AKShare
    - 监管风险 (Regulatory Risk) - Not available from AKShare
    - 行业周期位置 (Industry Cycle Position) - Not available from AKShare
    - 竞争优势类型 (Moat Type) - Not available from AKShare
    - 管理层评分 (Management Score) - Not available from AKShare
    - 内部人所有权 (Insider Ownership) - Not available from AKShare
    - 回购效率 (Buyback Efficiency) - Not available from AKShare
    - 资本配置 (Capital Allocation) - Not available from AKShare
    - 行业增长率 (Industry Growth) - Not available from AKShare
    - 品牌价值 (Brand Value) - Not available from AKShare
    - 行业市净率 (Industry PB) - Not available from AKShare
  - **Peter Lynch Strategy Metrics:**
    - PEG比率 (PEG Ratio) - Inconsistent availability from AKShare
    - 盈利增长稳定性 (Earnings Stability) - Not available from AKShare
    - 现金流与盈利比 (Cash Flow vs. Earnings) - Not available from AKShare
    - 分析师预期 (Analyst Expectations) - Not available from AKShare
- **Multi-Agent Analysis:**
  - **Technical Analysis:** Short-term signals based on price, volume, and technical indicators.
  - **Peter Lynch Strategy:** Growth/value analysis for long-term investing.
  - **Warren Buffett Strategy:** Value investing with focus on moat, management, and financial health.
- **LLM Integration:** Uses a local Ollama server (Qwen3 14B) for natural language reasoning and summary.
- **Web Interface:** Provides a web-based user interface for easy access and visualization.

## Requirements
- Python 3.10+
- macOS (for local Ollama LLM integration)
- [Ollama](https://ollama.com/) with Qwen3 14B model installed and running
- See `requirements.txt` for Python dependencies

## Installation
```sh
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install required packages
pip install -r requirements.txt
```

## Usage
### Running the Web Application
1. **Start Ollama with Qwen3 14B:**
   ```sh
   ollama run qwen3:14b
   ```
2. **Run the web application:
   ```sh
   python src/web/app.py
   ```
3. **Access the web interface:**
   Open your browser and navigate to http://127.0.0.1:5001
4. **Analyze a stock:**
   Enter a stock code (e.g., 000333.SZ) in the web interface to receive three sets of expert opinions.

### Using the Command Line Interface
1. **Start Ollama with Qwen3 14B:**
   ```sh
   ollama run qwen3:14b
   ```
2. **Run the analysis tool:
   ```sh
   python src/main.py STOCK_CODE --vix 15.0
   # Example:
   python src/main.py 000333.SZ --vix 15.0
   ```
3. **Output:**
   - The tool will print and optionally save a JSON with three sets of opinions (technical, Peter Lynch, Warren Buffett) and a combined summary.

## Project Structure
- `src/models/akshare_collector.py` — Data collection from AKShare
- `src/analysis/analysis_orchestrator.py` — Orchestrates data flow and agent execution
- `src/strategies/` — Contains the three strategy agents
- `src/llm/stock_analyzer.py` — LLM integration (Ollama/Qwen3)
- `src/config.py` — Configuration for URLs, LLM, and prompts
- `src/web/` — Web interface implementation

## Configuration
- Edit `src/config.py` to adjust LLM settings, system prompts, or add more stock URLs.
- Configuration options include:
  - LLM settings (temperature, max tokens, etc.)
  - System prompts for different strategies
  - AKShare API endpoints

## Development
```sh
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/
```

## Testing
The project includes comprehensive unit tests and end-to-end tests:
- Unit tests cover individual components (`tests/test_*.py`)
- End-to-end tests verify the complete analysis pipeline (`tests/test_e2e.py`)

## TODO
- Retrieve data from multiple sources, e.g., Yahoo Finance


## License
MIT

## TODO
- Retrieve data from multiple sources, e.g., Yahoo Finance

## Acknowledgements
- [AKShare](https://github.com/jindaxiang/akshare)
- [Ollama](https://ollama.com/)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Qwen3](https://github.com/QwenLM)

## Roadmap
- Implement additional investment strategies
- Add support for multiple LLM providers
- Enhance data validation and error handling
- Improve performance through caching and parallel processing
- Expand test coverage for all modules