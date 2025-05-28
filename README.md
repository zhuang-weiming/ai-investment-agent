# AI Investment Agent

A modular, agent-based stock analysis tool that collects data from AKShare and provides multi-strategy investment opinions using a local LLM (Qwen3 14B via Ollama on macOS).

## Features
- **Data Collection:** Fetches real-time and historical stock data from AKShare.
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
``sh
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
   Enter a stock code (e.g., 000688.SZ) in the web interface to receive three sets of expert opinions.

### Using the Command Line Interface
1. **Start Ollama with Qwen3 14B:**
   ```sh
   ollama run qwen3:14b
   ```
2. **Run the analysis tool:**
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

## License
MIT

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