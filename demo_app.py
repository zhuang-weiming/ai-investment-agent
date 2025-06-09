import sys
import asyncio
from flask import Flask, render_template_string, request, jsonify
from src.data.eastmoney_collector import EastmoneyCollector
import time

app = Flask(__name__)
collector = EastmoneyCollector()

# Track cache statistics
cache_stats = {
    'hits': 0,
    'misses': 0,
    'total_requests': 0,
    'last_reset': time.time()
}

# HTML template for the demo interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>EastmoneyCollector Demo</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { color: #333; }
        .container { max-width: 1200px; margin: 0 auto; }
        .form-group { margin-bottom: 15px; }
        label { display: block; margin-bottom: 5px; }
        input[type="text"], select { padding: 8px; width: 300px; }
        button { padding: 8px 15px; background: #4CAF50; color: white; border: none; cursor: pointer; }
        button:hover { background: #45a049; }
        .results { margin-top: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 4px; }
        .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
        .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 16px; }
        .tab button:hover { background-color: #ddd; }
        .tab button.active { background-color: #ccc; }
        .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        tr:nth-child(even) { background-color: #f9f9f9; }
        .cache-stats { background-color: #f0f8ff; padding: 15px; margin: 15px 0; border-radius: 5px; }
        .cache-stats h3 { margin-top: 0; }
        .cache-stats-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; }
        .cache-stats-item { background: #fff; padding: 10px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .cache-stats-item p { margin: 0; font-weight: bold; }
        .cache-stats-item span { font-size: 1.2em; }
        .response-time { font-size: 0.8em; color: #666; margin-top: 10px; }
        .grid-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; }
        .grid-item { background: #fff; padding: 10px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .market-data-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }
    </style>
</head>
<body>
    <div class="container">
        <h1>EastmoneyCollector Demo with Caching</h1>
        
        <div class="cache-stats">
            <h3>Cache Statistics</h3>
            <div class="cache-stats-grid" id="cache-stats">
                <div class="cache-stats-item">
                    <p>Total Requests</p>
                    <span id="total-requests">0</span>
                </div>
                <div class="cache-stats-item">
                    <p>Cache Hits</p>
                    <span id="cache-hits">0</span>
                </div>
                <div class="cache-stats-item">
                    <p>Cache Misses</p>
                    <span id="cache-misses">0</span>
                </div>
                <div class="cache-stats-item">
                    <p>Hit Rate</p>
                    <span id="hit-rate">0%</span>
                </div>
                <div class="cache-stats-item">
                    <p>Uptime</p>
                    <span id="uptime">0s</span>
                </div>
            </div>
            <div style="margin-top: 10px;">
                <button onclick="fetchCacheStats()">Refresh Stats</button>
                <button onclick="resetCacheStats()">Reset Stats</button>
            </div>
        </div>
        
        <div class="form-group">
            <label for="symbol">Enter Stock Symbol (e.g., 000333.SZ):</label>
            <input type="text" id="symbol" name="symbol" value="000333.SZ">
            <button onclick="fetchData()">Fetch All Data</button>
        </div>
        
        <div class="tab">
            <button class="tablinks" onclick="openTab(event, 'Financials')" id="defaultOpen">Financials</button>
            <button class="tablinks" onclick="openTab(event, 'News')">News</button>
            <button class="tablinks" onclick="openTab(event, 'InsiderTrades')">Insider Trades</button>
            <button class="tablinks" onclick="openTab(event, 'MarketData')">Market Data</button>
        </div>
        
        <div id="Financials" class="tabcontent">
            <h3>Financial Data</h3>
            <div id="financials-results"></div>
            <div class="response-time" id="financials-time"></div>
        </div>
        
        <div id="News" class="tabcontent">
            <h3>News</h3>
            <div style="margin-bottom: 10px;">
                <label for="news-limit">Number of news items:</label>
                <select id="news-limit">
                    <option value="3">3</option>
                    <option value="5" selected>5</option>
                    <option value="10">10</option>
                </select>
                <button onclick="fetchNews()">Refresh News</button>
            </div>
            <div id="news-results"></div>
            <div class="response-time" id="news-time"></div>
        </div>
        
        <div id="InsiderTrades" class="tabcontent">
            <h3>Insider Trades</h3>
            <div style="margin-bottom: 10px;">
                <label for="insider-limit">Number of trades:</label>
                <select id="insider-limit">
                    <option value="5">5</option>
                    <option value="10" selected>10</option>
                    <option value="15">15</option>
                </select>
                <button onclick="fetchInsiderTrades()">Refresh Insider Trades</button>
            </div>
            <div id="insider-results"></div>
            <div class="response-time" id="insider-time"></div>
        </div>
        
        <div id="MarketData" class="tabcontent">
            <h3>Market Data</h3>
            <button onclick="fetchMarketData()">Refresh Market Data</button>
            <div id="marketcap-results"></div>
            <div class="response-time" id="marketcap-time"></div>
        </div>
    </div>
    
    <script>
        // Open the default tab on page load
        document.getElementById("defaultOpen").click();
        
        // Fetch cache statistics on page load
        fetchCacheStats();
        
        function openTab(evt, tabName) {
            var i, tabcontent, tablinks;
            tabcontent = document.getElementsByClassName("tabcontent");
            for (i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }
            tablinks = document.getElementsByClassName("tablinks");
            for (i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }
            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
        
        function fetchCacheStats() {
            fetch('/api/cache_stats')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('total-requests').textContent = data.total_requests;
                    document.getElementById('cache-hits').textContent = data.hits;
                    document.getElementById('cache-misses').textContent = data.misses;
                    document.getElementById('hit-rate').textContent = data.hit_rate;
                    document.getElementById('uptime').textContent = data.uptime;
                })
                .catch(error => {
                    console.error('Error fetching cache stats:', error);
                });
        }
        
        function resetCacheStats() {
            fetch('/api/reset_cache_stats')
                .then(response => response.json())
                .then(data => {
                    console.log(data.status);
                    fetchCacheStats();
                })
                .catch(error => {
                    console.error('Error resetting cache stats:', error);
                });
        }
        
        function fetchData() {
            const symbol = document.getElementById('symbol').value;
            if (!symbol) {
                alert('Please enter a stock symbol');
                return;
            }
            
            fetchFinancials();
            fetchNews();
            fetchInsiderTrades();
            fetchMarketData();
            fetchCacheStats();
        }
        
        function fetchFinancials() {
            const symbol = document.getElementById('symbol').value;
            const resultsDiv = document.getElementById('financials-results');
            const timeDiv = document.getElementById('financials-time');
            
            resultsDiv.innerHTML = '<p>Loading financial data...</p>';
            const startTime = performance.now();
            
            fetch(`/api/financials?symbol=${symbol}`)
                .then(response => response.json())
                .then(data => {
                    const endTime = performance.now();
                    const responseTime = endTime - startTime;
                    
                    if (data.length === 0) {
                        resultsDiv.innerHTML = '<p>No financial data found.</p>';
                        return;
                    }
                    
                    let tableHtml = '<table><tr><th>Report Date</th><th>Revenue</th><th>Net Profit</th><th>EPS</th><th>ROE</th><th>Debt Ratio</th><th>Profit Margin</th><th>Growth Rate</th><th>Cash Flow</th></tr>';
                    data.forEach(item => {
                        tableHtml += `<tr>
                            <td>${item.report_date}</td>
                            <td>${item.revenue}</td>
                            <td>${item.net_profit}</td>
                            <td>${item.eps}</td>
                            <td>${item.roe}</td>
                            <td>${item.debt_ratio}</td>
                            <td>${item.profit_margin}</td>
                            <td>${item.revenue_growth || 'N/A'}</td>
                            <td>${item.operating_cash_flow || 'N/A'}</td>
                        </tr>`;
                    });
                    tableHtml += '</table>';
                    resultsDiv.innerHTML = tableHtml;
                    timeDiv.innerHTML = `Response time: ${responseTime.toFixed(2)}ms`;
                    
                    // Refresh cache stats
                    fetchCacheStats();
                })
                .catch(error => {
                    resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                });
        }
        
        function fetchNews() {
            const symbol = document.getElementById('symbol').value;
            const limit = document.getElementById('news-limit').value;
            const resultsDiv = document.getElementById('news-results');
            const timeDiv = document.getElementById('news-time');
            
            resultsDiv.innerHTML = '<p>Loading news...</p>';
            const startTime = performance.now();
            
            fetch(`/api/news?symbol=${symbol}&limit=${limit}`)
                .then(response => response.json())
                .then(data => {
                    const endTime = performance.now();
                    const responseTime = endTime - startTime;
                    
                    if (data.length === 0) {
                        resultsDiv.innerHTML = '<p>No news found.</p>';
                        return;
                    }
                    
                    let html = '';
                    data.forEach(item => {
                        const sentimentClass = item.sentiment > 0 ? 'positive' : (item.sentiment < 0 ? 'negative' : 'neutral');
                        const sentimentText = item.sentiment > 0 ? 'Positive' : (item.sentiment < 0 ? 'Negative' : 'Neutral');
                        
                        html += `<div style="margin-bottom: 20px; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                            <h4 style="margin-top: 0;">${item.title}</h4>
                            <p><strong>Date:</strong> ${item.date} | <strong>Source:</strong> ${item.source}</p>
                            <p>${item.summary || 'No summary available'}</p>
                            <p><strong>Sentiment:</strong> <span style="color: ${sentimentClass === 'positive' ? 'green' : (sentimentClass === 'negative' ? 'red' : 'gray')}">${sentimentText}</span></p>
                            <p><a href="${item.url}" target="_blank">Read more</a></p>
                        </div>`;
                    });
                    resultsDiv.innerHTML = html;
                    timeDiv.innerHTML = `Response time: ${responseTime.toFixed(2)}ms`;
                    
                    // Refresh cache stats
                    fetchCacheStats();
                })
                .catch(error => {
                    resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                });
        }
        
        function fetchInsiderTrades() {
            const symbol = document.getElementById('symbol').value;
            const limit = document.getElementById('insider-limit').value;
            const resultsDiv = document.getElementById('insider-results');
            const timeDiv = document.getElementById('insider-time');
            
            resultsDiv.innerHTML = '<p>Loading insider trades...</p>';
            const startTime = performance.now();
            
            fetch(`/api/insider_trades?symbol=${symbol}&limit=${limit}`)
                .then(response => response.json())
                .then(data => {
                    const endTime = performance.now();
                    const responseTime = endTime - startTime;
                    
                    if (data.length === 0) {
                        resultsDiv.innerHTML = '<p>No insider trades found.</p>';
                        return;
                    }
                    
                    let tableHtml = '<table><tr><th>Name</th><th>Position</th><th>Type</th><th>Shares</th><th>Price</th><th>Total Value</th><th>Date</th><th>Reason</th></tr>';
                    data.forEach(item => {
                        const changeType = item.change_type;
                        const typeColor = changeType === 'Buy' ? 'green' : 'red';
                        
                        tableHtml += `<tr>
                            <td>${item.name}</td>
                            <td>${item.position}</td>
                            <td style="color: ${typeColor}; font-weight: bold;">${changeType}</td>
                            <td>${item.change_shares.toLocaleString()}</td>
                            <td>${item.price.toLocaleString()}</td>
                            <td>${item.total_value ? item.total_value.toLocaleString() : 'N/A'}</td>
                            <td>${item.date}</td>
                            <td>${item.reason || 'N/A'}</td>
                        </tr>`;
                    });
                    tableHtml += '</table>';
                    resultsDiv.innerHTML = tableHtml;
                    timeDiv.innerHTML = `Response time: ${responseTime.toFixed(2)}ms`;
                    
                    // Refresh cache stats
                    fetchCacheStats();
                })
                .catch(error => {
                    resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                });
        }
        
        function fetchMarketData() {
            const symbol = document.getElementById('symbol').value;
            const resultsDiv = document.getElementById('marketcap-results');
            const timeDiv = document.getElementById('marketcap-time');
            
            resultsDiv.innerHTML = '<p>Loading market data...</p>';
            const startTime = performance.now();
            
            fetch(`/api/market_cap?symbol=${symbol}`)
                .then(response => response.json())
                .then(data => {
                    const endTime = performance.now();
                    const responseTime = endTime - startTime;
                    
                    if (!data || Object.keys(data).length === 0) {
                        resultsDiv.innerHTML = '<p>No market data found.</p>';
                        return;
                    }
                    
                    // Format large numbers
                    const formatNumber = (num) => {
                        if (num >= 1e12) return (num / 1e12).toFixed(2) + ' T';
                        if (num >= 1e9) return (num / 1e9).toFixed(2) + ' B';
                        if (num >= 1e6) return (num / 1e6).toFixed(2) + ' M';
                        if (num >= 1e3) return (num / 1e3).toFixed(2) + ' K';
                        return num.toFixed(2);
                    };
                    
                    // Create a grid layout for market data
                    let html = `<div class="market-data-grid">
                        <div class="grid-item">
                            <h4>Market Cap</h4>
                            <p>${formatNumber(data.market_cap)} ${data.currency}</p>
                            <p>Float Market Cap: ${formatNumber(data.float_market_cap)} ${data.currency}</p>
                        </div>
                        
                        <div class="grid-item">
                            <h4>Shares</h4>
                            <p>Total Shares: ${formatNumber(data.total_shares)}</p>
                            <p>Float Shares: ${formatNumber(data.float_shares)}</p>
                        </div>
                        
                        <div class="grid-item">
                            <h4>Price</h4>
                            <p>Current: ${data.price.toFixed(2)} ${data.currency}</p>
                            <p>Change: <span style="color: ${data.price_change >= 0 ? 'green' : 'red'}">${data.price_change.toFixed(2)} (${data.price_change_percent.toFixed(2)}%)</span></p>
                        </div>
                        
                        <div class="grid-item">
                            <h4>Trading Range</h4>
                            <p>Open: ${data.open_price.toFixed(2)}</p>
                            <p>High: ${data.high_price.toFixed(2)} | Low: ${data.low_price.toFixed(2)}</p>
                        </div>
                        
                        <div class="grid-item">
                            <h4>Volume</h4>
                            <p>Volume: ${formatNumber(data.volume)}</p>
                            <p>Turnover Rate: ${data.turnover_rate.toFixed(2)}%</p>
                        </div>
                        
                        <div class="grid-item">
                            <h4>Valuation</h4>
                            <p>PE Ratio: ${data.pe_ratio.toFixed(2)}</p>
                            <p>PB Ratio: ${data.pb_ratio.toFixed(2)}</p>
                            <p>Dividend Yield: ${data.dividend_yield.toFixed(2)}%</p>
                        </div>
                    </div>
                    <p style="margin-top: 10px; font-size: 0.8em;">Data Source: ${data.data_source || 'Unknown'}</p>`;
                    
                    resultsDiv.innerHTML = html;
                    timeDiv.innerHTML = `Response time: ${responseTime.toFixed(2)}ms`;
                    
                    // Refresh cache stats
                    fetchCacheStats();
                })
                .catch(error => {
                    resultsDiv.innerHTML = `<p>Error: ${error.message}</p>`;
                });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

# Function to update cache statistics
def update_cache_stats(endpoint, is_hit):
    global cache_stats
    cache_stats['total_requests'] += 1
    if is_hit:
        cache_stats['hits'] += 1
    else:
        cache_stats['misses'] += 1

@app.route('/api/cache_stats')
def get_cache_stats():
    # Calculate hit rate
    if cache_stats['total_requests'] > 0:
        hit_rate = (cache_stats['hits'] / cache_stats['total_requests']) * 100
    else:
        hit_rate = 0
        
    # Add uptime
    uptime = time.time() - cache_stats['last_reset']
    
    stats = {
        **cache_stats,
        'hit_rate': f"{hit_rate:.2f}%",
        'uptime': f"{uptime:.2f} seconds"
    }
    
    return jsonify(stats)

@app.route('/api/reset_cache_stats')
def reset_cache_stats():
    global cache_stats
    cache_stats = {
        'hits': 0,
        'misses': 0,
        'total_requests': 0,
        'last_reset': time.time()
    }
    return jsonify({"status": "Cache statistics reset"})

@app.route('/api/financials')
def get_financials():
    symbol = request.args.get('symbol', '000333.SZ')
    try:
        start_time = time.time()
        data = collector.get_financials(symbol)
        response_time = time.time() - start_time
        
        # Update cache stats - assume it's a cache hit if response time is very fast
        update_cache_stats('financials', response_time < 0.1)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/news')
def get_news():
    symbol = request.args.get('symbol', '000333.SZ')
    limit = request.args.get('limit', 5, type=int)
    try:
        start_time = time.time()
        data = collector.get_news(symbol, limit=limit)
        response_time = time.time() - start_time
        
        # Update cache stats
        update_cache_stats('news', response_time < 0.1)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/insider_trades')
def get_insider_trades():
    symbol = request.args.get('symbol', '000333.SZ')
    limit = request.args.get('limit', 10, type=int)
    try:
        start_time = time.time()
        data = collector.get_insider_trades(symbol, limit=limit)
        response_time = time.time() - start_time
        
        # Update cache stats
        update_cache_stats('insider_trades', response_time < 0.1)
        
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/market_cap')
def get_market_cap():
    symbol = request.args.get('symbol', '000333.SZ')
    try:
        start_time = time.time()
        market_data = collector.get_market_cap(symbol)
        response_time = time.time() - start_time
        
        # Update cache stats
        update_cache_stats('market_cap', response_time < 0.1)
        
        return jsonify(market_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)