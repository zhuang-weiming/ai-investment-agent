import pytest
from src.strategies.warren_buffett import WarrenBuffettStrategy, BuffettMetrics

@pytest.fixture
def sample_stock_data():
    return {
        'current_price': 150.0,
        'owner_earnings': 1000000000,  # Strong owner earnings
        'roe': 0.185,  # Good ROE > 15%
        'debt_ratio': 0.3,  # Conservative debt
        'fcf': 120000000,  # Strong free cash flow
        'moat_type': 'brand',  # Strong brand moat
        'mgmt_score': 85.0,  # Good management
        'pe_ratio': 15.0,
        'industry_pe': 20.0,
        'pb_ratio': 2.5,
        'historical_pb': 3.0,
        'market_share': 25.0,  # Good market share
        'brand_value': 85.0,  # Strong brand
        'insider_ownership': 12.0,  # Good insider alignment
        'buyback_efficiency': 85.0,  # Efficient buybacks
        'capital_allocation': 90.0,  # Efficient capital allocation
        'industry_growth': 12.0,
        'market_concentration': 70.0,
        'regulatory_risk': 25.0,  # Low regulatory risk
        'industry_cycle_position': 'growth',
        'competitive_position': 80.0
    }

@pytest.fixture
def declining_industry_data(sample_stock_data):
    """Sample data for a company in a declining industry"""
    data = sample_stock_data.copy()
    data.update({
        'industry_cycle_position': 'decline',
        'industry_growth': -2.0,
        'market_concentration': 90.0,
        'competitive_position': 60.0,
        'market_share': 15.0,
        'regulatory_risk': 45.0,
        'owner_earnings': 500000000,  # Lower earnings in decline
        'roe': 0.12  # Lower ROE in decline
    })
    return data

@pytest.fixture
def high_risk_data(sample_stock_data):
    """Sample data for a high-risk company"""
    data = sample_stock_data.copy()
    data.update({
        'regulatory_risk': 85.0,
        'debt_ratio': 0.8,  # Very high debt
        'moat_type': 'none',
        'market_share': 3.0,
        'brand_value': 30.0,
        'insider_ownership': 2.0,
        'capital_allocation': 30.0,  # Very poor capital allocation
        'mgmt_score': 35.0,  # Very poor management
        'buyback_efficiency': 40.0,  # Poor buybacks
        'fcf': -50000000,  # Negative free cash flow
        'roe': 0.05  # Poor ROE
    })
    return data

def test_validate_data(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    assert strategy.validate_data(sample_stock_data) == True

def test_validate_data_missing_metrics():
    """Test data validation with missing metrics"""
    strategy = WarrenBuffettStrategy()
    incomplete_data = {
        'roe': 0.15,
        'fcf': 1000000
    }
    assert strategy.validate_data(incomplete_data) == False

def test_moat_score(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    data = sample_stock_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_moat_score(metrics)
    assert 75 <= score <= 100  # Strong moat characteristics

def test_moat_score_weak_position(high_risk_data):
    """Test moat score calculation for weak competitive position"""
    strategy = WarrenBuffettStrategy()
    data = high_risk_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_moat_score(metrics)
    assert score < 50  # Weak moat characteristics

def test_management_score(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    data = sample_stock_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_management_score(metrics)
    assert 80 <= score <= 100  # Good management score

def test_management_score_poor_allocation(high_risk_data):
    """Test management score with poor capital allocation"""
    strategy = WarrenBuffettStrategy()
    data = high_risk_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_management_score(metrics)
    assert score < 50  # Poor management characteristics

def test_financial_score(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    data = sample_stock_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_financial_score(metrics)
    assert score >= 70  # Good financial position

def test_financial_score_high_debt(high_risk_data):
    """Test financial score with high debt"""
    strategy = WarrenBuffettStrategy()
    data = high_risk_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_financial_score(metrics)
    assert score < 50  # Poor financial health due to high debt and negative FCF

def test_valuation_score(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    data = sample_stock_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    score = strategy.calculate_valuation_score(metrics)
    assert score >= 70  # Good valuation

def test_competitive_advantage_period():
    """Test CAP calculation for different scenarios"""
    strategy = WarrenBuffettStrategy()
    
    # Test dominant brand
    metrics = BuffettMetrics(
        owner_earnings=1000000000,
        roe=0.22,
        debt_ratio=0.2,
        fcf=800000000,
        moat_type='brand',
        mgmt_score=90.0,
        pe_ratio=15.0,
        industry_pe=20.0,
        pb_ratio=2.5,
        historical_pb=3.0,
        market_share=45.0,
        brand_value=90.0,
        insider_ownership=15.0,
        buyback_efficiency=85.0,
        capital_allocation=90.0,
        industry_growth=8.0,
        market_concentration=75.0,
        regulatory_risk=20.0,
        industry_cycle_position='mature',
        competitive_position=85.0
    )
    cap = strategy.estimate_competitive_advantage_period(metrics)
    assert 15 <= cap <= 20  # Strong brand should have long CAP

    # Test weak position
    metrics.moat_type = 'none'
    metrics.market_share = 3.0
    metrics.brand_value = 30.0
    metrics.roe = 0.08
    cap = strategy.estimate_competitive_advantage_period(metrics)
    assert 3 <= cap <= 5  # Weak position should have short CAP

def test_intrinsic_value_calculation(sample_stock_data):
    """Test intrinsic value calculation"""
    strategy = WarrenBuffettStrategy()
    data = sample_stock_data.copy()
    current_price = data.pop('current_price')
    metrics = BuffettMetrics(**data)
    value = strategy.calculate_intrinsic_value(metrics)
    assert value is not None
    assert value > 0
    assert isinstance(value, float)

def test_intrinsic_value_declining_industry(declining_industry_data, sample_stock_data):
    """Test intrinsic value calculation for declining industry"""
    strategy = WarrenBuffettStrategy()
    
    # Calculate value for declining industry
    data = declining_industry_data.copy()
    current_price = data.pop('current_price')
    metrics_declining = BuffettMetrics(**data)
    value_declining = strategy.calculate_intrinsic_value(metrics_declining)
    
    # Calculate value for growth industry
    data_growth = sample_stock_data.copy()
    current_price = data_growth.pop('current_price')
    metrics_growth = BuffettMetrics(**data_growth)
    value_growth = strategy.calculate_intrinsic_value(metrics_growth)
    
    assert value_declining < value_growth

@pytest.mark.asyncio
async def test_analyze(sample_stock_data):
    strategy = WarrenBuffettStrategy()
    result = await strategy.analyze(sample_stock_data)
    assert result.signal == "bullish"  # Should be bullish given the strong metrics
    assert result.confidence >= 80
    assert "moat" in result.reasoning.lower()
    assert "management" in result.reasoning.lower()
    assert "margin of safety" in result.reasoning.lower()

@pytest.mark.asyncio
async def test_analyze_high_risk(high_risk_data):
    """Test analysis for high-risk company"""
    strategy = WarrenBuffettStrategy()
    result = await strategy.analyze(high_risk_data)
    assert result.signal in ["neutral", "bearish"]  # Should be bearish due to high risk
    assert result.confidence < 60  # Lower confidence due to risks
    assert "regulatory risk" in result.reasoning.lower()
    assert result.raw_data['moat_score'] < 50

@pytest.mark.asyncio
async def test_analyze_declining_industry(declining_industry_data):
    """Test analysis for company in declining industry"""
    strategy = WarrenBuffettStrategy()
    result = await strategy.analyze(declining_industry_data)
    assert result.signal in ["neutral", "bearish"]  # Should be cautious in declining industry
    assert any(word in result.reasoning.lower() for word in ["declining", "decline"])  # Check for any form of "decline"
    assert "/no_think" in result.reasoning  # Verify /no_think is present
    assert "industry" in result.reasoning.lower()
    assert result.raw_data['intrinsic_value'] is not None
    assert float(result.raw_data['intrinsic_value']) > 0  # Ensure positive value

@pytest.mark.asyncio
async def test_analyze_missing_price():
    """Test analysis with missing current price"""
    strategy = WarrenBuffettStrategy()
    test_data = {
        'current_price': 0,  # Invalid price
        'owner_earnings': 1000000000,
        'roe': 0.15,
        'debt_ratio': 0.3,
        'fcf': 100000000,
        'moat_type': 'brand',
        'mgmt_score': 80.0,
        'pe_ratio': 15.0,
        'industry_pe': 20.0,
        'pb_ratio': 2.5,
        'historical_pb': 3.0,
        'market_share': 20.0,
        'brand_value': 80.0,
        'insider_ownership': 10.0,
        'buyback_efficiency': 80.0,
        'capital_allocation': 85.0,
        'industry_growth': 10.0,
        'market_concentration': 65.0,
        'regulatory_risk': 30.0,
        'industry_cycle_position': 'mature',
        'competitive_position': 75.0
    }
    result = await strategy.analyze(test_data)
    assert result.signal == "neutral"
    assert result.confidence == 20
    assert "price" in result.reasoning.lower()
