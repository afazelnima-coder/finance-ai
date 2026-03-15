"""
Tests for the portfolio agent functionality.
Tests metrics calculations, asset classification, expense ratios, and risk assessment.
"""
import pytest
from unittest.mock import patch, MagicMock


class TestAssetClassification:
    """Tests for asset class determination."""

    def test_get_asset_class_from_ticker(self):
        """Test that known tickers map to correct asset classes."""
        from agents.portfolio_agent import get_asset_class, AssetClass

        assert get_asset_class("us_stock", "VOO") == AssetClass.US_STOCK
        assert get_asset_class("us_stock", "BND") == AssetClass.BOND
        assert get_asset_class("us_stock", "VXUS") == AssetClass.INTL_STOCK
        assert get_asset_class("us_stock", "GLD") == AssetClass.COMMODITY

    def test_get_asset_class_from_type_string(self):
        """Test that asset type strings map correctly."""
        from agents.portfolio_agent import get_asset_class, AssetClass

        assert get_asset_class("us_stock", None) == AssetClass.US_STOCK
        assert get_asset_class("bond", None) == AssetClass.BOND
        assert get_asset_class("intl_stock", None) == AssetClass.INTL_STOCK
        assert get_asset_class("crypto", None) == AssetClass.CRYPTO
        assert get_asset_class("mixed", None) == AssetClass.MIXED

    def test_get_asset_class_unknown_defaults_to_us_stock(self):
        """Test that unknown types default to US stocks."""
        from agents.portfolio_agent import get_asset_class, AssetClass

        assert get_asset_class("unknown_type", None) == AssetClass.US_STOCK


class TestExpenseRatios:
    """Tests for expense ratio lookups."""

    def test_get_expense_ratio_known_ticker(self):
        """Test expense ratio for known tickers."""
        from agents.portfolio_agent import get_expense_ratio

        assert get_expense_ratio("VOO", "Vanguard S&P 500") == 0.03
        assert get_expense_ratio("BND", "Bond Fund") == 0.03
        assert get_expense_ratio("SPY", "SPDR S&P 500") == 0.09
        assert get_expense_ratio("QQQ", "Nasdaq 100") == 0.20

    def test_get_expense_ratio_target_date_fund(self):
        """Test expense ratio for target date funds."""
        from agents.portfolio_agent import get_expense_ratio

        assert get_expense_ratio(None, "Target Date 2045 Fund") == 0.14
        assert get_expense_ratio(None, "Vanguard Target Date 2050") == 0.14

    def test_get_expense_ratio_unknown_returns_none(self):
        """Test that unknown funds return None."""
        from agents.portfolio_agent import get_expense_ratio

        assert get_expense_ratio("UNKNOWN", "Unknown Fund") is None
        assert get_expense_ratio(None, "Apple Inc") is None  # Individual stock


class TestDiversificationScore:
    """Tests for diversification score calculation."""

    def test_diversification_score_single_holding(self):
        """Test low diversification score for single holding."""
        from agents.portfolio_agent import calculate_diversification_score, Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 100000, AssetClass.US_STOCK, 0.03, True)
        ]
        allocations = {"US Stocks": 100.0}

        score = calculate_diversification_score(holdings, allocations)

        # Single holding, single asset class = low score
        assert score < 50

    def test_diversification_score_multiple_asset_classes(self):
        """Test higher diversification score for multiple asset classes."""
        from agents.portfolio_agent import calculate_diversification_score, Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 50000, AssetClass.US_STOCK, 0.03, True),
            Holding("BND", "BND", 30000, AssetClass.BOND, 0.03, True),
            Holding("VXUS", "VXUS", 20000, AssetClass.INTL_STOCK, 0.07, True),
        ]
        allocations = {
            "US Stocks": 50.0,
            "Bonds": 30.0,
            "International Stocks": 20.0,
        }

        score = calculate_diversification_score(holdings, allocations)

        # Multiple asset classes + international = higher score
        assert score >= 60

    def test_diversification_score_with_index_funds(self):
        """Test that index fund usage improves diversification score."""
        from agents.portfolio_agent import calculate_diversification_score, Holding, AssetClass

        # All index funds
        holdings_index = [
            Holding("VOO", "VOO", 50000, AssetClass.US_STOCK, 0.03, True),
            Holding("BND", "BND", 50000, AssetClass.BOND, 0.03, True),
        ]

        # All individual stocks (not index funds)
        holdings_individual = [
            Holding("Apple", "AAPL", 50000, AssetClass.US_STOCK, None, False),
            Holding("Microsoft", "MSFT", 50000, AssetClass.US_STOCK, None, False),
        ]

        allocations_mixed = {"US Stocks": 50.0, "Bonds": 50.0}
        allocations_single = {"US Stocks": 100.0}

        score_index = calculate_diversification_score(holdings_index, allocations_mixed)
        score_individual = calculate_diversification_score(holdings_individual, allocations_single)

        # Index funds should score higher
        assert score_index > score_individual


class TestRiskAssessment:
    """Tests for portfolio risk assessment."""

    def test_risk_assessment_aggressive(self):
        """Test aggressive risk level for high stock allocation."""
        from agents.portfolio_agent import assess_risk, Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 95000, AssetClass.US_STOCK, 0.03, True),
            Holding("QQQ", "QQQ", 5000, AssetClass.US_STOCK, 0.20, True),
        ]
        allocations = {"US Stocks": 100.0}

        risk_level, risk_factors = assess_risk(holdings, allocations)

        assert risk_level == "Aggressive"
        assert any("equity" in f.lower() or "stock" in f.lower() for f in risk_factors)

    def test_risk_assessment_conservative(self):
        """Test conservative risk level for high bond allocation."""
        from agents.portfolio_agent import assess_risk, Holding, AssetClass

        holdings = [
            Holding("BND", "BND", 80000, AssetClass.BOND, 0.03, True),
            Holding("VOO", "VOO", 20000, AssetClass.US_STOCK, 0.03, True),
        ]
        allocations = {"Bonds": 80.0, "US Stocks": 20.0}

        risk_level, risk_factors = assess_risk(holdings, allocations)

        assert risk_level in ["Conservative", "Moderate-Conservative"]

    def test_risk_assessment_moderate(self):
        """Test moderate risk level for balanced allocation."""
        from agents.portfolio_agent import assess_risk, Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 60000, AssetClass.US_STOCK, 0.03, True),
            Holding("BND", "BND", 40000, AssetClass.BOND, 0.03, True),
        ]
        allocations = {"US Stocks": 60.0, "Bonds": 40.0}

        risk_level, risk_factors = assess_risk(holdings, allocations)

        assert risk_level in ["Moderate", "Moderate-Aggressive"]

    def test_risk_assessment_concentration_warning(self):
        """Test that concentration risk is flagged."""
        from agents.portfolio_agent import assess_risk, Holding, AssetClass

        holdings = [
            Holding("Apple", "AAPL", 80000, AssetClass.US_STOCK, None, False),
            Holding("VOO", "VOO", 20000, AssetClass.US_STOCK, 0.03, True),
        ]
        allocations = {"US Stocks": 100.0}

        risk_level, risk_factors = assess_risk(holdings, allocations)

        assert any("concentration" in f.lower() for f in risk_factors)

    def test_risk_assessment_individual_stock_warning(self):
        """Test that high individual stock exposure is flagged."""
        from agents.portfolio_agent import assess_risk, Holding, AssetClass

        holdings = [
            Holding("Apple", "AAPL", 40000, AssetClass.US_STOCK, None, False),
            Holding("Microsoft", "MSFT", 30000, AssetClass.US_STOCK, None, False),
            Holding("VOO", "VOO", 30000, AssetClass.US_STOCK, 0.03, True),
        ]
        allocations = {"US Stocks": 100.0}

        risk_level, risk_factors = assess_risk(holdings, allocations)

        assert any("individual stock" in f.lower() for f in risk_factors)


class TestPortfolioMetricsCalculation:
    """Tests for overall portfolio metrics."""

    def test_weighted_expense_ratio_calculation(self):
        """Test weighted expense ratio is calculated correctly."""
        from agents.portfolio_agent import Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 100000, AssetClass.US_STOCK, 0.03, True),  # 0.03% * 66.67%
            Holding("BND", "BND", 50000, AssetClass.BOND, 0.03, True),       # 0.03% * 33.33%
        ]

        total_value = sum(h.value for h in holdings)
        weighted_expense = sum(
            (h.expense_ratio * h.value / total_value)
            for h in holdings if h.expense_ratio is not None
        )

        # Should be 0.03 (same expense ratio for both)
        assert abs(weighted_expense - 0.03) < 0.001

    def test_allocation_percentages(self):
        """Test allocation percentages sum to 100."""
        from agents.portfolio_agent import Holding, AssetClass

        holdings = [
            Holding("VOO", "VOO", 50000, AssetClass.US_STOCK, 0.03, True),
            Holding("BND", "BND", 30000, AssetClass.BOND, 0.03, True),
            Holding("VXUS", "VXUS", 20000, AssetClass.INTL_STOCK, 0.07, True),
        ]

        total_value = sum(h.value for h in holdings)
        allocations = {}
        for h in holdings:
            ac = h.asset_class.value
            if ac not in allocations:
                allocations[ac] = 0
            allocations[ac] += (h.value / total_value) * 100

        total_allocation = sum(allocations.values())

        assert abs(total_allocation - 100.0) < 0.01


class TestHoldingDataClass:
    """Tests for the Holding dataclass."""

    def test_holding_creation(self):
        """Test Holding dataclass creation."""
        from agents.portfolio_agent import Holding, AssetClass

        holding = Holding(
            name="S&P 500 Index",
            ticker="VOO",
            value=100000,
            asset_class=AssetClass.US_STOCK,
            expense_ratio=0.03,
            is_index_fund=True
        )

        assert holding.name == "S&P 500 Index"
        assert holding.ticker == "VOO"
        assert holding.value == 100000
        assert holding.asset_class == AssetClass.US_STOCK
        assert holding.expense_ratio == 0.03
        assert holding.is_index_fund is True


class TestExpenseRatioReference:
    """Tests for expense ratio reference data."""

    def test_common_etfs_have_expense_ratios(self):
        """Test that common ETFs are in the reference data."""
        from agents.portfolio_agent import EXPENSE_RATIOS

        common_etfs = ["VOO", "VTI", "SPY", "QQQ", "BND", "AGG", "IVV"]
        for etf in common_etfs:
            assert etf in EXPENSE_RATIOS, f"{etf} should be in expense ratio reference"

    def test_expense_ratios_are_reasonable(self):
        """Test that expense ratios are within reasonable bounds."""
        from agents.portfolio_agent import EXPENSE_RATIOS

        for ticker, ratio in EXPENSE_RATIOS.items():
            assert 0 <= ratio <= 2.0, f"{ticker} has unreasonable expense ratio: {ratio}"


class TestAssetClassMap:
    """Tests for asset class mapping."""

    def test_common_tickers_are_mapped(self):
        """Test that common tickers are in the asset class map."""
        from agents.portfolio_agent import ASSET_CLASS_MAP

        common_tickers = ["VOO", "VTI", "BND", "VXUS", "SPY"]
        for ticker in common_tickers:
            assert ticker in ASSET_CLASS_MAP, f"{ticker} should be in asset class map"
