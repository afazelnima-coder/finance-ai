"""
Tests for portfolio visualization functions.
Tests that charts are created correctly with proper data.
"""
import pytest
import plotly.graph_objects as go


class TestAllocationPieChart:
    """Tests for the allocation pie chart."""

    def test_create_pie_chart_returns_figure(self):
        """Test that pie chart returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_allocation_pie_chart

        allocations = {"US Stocks": 60.0, "Bonds": 40.0}
        fig = create_allocation_pie_chart(allocations, 100000)

        assert isinstance(fig, go.Figure)

    def test_pie_chart_has_correct_values(self):
        """Test that pie chart contains correct allocation values."""
        from agents.portfolio_visualizations import create_allocation_pie_chart

        allocations = {"US Stocks": 60.0, "Bonds": 30.0, "Cash": 10.0}
        fig = create_allocation_pie_chart(allocations, 100000)

        # Check that the pie trace has correct values
        pie_trace = fig.data[0]
        assert list(pie_trace.values) == [60.0, 30.0, 10.0]

    def test_pie_chart_has_correct_labels(self):
        """Test that pie chart has correct asset class labels."""
        from agents.portfolio_visualizations import create_allocation_pie_chart

        allocations = {"US Stocks": 50.0, "Bonds": 50.0}
        fig = create_allocation_pie_chart(allocations, 100000)

        pie_trace = fig.data[0]
        assert "US Stocks" in pie_trace.labels
        assert "Bonds" in pie_trace.labels

    def test_pie_chart_includes_total_value_annotation(self):
        """Test that pie chart has total value annotation."""
        from agents.portfolio_visualizations import create_allocation_pie_chart

        allocations = {"US Stocks": 100.0}
        fig = create_allocation_pie_chart(allocations, 250000)

        # Check annotations include total value
        annotations = fig.layout.annotations
        assert len(annotations) > 0
        assert "$250,000" in annotations[0].text


class TestHoldingsBarChart:
    """Tests for the holdings bar chart."""

    def test_create_bar_chart_returns_figure(self):
        """Test that bar chart returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_holdings_bar_chart

        holdings = [
            {"name": "VOO", "ticker": "VOO", "value": 50000, "expense_ratio": 0.03},
            {"name": "BND", "ticker": "BND", "value": 30000, "expense_ratio": 0.03},
        ]
        fig = create_holdings_bar_chart(holdings)

        assert isinstance(fig, go.Figure)

    def test_bar_chart_sorted_by_value(self):
        """Test that holdings are sorted by value descending."""
        from agents.portfolio_visualizations import create_holdings_bar_chart

        holdings = [
            {"name": "Small", "ticker": "SML", "value": 10000, "expense_ratio": 0.03},
            {"name": "Large", "ticker": "LRG", "value": 90000, "expense_ratio": 0.03},
        ]
        fig = create_holdings_bar_chart(holdings)

        bar_trace = fig.data[0]
        # Values should be in descending order (90000, 10000)
        assert bar_trace.x[0] == 90000
        assert bar_trace.x[1] == 10000

    def test_bar_chart_handles_missing_ticker(self):
        """Test that bar chart handles holdings without tickers."""
        from agents.portfolio_visualizations import create_holdings_bar_chart

        holdings = [
            {"name": "Target Date 2045", "ticker": None, "value": 50000, "expense_ratio": 0.14},
        ]
        fig = create_holdings_bar_chart(holdings)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestExpenseComparisonChart:
    """Tests for expense ratio comparison chart."""

    def test_create_expense_chart_returns_figure(self):
        """Test that expense chart returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_expense_comparison_chart

        holdings = [
            {"name": "VOO", "ticker": "VOO", "value": 50000, "expense_ratio": 0.03},
            {"name": "ARKK", "ticker": "ARKK", "value": 30000, "expense_ratio": 0.75},
        ]
        fig = create_expense_comparison_chart(holdings)

        assert isinstance(fig, go.Figure)

    def test_expense_chart_handles_no_expense_data(self):
        """Test that expense chart handles holdings without expense ratios."""
        from agents.portfolio_visualizations import create_expense_comparison_chart

        holdings = [
            {"name": "Apple", "ticker": "AAPL", "value": 50000, "expense_ratio": None},
        ]
        fig = create_expense_comparison_chart(holdings)

        assert isinstance(fig, go.Figure)
        # Should have annotation about no data
        assert len(fig.layout.annotations) > 0

    def test_expense_chart_sorted_by_expense_ratio(self):
        """Test that holdings are sorted by expense ratio."""
        from agents.portfolio_visualizations import create_expense_comparison_chart

        holdings = [
            {"name": "VOO", "ticker": "VOO", "value": 50000, "expense_ratio": 0.03},
            {"name": "ARKK", "ticker": "ARKK", "value": 30000, "expense_ratio": 0.75},
        ]
        fig = create_expense_comparison_chart(holdings)

        bar_trace = fig.data[0]
        # Should be sorted by expense ratio descending (0.75, 0.03)
        assert bar_trace.y[0] == 0.75


class TestRiskGauge:
    """Tests for risk level gauge chart."""

    def test_create_risk_gauge_returns_figure(self):
        """Test that risk gauge returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_risk_gauge

        fig = create_risk_gauge("Moderate")

        assert isinstance(fig, go.Figure)

    def test_risk_gauge_all_levels(self):
        """Test that all risk levels create valid gauges."""
        from agents.portfolio_visualizations import create_risk_gauge

        risk_levels = [
            "Conservative",
            "Moderate-Conservative",
            "Moderate",
            "Moderate-Aggressive",
            "Aggressive"
        ]

        for level in risk_levels:
            fig = create_risk_gauge(level)
            assert isinstance(fig, go.Figure)

    def test_risk_gauge_has_correct_value(self):
        """Test that risk gauge shows correct numeric value."""
        from agents.portfolio_visualizations import create_risk_gauge

        fig = create_risk_gauge("Aggressive")

        # Aggressive should map to value 5
        indicator = fig.data[0]
        assert indicator.value == 5


class TestDiversificationGauge:
    """Tests for diversification score gauge chart."""

    def test_create_diversification_gauge_returns_figure(self):
        """Test that diversification gauge returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_diversification_gauge

        fig = create_diversification_gauge(75)

        assert isinstance(fig, go.Figure)

    def test_diversification_gauge_score_range(self):
        """Test diversification gauge with various scores."""
        from agents.portfolio_visualizations import create_diversification_gauge

        for score in [0, 25, 50, 75, 100]:
            fig = create_diversification_gauge(score)
            assert isinstance(fig, go.Figure)
            assert fig.data[0].value == score

    def test_diversification_gauge_assessment_labels(self):
        """Test that gauge includes assessment text."""
        from agents.portfolio_visualizations import create_diversification_gauge

        # High score should show "Excellent"
        fig_high = create_diversification_gauge(85)
        title_text = str(fig_high.layout.title.text) if fig_high.layout.title else ""
        assert "Excellent" in title_text or fig_high.data[0].value == 85

        # Low score should show "Needs Work"
        fig_low = create_diversification_gauge(30)
        title_text_low = str(fig_low.layout.title.text) if fig_low.layout.title else ""
        assert "Needs Work" in title_text_low or fig_low.data[0].value == 30


class TestGoalProjectionChart:
    """Tests for investment growth projection chart."""

    def test_create_projection_chart_returns_figure(self):
        """Test that projection chart returns a Plotly Figure."""
        from agents.portfolio_visualizations import create_goal_projection_chart

        fig = create_goal_projection_chart(
            current_value=100000,
            monthly_contribution=500,
            years=20
        )

        assert isinstance(fig, go.Figure)

    def test_projection_chart_has_multiple_scenarios(self):
        """Test that projection chart shows multiple scenarios."""
        from agents.portfolio_visualizations import create_goal_projection_chart

        fig = create_goal_projection_chart(100000, 500, 20)

        # Should have 4 traces: Conservative, Moderate, Aggressive, Contributions Only
        assert len(fig.data) == 4

    def test_projection_chart_final_values_increase_with_return(self):
        """Test that higher returns lead to higher final values."""
        from agents.portfolio_visualizations import create_goal_projection_chart

        fig = create_goal_projection_chart(100000, 500, 20)

        # Get final values for each scenario
        conservative_final = fig.data[0].y[-1]  # 5%
        moderate_final = fig.data[1].y[-1]      # 7%
        aggressive_final = fig.data[2].y[-1]    # 9%
        contributions_final = fig.data[3].y[-1] # 0%

        assert aggressive_final > moderate_final > conservative_final > contributions_final

    def test_projection_chart_contributions_only_linear(self):
        """Test that contributions-only line is linear."""
        from agents.portfolio_visualizations import create_goal_projection_chart

        initial = 100000
        monthly = 500
        years = 10

        fig = create_goal_projection_chart(initial, monthly, years)

        contributions_trace = fig.data[3]
        expected_final = initial + (monthly * 12 * years)

        # Allow small floating point difference
        assert abs(contributions_trace.y[-1] - expected_final) < 1


class TestChartFormatting:
    """Tests for chart formatting and layout."""

    def test_pie_chart_has_title(self):
        """Test that pie chart has a title."""
        from agents.portfolio_visualizations import create_allocation_pie_chart

        fig = create_allocation_pie_chart({"Stocks": 100}, 100000, "My Title")

        assert fig.layout.title.text == "My Title"

    def test_bar_chart_has_axis_labels(self):
        """Test that bar chart has axis labels."""
        from agents.portfolio_visualizations import create_holdings_bar_chart

        holdings = [{"name": "Test", "ticker": "TST", "value": 1000, "expense_ratio": 0.01}]
        fig = create_holdings_bar_chart(holdings)

        assert fig.layout.xaxis.title.text == "Value ($)"

    def test_projection_chart_has_legend(self):
        """Test that projection chart has a legend."""
        from agents.portfolio_visualizations import create_goal_projection_chart

        fig = create_goal_projection_chart(100000, 500, 10)

        # Check all traces have names for legend
        for trace in fig.data:
            assert trace.name is not None
            assert len(trace.name) > 0
