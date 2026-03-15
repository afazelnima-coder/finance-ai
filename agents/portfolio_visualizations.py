"""
Portfolio Visualization Module
Generates charts for portfolio analysis using Plotly.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Optional
import math


def create_allocation_pie_chart(
    allocations: Dict[str, float],
    total_value: float,
    title: str = "Asset Allocation"
) -> go.Figure:
    """
    Create a pie chart showing asset allocation.

    Args:
        allocations: Dict of {asset_class: percentage}
        total_value: Total portfolio value for showing dollar amounts
        title: Chart title

    Returns:
        Plotly Figure object
    """
    labels = list(allocations.keys())
    values = list(allocations.values())

    # Calculate dollar values for hover
    dollar_values = [total_value * pct / 100 for pct in values]

    # Custom colors for asset classes
    color_map = {
        "US Stocks": "#2E86AB",
        "International Stocks": "#A23B72",
        "Bonds": "#F18F01",
        "Cash": "#C73E1D",
        "Real Estate": "#3B1F2B",
        "Commodities": "#95190C",
        "Cryptocurrency": "#610345",
        "Mixed/Target Date": "#44AF69",
    }

    colors = [color_map.get(label, "#666666") for label in labels]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
        textinfo='label+percent',
        textposition='outside',
        hovertemplate="<b>%{label}</b><br>" +
                      "Allocation: %{percent}<br>" +
                      "Value: $%{customdata:,.0f}<extra></extra>",
        customdata=dollar_values
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=20, r=20, t=60, b=80),
        annotations=[dict(
            text=f"${total_value:,.0f}",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )]
    )

    return fig


def create_holdings_bar_chart(
    holdings: List[Dict],
    title: str = "Holdings Breakdown"
) -> go.Figure:
    """
    Create a horizontal bar chart showing individual holdings.

    Args:
        holdings: List of dicts with 'name', 'value', 'ticker', 'expense_ratio'
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Sort by value descending
    sorted_holdings = sorted(holdings, key=lambda x: x.get('value', 0), reverse=True)

    names = []
    values = []
    colors = []
    expense_ratios = []

    for h in sorted_holdings:
        name = h.get('name', 'Unknown')
        ticker = h.get('ticker')
        if ticker:
            name = f"{name} ({ticker})"
        names.append(name)
        values.append(h.get('value', 0))
        expense_ratios.append(h.get('expense_ratio'))

        # Color based on whether it has expense ratio data
        if h.get('expense_ratio') is not None:
            colors.append("#2E86AB")
        else:
            colors.append("#A23B72")

    fig = go.Figure(data=[go.Bar(
        y=names,
        x=values,
        orientation='h',
        marker_color=colors,
        text=[f"${v:,.0f}" for v in values],
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>" +
                      "Value: $%{x:,.0f}<br>" +
                      "<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Value ($)",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(holdings) * 50),
        margin=dict(l=20, r=100, t=60, b=40),
        showlegend=False
    )

    return fig


def create_expense_comparison_chart(
    holdings: List[Dict],
    title: str = "Expense Ratios by Holding"
) -> go.Figure:
    """
    Create a bar chart comparing expense ratios across holdings.

    Args:
        holdings: List of dicts with 'name', 'ticker', 'expense_ratio', 'value'
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Filter holdings with expense ratios and sort by expense ratio
    holdings_with_er = [h for h in holdings if h.get('expense_ratio') is not None]
    sorted_holdings = sorted(holdings_with_er, key=lambda x: x.get('expense_ratio', 0), reverse=True)

    if not sorted_holdings:
        fig = go.Figure()
        fig.add_annotation(
            text="No expense ratio data available",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14)
        )
        fig.update_layout(height=200)
        return fig

    names = []
    expense_ratios = []
    values = []

    for h in sorted_holdings:
        name = h.get('name', 'Unknown')
        ticker = h.get('ticker')
        if ticker:
            name = f"{ticker}"
        names.append(name)
        expense_ratios.append(h.get('expense_ratio', 0))
        values.append(h.get('value', 0))

    # Color gradient based on expense ratio (green=low, red=high)
    colors = []
    for er in expense_ratios:
        if er <= 0.05:
            colors.append("#44AF69")  # Green - very low
        elif er <= 0.15:
            colors.append("#F18F01")  # Orange - moderate
        else:
            colors.append("#C73E1D")  # Red - high

    fig = go.Figure(data=[go.Bar(
        x=names,
        y=expense_ratios,
        marker_color=colors,
        text=[f"{er:.2f}%" for er in expense_ratios],
        textposition='outside',
        hovertemplate="<b>%{x}</b><br>" +
                      "Expense Ratio: %{y:.3f}%<br>" +
                      "<extra></extra>"
    )])

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        yaxis_title="Expense Ratio (%)",
        xaxis_title="",
        height=350,
        margin=dict(l=20, r=20, t=60, b=40),
        showlegend=False
    )

    # Add reference line for average expense ratio
    avg_er = sum(expense_ratios) / len(expense_ratios)
    fig.add_hline(
        y=avg_er,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"Avg: {avg_er:.2f}%",
        annotation_position="right"
    )

    return fig


def create_risk_gauge(
    risk_level: str,
    title: str = "Risk Level"
) -> go.Figure:
    """
    Create a gauge chart showing portfolio risk level.

    Args:
        risk_level: One of "Conservative", "Moderate-Conservative", "Moderate",
                   "Moderate-Aggressive", "Aggressive"
        title: Chart title

    Returns:
        Plotly Figure object
    """
    risk_values = {
        "Conservative": 1,
        "Moderate-Conservative": 2,
        "Moderate": 3,
        "Moderate-Aggressive": 4,
        "Aggressive": 5
    }

    value = risk_values.get(risk_level, 3)

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 18}},
        number={'suffix': f" - {risk_level}", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [0, 5], 'tickvals': [1, 2, 3, 4, 5],
                    'ticktext': ['Cons.', 'Mod-Cons.', 'Mod.', 'Mod-Agg.', 'Agg.']},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 1], 'color': "#44AF69"},
                {'range': [1, 2], 'color': "#90BE6D"},
                {'range': [2, 3], 'color': "#F9C74F"},
                {'range': [3, 4], 'color': "#F8961E"},
                {'range': [4, 5], 'color': "#F94144"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=60, b=20)
    )

    return fig


def create_diversification_gauge(
    score: int,
    title: str = "Diversification Score"
) -> go.Figure:
    """
    Create a gauge chart showing diversification score.

    Args:
        score: Diversification score (0-100)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    # Determine assessment
    if score >= 80:
        assessment = "Excellent"
        color = "#44AF69"
    elif score >= 60:
        assessment = "Good"
        color = "#90BE6D"
    elif score >= 40:
        assessment = "Fair"
        color = "#F9C74F"
    else:
        assessment = "Needs Work"
        color = "#F94144"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': f"{title}<br><span style='font-size:14px;color:gray'>{assessment}</span>",
               'font': {'size': 18}},
        number={'suffix': "/100", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'tickvals': [0, 25, 50, 75, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': "#FFE5E5"},
                {'range': [40, 60], 'color': "#FFF3E0"},
                {'range': [60, 80], 'color': "#E8F5E9"},
                {'range': [80, 100], 'color': "#C8E6C9"}
            ],
        }
    ))

    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=80, b=20)
    )

    return fig


def create_goal_projection_chart(
    current_value: float,
    monthly_contribution: float,
    years: int,
    expected_return: float = 0.07,
    title: str = "Investment Growth Projection"
) -> go.Figure:
    """
    Create a line chart showing projected portfolio growth over time.

    Args:
        current_value: Current portfolio value
        monthly_contribution: Monthly contribution amount
        years: Number of years to project
        expected_return: Expected annual return (default 7%)
        title: Chart title

    Returns:
        Plotly Figure object
    """
    months = years * 12
    monthly_return = (1 + expected_return) ** (1/12) - 1

    # Calculate projections for different scenarios
    scenarios = {
        'Conservative (5%)': 0.05,
        'Moderate (7%)': 0.07,
        'Aggressive (9%)': 0.09
    }

    fig = go.Figure()

    colors = ['#44AF69', '#2E86AB', '#A23B72']

    for idx, (scenario_name, annual_return) in enumerate(scenarios.items()):
        monthly_ret = (1 + annual_return) ** (1/12) - 1
        values = []
        value = current_value

        for month in range(months + 1):
            values.append(value)
            value = value * (1 + monthly_ret) + monthly_contribution

        years_list = [m / 12 for m in range(months + 1)]

        fig.add_trace(go.Scatter(
            x=years_list,
            y=values,
            mode='lines',
            name=scenario_name,
            line=dict(color=colors[idx], width=2 if 'Moderate' in scenario_name else 1),
            hovertemplate=f"<b>{scenario_name}</b><br>" +
                          "Year: %{x:.1f}<br>" +
                          "Value: $%{y:,.0f}<extra></extra>"
        ))

    # Add contribution line (no growth)
    contribution_only = [current_value + monthly_contribution * m for m in range(months + 1)]
    fig.add_trace(go.Scatter(
        x=[m / 12 for m in range(months + 1)],
        y=contribution_only,
        mode='lines',
        name='Contributions Only',
        line=dict(color='gray', width=1, dash='dash'),
        hovertemplate="<b>Contributions Only</b><br>" +
                      "Year: %{x:.1f}<br>" +
                      "Value: $%{y:,.0f}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=18)),
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        hovermode='x unified'
    )

    # Format y-axis with dollar signs
    fig.update_yaxes(tickprefix="$", tickformat=",.0f")

    return fig


def create_portfolio_summary_dashboard(
    total_value: float,
    allocations: Dict[str, float],
    holdings: List[Dict],
    risk_level: str,
    diversification_score: int,
    weighted_expense_ratio: float
) -> go.Figure:
    """
    Create a comprehensive dashboard with multiple visualizations.

    Args:
        total_value: Total portfolio value
        allocations: Dict of {asset_class: percentage}
        holdings: List of holding dicts
        risk_level: Risk level string
        diversification_score: Score 0-100
        weighted_expense_ratio: Weighted average expense ratio

    Returns:
        Plotly Figure object with subplots
    """
    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "pie"}, {"type": "bar"}],
            [{"type": "indicator"}, {"type": "indicator"}]
        ],
        subplot_titles=(
            "Asset Allocation",
            "Holdings Breakdown",
            "Risk Level",
            "Diversification Score"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    # 1. Pie Chart - Asset Allocation
    labels = list(allocations.keys())
    values = list(allocations.values())

    color_map = {
        "US Stocks": "#2E86AB",
        "International Stocks": "#A23B72",
        "Bonds": "#F18F01",
        "Cash": "#C73E1D",
        "Real Estate": "#3B1F2B",
        "Commodities": "#95190C",
        "Cryptocurrency": "#610345",
        "Mixed/Target Date": "#44AF69",
    }
    colors = [color_map.get(label, "#666666") for label in labels]

    fig.add_trace(
        go.Pie(labels=labels, values=values, hole=0.4, marker=dict(colors=colors)),
        row=1, col=1
    )

    # 2. Bar Chart - Holdings
    sorted_holdings = sorted(holdings, key=lambda x: x.get('value', 0), reverse=True)[:8]
    names = [h.get('ticker') or h.get('name', 'Unknown')[:10] for h in sorted_holdings]
    values = [h.get('value', 0) for h in sorted_holdings]

    fig.add_trace(
        go.Bar(x=names, y=values, marker_color="#2E86AB"),
        row=1, col=2
    )

    # 3. Risk Gauge
    risk_values = {
        "Conservative": 1, "Moderate-Conservative": 2, "Moderate": 3,
        "Moderate-Aggressive": 4, "Aggressive": 5
    }
    risk_value = risk_values.get(risk_level, 3)

    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=risk_value,
            gauge={
                'axis': {'range': [0, 5]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 2], 'color': "#44AF69"},
                    {'range': [2, 3], 'color': "#F9C74F"},
                    {'range': [3, 5], 'color': "#F94144"}
                ]
            }
        ),
        row=2, col=1
    )

    # 4. Diversification Gauge
    fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=diversification_score,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#2E86AB"},
                'steps': [
                    {'range': [0, 40], 'color': "#FFE5E5"},
                    {'range': [40, 60], 'color': "#FFF3E0"},
                    {'range': [60, 100], 'color': "#E8F5E9"}
                ]
            }
        ),
        row=2, col=2
    )

    fig.update_layout(
        height=700,
        showlegend=False,
        title=dict(
            text=f"Portfolio Dashboard - Total Value: ${total_value:,.0f}",
            x=0.5,
            font=dict(size=20)
        ),
        margin=dict(l=40, r=40, t=80, b=40)
    )

    return fig
