"""
Tests for the router agent functionality.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestRouterNode:
    """Tests for the router_node function."""

    def test_routes_to_qa_for_definitions(self, mock_openai_llm):
        """Test that definition questions route to QA agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("qa")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="What is a stock?")]}
            result = router_node(state)

            assert result["next_agent"] == "qa"

    def test_routes_to_market_for_prices(self, mock_openai_llm):
        """Test that price questions route to Market agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("market")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="What's AAPL's price?")]}
            result = router_node(state)

            assert result["next_agent"] == "market"

    def test_routes_to_news_for_news_queries(self, mock_openai_llm):
        """Test that news queries route to News agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("news")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="Latest financial news")]}
            result = router_node(state)

            assert result["next_agent"] == "news"

    def test_routes_to_tax_for_tax_questions(self, mock_openai_llm):
        """Test that tax questions route to Tax agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("tax")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="How do capital gains taxes work?")]}
            result = router_node(state)

            assert result["next_agent"] == "tax"

    def test_routes_to_goal_for_planning(self, mock_openai_llm):
        """Test that planning questions route to Goal agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("goal")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="How should I plan for retirement?")]}
            result = router_node(state)

            assert result["next_agent"] == "goal"

    def test_routes_to_portfolio_for_allocation(self, mock_openai_llm):
        """Test that portfolio questions route to Portfolio agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("portfolio")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="How should I allocate my investments?")]}
            result = router_node(state)

            assert result["next_agent"] == "portfolio"

    def test_defaults_to_qa_for_invalid_response(self, mock_openai_llm):
        """Test that invalid LLM responses default to QA agent."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("invalid_agent")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="Some question")]}
            result = router_node(state)

            assert result["next_agent"] == "qa"

    def test_handles_response_with_whitespace(self, mock_openai_llm):
        """Test that responses with whitespace are handled correctly."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("  market  \n")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="Stock price")]}
            result = router_node(state)

            assert result["next_agent"] == "market"

    def test_handles_uppercase_response(self, mock_openai_llm):
        """Test that uppercase responses are handled correctly."""
        with patch("agents.router_agent_v2.llm", mock_openai_llm("MARKET")):
            from agents.router_agent_v2 import router_node

            state = {"messages": [HumanMessage(content="Stock price")]}
            result = router_node(state)

            assert result["next_agent"] == "market"


class TestRouteToAgent:
    """Tests for the route_to_agent conditional edge function."""

    def test_returns_next_agent_from_state(self):
        """Test that route_to_agent returns the agent from state."""
        from agents.router_agent_v2 import route_to_agent

        agents = ["qa", "market", "news", "tax", "goal", "portfolio"]
        for agent in agents:
            state = {"next_agent": agent}
            result = route_to_agent(state)
            assert result == agent


class TestAgentNodes:
    """Tests for individual agent node functions."""

    def test_qa_node_invokes_qa_agent(self):
        """Test that qa_node invokes the QA agent."""
        with patch("agents.router_agent_v2.qa_agent") as mock_qa:
            mock_response = {"messages": [AIMessage(content="Stock definition")]}
            mock_qa.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import qa_node

            state = {"messages": [HumanMessage(content="What is a stock?")]}
            result = qa_node(state)

            mock_qa.agent.invoke.assert_called_once()
            assert "messages" in result
            assert len(result["messages"]) == 1

    def test_market_node_invokes_market_agent(self):
        """Test that market_node invokes the Market agent."""
        with patch("agents.router_agent_v2.market_agent") as mock_market:
            mock_response = {"messages": [AIMessage(content="AAPL is at $150")]}
            mock_market.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import market_node

            state = {"messages": [HumanMessage(content="Apple stock price")]}
            result = market_node(state)

            mock_market.agent.invoke.assert_called_once()
            assert "messages" in result

    def test_news_node_invokes_news_agent(self):
        """Test that news_node invokes the News agent."""
        with patch("agents.router_agent_v2.news_agent") as mock_news:
            mock_response = {"messages": [AIMessage(content="Latest news...")]}
            mock_news.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import news_node

            state = {"messages": [HumanMessage(content="Financial news")]}
            result = news_node(state)

            mock_news.agent.invoke.assert_called_once()
            assert "messages" in result

    def test_tax_node_invokes_tax_agent(self):
        """Test that tax_node invokes the Tax agent."""
        with patch("agents.router_agent_v2.tax_agent") as mock_tax:
            mock_response = {"messages": [AIMessage(content="Tax info...")]}
            mock_tax.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import tax_node

            state = {"messages": [HumanMessage(content="Tax question")]}
            result = tax_node(state)

            mock_tax.agent.invoke.assert_called_once()
            assert "messages" in result

    def test_goal_node_invokes_goal_agent(self):
        """Test that goal_node invokes the Goal agent."""
        with patch("agents.router_agent_v2.goal_agent") as mock_goal:
            mock_response = {"messages": [AIMessage(content="Goal planning...")]}
            mock_goal.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import goal_node

            state = {"messages": [HumanMessage(content="Retirement planning")]}
            result = goal_node(state)

            mock_goal.agent.invoke.assert_called_once()
            assert "messages" in result

    def test_portfolio_node_invokes_portfolio_agent(self):
        """Test that portfolio_node invokes the Portfolio agent."""
        with patch("agents.router_agent_v2.portfolio_agent") as mock_portfolio:
            mock_response = {"messages": [AIMessage(content="Portfolio advice...")]}
            mock_portfolio.agent.invoke.return_value = mock_response

            from agents.router_agent_v2 import portfolio_node

            state = {"messages": [HumanMessage(content="Portfolio allocation")]}
            result = portfolio_node(state)

            mock_portfolio.agent.invoke.assert_called_once()
            assert "messages" in result


class TestStateGraphStructure:
    """Tests for the StateGraph workflow structure."""

    def test_graph_has_required_nodes(self):
        """Test that the compiled graph has all required nodes."""
        from agents.router_agent_v2 import agent

        graph = agent.get_graph()
        # graph.nodes is a dict where keys are node names (strings)
        node_names = list(graph.nodes.keys())

        required_nodes = ["guardrail", "off_topic", "router", "qa", "market", "news", "tax", "goal", "portfolio"]
        for node in required_nodes:
            assert node in node_names, f"Missing node: {node}"

    def test_graph_starts_with_guardrail(self):
        """Test that the graph starts with the guardrail node."""
        from agents.router_agent_v2 import agent

        graph = agent.get_graph()

        # Find the start node edges
        start_edges = [e for e in graph.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "guardrail"

    def test_graph_can_be_visualized(self):
        """Test that the graph can generate mermaid visualization."""
        from agents.router_agent_v2 import agent

        mermaid = agent.get_graph().draw_mermaid()

        assert "guardrail" in mermaid
        assert "router" in mermaid
        assert "off_topic" in mermaid
