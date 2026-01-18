"""
Tests for the guardrails functionality.
Tests both the validator in router_agent_v2.py and is_finance_related in streamlit_app.py
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage


class TestFinanceTopicValidator:
    """Tests for the FinanceTopicValidator class."""

    def test_validator_passes_finance_topic(self, mock_openai_llm):
        """Test that finance-related queries pass validation."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            MockLLM.return_value = mock_openai_llm("yes")

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_openai_llm("yes")

            result = validator.validate("What is a stock?")

            assert result.outcome == "pass"

    def test_validator_fails_non_finance_topic(self, mock_openai_llm):
        """Test that non-finance queries fail validation."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            MockLLM.return_value = mock_openai_llm("no")

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_openai_llm("no")

            result = validator.validate("Best pizza recipe?")

            assert result.outcome == "fail"
            assert "not related to finance" in result.error_message

    def test_validator_handles_lowercase_yes(self, mock_openai_llm):
        """Test that validator handles 'Yes' with different cases."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            MockLLM.return_value = mock_openai_llm("YES")

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_openai_llm("YES")

            result = validator.validate("How do bonds work?")

            assert result.outcome == "pass"

    def test_validator_handles_whitespace(self, mock_openai_llm):
        """Test that validator handles responses with whitespace."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            MockLLM.return_value = mock_openai_llm("  yes  \n")

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_openai_llm("  yes  \n")

            result = validator.validate("What is inflation?")

            assert result.outcome == "pass"

    def test_validator_valid_topics_list(self):
        """Test that validator has expected valid topics configured."""
        with patch("agents.router_agent_v2.ChatOpenAI"):
            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()

            expected_topics = ["stocks", "bonds", "taxes", "cryptocurrency", "portfolio management"]
            for topic in expected_topics:
                assert topic in validator.valid_topics


class TestGuardrailNode:
    """Tests for the guardrail_node function."""

    def test_guardrail_returns_on_topic_true(self):
        """Test that guardrail_node sets is_on_topic True for finance queries."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {"messages": [HumanMessage(content="What is a stock?")]}
            result = guardrail_node(state)

            assert result["is_on_topic"] is True

    def test_guardrail_returns_on_topic_false(self):
        """Test that guardrail_node sets is_on_topic False for non-finance queries."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = False
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {"messages": [HumanMessage(content="Best pizza recipe?")]}
            result = guardrail_node(state)

            assert result["is_on_topic"] is False

    def test_guardrail_handles_exception(self):
        """Test that guardrail_node defaults to on_topic on error."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_guard.validate.side_effect = Exception("API Error")

            from agents.router_agent_v2 import guardrail_node

            state = {"messages": [HumanMessage(content="Test query")]}
            result = guardrail_node(state)

            # Should default to True on error
            assert result["is_on_topic"] is True

    def test_guardrail_handles_empty_messages(self):
        """Test that guardrail_node handles empty message list."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {"messages": []}
            result = guardrail_node(state)

            # Should validate empty string
            mock_guard.validate.assert_called_once_with("")


class TestCheckTopic:
    """Tests for the check_topic conditional function."""

    def test_check_topic_returns_router_when_on_topic(self):
        """Test that check_topic returns 'router' for on-topic queries."""
        from agents.router_agent_v2 import check_topic

        state = {"is_on_topic": True}
        result = check_topic(state)

        assert result == "router"

    def test_check_topic_returns_off_topic_when_not_on_topic(self):
        """Test that check_topic returns 'off_topic' for off-topic queries."""
        from agents.router_agent_v2 import check_topic

        state = {"is_on_topic": False}
        result = check_topic(state)

        assert result == "off_topic"

    def test_check_topic_defaults_to_router(self):
        """Test that check_topic defaults to 'router' when key missing."""
        from agents.router_agent_v2 import check_topic

        state = {}
        result = check_topic(state)

        assert result == "router"


class TestOffTopicNode:
    """Tests for the off_topic_node function."""

    def test_off_topic_returns_message(self):
        """Test that off_topic_node returns appropriate decline message."""
        from agents.router_agent_v2 import off_topic_node
        from langchain_core.messages import AIMessage

        state = {"messages": [HumanMessage(content="Best pizza recipe?")]}
        result = off_topic_node(state)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
        assert "finance" in result["messages"][0].content.lower()
