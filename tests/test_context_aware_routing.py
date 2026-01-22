"""
Tests for context-aware routing and guardrail functionality.
Tests that follow-up questions use conversation context for proper routing.
"""
import pytest
from unittest.mock import patch, MagicMock
from langchain_core.messages import HumanMessage, AIMessage


class TestGuardrailWithContext:
    """Tests for guardrail_node with conversation context."""

    def test_guardrail_passes_context_to_validator(self):
        """Test that guardrail passes conversation context to validator."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            # Simulate conversation with context
            state = {
                "messages": [
                    HumanMessage(content="What is an index fund?"),
                    AIMessage(content="An index fund is a type of mutual fund..."),
                    HumanMessage(content="What are the most common?"),
                ]
            }

            guardrail_node(state)

            # Check that validate was called with metadata containing context
            call_args = mock_guard.validate.call_args
            assert call_args is not None

            # Should have been called with the last message
            assert call_args[0][0] == "What are the most common?"

            # Should have metadata with context
            metadata = call_args.kwargs.get("metadata", {})
            assert "conversation_context" in metadata
            assert "index fund" in metadata["conversation_context"].lower()

    def test_guardrail_without_context_for_first_message(self):
        """Test that first message has no conversation context."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {
                "messages": [
                    HumanMessage(content="What is a stock?"),
                ]
            }

            guardrail_node(state)

            call_args = mock_guard.validate.call_args
            metadata = call_args.kwargs.get("metadata", {})

            # First message should have empty context
            assert metadata.get("conversation_context", "") == ""

    def test_guardrail_truncates_long_context(self):
        """Test that guardrail truncates very long messages in context."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            long_response = "A" * 500  # Very long response

            state = {
                "messages": [
                    HumanMessage(content="What is a stock?"),
                    AIMessage(content=long_response),
                    HumanMessage(content="Tell me more"),
                ]
            }

            guardrail_node(state)

            call_args = mock_guard.validate.call_args
            metadata = call_args.kwargs.get("metadata", {})
            context = metadata.get("conversation_context", "")

            # Context should be truncated (200 chars + "...")
            assert "..." in context
            assert len(context) < 500


class TestRouterWithContext:
    """Tests for router_node with conversation context."""

    def test_router_includes_context_in_prompt(self):
        """Test that router includes conversation context in LLM prompt."""
        with patch("agents.router_agent_v2.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "portfolio"
            mock_llm.invoke.return_value = mock_response

            from agents.router_agent_v2 import router_node

            state = {
                "messages": [
                    HumanMessage(content="Analyze my portfolio"),
                    AIMessage(content="Here's your portfolio analysis..."),
                    HumanMessage(content="What about adding bonds?"),
                ]
            }

            router_node(state)

            # Check the prompt sent to LLM
            call_args = mock_llm.invoke.call_args
            prompt_messages = call_args[0][0]
            prompt_content = prompt_messages[0].content

            # Should include context
            assert "portfolio" in prompt_content.lower()
            assert "Recent conversation" in prompt_content

    def test_router_routes_follow_up_correctly(self):
        """Test that follow-up questions route to same agent."""
        with patch("agents.router_agent_v2.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "portfolio"
            mock_llm.invoke.return_value = mock_response

            from agents.router_agent_v2 import router_node

            # Follow-up to portfolio question
            state = {
                "messages": [
                    HumanMessage(content="Analyze my portfolio: $100K in VOO"),
                    AIMessage(content="Your portfolio analysis..."),
                    HumanMessage(content="Which funds would you recommend?"),
                ]
            }

            result = router_node(state)

            assert result["next_agent"] == "portfolio"

    def test_router_no_context_for_first_message(self):
        """Test that first message doesn't include context in prompt."""
        with patch("agents.router_agent_v2.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "qa"
            mock_llm.invoke.return_value = mock_response

            from agents.router_agent_v2 import router_node

            state = {
                "messages": [
                    HumanMessage(content="What is a stock?"),
                ]
            }

            router_node(state)

            call_args = mock_llm.invoke.call_args
            prompt_messages = call_args[0][0]
            prompt_content = prompt_messages[0].content

            # Should NOT include "Recent conversation" for first message
            assert "Recent conversation" not in prompt_content

    def test_router_validates_agent_choice(self):
        """Test that router validates and defaults invalid agent choices."""
        with patch("agents.router_agent_v2.llm") as mock_llm:
            mock_response = MagicMock()
            mock_response.content = "invalid_agent"
            mock_llm.invoke.return_value = mock_response

            from agents.router_agent_v2 import router_node

            state = {
                "messages": [
                    HumanMessage(content="Test question"),
                ]
            }

            result = router_node(state)

            # Should default to 'qa' for invalid choice
            assert result["next_agent"] == "qa"


class TestFinanceTopicValidatorWithContext:
    """Tests for FinanceTopicValidator with conversation context."""

    def test_validator_uses_context_prompt(self, mock_openai_llm):
        """Test that validator uses context-aware prompt when context provided."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            MockLLM.return_value = mock_openai_llm("yes")

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_openai_llm("yes")

            # Call validate with context
            result = validator.validate(
                "What are the most common?",
                metadata={"conversation_context": "User: What is an index fund?\nAssistant: An index fund is..."}
            )

            assert result.outcome == "pass"

    def test_validator_context_prompt_mentions_follow_up(self, mock_openai_llm):
        """Test that context prompt instructs about follow-up questions."""
        with patch("agents.router_agent_v2.ChatOpenAI") as MockLLM:
            # Create a mock that captures the prompt
            captured_prompt = []

            def capture_invoke(messages):
                captured_prompt.append(messages[0].content)
                mock_response = MagicMock()
                mock_response.content = "yes"
                return mock_response

            mock_llm = MagicMock()
            mock_llm.invoke = capture_invoke
            MockLLM.return_value = mock_llm

            from agents.router_agent_v2 import FinanceTopicValidator
            validator = FinanceTopicValidator()
            validator.llm = mock_llm

            validator.validate(
                "Tell me more",
                metadata={"conversation_context": "Previous discussion about stocks"}
            )

            # Check that prompt mentions follow-up questions
            assert len(captured_prompt) > 0
            assert "follow-up" in captured_prompt[0].lower()


class TestContextBuilding:
    """Tests for conversation context building logic."""

    def test_context_includes_recent_messages(self):
        """Test that context includes recent messages."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {
                "messages": [
                    HumanMessage(content="First question"),
                    AIMessage(content="First answer"),
                    HumanMessage(content="Second question"),
                    AIMessage(content="Second answer"),
                    HumanMessage(content="Third question"),
                ]
            }

            guardrail_node(state)

            call_args = mock_guard.validate.call_args
            metadata = call_args.kwargs.get("metadata", {})
            context = metadata.get("conversation_context", "")

            # Should include previous messages
            assert "First question" in context or "Second question" in context

    def test_context_labels_user_and_assistant(self):
        """Test that context correctly labels User and Assistant messages."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            state = {
                "messages": [
                    HumanMessage(content="User message here"),
                    AIMessage(content="Assistant response here"),
                    HumanMessage(content="Follow up"),
                ]
            }

            guardrail_node(state)

            call_args = mock_guard.validate.call_args
            metadata = call_args.kwargs.get("metadata", {})
            context = metadata.get("conversation_context", "")

            # Should have User and Assistant labels
            assert "User:" in context
            assert "Assistant:" in context

    def test_context_limits_to_recent_messages(self):
        """Test that context is limited to last few messages."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard:
            mock_result = MagicMock()
            mock_result.validation_passed = True
            mock_guard.validate.return_value = mock_result

            from agents.router_agent_v2 import guardrail_node

            # Create many messages
            messages = []
            for i in range(10):
                messages.append(HumanMessage(content=f"Question {i}"))
                messages.append(AIMessage(content=f"Answer {i}"))
            messages.append(HumanMessage(content="Final question"))

            state = {"messages": messages}

            guardrail_node(state)

            call_args = mock_guard.validate.call_args
            metadata = call_args.kwargs.get("metadata", {})
            context = metadata.get("conversation_context", "")

            # Should NOT include very old messages (Question 0)
            # Context should be limited to recent messages
            lines = context.split("\n")
            assert len(lines) <= 8  # At most 4 message pairs


class TestEndToEndRouting:
    """Integration tests for context-aware routing."""

    def test_portfolio_follow_up_routes_correctly(self):
        """Test that portfolio follow-ups route to portfolio agent."""
        with patch("agents.router_agent_v2.finance_guard") as mock_guard, \
             patch("agents.router_agent_v2.llm") as mock_llm:

            # Guardrail passes
            mock_guard_result = MagicMock()
            mock_guard_result.validation_passed = True
            mock_guard.validate.return_value = mock_guard_result

            # Router should return portfolio
            mock_router_response = MagicMock()
            mock_router_response.content = "portfolio"
            mock_llm.invoke.return_value = mock_router_response

            from agents.router_agent_v2 import guardrail_node, router_node

            state = {
                "messages": [
                    HumanMessage(content="Analyze my portfolio: $100K VOO, $50K BND"),
                    AIMessage(content="Here's your portfolio analysis..."),
                    HumanMessage(content="What about international exposure?"),
                ]
            }

            # Run through guardrail
            guardrail_result = guardrail_node(state)
            state.update(guardrail_result)

            # Run through router
            router_result = router_node(state)

            assert router_result["next_agent"] == "portfolio"
