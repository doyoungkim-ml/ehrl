import json

import pytest

from rllm.parser import QwenToolParser, R1ToolParser
from rllm.tools.tool_base import ToolCall


class TestQwenToolParser:
    @pytest.fixture
    def parser(self):
        return QwenToolParser()

    def test_empty_response(self, parser):
        """Test parsing empty response."""
        result = parser.parse("")
        assert len(result) == 0

    def test_no_tool_calls(self, parser):
        """Test response with no tool calls."""
        response = "This is a normal response without any tool calls."
        result = parser.parse(response)
        assert len(result) == 0

    def test_single_valid_tool_call(self, parser):
        """Test parsing a single valid tool call."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York", "unit": "celsius"}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].name == "search_weather"
        assert result[0].arguments == {"location": "New York", "unit": "celsius"}

    def test_multiple_valid_tool_calls(self, parser):
        """Test parsing multiple valid tool calls."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York"}}</tool_call>
        <tool_call>{"name": "search_restaurants", "arguments": {"location": "New York", "cuisine": "Italian"}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 2
        assert result[0].name == "search_weather"
        assert result[1].name == "search_restaurants"

    def test_invalid_json_tool_call(self, parser):
        """Test parsing tool call with invalid JSON."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {invalid json}}</tool_call>
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_tool_call_end(self, parser):
        """Test parsing tool call with missing end tag."""
        response = """
        <tool_call>{"name": "search_weather", "arguments": {"location": "New York"}}
        """
        result = parser.parse(response)
        assert len(result) == 0

    @pytest.mark.parametrize(
        "payload",
        [
            {},
            {"arguments": {"code": "print('bad')"}},
            {"name": "execute"},
            {"name": None, "arguments": {}},
            {"name": 1, "arguments": {}},
            {"name": [], "arguments": {}},
            {"name": "", "arguments": {}},
            {"name": "  ", "arguments": {}},
            {"name": "execute", "arguments": None},
            {"name": "execute", "arguments": []},
            {"name": "execute", "arguments": "{}"},
            {"name": "execute", "arguments": 1},
            {"name": "execute", "arguments": True},
            None,
            [],
            "execute",
            1,
            True,
        ],
    )
    def test_malformed_call_is_skipped_without_losing_valid_calls(self, parser, payload):
        response = (
            '<tool_call>{"name": "first", "arguments": {"code": "print(1)"}}</tool_call>\n'
            f"<tool_call>{json.dumps(payload)}</tool_call>\n"
            '<tool_call>{"name": "last", "arguments": {}}</tool_call>'
        )
        assert parser.parse(response) == [
            ToolCall(name="first", arguments={"code": "print(1)"}),
            ToolCall(name="last", arguments={}),
        ]
        assert parser.parse(f"<tool_call>{json.dumps(payload)}</tool_call>") == []

    def test_get_tool_prompt(self, parser):
        """Test tool prompt generation."""
        tools_schema = """
        {
            "name": "search_weather",
            "description": "Search for weather information",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                }
            }
        }
        """
        prompt = parser.get_tool_prompt(tools_schema)
        assert "<tools>" in prompt
        assert tools_schema in prompt
        assert "<tool_call>" in prompt


class TestR1ToolParser:
    @pytest.fixture
    def parser(self):
        return R1ToolParser()

    def test_empty_response(self, parser):
        """Test parsing empty response."""
        result = parser.parse("")
        assert len(result) == 0

    def test_no_tool_calls(self, parser):
        """Test response with no tool calls."""
        response = "This is a normal response without any tool calls."
        result = parser.parse(response)
        assert len(result) == 0

    def test_single_valid_tool_call(self, parser):
        """Test parsing a single valid tool call."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York", "unit": "celsius"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 1
        assert isinstance(result[0], ToolCall)
        assert result[0].name == "search_weather"
        assert result[0].arguments == {"location": "New York", "unit": "celsius"}

    def test_multiple_valid_tool_calls(self, parser):
        """Test parsing multiple valid tool calls."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York"}}
        ```
        {parser.tool_call_end}
        {parser.tool_call_begin}function{parser.tool_sep}search_restaurants
        ```json
        {{"location": "New York", "cuisine": "Italian"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 2
        assert result[0].name == "search_weather"
        assert result[1].name == "search_restaurants"

    def test_invalid_json_tool_call(self, parser):
        """Test parsing tool call with invalid JSON."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York", invalid json}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_tool_call_end(self, parser):
        """Test parsing tool call with missing end tag."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        ```json
        {{"location": "New York"}}
        ```
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_function_prefix(self, parser):
        """Test parsing tool call with missing function prefix."""
        response = f"""
        {parser.tool_call_begin}search_weather
        ```json
        {{"location": "New York"}}
        ```
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0

    def test_missing_json_block(self, parser):
        """Test parsing tool call with missing JSON block."""
        response = f"""
        {parser.tool_call_begin}function{parser.tool_sep}search_weather
        {parser.tool_call_end}
        """
        result = parser.parse(response)
        assert len(result) == 0
