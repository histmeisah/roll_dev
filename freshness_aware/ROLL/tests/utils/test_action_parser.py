import re

import pytest

from roll.pipeline.agentic.env.parse_action_utils import default_parser_action_func
from roll.pipeline.agentic.tools.action_parser import Qwen3CoderActionParser


SOKOBAN_ACTION_LOOKUP = {1: "Up", 2: "Down", 3: "Left", 4: "Right"}


def test_default_parser_action_func_parses_wrapped_lookup_action():
    parsed = default_parser_action_func(
        text="<answer>Right</answer><|im_end|>",
        action_pattern=r"<answer>(.*?)</answer>",
        action_lookup=SOKOBAN_ACTION_LOOKUP,
        special_token_list=("<|im_end|>",),
    )

    assert parsed["action"] == 4
    assert parsed["action_content"] == "Right"
    assert parsed["think_content"] == ""


def test_default_parser_action_func_falls_back_to_bare_lookup_action():
    parsed = default_parser_action_func(
        text=" right <|im_end|>",
        action_pattern=r"<answer>(.*?)</answer>",
        action_lookup=SOKOBAN_ACTION_LOOKUP,
        special_token_list=("<|im_end|>",),
    )

    assert parsed["action"] == 4
    assert parsed["action_content"] == "right"
    assert parsed["think_content"] == ""


def test_default_parser_action_func_keeps_unknown_bare_action_invalid():
    parsed = default_parser_action_func(
        text="Action<|im_end|>",
        action_pattern=r"<answer>(.*?)</answer>",
        action_lookup=SOKOBAN_ACTION_LOOKUP,
        special_token_list=("<|im_end|>",),
    )

    assert parsed["action"] is None
    assert parsed["action_content"] == ""
    assert parsed["think_content"] == ""


def test_default_parser_action_func_does_not_fallback_without_lookup():
    parsed = default_parser_action_func(
        text="Right<|im_end|>",
        action_pattern=r"<answer>(.*?)</answer>",
        action_lookup=None,
        special_token_list=("<|im_end|>",),
    )

    assert parsed["action"] is None
    assert parsed["action_content"] == ""
    assert parsed["think_content"] == ""


def test_qwen3coder_action_parser_parse_action_single_call():
    tool = Qwen3CoderActionParser()
    response = (
        "Let me check the current directory."
        "<tool_call><function=list_directory><parameter=path>.</parameter></function></tool_call>"
    )

    ok, actions = tool.parse_action(response=response)

    assert ok is True
    assert isinstance(actions, list)
    assert len(actions) == 1

    action = actions[0]
    assert action["type"] == "function"
    assert action["function"]["name"] == "list_directory"
    assert action["function"]["arguments"] == '{"path": "."}'
