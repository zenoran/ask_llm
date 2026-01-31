from __future__ import annotations

import json
import logging
import re
from typing import Any

from .base import ToolCallRequest, ToolFormatHandler

logger = logging.getLogger(__name__)


_THOUGHT_MARKERS = (
    "thought:",
    "action:",
    "action input:",
    "observation:",
)


def _try_fix_json(json_str: str) -> str | None:
    original = json_str

    open_count = json_str.count("{")
    close_count = json_str.count("}")
    if close_count > open_count:
        excess = close_count - open_count
        while excess > 0 and json_str.rstrip().endswith("}"):
            json_str = json_str.rstrip()[:-1]
            excess -= 1

    if open_count > close_count:
        json_str = json_str + ("}" * (open_count - close_count))

    json_str = re.sub(r",\s*}", "}", json_str)
    json_str = re.sub(r",\s*]", "]", json_str)

    if json_str != original:
        return json_str
    return None


def _load_json_payload(payload: str) -> dict[str, Any] | None:
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        pass

    try:
        import json5  # type: ignore

        return json5.loads(payload)
    except Exception:
        pass

    fixed = _try_fix_json(payload)
    if fixed:
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None

    return None


def _extract_json_object(text: str, start_idx: int) -> tuple[str | None, int | None]:
    brace_idx = text.find("{", start_idx)
    if brace_idx == -1:
        return None, None

    depth = 0
    in_string = False
    escape = False
    for i in range(brace_idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_idx : i + 1], i + 1

    return None, None


def _tool_list_to_prompt(tools: list) -> str:
    lines: list[str] = []
    for tool in tools or []:
        if hasattr(tool, "to_prompt_string"):
            lines.append(tool.to_prompt_string())
        elif isinstance(tool, dict):
            name = tool.get("name", "")
            desc = tool.get("description", "")
            params = tool.get("parameters") or []
            param_lines = []
            for param in params:
                if not isinstance(param, dict):
                    continue
                req = "(required)" if param.get("required", True) else "(optional)"
                param_lines.append(
                    f"    - {param.get('name')} ({param.get('type')}): {param.get('description')} {req}"
                )
            if param_lines:
                lines.append(f"- **{name}**: {desc}\n" + "\n".join(param_lines))
            else:
                lines.append(f"- **{name}**: {desc}")
        else:
            lines.append(str(tool))

    return "\n".join(lines)


def _tool_names(tools: list) -> str:
    names: list[str] = []
    for tool in tools or []:
        if hasattr(tool, "name"):
            names.append(tool.name)
        elif isinstance(tool, dict):
            name = tool.get("name")
            if name:
                names.append(name)
    return ", ".join(names)


class ReActFormatHandler(ToolFormatHandler):
    """Text-based Thought/Action/Action Input tool calling."""

    def get_system_prompt(self, tools: list) -> str:
        tool_descriptions = _tool_list_to_prompt(tools)
        tool_names = _tool_names(tools)
        return (
            "## Tools\n\n"
            "You have access to the following tools:\n\n"
            f"{tool_descriptions}\n\n"
            "## How to Use Tools\n\n"
            "When you need to use a tool, respond with this EXACT format:\n\n"
            "Thought: [explain what you need to do and why]\n"
            "Action: [tool_name from the list above]\n"
            "Action Input: {\"param\": \"value\"}\n\n"
            "Then STOP. The system will execute the tool and provide an Observation.\n\n"
            "## When You Have the Answer\n\n"
            "Thought: I now have the information needed\n"
            "Final Answer: [your complete response to the user]\n\n"
            "## Important Rules\n\n"
            "1. Use EXACTLY ONE tool per response - no more\n"
            "2. STOP after Action Input - do not continue\n"
            "3. Wait for Observation before your next Thought\n"
            f"4. Tool names must be exactly: {tool_names}\n"
            "5. Action Input must be valid JSON\n"
        )

    def get_stop_sequences(self) -> list[str]:
        return ["\nObservation:", "\nObservation"]

    def parse_response(self, response: str) -> tuple[list[ToolCallRequest], str]:
        if not response:
            return [], ""

        action_match = re.search(r"(?im)^\s*Action\s*:\s*(.+?)\s*$", response)
        if action_match:
            tool_name = action_match.group(1).strip()
            search_start = action_match.end()
            action_input_match = re.search(
                r"(?im)^\s*Action\s*Input\s*:\s*", response[search_start:]
            )
            if action_input_match:
                input_start = search_start + action_input_match.end()
                payload, end_idx = _extract_json_object(response, input_start)
                if payload and end_idx:
                    parsed = _load_json_payload(payload)
                    if isinstance(parsed, dict):
                        raw_text = response[action_match.start() : end_idx]
                        remaining = (response[: action_match.start()] + response[end_idx:]).strip()
                        return (
                            [
                                ToolCallRequest(
                                    name=tool_name,
                                    arguments=parsed,
                                    raw_text=raw_text,
                                )
                            ],
                            remaining,
                        )

        final_match = re.search(r"(?is)Final\s*Answer\s*:\s*(.*)$", response)
        if final_match:
            return [], final_match.group(1).strip()

        return [], response.strip()

    def format_result(
        self,
        tool_name: str,
        result: Any,
        tool_call_id: str | None = None,
        error: str | None = None,
    ) -> str:
        if error:
            return f"Observation: ERROR: {error}"

        result_str = self._normalize_result(result)
        return f"Observation: {result_str}"

    def sanitize_response(self, response: str) -> str:
        if not response:
            return ""

        final_match = re.search(r"(?is)Final\s*Answer\s*:\s*(.*)$", response)
        if final_match:
            response = final_match.group(1).strip()

        cleaned_lines = []
        for line in response.splitlines():
            stripped = line.strip().lower()
            if stripped.startswith(_THOUGHT_MARKERS):
                continue
            cleaned_lines.append(line)

        cleaned = "\n".join(cleaned_lines)
        cleaned = re.sub(r"<tool_call>.*?</tool_call>", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
        return cleaned.strip()

    def _normalize_result(self, result: Any) -> str:
        if isinstance(result, (dict, list)):
            return json.dumps(result, indent=2, default=str)
        result_str = str(result)
        # Strip legacy tool_result wrappers if present
        if "<tool_result" in result_str:
            result_str = re.sub(r"<tool_result[^>]*>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"</tool_result>", "", result_str, flags=re.IGNORECASE)
            result_str = re.sub(r"\\[IMPORTANT:.*?\\]", "", result_str, flags=re.DOTALL)
        return result_str.strip()
