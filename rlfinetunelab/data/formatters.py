"""
Data formatters converting raw datasets (Alpaca, ShareGPT, custom) into standard ChatML and TRL formats.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class StandardMessage(BaseModel):
    role: str
    content: str


def format_alpaca_to_chatml(
    example: Dict[str, Any],
    instruction_key: str = "instruction",
    input_key: str = "input",
    output_key: str = "output",
    system_prompt: Optional[str] = "You are a helpful, respectful, and honest assistant."
) -> List[Dict[str, str]]:
    """Converts classic Alpaca format (instruction, input, output) to ChatML messages list."""
    instruction = str(example.get(instruction_key, "")).strip()
    input_text = str(example.get(input_key, "")).strip()
    output_text = str(example.get(output_key, "")).strip()

    if input_text:
        user_content = f"{instruction}\n\nContext:\n{input_text}"
    else:
        user_content = instruction

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})
    messages.append({"role": "assistant", "content": output_text})
    return messages


def format_sharegpt_to_chatml(
    example: Dict[str, Any],
    conversations_key: str = "conversations",
    system_prompt: Optional[str] = None
) -> List[Dict[str, str]]:
    """Converts ShareGPT conversation list (from/value) to standard ChatML messages."""
    raw_convs = example.get(conversations_key, [])
    messages: List[Dict[str, str]] = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    role_mapping = {
        "human": "user",
        "user": "user",
        "gpt": "assistant",
        "assistant": "assistant",
        "system": "system",
    }

    for turn in raw_convs:
        raw_role = turn.get("from", turn.get("role", "")).lower()
        content = turn.get("value", turn.get("content", ""))
        mapped_role = role_mapping.get(raw_role, "user")
        messages.append({"role": mapped_role, "content": str(content).strip()})

    return messages


def apply_chatml_template(
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = False
) -> str:
    """Renders ChatML text representation if tokenizer chat template is not available."""
    text = ""
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    if add_generation_prompt:
        text += "<|im_start|>assistant\n"

    return text
