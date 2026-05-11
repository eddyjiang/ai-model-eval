import json
import re
import uuid
from pathlib import Path

import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_config() -> tuple[dict, dict, dict]:
    base = Path(__file__).parent / "config"
    groups = load_yaml(base / "groups.yaml")
    templates = load_yaml(base / "templates.yaml")
    models = load_yaml(base / "models.yaml")
    return groups, templates, models


def build_prompt(template: str, group_name: str) -> str:
    return template.replace("{group}", group_name)


def generate_run_id() -> str:
    return str(uuid.uuid4())[:8]


def extract_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown code fences."""
    if not text:
        raise ValueError("extract_json received empty or None text")
    # Strip markdown code fences
    code_block = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if code_block:
        text = code_block.group(1)

    # Find the outermost JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in text: {text[:200]}")

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])

    raise ValueError(f"Malformed JSON in text: {text[:200]}")
