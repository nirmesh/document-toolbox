#!/usr/bin/env python3
"""
word_to_mcp.py

Extract Word document content (paragraphs, headings, TOC, tables, images),
call Groq LLM to convert into structured JSON (schema-driven if provided, or best-effort),
robustly parse the LLM output, and optionally push results to an MCP endpoint.

Requirements:
  pip install python-docx unstructured requests python-dotenv

Usage:
  export GROQ_API_KEY="..."
  export MCP_ENDPOINT="https://mcp.example/api/push"
  export MCP_API_KEY="..."
  python word_to_mcp.py /path/to/file.docx --schema schema.json --model <model-name>
"""

import os
import re
import json
import argparse
import requests
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
from unstructured.partition.docx import partition_docx

# -----------------------
# ENV
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_URL = os.getenv("GROQ_URL", "https://api.groq.com/openai/v1/chat/completions")
MCP_ENDPOINT = os.getenv("MCP_ENDPOINT")
MCP_API_KEY = os.getenv("MCP_API_KEY")


# -----------------------
# DOCX extraction
# -----------------------
def extract_text_from_docx_all(path: str) -> Dict[str, str]:
    """Extracts ALL content (paragraphs, headings, tables, lists, images) from Word."""
    elements = partition_docx(filename=path)
    out = {}
    block_lines = []
    for el in elements:
        block_type = el.category or "Unknown"
        text = el.text.strip() if el.text else ""
        block_lines.append(f"[{block_type}] {text}")
    out["Document"] = "\n".join(block_lines)
    return out


def extract_text_from_docx_by_schema(path: str, schema: Dict[str, Any]) -> Dict[str, str]:
    """
    Extracts only content relevant to schema keys.
    Example schema:
      {
        "Participants": [ { "Name": "string", "Role": "string" } ],
        "Summary": [ { "Topic": "string", "Details": "string" } ]
      }
    """
    elements = partition_docx(filename=path)
    text_blocks = {}
    doc_text = "\n".join([f"{el.category}: {el.text}" for el in elements if el.text])
    for section in schema.keys():
        # Very simple matching: include whole text for now, let LLM filter
        text_blocks[section] = f"Section: {section}\n{doc_text}"
    return text_blocks


# -----------------------
# Prompt creation
# -----------------------
def create_prompt(section: str, text: str, fields_schema: Optional[List[Dict[str, Any]]] = None) -> str:
    if fields_schema:
        schema_block = json.dumps(fields_schema, indent=2)
        return f"""
You are an intelligent document parser.

Extract information from section '{section}' strictly matching the fields below.
Return ONLY valid JSON, no markdown or comments.
Exclude any fields not in the schema.

Schema for '{section}':
{schema_block}

Section text:
\"\"\"{text}\"\"\"

Return JSON like:
{{ "{section}": [ {{ "field1": value1, ... }} ] }}
"""
    else:
        return f"""
You are a JSON extraction agent.

Convert the following Word content from section '{section}' into structured JSON.
Return ONLY valid JSON, no explanations or markdown.

Section text:
\"\"\"{text}\"\"\"
"""


# -----------------------
# LLM call
# -----------------------
def call_groq_llm(prompt: str, model: str) -> str:
    if not GROQ_API_KEY:
        raise EnvironmentError("GROQ_API_KEY is not set.")
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
    }
    r = requests.post(GROQ_URL, headers=headers, json=payload)
    r.raise_for_status()
    parsed = r.json()
    if "choices" in parsed and parsed["choices"]:
        return parsed["choices"][0]["message"]["content"]
    if "text" in parsed:
        return parsed["text"]
    return json.dumps(parsed)


# -----------------------
# Robust JSON parsing
# -----------------------
def _find_top_level_json_blocks(s: str) -> List[Tuple[int, int]]:
    starts, blocks, stack = [], [], []
    for i, ch in enumerate(s):
        if ch in "{[":
            stack.append((ch, i))
        elif ch in "}]":
            if not stack:
                continue
            opening, start_idx = stack.pop()
            if (opening == "{" and ch == "}") or (opening == "[" and ch == "]"):
                if not stack:
                    blocks.append((start_idx, i + 1))
    return blocks


def _attempt_json_load(candidate: str) -> Any:
    cand = candidate.strip()
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        cand = re.sub(r"^```(?:json)?\s*", "", cand, flags=re.IGNORECASE)
        cand = re.sub(r"\s*```$", "", cand, flags=re.IGNORECASE)
        cand = re.sub(r"^[^\{\[\n]*\n", "", cand)
        cand = re.sub(r",\s*([\]\}])", r"\1", cand)
        if "'" in cand and '"' not in cand:
            try:
                return json.loads(cand.replace("'", '"'))
            except Exception:
                pass
        return json.loads(cand)


def extract_json_from_llm_output(raw_output: str) -> Any:
    if not raw_output:
        return {"raw_output": ""}
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw_output, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    cleaned = cleaned.strip()
    blocks = _find_top_level_json_blocks(cleaned)
    if not blocks:
        m = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", cleaned)
        if not m:
            return {"raw_output": cleaned}
        return _attempt_json_load(m.group(1))
    for start, end in reversed(blocks):
        try:
            return _attempt_json_load(cleaned[start:end])
        except Exception:
            continue
    return {"raw_output": cleaned}


# -----------------------
# Push to MCP
# -----------------------
def push_to_mcp(mcp_endpoint: str, api_key: Optional[str], payload: Dict[str, Any]) -> Dict[str, Any]:
    if not mcp_endpoint:
        raise EnvironmentError("MCP_ENDPOINT not provided.")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    r = requests.post(mcp_endpoint, headers=headers, json=payload, timeout=30)
    r.raise_for_status()
    try:
        return r.json()
    except Exception:
        return {"status": "ok", "raw_response_text": r.text}


# -----------------------
# Orchestration
# -----------------------
def run_word_extraction(path: str, schema_path: Optional[str], model: str, push_to_mcp_flag: bool = False):
    schema = None
    if schema_path:
        if not os.path.exists(schema_path):
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        with open(schema_path, "r") as f:
            schema = json.load(f)

    if schema:
        section_texts = extract_text_from_docx_by_schema(path, schema)
    else:
        section_texts = extract_text_from_docx_all(path)

    results = {}
    for section, block in section_texts.items():
        fields_schema = schema.get(section) if schema else None
        prompt = create_prompt(section, block, fields_schema)
        print(f"[i] Calling LLM for section '{section}'...")
        raw_output = call_groq_llm(prompt, model)
        parsed = extract_json_from_llm_output(raw_output)

        if isinstance(parsed, dict) and section in parsed:
            results[section] = parsed[section]
        else:
            results[section] = parsed

        if push_to_mcp_flag:
            final_payload = {"file": os.path.basename(path), "section": section, "data": results[section]}
            try:
                resp = push_to_mcp(MCP_ENDPOINT, MCP_API_KEY, final_payload)
                print(f"[i] Pushed section '{section}' to MCP. Response: {resp}")
            except Exception as e:
                print(f"[⚠] Failed to push section '{section}' to MCP: {e}")

    full = {"file": os.path.basename(path), "results": results}
    print(json.dumps(full, indent=2, ensure_ascii=False))
    return full


# -----------------------
# CLI
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to Word (.docx) file")
    parser.add_argument("--schema", help="Optional schema JSON file path")
    parser.add_argument("--model", default="meta-llama/llama-4-maverick-17b-128e-instruct", help="LLM model")
    parser.add_argument("--push-to-mcp", action="store_true", help="Push results to MCP endpoint")
    args = parser.parse_args()

    run_word_extraction(args.path, args.schema, args.model, push_to_mcp_flag=args.push_to_mcp)
