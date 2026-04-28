#!/usr/bin/env python3
"""
Agent0_new — MBPP Curriculum Designer
=====================================

Redesign of Agent0_new for code tasks from MBPP-style JSONL/JSON/Parquet data.

What this script does:
  1) Define taxonomy and difficulty for each problem
  2) Build curriculum tasks from templates
  3) Generate / attach solution + explanation
  4) Verify solution by executing code against tests
  5) If verification fails, optionally repair via LLM
  6) Save accepted examples into a knowledge base (JSONL)

Supported curriculum strategies:
  - easy_medium_hard : ascending curriculum
  - diversity         : maximize taxonomy coverage
  - mutate            : paraphrase / mutate problem statements
  - all               : run all three strategies and merge outputs

Supported providers:
  - offline  : no API; uses deterministic prompt mutation and reference code
  - ollama   : local Ollama endpoint
  - groq     : Groq OpenAI-compatible API
  - openai   : OpenAI API
  - custom   : any OpenAI-compatible endpoint (vLLM, LM Studio, etc.)

Examples:
  python run_agent0_mbpp_curriculum.py --interactive
  python run_agent0_mbpp_curriculum.py --data_file ../../Data/mbpp/mbpp.jsonl --provider offline
  python run_agent0_mbpp_curriculum.py --data_file ../../Data/mbpp/mbpp.jsonl --provider ollama --model qwen3:8b
  python run_agent0_mbpp_curriculum.py --strategy all --items_per_strategy 10
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import traceback
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_DATA_FILE = REPO_ROOT / "Data" / "mbpp" / "mbpp.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Data" / "mbpp" / "curriculum_outputs"
DEFAULT_KB_PATH = DEFAULT_OUTPUT_DIR / "knowledge_base.jsonl"
DEFAULT_REJECTED_PATH = DEFAULT_OUTPUT_DIR / "rejected.jsonl"
DEFAULT_SUMMARY_PATH = DEFAULT_OUTPUT_DIR / "summary.json"

OLLAMA_BASE_URL = "http://localhost:11434"
OPENAI_BASE_URL = "https://api.openai.com/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"

DEFAULT_MODEL = "llama-3.3-70b-versatile"
DEFAULT_PROVIDER = "auto"
DEFAULT_STRATEGY = "all"

TAXONOMY_RULES: Dict[str, List[str]] = {
    "math": [
        r"\bprime\b", r"\bfactorial\b", r"\bgcd\b", r"\blcm\b",
        r"\bdivisible\b", r"\bmod(?:ulo)?\b", r"\bsum\b", r"\bproduct\b",
        r"\bfibonacci\b", r"\bsequence\b", r"\bpermutation\b", r"\bcombination\b",
    ],
    "strings": [
        r"\bstring\b", r"\bsubstring\b", r"\bpalindrome\b", r"\banagram\b",
        r"\bregex\b", r"\bsplit\b", r"\bjoin\b", r"\breplace\b",
        r"\blowercase\b", r"\buppercase\b", r"\bcharacter\b", r"\btext\b",
    ],
    "lists": [
        r"\blist\b", r"\barray\b", r"\bslice\b", r"\bappend\b", r"\bindex\b",
        r"\bsort\b", r"\bsorted\b", r"\bsublist\b", r"\belement\b", r"\bsequence\b",
    ],
    "dict": [
        r"\bdict\b", r"\bdictionary\b", r"\bfrequency\b", r"\bcount\b", r"\bmap\b",
        r"\bhash\b", r"\blookup\b", r"\boccurrence\b", r"\boccurrences\b",
    ],
    "set": [
        r"\bset\b", r"\bunique\b", r"\bduplicate\b", r"\bintersection\b",
        r"\bunion\b", r"\bdeduplicate\b",
    ],
    "loops": [r"\bfor\b", r"\bwhile\b", r"\biterate\b", r"\bloop\b"],
    "recursion": [
        r"\brecursive\b", r"\brecursion\b", r"\bbacktracking\b", r"\bdfs\b",
        r"\btree\b", r"\bdepth\b", r"\bheight\b",
    ],
    "graph": [
        r"\bgraph\b", r"\bnode\b", r"\bedge\b", r"\bbfs\b", r"\bdfs\b",
        r"\bpath\b", r"\bshortest path\b", r"\bconnected\b",
    ],
    "dp": [
        r"dynamic programming", r"\bdp\b", r"\bmemo\b", r"\bmemoization\b",
        r"\bsubsequence\b", r"\bknapsack\b", r"\btabulation\b",
    ],
    "search": [
        r"\bsearch\b", r"\bfind\b", r"\blookup\b", r"\bbinary search\b",
        r"\bminimum\b", r"\bmaximum\b", r"\bmin\b", r"\bmax\b",
    ],
    "sorting": [
        r"\bsort\b", r"\bsorted\b", r"\bmerge sort\b", r"\bquick sort\b",
        r"\bascending\b", r"\bdescending\b",
    ],
    "bitwise": [r"\bxor\b", r"\band\b", r"\bor\b", r"\bshift\b", r"\bbitwise\b"],
    "datetime": [r"\bdate\b", r"\btime\b", r"\btimestamp\b", r"\bweekday\b", r"\bcalendar\b"],
    "regex": [r"\bregex\b", r"\bregular expression\b", r"\bre\."],
    "io": [r"\bfile\b", r"\bpath\b", r"\bjson\b", r"\bcsv\b", r"\bread\b", r"\bwrite\b"],
    "datastructures": [r"\bstack\b", r"\bqueue\b", r"\bheap\b", r"\bdeque\b", r"\bpriority queue\b"],
}

DIFFICULTY_WEIGHTS = {
    "math": 0.10,
    "strings": 0.08,
    "lists": 0.08,
    "dict": 0.10,
    "set": 0.08,
    "loops": 0.05,
    "recursion": 0.20,
    "graph": 0.25,
    "dp": 0.25,
    "search": 0.08,
    "sorting": 0.08,
    "bitwise": 0.12,
    "datetime": 0.06,
    "regex": 0.16,
    "io": 0.05,
    "datastructures": 0.10,
}

PROVIDER_ENV_KEYS = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "custom": None,
    "ollama": None,
    "offline": None,
}


def load_env_files() -> None:
    if load_dotenv is None:
        return
    for candidate in [
        SCRIPT_DIR / ".env",
        REPO_ROOT / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate)
            break


load_env_files()

# ─────────────────────────────────────────────────────────────────────────────
# Global logger
# ─────────────────────────────────────────────────────────────────────────────

logger: logging.Logger = logging.getLogger(__name__)
LOG_LLM_IO: bool = False
LLM_IO_MAX_CHARS: int = 20000


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return text
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated {len(text) - max_chars} chars>"

def setup_logger(
    log_file: Optional[str] = None,
    log_level: str = "INFO",
    quiet: bool = False,
    log_llm_io: bool = False,
    llm_io_max_chars: int = 20000,
) -> None:
    global logger
    global LOG_LLM_IO
    global LLM_IO_MAX_CHARS
    logger = logging.getLogger(__name__)
    LOG_LLM_IO = bool(log_llm_io)
    LLM_IO_MAX_CHARS = max(0, int(llm_io_max_chars))
    resolved_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(resolved_level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = logging.WARNING if quiet else resolved_level
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter(
        fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(resolved_level)
        file_formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging initialized to {log_file}")
    logger.info("LLM I/O logging: %s (max_chars=%s)", "enabled" if LOG_LLM_IO else "disabled", LLM_IO_MAX_CHARS)


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_jsonl_or_json(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL at line {line_no}: {e}") from e
        return items

    if suffix == ".json":
        payload = json.loads(read_text(path))
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return payload["data"]
        raise ValueError("JSON input must be a list or a {'data': [...]} object")

    if suffix == ".parquet":
        try:
            import pandas as pd  # type: ignore
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Reading parquet requires pandas/pyarrow. Please install them first."
            ) from e
        return pd.read_parquet(path).to_dict(orient="records")

    raise ValueError("Unsupported data format. Use .jsonl, .json, or .parquet")


def resolve_existing_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    candidates = [
        path,
        Path.cwd() / path,
        SCRIPT_DIR / path,
        REPO_ROOT / path,
    ]
    seen = set()
    for candidate in candidates:
        resolved = candidate.resolve(strict=False)
        if str(resolved) in seen:
            continue
        seen.add(str(resolved))
        if resolved.exists():
            return resolved
    return path.resolve(strict=False)


def safe_strip_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def normalize_answer(answer: str) -> str:
    answer = safe_strip_text(answer).replace("$", "").replace(",", "").replace(" ", "")
    try:
        return str(int(float(answer)))
    except (ValueError, OverflowError):
        return answer


def compact_sentence(text: str) -> str:
    text = re.sub(r"\s+", " ", safe_strip_text(text)).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Runtime / provider selection
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    provider: str
    model: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


GROQ_MODELS = {
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "qwen/qwen3-32b",
    "qwen/qwen3-235b-a22b",
}


def prompt_runtime_selection(args: argparse.Namespace) -> None:
    if not args.interactive:
        return

    provider = input("Choose provider [auto/offline/ollama/groq/openai/custom] (default: auto): ").strip().lower()
    if provider:
        args.provider = provider

    model = input(f"Model name (default: {args.model}): ").strip()
    if model:
        args.model = model

    data_file = input(f"Data file (default: {args.data_file or DEFAULT_DATA_FILE}): ").strip()
    if data_file:
        args.data_file = data_file

    if args.provider in {"openai", "custom"}:
        base_url = input(f"Base URL (default: {args.base_url or OPENAI_BASE_URL}): ").strip()
        if base_url:
            args.base_url = base_url

    if args.provider in {"groq", "openai", "custom"}:
        api_key = input("API key override (blank to use env): ").strip()
        if api_key:
            args.api_key = api_key


def resolve_runtime(args: argparse.Namespace) -> RuntimeConfig:
    provider = (args.provider or "auto").lower()
    model = args.model

    if provider == "auto":
        if args.base_url:
            provider = "custom"
        elif model in GROQ_MODELS or ("-" in model and not model.startswith("http")):
            provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"
        elif ":" in model:
            provider = "ollama"
        else:
            provider = "groq" if os.getenv("GROQ_API_KEY") else "ollama"

    if provider == "offline":
        return RuntimeConfig(provider="offline", model=model, backend="Offline deterministic mode")

    if provider == "ollama":
        base_url = args.base_url or OLLAMA_BASE_URL
        _check_ollama(model, base_url)
        return RuntimeConfig(provider="ollama", model=model, backend="Ollama local", base_url=base_url)

    if provider == "groq":
        api_key = args.api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set for Groq provider")
        return RuntimeConfig(
            provider="groq",
            model=model,
            backend="Groq API",
            base_url=GROQ_BASE_URL,
            api_key=api_key,
        )

    if provider == "openai":
        api_key = args.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set for OpenAI provider")
        return RuntimeConfig(
            provider="openai",
            model=model,
            backend="OpenAI API",
            base_url=args.base_url or OPENAI_BASE_URL,
            api_key=api_key,
        )

    if provider == "ollama-cloud":
        api_key = args.api_key or os.getenv("OLLAMA_API_KEY")
        if not api_key:
            raise ValueError("OLLAMA_API_KEY not set for Ollama Cloud provider")
        return RuntimeConfig(
            provider="ollama-cloud",
            model=model,
            backend="Ollama Cloud",
            base_url=args.base_url or "https://ollama.com",
            api_key=api_key,
        )

    if provider == "custom":
        base_url = args.base_url or OPENAI_BASE_URL
        api_key = args.api_key or "not-used"
        return RuntimeConfig(
            provider="custom",
            model=model,
            backend="Custom OpenAI-compatible API",
            base_url=base_url,
            api_key=api_key,
        )

    raise ValueError("Invalid provider. Choose from auto/offline/ollama/ollama-cloud/groq/openai/custom")


def _check_ollama(model: str, base_url: str) -> None:
    try:
        tags = http_get_json(f"{base_url.rstrip('/')}/api/tags", timeout=5)
        models = [m.get("name", "") for m in tags.get("models", [])]
        if models and not any(model in m for m in models):
            print(f"WARNING: '{model}' not found in Ollama. Available: {models}")
    except Exception as e:
        raise ValueError(f"Cannot connect to Ollama at {base_url}: {e}") from e


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers (no external SDK required)
# ─────────────────────────────────────────────────────────────────────────────

def http_get_json(url: str, timeout: int = 30, headers: Optional[dict] = None) -> dict:
    default_headers = {"User-Agent": "Agent0/1.0"}
    if headers:
        default_headers.update(headers)
    req = urllib.request.Request(url, headers=default_headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}


def http_post_json(url: str, payload: dict, timeout: int = 300, headers: Optional[dict] = None) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req_headers = {"Content-Type": "application/json", "User-Agent": "Agent0/1.0"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(url, data=data, headers=req_headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HTTP {e.code} from {url}: {err}") from e


def chat_completion(
    runtime: RuntimeConfig,
    messages: List[dict],
    temperature: float = 0.2,
    max_tokens: int = 2048,
) -> str:
    if runtime.provider == "offline":
        raise RuntimeError("Offline mode does not support LLM chat calls")

    logger.debug(
        "LLM call start | provider=%s model=%s temperature=%s max_tokens=%s messages=%s",
        runtime.provider,
        runtime.model,
        temperature,
        max_tokens,
        len(messages),
    )
    if LOG_LLM_IO:
        try:
            request_dump = json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception:
            request_dump = str(messages)
        logger.info(
            "LLM INPUT | provider=%s model=%s\n%s",
            runtime.provider,
            runtime.model,
            _truncate_text(request_dump, LLM_IO_MAX_CHARS),
        )

    if runtime.provider in ("ollama", "ollama-cloud"):
        payload = {
            "model": runtime.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        headers = {}
        if runtime.api_key:
            headers["Authorization"] = f"Bearer {runtime.api_key}"
        url = f"{runtime.base_url.rstrip('/')}/api/chat"
        for _retry in range(3):
            try:
                body = http_post_json(url, payload, timeout=300, headers=headers)
                break
            except RuntimeError as e:
                if "429" in str(e) and _retry < 2:
                    logger.info("Rate limited, waiting 40s before retry...")
                    time.sleep(40)
                    continue
                raise
        else:
            raise RuntimeError(f"Rate limit exceeded after 3 retries for {runtime.provider}")
        content = safe_strip_text(body.get("message", {}).get("content", ""))
        if LOG_LLM_IO:
            logger.info("LLM OUTPUT | provider=%s\n%s", runtime.provider, _truncate_text(content, LLM_IO_MAX_CHARS))
        logger.debug("LLM call done | provider=%s output_chars=%s", runtime.provider, len(content))
        return content

    # Groq / OpenAI / custom all use OpenAI-compatible chat/completions
    assert runtime.base_url is not None
    payload = {
        "model": runtime.model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {}
    if runtime.api_key and runtime.api_key != "not-used":
        headers["Authorization"] = f"Bearer {runtime.api_key}"
    url = f"{runtime.base_url.rstrip('/')}/chat/completions"
    for _retry in range(3):
        try:
            body = http_post_json(url, payload, timeout=300, headers=headers)
            break
        except RuntimeError as e:
            if "429" in str(e) and _retry < 2:
                logger.info("Rate limited, waiting 40s before retry...")
                time.sleep(40)
                continue
            raise
    else:
        raise RuntimeError(f"Rate limit exceeded after 3 retries for {runtime.provider} API")
    choices = body.get("choices", [])
    if not choices:
        logger.error("LLM call failed | provider=%s no choices returned", runtime.provider)
        raise RuntimeError(f"No choices returned by {runtime.provider} API")
    content = safe_strip_text(choices[0].get("message", {}).get("content", ""))
    if LOG_LLM_IO:
        logger.info("LLM OUTPUT | provider=%s\n%s", runtime.provider, _truncate_text(content, LLM_IO_MAX_CHARS))
    logger.debug("LLM call done | provider=%s output_chars=%s", runtime.provider, len(content))
    return content


# ─────────────────────────────────────────────────────────────────────────────
# MBPP loading / taxonomy / difficulty
# ─────────────────────────────────────────────────────────────────────────────

def normalize_mbpp_record(
    raw: dict,
    prompt_key: str = "text",
    code_key: str = "code",
    test_key: str = "test_list",
    setup_key: str = "test_setup_code",
) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    prompt = raw.get(prompt_key)
    if prompt is None:
        for k in ("prompt", "question", "text", "instruction"):
            if raw.get(k) is not None:
                prompt = raw.get(k)
                break

    code = raw.get(code_key)
    if code is None:
        for k in ("code", "answer", "solution"):
            if raw.get(k) is not None:
                code = raw.get(k)
                break

    tests = raw.get(test_key)
    if tests is None:
        for k in ("test_list", "tests", "assertions"):
            if raw.get(k) is not None:
                tests = raw.get(k)
                break

    setup_code = raw.get(setup_key)
    if setup_code is None:
        setup_code = raw.get("test_setup_code", "")

    challenge_tests = raw.get("challenge_test_list", [])

    if prompt is None or code is None or tests is None:
        return None

    if isinstance(tests, str):
        tests = [x for x in tests.split("\n") if x.strip()]
    if isinstance(challenge_tests, str):
        challenge_tests = [x for x in challenge_tests.split("\n") if x.strip()]

    if not isinstance(tests, list):
        tests = [str(tests)]
    if not isinstance(challenge_tests, list):
        challenge_tests = [str(challenge_tests)] if challenge_tests else []

    record = {
        "task_id": raw.get("task_id"),
        "prompt": safe_strip_text(prompt),
        "reference_solution": safe_strip_text(code),
        "test_setup_code": safe_strip_text(setup_code),
        "test_list": [safe_strip_text(t) for t in tests if safe_strip_text(t)],
        "challenge_test_list": [safe_strip_text(t) for t in challenge_tests if safe_strip_text(t)],
        "source_file": raw.get("source_file", "mbpp.jsonl"),
        "raw": raw,
    }
    return record


def load_mbpp_records(
    data_file: str,
    prompt_key: str = "text",
    code_key: str = "code",
    test_key: str = "test_list",
    setup_key: str = "test_setup_code",
    limit: Optional[int] = None,
) -> List[dict]:
    path = resolve_existing_path(data_file)
    raw_items = load_jsonl_or_json(path)
    records: List[dict] = []
    for raw in raw_items:
        rec = normalize_mbpp_record(raw, prompt_key=prompt_key, code_key=code_key, test_key=test_key, setup_key=setup_key)
        if rec:
            records.append(rec)
        if limit and len(records) >= limit:
            break

    if not records:
        raise ValueError(
            f"No valid MBPP records found in {path} with keys prompt='{prompt_key}', code='{code_key}', test='{test_key}'"
        )
    return records


def synthetic_reference_stub(function_name: str) -> str:
    return textwrap.dedent(
        f"""
        def {function_name}(*args, **kwargs):
            raise NotImplementedError(\"Synthetic skeleton; generate solution before verify\")
        """
    ).strip()


def generate_synthetic_records(count: int, seed: int = 42) -> List[dict]:
    """Fallback: generates minimal placeholder records when no LLM is available."""
    random.seed(seed)
    records: List[dict] = []
    for i in range(1, count + 1):
        records.append({
            "task_id": f"syn_{i}",
            "prompt": f"Write a Python function to solve coding task #{i}.",
            "reference_solution": synthetic_reference_stub(f"solve_{i}"),
            "test_setup_code": "",
            "test_list": [f"assert solve_{i}() is not None"],
            "challenge_test_list": [],
            "source_file": "synthetic",
            "raw": {},
        })
    random.shuffle(records)
    return records


def normalize_llm_synthetic_task(raw: dict, idx: int) -> Optional[dict]:
    if not isinstance(raw, dict):
        return None

    prompt = safe_strip_text(raw.get("prompt") or raw.get("task") or raw.get("instruction"))
    function_name = safe_strip_text(raw.get("function_name") or raw.get("name"))
    tests = raw.get("test_list") or raw.get("tests") or []
    challenge_tests = raw.get("challenge_test_list") or raw.get("challenge_tests") or []
    setup_code = safe_strip_text(raw.get("test_setup_code") or raw.get("setup_code") or "")

    if isinstance(tests, str):
        tests = [x.strip() for x in tests.split("\n") if x.strip()]
    if isinstance(challenge_tests, str):
        challenge_tests = [x.strip() for x in challenge_tests.split("\n") if x.strip()]

    if not isinstance(tests, list):
        tests = [str(tests)]
    if not isinstance(challenge_tests, list):
        challenge_tests = [str(challenge_tests)] if challenge_tests else []

    tests = [safe_strip_text(t) for t in tests if safe_strip_text(t)]
    challenge_tests = [safe_strip_text(t) for t in challenge_tests if safe_strip_text(t)]

    if not prompt or not tests:
        return None

    if not function_name:
        detected = re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)\s*\(", prompt)
        function_name = detected[0] if detected else f"solve_{idx}"

    return {
        "task_id": f"synllm_{idx}",
        "prompt": prompt,
        "reference_solution": synthetic_reference_stub(function_name),
        "test_setup_code": setup_code,
        "test_list": tests,
        "challenge_test_list": challenge_tests,
        "source_file": "synthetic_llm",
        "raw": raw,
    }


def _plan_subtopics(runtime: RuntimeConfig, domain: str, count: int, reflection_rounds: int = 1) -> List[dict]:
    """Phase 1: Delegates to curriculum_planner for subtopic planning with reflection."""
    from curriculum_planner import plan_curriculum, RuntimeConfig as PlannerRC

    planner_runtime = PlannerRC(
        provider=runtime.provider,
        model=runtime.model,
        backend=runtime.backend,
        base_url=runtime.base_url,
        api_key=runtime.api_key,
    )
    return plan_curriculum(
        runtime=planner_runtime,
        domain=domain,
        total_tasks=count,
        reflection_rounds=reflection_rounds,
    )


def _generate_tasks_for_subtopic(
    runtime: RuntimeConfig,
    subtopic: dict,
    start_idx: int,
    seed: int,
) -> List[dict]:
    """Phase 2: LLM generates tasks for a single subtopic."""
    task_count = subtopic["task_count"]
    messages = [
        {
            "role": "system",
            "content": (
                "You generate coding benchmark tasks. Return strict JSON only as an array. "
                "Each item must have: prompt, function_name, test_setup_code, test_list, challenge_test_list. "
                "Tests must be Python assert statements and executable. No markdown. No explanation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "subtopic": subtopic["name"],
                    "description": subtopic["description"],
                    "difficulty": subtopic["difficulty"],
                    "count": task_count,
                    "seed": seed,
                    "requirements": [
                        f"Generate exactly {task_count} standalone Python coding problems about '{subtopic['name']}'.",
                        f"Difficulty level: {subtopic['difficulty']}.",
                        "Each task targets one primary function with deterministic behavior.",
                        "Provide 3-5 assert tests in test_list.",
                        "Optionally provide 0-2 harder asserts in challenge_test_list.",
                        "No markdown wrappers; JSON array only.",
                    ],
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    content = chat_completion(runtime, messages, temperature=0.5, max_tokens=3500)

    # Strip <think>...</think> blocks
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()

    parsed: Any
    try:
        parsed = json.loads(content)
    except Exception:
        parsed = parse_json_object(content)
        if parsed is None:
            parsed = []

    if isinstance(parsed, dict):
        items = parsed.get("tasks") or parsed.get("data") or []
    elif isinstance(parsed, list):
        items = parsed
    else:
        items = []

    collected: List[dict] = []
    idx = start_idx
    for raw in items:
        normalized = normalize_llm_synthetic_task(raw, idx)
        if normalized:
            normalized["subtopic"] = subtopic["name"]
            normalized["subtopic_difficulty"] = subtopic["difficulty"]
            collected.append(normalized)
            idx += 1
            if len(collected) >= task_count:
                break

    return collected


def generate_synthetic_records_with_llm(
    runtime: RuntimeConfig,
    count: int,
    seed: int = 42,
    domain: str = "coding",
) -> List[dict]:
    if runtime.provider == "offline":
        raise ValueError("LLM synthetic generation requires non-offline provider")

    logger.info(
        "Synthetic LLM generation start | domain=%s count=%s provider=%s model=%s",
        domain,
        count,
        runtime.provider,
        runtime.model,
    )

    # ── Phase 1: LLM plans subtopics ──
    subtopics = _plan_subtopics(runtime, domain, count)

    # ── Phase 2: LLM generates tasks per subtopic ──
    collected: List[dict] = []
    global_idx = 1

    for i, subtopic in enumerate(subtopics, start=1):
        logger.info(
            "Phase 2: Generating tasks | subtopic=%s (%s/%s) difficulty=%s task_count=%s",
            subtopic["name"],
            i,
            len(subtopics),
            subtopic["difficulty"],
            subtopic["task_count"],
        )

        for attempt in range(3):
            try:
                tasks = _generate_tasks_for_subtopic(
                    runtime, subtopic, global_idx, seed + i + attempt
                )
                break
            except Exception as e:
                logger.warning(
                    "Phase 2 failed | subtopic=%s attempt=%s error=%s",
                    subtopic["name"],
                    attempt + 1,
                    e,
                )
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    logger.info("Rate limited, waiting 40s before retry...")
                    time.sleep(40)
                if attempt == 2:
                    tasks = []

        logger.info(
            "Phase 2 done | subtopic=%s generated=%s/%s",
            subtopic["name"],
            len(tasks),
            subtopic["task_count"],
        )
        collected.extend(tasks)
        global_idx += len(tasks)

    if len(collected) < count:
        logger.warning(
            "Synthetic LLM generation partial | collected=%s/%s",
            len(collected),
            count,
        )
        if len(collected) == 0:
            raise ValueError("Could not generate any synthetic tasks via LLM.")
    else:
        collected = collected[:count]

    random.seed(seed)
    random.shuffle(collected)
    logger.info("Synthetic LLM generation completed | final_count=%s", len(collected[:count]))
    return collected[:count]


def infer_taxonomy(record: dict) -> dict:
    text = "\n".join(
        [
            record.get("prompt", ""),
            record.get("reference_solution", ""),
            "\n".join(record.get("test_list", [])),
            "\n".join(record.get("challenge_test_list", [])),
        ]
    ).lower()

    counts: Dict[str, int] = {}
    for label, patterns in TAXONOMY_RULES.items():
        score = 0
        for pattern in patterns:
            if re.search(pattern, text, flags=re.I):
                score += 1
        if score:
            counts[label] = score

    # Code-structure hints
    code = record.get("reference_solution", "")
    code_features = {
        "for_loops": len(re.findall(r"\bfor\b", code)),
        "while_loops": len(re.findall(r"\bwhile\b", code)),
        "ifs": len(re.findall(r"\bif\b", code)),
        "recursion": bool(re.search(r"\breturn\s+\w+\(", code)) and any(k in counts for k in ["recursion", "dp", "graph"]),
        "comprehensions": len(re.findall(r"\[[^\]]+for[^\]]+\]", code)),
        "imports": len(re.findall(r"^\s*(?:from\s+\S+\s+import|import\s+\S+)", code, flags=re.M)),
    }

    primary = max(counts.items(), key=lambda kv: (kv[1], DIFFICULTY_WEIGHTS.get(kv[0], 0.05)))[0] if counts else "general"
    secondary = [label for label, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if label != primary][:4]

    prompt_len = len(record.get("prompt", ""))
    code_len = len(code)
    signals = len(counts)
    difficulty_score = 0.15
    difficulty_score += min(prompt_len / 600.0, 0.20)
    difficulty_score += min(code_len / 1200.0, 0.15)
    difficulty_score += min(signals * 0.08, 0.30)
    difficulty_score += sum(DIFFICULTY_WEIGHTS.get(label, 0.0) for label in counts)
    if code_features["for_loops"] + code_features["while_loops"] >= 2:
        difficulty_score += 0.08
    if code_features["comprehensions"]:
        difficulty_score += 0.05
    if code_features["imports"] >= 2:
        difficulty_score += 0.03

    if difficulty_score < 0.55:
        difficulty = "easy"
    elif difficulty_score < 0.80:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return {
        "taxonomy": [primary] + secondary,
        "primary_taxonomy": primary,
        "secondary_taxonomy": secondary,
        "difficulty_score": round(difficulty_score, 3),
        "difficulty": difficulty,
        "signals": counts,
        "code_features": code_features,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Curriculum strategies
# ─────────────────────────────────────────────────────────────────────────────

def bucket_by_difficulty(records: List[dict]) -> Dict[str, List[dict]]:
    buckets = {"easy": [], "medium": [], "hard": []}
    for rec in records:
        buckets[rec["difficulty"]].append(rec)
    for bucket in buckets.values():
        bucket.sort(key=lambda r: (r["difficulty_score"], len(r["prompt"]), r.get("task_id") or 0))
    return buckets


def order_easy_medium_hard(records: List[dict], per_stage: int) -> List[dict]:
    buckets = bucket_by_difficulty(records)
    ordered = []
    for stage in ["easy", "medium", "hard"]:
        ordered.extend([dict(r, curriculum_stage=stage, curriculum_strategy="easy_medium_hard") for r in buckets[stage][:per_stage]])
    return ordered


def order_diversity(records: List[dict], per_stage: int) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        grouped[rec["primary_taxonomy"]].append(rec)
    for label in grouped:
        grouped[label].sort(key=lambda r: (r["difficulty_score"], len(r["prompt"]), r.get("task_id") or 0))

    labels = sorted(grouped.keys(), key=lambda lbl: (-len(grouped[lbl]), lbl))
    selected = []
    idx = 0
    while len(selected) < per_stage and labels:
        for label in labels:
            if grouped[label]:
                rec = grouped[label].pop(0)
                selected.append(dict(rec, curriculum_stage=rec["difficulty"], curriculum_strategy="diversity"))
                if len(selected) >= per_stage:
                    break
        idx += 1
        if idx > 1000:
            break
    return selected


def order_mutate(records: List[dict], per_stage: int) -> List[dict]:
    # Prefer medium tasks first, then easy/hard, to mutate around a stable middle ground.
    scored = sorted(records, key=lambda r: (abs(r["difficulty_score"] - 0.55), r["difficulty_score"], len(r["prompt"])))
    chosen = []
    for rec in scored[:per_stage]:
        chosen.append(dict(rec, curriculum_stage=rec["difficulty"], curriculum_strategy="mutate"))
    return chosen


def order_all_strategies(records: List[dict], per_stage: int) -> List[dict]:
    combined: List[dict] = []
    combined.extend(order_easy_medium_hard(records, per_stage))
    combined.extend(order_diversity(records, per_stage))
    combined.extend(order_mutate(records, per_stage))

    # Deduplicate by task_id + strategy + prompt hash-like signature.
    seen = set()
    unique = []
    for rec in combined:
        signature = (rec.get("task_id"), rec.get("curriculum_strategy"), rec.get("curriculum_stage"), rec.get("primary_taxonomy"), rec.get("prompt")[:120])
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(rec)
    return unique


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction and explanation generation
# ─────────────────────────────────────────────────────────────────────────────

def mutate_prompt_text(prompt: str, strategy: str, stage: str, taxonomy: Sequence[str]) -> Tuple[str, str]:
    prompt = compact_sentence(prompt)
    primary = taxonomy[0] if taxonomy else "general"

    if strategy == "easy_medium_hard":
        lead = {
            "easy": "Beginner-friendly curriculum task:",
            "medium": "Intermediate curriculum task:",
            "hard": "Advanced curriculum task:",
        }.get(stage, "Curriculum task:")
        mutated = f"{lead} {prompt}"
        notes = f"Stage-wrapped task for {stage} curriculum with primary taxonomy '{primary}'."
        return mutated, notes

    if strategy == "diversity":
        prefix = f"[Taxonomy: {primary} | Stage: {stage}]"
        mutated = f"{prefix} {prompt}"
        notes = f"Diversity-focused prompt that exposes taxonomy '{primary}' and stage '{stage}'."
        return mutated, notes

    # mutate strategy
    replacements = [
        (r"\bWrite a function to\b", "Implement a Python function to"),
        (r"\bWrite a function that\b", "Implement a Python function that"),
        (r"\bGiven\b", "Consider"),
        (r"\bYou need to\b", "Your task is to"),
        (r"\bFind\b", "Determine"),
        (r"\bCompute\b", "Calculate"),
        (r"\bReturn\b", "Produce"),
    ]
    mutated = prompt
    for pattern, replacement in replacements:
        mutated = re.sub(pattern, replacement, mutated, flags=re.I)

    if mutated == prompt:
        mutated = f"Rephrased coding task: {prompt}"
    notes = f"Paraphrased the prompt while preserving semantics; taxonomy '{primary}' remains unchanged."
    return mutated, notes


def heuristic_solution_pattern(record: dict, taxonomy: Sequence[str]) -> str:
    code = record.get("reference_solution", "")
    features = infer_taxonomy(record)["code_features"]
    patterns = []
    if "dp" in taxonomy:
        patterns.append("dynamic programming / memoization")
    if "graph" in taxonomy:
        patterns.append("graph traversal")
    if "recursion" in taxonomy:
        patterns.append("recursive decomposition")
    if "dict" in taxonomy:
        patterns.append("dictionary / frequency counting")
    if "strings" in taxonomy:
        patterns.append("string normalization")
    if "lists" in taxonomy:
        patterns.append("list traversal and aggregation")
    if features["for_loops"]:
        patterns.append("iterative loops")
    if features["comprehensions"]:
        patterns.append("comprehension-based data processing")
    if not patterns:
        patterns.append("straightforward control flow")
    return ", ".join(dict.fromkeys(patterns))


def build_explanation(record: dict, curriculum_meta: dict) -> str:
    taxonomy = curriculum_meta["taxonomy"]
    stage = curriculum_meta["curriculum_stage"]
    pattern = heuristic_solution_pattern(record, taxonomy)
    prompt = record.get("prompt", "")
    return textwrap.dedent(
        f"""
        This curriculum example is a {stage} problem focused on {curriculum_meta['primary_taxonomy']}.
        The prompt is rewritten to emphasize the same behavior while keeping the MBPP tests valid.
        The reference solution succeeds because it uses {pattern}.
        Verification is performed by executing the solution against the supplied tests.
        """
    ).strip()


def maybe_enrich_explanation_with_llm(
    runtime: RuntimeConfig,
    record: dict,
    curriculum_meta: dict,
    curriculum_prompt: str,
    mutation_notes: str,
) -> Tuple[str, str]:
    if runtime.provider == "offline":
        return build_explanation(record, curriculum_meta), mutation_notes

    messages = [
        {
            "role": "system",
            "content": (
                "You write curriculum metadata for coding tasks. Return concise JSON only. "
                "Do not change the solution code."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task_id": record.get("task_id"),
                    "strategy": curriculum_meta.get("curriculum_strategy"),
                    "stage": curriculum_meta.get("curriculum_stage"),
                    "taxonomy": curriculum_meta.get("taxonomy"),
                    "difficulty": curriculum_meta.get("difficulty"),
                    "prompt": curriculum_prompt,
                    "original_prompt": record.get("prompt"),
                    "solution_pattern": heuristic_solution_pattern(record, curriculum_meta.get("taxonomy", [])),
                    "mutation_notes": mutation_notes,
                },
                ensure_ascii=False,
                indent=2,
            ),
        },
    ]

    try:
        content = chat_completion(runtime, messages, temperature=0.2, max_tokens=700)
        parsed = parse_json_object(content)
        explanation = safe_strip_text(parsed.get("explanation") or parsed.get("summary") or "")
        mutated_notes = safe_strip_text(parsed.get("mutation_notes") or mutation_notes)
        if explanation:
            return explanation, mutated_notes
    except Exception:
        pass

    return build_explanation(record, curriculum_meta), mutation_notes


# ─────────────────────────────────────────────────────────────────────────────
# Verification / repair
# ─────────────────────────────────────────────────────────────────────────────

def build_verification_script(record: dict, solution_code: str, use_challenge_tests: bool = True) -> str:
    parts = []
    if record.get("test_setup_code"):
        parts.append(record["test_setup_code"])
    parts.append(solution_code)
    tests = list(record.get("test_list", []))
    if use_challenge_tests and record.get("challenge_test_list"):
        tests.extend(record["challenge_test_list"])
    parts.extend(tests)
    return "\n\n".join(parts)


def verify_solution(record: dict, solution_code: str, timeout: int = 30, use_challenge_tests: bool = True) -> Dict[str, Any]:
    logger.debug(
        "Verification start | task_id=%s timeout=%ss challenge_tests=%s",
        record.get("task_id"),
        timeout,
        use_challenge_tests,
    )
    script = build_verification_script(record, solution_code, use_challenge_tests=use_challenge_tests)
    tmpdir = Path(tempfile.mkdtemp(prefix="agent0_mbpp_"))
    script_path = tmpdir / "verify.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmpdir,
        )
        logger.info(
            "Verification %s | task_id=%s returncode=%s",
            "PASS" if proc.returncode == 0 else "FAIL",
            record.get("task_id"),
            proc.returncode,
        )
        return {
            "passed": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
            "script_path": str(script_path),
        }
    except subprocess.TimeoutExpired as e:
        logger.warning("Verification timeout | task_id=%s timeout=%ss", record.get("task_id"), timeout)
        return {
            "passed": False,
            "returncode": -1,
            "stdout": (e.stdout or "").strip() if e.stdout else "",
            "stderr": f"Timeout after {timeout}s",
            "script_path": str(script_path),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _diagnose_failure(
    runtime: RuntimeConfig,
    record: dict,
    code: str,
    failure_text: str,
) -> str:
    """Ask LLM to analyze the root cause before attempting a fix."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a senior Python debugger. Analyze the failure and explain the root cause. "
                "Be concise (3-5 sentences). Focus on: "
                "1) Is it a syntax error, logic error, or wrong test expectation? "
                "2) Which specific line or expression is wrong? "
                "3) What is the correct fix?"
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Task prompt:
                {record.get('prompt')}

                Tests:
                {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}

                Code:
                {code}

                Error:
                {failure_text}

                Analyze: what exactly went wrong and where?
                """
            ).strip(),
        },
    ]
    try:
        diagnosis = chat_completion(runtime, messages, temperature=0.1, max_tokens=500)
        # Strip think blocks
        diagnosis = re.sub(r"<think>.*?</think>", "", diagnosis, flags=re.DOTALL).strip()
        logger.info("Diagnosis | task_id=%s\n%s", record.get("task_id"), diagnosis[:500])
        return diagnosis
    except Exception as e:
        logger.warning("Diagnosis failed | task_id=%s error=%s", record.get("task_id"), e)
        return ""


def repair_solution_with_llm(
    runtime: RuntimeConfig,
    record: dict,
    current_code: str,
    verification_result: Dict[str, Any],
    max_rounds: int = 2,
    timeout: int = 30,
) -> Tuple[str, Dict[str, Any], int]:
    if runtime.provider == "offline":
        return current_code, verification_result, 0

    logger.info(
        "Repair loop start | task_id=%s max_rounds=%s",
        record.get("task_id"),
        max_rounds,
    )

    code = current_code
    result = verification_result
    rounds = 0

    for round_idx in range(max_rounds):
        if result.get("passed"):
            logger.info("Repair loop early stop | task_id=%s already passed", record.get("task_id"))
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown verification failure"

        # Step 1: Diagnose the failure
        diagnosis = _diagnose_failure(runtime, record, code, failure_text)

        # Step 2: Repair with diagnosis context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Python engineer. Return only corrected Python code, "
                    "without markdown fences or explanation."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Task prompt:
                    {record.get('prompt')}

                    Tests:
                    {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}

                    Optional challenge tests:
                    {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}

                    Current code:
                    {code}

                    Verification failure:
                    {failure_text}

                    Root cause analysis:
                    {diagnosis}

                    Based on the analysis above, fix the code so it passes the tests.
                    """
                ).strip(),
            },
        ]
        try:
            logger.info("Repair attempt | task_id=%s round=%s", record.get("task_id"), round_idx + 1)
            repaired = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
            extracted = extract_code_block(repaired) or repaired.strip()
            if extracted:
                code = extracted
                result = verify_solution(record, code, timeout=timeout, use_challenge_tests=True)
                rounds += 1
                logger.info(
                    "Repair attempt result | task_id=%s round=%s passed=%s",
                    record.get("task_id"),
                    round_idx + 1,
                    result.get("passed"),
                )
        except Exception as e:
            logger.warning("Repair attempt exception | task_id=%s round=%s error=%s", record.get("task_id"), round_idx + 1, e)
            result = {
                "passed": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Repair attempt failed: {e}",
            }
            rounds += 1

    return code, result, rounds


def repair_tests_with_llm(
    runtime: RuntimeConfig,
    record: dict,
    solution_code: str,
    verification_result: Dict[str, Any],
    max_rounds: int = 2,
    timeout: int = 30,
) -> Tuple[dict, Dict[str, Any], int]:
    """Stage 2 repair: fix test cases instead of code, then re-verify."""
    if runtime.provider == "offline":
        return record, verification_result, 0

    logger.info(
        "Test repair loop start | task_id=%s max_rounds=%s",
        record.get("task_id"),
        max_rounds,
    )

    patched_record = dict(record)
    result = verification_result
    rounds = 0

    for round_idx in range(max_rounds):
        if result.get("passed"):
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown failure"

        # Step 1: Diagnose — is the code wrong or are the tests wrong?
        diagnosis = _diagnose_failure(runtime, record, solution_code, failure_text)

        # Step 2: Fix tests based on diagnosis
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior Python test engineer. "
                    "The solution code is correct, but the test assertions may be wrong. "
                    "Return ONLY a JSON object with keys: test_list, challenge_test_list. "
                    "Each is an array of corrected Python assert strings. No markdown. No explanation."
                ),
            },
            {
                "role": "user",
                "content": textwrap.dedent(
                    f"""
                    Task prompt:
                    {record.get('prompt')}

                    Solution code (treat as correct):
                    {solution_code}

                    Current test_list:
                    {json.dumps(patched_record.get('test_list', []), ensure_ascii=False, indent=2)}

                    Current challenge_test_list:
                    {json.dumps(patched_record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}

                    Verification failure:
                    {failure_text}

                    Root cause analysis:
                    {diagnosis}

                    Based on the analysis, fix the assert statements so they match the actual output of the code.
                    Return JSON only: {{"test_list": [...], "challenge_test_list": [...]}}
                    """
                ).strip(),
            },
        ]
        try:
            logger.info("Test repair attempt | task_id=%s round=%s", record.get("task_id"), round_idx + 1)
            raw = chat_completion(runtime, messages, temperature=0.1, max_tokens=1500)

            # Strip think blocks and parse JSON
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            new_tests = parsed.get("test_list", [])
            new_challenge = parsed.get("challenge_test_list", [])

            if new_tests and isinstance(new_tests, list):
                patched_record["test_list"] = [t.strip() for t in new_tests if isinstance(t, str) and t.strip()]
            if isinstance(new_challenge, list):
                patched_record["challenge_test_list"] = [t.strip() for t in new_challenge if isinstance(t, str) and t.strip()]

            result = verify_solution(patched_record, solution_code, timeout=timeout, use_challenge_tests=True)
            rounds += 1
            logger.info(
                "Test repair attempt result | task_id=%s round=%s passed=%s",
                record.get("task_id"),
                round_idx + 1,
                result.get("passed"),
            )
        except Exception as e:
            logger.warning("Test repair attempt exception | task_id=%s round=%s error=%s", record.get("task_id"), round_idx + 1, e)
            result = {
                "passed": False,
                "returncode": -1,
                "stdout": "",
                "stderr": f"Test repair failed: {e}",
            }
            rounds += 1

    return patched_record, result, rounds


# ─────────────────────────────────────────────────────────────────────────────
# Strategy orchestration
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> Optional[str]:
    matches = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if matches:
        return matches[-1].strip()
    return None


def parse_json_object(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try fenced JSON
    fenced = re.findall(r"```json\s*(.*?)```", text, flags=re.DOTALL | re.I)
    for candidate in reversed(fenced):
        try:
            return json.loads(candidate)
        except Exception:
            continue

    # Try first balanced JSON object
    start = text.find("{")
    if start >= 0:
        for end in range(len(text), start, -1):
            candidate = text[start:end]
            try:
                return json.loads(candidate)
            except Exception:
                continue

    return {}


def offline_generate_solution_from_template(record: dict) -> Optional[str]:
    template_name = safe_strip_text(record.get("synthetic_template"))
    templates = {
        "sum_even": textwrap.dedent(
            """
            def sum_even(nums):
                return sum(x for x in nums if x % 2 == 0)
            """
        ).strip(),
        "reverse_words": textwrap.dedent(
            """
            def reverse_words(s):
                parts = s.split()
                return " ".join(reversed(parts))
            """
        ).strip(),
        "group_anagrams": textwrap.dedent(
            """
            def group_anagrams(words):
                groups = {}
                for word in words:
                    key = "".join(sorted(word))
                    groups.setdefault(key, []).append(word)
                return list(groups.values())
            """
        ).strip(),
        "two_sum": textwrap.dedent(
            """
            def two_sum(nums, target):
                seen = {}
                for idx, value in enumerate(nums):
                    need = target - value
                    if need in seen:
                        return [seen[need], idx]
                    seen[value] = idx
                return []
            """
        ).strip(),
        "fibonacci": textwrap.dedent(
            """
            def fibonacci(n):
                if n <= 1:
                    return n
                first, second = 0, 1
                for _ in range(2, n + 1):
                    first, second = second, first + second
                return second
            """
        ).strip(),
        "merge_intervals": textwrap.dedent(
            """
            def merge_intervals(intervals):
                if not intervals:
                    return []
                ordered = sorted(intervals)
                merged = [ordered[0]]
                for start, end in ordered[1:]:
                    if start <= merged[-1][1]:
                        merged[-1][1] = max(merged[-1][1], end)
                    else:
                        merged.append([start, end])
                return merged
            """
        ).strip(),
    }
    if template_name in templates:
        return templates[template_name]

    reference_code = safe_strip_text(record.get("reference_solution"))
    if "NotImplementedError" not in reference_code:
        return reference_code
    return None


def generate_solution_with_llm(runtime: RuntimeConfig, record: dict, timeout: int = 45) -> Optional[str]:
    if runtime.provider == "offline":
        return None

    logger.debug(
        "LLM solution generation start | task_id=%s provider=%s",
        record.get("task_id"),
        runtime.provider,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding assistant. Return only runnable Python code, "
                "no markdown and no explanation."
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Implement Python code to satisfy this task and tests.

                Task:
                {record.get('prompt', '')}

                Tests:
                {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}

                Optional challenge tests:
                {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}

                Return only the Python solution code.
                """
            ).strip(),
        },
    ]
    try:
        generated = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
        solution = (extract_code_block(generated) or generated).strip()
        logger.debug("LLM solution generation done | task_id=%s chars=%s", record.get("task_id"), len(solution))
        return solution
    except Exception as e:
        logger.warning("LLM solution generation failed | task_id=%s error=%s", record.get("task_id"), e)
        return None


def generate_solution_from_seed(runtime: RuntimeConfig, record: dict, timeout: int = 45) -> str:
    logger.debug(
        "Generate solution start | task_id=%s provider=%s template=%s",
        record.get("task_id"),
        runtime.provider,
        record.get("synthetic_template"),
    )
    if runtime.provider == "offline":
        offline_solution = offline_generate_solution_from_template(record)
        if offline_solution:
            logger.debug("Generate solution path=offline_template | task_id=%s", record.get("task_id"))
            return offline_solution

    llm_solution = generate_solution_with_llm(runtime, record, timeout=timeout)
    if llm_solution:
        logger.debug("Generate solution path=llm | task_id=%s", record.get("task_id"))
        return llm_solution

    reference_code = safe_strip_text(record.get("reference_solution"))
    if reference_code:
        logger.debug("Generate solution path=reference_fallback | task_id=%s", record.get("task_id"))
        return reference_code

    logger.error("Generate solution failed | task_id=%s", record.get("task_id"))
    raise ValueError(f"Unable to generate solution for task_id={record.get('task_id')}")


def select_seed_records(records: List[dict], strategy: str, items_per_strategy: int) -> List[dict]:
    prepared = []
    for rec in records:
        meta = infer_taxonomy(rec)
        prepared.append({**rec, **meta})

    if strategy == "easy_medium_hard":
        return order_easy_medium_hard(prepared, items_per_strategy)
    if strategy == "diversity":
        return order_diversity(prepared, items_per_strategy)
    if strategy == "mutate":
        return order_mutate(prepared, items_per_strategy)
    if strategy == "all":
        return order_all_strategies(prepared, items_per_strategy)
    raise ValueError(f"Unknown strategy: {strategy}")


def curriculum_entry_from_seed(
    seed: dict,
    runtime: RuntimeConfig,
    item_index: int,
    total_items: int,
    use_llm_enrichment: bool,
    repair_rounds: int,
    verify_timeout: int,
) -> Tuple[dict, Optional[dict]]:
    taxonomy = seed.get("taxonomy", [])
    strategy = seed.get("curriculum_strategy", "all")
    stage = seed.get("curriculum_stage", seed.get("difficulty", "medium"))

    curriculum_prompt, mutation_notes = mutate_prompt_text(seed["prompt"], strategy, stage, taxonomy)
    explanation, mutation_notes = maybe_enrich_explanation_with_llm(
        runtime=runtime,
        record=seed,
        curriculum_meta=seed,
        curriculum_prompt=curriculum_prompt,
        mutation_notes=mutation_notes,
    )

    from executor import execute_task as _execute_task, RuntimeConfig as ExecRC

    exec_runtime = ExecRC(
        provider=runtime.provider, model=runtime.model, backend=runtime.backend,
        base_url=runtime.base_url, api_key=runtime.api_key,
    )
    exec_result = _execute_task(exec_runtime, seed, repair_rounds=repair_rounds, verify_timeout=verify_timeout)

    solution_code = exec_result.solution_code
    verification = exec_result.verification
    repair_rounds_used = exec_result.repair_rounds_used
    test_repair_rounds_used = exec_result.test_repair_rounds_used
    accepted = exec_result.accepted
    reasoning = exec_result.reasoning
    if exec_result.record is not seed:
        seed = exec_result.record

    entry = {
        "uid": f"{strategy}-{seed.get('task_id', item_index)}-{item_index}",
        "source_task_id": seed.get("task_id"),
        "source_file": seed.get("source_file"),
        "strategy": strategy,
        "curriculum_stage": stage,
        "taxonomy": taxonomy,
        "difficulty_score": seed.get("difficulty_score"),
        "difficulty": seed.get("difficulty"),
        "primary_taxonomy": seed.get("primary_taxonomy"),
        "secondary_taxonomy": seed.get("secondary_taxonomy", []),
        "curriculum_prompt": curriculum_prompt,
        "original_prompt": seed["prompt"],
        "solution_code": solution_code,
        "reasoning": reasoning,
        "explanation": explanation,
        "mutation_notes": mutation_notes,
        "test_setup_code": seed.get("test_setup_code", ""),
        "test_list": seed.get("test_list", []),
        "challenge_test_list": seed.get("challenge_test_list", []),
        "verification": verification,
        "repair_rounds_used": repair_rounds_used,
        "test_repair_rounds_used": test_repair_rounds_used,
        "accepted": accepted,
        "created_at": int(time.time()),
    }

    reject_entry = None
    if not accepted:
        reject_entry = {
            "uid": entry["uid"],
            "source_task_id": seed.get("task_id"),
            "strategy": strategy,
            "curriculum_stage": stage,
            "taxonomy": taxonomy,
            "curriculum_prompt": curriculum_prompt,
            "original_prompt": seed["prompt"],
            "verification": verification,
            "reason": verification.get("stderr") or verification.get("stdout") or "verification failed",
        }
    return entry, reject_entry


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Agent0_new MBPP curriculum redesign: taxonomy -> template -> solution -> verify -> KB"
    )
    parser.add_argument(
        "--data_file",
        type=str,
        default=str(DEFAULT_DATA_FILE) if DEFAULT_DATA_FILE.exists() else None,
        help="Local MBPP-style data file (.jsonl/.json/.parquet). Default: Data/mbpp/mbpp.jsonl if present.",
    )
    parser.add_argument(
        "--synthetic_only",
        action="store_true",
        default=False,
        help="Do not use source data file; generate synthetic coding tasks only.",
    )
    parser.add_argument(
        "--synthetic_count",
        type=int,
        default=60,
        help="Number of synthetic tasks to generate when --synthetic_only is enabled.",
    )
    parser.add_argument(
        "--synthetic_generator",
        type=str,
        default="auto",
        choices=["auto", "llm", "template"],
        help="Source of synthetic tasks: llm (LLM-generated), template (built-in templates), or auto.",
    )
    parser.add_argument(
        "--synthetic_domain",
        type=str,
        default="coding",
        help="High-level domain description for LLM synthetic generation (e.g. coding, algorithms).",
    )
    parser.add_argument(
        "--prompt_key",
        type=str,
        default="text",
        help="Prompt field name in local data file.",
    )
    parser.add_argument(
        "--code_key",
        type=str,
        default="code",
        help="Reference solution field name in local data file.",
    )
    parser.add_argument(
        "--test_key",
        type=str,
        default="test_list",
        help="Unit-test field name in local data file.",
    )
    parser.add_argument(
        "--setup_key",
        type=str,
        default="test_setup_code",
        help="Optional setup/import code field name in local data file.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default=DEFAULT_PROVIDER,
        choices=["auto", "offline", "ollama", "ollama-cloud", "groq", "openai", "custom"],
        help="LLM backend for explanation enrichment and repair.",
    )
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument(
        "--strategy",
        type=str,
        default=DEFAULT_STRATEGY,
        choices=["easy_medium_hard", "diversity", "mutate", "all"],
        help="Curriculum strategy.",
    )
    parser.add_argument("--items_per_strategy", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="Limit source records loaded from the dataset.")
    parser.add_argument("--repair_rounds", type=int, default=2)
    parser.add_argument("--verify_timeout", type=int, default=20)
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--kb_path", type=str, default=str(DEFAULT_KB_PATH))
    parser.add_argument("--rejected_path", type=str, default=str(DEFAULT_REJECTED_PATH))
    parser.add_argument("--summary_path", type=str, default=str(DEFAULT_SUMMARY_PATH))
    parser.add_argument("--max_output_examples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file (e.g., ./logs/run.log)")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")
    parser.add_argument("--log_llm_io", action="store_true", default=False, help="Log full LLM request/response content")
    parser.add_argument("--llm_io_max_chars", type=int, default=20000, help="Max characters to keep per LLM input/output log entry (0 = unlimited)")
    parser.add_argument("--quiet", action="store_true", default=False)
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    random.seed(args.seed)
    # Setup logging
    setup_logger(
        log_file=args.log_file,
        log_level=args.log_level,
        quiet=args.quiet,
        log_llm_io=args.log_llm_io,
        llm_io_max_chars=args.llm_io_max_chars,
    )
    
    logger.info('=' * 80)
    logger.info('Agent0_new MBPP Curriculum Redesign - Starting')
    logger.info('=' * 80)


    if args.interactive:
        prompt_runtime_selection(args)

    if not args.synthetic_only and not args.data_file:
        raise ValueError("Please provide --data_file, or run with --synthetic_only")

    runtime = resolve_runtime(args)
    use_llm_enrichment = runtime.provider != "offline"
    logger.info("Runtime resolved | provider=%s backend=%s model=%s", runtime.provider, runtime.backend, runtime.model)

    data_path = resolve_existing_path(args.data_file) if args.data_file else None
    output_dir = resolve_existing_path(args.output_dir)

    # Organize KB by model name
    model_safe = runtime.model.replace("/", "_").replace(":", "_")
    model_dir = output_dir / model_safe
    kb_path = model_dir / "knowledge_base.jsonl"
    rejected_path = model_dir / "rejected.jsonl"
    summary_path = model_dir / "summary.json"

    # Global KB (all models combined)
    global_kb_path = output_dir / "all_models" / "knowledge_base.jsonl"
    logger.debug("Paths | data=%s output=%s kb=%s rejected=%s summary=%s", data_path, output_dir, kb_path, rejected_path, summary_path)

    resolved_synthetic_generator = args.synthetic_generator
    if args.synthetic_only:
        if resolved_synthetic_generator == "auto":
            resolved_synthetic_generator = "llm" if runtime.provider != "offline" else "template"
        logger.info("Synthetic generator resolved | requested=%s resolved=%s", args.synthetic_generator, resolved_synthetic_generator)

        if resolved_synthetic_generator == "llm":
            records = generate_synthetic_records_with_llm(
                runtime=runtime,
                count=args.synthetic_count,
                seed=args.seed,
                domain=args.synthetic_domain,
            )
        else:
            records = generate_synthetic_records(count=args.synthetic_count, seed=args.seed)
    else:
        records = load_mbpp_records(
            data_file=str(data_path),
            prompt_key=args.prompt_key,
            code_key=args.code_key,
            test_key=args.test_key,
            setup_key=args.setup_key,
            limit=args.limit,
        )

    selected = select_seed_records(records, args.strategy, args.items_per_strategy)
    logger.info("Selection completed | loaded=%s selected=%s strategy=%s", len(records), len(selected), args.strategy)
    if not selected:
        raise ValueError("No curriculum seeds selected. Check strategy/limit/items_per_strategy.")

    output_dir.mkdir(parents=True, exist_ok=True)
    kb_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print("=" * 80)
        print("  Agent0_new MBPP Curriculum Redesign")
        print("=" * 80)
        print(f"  Provider      : {runtime.provider} ({runtime.backend})")
        print(f"  Model         : {runtime.model}")
        if runtime.base_url:
            print(f"  Base URL      : {runtime.base_url}")
        print(f"  Data mode     : {'synthetic_only' if args.synthetic_only else 'from_file'}")
        if args.synthetic_only:
            print(f"  Synthetic count: {args.synthetic_count}")
            print(f"  Synthetic source: {resolved_synthetic_generator}")
            print(f"  Synthetic domain: {args.synthetic_domain}")
        else:
            print(f"  Data file     : {data_path}")
        print(f"  Strategy      : {args.strategy}")
        print(f"  Items/strategy: {args.items_per_strategy}")
        print(f"  Seeds loaded  : {len(records)}")
        print(f"  Seeds selected: {len(selected)}")
        print(f"  Output dir    : {output_dir}")
        print(f"  KB path       : {kb_path}")
        print(f"  Rejected path  : {rejected_path}")
        print(f"  Repair rounds  : {args.repair_rounds}")
        print(f"  Verify timeout : {args.verify_timeout}s")
        print("=" * 80)

    accepted_rows: List[dict] = []
    rejected_rows: List[dict] = []
    stats = Counter()
    taxonomy_counter = Counter()
    failed_task_ids: set = set()

    # Load existing KB to avoid duplicates across runs
    existing_kb_prompts: set = set()
    existing_kb_count = 0
    if kb_path.exists():
        try:
            with kb_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    # Dedup by full curriculum_prompt (exact match only)
                    sig = row.get("curriculum_prompt", row.get("original_prompt", ""))
                    existing_kb_prompts.add(sig)
                    existing_kb_count += 1
            logger.info("Loaded existing KB | %s entries (%s unique prompts)", existing_kb_count, len(existing_kb_prompts))
        except Exception as e:
            logger.warning("Failed to load existing KB: %s", e)

    # Initialize retriever for live vector index updates
    kb_retriever = None
    try:
        from knowledge_retriever import KnowledgeRetriever
        kb_retriever = KnowledgeRetriever(kb_path=str(kb_path))
        kb_retriever.build_index()
        logger.info("Knowledge retriever initialized | %s entries indexed", len(kb_retriever.entries))
    except Exception as e:
        logger.warning("Knowledge retriever init failed (non-fatal): %s", e)

    # Track prompts accepted in THIS run to avoid intra-run duplicates
    accepted_prompts_this_run: set = set()

    for idx, seed in enumerate(selected, start=1):
        task_id = seed.get("task_id")

        # Skip tasks whose exact prompt already exists in KB (exact match)
        prompt_sig = seed.get("prompt", "")
        if prompt_sig in existing_kb_prompts or prompt_sig in accepted_prompts_this_run:
            logger.info(
                "Skipping seed %s/%s | task_id=%s (already in KB from previous run)",
                idx, len(selected), task_id,
            )
            stats["skipped_existing_kb"] += 1
            if not args.quiet:
                print(
                    f"[{idx:02d}/{len(selected):02d}] SKIP | "
                    f"task_id={task_id} | already in KB"
                )
            continue

        # Skip tasks that already failed in a previous strategy
        if task_id in failed_task_ids:
            logger.info(
                "Skipping seed %s/%s | task_id=%s (already failed in previous strategy)",
                idx, len(selected), task_id,
            )
            stats["skipped_duplicate_fail"] += 1
            if not args.quiet:
                print(
                    f"[{idx:02d}/{len(selected):02d}] SKIP | "
                    f"task_id={task_id} | already failed — skipping"
                )
            continue

        logger.info(
            "Processing seed %s/%s | task_id=%s stage=%s taxonomy=%s",
            idx,
            len(selected),
            task_id,
            seed.get("curriculum_stage", seed.get("difficulty")),
            seed.get("primary_taxonomy"),
        )
        entry, rejected = curriculum_entry_from_seed(
            seed=seed,
            runtime=runtime,
            item_index=idx,
            total_items=len(selected),
            use_llm_enrichment=use_llm_enrichment,
            repair_rounds=args.repair_rounds,
            verify_timeout=args.verify_timeout,
        )

        taxonomy_counter.update(entry.get("taxonomy", []))
        stats["accepted" if entry["accepted"] else "rejected"] += 1
        stats[f"stage_{entry['curriculum_stage']}"] += 1
        stats[f"strategy_{entry['strategy']}"] += 1
        if entry["verification"].get("passed"):
            stats["verified_pass"] += 1
        else:
            stats["verified_fail"] += 1
            failed_task_ids.add(task_id)

        if entry["accepted"]:
            accepted_rows.append(entry)
            append_jsonl(kb_path, entry)
            # Append to global KB (all models combined) with source model tag
            global_entry = dict(entry)
            global_entry["source_model"] = runtime.model
            append_jsonl(global_kb_path, global_entry)
            accepted_prompts_this_run.add(entry.get("curriculum_prompt", entry.get("original_prompt", "")))
            # Live update vector index
            if kb_retriever is not None:
                kb_retriever.add_entry(entry)
            logger.info("Accepted | uid=%s task_id=%s", entry.get("uid"), entry.get("source_task_id"))
        else:
            if rejected is not None:
                rejected_rows.append(rejected)
                append_jsonl(rejected_path, rejected)
            logger.warning(
                "Rejected | uid=%s task_id=%s reason=%s",
                entry.get("uid"),
                entry.get("source_task_id"),
                entry["verification"].get("stderr") or entry["verification"].get("stdout") or "verification failed",
            )

        if not args.quiet:
            status = "PASS" if entry["accepted"] else "FAIL"
            print(
                f"[{idx:02d}/{len(selected):02d}] {status} | "
                f"task_id={entry.get('source_task_id')} | "
                f"stage={entry['curriculum_stage']} | "
                f"taxonomy={entry.get('primary_taxonomy')} | "
                f"difficulty={entry.get('difficulty')} ({entry.get('difficulty_score')})"
            )
            print(f"      prompt: {entry['curriculum_prompt'][:120]}...")
            if not entry["accepted"]:
                print(f"      reason: {entry['verification'].get('stderr') or entry['verification'].get('stdout')}")

    summary = {
        "config": {
            "provider": runtime.provider,
            "backend": runtime.backend,
            "model": runtime.model,
            "base_url": runtime.base_url,
            "data_file": str(data_path) if data_path else None,
            "synthetic_only": args.synthetic_only,
            "synthetic_count": args.synthetic_count,
            "synthetic_generator": resolved_synthetic_generator,
            "synthetic_domain": args.synthetic_domain,
            "strategy": args.strategy,
            "items_per_strategy": args.items_per_strategy,
            "limit": args.limit,
            "repair_rounds": args.repair_rounds,
            "verify_timeout": args.verify_timeout,
            "prompt_key": args.prompt_key,
            "code_key": args.code_key,
            "test_key": args.test_key,
            "setup_key": args.setup_key,
            "log_file": args.log_file,
            "log_level": args.log_level,
            "log_llm_io": args.log_llm_io,
            "llm_io_max_chars": args.llm_io_max_chars,
        },
        "stats": {
            "selected": len(selected),
            "accepted": len(accepted_rows),
            "rejected": len(rejected_rows),
            "skipped_duplicate_fail": stats.get("skipped_duplicate_fail", 0),
            "skipped_existing_kb": stats.get("skipped_existing_kb", 0),
            "kb_existing": existing_kb_count,
            "kb_total": existing_kb_count + len(accepted_rows),
            "taxonomy_top": taxonomy_counter.most_common(15),
            "counts": dict(stats),
        },
        "accepted_examples": [
            {
                "uid": row["uid"],
                "source_task_id": row.get("source_task_id"),
                "strategy": row["strategy"],
                "stage": row["curriculum_stage"],
                "taxonomy": row["taxonomy"],
                "difficulty": row["difficulty"],
                "score": row["difficulty_score"],
            }
            for row in accepted_rows
        ],
    }
    write_json(summary_path, summary)

    if args.max_output_examples > 0 and accepted_rows:
        sample_path = output_dir / "sample_accepted.json"
        write_json(sample_path, {"examples": accepted_rows[: args.max_output_examples]})

    total_kb = existing_kb_count + len(accepted_rows)

    print()
    print("=" * 80)
    print("  MBPP Curriculum Summary")
    print("=" * 80)
    print(f"  This run     : +{len(accepted_rows)} accepted, {len(rejected_rows)} rejected")
    if stats.get("skipped_duplicate_fail", 0):
        print(f"  Skipped (fail): {stats['skipped_duplicate_fail']}")
    if stats.get("skipped_existing_kb", 0):
        print(f"  Skipped (dup) : {stats['skipped_existing_kb']}")
    print(f"  KB total      : {total_kb} entries ({existing_kb_count} old + {len(accepted_rows)} new)")
    print(f"  KB path       : {kb_path}")
    print(f"  Summary       : {summary_path}")
    print(f"  Rejected      : {rejected_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
