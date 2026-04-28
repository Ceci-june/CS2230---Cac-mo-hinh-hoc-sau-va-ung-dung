#!/usr/bin/env python3
"""
Executor — Generate, Verify, Diagnose, and Repair code solutions
================================================================

This module handles all code execution logic:
  1) Generate solution code via LLM
  2) Verify solution against test cases
  3) Diagnose failures (root cause analysis)
  4) Repair code (Stage 1)
  5) Judge whether code or tests are wrong
  6) Repair tests (Stage 2)

Used by run_agent0_mbpp_curriculum.py but can also be used standalone.

Usage:
  from executor import execute_task, ExecutionResult
  result = execute_task(runtime, record, repair_rounds=2, verify_timeout=20)
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("executor")

# ─────────────────────────────────────────────────────────────────────────────
# Runtime config (shared with other modules)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeConfig:
    provider: str
    model: str
    backend: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of executing a task through the full pipeline."""
    solution_code: str
    verification: Dict[str, Any]
    accepted: bool
    record: dict  # possibly patched (if tests were fixed)
    repair_rounds_used: int = 0
    test_repair_rounds_used: int = 0
    diagnosis_log: List[str] = field(default_factory=list)
    judge_verdict: Optional[str] = None
    reasoning: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP + LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

LOG_LLM_IO: bool = False
LLM_IO_MAX_CHARS: int = 20000


def _truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n...<truncated {len(text) - max_chars} chars>"


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


def chat_completion(runtime: RuntimeConfig, messages: List[dict], temperature: float = 0.2, max_tokens: int = 2048) -> str:
    if LOG_LLM_IO:
        try:
            request_dump = json.dumps(messages, ensure_ascii=False, indent=2)
        except Exception:
            request_dump = str(messages)
        logger.info("LLM INPUT | provider=%s model=%s\n%s", runtime.provider, runtime.model, _truncate_text(request_dump, LLM_IO_MAX_CHARS))

    # Ollama local and cloud
    if runtime.provider in ("ollama", "ollama-cloud"):
        payload = {"model": runtime.model, "messages": messages, "stream": False, "options": {"temperature": temperature, "num_predict": max_tokens}}
        hdrs = {}
        if runtime.api_key:
            hdrs["Authorization"] = f"Bearer {runtime.api_key}"
        url = f"{runtime.base_url.rstrip('/')}/api/chat"
        for _retry in range(3):
            try:
                body = http_post_json(url, payload, timeout=300, headers=hdrs)
                break
            except RuntimeError as e:
                if "429" in str(e) and _retry < 2:
                    logger.info("Rate limited, waiting 40s...")
                    time.sleep(40)
                    continue
                raise
        else:
            raise RuntimeError("Rate limit exceeded after 3 retries")
        content = (body.get("message", {}).get("content", "") or "").strip()
        if LOG_LLM_IO:
            logger.info("LLM OUTPUT | provider=%s\n%s", runtime.provider, _truncate_text(content, LLM_IO_MAX_CHARS))
        return content

    # OpenAI-compatible (Groq, OpenAI, custom)
    assert runtime.base_url is not None
    payload = {"model": runtime.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    hdrs = {}
    if runtime.api_key and runtime.api_key != "not-used":
        hdrs["Authorization"] = f"Bearer {runtime.api_key}"
    url = f"{runtime.base_url.rstrip('/')}/chat/completions"
    for _retry in range(3):
        try:
            body = http_post_json(url, payload, timeout=300, headers=hdrs)
            break
        except RuntimeError as e:
            if "429" in str(e) and _retry < 2:
                logger.info("Rate limited, waiting 40s...")
                time.sleep(40)
                continue
            raise
    else:
        raise RuntimeError("Rate limit exceeded after 3 retries")
    choices = body.get("choices", [])
    if not choices:
        raise RuntimeError("No choices returned by API")
    content = (choices[0].get("message", {}).get("content", "") or "").strip()
    if LOG_LLM_IO:
        logger.info("LLM OUTPUT | provider=%s\n%s", runtime.provider, _truncate_text(content, LLM_IO_MAX_CHARS))
    return content


# ─────────────────────────────────────────────────────────────────────────────
# Code extraction helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_block(text: str) -> Optional[str]:
    matches = re.findall(r"```python\s*(.*?)```", text, flags=re.DOTALL | re.I)
    if matches:
        return matches[-1].strip()
    return None


def strip_think_blocks(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ─────────────────────────────────────────────────────────────────────────────
# 1) Generate solution
# ─────────────────────────────────────────────────────────────────────────────

def generate_solution(runtime: RuntimeConfig, record: dict) -> str:
    """Generate a Python solution for the given task using LLM."""
    messages = [
        {
            "role": "system",
            "content": "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation.",
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
        solution = strip_think_blocks(solution)
        logger.info("Generate solution done | task_id=%s chars=%s", record.get("task_id"), len(solution))
        return solution
    except Exception as e:
        logger.warning("Generate solution failed | task_id=%s error=%s", record.get("task_id"), e)
        # Fallback to reference if available
        ref = (record.get("reference_solution") or "").strip()
        if ref and "NotImplementedError" not in ref:
            return ref
        raise


# ─────────────────────────────────────────────────────────────────────────────
# 1b) Generate reasoning for a verified solution
# ─────────────────────────────────────────────────────────────────────────────

def generate_reasoning(runtime: RuntimeConfig, record: dict, solution_code: str) -> str:
    """Generate step-by-step reasoning explaining the approach and key ideas."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Python coding tutor. Explain the approach and reasoning "
                "behind a solution in 3-5 concise bullet points. Focus on: "
                "1) What algorithm/technique is used "
                "2) Key steps of the approach "
                "3) Edge cases handled. "
                "No code. No markdown. Just plain text bullets."
            ),
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}

                Solution:
                {solution_code}

                Explain the reasoning behind this solution.
                """
            ).strip(),
        },
    ]
    try:
        reasoning = chat_completion(runtime, messages, temperature=0.1, max_tokens=500)
        reasoning = strip_think_blocks(reasoning)
        logger.info("Reasoning generated | task_id=%s chars=%s", record.get("task_id"), len(reasoning))
        return reasoning
    except Exception as e:
        logger.warning("Reasoning generation failed | task_id=%s error=%s", record.get("task_id"), e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 2) Verify solution
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
    """Run solution code against test assertions."""
    script = build_verification_script(record, solution_code, use_challenge_tests=use_challenge_tests)
    tmpdir = Path(tempfile.mkdtemp(prefix="agent0_exec_"))
    script_path = tmpdir / "verify.py"
    script_path.write_text(script, encoding="utf-8")
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
        )
        passed = proc.returncode == 0
        logger.info("Verify %s | task_id=%s", "PASS" if passed else "FAIL", record.get("task_id"))
        return {"passed": passed, "returncode": proc.returncode, "stdout": proc.stdout.strip(), "stderr": proc.stderr.strip()}
    except subprocess.TimeoutExpired:
        logger.warning("Verify timeout | task_id=%s", record.get("task_id"))
        return {"passed": False, "returncode": -1, "stdout": "", "stderr": f"Timeout after {timeout}s"}
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3) Diagnose failure
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_failure(runtime: RuntimeConfig, record: dict, code: str, failure_text: str) -> str:
    """Ask LLM to analyze root cause of verification failure."""
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
                Task: {record.get('prompt', '')[:300]}
                Tests: {json.dumps(record.get('test_list', []), ensure_ascii=False)[:500]}
                Code:
                {code[:800]}
                Error:
                {failure_text[:500]}

                Analyze: what exactly went wrong and where?
                """
            ).strip(),
        },
    ]
    try:
        diagnosis = chat_completion(runtime, messages, temperature=0.1, max_tokens=500)
        diagnosis = strip_think_blocks(diagnosis)
        logger.info("Diagnosis | task_id=%s\n%s", record.get("task_id"), diagnosis[:500])
        return diagnosis
    except Exception as e:
        logger.warning("Diagnosis failed | task_id=%s error=%s", record.get("task_id"), e)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# 4) Repair code (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def repair_code(
    runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any],
    max_rounds: int = 2, timeout: int = 30,
) -> Tuple[str, Dict[str, Any], int, List[str]]:
    """Diagnose then repair code. Returns (code, verification, rounds_used, diagnoses)."""
    result = verification
    diagnoses: List[str] = []
    rounds = 0

    for r in range(max_rounds):
        if result.get("passed"):
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown failure"

        # Diagnose
        diagnosis = diagnose_failure(runtime, record, code, failure_text)
        diagnoses.append(diagnosis)

        # Repair with diagnosis context
        messages = [
            {"role": "system", "content": "You are a senior Python engineer. Return only corrected Python code, without markdown fences or explanation."},
            {"role": "user", "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}
                Tests: {json.dumps(record.get('test_list', []), ensure_ascii=False, indent=2)}
                Challenge tests: {json.dumps(record.get('challenge_test_list', []), ensure_ascii=False, indent=2)}
                Current code:
                {code}
                Error: {failure_text}
                Root cause: {diagnosis}

                Based on the analysis above, fix the code so it passes the tests.
                """
            ).strip()},
        ]
        try:
            logger.info("Repair code attempt %s | task_id=%s", r + 1, record.get("task_id"))
            repaired = chat_completion(runtime, messages, temperature=0.1, max_tokens=1800)
            extracted = extract_code_block(repaired) or strip_think_blocks(repaired)
            if extracted:
                code = extracted
                result = verify_solution(record, code, timeout=timeout)
                rounds += 1
                logger.info("Repair code result | task_id=%s round=%s passed=%s", record.get("task_id"), r + 1, result.get("passed"))
        except Exception as e:
            logger.warning("Repair code exception | task_id=%s round=%s error=%s", record.get("task_id"), r + 1, e)
            result = {"passed": False, "returncode": -1, "stdout": "", "stderr": f"Repair failed: {e}"}
            rounds += 1

    return code, result, rounds, diagnoses


# ─────────────────────────────────────────────────────────────────────────────
# 5) Judge: code wrong or tests wrong?
# ─────────────────────────────────────────────────────────────────────────────

def judge_failure(runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any]) -> str:
    """Ask LLM whether the failure is caused by wrong CODE or wrong TEST. Returns 'CODE' or 'TEST'."""
    failure_text = verification.get("stderr") or verification.get("stdout") or ""
    messages = [
        {"role": "system", "content": "You judge whether a test failure is caused by wrong CODE or wrong TEST assertions. Answer with exactly one word: CODE or TEST. Nothing else."},
        {"role": "user", "content": textwrap.dedent(
            f"""
            Task: {record.get('prompt', '')[:300]}
            Code:
            {code[:800]}
            Tests: {json.dumps(record.get('test_list', []), ensure_ascii=False)[:500]}
            Error: {failure_text[:500]}

            Is the bug in the CODE or in the TEST assertions?
            """
        ).strip()},
    ]
    try:
        verdict = chat_completion(runtime, messages, temperature=0.0, max_tokens=10)
        verdict = strip_think_blocks(verdict).upper()
        logger.info("Judge verdict | task_id=%s → %s", record.get("task_id"), verdict)
        return "TEST" if "TEST" in verdict else "CODE"
    except Exception as e:
        logger.warning("Judge failed | task_id=%s error=%s", record.get("task_id"), e)
        return "CODE"


# ─────────────────────────────────────────────────────────────────────────────
# 6) Repair tests (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def repair_tests(
    runtime: RuntimeConfig, record: dict, code: str, verification: Dict[str, Any],
    max_rounds: int = 2, timeout: int = 30,
) -> Tuple[dict, Dict[str, Any], int, List[str]]:
    """Diagnose then repair test assertions. Returns (patched_record, verification, rounds_used, diagnoses)."""
    patched = dict(record)
    result = verification
    diagnoses: List[str] = []
    rounds = 0

    for r in range(max_rounds):
        if result.get("passed"):
            break

        failure_text = result.get("stderr") or result.get("stdout") or "Unknown failure"

        # Diagnose
        diagnosis = diagnose_failure(runtime, record, code, failure_text)
        diagnoses.append(diagnosis)

        # Fix tests
        messages = [
            {"role": "system", "content": (
                "You are a senior Python test engineer. "
                "The solution code is correct, but the test assertions may be wrong. "
                "Return ONLY a JSON object with keys: test_list, challenge_test_list. "
                "Each is an array of corrected Python assert strings. No markdown. No explanation."
            )},
            {"role": "user", "content": textwrap.dedent(
                f"""
                Task: {record.get('prompt', '')}
                Solution code (correct):
                {code}
                Current test_list: {json.dumps(patched.get('test_list', []), ensure_ascii=False, indent=2)}
                Current challenge_test_list: {json.dumps(patched.get('challenge_test_list', []), ensure_ascii=False, indent=2)}
                Error: {failure_text}
                Root cause: {diagnosis}

                Fix the assert statements to match the actual output. Return JSON only.
                """
            ).strip()},
        ]
        try:
            logger.info("Repair tests attempt %s | task_id=%s", r + 1, record.get("task_id"))
            raw = chat_completion(runtime, messages, temperature=0.1, max_tokens=1500)
            raw = strip_think_blocks(raw)
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
            parsed = json.loads(raw)

            new_tests = parsed.get("test_list", [])
            new_challenge = parsed.get("challenge_test_list", [])
            if new_tests and isinstance(new_tests, list):
                patched["test_list"] = [t.strip() for t in new_tests if isinstance(t, str) and t.strip()]
            if isinstance(new_challenge, list):
                patched["challenge_test_list"] = [t.strip() for t in new_challenge if isinstance(t, str) and t.strip()]

            result = verify_solution(patched, code, timeout=timeout)
            rounds += 1
            logger.info("Repair tests result | task_id=%s round=%s passed=%s", record.get("task_id"), r + 1, result.get("passed"))
        except Exception as e:
            logger.warning("Repair tests exception | task_id=%s round=%s error=%s", record.get("task_id"), r + 1, e)
            result = {"passed": False, "returncode": -1, "stdout": "", "stderr": f"Test repair failed: {e}"}
            rounds += 1

    return patched, result, rounds, diagnoses


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def execute_task(
    runtime: RuntimeConfig,
    record: dict,
    repair_rounds: int = 2,
    verify_timeout: int = 20,
) -> ExecutionResult:
    """
    Full execution pipeline for a single task:
      1) Generate solution
      2) Verify
      3) Stage 1: Repair code (if failed)
      4) Judge: CODE or TEST?
      5) Stage 2: Repair tests (if TEST) or repair code once more (if CODE)
    """
    task_id = record.get("task_id", "?")
    all_diagnoses: List[str] = []

    # 1) Generate
    logger.info("Execute start | task_id=%s", task_id)
    solution_code = generate_solution(runtime, record)

    # 2) Verify
    verification = verify_solution(record, solution_code, timeout=verify_timeout)

    repair_rounds_used = 0
    test_repair_rounds_used = 0
    judge_verdict = None
    patched_record = record

    if verification["passed"]:
        reasoning = generate_reasoning(runtime, record, solution_code)
        return ExecutionResult(
            solution_code=solution_code, verification=verification, accepted=True,
            record=record, diagnosis_log=all_diagnoses, reasoning=reasoning,
        )

    # 3) Stage 1: Repair code
    if repair_rounds > 0 and runtime.provider != "offline":
        solution_code, verification, repair_rounds_used, diagnoses = repair_code(
            runtime, record, solution_code, verification,
            max_rounds=repair_rounds, timeout=verify_timeout,
        )
        all_diagnoses.extend(diagnoses)

    if verification["passed"]:
        reasoning = generate_reasoning(runtime, record, solution_code)
        return ExecutionResult(
            solution_code=solution_code, verification=verification, accepted=True,
            record=record, repair_rounds_used=repair_rounds_used, diagnosis_log=all_diagnoses, reasoning=reasoning,
        )

    # 4) Judge
    if repair_rounds > 0 and runtime.provider != "offline":
        judge_verdict = judge_failure(runtime, record, solution_code, verification)

        if judge_verdict == "TEST":
            # 5a) Stage 2: Repair tests
            patched_record, verification, test_repair_rounds_used, diagnoses = repair_tests(
                runtime, record, solution_code, verification,
                max_rounds=repair_rounds, timeout=verify_timeout,
            )
            all_diagnoses.extend(diagnoses)
        else:
            # 5b) Code is wrong — one more repair attempt
            logger.info("Judge says CODE wrong, one more repair | task_id=%s", task_id)
            solution_code, verification, extra, diagnoses = repair_code(
                runtime, record, solution_code, verification,
                max_rounds=1, timeout=verify_timeout,
            )
            repair_rounds_used += extra
            all_diagnoses.extend(diagnoses)

    reasoning = ""
    if verification["passed"]:
        reasoning = generate_reasoning(runtime, record, solution_code)

    return ExecutionResult(
        solution_code=solution_code,
        verification=verification,
        accepted=bool(verification["passed"]),
        record=patched_record,
        repair_rounds_used=repair_rounds_used,
        test_repair_rounds_used=test_repair_rounds_used,
        diagnosis_log=all_diagnoses,
        judge_verdict=judge_verdict,
        reasoning=reasoning,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Solve with Knowledge Base (few-shot RAG)
# ─────────────────────────────────────────────────────────────────────────────

def solve_with_knowledge(
    runtime: RuntimeConfig,
    record: dict,
    few_shot_text: str,
    repair_rounds: int = 2,
    verify_timeout: int = 20,
) -> ExecutionResult:
    """
    Solve a coding task using KB few-shot examples.

    Args:
        runtime: LLM runtime config
        record: task dict with prompt, test_list, etc.
        few_shot_text: formatted few-shot examples from KnowledgeRetriever.format_few_shot()
        repair_rounds: max repair attempts
        verify_timeout: seconds per verification
    """
    task_id = record.get("task_id", "?")
    all_diagnoses: List[str] = []

    # 1) Generate with few-shot context
    logger.info("Solve with KB | task_id=%s", task_id)
    messages = [
        {
            "role": "system",
            "content": "You are a Python coding assistant. Return only runnable Python code, no markdown and no explanation.",
        },
        {
            "role": "user",
            "content": textwrap.dedent(
                f"""
                Here are similar solved tasks for reference:

                {few_shot_text}

                Now solve the new task below following the same patterns.

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
        solution_code = (extract_code_block(generated) or generated).strip()
        solution_code = strip_think_blocks(solution_code)
        logger.info("KB-solve generated | task_id=%s chars=%s", task_id, len(solution_code))
    except Exception as e:
        logger.warning("KB-solve generation failed | task_id=%s error=%s", task_id, e)
        ref = (record.get("reference_solution") or "").strip()
        if ref and "NotImplementedError" not in ref:
            solution_code = ref
        else:
            return ExecutionResult(
                solution_code="", verification={"passed": False, "stderr": str(e)},
                accepted=False, record=record,
            )

    # 2) Verify
    verification = verify_solution(record, solution_code, timeout=verify_timeout)

    repair_rounds_used = 0
    test_repair_rounds_used = 0
    judge_verdict = None
    patched_record = record

    if verification["passed"]:
        return ExecutionResult(
            solution_code=solution_code, verification=verification, accepted=True,
            record=record, diagnosis_log=all_diagnoses,
        )

    # 3) Repair code
    if repair_rounds > 0 and runtime.provider != "offline":
        solution_code, verification, repair_rounds_used, diagnoses = repair_code(
            runtime, record, solution_code, verification,
            max_rounds=repair_rounds, timeout=verify_timeout,
        )
        all_diagnoses.extend(diagnoses)

    if verification["passed"]:
        return ExecutionResult(
            solution_code=solution_code, verification=verification, accepted=True,
            record=record, repair_rounds_used=repair_rounds_used, diagnosis_log=all_diagnoses,
        )

    # 4) Judge + Stage 2
    if repair_rounds > 0 and runtime.provider != "offline":
        judge_verdict = judge_failure(runtime, record, solution_code, verification)
        if judge_verdict == "TEST":
            patched_record, verification, test_repair_rounds_used, diagnoses = repair_tests(
                runtime, record, solution_code, verification,
                max_rounds=repair_rounds, timeout=verify_timeout,
            )
            all_diagnoses.extend(diagnoses)
        else:
            solution_code, verification, extra, diagnoses = repair_code(
                runtime, record, solution_code, verification,
                max_rounds=1, timeout=verify_timeout,
            )
            repair_rounds_used += extra
            all_diagnoses.extend(diagnoses)

    return ExecutionResult(
        solution_code=solution_code,
        verification=verification,
        accepted=bool(verification["passed"]),
        record=patched_record,
        repair_rounds_used=repair_rounds_used,
        test_repair_rounds_used=test_repair_rounds_used,
        diagnosis_log=all_diagnoses,
        judge_verdict=judge_verdict,
    )
