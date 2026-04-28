#!/usr/bin/env python3
"""
Knowledge Retriever — Vector-based KB search for few-shot code generation
=========================================================================

Converts the knowledge base (JSONL) into a vector index using embeddings,
then retrieves the most similar tasks as few-shot examples.

Usage:
  from knowledge_retriever import KnowledgeRetriever

  retriever = KnowledgeRetriever(kb_path="path/to/knowledge_base.jsonl")
  retriever.build_index()  # generates embeddings and builds index

  examples = retriever.query("Write a function to sort a list", n=3)
  # Returns list of {"original_prompt": ..., "solution_code": ...}

Supports:
  - Ollama local embeddings (default: mxbai-embed-large)
  - Ollama cloud embeddings
  - Persistent index (saves to .index.json alongside KB)
"""

from __future__ import annotations

import json
import logging
import math
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("knowledge_retriever")

# ─────────────────────────────────────────────────────────────────────────────
# Embedding helpers
# ─────────────────────────────────────────────────────────────────────────────

def _http_post(url: str, payload: dict, headers: Optional[dict] = None, timeout: int = 60) -> dict:
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


def get_embedding(
    text: str,
    model: str = "mxbai-embed-large",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
) -> List[float]:
    """Get embedding vector for a text string via Ollama API."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": text}
    result = _http_post(f"{base_url.rstrip('/')}/api/embed", payload, headers=headers)

    embeddings = result.get("embeddings", [])
    if not embeddings or not embeddings[0]:
        raise RuntimeError(f"Empty embedding returned for model {model}")
    return embeddings[0]


def get_embeddings_batch(
    texts: List[str],
    model: str = "mxbai-embed-large",
    base_url: str = "http://localhost:11434",
    api_key: Optional[str] = None,
) -> List[List[float]]:
    """Get embeddings for multiple texts."""
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {"model": model, "input": texts}
    result = _http_post(f"{base_url.rstrip('/')}/api/embed", payload, headers=headers, timeout=120)

    embeddings = result.get("embeddings", [])
    if len(embeddings) != len(texts):
        raise RuntimeError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# Vector math (no numpy dependency)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ─────────────────────────────────────────────────────────────────────────────
# KnowledgeRetriever
# ─────────────────────────────────────────────────────────────────────────────

class KnowledgeRetriever:
    def __init__(
        self,
        kb_path: str,
        embed_model: str = "mxbai-embed-large",
        embed_base_url: str = "http://localhost:11434",
        embed_api_key: Optional[str] = None,
    ):
        self.kb_path = Path(kb_path)
        self.index_path = self.kb_path.with_suffix(".index.json")
        self.embed_model = embed_model
        self.embed_base_url = embed_base_url
        self.embed_api_key = embed_api_key

        self.entries: List[dict] = []
        self.embeddings: List[List[float]] = []
        self._loaded = False

    def _load_kb(self) -> List[dict]:
        """Load and deduplicate KB entries."""
        if not self.kb_path.exists():
            logger.warning("KB not found: %s", self.kb_path)
            return []

        entries = []
        seen_prompts = set()
        with self.kb_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("original_prompt", row.get("curriculum_prompt", ""))
                code = row.get("solution_code", "")
                if not prompt or not code:
                    continue
                # Deduplicate
                sig = prompt[:150]
                if sig in seen_prompts:
                    continue
                seen_prompts.add(sig)
                entries.append({
                    "original_prompt": prompt,
                    "solution_code": code,
                    "reasoning": row.get("reasoning", ""),
                    "difficulty": row.get("difficulty", "unknown"),
                    "taxonomy": row.get("taxonomy", []),
                    "uid": row.get("uid", ""),
                })
        logger.info("Loaded KB | %s unique entries from %s", len(entries), self.kb_path)
        return entries

    def _load_index(self) -> bool:
        """Try to load cached index from disk."""
        if not self.index_path.exists():
            return False
        try:
            with self.index_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if data.get("embed_model") != self.embed_model:
                logger.info("Index model mismatch, will rebuild")
                return False
            if data.get("entry_count") != len(self.entries):
                logger.info("Index entry count mismatch (%s vs %s), will rebuild",
                           data.get("entry_count"), len(self.entries))
                return False
            self.embeddings = data["embeddings"]
            logger.info("Loaded cached index | %s embeddings from %s", len(self.embeddings), self.index_path)
            return True
        except Exception as e:
            logger.warning("Failed to load index: %s", e)
            return False

    def _save_index(self) -> None:
        """Save index to disk for reuse."""
        data = {
            "embed_model": self.embed_model,
            "entry_count": len(self.entries),
            "embeddings": self.embeddings,
        }
        with self.index_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.info("Saved index | %s embeddings to %s", len(self.embeddings), self.index_path)

    def build_index(self, force: bool = False) -> int:
        """Load KB and generate embeddings. Returns number of indexed entries."""
        self.entries = self._load_kb()
        if not self.entries:
            self._loaded = True
            return 0

        # Try cached index
        if not force and self._load_index():
            self._loaded = True
            return len(self.entries)

        # Generate embeddings in batches
        logger.info("Building index | %s entries, model=%s", len(self.entries), self.embed_model)
        prompts = [e["original_prompt"] for e in self.entries]
        batch_size = 32
        self.embeddings = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            logger.info("Embedding batch %s/%s", i // batch_size + 1, (len(prompts) + batch_size - 1) // batch_size)
            batch_embeddings = get_embeddings_batch(
                batch,
                model=self.embed_model,
                base_url=self.embed_base_url,
                api_key=self.embed_api_key,
            )
            self.embeddings.extend(batch_embeddings)

        self._save_index()
        self._loaded = True
        logger.info("Index built | %s entries indexed", len(self.entries))
        return len(self.entries)

    def add_entry(self, entry: dict) -> bool:
        """
        Add a new accepted task to the index immediately.
        Embeds the prompt and appends to both entries and embeddings.
        Saves updated index to disk.

        Returns True if added, False if duplicate or missing data.
        """
        prompt = entry.get("original_prompt", entry.get("curriculum_prompt", ""))
        code = entry.get("solution_code", "")
        if not prompt or not code:
            return False

        # Check duplicate
        sig = prompt[:150]
        existing_sigs = {e["original_prompt"][:150] for e in self.entries}
        if sig in existing_sigs:
            logger.debug("add_entry skipped duplicate | %s", sig[:60])
            return False

        # Embed and append
        emb = get_embedding(
            prompt,
            model=self.embed_model,
            base_url=self.embed_base_url,
            api_key=self.embed_api_key,
        )

        new_entry = {
            "original_prompt": prompt,
            "solution_code": code,
            "difficulty": entry.get("difficulty", "unknown"),
            "taxonomy": entry.get("taxonomy", []),
            "uid": entry.get("uid", ""),
        }
        self.entries.append(new_entry)
        self.embeddings.append(emb)
        self._save_index()

        logger.info("add_entry | indexed new entry (%s total) | %s", len(self.entries), prompt[:60])
        return True

    def rewrite_query(self, prompt: str, runtime_config: Optional[dict] = None) -> str:
        """
        Rewrite/expand the query prompt for better KB retrieval.
        Uses LLM to rephrase and add related keywords.
        """
        if not runtime_config:
            return prompt

        import urllib.request, json
        provider = runtime_config.get("provider", "ollama")
        model = runtime_config.get("model", "gemma3:4b")
        base_url = runtime_config.get("base_url", "http://localhost:11434")
        api_key = runtime_config.get("api_key")

        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the coding task below into a clearer, more searchable form. "
                    "Add related algorithm/technique keywords. "
                    "Return only the rewritten query, one paragraph, no code."
                ),
            },
            {"role": "user", "content": prompt},
        ]

        try:
            if provider in ("ollama", "ollama-cloud"):
                payload = {"model": model, "messages": messages, "stream": False, "options": {"temperature": 0.1, "num_predict": 200}}
                headers = {"Content-Type": "application/json", "User-Agent": "Agent0/1.0"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                data = json.dumps(payload).encode()
                req = urllib.request.Request(f"{base_url.rstrip('/')}/api/chat", data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                rewritten = (body.get("message", {}).get("content", "") or "").strip()
            else:
                payload = {"model": model, "messages": messages, "temperature": 0.1, "max_tokens": 200}
                headers = {"Content-Type": "application/json", "User-Agent": "Agent0/1.0"}
                if api_key:
                    headers["Authorization"] = f"Bearer {api_key}"
                data = json.dumps(payload).encode()
                req = urllib.request.Request(f"{base_url.rstrip('/')}/chat/completions", data=data, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = json.loads(resp.read().decode())
                rewritten = (body.get("choices", [{}])[0].get("message", {}).get("content", "") or "").strip()

            # Strip think blocks
            import re
            rewritten = re.sub(r"<think>.*?</think>", "", rewritten, flags=re.DOTALL).strip()

            if rewritten:
                logger.info("Query rewritten | original=%s... → rewritten=%s...", prompt[:60], rewritten[:60])
                return rewritten
        except Exception as e:
            logger.warning("Query rewrite failed: %s", e)

        return prompt

    def query(self, prompt: str, n: int = 3, runtime_config: Optional[dict] = None) -> List[dict]:
        """
        Find the n most similar KB entries to the given prompt.
        Optionally rewrites query via LLM before embedding.

        Returns list of dicts with: original_prompt, solution_code, similarity
        """
        if not self._loaded:
            self.build_index()

        if not self.entries or not self.embeddings:
            return []

        # Rewrite query for better retrieval
        search_prompt = self.rewrite_query(prompt, runtime_config) if runtime_config else prompt

        # Embed query
        query_emb = get_embedding(
            search_prompt,
            model=self.embed_model,
            base_url=self.embed_base_url,
            api_key=self.embed_api_key,
        )

        # Compute similarities
        scored: List[Tuple[float, int]] = []
        for idx, emb in enumerate(self.embeddings):
            sim = _cosine_similarity(query_emb, emb)
            scored.append((sim, idx))

        # Top-n
        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for sim, idx in scored[:n]:
            entry = self.entries[idx]
            results.append({
                "original_prompt": entry["original_prompt"],
                "solution_code": entry["solution_code"],
                "reasoning": entry.get("reasoning", ""),
                "difficulty": entry["difficulty"],
                "similarity": round(sim, 4),
            })

        return results

    def format_few_shot(self, examples: List[dict]) -> str:
        """Format retrieved examples as few-shot prompt text with reasoning."""
        if not examples:
            return ""

        parts = []
        for i, ex in enumerate(examples, 1):
            reasoning_block = ""
            if ex.get("reasoning"):
                reasoning_block = f"Reasoning:\n{ex['reasoning']}\n"
            parts.append(
                f"### Example {i}\n"
                f"Task: {ex['original_prompt']}\n"
                f"{reasoning_block}"
                f"Solution:\n```python\n{ex['solution_code']}\n```"
            )
        return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# CLI — test retrieval
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Knowledge Retriever — query KB for similar tasks")
    parser.add_argument("--kb_path", type=str, required=True, help="Path to knowledge_base.jsonl")
    parser.add_argument("--query", type=str, required=True, help="Task prompt to search for")
    parser.add_argument("--n", type=int, default=3, help="Number of results")
    parser.add_argument("--embed_model", type=str, default="mxbai-embed-large")
    parser.add_argument("--embed_base_url", type=str, default="http://localhost:11434")
    parser.add_argument("--embed_api_key", type=str, default=None)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    retriever = KnowledgeRetriever(
        kb_path=args.kb_path,
        embed_model=args.embed_model,
        embed_base_url=args.embed_base_url,
        embed_api_key=args.embed_api_key,
    )
    retriever.build_index(force=args.force_rebuild)

    results = retriever.query(args.query, n=args.n)

    print(f"\nQuery: {args.query}")
    print(f"{'='*60}")
    for i, r in enumerate(results, 1):
        print(f"\n{i}. [sim={r['similarity']:.4f}] {r['original_prompt'][:100]}")
        print(f"   Code: {r['solution_code'][:150]}...")
    print(f"\n{'='*60}")
    print(retriever.format_few_shot(results))
