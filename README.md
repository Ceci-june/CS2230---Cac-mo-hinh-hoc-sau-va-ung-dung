# Agent0 — Self-Evolving Code Knowledge Base

Hệ thống tự sinh Knowledge Base (KB) bằng LLM, sau đó dùng KB qua **Few-shot RAG** để cải thiện khả năng giải bài code. Đánh giá trên **MBPP / MBPP+** (EvalPlus).

## Kiến trúc

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: BUILD KNOWLEDGE BASE                              │
│                                                             │
│  curriculum_planner.py                                      │
│    └── LLM tự plan subtopics + reflection rounds           │
│                    │                                        │
│                    ▼                                        │
│  run_agent0_mbpp_curriculum.py                             │
│    └── Sinh tasks → Generate code → Verify → Repair        │
│                    │                                        │
│  executor.py        ▼                                       │
│    ├── generate_solution: LLM viết code                    │
│    ├── verify: chạy assert tests                           │
│    ├── diagnose: phân tích root cause                      │
│    ├── repair_code: sửa code (max 2 rounds)                │
│    ├── judge: code sai hay test sai?                       │
│    ├── repair_tests: sửa test (nếu test sai)               │
│    └── generate_reasoning: giải thích solution             │
│                    │                                        │
│                    ▼                                        │
│        ┌────────────────────────┐                           │
│        │ knowledge_base.jsonl   │                           │
│        │ + .index.json (vector) │                           │
│        └────────────────────────┘                           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 2: SOLVE NEW TASKS (Few-shot RAG)                    │
│                                                             │
│  knowledge_retriever.py                                     │
│    ├── embed query (mxbai-embed-large, 1024-dim)           │
│    ├── cosine similarity vs all KB vectors                 │
│    └── return top-N similar examples                       │
│                    │                                        │
│                    ▼                                        │
│  Few-shot prompt:                                          │
│    Example 1: task + reasoning + code                      │
│    Example 2: task + reasoning + code                      │
│    Now solve: <new task>                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  PHASE 3: BENCHMARK (MBPP / MBPP+ EvalPlus)                 │
│                                                             │
│  benchmark_mbpp.py                                          │
│    └── 378 tasks × {with KB, baseline} → pass@1            │
└─────────────────────────────────────────────────────────────┘
```

## Cài đặt

```bash
cd Source_code/Agent0/Agent0_new
pip install -r requirements_agent0_lite.txt
pip install evalplus  # cho benchmark
```

### API keys

Tạo `.env` ở thư mục gốc project:

```bash
GROQ_API_KEY="..."           # Groq (rate limit 6000 TPM, 100K TPD)
OLLAMA_API_KEY="..."              # Ollama Cloud (không rate limit)
```

### Embedding model

Cần Ollama local cho vector search:

```bash
ollama pull mxbai-embed-large
```

### Providers hỗ trợ

| Provider | Model ví dụ | Rate limit |
|---|---|---|
| `ollama-cloud` | `gemma3:4b`, `gemma3:12b`, `ministral-3:8b`, `qwen3-coder:480b` | Không |
| `ollama` | `qwen3:4b`, `llama3.1:8b` (local) | Không |
| `groq` | `llama-3.3-70b-versatile`, `llama-3.1-8b-instant` | 6000 TPM, 100K TPD |

---

## 1. Tạo Knowledge Base

### Plan subtopics (xem trước)

```bash
python curriculum_planner.py \
  --domain "coding" \
  --total_tasks 30 \
  --reflection_rounds 2 \
  --provider ollama-cloud \
  --model "gemma3:4b" \
  --output logs/plan.json
```

LLM sẽ:
1. **Initial plan**: tự nghĩ ~20 subtopics (basics, sorting, OOP, recursion, ...)
2. **Reflection rounds**: review plan, thêm topics thiếu (DP, graph, sliding window, ...)
3. Output: 50+ subtopics với difficulty + task_count

### Sinh KB tự động

```bash
python run_agent0_mbpp_curriculum.py \
  --provider ollama-cloud \
  --model "gemma3:4b" \
  --synthetic_only \
  --synthetic_generator llm \
  --synthetic_count 30 \
  --synthetic_domain "coding" \
  --strategy all \
  --items_per_strategy 10 \
  --log_file logs/kb_run.log
```

Pipeline cho mỗi task:
```
Generate → Verify → FAIL?
  ├─ Diagnose root cause
  ├─ Repair code (×2)
  └─ Judge: CODE or TEST wrong?
       ├─ TEST → Repair tests (×2)
       └─ CODE → Repair code (+1)
  → PASS → Generate reasoning → KB
```

### Chạy nhiều batches (KB tích lũy)

```bash
# Mỗi seed sinh tasks khác nhau, append vào KB hiện có
for seed in 100 200 300 400 500; do
  python run_agent0_mbpp_curriculum.py \
    --provider ollama-cloud --model "gemma3:4b" \
    --synthetic_only --synthetic_generator llm \
    --synthetic_count 30 --synthetic_domain "coding" \
    --strategy all --items_per_strategy 10 --seed $seed \
    --quiet --log_file logs/kb_${seed}.log &
done
wait
```

### Output

```
Data/mbpp/curriculum_outputs/
├── gemma3_4b/                          ← KB của model gemma3:4b
│   ├── knowledge_base.jsonl
│   ├── knowledge_base.index.json       ← Vector index (auto-update)
│   ├── rejected.jsonl
│   └── summary.json
├── ministral-3_8b/
├── qwen3-coder_480b/
├── llama-3.3-70b-versatile/
└── all_models/                         ← KB tổng (gộp tất cả models)
    └── knowledge_base.jsonl            ← Có field source_model
```

---

## 2. Query Knowledge Base

```bash
python knowledge_retriever.py \
  --kb_path Data/mbpp/curriculum_outputs/all_models/knowledge_base.jsonl \
  --query "Write a function to find the GCD of two numbers" \
  --n 3
```

Output: top-3 bài tương tự nhất với cosine similarity.

### Dùng trong code

```python
from knowledge_retriever import KnowledgeRetriever

retriever = KnowledgeRetriever(kb_path="path/to/knowledge_base.jsonl")
retriever.build_index()  # Lần đầu embed, sau load từ cache

examples = retriever.query("Write a function to find GCD", n=2)
few_shot = retriever.format_few_shot(examples)
```

---

## 3. Giải bài mới (Few-shot RAG)

```python
from executor import solve_with_knowledge, RuntimeConfig
from knowledge_retriever import KnowledgeRetriever

retriever = KnowledgeRetriever(kb_path="path/to/knowledge_base.jsonl")
retriever.build_index()

task = {
    "task_id": "new_1",
    "prompt": "Write a function to find min cost path...",
    "test_list": ["assert min_cost([[1,2],[3,4]], 1, 1) == 5"],
    "challenge_test_list": [],
    "test_setup_code": "",
}

examples = retriever.query(task["prompt"], n=2)
few_shot = retriever.format_few_shot(examples)

rt = RuntimeConfig(
    provider="ollama-cloud",
    model="gemma3:12b",
    backend="Ollama Cloud",
    base_url="https://ollama.com",
    api_key="your_key",
)

result = solve_with_knowledge(rt, task, few_shot, repair_rounds=2)
print(f"Accepted: {result.accepted}")
print(result.solution_code)
```

---

## 4. Demo Web App (Gradio)

Chạy demo web UI để gõ câu hỏi coding và nhận code + giải thích, có sử dụng KB few-shot RAG.

```bash
pip install gradio
python demo.py
```

Mở browser tại **http://localhost:7860**.

Demo features:
- Input: câu hỏi tiếng Việt hoặc tiếng Anh
- Output: code Python + explanation
- Backend: Ollama Cloud (`gemma3:12b`) + KB 485 entries

Yêu cầu:
- File `.env` có `OLLAMA_API_KEY`
- KB tồn tại tại `Data/mbpp/curriculum_outputs/gemma3_4b/knowledge_base.jsonl`
- Ollama local chạy `mxbai-embed-large` cho embedding

---

## 5. Benchmark MBPP / MBPP+

### Full benchmark (with KB + baseline)

```bash
python benchmark_mbpp.py run \
  --provider ollama-cloud \
  --model "gemma3:12b" \
  --kb_path "Data/mbpp/curriculum_outputs/all_models/knowledge_base.jsonl" \
  --n_examples 2
```

### Chỉ generate (không evaluate)

```bash
python benchmark_mbpp.py generate \
  --provider ollama-cloud --model "gemma3:12b" \
  --mode with_kb --n_examples 2 \
  --kb_path "Data/mbpp/curriculum_outputs/all_models/knowledge_base.jsonl" \
  --output Data/mbpp/benchmark_results/gemma3_12b/with_kb_n2.jsonl
```

### Chỉ evaluate

```bash
python benchmark_mbpp.py evaluate \
  --samples_file Data/mbpp/benchmark_results/gemma3_12b/with_kb_n2.jsonl
```

---

## Kết quả thực nghiệm

### Knowledge Base

**485 unique entries** từ 4 models:

| Model | Entries | Accept rate |
|---|---|---|
| gemma3:4b | 382 | ~60% |
| qwen3-coder:480b | 58 | ~79% |
| llama-3.1-8b-instant | 24 | ~50% |
| llama-3.3-70b-versatile | 21 | ~83% |

Topics covered (16): math, strings, lists, dict, set, sorting, search, recursion, dp, graph, datastructures, bitwise, loops, io, regex, datetime.

### Benchmark trên MBPP+ (378 tasks, pass@1)

| Model | Size | Baseline | Best With KB | Improvement |
|---|---|---|---|---|
| **ministral-3:8b** | **8B** | 80.9% / 50.9% | **83.2% / 52.1%** (n=4) | **+2.3% / +1.2%** |
| **gemma3:12b** | **12B** | 87.8% / 54.2% | **89.4% / 54.8%** (n=2) | **+1.6% / +0.6%** |

Format: `MBPP (base) / MBPP+`

### Phát hiện chính

**KB hiệu quả với model trung bình (8B-12B):**
- Model quá yếu (4B): không đủ khả năng tận dụng few-shot
- Model trung bình (8-12B): **được hưởng lợi rõ rệt từ KB**
- Model quá mạnh (20B+): đã đủ giỏi, KB gây nhiễu

**n=2 là sweet spot** cho cả ministral-3:8b và gemma3:12b.

### Ablation study cho gemma3:12b

| n | MBPP (base) | MBPP+ |
|---|---|---|
| Baseline | 87.8% | 54.2% |
| n=1 | 88.4% | 52.1% |
| **n=2** | **89.4%** | **54.8%** |
| n=3 | 88.1% | 53.4% |
| n=4 | 87.3% | 53.4% |
| n=5 | 88.1% | 53.4% |

---

## Cấu trúc files

```
Agent0_new/
├── curriculum_planner.py    Phase 1: plan subtopics + reflection
├── executor.py              Generate / verify / diagnose / repair / judge
├── knowledge_retriever.py   Vector DB: embed, search, format few-shot
├── benchmark_mbpp.py        MBPP/MBPP+ benchmark (EvalPlus)
├── run_agent0_mbpp_curriculum.py  Main pipeline
├── README.md                (file này)
└── requirements_agent0_lite.txt
```

---

## Tham số chính

### `run_agent0_mbpp_curriculum.py`

| Tham số | Mặc định | Mô tả |
|---|---|---|
| `--provider` | `auto` | `ollama`, `ollama-cloud`, `groq`, `openai` |
| `--model` | `llama-3.3-70b-versatile` | Tên model |
| `--synthetic_only` | `False` | Tự sinh task, không dùng MBPP gốc |
| `--synthetic_generator` | `auto` | `llm` hoặc `template` |
| `--synthetic_count` | `60` | Số task cần sinh |
| `--synthetic_domain` | `coding` | Domain cho LLM plan |
| `--strategy` | `all` | `easy_medium_hard`, `diversity`, `mutate`, `all` |
| `--repair_rounds` | `2` | Số repair tối đa |
| `--seed` | `42` | Random seed (đổi để đa dạng) |

### `benchmark_mbpp.py`

| Tham số | Mô tả |
|---|---|
| `--provider`, `--model` | Model đánh giá |
| `--kb_path` | KB file (jsonl) |
| `--n_examples` | Số few-shot examples (1-5) |
| `--limit` | Số tasks tối đa (testing) |
| `--mode` | `with_kb` hoặc `baseline` |
