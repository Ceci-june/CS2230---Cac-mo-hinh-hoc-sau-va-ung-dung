#!/bin/bash
###############################################################################
# Agent0 FULL RL Training trên 2x T4 16GB
#
# Dùng Agent0 framework thật (ADPO + VeRL-Tool + Sandbox)
# Đã chỉnh config cho 2x T4 (giảm batch, bật offload, giảm VRAM vLLM)
#
# Cách dùng:
#   1. Upload toàn bộ Source_code/Agent0/ lên server
#   2. Upload thư mục Data/aime2025/ lên server
#   3. Chỉnh AGENT0_DIR và DATA_DIR bên dưới
#   4. bash run_agent0_train.sh setup     # Cài đặt 1 lần
#      bash run_agent0_train.sh data      # Chuẩn bị data
#      bash run_agent0_train.sh train     # Train executor agent
#      bash run_agent0_train.sh eval      # Eval kết quả
#      bash run_agent0_train.sh all       # Chạy tất cả
###############################################################################
set -x

# Load .env file nếu có
ENV_FILE="$(dirname "$0")/../../.env"
[ -f "$ENV_FILE" ] && export $(grep -v '^#' "$ENV_FILE" | xargs)

# ========================== ĐƯỜNG DẪN ==========================
# Tự detect đường dẫn dựa trên vị trí script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

AGENT0_DIR="${AGENT0_DIR:-$PROJECT_DIR/Source_code/Agent0/Agent0}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR}"
STORAGE_PATH="${STORAGE_PATH:-$PROJECT_DIR/agent0_storage}"
# ===============================================================

# ========================== CONFIG CHO 2x T4 ==========================
# Paper dùng 8x A100 80GB, mình adapt cho 2x T4 16GB:
#   - Giảm batch size: 128 → 8
#   - Bật FSDP offload: param + optimizer offload về CPU
#   - Giảm vLLM memory: 0.7 → 0.35
#   - Giảm rollout samples: 16 → 4
#   - Giảm response length: 4096 → 2048
#   - Rollout mode: async → sync (ít VRAM hơn)

model_name="Qwen/Qwen3-4B-Base"
rl_alg=adpo

# === GPU ===
n_gpus_per_node=2                    # 2x T4 (paper: 8)
n_nodes=1
tensor_model_parallel_size=1

# === Batch sizes (giảm mạnh cho T4) ===
n=4                                  # rollout samples per prompt (paper: 16)
batch_size=8                         # train batch size (paper: 128)
ppo_mini_batch_size=8                # (paper: 128)
ppo_micro_batch_size_per_gpu=1       # giữ nguyên
log_prob_micro_batch_size_per_gpu=2  # (paper: 8) giảm cho T4

# === Sequence lengths (giảm để tiết kiệm VRAM) ===
max_prompt_length=512                # (paper: 1024)
max_response_length=2048             # (paper: 4096)
max_obs_length=256                   # (paper: 512)
max_action_length=1024               # (paper: 2048)

# === Memory optimization cho T4 ===
gpu_memory_utilization=0.35          # (paper: 0.7) giảm cho vLLM
do_offload=True                      # (paper: False) BẬT offload CPU
use_dynamic_bsz=False
strategy="fsdp"
fsdp_size=-1
ulysses_sequence_parallel_size=1
rollout_mode='sync'                  # (paper: async) sync ít VRAM hơn

# === Training ===
temperature=1.0
top_p=1.0
lr=1e-6
kl_loss_coef=1e-2
kl_coef=1e-2
entropy_coeff=0
kl_loss_type=low_var_kl
reward_manager=torl
enable_agent=True
max_turns=4
action_stop_tokens='```output'
additional_eos_token_ids=[151645]
mask_observations=True
enable_mtrl=False

# === Training schedule (giảm cho demo) ===
total_epochs=3                       # (paper: 10)
total_training_steps=30              # (paper: 100)
save_freq=10
test_freq=5

run_name="agent0_aime2025_2xT4"
export VERL_RUN_ID=$run_name
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false

# ========================== FUNCTIONS ==========================

setup() {
    echo "============================================"
    echo "  [Setup] Installing Agent0 dependencies"
    echo "============================================"

    # Check AGENT0_DIR exists
    if [ ! -d "$AGENT0_DIR" ]; then
        echo "ERROR: AGENT0_DIR not found: $AGENT0_DIR"
        echo "Please edit this script and set the correct path."
        exit 1
    fi

    # Conda (optional)
    if command -v conda &>/dev/null; then
        if ! conda env list | grep -q "agent0"; then
            echo "Creating conda env 'agent0'..."
            conda create -n agent0 python=3.12 -y
        fi
        eval "$(conda shell.bash hook)"
        conda activate agent0
    fi

    # Install base dependencies first
    pip install datasets pandas openai huggingface_hub stopit mathruler 2>&1 | tail -5

    cd "$AGENT0_DIR"

    # Install requirements
    echo "Installing requirements..."
    if [ -f requirements.txt ]; then
        pip install -r requirements.txt 2>&1 | tail -5
    else
        pip install torch transformers accelerate vllm ray wandb 2>&1 | tail -5
    fi

    # Install VeRL
    if [ -d executor_train/verl ]; then
        cd executor_train
        pip install -e verl 2>&1 | tail -3
    fi

    # Install flash-attn (optional, may fail on T4 - that's OK)
    echo "Installing flash-attn (optional)..."
    pip install "flash-attn==2.8.3" --no-build-isolation --no-cache-dir 2>&1 | tail -3 || \
        echo "WARNING: flash-attn failed to install. Training will still work without it."

    # Create directories
    mkdir -p "$STORAGE_PATH"/{models,evaluation,generated_question,temp_results}
    mkdir -p "$DATA_DIR"

    echo "Setup complete!"
    nvidia-smi
}

prepare_data() {
    echo "============================================"
    echo "  [Data] Preparing AIME2025 training data"
    echo "============================================"

    eval "$(conda shell.bash hook)"
    conda activate agent0

    mkdir -p "$DATA_DIR"

    python3 << PYEOF
import os
import json
import pandas as pd
from datasets import load_dataset, concatenate_datasets, Dataset

DATA_DIR = "$DATA_DIR"

# 1. Download AIME2025
print("Downloading AIME2025...")
aime1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
aime2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
full = concatenate_datasets([aime1, aime2])

# 2. Convert to Agent0 format
# Agent0 expects: prompt (question text), extra_info.answer
records = []
for row in full:
    records.append({
        "prompt": row["question"],
        "answer": str(row["answer"]),
    })

ds = Dataset.from_list(records)

# Save train and val (dùng cùng data vì AIME2025 chỉ có 30 bài)
train_path = os.path.join(DATA_DIR, "aime2025_train.parquet")
val_path = os.path.join(DATA_DIR, "aime2025_val.parquet")

ds.to_parquet(train_path)
ds.to_parquet(val_path)

print(f"Train data: {train_path} ({len(ds)} examples)")
print(f"Val data:   {val_path} ({len(ds)} examples)")
PYEOF
    echo "Done."
}

train_executor() {
    echo "============================================"
    echo "  [Train] Agent0 Executor Agent (ADPO)"
    echo "  Model:  $model_name"
    echo "  GPUs:   ${n_gpus_per_node}x T4 16GB"
    echo "  Config: batch=$batch_size, n=$n, offload=$do_offload"
    echo "============================================"

    eval "$(conda shell.bash hook)"
    conda activate agent0

    cd "$AGENT0_DIR/executor_train"

    # Training data paths
    train_data="$DATA_DIR/aime2025_train.parquet"
    val_data="[$DATA_DIR/aime2025_val.parquet]"

    # Temp file for action stop tokens
    action_stop_tokens_file=$(mktemp)
    echo -e -n "$action_stop_tokens" > "$action_stop_tokens_file"
    echo "action_stop_tokens_file=$action_stop_tokens_file"

    # Start tool server (Python code sandbox)
    host=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    port=$(shuf -i 30000-31000 -n 1)
    tool_server_url="http://$host:$port/get_observation"

    echo "Starting tool server at $tool_server_url..."
    python -m verl_tool.servers.serve \
        --host "$host" \
        --port "$port" \
        --tool_type "python_code" \
        --workers_per_tool 4 &
    server_pid=$!
    sleep 5
    echo "Tool server started (pid=$server_pid)"

    # ===================== LAUNCH TRAINING =====================
    echo ""
    echo "Launching Agent0 ADPO training..."
    echo ""

    PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
        algorithm.adv_estimator=$rl_alg \
        +actor_rollout_ref.actor.policy_loss_fn=$rl_alg \
        +algorithm.min_score_for_scaling=0.3 \
        +algorithm.max_score_for_scaling=0.8 \
        +algorithm.min_advantage_scale=0.6 \
        +actor_rollout_ref.actor.max_epsilon_bonus=0.1 \
        data.train_files=$train_data \
        data.val_files=$val_data \
        data.train_batch_size=$batch_size \
        data.val_batch_size=32 \
        data.max_prompt_length=$max_prompt_length \
        data.max_response_length=$max_response_length \
        data.truncation='right' \
        reward_model.reward_manager=$reward_manager \
        reward_model.launch_reward_fn_async=False \
        actor_rollout_ref.model.path=$model_name \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=$lr \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']" \
        actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        actor_rollout_ref.actor.use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.strategy=$strategy \
        actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
        actor_rollout_ref.actor.kl_loss_type=$kl_loss_type \
        actor_rollout_ref.actor.entropy_coeff=$entropy_coeff \
        actor_rollout_ref.actor.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=$do_offload \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=$fsdp_size \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        actor_rollout_ref.agent.enable_agent=$enable_agent \
        actor_rollout_ref.agent.tool_server_url=$tool_server_url \
        actor_rollout_ref.agent.max_prompt_length=$max_prompt_length \
        actor_rollout_ref.agent.max_response_length=$max_response_length \
        actor_rollout_ref.agent.max_start_length=$max_prompt_length \
        actor_rollout_ref.agent.max_obs_length=$max_obs_length \
        actor_rollout_ref.agent.max_turns=$max_turns \
        actor_rollout_ref.agent.additional_eos_token_ids=$additional_eos_token_ids \
        actor_rollout_ref.agent.mask_observations=$mask_observations \
        actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
        actor_rollout_ref.agent.enable_mtrl=$enable_mtrl \
        actor_rollout_ref.agent.max_action_length=$max_action_length \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
        actor_rollout_ref.rollout.temperature=$temperature \
        actor_rollout_ref.rollout.top_p=$top_p \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.n=$n \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.rollout.max_num_seqs=64 \
        actor_rollout_ref.rollout.mode=$rollout_mode \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=$use_dynamic_bsz \
        actor_rollout_ref.ref.fsdp_config.param_offload=$do_offload \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$log_prob_micro_batch_size_per_gpu \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        critic.optim.lr=1e-5 \
        critic.strategy=$strategy \
        critic.model.path=$model_name \
        critic.model.fsdp_config.fsdp_size=$fsdp_size \
        critic.model.fsdp_config.param_offload=$do_offload \
        critic.ppo_micro_batch_size_per_gpu=$ppo_micro_batch_size_per_gpu \
        critic.ulysses_sequence_parallel_size=$ulysses_sequence_parallel_size \
        algorithm.kl_ctrl.kl_coef=$kl_coef \
        trainer.logger="['console']" \
        trainer.project_name=$reward_manager \
        trainer.experiment_name=$run_name \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$n_gpus_per_node \
        trainer.nnodes=$n_nodes \
        +trainer.remove_previous_ckpt_in_save=False \
        trainer.save_freq=$save_freq \
        trainer.test_freq=$test_freq \
        trainer.total_epochs=$total_epochs \
        trainer.total_training_steps=$total_training_steps \
    2>&1 | tee "$STORAGE_PATH/training_log.txt"

    # Cleanup
    echo "Stopping tool server..."
    kill -9 $server_pid 2>/dev/null
    rm -f "$action_stop_tokens_file"

    echo ""
    echo "Training complete!"
    echo "Checkpoints saved in: checkpoints/$reward_manager/$run_name/"
    ls -lh "checkpoints/$reward_manager/$run_name/" 2>/dev/null || echo "(no checkpoints found)"
}

eval_checkpoint() {
    echo "============================================"
    echo "  [Eval] Evaluating trained model on AIME2025"
    echo "============================================"

    eval "$(conda shell.bash hook)"
    conda activate agent0

    cd "$AGENT0_DIR/executor_train"

    # Find latest checkpoint
    CKPT_DIR="checkpoints/$reward_manager/$run_name"
    if [ ! -d "$CKPT_DIR" ]; then
        echo "No checkpoint found at $CKPT_DIR"
        echo "Run training first: bash run_agent0_train.sh train"
        exit 1
    fi

    LATEST_CKPT=$(ls -d ${CKPT_DIR}/global_step_*/actor/huggingface 2>/dev/null | sort -V | tail -1)
    if [ -z "$LATEST_CKPT" ]; then
        echo "No HF checkpoint found in $CKPT_DIR"
        exit 1
    fi
    echo "Using checkpoint: $LATEST_CKPT"

    # Start tool server
    host=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    port=$(shuf -i 30000-31000 -n 1)
    tool_server_url="http://$host:$port/get_observation"
    python -m verl_tool.servers.serve --host "$host" --port "$port" \
        --tool_type "python_code" --workers_per_tool 4 --done_if_invalid True --slient True &
    server_pid=$!
    sleep 3

    # Start eval API service
    api_port=5000
    action_stop_tokens_file=$(mktemp)
    echo "$action_stop_tokens" > "$action_stop_tokens_file"

    python eval_service/app.py \
        --host 0.0.0.0 \
        --port $api_port \
        --tool_server_url $tool_server_url \
        --model "$LATEST_CKPT" \
        --max_turns 4 \
        --min_turns 0 \
        --action_stop_tokens "$action_stop_tokens_file" \
        --tensor_parallel_size 1 \
        --num_models 1 \
        --enable_mtrl False &
    api_pid=$!
    echo "Waiting for model to load..."
    sleep 30

    # Run evaluation
    python3 << PYEOF
import json, re, pandas as pd
from openai import OpenAI

df = pd.read_parquet("$DATA_DIR/aime2025_val.parquet")
client = OpenAI(api_key="not-used", base_url="http://localhost:$api_port/v1")

correct, total = 0, len(df)
results = []

for idx, row in df.iterrows():
    q, gt = row["prompt"], str(row["answer"]).strip()
    print(f"[{idx+1}/{total}] {q[:60]}...", end=" ")
    try:
        resp = client.chat.completions.create(
            model="default",
            messages=[
                {"role": "system", "content": "Please reason step by step, and put your final answer within \\\\boxed{}."},
                {"role": "user", "content": q},
            ],
            temperature=0.0, max_tokens=2048,
        )
        pred = resp.choices[0].message.content
        # Extract boxed
        ext = None
        matches = list(re.finditer(r"\\\\boxed\{", pred))
        if matches:
            m = matches[-1]; s, d, i = m.end(), 1, m.end()
            while i < len(pred) and d > 0:
                if pred[i] == "{": d += 1
                elif pred[i] == "}": d -= 1
                i += 1
            if d == 0: ext = pred[s:i-1].strip()
        def norm(a):
            if a is None: return ""
            try: n = float(str(a).strip().replace(",","")); return str(int(n)) if n == int(n) else str(n)
            except: return str(a).strip()
        ok = norm(ext) == norm(gt)
        if ok: correct += 1
        print(f"{'✓' if ok else '✗'} ({ext})")
        results.append({"question": q, "ground_truth": gt, "extracted": ext, "correct": ok})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"question": q, "ground_truth": gt, "extracted": None, "correct": False})

acc = correct / total * 100
print(f"\nAIME2025 Accuracy: {correct}/{total} = {acc:.1f}%")
with open("$STORAGE_PATH/eval_results.json", "w") as f:
    json.dump({"accuracy": acc, "correct": correct, "total": total, "results": results}, f, indent=2, ensure_ascii=False)
print(f"Results saved to $STORAGE_PATH/eval_results.json")
PYEOF

    # Cleanup
    kill -9 $api_pid $server_pid 2>/dev/null
    rm -f "$action_stop_tokens_file"
}

# ========================== MAIN ==========================

case "${1:-help}" in
    setup)    setup ;;
    data)     prepare_data ;;
    train)    train_executor ;;
    eval)     eval_checkpoint ;;
    all)
        setup
        prepare_data
        train_executor
        eval_checkpoint
        echo ""
        echo "============================================"
        echo "  ALL DONE!"
        echo "  Training log: $STORAGE_PATH/training_log.txt"
        echo "  Eval results: $STORAGE_PATH/eval_results.json"
        echo "============================================"
        ;;
    *)
        echo "Usage: bash $0 {setup|data|train|eval|all}"
        echo ""
        echo "  setup  - Install Agent0 dependencies (1 lần)"
        echo "  data   - Download & prepare AIME2025"
        echo "  train  - Train executor agent (ADPO + VeRL-Tool)"
        echo "  eval   - Evaluate trained checkpoint"
        echo "  all    - Run full pipeline"
        ;;
esac
