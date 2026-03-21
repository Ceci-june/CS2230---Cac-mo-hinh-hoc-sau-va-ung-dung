#!/bin/bash
###############################################################################
# Agent0 FULL CO-EVOLUTION Pipeline (2x T4 16GB)
#
# Đúng như paper: Curriculum Agent ↔ Executor Agent co-evolution
#
# Pipeline:
#   Iter 1:
#     Step 1: Train Curriculum Agent (GRPO)
#     Step 2: Curriculum Agent sinh 250 câu hỏi (paper: 1000x8 GPU)
#     Step 3: Executor Agent đánh giá + lọc câu hỏi (self-consistency)
#     Step 4: Train Executor Agent (ADPO) trên câu hỏi đã lọc
#     Step 5: Eval trên AIME2025
#   Iter 2-3: Lặp lại với model mới
#
# Cách dùng:
#   1. Upload Source_code/Agent0/ + Data/aime2025/ lên server
#   2. Chỉnh AGENT0_DIR bên dưới
#   3. bash run_agent0_full.sh setup
#   4. bash run_agent0_full.sh test         # Test GPU trước khi chạy
#      bash run_agent0_full.sh all          # Full 3 iterations
#      bash run_agent0_full.sh iter 1       # Chỉ chạy iteration 1
#      bash run_agent0_full.sh curriculum 1 # Chỉ train curriculum iter 1
#      bash run_agent0_full.sh generate 1   # Chỉ sinh câu hỏi iter 1
#      bash run_agent0_full.sh executor 1   # Chỉ train executor iter 1
#      bash run_agent0_full.sh eval 1       # Chỉ eval iter 1
#
# Yêu cầu: 2x NVIDIA T4 16GB, ~60GB disk, ~8GB RAM
###############################################################################
set -x

# Load .env
ENV_FILE="$(dirname "$0")/../../.env"
[ -f "$ENV_FILE" ] && export $(grep -v '^#' "$ENV_FILE" | xargs)

# ========================== ĐƯỜNG DẪN ==========================
# >>> CHỈNH CHO ĐÚNG SERVER <<<
# Tự detect đường dẫn dựa trên vị trí script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

AGENT0_DIR="${AGENT0_DIR:-$PROJECT_DIR/Source_code/Agent0/Agent0}"
DATA_DIR="${DATA_DIR:-$SCRIPT_DIR}"
STORAGE_PATH="${STORAGE_PATH:-$PROJECT_DIR/agent0_storage}"
export STORAGE_PATH
# ===============================================================

# ========================== CONFIG CHUNG ==========================
BASE_MODEL="Qwen/Qwen3-4B-Base"
NUM_ITERATIONS=3                    # Paper: 3 co-evolution iterations
NUM_QUESTIONS_PER_GPU=125           # Paper: 1000/8=125 per GPU. Giảm nếu chậm
NUM_EVAL_SAMPLES=9                  # Paper: 9 samples per question
MIN_SCORE=0.3                       # Lọc câu hỏi: min self-consistency
MAX_SCORE=0.8                       # Lọc câu hỏi: max self-consistency

# GPU config cho 2x T4
N_GPUS=2

export VLLM_DISABLE_COMPILE_CACHE=1
export NCCL_DEBUG=WARN
export VLLM_USE_V1=1
export TOKENIZERS_PARALLELISM=false
# T4 chỉ hỗ trợ fp16, không hỗ trợ bf16
export VERL_ACTOR_DTYPE=fp16
export VERL_CRITIC_DTYPE=fp16
# ===============================================================

log() {
    echo ""
    echo "============================================"
    echo "  $1"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"
}

# ========================== SETUP ==========================
setup() {
    log "Installing Agent0 dependencies"

    # Check AGENT0_DIR exists
    if [ ! -d "$AGENT0_DIR" ]; then
        echo "ERROR: AGENT0_DIR not found: $AGENT0_DIR"
        echo "Please edit this script and set the correct path."
        echo "Run: find /home -name 'requirements.txt' -path '*/Agent0/*' 2>/dev/null"
        exit 1
    fi

    # Conda (optional - skip if not available)
    if command -v conda &>/dev/null; then
        if ! conda env list | grep -q "agent0"; then
            conda create -n agent0 python=3.12 -y
        fi
        eval "$(conda shell.bash hook)"
        conda activate agent0
    fi

    # Install base dependencies first
    pip install datasets pandas openai huggingface_hub stopit mathruler 2>&1 | tail -5

    # Install curriculum_train deps (if requirements.txt exists)
    if [ -f "$AGENT0_DIR/requirements.txt" ]; then
        cd "$AGENT0_DIR"
        pip install -r requirements.txt 2>&1 | tail -5
    else
        echo "WARNING: $AGENT0_DIR/requirements.txt not found, installing core deps..."
        pip install torch transformers accelerate vllm ray datasets wandb 2>&1 | tail -5
    fi

    # Install executor_train deps (if verl exists)
    if [ -d "$AGENT0_DIR/executor_train/verl" ]; then
        cd "$AGENT0_DIR/executor_train"
        pip install -e verl 2>&1 | tail -3
    else
        echo "WARNING: verl not found at $AGENT0_DIR/executor_train/verl"
    fi

    # Flash attention (optional, may fail on T4)
    pip install "flash-attn==2.8.3" --no-build-isolation --no-cache-dir 2>&1 | tail -3 || true

    # Create directories
    mkdir -p "$STORAGE_PATH"/{models,evaluation,generated_question,temp_results}
    mkdir -p "$DATA_DIR"

    # Prepare AIME2025 eval data
    python3 << 'PYEOF'
import os
from datasets import load_dataset, concatenate_datasets, Dataset
DATA_DIR = os.environ["DATA_DIR"]
aime1 = load_dataset("opencompass/AIME2025", "AIME2025-I", split="test")
aime2 = load_dataset("opencompass/AIME2025", "AIME2025-II", split="test")
full = concatenate_datasets([aime1, aime2])
# Format cho executor_train eval
records = [{"prompt": r["question"], "answer": str(r["answer"])} for r in full]
Dataset.from_list(records).to_parquet(f"{DATA_DIR}/aime2025_val.parquet")
print(f"AIME2025 val set: {len(records)} problems")
PYEOF

    nvidia-smi
    log "Setup complete!"
}

# ========================== CURRICULUM AGENT TRAINING ==========================
train_curriculum() {
    local ITER=$1
    local EXECUTOR_PATH=$2
    local CURRICULUM_PATH=$3
    local SAVE_NAME="curriculum_iter${ITER}"

    log "Iter $ITER: Training Curriculum Agent (GRPO)"
    echo "  Executor:   $EXECUTOR_PATH"
    echo "  Curriculum:  $CURRICULUM_PATH"
    echo "  Save:        $SAVE_NAME"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/curriculum_train"

    export RUN_ID=$(date +%s%N)

    # Start vLLM server for executor (để curriculum reward tính self-consistency)
    # Trên 2x T4, dùng 1 GPU cho vLLM server, 1 GPU cho curriculum training
    echo "Starting vLLM executor service on GPU 1..."
    CUDA_VISIBLE_DEVICES=1 python vllm_service_init/start_vllm_server_tool.py \
        --model "$EXECUTOR_PATH" --run_id "$RUN_ID" &
    vllm_pid=$!
    sleep 30  # Đợi model load

    # Train curriculum agent trên GPU 0
    echo "Training curriculum agent on GPU 0..."
    CUDA_VISIBLE_DEVICES=0 python3 -m verl.trainer.main \
        config=examples/config.yaml \
        data.max_response_length=2048 \
        worker.actor.model.model_path="$CURRICULUM_PATH" \
        trainer.experiment_name="$SAVE_NAME" \
        trainer.save_checkpoint_path="${STORAGE_PATH}/models/$SAVE_NAME" \
        trainer.total_epochs=1000 \
        worker.reward.reward_function=./examples/reward_function/curriculum_reward.py:compute_score \
        trainer.val_freq=-1 \
        trainer.n_gpus_per_node=1 \
        data.format_prompt=./examples/format_prompt/questioner.jinja \
        worker.rollout.n=4 \
        worker.actor.global_batch_size=16 \
        trainer.logger="['console']" \
        trainer.project_name=agent0 \
        trainer.max_steps=6 \
        trainer.save_freq=1 \
    2>&1 | tee "$STORAGE_PATH/log_curriculum_iter${ITER}.txt"

    # Merge model
    echo "Merging curriculum model..."
    python scripts/model_merger.py \
        --local_dir "${STORAGE_PATH}/models/$SAVE_NAME/global_step_5/actor" 2>/dev/null || true

    # Cleanup
    kill -9 $vllm_pid 2>/dev/null
    sleep 5

    log "Iter $ITER: Curriculum Agent training done"
    echo "  Model: ${STORAGE_PATH}/models/$SAVE_NAME/global_step_5/actor/huggingface"
}

# ========================== QUESTION GENERATION ==========================
generate_questions() {
    local ITER=$1
    local CURRICULUM_PATH=$2
    local SAVE_NAME="iter${ITER}"

    log "Iter $ITER: Generating questions with Curriculum Agent"
    echo "  Model: $CURRICULUM_PATH"
    echo "  Questions per GPU: $NUM_QUESTIONS_PER_GPU"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/curriculum_train"

    # 2x T4: chạy song song trên 2 GPU
    for i in $(seq 0 $((N_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python -m question_generate.question_generate \
            --model "$CURRICULUM_PATH" \
            --suffix $i \
            --num_samples $NUM_QUESTIONS_PER_GPU \
            --save_name "$SAVE_NAME" &
    done
    wait

    # Count generated questions
    total=0
    for i in $(seq 0 $((N_GPUS - 1))); do
        f="${STORAGE_PATH}/generated_question/${SAVE_NAME}_${i}.json"
        if [ -f "$f" ]; then
            n=$(python3 -c "import json; print(len(json.load(open('$f'))))")
            total=$((total + n))
            echo "  GPU $i: $n questions"
        fi
    done
    echo "  Total: $total questions generated"

    log "Iter $ITER: Question generation done ($total questions)"
}

# ========================== QUESTION EVALUATION ==========================
evaluate_questions() {
    local ITER=$1
    local EXECUTOR_PATH=$2
    local SAVE_NAME="iter${ITER}"

    log "Iter $ITER: Evaluating questions with Executor Agent"
    echo "  Executor: $EXECUTOR_PATH"
    echo "  Samples per question: $NUM_EVAL_SAMPLES"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/curriculum_train"

    # 2x T4: chạy song song trên 2 GPU
    pids=()
    for i in $(seq 0 $((N_GPUS - 1))); do
        CUDA_VISIBLE_DEVICES=$i python question_evaluate/evaluate.py \
            --model "$EXECUTOR_PATH" \
            --suffix $i \
            --save_name "$SAVE_NAME" &
        pids[$i]=$!
    done

    # Wait with timeout (1 hour)
    wait ${pids[0]}
    echo "GPU 0 evaluation done."

    (
        sleep 3600
        echo "Timeout! Killing remaining eval..."
        for i in $(seq 1 $((N_GPUS - 1))); do
            kill -9 ${pids[$i]} 2>/dev/null
        done
    ) &
    timeout_pid=$!

    for i in $(seq 1 $((N_GPUS - 1))); do
        wait ${pids[$i]} 2>/dev/null
    done
    kill -9 $timeout_pid 2>/dev/null

    log "Iter $ITER: Question evaluation done"
}

# ========================== DATA FILTERING ==========================
filter_data() {
    local ITER=$1
    local SAVE_NAME="iter${ITER}"

    log "Iter $ITER: Filtering questions (score $MIN_SCORE - $MAX_SCORE)"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/curriculum_train"

    # Adapt upload.py for N_GPUS (paper uses 8)
    python3 << PYEOF
import json, os
from datasets import Dataset

STORAGE_PATH = os.environ["STORAGE_PATH"]
save_name = "$SAVE_NAME"
min_score, max_score = $MIN_SCORE, $MAX_SCORE

datas = []
for i in range($N_GPUS):
    f = f"{STORAGE_PATH}/generated_question/{save_name}_{i}_results.json"
    if os.path.exists(f):
        with open(f) as fh:
            data = json.load(fh)
            datas.extend(data)
            print(f"  Loaded {len(data)} from GPU {i}")

filtered = [
    {"problem": d["question"], "answer": d["answer"], "score": d["score"]}
    for d in datas
    if min_score <= d.get("score", 0) <= max_score and d.get("answer")
]

print(f"  Total: {len(datas)} → Filtered: {len(filtered)}")

if filtered:
    save_dir = f"{STORAGE_PATH}/generated_question/{save_name}"
    os.makedirs(save_dir, exist_ok=True)
    save_path = f"{save_dir}/train.parquet"
    Dataset.from_list(filtered).to_parquet(save_path)
    print(f"  Saved to {save_path}")
else:
    print("  WARNING: No data after filtering!")
PYEOF

    log "Iter $ITER: Data filtering done"
}

# ========================== EXECUTOR AGENT TRAINING (ADPO) ==========================
train_executor() {
    local ITER=$1
    local TRAIN_DATA=$2
    local EXECUTOR_PATH=$3
    local RUN_NAME="executor_iter${ITER}"

    log "Iter $ITER: Training Executor Agent (ADPO)"
    echo "  Train data: $TRAIN_DATA"
    echo "  Model:      $EXECUTOR_PATH"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/executor_train"

    val_data="[$DATA_DIR/aime2025_val.parquet]"

    # Action stop tokens
    action_stop_tokens_file=$(mktemp)
    echo -e -n '```output' > "$action_stop_tokens_file"

    # Start tool server
    host=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    port=$(shuf -i 30000-31000 -n 1)
    tool_server_url="http://$host:$port/get_observation"
    python -m verl_tool.servers.serve --host "$host" --port "$port" \
        --tool_type "python_code" --workers_per_tool 4 &
    server_pid=$!
    sleep 5

    # ==================== CONFIG CHO 2x T4 ====================
    PYTHONUNBUFFERED=1 python3 -m verl_tool.trainer.main_ppo \
        algorithm.adv_estimator=adpo \
        +actor_rollout_ref.actor.policy_loss_fn=adpo \
        +algorithm.min_score_for_scaling=0.3 \
        +algorithm.max_score_for_scaling=0.8 \
        +algorithm.min_advantage_scale=0.6 \
        +actor_rollout_ref.actor.max_epsilon_bonus=0.1 \
        data.train_files="$TRAIN_DATA" \
        data.val_files="$val_data" \
        data.train_batch_size=8 \
        data.val_batch_size=32 \
        data.max_prompt_length=512 \
        data.max_response_length=2048 \
        data.truncation='right' \
        reward_model.reward_manager=torl \
        reward_model.launch_reward_fn_async=False \
        actor_rollout_ref.model.path="$EXECUTOR_PATH" \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=5 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.model.trust_remote_code=True \
        actor_rollout_ref.actor.checkpoint.save_contents="['model','hf_model']" \
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_dynamic_bsz=False \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.strategy=fsdp \
        actor_rollout_ref.actor.kl_loss_coef=1e-2 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
        actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
        actor_rollout_ref.agent.enable_agent=True \
        actor_rollout_ref.agent.tool_server_url=$tool_server_url \
        actor_rollout_ref.agent.max_prompt_length=512 \
        actor_rollout_ref.agent.max_response_length=2048 \
        actor_rollout_ref.agent.max_start_length=512 \
        actor_rollout_ref.agent.max_obs_length=256 \
        actor_rollout_ref.agent.max_turns=4 \
        actor_rollout_ref.agent.additional_eos_token_ids=[151645] \
        actor_rollout_ref.agent.mask_observations=True \
        actor_rollout_ref.agent.action_stop_tokens=$action_stop_tokens_file \
        actor_rollout_ref.agent.enable_mtrl=False \
        actor_rollout_ref.agent.max_action_length=1024 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.enforce_eager=True \
        actor_rollout_ref.rollout.free_cache_engine=True \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.35 \
        actor_rollout_ref.rollout.temperature=1.0 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.n=4 \
        actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.rollout.max_num_seqs=64 \
        actor_rollout_ref.rollout.mode=sync \
        actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.ulysses_sequence_parallel_size=1 \
        critic.optim.lr=1e-5 \
        critic.strategy=fsdp \
        critic.model.path="$EXECUTOR_PATH" \
        critic.model.fsdp_config.fsdp_size=-1 \
        critic.model.fsdp_config.param_offload=True \
        critic.ppo_micro_batch_size_per_gpu=1 \
        critic.ulysses_sequence_parallel_size=1 \
        algorithm.kl_ctrl.kl_coef=1e-2 \
        trainer.logger="['console']" \
        trainer.project_name=torl \
        trainer.experiment_name="$RUN_NAME" \
        trainer.val_before_train=True \
        trainer.default_hdfs_dir=null \
        trainer.n_gpus_per_node=$N_GPUS \
        trainer.nnodes=1 \
        +trainer.remove_previous_ckpt_in_save=False \
        trainer.save_freq=10 \
        trainer.test_freq=5 \
        trainer.total_epochs=3 \
        trainer.total_training_steps=30 \
    2>&1 | tee "$STORAGE_PATH/log_executor_iter${ITER}.txt"

    # Cleanup
    kill -9 $server_pid 2>/dev/null
    rm -f "$action_stop_tokens_file"

    log "Iter $ITER: Executor Agent training done"
    echo "  Checkpoints: checkpoints/torl/$RUN_NAME/"
}

# ========================== EVAL ==========================
eval_aime2025() {
    local ITER=$1
    local MODEL_PATH=$2

    log "Iter $ITER: Evaluating on AIME2025"
    echo "  Model: $MODEL_PATH"

    eval "$(conda shell.bash hook)"
    conda activate agent0
    cd "$AGENT0_DIR/executor_train"

    # Start tool server
    host=$(hostname -i 2>/dev/null | awk '{print $1}' || echo "127.0.0.1")
    port=$(shuf -i 30000-31000 -n 1)
    tool_server_url="http://$host:$port/get_observation"
    python -m verl_tool.servers.serve --host "$host" --port "$port" \
        --tool_type "python_code" --workers_per_tool 4 --done_if_invalid True --slient True &
    server_pid=$!
    sleep 3

    # Start API service
    api_port=5000
    action_stop_tokens_file=$(mktemp)
    echo '```output' > "$action_stop_tokens_file"

    python eval_service/app.py \
        --host 0.0.0.0 --port $api_port \
        --tool_server_url $tool_server_url \
        --model "$MODEL_PATH" \
        --max_turns 4 --min_turns 0 \
        --action_stop_tokens "$action_stop_tokens_file" \
        --tensor_parallel_size 1 --num_models 1 \
        --enable_mtrl False &
    api_pid=$!
    sleep 30

    # Evaluate
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
            messages=[{"role":"system","content":"Please reason step by step, and put your final answer within \\\\boxed{}."},
                      {"role":"user","content":q}],
            temperature=0.0, max_tokens=2048)
        pred = resp.choices[0].message.content
        ext = None
        for m in re.finditer(r"\\\\boxed\{", pred):
            s, d, i = m.end(), 1, m.end()
            while i < len(pred) and d > 0:
                if pred[i]=="{": d+=1
                elif pred[i]=="}": d-=1
                i+=1
            if d==0: ext = pred[s:i-1].strip()
        def norm(a):
            if a is None: return ""
            try: n=float(str(a).strip().replace(",","")); return str(int(n)) if n==int(n) else str(n)
            except: return str(a).strip()
        ok = norm(ext)==norm(gt)
        if ok: correct+=1
        print(f"{'✓' if ok else '✗'} ({ext})")
        results.append({"question":q,"ground_truth":gt,"extracted":ext,"correct":ok})
    except Exception as e:
        print(f"ERROR: {e}")
        results.append({"question":q,"ground_truth":gt,"extracted":None,"correct":False})

acc = correct/total*100
print(f"\nIter $ITER AIME2025: {correct}/{total} = {acc:.1f}%")
with open("$STORAGE_PATH/eval_iter${ITER}.json","w") as f:
    json.dump({"iter":$ITER,"accuracy":acc,"correct":correct,"total":total,"model":"$MODEL_PATH","results":results},f,indent=2,ensure_ascii=False)
PYEOF

    kill -9 $api_pid $server_pid 2>/dev/null
    rm -f "$action_stop_tokens_file"

    log "Iter $ITER: Evaluation done"
}

# ========================== 1 ITERATION ==========================
run_iteration() {
    local ITER=$1
    local EXECUTOR_PATH=$2
    local CURRICULUM_PATH=$3

    log "========== ITERATION $ITER / $NUM_ITERATIONS =========="
    echo "  Executor:   $EXECUTOR_PATH"
    echo "  Curriculum:  $CURRICULUM_PATH"

    # Step 1: Train Curriculum Agent
    train_curriculum "$ITER" "$EXECUTOR_PATH" "$CURRICULUM_PATH"
    local NEW_CURRICULUM="${STORAGE_PATH}/models/curriculum_iter${ITER}/global_step_5/actor/huggingface"

    # Step 2: Generate questions
    generate_questions "$ITER" "$NEW_CURRICULUM"

    # Step 3: Evaluate questions (lọc theo self-consistency)
    evaluate_questions "$ITER" "$EXECUTOR_PATH"

    # Step 4: Filter data
    filter_data "$ITER"
    local TRAIN_DATA="${STORAGE_PATH}/generated_question/iter${ITER}/train.parquet"

    # Step 5: Train Executor Agent
    train_executor "$ITER" "$TRAIN_DATA" "$EXECUTOR_PATH"

    # Find latest executor checkpoint
    local CKPT_DIR="checkpoints/torl/executor_iter${ITER}"
    local NEW_EXECUTOR=$(ls -d ${CKPT_DIR}/global_step_*/actor/huggingface 2>/dev/null | sort -V | tail -1)

    if [ -z "$NEW_EXECUTOR" ]; then
        echo "WARNING: No executor checkpoint found, using previous: $EXECUTOR_PATH"
        NEW_EXECUTOR="$EXECUTOR_PATH"
    fi

    # Step 6: Eval on AIME2025
    eval_aime2025 "$ITER" "$NEW_EXECUTOR"

    # Return new paths for next iteration
    echo "EXECUTOR=$NEW_EXECUTOR"
    echo "CURRICULUM=$NEW_CURRICULUM"
}

# ========================== FULL PIPELINE ==========================
run_all() {
    log "Agent0 Full Co-Evolution Pipeline"
    echo "  Base model:  $BASE_MODEL"
    echo "  Iterations:  $NUM_ITERATIONS"
    echo "  GPUs:        ${N_GPUS}x T4"

    # Eval base model first
    eval_aime2025 0 "$BASE_MODEL"

    # Iterative co-evolution
    EXECUTOR_PATH="$BASE_MODEL"
    CURRICULUM_PATH="$BASE_MODEL"

    for ITER in $(seq 1 $NUM_ITERATIONS); do
        run_iteration "$ITER" "$EXECUTOR_PATH" "$CURRICULUM_PATH"

        # Update paths for next iteration
        NEW_CURRICULUM="${STORAGE_PATH}/models/curriculum_iter${ITER}/global_step_5/actor/huggingface"
        NEW_EXECUTOR=$(ls -d "$AGENT0_DIR/executor_train/checkpoints/torl/executor_iter${ITER}"/global_step_*/actor/huggingface 2>/dev/null | sort -V | tail -1)

        [ -d "$NEW_CURRICULUM" ] && CURRICULUM_PATH="$NEW_CURRICULUM"
        [ -n "$NEW_EXECUTOR" ] && EXECUTOR_PATH="$NEW_EXECUTOR"

        echo ""
        echo "Iter $ITER done. Next iteration will use:"
        echo "  Executor:   $EXECUTOR_PATH"
        echo "  Curriculum:  $CURRICULUM_PATH"
    done

    # Final summary
    log "ALL ITERATIONS COMPLETE"
    echo ""
    echo "Results:"
    for i in $(seq 0 $NUM_ITERATIONS); do
        f="$STORAGE_PATH/eval_iter${i}.json"
        if [ -f "$f" ]; then
            acc=$(python3 -c "import json; print(json.load(open('$f'))['accuracy'])")
            echo "  Iter $i: ${acc}%"
        fi
    done
    echo ""
    echo "Logs:        $STORAGE_PATH/log_*.txt"
    echo "Checkpoints: $AGENT0_DIR/executor_train/checkpoints/torl/"
    echo "Eval:        $STORAGE_PATH/eval_iter*.json"
}

# ========================== MAIN ==========================
test_gpu() {
    log "Testing GPU compatibility"

    eval "$(conda shell.bash hook)"
    conda activate agent0

    python3 << 'PYEOF'
import torch
import sys

print("=" * 50)
print("GPU COMPATIBILITY TEST")
print("=" * 50)

# Check CUDA
if not torch.cuda.is_available():
    print("FAIL: CUDA not available!")
    sys.exit(1)

n_gpus = torch.cuda.device_count()
print(f"GPUs: {n_gpus}")

total_mem = 0
for i in range(n_gpus):
    name = torch.cuda.get_device_name(i)
    mem = torch.cuda.get_device_properties(i).total_mem / 1e9
    cap = torch.cuda.get_device_capability(i)
    total_mem += mem
    print(f"  GPU {i}: {name} | {mem:.1f} GB | Compute {cap[0]}.{cap[1]}")

# Check bf16 support
supports_bf16 = torch.cuda.is_bf16_supported()
print(f"\nbf16 support: {'Yes' if supports_bf16 else 'No (will use fp16)'}")

# Check memory estimate for Qwen3-4B ADPO
model_size_fp16 = 8  # ~8GB for 4B params in fp16
needed = model_size_fp16 * 3  # actor + critic + ref (rough estimate)
print(f"\nMemory estimate:")
print(f"  Model (fp16):        ~{model_size_fp16} GB")
print(f"  ADPO (actor+critic+ref): ~{needed} GB")
print(f"  Available (total):   ~{total_mem:.0f} GB")
print(f"  With CPU offload:    Feasible" if total_mem >= 20 else f"  WARNING: Very tight!")

# Try loading a small tensor on each GPU
for i in range(n_gpus):
    try:
        t = torch.randn(1000, 1000, device=f"cuda:{i}")
        del t
        torch.cuda.empty_cache()
        print(f"  GPU {i}: OK")
    except Exception as e:
        print(f"  GPU {i}: FAIL - {e}")

# Check vllm
try:
    import vllm
    print(f"\nvLLM: {vllm.__version__}")
except ImportError:
    print("\nvLLM: NOT INSTALLED (run setup first)")

# Check verl
try:
    import verl
    print(f"VeRL: OK")
except ImportError:
    print("VeRL: NOT INSTALLED (run setup first)")

print("\n" + "=" * 50)
if total_mem >= 28:
    print("RESULT: Should work with default config")
elif total_mem >= 20:
    print("RESULT: Tight - may need to reduce batch_size/n")
    print("  Try: batch_size=4, n=2, max_response_length=1024")
else:
    print("RESULT: Too little VRAM for RL training")
    print("  Recommend: use run_train.sh (SFT) instead")
print("=" * 50)
PYEOF
}

case "${1:-help}" in
    setup)
        setup
        ;;
    test)
        test_gpu
        ;;
    curriculum)
        ITER=${2:-1}
        EXECUTOR=${3:-$BASE_MODEL}
        CURRICULUM=${4:-$BASE_MODEL}
        train_curriculum "$ITER" "$EXECUTOR" "$CURRICULUM"
        ;;
    generate)
        ITER=${2:-1}
        CURRICULUM="${3:-${STORAGE_PATH}/models/curriculum_iter${2}/global_step_5/actor/huggingface}"
        generate_questions "$ITER" "$CURRICULUM"
        ;;
    filter)
        ITER=${2:-1}
        EXECUTOR=${3:-$BASE_MODEL}
        evaluate_questions "$ITER" "$EXECUTOR"
        filter_data "$ITER"
        ;;
    executor)
        ITER=${2:-1}
        TRAIN_DATA="${3:-${STORAGE_PATH}/generated_question/iter${ITER}/train.parquet}"
        EXECUTOR=${4:-$BASE_MODEL}
        train_executor "$ITER" "$TRAIN_DATA" "$EXECUTOR"
        ;;
    eval)
        ITER=${2:-1}
        MODEL="${3:-$BASE_MODEL}"
        eval_aime2025 "$ITER" "$MODEL"
        ;;
    iter)
        ITER=${2:-1}
        EXECUTOR=${3:-$BASE_MODEL}
        CURRICULUM=${4:-$BASE_MODEL}
        run_iteration "$ITER" "$EXECUTOR" "$CURRICULUM"
        ;;
    all)
        run_all
        ;;
    *)
        echo "Agent0 Full Co-Evolution Pipeline (2x T4)"
        echo ""
        echo "Usage: bash $0 <command> [args]"
        echo ""
        echo "Commands:"
        echo "  setup                          Install dependencies"
        echo "  all                            Full pipeline (3 iterations)"
        echo "  iter <N>                       Run iteration N"
        echo "  curriculum <N> [exec] [curr]   Train curriculum agent"
        echo "  generate <N> [curriculum]      Generate questions"
        echo "  filter <N> [executor]          Evaluate & filter questions"
        echo "  executor <N> [data] [model]    Train executor agent"
        echo "  eval <N> [model]               Evaluate on AIME2025"
        ;;
esac
