#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${GPTQ_MODEL_DIR:-$repo_dir/../models-legacy/Qwen3.8-27B-GPTQ-G128}"
image="${VLLM_GPTQ_REFERENCE_IMAGE:-local/b70-qwen38-vllm:q128-m04-196k-180w}"
expected_image_id="sha256:76ddd6049aaf5c0f9a2da9fd62893d7344682d441f03e290205f0ce5104f78c8"
host="${REFERENCE_HOST:-127.0.0.1}"
port="${REFERENCE_PORT:-8082}"
render_node="${RENDER_NODE:-/dev/dri/renderD128}"
expected_model="mikeinnyc/Qwen3.8-27B-GPTQ-Int4-sym-G128-MTP-BF16"
expected_revision="a47b0c6f0d756bc394c4cc629d5b0ded1acc7001"

[[ -d "$model_dir" ]] || { echo "Model directory not found: $model_dir" >&2; exit 2; }
[[ -e "$render_node" ]] || { echo "Render node not found: $render_node" >&2; exit 2; }
mapfile -t source_pin < "$model_dir/SOURCE-REVISION.txt"
[[ "${source_pin[0]:-}" == "$expected_model" && "${source_pin[1]:-}" == "$expected_revision" ]] || {
  echo "Checkpoint source pin does not match the GPTQ reference" >&2
  exit 2
}

actual_image_id="$(docker image inspect --format '{{.Id}}' "$image")"
[[ "$actual_image_id" == "$expected_image_id" ]] || {
  echo "Reference image mismatch: expected $expected_image_id, got $actual_image_id" >&2
  exit 2
}
render_gid="$(stat -c '%g' "$render_node")"

exec docker run --rm --pull=never --name b70-gptq-no-mtp-reference \
  --device /dev/dri --group-add "$render_gid" --shm-size 8g \
  --publish "$host:$port:8000" \
  --mount "type=bind,source=$model_dir,target=/model,readonly" \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e B70_GPTQ_W4A8_PREFILL=0 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  --entrypoint vllm "$image" serve /model \
  --quantization gptq \
  --dtype float16 \
  --max-model-len "${REFERENCE_MAX_MODEL_LEN:-4096}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.93}" \
  --kv-cache-dtype auto \
  --max-num-seqs 1 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-512}" \
  --no-enable-prefix-caching \
  --enforce-eager \
  --served-model-name B70-GPTQ-INT4-Reference
