#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
model_dir="${GPTQ_MODEL_DIR:-$repo_dir/../models-legacy/Qwen3.8-27B-GPTQ-G128}"
image="${VLLM_GPTQ_REFERENCE_IMAGE:-local/b70-qwen38-vllm:q128-m04-196k-180w}"
expected_image_id="sha256:76ddd6049aaf5c0f9a2da9fd62893d7344682d441f03e290205f0ce5104f78c8"
evidence_dir="$repo_dir/B70_GPTQ_INT4_Plan/evidence"
fixture="$evidence_dir/gptq00_python_w4a16_m1_k5120_n1024.safetensors"
log_file="$evidence_dir/gptq00_python_w4a16_m1_k5120_n1024.log"
render_node="${RENDER_NODE:-/dev/dri/renderD128}"

[[ -d "$model_dir" ]] || { echo "Model directory not found: $model_dir" >&2; exit 2; }
[[ -e "$render_node" ]] || { echo "Render node not found: $render_node" >&2; exit 2; }
[[ ! -e "$fixture" && ! -e "${fixture%.safetensors}.json" ]] || {
  echo "Refusing to overwrite fixture: $fixture" >&2
  exit 2
}
[[ ! -e "$log_file" ]] || { echo "Refusing to overwrite log: $log_file" >&2; exit 2; }

actual_image_id="$(docker image inspect --format '{{.Id}}' "$image")"
[[ "$actual_image_id" == "$expected_image_id" ]] || {
  echo "Reference image mismatch: expected $expected_image_id, got $actual_image_id" >&2
  exit 2
}
render_gid="$(stat -c '%g' "$render_node")"
mkdir -p "$evidence_dir"

docker run --rm --pull=never --network none \
  --device /dev/dri --group-add "$render_gid" --shm-size 8g \
  --mount "type=bind,source=$model_dir,target=/model,readonly" \
  --mount "type=bind,source=$repo_dir/tools/gptq_reference,target=/script,readonly" \
  --mount "type=bind,source=$evidence_dir,target=/output" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_TARGET_DEVICE=xpu \
  -e ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE \
  -e ZE_AFFINITY_MASK=0 \
  -e VLLM_XPU_ENABLE_XPU_GRAPH=0 \
  -e DNNL_VERBOSE=1 \
  -e ONEDNN_VERBOSE=1 \
  --entrypoint python3 "$image" \
  /script/export_w4a16_fixture.py \
  --model-dir /model \
  --prefix model.language_model.layers.3.self_attn.k_proj \
  --output "/output/$(basename "$fixture")" \
  --seed 20260924 \
  --image-id "$actual_image_id" 2>&1 | tee "$log_file"
