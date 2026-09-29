#!/usr/bin/env bash
# Pinned Qwen3.8 TensorFold/vllm.cpp endpoint launcher and evidence capture.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HARNESS="$ROOT/tools/bench/qwen38_endpoint_bench.py"
CORPUS="$ROOT/benchmarks/manifests/qwen38_tensorfold/corpus.json"
CLOCK_SAMPLER=${QWEN38_CLOCK_SAMPLER:-$ROOT/tools/bench/gpu_clock_state.py}
MEMINFO=${QWEN38_MEMINFO_PATH:-/proc/meminfo}

fail() { echo "qwen38-tensorfold-gap: $*" >&2; exit 2; }
require_var() { test -n "${!1:-}" || fail "$1 is required"; }

load_env() {
  local path=$1
  test -f "$path" || fail "environment file does not exist: $path"
  # Environment files are operator-owned shell configuration so launch commands
  # can retain quoting. Repository examples contain no credentials.
  set -a
  # shellcheck disable=SC1090
  source "$path"
  set +a
}

check_revision() {
  local directory=$1 expected=$2 label=$3 actual
  test -d "$directory/.git" || git -C "$directory" rev-parse --git-dir >/dev/null 2>&1 ||
    fail "$label source is not a git checkout: $directory"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail "$label expected revision must be a full 40-hex object ID"
  actual=$(git -C "$directory" rev-parse HEAD) || fail "cannot read $label revision"
  test "$actual" = "$expected" || fail "$label revision mismatch: expected $expected, got $actual"
  test -z "$(git -C "$directory" status --porcelain=v1 --untracked-files=normal)" ||
    fail "$label revision is dirty"
}

check_artifacts() {
  local manifest=$1 base line digest relative actual count=0
  test -s "$manifest" || fail "artifact hash manifest is missing or empty: $manifest"
  base=$(cd "$(dirname "$manifest")" && pwd)
  while IFS= read -r line || test -n "$line"; do
    test -z "$line" && continue
    [[ "$line" =~ ^([0-9a-f]{64})[[:space:]][[:space:]]([^/].*)$ ]] ||
      fail "artifact manifest entry lacks a measured sha256: $line"
    digest=${BASH_REMATCH[1]}; relative=${BASH_REMATCH[2]}
    [[ "$relative" != *".."* ]] || fail "artifact path must remain under its manifest directory"
    test -f "$base/$relative" || fail "artifact is missing: $base/$relative"
    actual=$(sha256sum "$base/$relative" | cut -d' ' -f1)
    test "$actual" = "$digest" || fail "artifact hash mismatch: $relative"
    count=$((count + 1))
  done < "$manifest"
  test "$count" -gt 0 || fail "artifact hash manifest contains no hashes"
}

probe_endpoint() {
  if test -n "${QWEN38_ENDPOINT_PROBE:-}"; then
    "$QWEN38_ENDPOINT_PROBE" "$ENDPOINT_READY_URL" >/dev/null 2>&1 || fail "endpoint is not ready"
  else
    command -v curl >/dev/null || fail "curl is required for endpoint readiness"
    curl --fail --silent --show-error --max-time 5 "$ENDPOINT_READY_URL" >/dev/null ||
      fail "endpoint is not ready: $ENDPOINT_READY_URL"
  fi
}

check_one() {
  local env_file=$1 available
  unset ENGINE SOURCE_DIR EXPECTED_REVISION RECIPE_SOURCE_DIR EXPECTED_RECIPE_REVISION
  unset ARTIFACT_MANIFEST ARTIFACT_ID ENDPOINT ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY
  unset LAUNCH_COMMAND MIN_FREE_MEMORY_KIB DRAFT CONCURRENCY WAVES
  load_env "$env_file"
  for variable in ENGINE SOURCE_DIR EXPECTED_REVISION ARTIFACT_MANIFEST ENDPOINT \
                  ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY LAUNCH_COMMAND MIN_FREE_MEMORY_KIB; do
    require_var "$variable"
  done
  case "$ENGINE" in tensorfold|vllm-cpp) ;; *) fail "ENGINE must be tensorfold or vllm-cpp" ;; esac
  check_revision "$SOURCE_DIR" "$EXPECTED_REVISION" "$ENGINE"
  if test -n "${RECIPE_SOURCE_DIR:-}${EXPECTED_RECIPE_REVISION:-}"; then
    require_var RECIPE_SOURCE_DIR; require_var EXPECTED_RECIPE_REVISION
    check_revision "$RECIPE_SOURCE_DIR" "$EXPECTED_RECIPE_REVISION" recipe
  fi
  check_artifacts "$ARTIFACT_MANIFEST"
  for variable in RC_DEVICE RC_JOB_ID RC_TOKEN; do require_var "$variable"; done
  test -r "$MEMINFO" || fail "memory sampler is unavailable: $MEMINFO"
  available=$(awk '/^MemAvailable:/{print $2; exit}' "$MEMINFO")
  [[ "${available:-}" =~ ^[0-9]+$ ]] || fail "MemAvailable is missing from $MEMINFO"
  [[ "$MIN_FREE_MEMORY_KIB" =~ ^[0-9]+$ ]] || fail "MIN_FREE_MEMORY_KIB must be an integer"
  test "$available" -ge "$MIN_FREE_MEMORY_KIB" ||
    fail "free memory ${available} KiB is below required ${MIN_FREE_MEMORY_KIB} KiB"
  test -r "$CLOCK_SAMPLER" || fail "clock sampler is unavailable: $CLOCK_SAMPLER"
  python3 -m py_compile "$CLOCK_SAMPLER" || fail "clock sampler cannot be loaded"
  probe_endpoint
  echo "qwen38-tensorfold-gap: check PASS ($ENGINE)"
}

launch_one() {
  local wanted=$1 env_file=$2
  load_env "$env_file"
  test "${ENGINE:-}" = "$wanted" || fail "$env_file configures ${ENGINE:-no engine}, expected $wanted"
  require_var LAUNCH_COMMAND
  exec bash -c "$LAUNCH_COMMAND"
}

run_harness() {
  local env_file=$1 output=$2 adapter
  load_env "$env_file"
  adapter=$ENGINE
  python3 "$HARNESS" --endpoint "$ENDPOINT" --model "$MODEL" --corpus "$CORPUS" \
    --output "$output" --tokenizer-identity "$TOKENIZER_IDENTITY" --adapter "$adapter" \
    --draft "${DRAFT:-off}" --concurrency "${CONCURRENCY:-1}" --waves "${WAVES:-5}"
}

compare_results() {
  local left=$1 right=$2 left_artifact=$3 right_artifact=$4 output=$5
  python3 - "$ROOT" "$left" "$right" "$left_artifact" "$right_artifact" "$output" <<'PY'
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from tools.bench.qwen38_endpoint_bench import comparison_verdict
left = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
right = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
out = comparison_verdict(left, right)
out["left_artifact"] = sys.argv[4]
out["right_artifact"] = sys.argv[5]
if sys.argv[4] != sys.argv[5] and out["verdict"] != "REFUSED":
    out["verdict"] = "PROFILE_COMPARISON"
    out["no_ratio_reason"] = "artifacts differ; cross-engine results are absolute only"
out.pop("ratio", None)
pathlib.Path(sys.argv[6]).write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(out, sort_keys=True))
raise SystemExit(2 if out["verdict"] == "REFUSED" else 0)
PY
}

capture() {
  local tf_env=$1 cpp_env=$2 base=${3:-$ROOT/.agents/evidence/bench-qwen38-tensorfold-gap}
  local stamp out tf_artifact cpp_artifact
  check_one "$tf_env"; check_one "$cpp_env"
  stamp=$(date -u +%Y%m%dT%H%M%SZ)
  out="$base/$stamp"
  test ! -e "$out" || fail "refusing to overwrite evidence directory: $out"
  mkdir -p "$out"
  cp "$tf_env" "$out/tensorfold.env"; cp "$cpp_env" "$out/vllm_cpp.env"
  run_harness "$tf_env" "$out/tensorfold.json"
  run_harness "$cpp_env" "$out/vllm_cpp.json"
  load_env "$tf_env"; tf_artifact=${ARTIFACT_ID:-$(sha256sum "$ARTIFACT_MANIFEST" | cut -d' ' -f1)}
  load_env "$cpp_env"; cpp_artifact=${ARTIFACT_ID:-$(sha256sum "$ARTIFACT_MANIFEST" | cut -d' ' -f1)}
  compare_results "$out/tensorfold.json" "$out/vllm_cpp.json" "$tf_artifact" "$cpp_artifact" "$out/comparison.json"
  printf '%s\n' "$out"
}

usage() {
  cat <<EOF
usage:
  $0 check ENV_FILE
  $0 tensorfold ENV_FILE
  $0 vllm-cpp ENV_FILE
  $0 capture TENSORFOLD_ENV VLLM_CPP_ENV [EVIDENCE_ROOT]
EOF
}

command=${1:-}; shift || true
case "$command" in
  check) test "$#" -eq 1 || fail "check requires ENV_FILE"; check_one "$1" ;;
  tensorfold) test "$#" -eq 1 || fail "tensorfold requires ENV_FILE"; launch_one tensorfold "$1" ;;
  vllm-cpp) test "$#" -eq 1 || fail "vllm-cpp requires ENV_FILE"; launch_one vllm-cpp "$1" ;;
  capture) test "$#" -ge 2 -a "$#" -le 3 || fail "capture requires two env files and optional output root"; capture "$@" ;;
  compare-results) test "$#" -eq 5 || fail "compare-results requires LEFT RIGHT LEFT_ARTIFACT RIGHT_ARTIFACT OUTPUT"; compare_results "$@" ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
