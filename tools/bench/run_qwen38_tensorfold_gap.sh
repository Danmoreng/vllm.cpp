#!/usr/bin/env bash
# Pinned Qwen3.8 TensorFold/vllm.cpp launcher and serial evidence capture.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
HARNESS=${QWEN38_HARNESS:-$ROOT/tools/bench/qwen38_endpoint_bench.py}
CORPUS="$ROOT/benchmarks/manifests/qwen38_tensorfold/corpus.json"
CLOCK_SAMPLER=${QWEN38_CLOCK_SAMPLER:-$ROOT/tools/bench/gpu_clock_state.py}
MEMINFO=${QWEN38_MEMINFO_PATH:-/proc/meminfo}
RC_COMMAND=${QWEN38_RC_COMMAND:-rc}
GPU_STATE_COMMAND=${QWEN38_GPU_STATE_COMMAND:-nvidia-smi --query-gpu=index,name,uuid,driver_version,temperature.gpu,power.draw,power.limit,memory.total,memory.free --format=csv,noheader,nounits}
CONFIG_VARS=(ENGINE SOURCE_DIR EXPECTED_REVISION RECIPE_SOURCE_DIR EXPECTED_RECIPE_REVISION ARTIFACT_MANIFEST ENDPOINT ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY LAUNCH_COMMAND START_COMMAND STOP_COMMAND STATUS_COMMAND SERVER_LOG_PATH REDACTION_COMMAND MIN_FREE_MEMORY_KIB DRAFT CONCURRENCY WAVES)
fail() { echo "qwen38-tensorfold-gap: $*" >&2; exit 2; }
require_var() { test -n "${!1:-}" || fail "$1 is required"; }

reset_config() { local v; for v in "${CONFIG_VARS[@]}"; do unset "$v"; done; }
load_env() {
  local path=$1 name allowed v
  test -f "$path" || fail "environment file does not exist: $path"
  while IFS= read -r name; do
    allowed=0
    for v in "${CONFIG_VARS[@]}"; do test "$name" = "$v" && allowed=1; done
    test "$allowed" = 1 || fail "unsupported or suspicious field in environment file: $name"
  done < <(sed -nE 's/^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=.*/\2/p' "$path")
  reset_config
  set -a
  # shellcheck disable=SC1090
  source "$path"
  set +a
}
check_revision() {
  local d=$1 expected=$2 label=$3 actual
  git -C "$d" rev-parse --git-dir >/dev/null 2>&1 || fail "$label source is not a git checkout: $d"
  [[ "$expected" =~ ^[0-9a-f]{40}$ ]] || fail "$label expected revision must be full 40-hex"
  actual=$(git -C "$d" rev-parse HEAD); test "$actual" = "$expected" || fail "$label revision mismatch"
  test -z "$(git -C "$d" status --porcelain=v1 --untracked-files=normal)" || fail "$label revision is dirty"
}
check_artifacts() {
  local manifest=$1 base line digest relative actual count=0
  test -s "$manifest" || fail "artifact hash manifest is missing or empty: $manifest"
  base=$(cd "$(dirname "$manifest")" && pwd)
  while IFS= read -r line || test -n "$line"; do
    test -z "$line" && continue
    [[ "$line" =~ ^([0-9a-f]{64})[[:space:]][[:space:]]([^/].*)$ ]] || fail "artifact manifest entry lacks a measured sha256: $line"
    digest=${BASH_REMATCH[1]}; relative=${BASH_REMATCH[2]}
    [[ "$relative" != *".."* ]] || fail "artifact path escapes manifest directory"
    test -f "$base/$relative" || fail "artifact is missing: $relative"
    actual=$(sha256sum "$base/$relative" | cut -d' ' -f1); test "$actual" = "$digest" || fail "artifact hash mismatch: $relative"
    count=$((count+1))
  done < "$manifest"
  test "$count" -gt 0 || fail "artifact manifest contains no hashes"
}
artifact_identity() {
  local manifest=$1 base
  base=$(cd "$(dirname "$manifest")" && pwd)
  while read -r digest relative; do printf '%s  %s\n' "$digest" "$relative"; done < "$manifest" | LC_ALL=C sort | sha256sum | cut -d' ' -f1
}
verify_lease() {
  require_var RC_DEVICE; require_var RC_JOB_ID
  test "$RC_DEVICE" = dgx:gpu0 || fail "campaign requires RC_DEVICE=dgx:gpu0"
  command -v "$RC_COMMAND" >/dev/null || fail "repository rc helper is unavailable"
  "$RC_COMMAND" jobs --device dgx:gpu0 --limit 100 --output json | python3 -c '
import json,os,sys
try: x=json.load(sys.stdin)
except Exception: raise SystemExit(2)
rows=x if isinstance(x,list) else x.get("jobs",x.get("items",[]))
j=os.environ["RC_JOB_ID"]
ok=any(str(r.get("id",r.get("job_id","")))==j and r.get("device",r.get("device_id"))=="dgx:gpu0" and str(r.get("state",r.get("status",""))).lower() in {"running","held","active"} for r in rows)
raise SystemExit(0 if ok else 2)' || fail "RC_JOB_ID is not an active dgx:gpu0 repository lease"
}
mem_available() { awk '/^MemAvailable:/{print $2; exit}' "$MEMINFO"; }
probe_ready() { if test -n "${QWEN38_ENDPOINT_PROBE:-}"; then "$QWEN38_ENDPOINT_PROBE" "$ENDPOINT_READY_URL" >/dev/null 2>&1; else curl --fail --silent --max-time 5 "$ENDPOINT_READY_URL" >/dev/null; fi; }
common_gate() {
  local env_file=$1 policy=$2 available
  load_env "$env_file"
  for v in ENGINE SOURCE_DIR EXPECTED_REVISION ARTIFACT_MANIFEST ENDPOINT ENDPOINT_READY_URL MODEL TOKENIZER_IDENTITY MIN_FREE_MEMORY_KIB; do require_var "$v"; done
  case "$ENGINE" in tensorfold|vllm-cpp) ;; *) fail "invalid ENGINE";; esac
  check_revision "$SOURCE_DIR" "$EXPECTED_REVISION" "$ENGINE"
  if test -n "${RECIPE_SOURCE_DIR:-}${EXPECTED_RECIPE_REVISION:-}"; then require_var RECIPE_SOURCE_DIR; require_var EXPECTED_RECIPE_REVISION; check_revision "$RECIPE_SOURCE_DIR" "$EXPECTED_RECIPE_REVISION" recipe; fi
  check_artifacts "$ARTIFACT_MANIFEST"; verify_lease
  test -r "$MEMINFO" || fail "memory sampler unavailable"
  available=$(mem_available); [[ "$available" =~ ^[0-9]+$ && "$MIN_FREE_MEMORY_KIB" =~ ^[0-9]+$ ]] || fail "invalid free-memory gate"
  test "$available" -ge "$MIN_FREE_MEMORY_KIB" || fail "free unified memory below baseline"
  test -r "$CLOCK_SAMPLER" || fail "clock sampler unavailable"; python3 -m py_compile "$CLOCK_SAMPLER"
  case "$policy" in down) ! probe_ready || fail "endpoint already active before launch";; ready) probe_ready || fail "endpoint is not ready";; either) :;; esac
}
check_one() { common_gate "$1" either; echo "qwen38-tensorfold-gap: check PASS ($ENGINE)"; }
launch_one() { local wanted=$1 file=$2; common_gate "$file" down; test "$ENGINE" = "$wanted" || fail "wrong engine"; require_var LAUNCH_COMMAND; exec bash -c "$LAUNCH_COMMAND"; }
run_harness() { python3 "$HARNESS" --endpoint "$ENDPOINT" --model "$MODEL" --corpus "$CORPUS" --output "$1" --tokenizer-identity "$TOKENIZER_IDENTITY" --adapter "$ENGINE" --draft "${DRAFT:-off}" --concurrency "${CONCURRENCY:-1}" --waves "${WAVES:-5}"; }
write_provenance() {
  local output=$1 identity; identity=$(artifact_identity "$ARTIFACT_MANIFEST")
  python3 - "$output" "$ENGINE" "$EXPECTED_REVISION" "${EXPECTED_RECIPE_REVISION:-}" "$identity" "$ENDPOINT" "$MODEL" "$TOKENIZER_IDENTITY" "$ARTIFACT_MANIFEST" "$SOURCE_DIR" "$SERVER_LOG_PATH" <<'PY'
import json,os,pathlib,sys
out,engine,rev,recipe,artifact,endpoint,model,tok,manifest,source,log=sys.argv[1:]
entries=[]; base=pathlib.Path(manifest).parent
for line in pathlib.Path(manifest).read_text().splitlines():
 d,p=line.split(None,1); p=p.strip(); q=base/p; s=q.stat(); entries.append({"path":p,"sha256":d,"size":s.st_size,"mtime_ns":s.st_mtime_ns})
data={"engine":engine,"revision":rev,"recipe_revision":recipe or None,"artifact_identity":artifact,"artifacts":sorted(entries,key=lambda x:x["path"]),"endpoint":endpoint,"model":model,"tokenizer_identity":tok,"source_dirty":False,"runtime":{"rc_device":os.environ["RC_DEVICE"],"rc_job_id":os.environ["RC_JOB_ID"]},"server_log":"server.log"}
pathlib.Path(out).write_text(json.dumps(data,indent=2,sort_keys=True)+"\n")
PY
}
capture_arm() (
  local file=$1 out=$2 before after sampler="" started=0
  # Invoked indirectly by trap.
  # shellcheck disable=SC2329
  cleanup_arm() {
    local status=$?
    if test -n "$sampler"; then kill -INT "$sampler" 2>/dev/null || true; wait "$sampler" 2>/dev/null || true; fi
    if test "$started" = 1; then bash -c "$STOP_COMMAND" >>"$out/server-stop.log" 2>&1 || status=2; fi
    if bash -c "$STATUS_COMMAND" >/dev/null 2>&1; then echo "endpoint remains active after teardown" >&2; status=2; fi
    exit "$status"
  }
  common_gate "$file" either
  for v in START_COMMAND STOP_COMMAND STATUS_COMMAND SERVER_LOG_PATH REDACTION_COMMAND; do require_var "$v"; done
  before=$(mem_available); printf '%s\n' "$before" >"$out/memory-before-kib.txt"
  if ! bash -c "$STATUS_COMMAND" >/dev/null 2>&1; then
    started=1; trap cleanup_arm EXIT INT TERM
    bash -c "$START_COMMAND" >"$out/server-start.log" 2>&1
  else
    # Capture must own teardown even when an operator declares an existing arm.
    started=1; trap cleanup_arm EXIT INT TERM
  fi
  bash -c "$STATUS_COMMAND" >/dev/null 2>&1 || fail "$ENGINE failed to become ready"; probe_ready || fail "$ENGINE endpoint not ready"
  bash -c "$GPU_STATE_COMMAND" >"$out/gpu-state-before.csv" || fail "GPU runtime identity probe failed"
  python3 "$CLOCK_SAMPLER" sample --output "$out/clocks.jsonl" --summary "$out/clocks-summary.json" --interval 1 --max-duration 3600 & sampler=$!
  run_harness "$out/result.json"
  kill -INT "$sampler" 2>/dev/null || true; wait "$sampler" || fail "clock sampler refused window"; sampler=""
  python3 - "$out/clocks-summary.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1])); n=x.get("sm_clock_mhz",{}).get("n",0)
raise SystemExit(0 if n>=30 else 2)
PY
  bash -c "$GPU_STATE_COMMAND" >"$out/gpu-state-after.csv" || fail "GPU power/thermal probe failed"
  bash -c "$REDACTION_COMMAND < \"$SERVER_LOG_PATH\" > \"$out/server.log\"" || fail "server-log redaction failed"
  write_provenance "$out/provenance.json"
  bash -c "$STOP_COMMAND" >"$out/server-stop.log" 2>&1; started=0
  ! bash -c "$STATUS_COMMAND" >/dev/null 2>&1 || fail "$ENGINE teardown failed"
  after=$(mem_available); printf '%s\n' "$after" >"$out/memory-after-kib.txt"
  test "$after" -ge "$MIN_FREE_MEMORY_KIB" || fail "free unified memory did not return to baseline"
  trap - EXIT INT TERM
)
compare_results() {
  python3 - "$ROOT" "$1" "$2" "$3" "$4" "$5" <<'PY'
import json,pathlib,sys
sys.path.insert(0,sys.argv[1]); from tools.bench.qwen38_endpoint_bench import comparison_verdict
out=comparison_verdict(json.loads(pathlib.Path(sys.argv[2]).read_text()),json.loads(pathlib.Path(sys.argv[3]).read_text()))
out.update(left_artifact=sys.argv[4],right_artifact=sys.argv[5]); out.pop("ratio",None)
if sys.argv[4]!=sys.argv[5] and out["verdict"]!="REFUSED": out.update(verdict="PROFILE_COMPARISON",no_ratio_reason="verified artifact manifests differ; cross-engine results are absolute only")
pathlib.Path(sys.argv[6]).write_text(json.dumps(out,indent=2,sort_keys=True)+"\n"); raise SystemExit(2 if out["verdict"]=="REFUSED" else 0)
PY
}
capture() {
  local tf=$1 cpp=$2 base=${3:-$ROOT/.agents/evidence/bench-qwen38-tensorfold-gap} stamp out ta ca
  stamp=$(date -u +%Y%m%dT%H%M%SZ); out="$base/$stamp"; test ! -e "$out" || fail "evidence exists"; mkdir -p "$out/tensorfold" "$out/vllm-cpp"
  capture_arm "$tf" "$out/tensorfold"; load_env "$tf"; ta=$(artifact_identity "$ARTIFACT_MANIFEST")
  capture_arm "$cpp" "$out/vllm-cpp"; load_env "$cpp"; ca=$(artifact_identity "$ARTIFACT_MANIFEST")
  compare_results "$out/tensorfold/result.json" "$out/vllm-cpp/result.json" "$ta" "$ca" "$out/comparison.json"; printf '%s\n' "$out"
}
usage(){ echo "usage: $0 {check|tensorfold|vllm-cpp|capture} ..."; }
command=${1:-}; shift || true
case "$command" in
 check) test $# = 1 || fail "check requires ENV_FILE"; check_one "$1";;
 tensorfold|vllm-cpp) test $# = 1 || fail "$command requires ENV_FILE"; launch_one "$command" "$1";;
 capture) test $# -ge 2 -a $# -le 3 || fail "capture requires two env files"; capture "$@";;
 compare-results) test $# = 5 || fail "compare-results requires five arguments"; compare_results "$@";;
 *) usage >&2; exit 2;;
esac
