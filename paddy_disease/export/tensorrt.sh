#!/usr/bin/env bash
set -euo pipefail

ONNX_PATH="${1:-models/onnx/paddy_resnet34.onnx}"
ENGINE_PATH="${2:-models/trt/paddy_resnet34.plan}"
MAX_BATCH="${3:-8}"
IMG="${4:-224}"
WORKSPACE_MB="${5:-2048}"
FP16="${6:-true}"

mkdir -p "$(dirname "$ENGINE_PATH")"

CMD=(trtexec
  --onnx="$ONNX_PATH"
  --saveEngine="$ENGINE_PATH"
  --explicitBatch
  --minShapes=input:1x3x${IMG}x${IMG}
  --optShapes=input:${MAX_BATCH}x3x${IMG}x${IMG}
  --maxShapes=input:${MAX_BATCH}x3x${IMG}x${IMG}
  --workspace=${WORKSPACE_MB}
)

if [[ "$FP16" == "true" ]]; then
  CMD+=(--fp16)
fi

"${CMD[@]}"
echo "TensorRT engine saved to: $ENGINE_PATH"
