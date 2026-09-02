#!/usr/bin/env bash
#
# update_qwen35.sh — copy freshly built Qwen3.5 / Qwen3.6-35B-A3B artifacts
#                    (xclbins + shared libs) from the FastFlowLM_IRON build
#                    tree into this repo.
#
# Usage:
#   ./update_qwen35.sh                 # copy the standard set (0.8B/2B/4B/9B + 35B-A3B)
#   ./update_qwen35.sh -n              # dry run: show what would change, copy nothing
#   ./update_qwen35.sh --gdn           # also refresh GateDeltaNet_prefill.xclbin
#   ./update_qwen35.sh --exe ~/flm_exe # additionally mirror into an installed flm_exe tree
#
# Sources can be overridden via env:
#   FLM_XCL=/path/to/FLM_Xclbin  FLM_DLL=/path/to/FLM_DLL  ./update_qwen35.sh
#
set -uo pipefail

IRON="${IRON:-/scratch/alfxu/FastFlowLM_IRON}"
FLM_XCL="${FLM_XCL:-$IRON/FLM_Xclbin}"
FLM_DLL="${FLM_DLL:-$IRON/FLM_DLL}"

# This script lives in the repo's src/ dir.
REPO_SRC="${REPO_SRC:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
XCL_DST="$REPO_SRC/xclbins"
LIB_DST="$REPO_SRC/lib/xrt"

DRY=0; DO_GDN=0; EXE_DIR=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--dry-run) DRY=1 ;;
    --gdn)        DO_GDN=1 ;;
    --exe)        EXE_DIR="${2:?--exe needs a path}"; shift ;;
    -h|--help)    sed -n '2,17p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *)            echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

n_copied=0; n_same=0; n_missing=0
MISSING_SRCS=()

# cp_art <src> <dst>
cp_art() {
  local src="$1" dst="$2"
  if [[ ! -f "$src" ]]; then
    printf '  MISSING  %s\n' "${dst#$REPO_SRC/}"
    MISSING_SRCS+=("${dst#$REPO_SRC/}  <-  $src")
    ((n_missing++)); return 1
  fi
  if [[ -f "$dst" ]] && cmp -s "$src" "$dst"; then
    printf '  same     %s\n' "${dst#$REPO_SRC/}"
    ((n_same++)); return 0
  fi
  if (( DRY )); then
    printf '  WOULD CP %s\n' "${dst#$REPO_SRC/}"
  else
    mkdir -p "$(dirname "$dst")"
    cp -f "$src" "$dst" || { echo "  FAILED   $dst" >&2; return 1; }
    printf '  updated  %s\n' "${dst#$REPO_SRC/}"
  fi
  ((n_copied++))
}

DECODE_35="$FLM_XCL/Qwen3_5/qwen3_5_decoding/build"
DECODE_36="$FLM_XCL/Qwen3_6/qwen3_6_moe_decoding/build"
LMHEAD="$FLM_XCL/Qwen3_5/lm_head_npu_bin/build"
MM512="$FLM_XCL/dequant_mm_512x512x512/build/xclbins/dequant_mm.xclbin"
MM256="$FLM_XCL/dequant_mm_256x512x512/build/xclbins/dequant_mm.xclbin"
GDN="$FLM_XCL/Qwen3_5/gate_delta_net_prefill/hostDeltaNetPrefill/build/xclbins/GateDeltaNet_prefill.xclbin"

# model dir  |  decoding MODEL_TYPE  |  lm_head MODEL_TYPE
# note: the 0.8B decoding target is QWEN3_5_08B but its lm_head target is QWEN3_5_0_8B
MODELS_35=(
  "Qwen3.5-0.8B-NPU2:QWEN3_5_08B:QWEN3_5_0_8B"
  "Qwen3.5-2B-NPU2:QWEN3_5_2B:QWEN3_5_2B"
  "Qwen3.5-4B-NPU2:QWEN3_5_4B:QWEN3_5_4B"
  "Qwen3.5-9B-NPU2:QWEN3_5_9B:QWEN3_5_9B"
)
DIR_36="Qwen3.6-35B-A3B-NPU2"

echo "src xclbins : $FLM_XCL"
echo "src libs    : $FLM_DLL/build/lib"
echo "dst repo    : $REPO_SRC"
(( DRY )) && echo "*** DRY RUN — nothing will be written ***"
echo

for entry in "${MODELS_35[@]}"; do
  IFS=: read -r dir dec lmh <<<"$entry"
  echo "[$dir]"
  cp_art "$DECODE_35/$dec/xclbins/layer.xclbin"  "$XCL_DST/$dir/layer.xclbin"
  cp_art "$LMHEAD/$lmh/xclbins/lm_head.xclbin"   "$XCL_DST/$dir/lm_head.xclbin"
  cp_art "$MM512"                                "$XCL_DST/$dir/mm.xclbin"
  echo
done

# Qwen3.6 MoE: own decoding target, borrows the Qwen3.5-2B lm_head,
# and unlike the 3.5 models its dequant_mm is the 256x512x512 bitstream.
echo "[$DIR_36]"
cp_art "$DECODE_36/QWEN3_6_35B_A3B/xclbins/layer.xclbin" "$XCL_DST/$DIR_36/layer.xclbin"
cp_art "$LMHEAD/QWEN3_5_2B/xclbins/lm_head.xclbin"       "$XCL_DST/$DIR_36/lm_head.xclbin"
cp_art "$MM512"                                          "$XCL_DST/$DIR_36/mm.xclbin"
cp_art "$MM256"                                          "$XCL_DST/$DIR_36/dequant_mm.xclbin"
(( DO_GDN )) && cp_art "$GDN" "$XCL_DST/$DIR_36/GateDeltaNet_prefill.xclbin"
echo

echo "[shared libs]"
cp_art "$FLM_DLL/build/lib/libqwen3_5vl_npu.so"   "$LIB_DST/libqwen3_5vl_npu.so"
cp_art "$FLM_DLL/build/lib/libqwen3_6_moe_npu.so" "$LIB_DST/libqwen3_6_moe_npu.so"
echo

# Optionally mirror the same files into an installed runtime tree
if [[ -n "$EXE_DIR" ]]; then
  echo "[mirror -> $EXE_DIR]"
  for entry in "${MODELS_35[@]}" "$DIR_36::"; do
    dir="${entry%%:*}"
    if [[ -d "$EXE_DIR/share/flm/xclbins/$dir" ]]; then
      for f in "$XCL_DST/$dir"/*.xclbin; do
        if (( DRY )); then printf '  WOULD CP %s\n' "$dir/$(basename "$f")"
        else cp -f "$f" "$EXE_DIR/share/flm/xclbins/$dir/" && printf '  synced   %s\n' "$dir/$(basename "$f")"; fi
      done
    fi
  done
  if [[ -d "$EXE_DIR/lib" ]]; then
    for so in libqwen3_5vl_npu.so libqwen3_6_moe_npu.so; do
      if (( DRY )); then printf '  WOULD CP %s\n' "$so"
      else cp -f "$LIB_DST/$so" "$EXE_DIR/lib/" && printf '  synced   %s\n' "$so"; fi
    done
  fi
  echo
fi

echo "updated: $n_copied   unchanged: $n_same   missing sources: $n_missing"
if (( n_missing > 0 )); then
  echo
  echo "Source files that do not exist (target left untouched):"
  printf '  %s\n' "${MISSING_SRCS[@]}"
fi
exit 0
