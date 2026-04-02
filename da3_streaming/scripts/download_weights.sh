#!/bin/bash
set -euo pipefail

# Resolve directories
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${ROOT_DIR}/weights"
CACHE_DIR="${XDG_CACHE_HOME:-${HOME}/.cache}/da3_streaming"

mkdir -p "${CACHE_DIR}"

# Point the expected weights folder to the shared cache
if [ -e "${TARGET_DIR}" ] && [ ! -L "${TARGET_DIR}" ]; then
	rm -rf "${TARGET_DIR}"
fi
ln -sfn "${CACHE_DIR}" "${TARGET_DIR}"

cd "${CACHE_DIR}"

fetch() {
	local url="$1" dst="$2"
	if [ -f "${dst}" ]; then
		echo "[skip] ${dst} already exists"
		return 0
	fi
	echo "[download] ${dst}"
	curl -L -C - "${url}" -o "${dst}"
}

# SALAD (~ 340 MiB)
echo "Downloading SALAD weights (~ 340 MiB) ..."
SALAD_URL="https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt"
fetch "$SALAD_URL" "dino_salad.ckpt"


# DA3NESTED-GIANT-LARGE-1.1
echo "Downloading DA3NESTED-GIANT-LARGE-1.1 weights and config (~ 6.76 GiB)..."
BASE_URL="https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main"

# download config.json (~ 3.1 KiB)
fetch "$BASE_URL/config.json" "config.json"

# download model.safetensors (~ 6.76 GiB)
fetch "$BASE_URL/model.safetensors" "model.safetensors"
