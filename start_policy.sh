#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TOKENIZERS_PARALLELISM=false
python_bin="${OCEANVLA_PYTHON:-python}"
exec "$python_bin" -u -m core.pipeline \
  --config "${OCEANVLA_CONFIG:-configs/train1.yaml}" \
  --episodes "${OCEANVLA_EPISODES:-data/episodes}" \
  --captions "${OCEANVLA_CAPTIONS:-data/captions}" \
  --data-root "${OCEANVLA_DATA_ROOT:-data/raw}" \
  --caption-root "${OCEANVLA_CAPTION_ROOT:-data/caption_images}" \
  --output "${OCEANVLA_OUTPUT:-runs/oceanvla}" "$@"
