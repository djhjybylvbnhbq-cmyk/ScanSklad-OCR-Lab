#!/usr/bin/env bash
set -euo pipefail
mkdir -p app/src/main/assets/rec
ROOT='https://huggingface.co/PaddlePaddle/cyrillic_PP-OCRv5_mobile_rec_onnx/resolve/main'
curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 30 --max-time 300 --output app/src/main/assets/rec/inference.onnx "$ROOT/inference.onnx?download=true"
curl --fail --location --retry 5 --retry-delay 3 --connect-timeout 30 --max-time 120 --output app/src/main/assets/rec/inference.yml "$ROOT/inference.yml?download=true"
python3 - <<'PY'
from pathlib import Path
b=Path('app/src/main/assets/rec/inference.onnx')
y=Path('app/src/main/assets/rec/inference.yml')
if b.stat().st_size < 1_000_000 or y.stat().st_size < 3000:
  raise SystemExit('Bad downloaded model files: sizes '+str((b.stat().st_size,y.stat().st_size)))
if b.read_bytes()[:30].lower().startswith(b'<!doctype'):
  raise SystemExit('ONNX download was HTML, not a model')
print('Model ready:', b.stat().st_size, 'bytes. Dictionary:', y.stat().st_size, 'bytes.')
PY
