#!/usr/bin/env bash
set -euo pipefail
MODO="${1:-prueba}"
MODELO="salamandra-7b-instruct"
SALIDA="docs/experimento/llm"

command -v ollama >/dev/null || { echo "Falta instalar Ollama (https://ollama.com)"; exit 1; }
ollama list | grep -q "^${MODELO}" || ollama create "${MODELO}" -f scripts/experimento/ollama/Modelfile
pip install -q requests pyyaml pandas scikit-learn >/dev/null

python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" --mostrar-prompt --limite 3 --out docs/experimento/prueba_prompt || true

if [ "${MODO}" = "prueba" ]; then
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" \
      --semilla 42 --variantes base --limite 50 --out "${SALIDA}_prueba"
  python scripts/experimento/monoagente_salamandra.py resumir --out "${SALIDA}_prueba"
else
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" \
      --semilla 42 --variantes base limpio --out "${SALIDA}"
  python scripts/experimento/monoagente_salamandra.py resumir --out "${SALIDA}"
fi
