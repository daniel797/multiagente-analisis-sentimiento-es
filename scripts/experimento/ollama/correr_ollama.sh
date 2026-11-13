#!/usr/bin/env bash
# Corrida local de Salamandra con Ollama. Ejecutar desde la raíz del repositorio.
# Uso: bash scripts/experimento/ollama/correr_ollama.sh [prueba|completa]
set -euo pipefail
MODO="${1:-prueba}"
MODELO="salamandra-7b-instruct"
SALIDA="docs/experimento/llm_ollama"

command -v ollama >/dev/null || { echo "Falta instalar Ollama (https://ollama.com)"; exit 1; }
ollama list | grep -q "^${MODELO}" || ollama create "${MODELO}" -f scripts/experimento/ollama/Modelfile
pip install -q requests pyyaml pandas scikit-learn >/dev/null

# 1) Revisar a mano el prompt que recibe el modelo
python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" --mostrar-prompt --limite 3 --out docs/experimento/prueba_prompt || true

if [ "${MODO}" = "prueba" ]; then
  # 2) Muestra de 50 tuits para revisar salidas, tiempos y salidas inválidas
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" \
      --semilla 42 --variantes base --limite 50 --out "${SALIDA}_prueba"
  python scripts/experimento/monoagente_salamandra.py resumir --out "${SALIDA}_prueba"
else
  # 3) Corrida completa en dev (2.465 tuits; se puede reanudar si se corta)
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo "${MODELO}" \
      --semilla 42 --variantes base limpio --out "${SALIDA}"
  python scripts/experimento/monoagente_salamandra.py resumir --out "${SALIDA}"
fi
