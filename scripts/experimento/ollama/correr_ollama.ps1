# Corrida local de Salamandra con Ollama (Windows). Ejecutar desde la raíz del repositorio.
param([string]$Modo = "prueba")
$Modelo = "salamandra-7b-instruct"
$Salida = "docs/experimento/llm_ollama"
if (-not (Get-Command ollama -ErrorAction SilentlyContinue)) { Write-Error "Falta instalar Ollama (https://ollama.com)"; exit 1 }
if (-not (ollama list | Select-String "^$Modelo")) { ollama create $Modelo -f scripts/experimento/ollama/Modelfile }
pip install -q requests pyyaml pandas scikit-learn | Out-Null
python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo $Modelo --mostrar-prompt --limite 3 --out docs/experimento/prueba_prompt
if ($Modo -eq "prueba") {
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo $Modelo --semilla 42 --variantes base --limite 50 --out "${Salida}_prueba"
  python scripts/experimento/monoagente_salamandra.py resumir --out "${Salida}_prueba"
} else {
  python scripts/experimento/monoagente_salamandra.py ejecutar --backend ollama --ollama-modelo $Modelo --semilla 42 --variantes base limpio --out $Salida
  python scripts/experimento/monoagente_salamandra.py resumir --out $Salida
}
