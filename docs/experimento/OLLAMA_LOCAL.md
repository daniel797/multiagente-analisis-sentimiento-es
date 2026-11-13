# Correr Salamandra en local con Ollama

Estado: **preparado, no ejecutado.** El entorno donde se escribió esto no tiene GPU ni acceso al modelo, así que el flujo se probó solo con un servidor Ollama simulado. La primera corrida real hay que revisarla a mano.

## Pasos

1. Instalar Ollama (https://ollama.com).
2. Descargar un GGUF de `BSC-LT/salamandra-7b-instruct` y dejarlo como `salamandra-7b-instruct.Q8_0.gguf` junto al Modelfile (o ajustar la ruta en `scripts/experimento/ollama/Modelfile`). Busca en Hugging Face el repositorio GGUF oficial del BSC o uno de la comunidad y comprueba que sea del modelo instruido; no verifiqué el nombre exacto del repositorio. Con `ollama pull hf.co/<repo>:<cuantización>` también sirve; en ese caso quita el bloque `TEMPLATE` del Modelfile si el GGUF ya trae la plantilla.
3. Desde la raíz del repo: `bash scripts/experimento/ollama/correr_ollama.sh prueba` (50 tuits) y, si las salidas se ven bien, `... completa` (Windows: `.\scripts\experimento\ollama\correr_ollama.ps1 -Modo completa`).

## Parámetros y por qué

| Parámetro | Valor | Motivo |
|---|---|---|
| temperature | 0 | Clasificación determinista. |
| top_k / top_p | 1 / 1,0 | Refuerza la decodificación greedy; no depende del valor de temperature. |
| seed | 42 | Misma semilla que `configs/model_config.yaml`. |
| num_ctx | 4096 | Sobra para prompt + tuit (~700 tokens) y evita contextos grandes más lentos. |
| num_predict | 24 | La salida es un JSON corto. |
| repeat_penalty | 1,0 | Sin penalización: la respuesta es una de tres etiquetas. |
| salida | JSON restringido a {NEG, NEU, POS} | Evita salidas inválidas; se sigue contando cualquier reintento. |
| cuantización | Q8_0 (o Q4_K_M si no cabe) | Q8_0 es casi fiel al modelo original; la corrida de Colab fue 4-bit, así que Q4_K_M es la comparable. |

## Qué mirar antes de confiar en los resultados

- Con `--mostrar-prompt`, comprobar que la plantilla de chat de Ollama produce el mismo prompt que Hugging Face.
- En la muestra de 50, revisar a mano la proporción de salidas inválidas y las etiquetas.
- El resultado es **otra configuración** (GGUF y cuantización distintos), no una repetición exacta de la corrida de Colab; el Macro-F1 puede diferir. Se guarda `registro_corrida.json` con la versión de Ollama, el digest del modelo y los parámetros.
- El Macro-F1 de esta corrida solo se mide en dev, una vez, como el resto.
