# Evaluación de una arquitectura multiagente basada en LLMs para el análisis de sentimientos en variedades del español

Repositorio de tesis de maestría — Universidad Nacional de Ingeniería (UNI), Maestría en Ciencias con mención en Inteligencia Artificial.

**Autor:** Julio Daniel Pozo García

---

## 

El proyecto compara, de forma experimental y controlada, un sistema **monoagente** (un LLM que clasifica el sentimiento de un tuit en una sola llamada) contra un sistema **multiagente** (el mismo LLM coordinando varios roles: percepción de fenómenos lingüísticos, razonamiento y resolución) sobre tuits en español de seis variedades dialectales: España, México, Perú, Chile, Costa Rica y Uruguay, usando el corpus multivariante de TASS 2020.

No se entrena ni se ajusta ningún modelo. El LLM base se usa preentrenado tal cual; lo que se diseña y evalúa es cómo se coordinan las llamadas que se le hacen.

## Solo inferencia, sin entrenamiento

Este punto se presta a confusión, así que queda explícito: el proyecto **no incluye ninguna etapa de entrenamiento ni de fine-tuning**. El modelo base se descarga o se consume por API exactamente como se publica, sin ajustar sus pesos, ni para la condición monoagente ni para la multiagente.

El corpus (TASS 2020) se usa en tres momentos, pero en ninguno para entrenar:

- **Desarrollo:** se analizan los errores del monoagente para escribir mejor los prompts de los agentes (ingeniería de prompts, no aprendizaje de parámetros).
- **Validación:** se elige el umbral de confianza τ y `max_iter` del meta-agente, por prueba y error sobre resultados, sin tocar el modelo.
- **Prueba bloqueada:** se mide el desempeño una sola vez, al final.

Esto es deliberado: si se entrenara o ajustara el modelo en alguna de las dos condiciones, ya no se podría saber si una mejora viene de la coordinación entre agentes o del entrenamiento extra. Dejar el modelo fijo y sin tocar en ambas condiciones es lo que permite atribuir limpiamente cualquier diferencia de Macro-F1 a la arquitectura.

Este repositorio documenta el avance del proyecto semana a semana. El estado actual corresponde al **avance de la Semana 2**: diseño del flujo reproducible (ver `docs/`). Los scripts de este repositorio preparan el terreno para la fase experimental: consolidar el corpus, auditar fenómenos lingüísticos y particionar los datos sin fuga de información, antes de implementar los agentes.

## Estructura del repositorio

```
.
├── agents/             # Esqueleto de los agentes (percepción, razonamiento, resolución, coordinador)
├── configs/             # Configuración del modelo base y de los experimentos (versión exacta, semillas, etc.)
├── data/               # Archivos originales de TASS 2020, tal como se reciben
├── docs/                # Avances entregados (PDFs de cada semana)
├── notebooks/           # Exploración del corpus, auditoría de fenómenos, prototipos rápidos
├── scripts/
│   ├── data/             # Consolidación, partición y auditoría del dataset
│   └── utils/            # Funciones compartidas (lectura/escritura, logging)
├── tests/               # Pruebas de los scripts de datos
├── requirements.txt
└── README.md
```

## El dataset: cómo se consigue y por qué no está en este repositorio

El corpus **InterTASS / TASS 2020 multivariante** no es de descarga directa ni de uso libre sin restricciones: se solicita en [tass.sepln.org](http://tass.sepln.org/) aceptando una licencia de datos, y el enlace de descarga llega por correo. Por eso los archivos originales no se versionan en este repositorio (quedan fuera por `.gitignore`), solo la estructura de carpetas donde deben colocarse.

Para reproducir el proyecto:

1. Registrarse y descargar el corpus en tass.sepln.org (ediciones ES, MX, PE, CL, CR, UY de TASS 2020).
2. Colocar los archivos originales, sin modificar, en `data/`
3. Correr `scripts/data/consolidate_dataset.py` para unificarlos en un único archivo tabular.
4. Correr `scripts/data/split_dataset.py` para generar las particiones de desarrollo, validación y prueba bloqueada.
5. (Opcional, recomendado) Correr `scripts/data/audit_phenomena.py` sobre el conjunto de desarrollo para contar ejemplos de sarcasmo, diminutivos afectivos, doble negación y code-switching, antes de diseñar los prompts de los agentes.

## Scripts disponibles

| Script | Qué hace |
|---|---|
| `scripts/data/consolidate_dataset.py` | Lee los archivos crudos por variedad dialectal (XML/TSV/CSV, según venga de TASS) y los consolida en un único CSV con columnas `id, texto, variedad, etiqueta`. |
| `scripts/data/split_dataset.py` | Genera una partición estratificada por variedad y etiqueta en `dev` / `val` / `test`, dejando el test bloqueado (no se vuelve a tocar hasta la evaluación final). |
| `scripts/data/audit_phenomena.py` | Corre heurísticas simples (no un modelo) para contar menciones aproximadas de diminutivos afectivos, doble negación, code-switching y marcadores de sarcasmo/ironía, y deja un reporte por variedad. Sirve para decidir si hay evidencia suficiente en el corpus antes de comprometer hipótesis por fenómeno. |
| `scripts/utils/io_utils.py` | Funciones compartidas de lectura/escritura y logging usadas por los scripts anteriores. |

Ninguno de estos scripts entrena un modelo: preparan los datos. La implementación de los agentes (`agents/`) queda como siguiente etapa, una vez cerrada la auditoría del corpus.

## Modelo base

Se fijó **Salamandra-7B-Instruct** (BSC-LT), un modelo abierto entrenado específicamente en español y lenguas cooficiales, en vez de un multilingüe genérico. La versión exacta (familia, tamaño, cuantización, parámetros de decodificación y semilla) queda documentada en `configs/model_config.yaml`, siguiendo la observación de que "LLaMA o Qwen, 7B–13B" no era una especificación reproducible por sí sola. El modelo se usa preentrenado, sin fine-tuning (ver sección anterior).

## Cómo correr esto

```bash
python -m venv .venv
source .venv/bin/activate         # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/data/consolidate_dataset.py --raw-dir data/raw --out data/processed/tass2020_consolidado.csv
python scripts/data/split_dataset.py --input data/processed/tass2020_consolidado.csv --out-dir data/processed
python scripts/data/audit_phenomena.py --input data/processed/dev.csv --out data/processed/auditoria_fenomenos.csv
```

## Estado del proyecto

- [x] Semana 2 — Diseño del flujo reproducible (`docs/Avance_Semana2_Julio_Pozo.pdf`)
- [x] Scripts de consolidación, partición y auditoría del corpus
- [ ] Implementación de los agentes (Percepción, Razonamiento, Resolución, Coordinador)
- [ ] Ejecución de las condiciones experimentales (B1, B3, B4, B5, B6)
- [ ] Evaluación y análisis estadístico

## Licencia y uso de datos

Este repositorio no redistribuye el corpus TASS/InterTASS, por sus condiciones de licencia. El código del repositorio puede usarse libremente citando la fuente.
