# Evaluación de una arquitectura multiagente basada en LLMs para el análisis de sentimientos en variedades del español

Repositorio de tesis de maestría — Universidad Nacional de Ingeniería (UNI), Maestría en Ciencias con mención en Inteligencia Artificial.

**Autor:** Julio Daniel Pozo García

---

## 

El proyecto compara, de forma experimental y controlada, un sistema **monoagente** (un LLM que clasifica el sentimiento de un tuit en una sola llamada) contra un sistema **multiagente** (el mismo LLM coordinando varios roles: percepción de fenómenos lingüísticos, razonamiento y resolución) sobre tuits en español de cinco variedades dialectales: España, México, Perú, Costa Rica y Uruguay, usando el corpus multivariante de TASS 2020.

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
├── data/
│   ├── train/            # Un TSV por variedad (cr, es, mx, pe, uy), sin encabezado: id, texto, etiqueta
│   └── dev/              # Ídem, partición de desarrollo
├── docs/
│   ├── eda/              # Resultados del EDA: RESUMEN_EDA.md, tablas/ (CSV) y figuras/ (PNG)
│   ├── features/         # Resultados de Feature Engineering: RESUMEN_FEATURES.md, tablas/ y figuras/
│   └── *.pdf             # Avances entregados (PDFs de cada semana)
├── notebooks/           # 01 exploración inicial · 02 EDA · 03 Feature Engineering
├── scripts/
│   ├── data/             # Consolidación, partición y auditoría del dataset
│   ├── eda/              # Análisis exploratorio (tablas y figuras)
│   ├── features/         # Catálogo de características y líneas base de referencia
│   └── utils/            # Funciones compartidas (lectura/escritura, logging)
├── tests/               # Pruebas de los scripts de datos, EDA y features
├── requirements.txt
└── README.md
```

## El dataset: cómo se consigue y por qué no está en este repositorio

El corpus **InterTASS / TASS 2020 multivariante** no es de descarga directa ni de uso libre sin restricciones: se solicita en [tass.sepln.org](http://tass.sepln.org/) aceptando una licencia de datos, y el enlace de descarga llega por correo. Por eso los archivos originales no se versionan en este repositorio (quedan fuera por `.gitignore`), solo la estructura de carpetas donde deben colocarse.

Para reproducir el proyecto:

1. Registrarse y descargar el corpus en tass.sepln.org (ediciones ES, MX, PE, CR, UY de TASS 2020).
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
| `scripts/eda/run_eda.py` | Genera el EDA completo: 23 tablas CSV en `docs/eda/tablas/` y 9 figuras en `docs/eda/figuras/` a partir de `data/train` y `data/dev`. |
| `scripts/eda/eda_utils.py` | Funciones del EDA (calidad de datos, etiquetas, rasgos del texto, fenómenos, vocabulario, fechas, línea base mayoritaria). Las usan el script y el notebook. |
| `scripts/eda/eda_extra.py` | Segunda etapa del EDA: intervalos de confianza, época vs. país, línea base TF-IDF (solo como referencia), potencia estadística y textos casi duplicados. |
| `scripts/eda/validar_heuristicas.py` | Calcula la precisión de las reglas de fenómenos a partir de la muestra anotada a mano (`docs/eda/tablas/muestra_validar_heuristicas.csv`). |
| `scripts/features/feature_utils.py` | Feature Engineering: catálogo de 24 características lingüísticas y de superficie, normalización del texto para el prompt, banderas de Percepción y modelos de referencia (TF-IDF). |
| `scripts/features/run_features.py` | Genera el análisis de Feature Engineering: 10 tablas CSV en `docs/features/tablas/` y 4 figuras en `docs/features/figuras/`. |
| `scripts/utils/io_utils.py` | Funciones compartidas de lectura/escritura y logging usadas por los scripts anteriores. |

Ninguno de estos scripts entrena un modelo: preparan los datos. La implementación de los agentes (`agents/`) queda como siguiente etapa, una vez cerrada la auditoría del corpus.

## Análisis exploratorio (EDA)

El EDA corre sobre los datos reales de `data/train` y `data/dev` (7.267 tuits). Resumen completo en [`docs/eda/RESUMEN_EDA.md`](docs/eda/RESUMEN_EDA.md) y notebook en `notebooks/02_eda.ipynb`. Lo más relevante:

- El proyecto trabaja con **cinco variedades** (CR, ES, MX, PE, UY). Chile no está en los datos disponibles y se omite. Tampoco hay partición de prueba.
- **Perú es la única variedad con NEU como clase mayoritaria** (54-57%), y la diferencia no se explica solo por la época: se mantiene al comparar los países dentro del mismo mes.
- Dos pisos de referencia en *dev*: predecir siempre la clase mayoritaria da un **Macro-F1 de 0,18 a 0,24**, y un clasificador TF-IDF simple da **0,50 a 0,59**.
- Con ~400 a 580 tuits de *dev* por país, la diferencia mínima detectable es de 5 a 6 puntos por país y de 2,5 con todo el *dev*: una mejora de 0,02 en Macro-F1 probablemente no se detecte.
- Los tuits son cortos (máx. 146 caracteres) y casi no tienen emojis.
- El **code-switching es muy escaso** (64 tuits en total) y el script de auditoría llama "sarcasmo" a lo que en realidad son risas.
- En ES y MX más del 80% de los tuits está concentrado en dos semanas, así que época, tema y dialecto se pueden confundir.
- Hay 6 tuits del *dev* con un casi duplicado en *train*.

```bash
python scripts/eda/run_eda.py                # regenera docs/eda/tablas y docs/eda/figuras
python scripts/eda/validar_heuristicas.py    # precisión de las reglas, cuando la muestra esté anotada
```

## Feature Engineering

El proyecto no entrena ningún modelo, pero sí tiene ingeniería de características: qué texto ve el LLM, qué señales lingüísticas hay en el corpus y qué pisos de referencia debe superar el sistema. Detalle en [`docs/features/RESUMEN_FEATURES.md`](docs/features/RESUMEN_FEATURES.md) y en `notebooks/03_feature_engineering.ipynb`. Lo más relevante:

- **Catálogo de 24 características** en 5 grupos (estructural, expresividad, conversacional, fenómeno y léxico), cada una con definición y justificación.
- **Texto del prompt:** solo se anonimizan menciones y enlaces. No se toca mayúsculas, puntuación ni emojis, y no se le pasa la variedad al modelo.
- La señal que más distingue las etiquetas viene del **léxico**, la **negación** y las **exclamaciones**. La longitud, los emojis y el code-switching casi no aportan.
- Las banderas del agente de Percepción, como reglas aisladas, dicen poco (salvo la doble negación). Su valor tendría que estar en interpretarlas en contexto.
- Las características conversacionales (menciones, hashtags, enlaces) se ligan más al país que al sentimiento, y el texto por sí solo delata parcialmente el país (Macro-F1 de 0,38 frente a 0,20 al azar).
- **Pisos de referencia en *dev*:** clase mayoritaria 0,19 y TF-IDF 0,56 a 0,57 de Macro-F1. Los modelos de referencia se ajustan solo con *train* y no forman parte del experimento.

```bash
python scripts/features/run_features.py      # regenera docs/features/tablas y docs/features/figuras
```

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

# EDA sobre data/train y data/dev
python scripts/eda/run_eda.py

# Feature Engineering
python scripts/features/run_features.py
```

## Estado del proyecto

- [x] Semana 2 — Diseño del flujo reproducible (`docs/Avance_Semana2_Julio_Pozo.pdf`)
- [x] Scripts de consolidación, partición y auditoría del corpus
- [x] EDA del corpus por variedad (`docs/eda/`, `notebooks/02_eda.ipynb`)
- [x] Feature Engineering (`docs/features/`, `notebooks/03_feature_engineering.ipynb`)
- [ ] Implementación de los agentes (Percepción, Razonamiento, Resolución, Coordinador)
- [ ] Ejecución de las condiciones experimentales (B1, B3, B4, B5, B6)
- [ ] Evaluación y análisis estadístico

## Licencia y uso de datos

Este repositorio no redistribuye el corpus TASS/InterTASS, por sus condiciones de licencia. El código del repositorio puede usarse libremente citando la fuente.
