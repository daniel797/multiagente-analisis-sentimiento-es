# Experimento: baseline y variantes de los datos

Resumen del experimento de baseline clásico con variantes del texto de entrada. Las cifras salen de `scripts/experimento/run_baseline_variantes.py` (tablas en `docs/experimento/tablas/`, figuras en `docs/experimento/figuras/`). El monoagente con LLM está escrito en `scripts/experimento/monoagente_salamandra.py`, pero **no se ejecutó con el modelo real** (ver el final).

## Diseño

Se compara una línea base contra una variante de los datos, con el mismo modelo y los mismos tuits de *dev*. Lo único que cambia es el texto de entrada, porque el LLM no se entrena y su entrada es la única palanca del diseño.

| Variante | Qué cambia |
|---|---|
| `base` | El tuit como lo ve el LLM: `@USUARIO`, `URL` y espacios colapsados. Conserva mayúsculas, tildes, alargamientos y emojis |
| **`limpio`** | Minúsculas, sin tildes (la ñ se conserva), sin letras alargadas y sin emojis |
| `minusculas`, `sin_tildes`, `sin_alargamientos`, `sin_emojis` | Un solo cambio de los cuatro anteriores |
| `sin_usuario_url` | Extra: quita los marcadores `@USUARIO` y `URL` |

Las funciones de variante están en `scripts/experimento/variantes_datos.py` y son las mismas para el clasificador de referencia y para el LLM.

- **Modelo del baseline:** TF-IDF de palabras (1 a 2) con regresión logística balanceada, sin ajustar hiperparámetros. Se ajusta solo con *train* y se mide solo en *dev* (2.465 tuits). La variante se aplica igual a *train* y a *dev*.
- **Métrica:** Macro-F1 con IC 95% por bootstrap (2.000 remuestreos). La variante se compara con la base con bootstrap pareado sobre la diferencia, porque ambas se miden sobre los mismos tuits.

## Resultados (`resultados_por_variante.csv`, `diferencias_pareadas_variantes.csv`)

El texto limpio no cambia el rendimiento: Macro-F1 de **0,557** contra **0,556** de la base, con una diferencia pareada de +0,001 (IC 95% de −0,008 a +0,011).

| Ámbito | Base | Limpio | Diferencia (IC 95%) | Predicciones que cambian |
|---|---|---|---|---|
| Todas | 0,556 | 0,557 | +0,001 (−0,008 a +0,011) | 178 de 2.465 |
| CR | 0,586 | 0,586 | 0,000 (−0,020 a +0,020) | 16 de 390 |
| ES | 0,538 | 0,539 | +0,001 (−0,020 a +0,020) | 49 de 581 |
| MX | 0,559 | 0,557 | −0,002 (−0,027 a +0,023) | 41 de 510 |
| PE | 0,501 | 0,503 | +0,002 (−0,020 a +0,023) | 41 de 498 |
| UY | 0,567 | 0,576 | +0,008 (−0,011 a +0,026) | 31 de 486 |

- **Cambia mucho texto y poco resultado.** El texto limpio altera 178 predicciones (7,2%), pero en ambas direcciones: 72 tuits pasan de error a acierto y 69 de acierto a error.
- **Ninguna variante se distingue de la base**, ni en total ni por variedad (`fig02`). En el total, los intervalos excluyen un cambio de 0,02 o más. Por variedad (390 a 581 tuits) no lo excluyen.
- **Dos variantes no pueden mover nada por construcción:** el vectorizador ya pasa a minúsculas, y solo el 0,2% de los tuits tiene emojis.
- **Perú sigue siendo la variedad más baja** con ambos textos (0,501 y 0,503).
- **Lo que esto no dice:** un LLM sí distingue mayúsculas, tildes y alargamientos, así que este resultado es el piso clásico y no predice qué pasará con Salamandra.

## Limitaciones

- No existe partición de prueba. Todo se reporta en *dev*, sin ajustar nada con él, pero los prompts del LLM sí se ajustarán con datos y eso exige reservar un subconjunto.
- 6 variantes por 6 ámbitos sin corrección por comparaciones múltiples. Como ninguna resulta distinta de la base, la corrección no cambiaría la conclusión.
- Las variantes son implementaciones sencillas hechas para este proyecto (por ejemplo, los alargamientos son letras repetidas 3 o más veces).
- Quedan 2 textos con etiquetas en conflicto y 40 pares casi duplicados (8 cruzan *train* y *dev*), que este experimento no depuró.

## El monoagente con LLM (pendiente de correr)

`scripts/experimento/monoagente_salamandra.py` clasifica cada tuit con Salamandra-7B-Instruct (zero-shot, salida JSON, sin pasar la variedad) para cada variante, guarda cada predicción apenas la obtiene y se puede reanudar. Se probó el flujo completo con un backend de reglas (`--backend stub`) y con las pruebas de `tests/test_experimento.py`. **No se ejecutó con el modelo real**, porque el entorno donde se escribió no tiene GPU ni acceso a Hugging Face.

Antes de la primera corrida real:

1. Fijar `revision` en `configs/model_config.yaml` (hoy está vacía).
2. Instalar `requirements-llm.txt` en una máquina con GPU (por ejemplo Colab).
3. Probar con `--limite 50` y revisar a mano las salidas.

```bash
python scripts/experimento/monoagente_salamandra.py ejecutar --variantes base limpio --limite 50 --out docs/experimento/llm
python scripts/experimento/monoagente_salamandra.py ejecutar --variantes base limpio --out docs/experimento/llm
python scripts/experimento/monoagente_salamandra.py resumir --out docs/experimento/llm
```

Las salidas que el modelo no logra devolver en formato válido (tras 2 reintentos) cuentan como `NEU` y su tasa se reporta aparte en `salidas_invalidas.csv`.
