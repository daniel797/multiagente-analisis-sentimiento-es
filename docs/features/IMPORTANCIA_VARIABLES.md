# Importancia de variables

Qué características del tuit usa el modelo de 24 variables para clasificar el sentimiento. Se miden de dos formas, las dos sobre el conjunto de desarrollo (2.465 tuits), con el modelo entrenado solo con train (regresión logística balanceada, variables estandarizadas).

- **Importancia por permutación.** Se desordena una columna de dev a la vez, 15 veces, y se mide cuánto cae el Macro-F1. Si cae, el modelo dependía de esa variable. Se calcula con las demás presentes, así que es más confiable que mirar cada variable sola.
- **Ablación por grupo.** Se reentrena el modelo sin un grupo de variables, y también solo con un grupo. El cambio de Macro-F1 se compara con la versión completa, con IC 95% por bootstrap pareado.

Tablas: `tablas/importancia_permutacion.csv` y `tablas/ablacion_por_grupo.csv`. Figuras: `figuras/fig06_importancia_permutacion.png` y `figuras/fig05_ablacion_por_grupo.png`.

## Importancia por permutación

| Variable | Grupo | Caída de Macro-F1 | Desv. |
| --- | --- | --- | --- |
| `n_lex_positivo` | léxico | 0,0800 | 0,0071 |
| `n_lex_negativo` | léxico | 0,0261 | 0,0034 |
| `n_negaciones` | fenómeno | 0,0128 | 0,0042 |
| `n_intensificadores` | léxico | 0,0092 | 0,0032 |
| `n_exclamaciones` | expresividad | 0,0092 | 0,0033 |
| `empieza_con_mencion` | conversacional | 0,0072 | 0,0032 |
| `n_palabras` | estructural | 0,0055 | 0,0037 |
| `n_interrogaciones` | expresividad | 0,0023 | 0,0024 |

Las otras 16 variables quedan entre −0,005 y +0,001, es decir, dentro del ruido. Entre ellas están emojis, emoticonos, enlaces, número de oraciones, risas y diminutivos.

## Ablación por grupo

| Configuración | Variables | Macro-F1 | Cambio contra el completo (IC 95%) |
| --- | --- | --- | --- |
| Completo | 24 | 0,5085 | |
| Sin léxico | 21 | 0,4331 | −0,0754 (−0,0962 a −0,0537) |
| Sin estructural | 18 | 0,5067 | −0,0017 (−0,0099 a +0,0068) |
| Sin fenómeno | 20 | 0,5113 | +0,0028 (−0,0080 a +0,0136) |
| Sin conversacional | 20 | 0,5168 | +0,0083 (−0,0054 a +0,0232) |
| Sin expresividad | 17 | 0,5174 | +0,0089 (−0,0019 a +0,0201) |
| Solo léxico | 3 | 0,4757 | −0,0328 (−0,0507 a −0,0135) |
| Solo estructural | 6 | 0,3927 | −0,1158 (−0,1398 a −0,0912) |
| Solo conversacional | 4 | 0,3720 | −0,1365 (−0,1591 a −0,1133) |
| Solo expresividad | 7 | 0,3542 | −0,1543 (−0,1783 a −0,1285) |
| Solo fenómeno | 4 | 0,3318 | −0,1767 (−0,2005 a −0,1543) |

## Lectura

- El léxico de valoración es lo único cuya ausencia se nota: quitarlo baja 0,075. Con solo sus 3 variables se llega a 0,48, contra 0,51 con las 24.
- Quitar cualquier otro grupo no cambia nada que se distinga de cero. Que el Macro-F1 suba un poco al quitar expresividad o conversacional no es una mejora: los intervalos incluyen el cero.
- Después del léxico siguen la negación, los intensificadores y las exclamaciones. Son las pistas con más sentido para pasarlas al agente de Percepción.
- `empieza_con_mencion` aparece con peso propio, pero las variables conversacionales se relacionan más con el país que con el sentimiento (ver `RESUMEN_FEATURES.md`, sección de atajos), así que conviene tratarla como un atajo posible y no como una señal de sentimiento.

## Límites

- La importancia es del modelo de 24 variables, no del LLM. Sirve para decidir qué pistas ofrecer al agente de Percepción, no para explicar al LLM.
- Las variables correlacionadas se reparten la importancia (ver `tablas/correlaciones_altas.csv`), así que una caída pequeña no prueba que la variable no sirva.
- El léxico es una lista propia de unas 30 palabras por lado, sin validar. Su importancia mide esa lista, no el léxico de valoración en general.
- Todo se mide en dev, porque no existe partición de prueba. No se ajustó ningún hiperparámetro con esos datos.
