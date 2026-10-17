# Feature Engineering

Resumen de la ingeniería de características del proyecto. Las cifras salen de `scripts/features/run_features.py` (tablas en `docs/features/tablas/`, figuras en `docs/features/figuras/`) y se explican paso a paso en `notebooks/03_feature_engineering.ipynb`.

El proyecto no entrena ningún modelo: un LLM recibe el tuit y devuelve una etiqueta. Por eso aquí "feature engineering" significa tres cosas: decidir qué texto ve el LLM, construir un catálogo de características para entender el corpus y contrastar al agente de Percepción, y fijar pisos de referencia. Todo lo que se ajusta (vectorizadores, escalado, modelos de referencia) se ajusta solo con *train* y se mide solo en *dev*. No se toca ninguna partición de prueba.

## 1. Qué ve el LLM

La función `normalizar_para_llm` cambia las @menciones por `@USUARIO` y los enlaces por `URL`, y colapsa espacios repetidos. No toca mayúsculas, puntuación, alargamientos, emojis ni tildes, porque son señales de tono. Y **no se le pasa la variedad**: el sistema no debe saber de qué país es el tuit. La variedad solo se usa para reportar resultados por país.

## 2. Catálogo de 24 características

Cinco grupos, cada uno con definición y justificación (`catalogo_features.csv`):

| Grupo | Características | Ejemplos |
|---|---|---|
| Estructural (6) | longitud y forma del texto | `n_caracteres`, `long_media_palabra`, `ratio_mayusculas` |
| Expresividad (7) | puntuación y marcas orales | `n_exclamaciones`, `n_alargamientos`, `n_emojis` |
| Conversacional (4) | rasgos de red social | `n_menciones`, `n_hashtags`, `tiene_url` |
| Fenómeno (4) | reglas del agente de Percepción | `n_diminutivos`, `n_negaciones`, `n_risas` |
| Léxico (3) | listas cortas propias, sin validar | `n_lex_positivo`, `n_lex_negativo`, `n_intensificadores` |

Lo que antes se llamaba "sarcasmo" es `n_risas`, porque la regla solo detecta risas. El léxico es una lista de unas 30 palabras por lado, y se incluye para medir cuánta señal da una lista pequeña, no como léxico de referencia.

## 3. Qué señales distinguen las etiquetas

- La señal más fuerte viene del **léxico mínimo**: `n_lex_positivo` (epsilon² = 0,139) y `n_lex_negativo` (0,064). Le siguen la **negación** (0,047) y las **exclamaciones** (0,046).
- Los tuits POS tienen en promedio 0,58 exclamaciones frente a 0,17 en los NEG. Los NEG tienen más negaciones (0,62 frente a 0,25 en los POS).
- La longitud, los emojis, los emoticonos y el code-switching casi no distinguen las etiquetas, y varias de esas características casi no aparecen (`n_emojis` en el 0,2% de los tuits, `n_marcadores_ingles` en el 0,9%).

**Banderas del agente de Percepción como reglas aisladas** (`perfil_pistas.csv`): la doble negación es la más informativa (55,1% de NEG entre los tuits que la activan, frente a 39,1% en el total). El diminutivo inclina hacia POS (36,5% frente a 28,9%) y la risa aleja de NEG (29,9%). El code-switching (64 tuits) se comporta como el total. Por tanto, el valor del agente de Percepción tendría que estar en interpretar los fenómenos en contexto, algo que una regla no puede hacer, y eso es lo que el experimento debe medir.

## 4. Redundancia

Dos pares son redundantes: `n_caracteres` con `n_palabras` (Spearman 0,94) y `n_menciones` con `empieza_con_mencion` (0,88). En modelos numéricos conviene dejar una de cada par.

## 5. ¿Delatan el país en lugar del sentimiento?

Las características más ligadas al país que al sentimiento son sobre todo las conversacionales: `n_menciones` (efecto 0,041 sobre la variedad frente a 0,027 sobre la etiqueta), `empieza_con_mencion`, `tiene_url` y `n_hashtags` (`fig02`). Además, solo con el texto un clasificador de palabras adivina el país con un Macro-F1 de 0,38 frente a 0,20 al azar (`prediccion_de_variedad.csv`). El texto delata parcialmente el país aunque no se lo digamos al LLM, así que no pasarle la variedad no es una garantía total de que no la use.

## 6. Pisos de referencia (`fig01`, `fig03`)

Regresión logística entrenada con *train* y medida en *dev* (2.465 tuits), con intervalo de confianza del 95% por *bootstrap*:

| Representación | Macro-F1 (IC 95%) |
|---|---|
| Siempre la clase mayoritaria | 0,19 (0,18 a 0,19) |
| Superficie (estructural, expresividad, conversacional) | 0,43 (0,41 a 0,45) |
| Superficie + fenómenos + léxico mínimo | 0,51 (0,49 a 0,53) |
| TF-IDF de palabras | 0,56 (0,54 a 0,57) |
| TF-IDF de caracteres | 0,57 (0,55 a 0,59) |
| TF-IDF palabras + caracteres | 0,57 (0,55 a 0,59) |
| TF-IDF palabras + todas las numéricas | 0,56 (0,54 a 0,58) |

Combinar las características hechas a mano con las palabras **no mejora** sobre TF-IDF (ver la prueba pareada en la sección 7). Por país, Perú queda más abajo en casi todos los modelos (0,44 a 0,52), en línea con su exceso de neutrales visto en el EDA.

En `terminos_por_clase.csv`, las palabras de NEG (*triste, feo, peor, odio*) y POS (*feliz, gracias, amo, bonito*) son afectivas. En NEU aparecen palabras conversacionales (*hola, creo, igual, normal*) y `usuario`, que viene de las menciones normalizadas.

## 7. ¿Las diferencias entre representaciones son reales? (`diferencias_pareadas.csv`)

Se usa un *bootstrap* pareado sobre la diferencia de Macro-F1, porque los modelos se miden sobre los mismos tuits. Comparar si dos intervalos separados se solapan no basta.

- **TF-IDF de palabras supera a las 24 características** por 0,047 (IC 0,025 a 0,068).
- **No se distingue** entre TF-IDF de palabras y de caracteres (0,012, IC −0,006 a 0,032), ni con su combinación (0,010), ni agregando las características numéricas a las palabras (0,002).

## 8. ¿Qué aporta cada grupo? (`ablacion_por_grupo.csv`, `importancia_permutacion.csv`, `fig05`, `fig06`)

- Quitar el **léxico** baja el Macro-F1 en 0,075 (IC −0,096 a −0,054), y es el único grupo cuya ausencia se nota. Quitar cualquier otro grupo no cambia nada distinguible.
- Solo con las **3 características de léxico** se llega a 0,48, contra 0,51 con las 24. Los otros grupos, cada uno solo, quedan entre 0,33 y 0,39.
- En la importancia por permutación dominan `n_lex_positivo` (caída de 0,080) y `n_lex_negativo` (0,026), seguidos de `n_negaciones` (0,013), `n_intensificadores` y `n_exclamaciones` (0,009). Emojis, emoticonos, enlaces y oraciones no aportan.

## 9. Estabilidad frente a las perturbaciones (`estabilidad_features.csv`, `robustez_modelos.csv`, `fig07`, `fig08`)

Se aplican las perturbaciones previstas para la prueba de robustez (sin tildes, mayúsculas, minúsculas, letras alargadas, tipeo y abreviaturas). Los modelos se entrenan con texto limpio.

- Las características de **mayúsculas** son las más frágiles: cambian en el 100% de los tuits al poner todo en mayúsculas, y el modelo de 24 características cae **0,084**. TF-IDF no cambia.
- Las **letras alargadas** bajan 0,026 a TF-IDF de palabras y 0,028 al modelo numérico, y casi no afectan a TF-IDF de caracteres (−0,005).
- El **tipeo** baja todos los modelos entre 0,014 y 0,020. Las **abreviaturas** y las **tildes** casi no los afectan.
- Para el experimento: un LLM no es invariante a mayúsculas como TF-IDF, así que conviene medir su caída en mayúsculas y letras alargadas, las perturbaciones más informativas.

## Decisiones

1. **Prompt:** normalización mínima, sin tocar mayúsculas, puntuación ni emojis, y sin pasar la variedad.
2. **Pisos:** comparar el LLM contra la clase mayoritaria (0,19) y contra TF-IDF (0,56 a 0,57). Entre las variantes de TF-IDF no hay diferencias distinguibles. Si el monoagente queda por debajo de TF-IDF, reportarlo.
3. **Percepción:** medir si el agente aporta algo sobre las banderas heurísticas, que como reglas aisladas dicen poco.
4. **Atajos:** reportar los resultados por país junto con el contexto temporal, y usar con cuidado las características conversacionales.
5. **Robustez:** medir la caída del monoagente y del multiagente bajo mayúsculas y letras alargadas, y compararla con estos números de referencia.
6. **Catálogo:** conservar léxico, negación, exclamaciones e intensificadores, quitar una característica de cada par redundante y descartar las que no aportan.
7. **Pendientes:** validar a mano las reglas (muestra de `docs/eda/tablas/muestra_validar_heuristicas.csv`) y sustituir el léxico mínimo por uno validado, que sería la mejora más rentable.
