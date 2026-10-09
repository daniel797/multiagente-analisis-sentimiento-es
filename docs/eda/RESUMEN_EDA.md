# EDA del corpus TASS 2020 por variedad

Resumen de lo que muestran los datos de `data/train` y `data/dev` antes de diseñar los agentes. Todas las cifras salen de `scripts/eda/run_eda.py` (tablas en `docs/eda/tablas/`, figuras en `docs/eda/figuras/`) y se pueden revisar con más detalle en `notebooks/02_eda.ipynb`. No se entrenó ni evaluó ningún modelo. Los textos de las tablas de muestra están anonimizados (`@USUARIO`).

## Qué hay en los datos

Son **7.267 tuits de cinco variedades**: Costa Rica, España, México, Perú y Uruguay. Cada país aporta entre 1.167 y 1.707 tuits, repartidos en *train* (4.802) y *dev* (2.465). Dos cosas no coinciden con el plan original: Las etiquetas son tres (POS, NEG, NEU) y están completas.

Los tuits son cortos: mediana de 92 caracteres y máximo de 146 (unas 16 palabras). Casi no tienen emojis (0 a 0,5%), pero entre el 40% y el 63% lleva una @mención, así que el texto es muy conversacional (`fig03`, `fig04`).

## Calidad

Los datos están limpios: ningún texto vacío y muy pocos repetidos (16 filas tras normalizar). Lo que sí hay que depurar son **2 textos repetidos con etiquetas distintas** y **3 textos que aparecen en *train* y en *dev***. Son pocos, pero afectan justo a la evaluación.

## Etiquetas

Aquí está la diferencia más marcada entre países (`fig02`). **En Perú la clase mayoritaria es NEU** (54% en *train*, 57% en *dev*), mientras que en las otras cuatro variedades domina NEG (entre 37% y 51%). La asociación entre etiqueta y variedad es moderada pero clara (V de Cramér ≈ 0,19), y dentro de cada país *train* y *dev* se parecen entre sí. Antes de leer esa diferencia como un rasgo del dialecto conviene revisar si el criterio de anotación de NEU/NONE fue el mismo en todos los subcorpus.

Predecir siempre la clase mayoritaria da un **Macro-F1 de 0,18 a 0,24** en *dev* (la Accuracy llega a 0,57 en Perú, de ahí que el Macro-F1 sea la métrica principal). Ese es el piso mínimo a superar (`fig07`).

## Fenómenos lingüísticos

Con reglas simples, estimamos la presencia de cada fenómeno (`fig05`). Son aproximaciones, no anotaciones: en una revisión rápida de unos 10 casos por regla, los diminutivos resultaron mayormente correctos (aunque marca palabras como `éxitos`), las "2+ negaciones" incluyen cualquier tuit con dos negaciones, y la regla que el script de auditoría llama "sarcasmo aparente" detecta en realidad **risas** (`jaja`), no sarcasmo.

| Fenómeno | % de tuits (rango por país) | En *dev*, por país |
|---|---|---|
| Diminutivo (-ito/-ita) | 10,1 – 15,8 | 46 – 79 |
| 2+ negaciones | 8,6 – 11,1 | 42 – 54 |
| Inglés (lista corta) | 0,7 – 1,1 | **2 – 7** |
| Risa / marcador de humor | 3,7 – 8,3 | 22 – 42 |

El code-switching es muy escaso (64 tuits en total) y no permite sacar conclusiones por país. Para el resto hará falta una anotación manual: `docs/eda/tablas/muestra_validar_heuristicas.csv` trae 30 casos anonimizados por regla con una columna para confirmar cada marca.

## Tiempo y tema

Las fechas se recuperan del id del tuit y van de **julio de 2016 a enero de 2017**, pero con una distribución muy irregular (`fig06`). En **España el 87% de los tuits cae en solo dos semanas** (finales de agosto de 2016) y en **México el 80%** (enero de 2017). En CR, PE y UY el *dev* es posterior al *train* (hasta siete semanas de diferencia en la mediana). El vocabulario distintivo de cada país mezcla dialecto real (`chamba`, `neta`, `chido`, `mae`, `vos`) con lugares y temas de la actualidad de esos meses (`valencia`, `madrid`, `lima`, `uruguay`, `reyes`). Dentro de las ráfagas de ES y MX la proporción de etiquetas es casi igual a la del resto (ES: 43% NEG dentro del pico, 44% fuera), así que no parecen sesgar la polaridad.

La consecuencia es que **las diferencias de desempeño entre países se confunden con diferencias de época y de tema**.

## Qué implica para el proyecto

1. **Conjunto de prueba.** Decidir si el *dev* oficial hace de prueba bloqueada (conserva el orden temporal en CR, PE y UY) o si se reparte todo de nuevo. Es una decisión metodológica que conviene fijar antes de diseñar los prompts.
2. **Robustez.** Quitar "emojis equivalentes" de las perturbaciones: no hay emojis donde aplicarlos. Quedan tildes, tipeo, mayúsculas, letras alargadas y abreviaturas.
3. **Fenómenos.** Renombrar el "sarcasmo aparente" del script de auditoría y planificar una anotación manual. El code-switching no alcanza para analizarlo por país.
4. **Lectura por país.** Reportar junto a cada resultado la concentración temporal y el posible efecto de anotación de NEU en Perú, y no atribuir diferencias al dialecto sin ese control.
5. **Potencia estadística.** Con ~400 a 580 tuits de *dev* por país, el margen de error máximo de una Accuracy es de ±4 a ±5 puntos. Una mejora de 0,02 en Macro-F1 cae dentro de ese margen en cada país por separado, así que serán imprescindibles las pruebas pareadas y el análisis conjunto.
6. **Configuración.** Con tuits de 146 caracteres como máximo, `contexto_maximo_tokens: 8192` está muy sobredimensionado: el costo lo marcarán el prompt y el número de muestras, no el tuit.
