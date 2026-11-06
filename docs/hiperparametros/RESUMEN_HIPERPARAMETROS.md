# Optimización de hiperparámetros

Avance del sprint hasta este punto. Cuaderno: `notebooks/06_optimizacion_hiperparametros.ipynb` (recalcula todo, con CPU, en unos 4 minutos). Script: `scripts/experimento/optimizar_hiperparametros.py`. Tablas: `tablas/`.

## Qué se optimiza

El **baseline clásico** (TF-IDF o las 24 variables, con regresión logística). Es el piso contra el que se compara al LLM. El LLM no se optimiza: Salamandra-7B-Instruct no se entrena, su decodificación es determinista y su configuración queda fija (revisión fija, 4 bits). Lo único ajustable sería el prompt, y se haría solo con una submuestra de train.

## Protocolo

- Búsqueda con validación cruzada de 5 pliegues **solo en train**, estratificada por etiqueta y variedad. El vectorizador y el escalador se ajustan dentro de cada pliegue.
- Dev (2.465 tuits) se mide **una sola vez**, con la configuración ya elegida. Diferencias pareadas con IC 95% por bootstrap.
- 182 configuraciones: palabras (126), caracteres (42) y 24 variables (14). C ∈ {0,03 … 30}, class_weight ∈ {balanced, ninguno}; n-gramas y `min_df` en las familias de texto.

## Resultados

Mejor configuración por familia (Macro-F1 de validación cruzada) y su medición en dev:

| Familia | Mejor configuración | CV | Dev (IC 95%) | Dif. contra referencia (IC 95%) |
| --- | --- | --- | --- | --- |
| Referencia (palabras 1–2, min_df 2, C = 1) | | 0,556 | 0,5556 (0,5361 a 0,5755) | |
| Palabras | (1,3), min_df 2, C = 1, balanced | 0,560 | 0,5576 (0,5375 a 0,5778) | +0,0019 (−0,0049 a +0,0085) |
| Caracteres | (2,5), min_df 3, C = 1, balanced | 0,580 | 0,5678 (0,5482 a 0,5869) | +0,0122 (−0,0062 a +0,0319) |
| 24 variables | C = 1, balanced | 0,542 | 0,5085 (0,4888 a 0,5286) | −0,0472 (−0,0683 a −0,0246) |

## Lectura

- **Optimizar casi no cambia nada.** La configuración original ya estaba cerca del techo. Con caracteres el Macro-F1 sube 0,012 y con palabras 0,002, y los intervalos pareados incluyen el cero. Por variedad pasa lo mismo.
- **C es lo que más mueve**, sobre todo en los extremos (sin balancear y con C muy bajo, el modelo cae al piso de 0,19). Los n-gramas y `min_df` casi no importan.
- Las mejores configuraciones difieren menos que la desviación entre pliegues (0,01 a 0,02), así que "la mejor" es una elección entre empates prácticos.
- El modelo de 24 variables queda con la misma configuración que ya tenía y sigue por debajo de los de texto.
- Para la tesis: el piso clásico queda en unos 0,56 a 0,57 de Macro-F1. No hay una versión afinada mucho mejor que cambie la comparación con el LLM.

## Límites

- Malla acotada; no se probaron otros clasificadores ni otros pesos de clase.
- Todo se mide en dev, porque no existe partición de prueba.
- Es el baseline clásico, no el LLM.
