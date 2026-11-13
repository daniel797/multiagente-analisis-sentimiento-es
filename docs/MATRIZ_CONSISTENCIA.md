# Matriz de consistencia

Tesis: evaluación de una arquitectura multiagente basada en LLMs para el análisis de sentimientos en variedades del español. Se compara un monoagente con un multiagente que usan el mismo modelo (Salamandra-7B-Instruct, sin entrenamiento adicional) sobre tuits de TASS 2020 de cinco variedades: Costa Rica (CR), España (ES), México (MX), Perú (PE) y Uruguay (UY).

| Problema | Objetivo | Hipótesis | Variables | Indicadores |
| --- | --- | --- | --- | --- |
| **General.** ¿En qué medida una arquitectura multiagente basada en un mismo LLM mejora el análisis de sentimientos de tuits en español frente a un monoagente, en cinco variedades? | **General.** Diseñar una arquitectura multiagente que mejore el análisis de sentimientos en variedades del español respecto a un monoagente que usa el mismo modelo base. | **General.** Con el mismo modelo y sin entrenamiento adicional, el multiagente obtiene un Macro-F1 mayor que el monoagente: la diferencia pareada es de al menos 0,03 y su intervalo de confianza no incluye el cero. | Independiente: arquitectura (monoagente, multiagente).<br>Dependiente: Macro-F1 de la clasificación en POS, NEG y NEU.<br>Fijos: modelo base (revisión fija, 4 bits), decodificación determinista y los mismos tuits. | Macro-F1 con IC 95% por bootstrap (2.000 remuestreos).<br>Diferencia pareada multiagente menos monoagente.<br>Exactitud (secundaria) y porcentaje de salidas inválidas. |
| **Específico 1.** ¿El rendimiento de cada arquitectura cambia según la variedad del español? | **Específico 1.** Lograr determinar si el Macro-F1 de cada arquitectura cambia entre variedades. | **Específica 1.** El Macro-F1 difiere entre variedades, y la ventaja del multiagente tiende a ser mayor donde el monoagente rinde peor. Con 390 a 581 tuits por variedad solo se detectan diferencias de 5 a 6 puntos, así que esta hipótesis se evalúa de forma descriptiva. | Independientes: arquitectura; variedad (CR, ES, MX, PE, UY).<br>Dependiente: Macro-F1 por variedad. | Macro-F1 por variedad con IC 95%.<br>Diferencia pareada por variedad, con corrección de Bonferroni para 5 comparaciones (p < 0,01). |
| **Específico 2.** ¿Qué aporta cada agente del sistema multiagente, en particular el de Percepción? | **Específico 2.** Lograr identificar el aporte de cada agente al retirarlo del sistema, uno por vez. | **Específica 2.** Incluir el agente de Percepción aumenta el Macro-F1 respecto al mismo sistema sin él. | Independiente: presencia del agente de Percepción (con, sin).<br>Dependiente: Macro-F1. | Diferencia pareada de Macro-F1 con y sin el agente, con IC 95%.<br>F1 por clase en cada configuración. |
| **Específico 3.** ¿Cada arquitectura detecta por igual las tres clases, o se inclina hacia alguna? | **Específico 3.** Lograr caracterizar el desempeño por clase y la distribución de las clases predichas en ambas arquitecturas. | **Específica 3.** El multiagente reduce la inclinación del monoagente hacia NEU y mejora el recobrado de NEG. | Independiente: arquitectura.<br>Dependientes: recobrado y F1 por clase; distribución de clases predichas. | Precisión, recobrado y F1 por clase.<br>Porcentaje de tuits predichos como NEG, NEU y POS frente al real.<br>Matriz de confusión. |
| **Específico 4.** ¿Qué parte del texto sostiene la predicción de cada arquitectura? | **Específico 4.** Lograr establecer cuánto depende cada arquitectura de la clase de palabras, reduciendo el tuit a una sola: solo artículos, solo sustantivos, solo adjetivos y verbos. | **Específica 4.** Con solo artículos el Macro-F1 cae hasta el piso de la clase mayoritaria (0,19). Con solo sustantivos pierde más que con solo adjetivos y verbos, que se mantiene cerca de la base. | Independientes: variante de los datos (base, solo artículos, solo sustantivos, solo adjetivos y verbos); arquitectura.<br>Dependiente: Macro-F1. | Macro-F1 por variante con IC 95%.<br>Diferencia pareada contra la base.<br>Predicciones que cambian, a favor y en contra.<br>Comparación con el piso de 0,19. |

Antecedente del específico 4: con el texto limpio y con las variantes de un solo componente (minúsculas, sin tildes, sin alargamientos, sin emojis, sin `@USUARIO` y `URL`), ni TF-IDF ni Salamandra cambian su Macro-F1 (diferencias de +0,001 a +0,003, con intervalos que incluyen el cero). Detalle en `docs/experimento/`.

## Hipótesis estadísticas

Para la hipótesis general, con la diferencia pareada d = Macro-F1 del multiagente − Macro-F1 del monoagente:

- H0: d ≤ 0.
- H1: d > 0, y se exige además d ≥ 0,03 (un efecto menor no se distingue del ruido con este tamaño de muestra).

Se rechaza H0 si el límite inferior del IC 95% por bootstrap pareado (2.000 remuestreos) es mayor que 0. Por variedad se aplica Bonferroni para 5 comparaciones (p < 0,01).

## Diseño metodológico

| Aspecto | Descripción |
| --- | --- |
| Tipo y nivel | Aplicada, cuantitativa, comparativa. Nivel explicativo. |
| Diseño | Experimental con medidas pareadas: los mismos tuits se clasifican con cada arquitectura. Solo inferencia, sin entrenamiento. |
| Población y muestra | TASS 2020: 7.267 tuits de cinco variedades. Train 4.802 (para el baseline y las variables) y dev 2.465 (donde se mide todo). No hay partición de prueba. |
| Modelo | Salamandra-7B-Instruct, revisión fija, 4 bits, decodificación determinista. |
| Instrumentos | Prompts de cada agente, scripts de `scripts/experimento/` y cuadernos 04 y 05. |
| Medición | Macro-F1 (principal), exactitud, F1 por clase, porcentaje de salidas inválidas. |
| Análisis | Bootstrap (IC 95%), bootstrap pareado de diferencias, corrección de Bonferroni por variedad. |
| Poder | Con todo dev, la diferencia mínima detectable es de unos 2,5 puntos; por variedad, de 5,2 a 6,3. |
