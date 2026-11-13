# Exploración de alternativas (solo validación cruzada sobre train)

Scripts exploratorios usados antes de fijar M1–M3. Ninguno toca dev. Los resultados están en `docs/validacion/tablas/exploracion_alternativas_cv.csv`. Se ejecutan desde la raíz del repo; los de spaCy necesitan `pip install spacy` y `es_core_news_md` 3.8.0, y `prep_spacy.py` guarda sus rasgos en `/tmp/spacy_feats.pkl` (las rutas `/tmp/explora2.py` y `/tmp/spacy_feats.pkl` hay que ajustarlas a donde copies los archivos).
