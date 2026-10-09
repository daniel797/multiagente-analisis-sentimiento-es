"""Pruebas de las funciones del EDA con datos mínimos sintéticos (no usan el corpus real)."""
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts" / "eda"))
import eda_utils as E  # noqa: E402


def _escribir(tmp_path, split, variedad, filas):
    carpeta = tmp_path / split
    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / f"{variedad}.tsv").write_text("\n".join("\t".join(f) for f in filas) + "\n", encoding="utf-8")


def test_cargar_corpus_mapea_etiquetas_y_tolera_comillas(tmp_path):
    _escribir(tmp_path, "train", "pe", [("1", 'Dijo "hola" y se fue', "P"), ("2", "nada especial", "NEU")])
    _escribir(tmp_path, "dev", "pe", [("3", "qué mal día", "N")])
    c = E.cargar_corpus(tmp_path)
    assert len(c) == 3
    assert set(c["etiqueta"]) == {"POS", "NEU", "NEG"}
    assert set(c["variedad"]) == {"PE"}
    assert c.loc[c.id == "1", "texto"].iloc[0] == 'Dijo "hola" y se fue'


def test_cargar_corpus_falla_si_falta_carpeta(tmp_path):
    try:
        E.cargar_corpus(tmp_path)
        assert False
    except FileNotFoundError:
        pass


def test_anonimizar_oculta_usuarios():
    assert E.anonimizar("hola @Maria_G y @pepe1") == "hola @USUARIO y @USUARIO"


def test_reporte_calidad_detecta_conflictos_y_solapamiento():
    c = pd.DataFrame({
        "id": ["1", "2", "3"], "variedad": ["ES"] * 3, "split": ["train", "dev", "train"],
        "texto": ["Mismo texto @a", "mismo texto", "otro"], "etiqueta": ["POS", "NEU", "NEG"],
    })
    r = E.reporte_calidad(c).set_index("chequeo")["casos"]
    assert r["Textos repetidos con etiquetas distintas (textos)"] == 1
    assert r["Textos presentes en train y dev a la vez"] == 1


def test_baseline_mayoritario_usa_clase_de_train():
    c = pd.DataFrame({
        "id": list("abcdef"), "variedad": ["PE"] * 6,
        "split": ["train"] * 4 + ["dev"] * 2,
        "texto": list("abcdef"), "etiqueta": ["NEU", "NEU", "NEU", "POS", "NEU", "POS"],
    })
    b = E.baseline_mayoritario(c).iloc[0]
    assert b["clase_mayoritaria_train"] == "NEU"
    assert b["accuracy_dev"] == 0.5


def test_agregar_fecha_snowflake():
    d = E.agregar_fecha(pd.DataFrame({"id": ["775087224857567232"], "variedad": ["PE"]}))
    assert str(d["fecha"].iloc[0].date()) == "2016-09-11"


def test_tokens_quitan_menciones_urls_y_stopwords():
    assert E.tokens("@user gracias por la ayuda https://t.co/x") == ["gracias", "ayuda"]


def test_cramers_v_es_cero_si_no_hay_asociacion():
    chi2, p, v = E.cramers_v(pd.DataFrame([[10, 10], [10, 10]]))
    assert v == 0 and p == 1
