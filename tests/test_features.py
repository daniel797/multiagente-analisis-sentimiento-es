import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts" / "features"))
import feature_utils as F  # noqa: E402


def test_catalogo_y_extraccion_tienen_las_mismas_columnas():
    cat = F.catalogo_features()
    assert list(cat.feature) == F.NOMBRES
    assert set(F.extraer_features("hola")) == set(F.NOMBRES)
    assert cat.feature.is_unique


def test_extraer_features_cuenta_senales_basicas():
    f = F.extraer_features("@ana Qué bueeeno!!! JAJAJA no no puedo :) https://t.co/x #tema")
    assert f["n_exclamaciones"] == 3
    assert f["n_menciones"] == 1 and f["empieza_con_mencion"] == 1
    assert f["n_hashtags"] == 1 and f["tiene_url"] == 1
    assert f["n_alargamientos"] >= 1
    assert f["n_risas"] >= 1
    assert f["n_negaciones"] == 2
    assert f["n_emoticonos"] == 1


def test_extraer_features_con_texto_vacio_no_falla():
    f = F.extraer_features("")
    assert f["n_caracteres"] == 0 and f["n_palabras"] == 0 and f["n_oraciones"] == 1


def test_lexico_ignora_acentos():
    assert F.extraer_features("Qué increíble, gracias")["n_lex_positivo"] == 2


def test_normalizar_para_llm_anonimiza_pero_conserva_senales():
    t = F.normalizar_para_llm("@juan  mira  https://t.co/xx  ¡QUÉ BUENOOO! 😂")
    assert t == "@USUARIO mira URL ¡QUÉ BUENOOO! 😂"


def test_pistas_percepcion_usa_marcador_risa_y_no_sarcasmo():
    p = F.pistas_percepcion("jajaja no no sé")
    assert set(p) == {"diminutivo_afectivo", "doble_negacion", "code_switching", "marcador_risa"}
    assert p["marcador_risa"] and p["doble_negacion"]


def test_correlaciones_altas_detecta_columnas_redundantes():
    m = pd.DataFrame({"a": range(50), "b": [x * 2 for x in range(50)], "c": [(x * 7) % 5 for x in range(50)]})
    r = F.correlaciones_altas(m, umbral=0.95)
    assert ((r.feature_a == "a") & (r.feature_b == "b")).any()
