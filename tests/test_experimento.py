import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.append(str(RAIZ / "scripts" / "experimento"))
import evaluacion as EV  # noqa: E402
import monoagente_salamandra as M  # noqa: E402
import variantes_datos as V  # noqa: E402


def test_variante_base_es_la_normalizacion_del_llm():
    assert V.aplicar("@ana mira https://t.co/x  hola", "base") == "@USUARIO mira URL hola"


def test_limpio_quita_tono_pero_conserva_la_enie():
    t = "Qué BUENO!!! Mañana celebramoooos 😀"
    assert V.aplicar(t, "limpio") == "que bueno!!! mañana celebramos"


def test_cada_componente_cambia_solo_lo_suyo():
    assert V.aplicar("Canción", "sin_tildes") == "Cancion"
    assert V.aplicar("sufriendoooo", "sin_alargamientos") == "sufriendo"
    assert V.aplicar("Hola", "minusculas") == "hola"
    assert V.aplicar("hola 😀", "sin_emojis") == "hola"
    assert V.aplicar("@ana hola https://t.co/x", "sin_usuario_url") == "hola"
    assert V.aplicar("Hola", "sin_tildes") == "Hola"  # no toca mayúsculas


def test_variante_desconocida_falla_con_mensaje_claro():
    try:
        V.aplicar("hola", "inventada")
    except ValueError as e:
        assert "inventada" in str(e)
    else:
        raise AssertionError("debía fallar")


def test_leer_etiqueta_acepta_json_texto_suelto_y_rechaza_basura():
    assert M.leer_etiqueta('{"label": "NEG"}') == "NEG"
    assert M.leer_etiqueta('Claro: {"label": "pos"} gracias') == "POS"
    assert M.leer_etiqueta("La etiqueta es NEU.") == "NEU"
    assert M.leer_etiqueta("no sé") is None
    assert M.leer_etiqueta("") is None


def test_el_prompt_solo_lleva_el_tuit_y_no_la_variedad():
    m = M.construir_mensajes("hola che")
    assert [x["role"] for x in m] == ["system", "user"]
    assert m[1]["content"] == "Tuit: hola che"
    for pais in ("Uruguay", "Perú", "México", "España", "Costa Rica"):
        assert pais.lower() not in M.PROMPT_SISTEMA.lower()


def test_reintenta_solo_las_salidas_ilegibles():
    class Raro:
        def __init__(self):
            self.llamadas = []

        def generar(self, lotes, temperatura, semilla, **_):
            self.llamadas.append((len(lotes), temperatura))
            return ["???" if len(self.llamadas) == 1 and i == 0 else '{"label": "POS"}' for i in range(len(lotes))]

    b = Raro()
    res = M.clasificar_lote(b, ["a", "b", "c"], 0.0, 1, 2, {})
    assert [r["pred"] for r in res] == ["POS", "POS", "POS"]
    assert [r["intentos"] for r in res] == [2, 1, 1]
    assert b.llamadas == [(3, 0.0), (1, 0.3)]


def test_flujo_completo_con_backend_de_reglas(tmp_path):
    datos = RAIZ / "data"
    if not (datos / "dev").exists():
        return
    args = M.argparse.Namespace(data_dir=str(datos), config=str(RAIZ / "configs" / "model_config.yaml"), split="dev",
                                variantes=["base", "limpio"], backend="stub", revision="", temperatura=0.0, lote=8,
                                limite=25, out=str(tmp_path))
    M.ejecutar(args)
    M.ejecutar(args)  # reanudar no duplica filas
    pred = pd.read_csv(tmp_path / "predicciones_llm.csv", dtype=str)
    assert len(pred) == 2 * 25 and not pred.duplicated(["id", "variante"]).any()
    M.resumir(M.argparse.Namespace(out=str(tmp_path), n_boot=50))
    assert (tmp_path / "tablas" / "diferencias_llm_pareadas.csv").exists()


def test_metricas_y_diferencia_pareada_basicas():
    y = np.array(["POS", "NEG", "NEU"] * 40)
    m = EV.metricas(y, y, n_boot=50)
    assert m["macro_f1"] == 1.0 and m["ic_inf"] <= m["macro_f1"] <= m["ic_sup"]
    var = pd.Series(["CR"] * 60 + ["ES"] * 60)
    filas = EV.comparar_con_base(y, y, y, var, "igual", n_boot=50)
    assert {f["ambito"] for f in filas} == {"Todas", "CR", "ES"}
    assert all(f["predicciones_distintas"] == 0 and f["conclusion"] == "no se distingue" for f in filas)
