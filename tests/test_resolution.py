"""
Prueba del cálculo de confianza por acuerdo (self-consistency).
Es la única pieza de agents/resolution.py que no depende de llamar al LLM,
así que es lo único que se puede probar en esta etapa.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from agents.resolution import calcular_confianza_por_acuerdo  # noqa: E402


def test_acuerdo_total():
    etiqueta, confianza = calcular_confianza_por_acuerdo(["POS", "POS", "POS"])
    assert etiqueta == "POS"
    assert confianza == 1.0


def test_acuerdo_parcial():
    etiqueta, confianza = calcular_confianza_por_acuerdo(["POS", "POS", "NEU", "NEG", "POS"])
    assert etiqueta == "POS"
    assert confianza == 3 / 5


def test_lista_vacia_lanza_error():
    try:
        calcular_confianza_por_acuerdo([])
        assert False, "Debería haber lanzado ValueError"
    except ValueError:
        pass
