"""Léxico de polaridad en español, más amplio que el mínimo de feature_utils, y rasgos con alcance de negación."""
from __future__ import annotations
import re, unicodedata

def _sa(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")

_POS = """
bueno bien mejor excelente genial fantastico maravilloso estupendo magnifico espectacular increible perfecto impecable
feliz alegre alegria contento contenta satisfecho encantado ilusion ilusionado emocionado entusiasmo entusiasta
amor amar querer cariño carino adorar encantar gustar disfrutar fascinar enamorar enamorado apasionar
lindo hermoso bonito guapo precioso bello encantador agradable simpatico amable carismatico atractivo
gracias agradecer agradecido agradecimiento felicitar felicidades enhorabuena aplaudir celebrar celebracion festejar fiesta
exito triunfo triunfar ganar ganador victoria campeon logro lograr conseguir avance avanzar progreso progresar mejora mejorar
orgullo orgulloso admirar admiracion respeto respetar apoyar apoyo ayuda ayudar solidaridad solidario generoso
divertido divertir diversion gracioso risa reir sonreir sonrisa disfrute
esperanza esperanzador optimista optimismo confianza confiar seguro tranquilo paz tranquilidad calma
bendecir bendicion bendito dios_bendiga gloria glorioso honor honrar noble valiente heroe heroico
inteligente brillante talentoso capaz eficaz eficiente competente honesto honrado honestidad transparente justo justicia
libre libertad democracia progresista innovador innovacion creativo exitoso prospero prosperidad crecimiento
recomendar recomendable favorito preferido ideal util ventaja beneficio beneficiar favorecer favorable positivo
saludable salud sano sanar curar recuperar recuperacion fortalecer fuerte fortaleza valioso valor
chevere bacan chido guay mola molar tuanis genial brutal buenisimo buenazo crack chulo chulada
buenisima delicioso rico sabroso suculento
unir union unidad armonia amistad amigo amiga hermano hermandad familia abrazo abrazar besar beso
bienvenido gozar gozo placer paraiso milagro
"""
_NEG = """
malo mal peor pesimo horrible terrible espantoso atroz desastre desastroso fatal lamentable penoso deplorable
triste tristeza pena penar llorar llanto lagrima sufrir sufrimiento dolor doler doloroso angustia angustiar desesperar desesperacion
odio odiar asco asqueroso repugnante detestar aborrecer rabia rabioso furioso furia enojo enojar enojado molestar molesto molestia
miedo temer temor terror aterrar asustar susto preocupar preocupacion preocupante peligro peligroso amenaza amenazar riesgo
feo fea horrendo repulsivo desagradable antipatico grosero maleducado insoportable pesado aburrido aburrir aburrimiento
cansado cansar harto hartar hartazgo agotar agotado fastidiar fastidio fastidioso
problema crisis caos caotico conflicto guerra violencia violento agresion agredir atacar ataque matar muerte morir muerto asesinar asesinato crimen criminal delincuente delincuencia
corrupto corrupcion corromper ladron robar robo estafa estafar fraude fraudulento mentir mentira mentiroso engañar engano engaño traicionar traicion traidor
culpa culpable culpar vergüenza verguenza vergonzoso lastima lastimar daño dano danar perjudicar perjuicio injusto injusticia
inutil incompetente incapaz ineficaz torpe estupido idiota imbecil tonto tonteria ridiculo patetico mediocre basura porqueria
fracaso fracasar perder perdida derrota derrotar fallar falla error equivocar decepcion decepcionar decepcionante desilusion frustracion frustrar
quejar queja protestar reclamar denunciar denuncia rechazar rechazo critica criticar insultar insulto ofensa ofender
pobre pobreza miseria hambre hambruna desempleo desempleado despedir despido quiebra quebrar deuda
enfermo enfermedad enfermar epidemia tragedia tragico catastrofe catastrofico accidente herido herir lesion
mierda carajo joder jodido jodar maldito maldecir puto cabron pendejo pendejada chungo cutre horroroso
soledad abandonar abandono olvidar ignorar desprecio despreciar humillar humillacion
lento tardar demora retraso retrasar cancelar suspender prohibir prohibicion censura censurar
negativo negar fatiga estres estresar ansiedad depresion deprimir deprimente
"""
POS_LEMAS = {_sa(w) for w in _POS.split() if "_" not in w}
NEG_LEMAS = {_sa(w) for w in _NEG.split() if "_" not in w}
POS_LEMAS -= NEG_LEMAS; NEG_LEMAS -= POS_LEMAS

NEGADORES = {"no", "nunca", "jamas", "ni", "tampoco", "sin", "nada", "nadie", "ningun", "ninguno", "ninguna"}
INTENSIF = {"muy", "super", "tan", "demasiado", "bastante", "totalmente", "absolutamente", "realmente", "mas", "tanto", "mucho"}
EMOJI_POS = set("😀😃😄😁😆😊🙂😍😘🥰😎👍👏🙌💪❤💕💖💗💙💚💛🎉🎊✨🌟😂🤣😅🙏☺😇🥳💯🔥")
EMOJI_NEG = set("😢😭😞😔😟😠😡🤬😤😩😫😖😣👎💔😱😰😨🤢🤮😒🙄😑😕☹🖕💩")
_EMOTICON_POS = re.compile(r"(?<!\w)[:;=][\-o]?[\)Dd\]]|<3|\^\^")
_EMOTICON_NEG = re.compile(r"(?<!\w)[:;=][\-o]?[\(\[/\\]|:'\(")

NOMBRES_RASGOS = [
    "n_lex2_positivo", "n_lex2_negativo", "n_lex2_pos_negada", "n_lex2_neg_negada",
    "n_lex2_saldo", "n_lex2_intensificado", "n_emoji_emoticono_pos", "n_emoji_emoticono_neg",
]

def rasgos(doc, texto: str) -> dict:
    """Rasgos del léxico amplio para un tuit procesado con spaCy (doc) y su texto original."""
    toks = [t for t in doc if not t.is_space and not t.is_punct]
    sa = [_sa(t.text) for t in toks]
    lem = [_sa(t.lemma_) for t in toks]
    pos = neg = pos_n = neg_n = inten = 0
    for i in range(len(toks)):
        esp = lem[i] in POS_LEMAS or sa[i] in POS_LEMAS
        esn = lem[i] in NEG_LEMAS or sa[i] in NEG_LEMAS
        if not (esp or esn):
            continue
        negada = any(w in NEGADORES for w in sa[max(0, i - 3):i])
        if i > 0 and sa[i - 1] in INTENSIF:
            inten += 1
        if esp:
            if negada: pos_n += 1
            else: pos += 1
        else:
            if negada: neg_n += 1
            else: neg += 1
    ef_pos, ef_neg = pos + neg_n, neg + pos_n
    ep = sum(c in EMOJI_POS for c in texto) + len(_EMOTICON_POS.findall(texto))
    en = sum(c in EMOJI_NEG for c in texto) + len(_EMOTICON_NEG.findall(texto))
    return {"n_lex2_positivo": pos, "n_lex2_negativo": neg, "n_lex2_pos_negada": pos_n, "n_lex2_neg_negada": neg_n,
            "n_lex2_saldo": ef_pos - ef_neg, "n_lex2_intensificado": inten, "n_emoji_emoticono_pos": ep, "n_emoji_emoticono_neg": en}
