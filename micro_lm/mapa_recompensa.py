"""Mapa de la recompensa: cual es el umbral que REALMENTE gobierna, y donde el optimo no es un extremo.

Motivo (2026-08-30). `ENMIENDA_RECOMPENSA_F.md` dejo escrita la regla: «cuando un chequeo dice que el
optimo de la perdida es un extremo, eso es un defecto y hay que cerrarlo antes de correr». El chequeo
que la enmienda propone verifica c* > 0 POR MUESTRA. Este script lo extiende a las tres politicas
constantes que compiten, porque la prediccion del 29 a la noche uso el umbral GLOBAL (0,657) para leer
el resultado de `f23_s3`, y hay que decidir cual de los dos umbrales aplica.

No entrena nada, no toca la GPU, no lee checkpoints. Es aritmetica sobre la misma formula que
`entrenar.py::_recompensa`, y lo primero que hace es verificar que sea la misma.
"""

import numpy as np

PI = 0.4065        # fraccion de preguntas sin respuesta en el banco (p_nose=0,4 medido)
PISO = 0.4065      # exactitud global trivial = contestar NOSE siempre


# --- las tres politicas constantes, en forma cerrada -------------------------------------------
# Por muestra CON respuesta:  R = q*(-F) + (1-q)*(c - (1-c)*M)
# Por muestra SIN respuesta:  R = q*(+L) + (1-q)*(-M)

def r_mudo(L, M, F):
    """q=1 en todas. Es el atractor medido el 29-ago (exactitud global 0,4065 clavada)."""
    return PI * L - (1 - PI) * F


def r_locuaz(c, L, M, F):
    """q=0 en todas. Colapsa a una recta en c: (1-pi)(1+M)c - M."""
    return (1 - PI) * (1 + M) * c - M


def r_oraculo(c, L, M, F):
    """q=1 donde no hay respuesta, q=0 donde si. Es la politica que el proyecto quiere."""
    return PI * L + (1 - PI) * ((1 + M) * c - M)


def umbral_por_muestra(L, M, F):
    """c* : en una pregunta CON respuesta conviene contestar si c > c*.  (M-F)/(1+M)."""
    return (M - F) / (1 + M)


def umbral_global(L, M, F):
    """c donde el locuaz alcanza al mudo. Es el que gobierna a un modelo que NO distingue."""
    return (r_mudo(L, M, F) + M) / ((1 - PI) * (1 + M))


def umbral_oraculo(L, M, F):
    """c donde el oraculo alcanza al mudo. Se demuestra abajo que es identico a c*."""
    return (M - F) / (1 + M)


# --- compuerta: la formula de acá es la de entrenar.py ------------------------------------------

def _verificar_identidad():
    """Reproduce `_recompensa` con numpy sobre un lote sintetico y compara con la forma cerrada.

    Puede fallar: si las dos difieren, todo lo que sigue es sobre una formula que no se entrena.
    """
    rng = np.random.default_rng(20260830)
    L, M, F = 0.5, 0.5, 0.2
    n = 200000

    es_nose = (rng.random(n) < PI).astype(np.float64)
    hay = 1.0 - es_nose
    c_muestra = rng.random(n) * hay          # c=0 donde tgt es NOSE, igual que el codigo
    q = rng.random(n)

    r_hay = q * (-F) + (1 - q) * (c_muestra - (1 - c_muestra) * M)
    r_no = q * L + (1 - q) * (-M)
    directo = (hay * r_hay + es_nose * r_no).mean()

    # forma cerrada con q y c constantes no aplica aca (son aleatorios); se verifica el caso q fijo
    for qf in (0.0, 1.0):
        cf = 0.37
        r_dir = (hay * (qf * (-F) + (1 - qf) * (cf - (1 - cf) * M))
                 + es_nose * (qf * L + (1 - qf) * (-M))).mean()
        pi_emp = es_nose.mean()
        r_cer = (1 - pi_emp) * (qf * (-F) + (1 - qf) * (cf * (1 + M) - M)) + pi_emp * (qf * L - (1 - qf) * M)
        assert abs(r_dir - r_cer) < 1e-12, f"forma cerrada != directa en q={qf}"

    # y la identidad que el script afirma: ventaja del oraculo sobre el mudo cruza cero en c*
    for (l, m, f) in ((0.5, 0.5, 0.2), (0.0, 0.2, 0.1), (1.0, 1.0, 0.3), (0.5, 0.5, 1.5)):
        cs = umbral_por_muestra(l, m, f)
        d = r_oraculo(cs, l, m, f) - r_mudo(l, m, f)
        assert abs(d) < 1e-12, f"el cruce oraculo-mudo no esta en c* para L={l} M={m} F={f}: {d}"

    return directo


# --- barrido ------------------------------------------------------------------------------------

def barrer():
    print("=" * 92)
    print("MAPA DE LA RECOMPENSA   ·   pi =", PI, "  piso trivial =", PISO)
    print("=" * 92)
    _verificar_identidad()
    print("compuerta: forma cerrada identica a la directa, y el cruce oraculo-mudo cae en c*  [OK]\n")

    print("IDENTIDAD CLAVE (demostrada arriba, no supuesta):")
    print("   R_oraculo(c) - R_mudo = (1-pi) * [ (1+M)c - M + F ]  ->  cruza cero exactamente en c*")
    print("   o sea: al modelo que DISTINGUE le conviene hablar desde c*, no desde el umbral global.\n")

    filas = []
    for L in (0.0, 0.25, 0.5):
        for M in (0.2, 0.35, 0.5, 1.0):
            for F in (0.05, 0.1, 0.2, 0.3, 0.35):
                if F >= M:
                    continue                      # sin umbral por muestra: defecto, no se considera
                filas.append((L, M, F))

    print(f"{'L':>5} {'M':>5} {'F':>5} | {'c* muestra':>10} {'c global':>9} | "
          f"{'R_mudo':>8} {'R_orac(.35)':>11} | veredicto")
    print("-" * 92)

    aptas = []
    for (L, M, F) in filas:
        cs = umbral_por_muestra(L, M, F)
        cg = umbral_global(L, M, F)
        rm = r_mudo(L, M, F)
        ro = r_oraculo(0.35, L, M, F)             # 0,35 = RECUP medido de las mudas (0,30-0,40)

        # criterios, fijados aca:
        #  A · existe umbral por muestra          -> c* en (0, 1)
        #  B · alcanzable por una unidad muda     -> c* <= 0,30  (por debajo del RECUP medido)
        #  C · el mudo NO cobra premio neto       -> R_mudo <= 0  (L no subsidia el silencio)
        ok_a = 0.0 < cs < 1.0
        ok_b = cs <= 0.30
        ok_c = rm <= 0.0
        v = []
        if not ok_a: v.append("sin umbral")
        if not ok_b: v.append("c* fuera de alcance")
        if not ok_c: v.append(f"mudo cobra +{rm:.4f}")
        veredicto = "APTA" if not v else " · ".join(v)
        if not v:
            aptas.append((L, M, F, cs, cg, rm, ro))

        print(f"{L:5.2f} {M:5.2f} {F:5.2f} | {cs:10.4f} {cg:9.4f} | "
              f"{rm:8.4f} {ro:11.4f} | {veredicto}")

    print("-" * 92)
    print(f"\nCELDAS APTAS: {len(aptas)} de {len(filas)}\n")

    print("EL PUNTO QUE SE ESTA CORRIENDO HOY (L=0,5 M=0,5 F=0,2), en detalle:")
    L, M, F = 0.5, 0.5, 0.2
    cs, cg, rm = umbral_por_muestra(L, M, F), umbral_global(L, M, F), r_mudo(L, M, F)
    print(f"   c* por muestra   = {cs:.4f}   <- gobierna a un modelo que DISTINGUE ausencia de error")
    print(f"   c  global        = {cg:.4f}   <- gobierna a un modelo que NO distingue")
    print(f"   R_mudo           = {rm:+.4f}  <- POSITIVO: L le paga al silencio")
    print(f"   R_locuaz(c=0)    = {r_locuaz(0.0, L, M, F):+.4f}")
    print(f"   R_oraculo(0,35)  = {r_oraculo(0.35, L, M, F):+.4f}\n")

    print("LA MEJOR CELDA POR c* MAS BAJO CON R_mudo <= 0:")
    if aptas:
        mejor = min(aptas, key=lambda t: (t[3], t[5]))
        L, M, F, cs, cg, rm, ro = mejor
        print(f"   L={L:.2f}  M={M:.2f}  F={F:.2f}   c*={cs:.4f}   c_global={cg:.4f}   R_mudo={rm:+.4f}")
    else:
        print("   ninguna")


if __name__ == "__main__":
    barrer()
