"""MICRO-LM · entrenamiento. Corre igual en CPU (lento) y en GPU de Colab.

    python entrenar.py --nivel 1 --pasos 20000

Metrica: exactitud del UNICO token de respuesta, desagregada en vigente / anterior, mas la tasa de
abstencion (cuando el modelo elige NOSE). Nada de jueces ni parsers: la respuesta es un token y se
compara con el token esperado.

Lo que este script hereda de lo medido en el brazo interno, y esta escrito a proposito:
  · el archivo se inyecta TEMPRANO (bloque 0);
  · la clave archivada lleva sello de orden co-entrenado;
  · el balance de preguntas es un parametro (`--p-vieja`), porque E-I3d mostro que si casi todas las
    preguntas son por la version vigente, el modelo aprende el ATAJO de la recencia y nunca aprende a
    ordenar. Por defecto 0,35, no 0,05.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jax
import jax.numpy as jnp
import optax

# --- la aritmetica no puede depender de que acelerador toco (2026-08-23) ------------------------
# Desde hoy el rotador pide TPU v5e1 cuando no hay T4, asi que una MISMA corrida puede hacer un
# tramo en GPU y el siguiente en TPU. Y por defecto no son la misma cuenta: en TPU jax resuelve los
# matmul de float32 pasando por bf16, mientras que en la T4 (que es Turing, sin TF32) los hace en
# fp32 de verdad. Sin fijar esto, «w3_s1 corrio en TPU» seria una variable escondida capaz de
# explicar una diferencia entre celdas — exactamente el confound que el proyecto ya evita
# registrando `hw` en el JSON, solo que registrado no alcanza cuando pasa DENTRO de una corrida.
#
# `highest` fuerza fp32 en las dos. En T4 no cambia nada (ya era su default, verificado bit a bit);
# en TPU cuesta velocidad —hace tres pasadas bf16— y esa es justamente la moneda con la que se paga
# que el numero signifique lo mismo.
jax.config.update("jax_default_matmul_precision", "highest")

import datos as DAT
import idioma as I
import modelo as M

NOSE = I.STOI["NOSE"]

# Donde entra la lectura del archivo dentro del bloque 0 (ver `modelo.tronco`). Es el eje del
# experimento del 22-ago. Se fija UNA vez en `main()`, antes de que jax tracee nada, y queda horneado
# en los grafos compilados —es un `if` sobre un string de Python, no un valor de array—. No puede
# cambiar en el medio de una corrida, y por eso `main` lo deja asentado en el JSON de config: si
# alguna vez una corrida sale con el valor que no era, el JSON es donde se ve.
_DONDE = "pre"
_ABST = "token"
_BLANCO = "ausencia"      # A5: blanco de la BCE de la cabeza — «ausencia» o «error»
_PERDIDA_CABEZA = "bce"   # 2026-08-29: forma de la perdida de la cabeza — bce/balance/ranking


def evaluar(params, rng, n=8, B=64, nivel=4, p_vieja=0.35, p_nose=0.0, pred_fn=None):
    """Devuelve un dict de metricas. Las dos caras de la abstencion van SEPARADAS:

      `nose`         acierta NOSE cuando la respuesta no esta en el archivo (lo que se quiere);
      `falsa_abst`   dice NOSE cuando la respuesta SI estaba (el costo de conseguirlo).

    Un modelo que contesta NOSE a todo tendria `nose` = 1,000, y por eso la segunda no es opcional.
    """
    col = {k: [] for k in ("vigente", "anterior", "nose", "nose_ent", "nose_rel",
                           "falsa_abst", "abstencion")}
    for _ in range(n):
        ses, cortes, turnos, mask, cons, pos, tgt, tipo = DAT.lote(
            rng, B, nivel=nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja, p_nose=p_nose)
        fn = pred_fn or predecir
        pred = np.array(fn(params, jnp.array(ses), jnp.array(cortes), jnp.array(turnos),
                           jnp.array(mask), jnp.array(cons), jnp.array(pos)))
        ok = pred == tgt
        sub = lambda m: ok[m].mean() if m.any() else np.nan
        col["vigente"].append(sub(tipo == 0))
        col["anterior"].append(sub(tipo == 1))
        col["nose"].append(sub(tipo >= 2))
        col["nose_ent"].append(sub(tipo == 2))
        col["nose_rel"].append(sub(tipo == 3))
        hay = tipo < 2
        col["falsa_abst"].append((pred[hay] == NOSE).mean() if hay.any() else np.nan)
        col["abstencion"].append((pred == NOSE).mean())
    return {k: float(np.nanmean(v)) for k, v in col.items()}


def logits_de(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg = M.responder(params, archivo, turnos, cons, mask, donde=_DONDE)
    return jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]


@jax.jit
def predecir(params, ses, cortes, turnos, mask, cons, pos):
    return logits_de(params, ses, cortes, turnos, mask, cons, pos).argmax(-1)


def perdida(params, ses, cortes, turnos, mask, cons, pos, tgt):
    lg = logits_de(params, ses, cortes, turnos, mask, cons, pos)
    if _PERDIDA_CABEZA == "recompensa":
        return _recompensa(lg, tgt, q=jax.nn.softmax(lg, -1)[:, NOSE],
                           s=lg[:, NOSE] - jax.nn.logsumexp(
                               lg.at[:, NOSE].set(-1e9), axis=-1))
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tgt).mean()
    return ce, (lg.argmax(-1) == tgt).mean()


# --- RECOMPENSA ESPERADA (2026-08-29, `PREREG_RECOMPENSA.md`, SHA f1f7bb66) ---------------------
# Idea de Maxi: «premio para que los aciertos [...] lo incentiven a querer siempre dar su mejor
# esfuerzo en saber, y resta cuando miente o da una respuesta incorrecta».
#
# Lo que esto arregla, y es el fracaso medido el 29 con `balance` y `ranking`: aquellas dos tocan
# SOLO al vigilante, asi que la perdida del valor y la de la cabeza nunca se hablan y nada le dice al
# modelo que equivocarse sea PEOR que callarse. Les sacamos el pago al silencio y las seis unidades
# se fueron a inventar (exactitud 0,2361-0,3536 contra un piso de 0,4065, invento hasta 0,1966).
#
# Aca la perdida es UNA SOLA y mira el resultado final. Cuatro casos, con el acierto normalizado a 1:
#
#     +1     acerto la respuesta
#     +L     dijo NOSE y de verdad no estaba
#     -M     contesto y erro (inventar incluido)
#     -F     dijo NOSE pero la respuesta SI estaba
#
# El corte «si NOSE gana, abstenerse» no es derivable, asi que NO se decide: se usa la MASA `q` que el
# modelo ya le da a NOSE como probabilidad de abstenerse, y se escribe la recompensa ESPERADA. Con la
# interfaz `token` esa masa sale del propio softmax de vocabulario —sin cabeza, sin parametro nuevo,
# que es lo que la hace escalable—; con `cabeza` sale de sigmoid(a).
#
# Los pesos estan FIJADOS en el pre-registro (L=M=0,5, F=1,5) y salen de la condicion derivada alli:
#     F > [ pi*(L+M) + (1-pi)*((1-c)*M - c) ] / (1-pi)
# que en este banco (pi=0,4065) da un minimo de 1,185 en c=0. El 1,5 deja 27 % de margen.
# 2026-08-29 22:40 · ENMIENDA_RECOMPENSA_F.md. La primera corrida (F=1,5) dio abstencion 0,0000 en
# las 8 unidades: contestaron TODO. La causa es un error de MI derivacion, no del diseno: la condicion
# se dedujo sobre un q GLOBAL, y el modelo elige q POR MUESTRA. Con q por muestra, en una pregunta con
# respuesta conviene contestar si  c > c* = (M - F) / (1 + M),  asi que para que EXISTA un umbral hace
# falta  F < M.  Con F=1,5 y M=0,5 el umbral daba -0,667: nunca convenia callarse.
_REC_L = 0.5
_REC_M = 0.5
_REC_F = 0.2      # F < M  ->  umbral de confianza c* = (0.5-0.2)/1.5 = 0.200
# Peso de la CE del valor, que se SUMA a la recompensa y no la reemplaza. Es necesario, no cosmetico:
# en la recompensa el termino que empuja a acertar va multiplicado por (1-q), asi que un modelo que se
# calla (q->1) deja de recibir gradiente hacia la recuperacion y el atractor mudo volveria, ahora sin
# nada que lo saque. La CE no depende de q y mantiene el aprendizaje del valor siempre vivo.
_REC_CE = 1.0
# Termino de ORDEN sobre la decision de callarse (2026-08-31, `DISENO_RECOMPENSA_RANKING.md`).
# Por que hace falta, y es un defecto de la recompensa que se midio hoy y no estaba visto:
# `rec` es LINEAL en `q`, asi que si `q` resulta independiente de (hay, c), E[rec] depende de `q`
# SOLO a traves de su MEDIA. Dos modelos con la misma tasa de abstencion y particiones distintas de
# las preguntas valen exactamente lo mismo -> hay una VARIEDAD de optimos equivalentes y ningun
# gradiente que elija entre ellos. Medido el 31: las unidades `t*` se callan segun la RELACION
# preguntada (pureza 0,98, nulo 0,52), lo que no correlaciona ni con la ausencia (+0,0000 de
# ganancia), ni con la dificultad (-0,0200, al reves), ni con su confianza (+0,024 y -0,003, con
# signos opuestos entre semillas), y con el logit saturado entre -18,4 y +22,3.
#
# `ranking` (29-ago) ya premiaba discriminar —«toda constante da el PEOR valor, el unico modo de
# bajar la perdida es ordenar»— pero tocaba SOLO al vigilante y las seis unidades se fueron a
# inventar. `recompensa` arreglo el acople y perdio la discriminacion. Este termino es la sintesis:
# el acople de una y el orden de la otra.
#
# Va sobre el LOGIT y no sobre `q` a proposito: `q` esta saturado (86-99 % de la masa en 0 y 1) y la
# diferencia de dos probabilidades saturadas no tiene gradiente. El logit si lo tiene.
_REC_RANK = 0.0


def _orden_nose(s, es_nose, hay):
    """Sustituto del AUC por pares sobre el logit de `NOSE`: s_i (sin respuesta) > s_j (con).

    Toda constante da el mismo valor y es el PEOR alcanzable, asi que no hay tasa que cobrar: el
    unico modo de bajarlo es ORDENAR. Con un lote sin pares validos el denominador seria 0 y se
    protege con `maximum(...,1)`, igual que en `perdida_cabeza`; ahi el termino da 0 y no hay
    informacion de orden que aprender, que es lo correcto y no un caso a rellenar.
    """
    dif = s[:, None] - s[None, :]
    par = es_nose[:, None] * hay[None, :]
    return (jax.nn.softplus(-dif) * par).sum() / jnp.maximum(par.sum(), 1.0)


def _recompensa(lg, tgt, q, s=None):
    """Perdida = -E[recompensa]. `q` es la probabilidad de abstenerse, derivable.

    `s` es el logit crudo del que sale `q`, y sólo lo usa el término de orden. Se pasa aparte porque
    con la interfaz `token` sale del softmax de vocabulario y con `cabeza` de la salida binaria.
    """
    es_nose = (tgt == NOSE).astype(jnp.float32)
    hay = 1.0 - es_nose

    # P(acertar | contesta). Se usa la probabilidad NORMALIZADA sobre los valores, no el argmax: el
    # argmax no es derivable y ademas descarta la informacion de cuanta confianza tenia el modelo,
    # que es justamente lo que la recompensa esperada necesita para poder moverse.
    lg_v = lg.at[:, NOSE].set(-1e9)
    p_val = jax.nn.softmax(lg_v, -1)
    c = jnp.take_along_axis(p_val, tgt[:, None], axis=1)[:, 0] * hay   # 0 donde tgt es NOSE

    # con respuesta: q*(-F) + (1-q)*(c*1 + (1-c)*(-M))
    r_hay = q * (-_REC_F) + (1.0 - q) * (c - (1.0 - c) * _REC_M)
    # sin respuesta: q*(+L) + (1-q)*(-M)   — contestar cuando no hay respuesta es siempre error
    r_no = q * _REC_L + (1.0 - q) * (-_REC_M)

    rec = hay * r_hay + es_nose * r_no

    # CE del valor, SIEMPRE activa y sin depender de q (ver el comentario de _REC_CE). Se normaliza
    # por la fraccion con respuesta para que su escala no dependa de p_nose, igual que en las otras.
    ce = optax.softmax_cross_entropy_with_integer_labels(lg_v, tgt)
    ce = (ce * hay).sum() / jnp.maximum(hay.sum(), 1.0)

    perdida = -rec.mean() + _REC_CE * ce
    # El termino de orden es OPCIONAL y por defecto vale 0, asi que con `--rec-rank 0` (el default)
    # la perdida es BIT A BIT la de antes y las campanias anteriores siguen siendo reproducibles.
    if _REC_RANK != 0.0:
        perdida = perdida + _REC_RANK * _orden_nose(
            jnp.log(jnp.clip(q, 1e-7, 1 - 1e-7)) - jnp.log(jnp.clip(1 - q, 1e-7, 1 - 1e-7))
            if s is None else s, es_nose, hay)

    pred = jnp.where(q > 0.5, NOSE, lg_v.argmax(-1))
    return perdida, (pred == tgt).mean()


# --- cabeza de abstencion separada (2026-08-18, `PREREG_CABEZA_ABSTENCION.md`) -------------------
# `NOSE` deja de ser una entrada del softmax de vocabulario y pasa a tener su propia salida binaria.
# Las dos decisiones —«¿esta?» y «¿que valor?»— dejan de competir por la misma masa de probabilidad.

def _partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask, donde=_DONDE, abst=_ABST)
    lg = jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]
    a = jnp.take_along_axis(a, pos[:, None], axis=1)[:, 0]
    return lg, a


@jax.jit
def predecir_cabeza(params, ses, cortes, turnos, mask, cons, pos):
    lg, a = _partes(params, ses, cortes, turnos, mask, cons, pos)
    # `NOSE` se excluye del argmax de valores: con la cabeza aparte, dejarlo seria darle dos rutas a
    # la misma decision y el contraste con `token` dejaria de ser limpio.
    lg = lg.at[:, NOSE].set(-jnp.inf)
    return jnp.where(a > 0.0, NOSE, lg.argmax(-1))


def perdida_cabeza(params, ses, cortes, turnos, mask, cons, pos, tgt):
    lg, a = _partes(params, ses, cortes, turnos, mask, cons, pos)
    es_nose = (tgt == NOSE).astype(jnp.float32)

    # --- BLANCO DE LA CABEZA (2026-08-26, `ALTERNATIVAS_DETECCION_20260826.md` A5) --------------
    # `ausencia` es lo que se venia haciendo: la BCE aprende «¿hay respuesta?».
    # `error` cambia el blanco por «¿me voy a equivocar SI contesto?», que es lo que un detector de
    # alucinacion tiene que anticipar. Los dos no son lo mismo: el modelo puede no tener la respuesta
    # y aun asi acertar por otra via, y puede tenerla y errarle igual — que es exactamente el
    # `err_identidad` que la cabeza de hoy no ve porque su blanco no lo contiene.
    #
    # El blanco sale del ARGMAX del propio modelo, con `stop_gradient` explicito: la etiqueta es una
    # medicion del estado actual, no algo por donde deba fluir gradiente. Y por eso se MUEVE mientras
    # entrena, que es el riesgo declarado por adelantado — si colapsa al prior va a ser por esto, no
    # por falta de capacidad, igual que le paso al slot nulo el 25-ago.
    if _BLANCO == "error":
        lg_arg = jax.lax.stop_gradient(lg.at[:, NOSE].set(-1e9).argmax(-1))
        blanco = (lg_arg != tgt).astype(jnp.float32)      # con tgt==NOSE da 1 siempre, por definicion
    else:
        blanco = es_nose
    # --- PERDIDA DE LA CABEZA (2026-08-29, `PREREG_PERDIDA_CABEZA.md`, SHA 0f57609d) -------------
    # Idea de Maxi: «estudiar tiene que tener una retribucion mucho mayor que no estudiar [...] para
    # descubrir tiene que saber y eso lo lleva a saber».
    #
    # El informe del 29 midio que con `bce` la cabeza de 4 de 9 unidades colapsa al prior, y que ese
    # colapso NO es una meseta: la constante `logit E[b]` es el MINIMO de la BCE. El modelo se queda
    # ahi porque ahi esta el optimo de lo que se le pide. Peor todavia, cuanto menos recupera mas
    # facil le queda a la cabeza, porque el blanco se vuelve casi constante.
    #
    # Las dos condiciones nuevas le sacan a la cabeza el pago por la constante, con distinta fuerza:
    #
    #   `balance`  pesa cada muestra por el inverso de la frecuencia de su clase EN EL LOTE. La mejor
    #              constante pasa a valer log 2 sea cual sea el prior: cobrarlo deja de pagar. Los
    #              pesos se renormalizan a media 1 para no cambiar la escala frente a la CE del valor;
    #              sin eso, `balance` estaria ademas subiendo el peso relativo de la cabeza y el
    #              contraste mediria dos cosas a la vez.
    #
    #   `ranking`  sustituto del AUC por pares. TODA constante da el mismo valor y es el PEOR
    #              alcanzable, asi que no hay prior que cobrar en absoluto y el unico modo de bajar la
    #              perdida es ordenar. Es la version literal de la idea.
    #              Con un lote sin pares validos (todo blanco 1, que es lo que pasa al principio) el
    #              denominador seria 0: se protege con `maximum(...,1)`, y en ese caso la perdida da 0
    #              y la cabeza simplemente no recibe gradiente en ese paso. Es lo correcto — no hay
    #              informacion de orden que aprender — y no un caso que haya que rellenar.
    if _PERDIDA_CABEZA == "recompensa":
        # interfaz H del PREREG_RECOMPENSA: la misma recompensa, pero la probabilidad de abstenerse
        # sale de la cabeza binaria en vez del softmax de vocabulario. Es el CONTRASTE de la campania,
        # no la condicion principal — la principal no tiene cabeza justamente porque escala.
        return _recompensa(lg, tgt, q=jax.nn.sigmoid(a), s=a)
    if _PERDIDA_CABEZA == "balance":
        f1 = jnp.mean(blanco)                       # fraccion de la clase «me equivoco»
        f0 = 1.0 - f1
        w = blanco / jnp.maximum(f1, 1e-6) + (1.0 - blanco) / jnp.maximum(f0, 1e-6)
        w = w / jnp.maximum(jnp.mean(w), 1e-6)      # media 1: no cambia la escala de la perdida
        bce = (optax.sigmoid_binary_cross_entropy(a, blanco) * w).mean()
    elif _PERDIDA_CABEZA == "ranking":
        dif = a[:, None] - a[None, :]               # (B, B)
        par = blanco[:, None] * (1.0 - blanco)[None, :]   # i con blanco 1, j con blanco 0
        bce = (jax.nn.softplus(-dif) * par).sum() / jnp.maximum(par.sum(), 1.0)
    else:
        bce = optax.sigmoid_binary_cross_entropy(a, blanco).mean()
    lg_v = lg.at[:, NOSE].set(-1e9)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg_v, tgt)
    # la CE del valor sólo cuenta donde HAY respuesta, y se normaliza por esa fraccion para que la
    # escala de la perdida no dependa de `p_nose` (si no, subir p_nose baja el peso del valor y el
    # contraste entre condiciones mediria eso).
    hay = 1.0 - es_nose
    ce = (ce * hay).sum() / jnp.maximum(hay.sum(), 1.0)
    pred = jnp.where(a > 0.0, NOSE, lg_v.argmax(-1))
    return bce + ce, (pred == tgt).mean()


# --- mezcla escalonada por capacidad (2026-08-23, `DISENO_ESCALONADO.md`) ------------------------
# Idea de Maxi: «¿por que todo tiene que terminar a 4000? Que cada cosa termine cuando le conviene y
# el resto continue hasta su turno.» En vez de fijar la mezcla de tipos de pregunta una vez a mano,
# se muestrea cada tipo con probabilidad proporcional a su ERROR actual. Una capacidad ya resuelta
# tiene error ~0 y deja de gastar muestras SOLA; el presupuesto se va a las que faltan sin que nadie
# elija un umbral. Esto importa porque la palanca ya existia y estaba clavada: `p_vieja` esta en 0,35
# porque E-I3d la subio a mano de 0,05 para que el atajo de la recencia no ganara. Esto generaliza
# esa correccion.
TIPOS = ("vigente", "anterior", "nose")


def probs_de_pesos(w):
    """(w_vigente, w_anterior, w_nose) normalizados -> (p_vieja, p_nose).

    `datos.lote` no acepta tres pesos: acepta las dos palancas que ya tenia. La traduccion es exacta
    y biyectiva, porque los tipos se eligen en cascada —primero si la consulta tiene respuesta, y
    despues cual—:

        p(nose)     = p_nose
        p(anterior) = (1 - p_nose) * p_vieja
        p(vigente)  = (1 - p_nose) * (1 - p_vieja)

    Asi que la mezcla dinamica no necesita tocar `datos.py`, que es codigo compartido con las
    campanias ya cerradas.
    """
    v, an, no = (max(0.0, float(x)) for x in w)
    tot = v + an + no
    if tot <= 0:
        return 0.35, 0.0
    v, an, no = v / tot, an / tot, no / tot
    con_resp = v + an
    p_vieja = an / con_resp if con_resp > 1e-9 else 0.0
    return float(p_vieja), float(no)


def pesos_de_probs(p_vieja, p_nose):
    """La inversa de `probs_de_pesos`. Existe para que el registro de una corrida FIJA diga la
    mezcla que de verdad uso: si anotara los pesos de la EMA, el JSON de la condicion de control
    mostraria un reparto que nadie muestreo."""
    return [(1 - p_nose) * (1 - p_vieja), (1 - p_nose) * p_vieja, p_nose]


def pesos_de_ema(ema, piso):
    """error EMA por tipo -> pesos de muestreo, con piso.

    El PISO no es decoracion: sin el, una capacidad resuelta deja de muestrearse del todo y se
    olvida —olvido catastrofico dentro del propio entrenamiento—. Con piso queda viva a costo bajo.
    """
    piso = max(0.0, min(piso, 1.0 / len(TIPOS)))
    libre = 1.0 - piso * len(TIPOS)
    err = [max(0.0, float(ema[t])) for t in TIPOS]
    tot = sum(err)
    if tot <= 1e-9:                       # todo resuelto: reparto parejo dentro de lo que sobra
        return [piso + libre / len(TIPOS)] * len(TIPOS)
    return [piso + libre * e / tot for e in err]


def estado_mezcla_inicial():
    # El error arranca en 1,0 para los tres: sin evidencia todavia, la mezcla inicial es uniforme y
    # no privilegia a ninguno. `acum` suma pesos POR PASO para poder reconstruir despues la mezcla
    # promedio que la corrida termino usando, que es lo que necesita el control `fijo_promedio`.
    return {"ema": {t: 1.0 for t in TIPOS}, "acum": [0.0] * len(TIPOS), "pasos_acum": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nivel", type=int, default=1)
    ap.add_argument("--pasos", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--capas", type=int, default=6)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--p-vieja", type=float, default=0.35)
    ap.add_argument("--p-nose", type=float, default=0.0,
                    help="fraccion de preguntas SIN respuesta en el archivo (respuesta = NOSE)")
    ap.add_argument("--abst", default="token", choices=("token", "escala", "cabeza", "slot"),
                    help="como se decide la abstencion (PREREG_CABEZA_ABSTENCION.md). "
                         "token = NOSE es una entrada mas del softmax de vocabulario (lo de hoy); "
                         "escala = idem pero renormalizando el vector de NOSE a la norma media de "
                         "los tokens de valor al arrancar la fase; "
                         "cabeza = salida binaria separada, con NOSE excluido del softmax de valores")
    ap.add_argument("--perdida-cabeza", default="bce",
                    choices=("bce", "balance", "ranking", "recompensa"),
                    help="forma de la perdida de la cabeza (PREREG_PERDIDA_CABEZA.md, SHA 0f57609d). "
                         "`bce` es lo que se venia haciendo y tiene un minimo en la constante del "
                         "prior, que es el atractor mudo del 29-ago. `balance` pesa por el inverso "
                         "de la frecuencia de clase, con lo que la mejor constante vale log 2 sea "
                         "cual sea el prior. `ranking` es el sustituto del AUC por pares, donde toda "
                         "constante da el PEOR valor posible. `recompensa` (PREREG_RECOMPENSA.md, "
                         "SHA f1f7bb66) es otra cosa: reemplaza las DOS perdidas por una sola "
                         "recompensa esperada sobre el resultado final, y es la unica que "
                         "funciona tambien con --abst token, sin cabeza.")
    # Pesos de `recompensa`. Dejan de ser constantes de modulo el 2026-08-30 (`PREREG_RECOMPENSA_L`):
    # editarlos a mano no queda en la config del checkpoint, asi que la guarda de identidad no puede
    # ver el cambio y dos tramos con pesos distintos se leerian como una sola curva. Es la misma razon
    # por la que `blanco` y `perdida_cabeza` son flags y no constantes.
    ap.add_argument("--rec-l", type=float, default=0.5,
                    help="premio por decir NOSE cuando de verdad no estaba. OJO: es el subsidio al "
                         "atractor mudo — con L=0,5 una unidad que se calla SIEMPRE cobra +0,0845 de "
                         "recompensa neta, que es el piso trivial 0,4065 metido adentro de la "
                         "perdida como premio. Con L=0 el mudo cobra negativo y c* no cambia, porque "
                         "L no entra en (M-F)/(1+M). Ver `mapa_recompensa.py`.")
    ap.add_argument("--rec-m", type=float, default=0.5,
                    help="castigo por contestar y errar (inventar incluido).")
    ap.add_argument("--rec-f", type=float, default=0.2,
                    help="castigo por decir NOSE cuando la respuesta SI estaba. Hace falta F < M para "
                         "que exista un umbral por muestra; con F >= M nunca conviene callarse y el "
                         "optimo es el extremo locuaz (ENMIENDA_RECOMPENSA_F.md).")
    ap.add_argument("--rec-ce", type=float, default=1.0,
                    help="peso de la CE del valor, que se SUMA a la recompensa. No es cosmetico: sin "
                         "el, un modelo que se calla deja de recibir gradiente hacia la recuperacion.")
    ap.add_argument("--rec-rank", type=float, default=0.0,
                    help="peso del termino de ORDEN sobre el logit de NOSE (DISENO_RECOMPENSA_RANKING"
                         ".md). Arregla un defecto medido el 31-ago: la recompensa es LINEAL en q, "
                         "asi que si la decision resulta independiente de la evidencia la perdida "
                         "depende solo de la TASA de abstencion y NO de cuales preguntas se callan. "
                         "Con 0,0 (el default) la perdida es bit a bit la de las campanias "
                         "anteriores. El valor se deriva del ratio de gradientes medido en el "
                         "checkpoint DE SIEMBRA con `medir_ratio_ce.py`, nunca a mitad de corrida.")
    ap.add_argument("--blanco", default="ausencia", choices=("ausencia", "error"),
                    help="que aprende la BCE de la cabeza (A5, ALTERNATIVAS_DETECCION_20260826.md). "
                         "«ausencia» = ¿hay respuesta?, que es lo de siempre. «error» = ¿me voy a "
                         "equivocar si contesto?, con el blanco tomado del argmax del propio modelo "
                         "y stop_gradient. Solo tiene efecto con --abst cabeza o slot.")
    ap.add_argument("--donde", default="pre", choices=("pre", "post", "lat", "lat2"),
                    help="en que punto del bloque 0 entra la lectura del archivo "
                         "(PREREG_QUERY_CONJUNTA.md). pre = antes de la conv y del mixer, sobre "
                         "emb[x], que es lo que se venia haciendo y deja la query como funcion pura "
                         "del token; post = despues del mixer del mismo bloque, con lo que la query "
                         "puede depender de la entidad y de la relacion a la vez (medido el 22-ago: "
                         "rompe el modelo, porque la lectura deja de entrar antes del computo); "
                         "lat = camino lateral, la inyeccion queda donde `pre` la tiene y solo la "
                         "QUERY se forma sobre conv3(ln1(h)), que da contexto local sin mover el "
                         "punto de inyeccion; "
                         "lat2 = igual que lat pero con conv PROPIA para la query (`convq`), "
                         "inicializada en [1,0,0] — arranca siendo exactamente `pre` y el modelo "
                         "decide por gradiente cuanto contexto usa, con lo cual no puede ser "
                         "estructuralmente peor. Corrige el acoplamiento diagnosticado el 22-ago")
    ap.add_argument("--reinit-adam", action="store_true",
                    help="reinicia el estado de Adam al reanudar. La condicion `cabeza` lo hace sola "
                         "porque el arbol de params cambia de forma; este flag existe para que "
                         "`token` y `escala` puedan hacer lo MISMO y el contraste sea pareado. Sin "
                         "el flag, reanudar es continuar la misma corrida (lo que hace la campania)")
    ap.add_argument("--cortes-vigente", default="",
                    help="lista de valores de `vigente` (ej. 0.85,0.90,0.95). Cada vez que la "
                         "metrica cruza uno por primera vez se guarda un checkpoint aparte, para "
                         "muestrear la frontera del margen (PREREG_FRONTERA.md)")
    ap.add_argument("--mezcla", default="fija", choices=("fija", "dinamica"),
                    help="como se reparte el presupuesto de muestras entre tipos de pregunta "
                         "(DISENO_ESCALONADO.md). fija = --p-vieja y --p-nose mandan de principio a "
                         "fin, que es lo que se venia haciendo; dinamica = cada tipo se muestrea "
                         "proporcional a su error EMA, asi la capacidad que ya se resolvio deja de "
                         "gastar muestras sola. OJO: el CONTROL `fijo_promedio` no es un modo "
                         "aparte — es `fija` con --p-vieja/--p-nose puestos en el promedio que la "
                         "corrida dinamica termino usando (sale de `mezcla_promedio` en su JSON). "
                         "Se hace asi a proposito: la mezcla del control queda escrita en la config "
                         "del control, en vez de depender de que otro archivo siga estando")
    ap.add_argument("--mezcla-piso", type=float, default=0.10,
                    help="piso de probabilidad por tipo en --mezcla dinamica. Evita que un tipo "
                         "resuelto deje de verse del todo y se olvide")
    ap.add_argument("--mezcla-alpha", type=float, default=0.1,
                    help="alpha de la EMA del error por tipo. Va LENTA a proposito: el error se mide "
                         "con 512 muestras y tiene ruido de +-0,02, y ademas hay realimentacion "
                         "—el muestreo cambia el error que decide el muestreo—, asi que una EMA "
                         "rapida perseguiria el ruido y oscilaria")
    ap.add_argument("--parar-si-estanca", type=int, default=0,
                    help="para la corrida si el acierto global no mejora su mejor valor en N "
                         "evaluaciones seguidas (0 = apagado, que es lo que corre la campania del "
                         "23-ago). Es el pedido de Maxi —«que se detenga cuando alcanza su mejor "
                         "capacidad»— y es lo que convierte el ahorro de S-2 en ahorro de reloj: "
                         "sin esto, escalonar llega antes al techo pero igual gasta los 20000 "
                         "pasos. VA APAGADO EN LA CAMPANIA QUE MIDE porque cortar antes deja la "
                         "curva sin cola, y la cola es justamente lo que S-2 y S-4 leen. Se "
                         "enciende DESPUES, cuando ya no se esta midiendo sino produciendo. "
                         "N cuenta EVALUACIONES, no pasos: con --cada 250, N=20 son 5000 pasos")
    ap.add_argument("--salida", default="resultados_micro.json")
    ap.add_argument("--pesos", default=None, help="ruta .pkl donde dejar los pesos finales")
    ap.add_argument("--idioma", type=int, default=2, choices=(1, 2, 3),
                    help="1 = verbo «pertenece_a» (campania del 14-ago), 2 = «posee» (corregido), "
                         "3 = 24 relaciones en vez de 6, para que la colision de clave baje del "
                         "72,4 %% al 23,5 %% de los episodios y el error deje de estar dominado por "
                         "el empate entre dos entradas (2026-08-23)")
    ap.add_argument("--horizonte", type=int, default=0,
                    help="sobre cuantos pasos se calcula el decaimiento de la lr (0 = usar --pasos). "
                         "Sirve para poder PARAR antes y CONTINUAR despues sin romper nada: si el "
                         "cosine se calcula sobre 12000 y luego se extiende a 20000, la lr ya toco "
                         "su minimo y la continuacion no equivale a haber corrido 20000 de una. "
                         "Fijando el horizonte en el maximo previsto, `--pasos` decide sólo hasta "
                         "donde se corre por ahora")
    ap.add_argument("--tramo", type=int, default=0,
                    help="cuantos pasos correr EN ESTA sesion (0 = todos los que falten). El total "
                         "sigue siendo --pasos: el tramo sólo dice hasta donde llega esta VM antes "
                         "de guardar y salir, para que otra sesion siga desde ahi")
    ap.add_argument("--cada", type=int, default=2000,
                    help="cada cuantos pasos se evalua y se guarda el checkpoint. Con VMs que se "
                         "caen conviene bajarlo: es lo que se pierde cuando muere la sesion")
    ap.add_argument("--ckpt", default=None,
                    help="ruta .pkl del checkpoint: se guarda en cada evaluacion y, si ya existe, "
                         "se REANUDA desde ahi (permite partir una corrida entre varias sesiones)")
    a = ap.parse_args()

    I.fijar_version(a.idioma)       # antes de construir el modelo: define I.V

    # Antes de que jax tracee nada: el valor queda horneado en los grafos compilados, asi que fijarlo
    # tarde seria peor que no fijarlo (compilaria con `pre` y el JSON diria `post`).
    # `_ABST` va en el mismo `global` que `_DONDE` y por el mismo motivo. Sin declararlo, la
    # asignacion crea una LOCAL y la global se queda en "token": `--abst slot` no tendria efecto y
    # la corrida diria `slot` en el JSON mientras entrena `token`. Es el mismo agujero que taparon
    # las guardas de identidad del checkpoint, y aca lo cazamos antes de gastar una unidad.
    global _DONDE, _ABST, _BLANCO, _PERDIDA_CABEZA, _REC_L, _REC_M, _REC_F, _REC_CE, _REC_RANK
    _REC_L, _REC_M, _REC_F, _REC_CE = a.rec_l, a.rec_m, a.rec_f, a.rec_ce
    _REC_RANK = a.rec_rank
    # Compuerta de la ENMIENDA, ahora ejecutable y no solo escrita: si el optimo de la perdida es un
    # extremo, eso no es un riesgo a vigilar sino un defecto, y se cierra ANTES de gastar GPU.
    if a.perdida_cabeza == "recompensa":
        c_est = (a.rec_m - a.rec_f) / (1.0 + a.rec_m)
        if not 0.0 < c_est < 1.0:
            sys.exit(f"ABORTA: con M={a.rec_m} y F={a.rec_f} el umbral por muestra es c*={c_est:.4f}, "
                     f"fuera de (0,1). Hace falta F < M para que EXISTA un punto en el que convenga "
                     f"callarse; si no, el optimo es un extremo. Ver ENMIENDA_RECOMPENSA_F.md.")
        r_mudo = 0.4065 * a.rec_l - (1.0 - 0.4065) * a.rec_f
        print(f"recompensa: L={a.rec_l} M={a.rec_m} F={a.rec_f} CE={a.rec_ce}  ->  "
              f"c*={c_est:.4f}   recompensa neta del modelo MUDO = {r_mudo:+.4f}"
              f"{'  <== POSITIVA: el silencio cobra premio' if r_mudo > 0 else ''}\n", flush=True)
    _DONDE = a.donde
    _ABST = a.abst
    _BLANCO = a.blanco
    _PERDIDA_CABEZA = a.perdida_cabeza

    print(f"MICRO-LM · nivel {a.nivel} · vocabulario {I.V} tokens · d={a.d} capas={a.capas} "
          f"· lectura {a.donde}", flush=True)
    # El hardware va al JSON, no sólo al log. Cuando Colab raciona las T4 hay que aceptar el
    # acelerador que haya, y entonces «en qué corrió esta celda» deja de ser un detalle de
    # operación: es una variable que podría explicar una diferencia entre celdas, y sin registrarla
    # no hay forma de descartarla después.
    hw = ", ".join(f"{d.platform}:{d.device_kind}" for d in jax.devices())
    print(f"dispositivos: {jax.devices()}  ->  hw={hw}", flush=True)

    params = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas, N_TURNOS=64)
    n = M.contar(params)
    print(f"parametros: {n:,} ({n * 4 / 1e6:.1f} MB en fp32)\n", flush=True)

    HOR = a.horizonte if a.horizonte > 0 else a.pasos
    warmup = min(500, max(1, HOR // 10))
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warmup, HOR, a.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    # El eje `--abst` elige QUE funcion de perdida y que regla de decision se usan. `escala` comparte
    # las dos con `token`: lo unico que cambia es el init del vector de NOSE al entrar en la fase.
    # `slot` reusa el camino de `cabeza` a proposito: misma BCE, mismo argmax con NOSE excluido, y
    # lo unico que cambia es de DONDE sale el logit binario —la masa del slot nulo en vez de una
    # proyeccion del estado final—. Asi el contraste mide «donde vive la abstencion» y no «que
    # perdida se uso», que es la forma del error que arruino a `post` el 22-ago.
    _bin = a.abst in ("cabeza", "slot")
    fn_perd = perdida_cabeza if _bin else perdida
    fn_pred = predecir_cabeza if _bin else predecir

    cortes_vigente = sorted(float(x) for x in a.cortes_vigente.split(",") if x.strip())
    cortes_hechos = set()

    @jax.jit
    def paso(params, state, ses, cortes, turnos, mask, cons, pos, tgt):
        (l, acc), g = jax.value_and_grad(fn_perd, has_aux=True)(
            params, ses, cortes, turnos, mask, cons, pos, tgt)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, acc

    rng = np.random.default_rng(1000 + a.semilla)
    DAT.reset_truncados()
    hist, t0 = [], time.time()
    paso0 = 0
    mez = estado_mezcla_inicial()

    # --- reanudacion -----------------------------------------------------------------------
    # Se guardan las CUATRO cosas que definen el estado del entrenamiento, no sólo los pesos:
    #   · params      — el modelo;
    #   · opt_state   — el momento de Adam Y el contador de pasos, del que depende el scheduler
    #                   (warmup + cosine): sin esto la tasa de aprendizaje volveria al warmup;
    #   · rng         — el estado del generador de lotes, para no repetir los mismos episodios;
    #   · hist        — la historia de evaluaciones, para que la curva quede entera.
    # Guardar sólo los pesos daria un modelo que «sigue» pero con otro optimizador y otra lr: no es
    # la misma corrida partida en dos, es otra corrida.
    if a.ckpt and os.path.exists(a.ckpt):
        with open(a.ckpt, "rb") as f:
            ck = pickle.load(f)
        # El scheduler (warmup + cosine hasta `pasos`) y el idioma forman parte de la identidad de
        # la corrida: reanudar con otros valores da una curva que NO es la continuacion de la de
        # antes, y quedaria pegada en el mismo JSON como si lo fuera.
        # `pasos` NO entra en la comparacion a proposito: es hasta donde se corre por ahora, y se
        # puede extender. Lo que no puede cambiar es el HORIZONTE, porque define la curva de lr.
        for k in ("nivel", "semilla", "lr", "idioma", "d", "capas"):
            if ck["config"].get(k) != vars(a).get(k):
                sys.exit(f"ABORTA: el checkpoint tiene {k}={ck['config'].get(k)} y se pidio "
                         f"{k}={vars(a).get(k)}. No es la misma corrida.")
        # `donde` va aparte porque los checkpoints anteriores al 22-ago no lo tienen en su config y
        # todos ellos son `pre` (era la unica posicion que existia). El chequeo importa mas que los
        # otros: la campania corre POR TRAMOS entre cuentas de Colab, y reanudar un tramo `post` sin
        # pasar el flag lo continuaria como `pre` sin decir nada —la arquitectura cambia a mitad de
        # corrida y el JSON mostraria una curva sola—. Es la misma familia de la D-1 del 20-ago:
        # el estado que dos pasos comparten tiene que estar declarado, no supuesto.
        # Misma familia que la guarda de `donde`, y por el mismo motivo: la campania corre por tramos
        # entre cuentas, asi que un tramo al que se le olvida el flag continuaria una corrida
        # dinamica como fija sin decir nada, y el JSON mostraria una curva sola. Los checkpoints
        # anteriores al 23-ago no traen la clave y todos ellos son `fija`, que es el default.
        if ck["config"].get("mezcla", "fija") != a.mezcla:
            sys.exit(f"ABORTA: el checkpoint se entreno con mezcla={ck['config'].get('mezcla', 'fija')} "
                     f"y se pidio mezcla={a.mezcla}. Es otra politica de muestreo, no la misma corrida.")
        # El piso distingue `ed` de `e0`, que sólo se diferencian en ese numero. Un tramo al que se
        # le olvida la variable continuaria `e0` con piso 0,10 y las dos celdas serian la misma
        # corrida con dos nombres — el mismo agujero que taparon las guardas de `donde` y `mezcla`.
        # Sólo se compara en dinamica: en `fija` el piso no se usa, y los ckpts previos no lo traen.
        if a.mezcla == "dinamica" and \
                abs(float(ck["config"].get("mezcla_piso", 0.10)) - a.mezcla_piso) > 1e-9:
            sys.exit(f"ABORTA: el checkpoint se entreno con mezcla_piso="
                     f"{ck['config'].get('mezcla_piso', 0.10)} y se pidio {a.mezcla_piso}. "
                     f"Es otra politica de muestreo, no la misma corrida.")
        if ck["config"].get("donde", "pre") != a.donde:
            sys.exit(f"ABORTA: el checkpoint se entreno con donde={ck['config'].get('donde', 'pre')} "
                     f"y se pidio donde={a.donde}. Es otra arquitectura, no la misma corrida.")
        # `blanco` (2026-08-26, A5): MISMA familia que `donde` y `mezcla`. Sin esta guarda, un tramo
        # al que se le olvida el flag continuaria una corrida con blanco `error` como `ausencia` sin
        # decir nada, y el JSON mostraria una curva sola.
        #
        # La guarda se aplica SOLO si el checkpoint ya trae la clave, y eso NO es aflojarla: la fase
        # de abstencion arranca SEMBRANDO desde una base (`tramo_abst.sh` copia `n3_sX.pkl`), y esas
        # bases son anteriores a hoy y no tienen `blanco`. Con `.get(..., "ausencia")` la comparacion
        # daria "ausencia" != "error" y la campania abortaria en su PRIMER tramo, antes de entrenar
        # un solo paso. Lo que la guarda tiene que impedir es que un tramo POSTERIOR cambie el
        # blanco, y eso sigue cubierto: todo checkpoint escrito desde hoy lleva la clave, porque la
        # config se vuelca con `vars(a)` completo.
        if "blanco" in ck["config"] and ck["config"]["blanco"] != a.blanco:
            sys.exit(f"ABORTA: el checkpoint se entreno con blanco="
                     f"{ck['config'].get('blanco', 'ausencia')} y se pidio blanco={a.blanco}. "
                     f"La cabeza aprende otra cosa, no es la misma corrida.")
        # `perdida_cabeza` (2026-08-29): MISMA familia que `blanco` y por la misma razon. Un tramo que
        # se reanuda con otra forma de perdida no continua la corrida, la bifurca — y el 29-ago se
        # midio que esta eleccion decide a cual de los dos puntos fijos cae la unidad, asi que
        # mezclarla a mitad de camino seria el confound mas caro posible en esta campania.
        if ("perdida_cabeza" in ck["config"]
                and ck["config"]["perdida_cabeza"] != a.perdida_cabeza):
            sys.exit(f"ABORTA: el checkpoint se entreno con perdida_cabeza="
                     f"{ck['config']['perdida_cabeza']} y se pidio {a.perdida_cabeza}. "
                     f"No es la misma corrida.")
        # Pesos de la recompensa (2026-08-30): MISMA familia que `perdida_cabeza`. Cambiar L a mitad
        # de camino no continua la corrida, la bifurca — L decide si el atractor mudo cobra premio, o
        # sea a cual de los dos puntos fijos del 29-ago cae la unidad. Se compara SOLO si el ckpt trae
        # la clave, por lo mismo que `blanco`: las bases de siembra son anteriores y no la tienen.
        if a.perdida_cabeza == "recompensa":
            for k, v in (("rec_l", a.rec_l), ("rec_m", a.rec_m), ("rec_f", a.rec_f),
                         ("rec_ce", a.rec_ce), ("rec_rank", a.rec_rank)):
                if k in ck["config"] and abs(float(ck["config"][k]) - v) > 1e-9:
                    sys.exit(f"ABORTA: el checkpoint se entreno con {k}={ck['config'][k]} y se pidio "
                             f"{v}. Los pesos de la recompensa definen donde esta el optimo, asi que "
                             f"no es la misma corrida.")
        hor_ck = ck["config"].get("horizonte") or ck["config"].get("pasos")
        if hor_ck != HOR:
            sys.exit(f"ABORTA: el checkpoint se entreno con horizonte de lr {hor_ck} y se pidio "
                     f"{HOR}. Continuar cambiaria la curva de aprendizaje a mitad de camino.")
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        # --- entrada a la fase de abstencion desde un checkpoint base (PREREG_CABEZA_ABSTENCION) ---
        # Un ckpt de la campania base no tiene la cabeza `abst`, asi que el arbol de params cambia de
        # forma y el estado de Adam deja de corresponderle. Se reinicializa el optimizador, y se hace
        # IGUAL en las tres condiciones —tambien en `token`— para que el contraste sea pareado: por
        # eso la campania `token` del 17-ago no se reusa como linea de base.
        # OJO: sólo se toca el arbol cuando la condicion lo NECESITA. Si `--abst token` reanudara
        # agregando la cabeza, cualquier corrida en curso de la campania `x` se reanudaria con Adam
        # reiniciado a mitad de camino — dejaria de ser la misma corrida partida en dos.
        # Misma familia que la guarda de arriba, y por el mismo motivo (2026-08-24): un ckpt anterior
        # a `lat2` no trae `convq` en sus bloques. Se agrega SOLO si la condicion la necesita —igual
        # que con `abst`—, para que ninguna corrida en curso de otra condicion se reanude con Adam
        # reiniciado a mitad de camino y deje de ser la misma corrida partida en dos.
        if a.donde == "lat2" and "convq" not in params["blocks"][0]:
            frescos = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas)
            for blk, fresco in zip(params["blocks"], frescos["blocks"]):
                blk["convq"] = fresco["convq"]
            state = opt.init(params)
            print("el checkpoint no traia `convq`: se agrega en [1,0,0] y se reinicia Adam\n",
                  flush=True)
        elif "abst" not in params and (a.abst != "token" or a.reinit_adam):
            params["abst"] = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas)["abst"]
            state = opt.init(params)
            print("el checkpoint no traia cabeza de abstencion: se agrega y se reinicia Adam\n",
                  flush=True)
        elif a.reinit_adam:
            state = opt.init(params)
            print("Adam reiniciado por pedido explicito (--reinit-adam)\n", flush=True)
        elif "opt_state" not in ck:
            # Un checkpoint escrito por `sembrar.py` no trae estado de Adam a proposito: es una BASE
            # de siembra, no la continuacion de la corrida de la que salio. Se reinicia avisando, y
            # solo si la procedencia esta declarada; si falta el opt_state sin que nadie haya
            # sembrado, eso es un checkpoint roto y no algo que haya que adivinar en silencio.
            if "sembrado_de" not in ck:
                sys.exit("ABORTA: el checkpoint no trae `opt_state` ni `sembrado_de`. Es un archivo "
                         "incompleto, no una siembra declarada.")
            state = opt.init(params)
            print(f"siembra declarada desde {ck['sembrado_de']['ruta']} "
                  f"(paso {ck['sembrado_de']['paso']}): Adam arranca de cero\n", flush=True)
        else:
            state = jax.tree_util.tree_map(jnp.asarray, ck["opt_state"])
        if a.abst == "escala" and ck["config"].get("abst", "token") != "escala":
            # La explicacion barata de la pista del 17-ago: el vector de NOSE mide 0,367 contra ~1,01
            # de un valor. Se lo lleva a la norma media de los tokens de VALOR (nombres y numeros),
            # en la salida y en la entrada, que es la version mas generosa de esa hipotesis.
            vals = [I.STOI[t] for t in list(I.NOMBRES) + list(I.NUMEROS) if t in I.STOI]
            for clave, mat, eje in (("head", params["head"]["w"], 0), ("emb", params["emb"], 1)):
                v = mat[:, vals] if eje == 0 else mat[vals, :]
                objetivo = float(jnp.linalg.norm(v, axis=eje).mean())
                col = mat[:, NOSE] if eje == 0 else mat[NOSE, :]
                escala = objetivo / float(jnp.maximum(jnp.linalg.norm(col), 1e-6))
                if eje == 0:
                    params["head"]["w"] = mat.at[:, NOSE].set(col * escala)
                else:
                    params["emb"] = mat.at[NOSE, :].set(col * escala)
                print(f"escala: {clave}[NOSE] x{escala:.3f} -> norma {objetivo:.4f}", flush=True)
            state = opt.init(params)
        # En una siembra el `rng` y la `historia` tampoco viajan: el flujo de datos arranca de la
        # semilla y la curva es NUEVA. Mezclar la historia de la corrida de origen dibujaria una sola
        # curva donde hubo dos regimenes, que es exactamente lo que la guarda de identidad impide.
        if "rng" in ck:
            rng.bit_generator.state = ck["rng"]
        hist, paso0 = ck.get("historia", []), ck["paso"]
        # La EMA del error y el acumulado de la mezcla son ESTADO del entrenamiento, igual que Adam.
        # Esta campania corre por tramos de 8000 pasos entre cuentas de Colab: sin esto, cada tramo
        # reiniciaria la EMA en 1,0 y la mezcla volveria a uniforme tres veces en una corrida, con lo
        # que «dinamica» ya no seria una sola politica sino tres arranques pegados.
        if "mezcla" in ck:
            mez = ck["mezcla"]
            mez.setdefault("pasos_acum", 0)
            mez.setdefault("acum", [0.0] * len(TIPOS))
        print(f"REANUDA desde {a.ckpt}: paso {paso0} de {a.pasos} "
              f"({len(hist)} evaluaciones ya hechas)\n", flush=True)

    if paso0 >= a.pasos:
        print("el checkpoint ya esta completo; nada que hacer")
        return

    def guardar_ckpt(s):
        if not a.ckpt:
            return
        tmp = a.ckpt + ".tmp"      # escritura atomica: si la VM muere a mitad, el ckpt viejo vive
        with open(tmp, "wb") as f:
            pickle.dump({"params": jax.device_get(params), "opt_state": jax.device_get(state),
                         "rng": rng.bit_generator.state, "historia": hist, "paso": s,
                         "config": vars(a), "mezcla": mez}, f)
        os.replace(tmp, a.ckpt)

    # El tramo se redondea al múltiplo de `--cada` para terminar SIEMPRE sobre un checkpoint: si
    # cortara en el medio, esos pasos se perderían igual y la VM los habría gastado al pedo.
    fin = a.pasos if a.tramo <= 0 else min(a.pasos, paso0 + max(a.cada, a.tramo - a.tramo % a.cada))
    if fin < a.pasos:
        print(f"tramo: se corre hasta el paso {fin} y se guarda para continuar despues\n", flush=True)

    # La mezcla de ENTRENAMIENTO puede moverse; la de EVALUACION nunca. Si la evaluacion siguiera a
    # la mezcla dinamica, cada condicion se estaria midiendo sobre una poblacion de preguntas
    # distinta y `dinamica` vs `fija` dejaria de ser comparable —el numero cambiaria por como se
    # midio, no por lo que aprendio—. `a.p_vieja` y `a.p_nose` quedan como la mezcla de REFERENCIA.
    if a.mezcla == "dinamica":
        w = pesos_de_ema(mez["ema"], a.mezcla_piso)
        p_vieja_tr, p_nose_tr = probs_de_pesos(w)
        print(f"mezcla dinamica · piso {a.mezcla_piso:.2f} · alpha {a.mezcla_alpha:.2f} · "
              f"arranca en vigente {w[0]:.3f} / anterior {w[1]:.3f} / nose {w[2]:.3f}\n", flush=True)
    else:
        p_vieja_tr, p_nose_tr = a.p_vieja, a.p_nose
        w = pesos_de_probs(p_vieja_tr, p_nose_tr)
    ult_eval = paso0
    # El mejor global visto, para `--parar-si-estanca`. Se siembra con la historia ya vivida: si no,
    # un tramo que reanuda arrancaria creyendo que nunca vio nada bueno y no cortaria nunca.
    mejor = {"valor": -1.0, "paso": 0, "seguidas": 0}
    for e in hist:
        v = 0.39 * e.get("vigente", 0) + 0.21 * e.get("anterior", 0) + 0.40 * e.get("nose", 0)
        if v > mejor["valor"]:
            mejor.update(valor=v, paso=e["paso"], seguidas=0)
        else:
            mejor["seguidas"] += 1

    for s in range(paso0 + 1, fin + 1):
        ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
            rng, a.batch, nivel=a.nivel, n_hechos=4, n_sesiones=4, p_vieja=p_vieja_tr,
            p_nose=p_nose_tr)
        params, state, l, acc = paso(params, state, jnp.array(ses), jnp.array(cortes),
                                     jnp.array(turnos), jnp.array(mask), jnp.array(cons),
                                     jnp.array(pos), jnp.array(tgt))
        if s % 500 == 0:
            print(f"  paso {s:6d}  loss {float(l):.4f}  acc {float(acc):.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        if s % a.cada == 0 or s == fin:
            trunc = DAT.tasa_truncados()            # la compuerta, en el registro permanente
            ev = np.random.default_rng(90000 + a.semilla)
            m = evaluar(params, ev, nivel=a.nivel, p_vieja=a.p_vieja, p_nose=a.p_nose,
                        pred_fn=fn_pred)
            # `p_nose` va en CADA evaluacion y no solo en la config, porque la guarda de identidad
            # del checkpoint no lo compara: una corrida puede reanudarse con otro valor. Eso es
            # deliberado —el curriculum de dos fases entrena primero sin preguntas sin respuesta y
            # las introduce despues— pero deja una curva cuya segunda mitad es OTRA tarea. Sin este
            # registro, mañana leeriamos un salto de metrica como si fuera aprendizaje.
            # --- escalonado: se acumula lo YA gastado y recien despues se mueve la mezcla ---------
            # El orden importa. Los pesos `w` son los que rigieron el bloque que termina en `s`, asi
            # que el acumulado se cierra con ELLOS; recien despues la evaluacion nueva mueve la EMA.
            # Al reves, el promedio le atribuiria a cada bloque la mezcla del bloque siguiente, y el
            # control `fijo_promedio` se correria con una mezcla que nadie uso.
            tramo_pasos = s - ult_eval
            ult_eval = s
            if tramo_pasos > 0:
                mez["acum"] = [c + wi * tramo_pasos for c, wi in zip(mez["acum"], w)]
                mez["pasos_acum"] += tramo_pasos
            prom = ([c / mez["pasos_acum"] for c in mez["acum"]] if mez["pasos_acum"] else list(w))
            usada = {"vigente": w[0], "anterior": w[1], "nose": w[2]}
            if a.mezcla == "dinamica":
                for i, t in enumerate(TIPOS):
                    # nan cuando el lote de eval no trajo ningun caso de ese tipo: se deja la EMA
                    # quieta en vez de empujarla con un valor inventado.
                    if not np.isnan(m[t]):
                        err = 1.0 - float(m[t])
                        mez["ema"][t] = (1 - a.mezcla_alpha) * mez["ema"][t] + a.mezcla_alpha * err
                w = pesos_de_ema(mez["ema"], a.mezcla_piso)
                p_vieja_tr, p_nose_tr = probs_de_pesos(w)
            hist.append({"paso": s, "truncados": float(trunc), "p_nose": float(a.p_nose),
                         "mezcla_usada": usada, "mezcla_promedio": dict(zip(TIPOS, prom)),
                         "p_vieja_tr": float(p_vieja_tr), "p_nose_tr": float(p_nose_tr), **m})
            extra = ("" if a.p_nose == 0 else
                     f" · nose {m['nose']:.4f} (ent {m['nose_ent']:.4f}/rel {m['nose_rel']:.4f})"
                     f" · falsa_abst {m['falsa_abst']:.4f}")
            print(f"    ── eval: vigente {m['vigente']:.4f} · anterior {m['anterior']:.4f}"
                  f" · abstencion {m['abstencion']:.4f} · truncados {trunc:.4f}{extra}", flush=True)
            if a.mezcla == "dinamica":
                print(f"       mezcla -> vigente {w[0]:.3f} · anterior {w[1]:.3f} · nose {w[2]:.3f}"
                      f"   (p_vieja {p_vieja_tr:.3f} · p_nose {p_nose_tr:.3f})", flush=True)
            json.dump({"config": vars(a), "params": n, "hw": hw, "historia": hist},
                      open(a.salida, "w"), indent=1)
            guardar_ckpt(s)
            # --- cortes por VALOR de vigente (PREREG_FRONTERA.md) --------------------------------
            # Para muestrear la frontera del margen hace falta detener la base cuando `vigente` llega
            # a un valor dado, no en un paso dado: lo que se controla es el MARGEN, y a paso fijo
            # cada semilla cae en un margen distinto. Se guarda al PRIMER cruce y una sola vez por
            # umbral; el nombre lleva el valor pedido, no el alcanzado, para que la unidad sea
            # identificable antes de correrla.
            for corte in cortes_vigente:
                if corte not in cortes_hechos and m["vigente"] >= corte:
                    cortes_hechos.add(corte)
                    destino = f"{a.ckpt}.v{int(corte * 100)}"
                    with open(destino, "wb") as f:
                        pickle.dump({"params": jax.device_get(params),
                                     "opt_state": jax.device_get(state), "rng": rng.bit_generator.state,
                                     "historia": hist, "paso": s, "config": vars(a)}, f)
                    print(f"    [corte por vigente {corte:.2f}: alcanzado {m['vigente']:.4f} en el "
                          f"paso {s} -> {os.path.basename(destino)}]", flush=True)
            # Lo del 13-ago no se repite: si se esta truncando, se corta ACA y no despues de 73 min.
            if trunc > 0.01:
                sys.exit(f"ABORTA: truncamiento {trunc:.4f} > 0,01 — se estaria midiendo el padding")

            # --- parada por techo alcanzado (pedido de Maxi, 23-ago) -----------------------------
            # El acierto global se pondera con la mezcla de REFERENCIA, no con la que la corrida
            # este usando: si se ponderara con la dinamica, mover el muestreo hacia lo dificil
            # bajaria el numero sola y la corrida se cortaria por haberse puesto un examen mas
            # dificil, no por haber dejado de aprender.
            if a.parar_si_estanca > 0:
                glob = 0.39 * m["vigente"] + 0.21 * m["anterior"] + 0.40 * m["nose"]
                if glob > mejor["valor"] + 1e-6:
                    mejor.update(valor=glob, paso=s, seguidas=0)
                else:
                    mejor["seguidas"] += 1
                    if mejor["seguidas"] >= a.parar_si_estanca:
                        print(f"\nPARA: el acierto global no mejora {mejor['valor']:.4f} "
                              f"(paso {mejor['paso']}) desde hace {mejor['seguidas']} evaluaciones "
                              f"= {mejor['seguidas'] * a.cada} pasos. Techo alcanzado.", flush=True)
                        break

    if a.pesos:
        with open(a.pesos, "wb") as f:
            pickle.dump({"params": jax.device_get(params), "config": vars(a),
                         "vocab": I.ITOS}, f)
        print(f"pesos guardados en {a.pesos} ({os.path.getsize(a.pesos)/1e6:.1f} MB)", flush=True)
    print("\nlisto:", hist[-1] if hist else "sin evaluaciones")
    # La receta del control, impresa donde se la va a buscar. `fijo_promedio` no se puede armar antes
    # de correr la dinamica —el promedio es un RESULTADO de la corrida—, asi que la corrida lo deja
    # listo para copiar y pegar en vez de dejar el calculo para despues.
    if a.mezcla == "dinamica" and mez["pasos_acum"]:
        pv, pn = probs_de_pesos([c / mez["pasos_acum"] for c in mez["acum"]])
        print(f"CONTROL fijo_promedio: --mezcla fija --p-vieja {pv:.4f} --p-nose {pn:.4f}",
              flush=True)


if __name__ == "__main__":
    main()
