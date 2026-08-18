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

import datos as DAT
import idioma as I
import modelo as M

NOSE = I.STOI["NOSE"]


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
    lg = M.responder(params, archivo, turnos, cons, mask)
    return jnp.take_along_axis(lg, pos[:, None, None], axis=1)[:, 0, :]


@jax.jit
def predecir(params, ses, cortes, turnos, mask, cons, pos):
    return logits_de(params, ses, cortes, turnos, mask, cons, pos).argmax(-1)


def perdida(params, ses, cortes, turnos, mask, cons, pos, tgt):
    lg = logits_de(params, ses, cortes, turnos, mask, cons, pos)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg, tgt).mean()
    return ce, (lg.argmax(-1) == tgt).mean()


# --- cabeza de abstencion separada (2026-08-18, `PREREG_CABEZA_ABSTENCION.md`) -------------------
# `NOSE` deja de ser una entrada del softmax de vocabulario y pasa a tener su propia salida binaria.
# Las dos decisiones —«¿esta?» y «¿que valor?»— dejan de competir por la misma masa de probabilidad.

def _partes(params, ses, cortes, turnos, mask, cons, pos):
    archivo = M.escribir(params, ses, cortes)
    lg, a = M.responder_con_abst(params, archivo, turnos, cons, mask)
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
    bce = optax.sigmoid_binary_cross_entropy(a, es_nose).mean()
    lg_v = lg.at[:, NOSE].set(-1e9)
    ce = optax.softmax_cross_entropy_with_integer_labels(lg_v, tgt)
    # la CE del valor sólo cuenta donde HAY respuesta, y se normaliza por esa fraccion para que la
    # escala de la perdida no dependa de `p_nose` (si no, subir p_nose baja el peso del valor y el
    # contraste entre condiciones mediria eso).
    hay = 1.0 - es_nose
    ce = (ce * hay).sum() / jnp.maximum(hay.sum(), 1.0)
    pred = jnp.where(a > 0.0, NOSE, lg_v.argmax(-1))
    return bce + ce, (pred == tgt).mean()


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
    ap.add_argument("--abst", default="token", choices=("token", "escala", "cabeza"),
                    help="como se decide la abstencion (PREREG_CABEZA_ABSTENCION.md). "
                         "token = NOSE es una entrada mas del softmax de vocabulario (lo de hoy); "
                         "escala = idem pero renormalizando el vector de NOSE a la norma media de "
                         "los tokens de valor al arrancar la fase; "
                         "cabeza = salida binaria separada, con NOSE excluido del softmax de valores")
    ap.add_argument("--reinit-adam", action="store_true",
                    help="reinicia el estado de Adam al reanudar. La condicion `cabeza` lo hace sola "
                         "porque el arbol de params cambia de forma; este flag existe para que "
                         "`token` y `escala` puedan hacer lo MISMO y el contraste sea pareado. Sin "
                         "el flag, reanudar es continuar la misma corrida (lo que hace la campania)")
    ap.add_argument("--salida", default="resultados_micro.json")
    ap.add_argument("--pesos", default=None, help="ruta .pkl donde dejar los pesos finales")
    ap.add_argument("--idioma", type=int, default=2, choices=(1, 2),
                    help="1 = verbo «pertenece_a» (campania del 14-ago), 2 = «posee» (corregido)")
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

    print(f"MICRO-LM · nivel {a.nivel} · vocabulario {I.V} tokens · d={a.d} capas={a.capas}",
          flush=True)
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
    fn_perd = perdida_cabeza if a.abst == "cabeza" else perdida
    fn_pred = predecir_cabeza if a.abst == "cabeza" else predecir

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
        if "abst" not in params and (a.abst != "token" or a.reinit_adam):
            params["abst"] = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas)["abst"]
            state = opt.init(params)
            print("el checkpoint no traia cabeza de abstencion: se agrega y se reinicia Adam\n",
                  flush=True)
        elif a.reinit_adam:
            state = opt.init(params)
            print("Adam reiniciado por pedido explicito (--reinit-adam)\n", flush=True)
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
        rng.bit_generator.state = ck["rng"]
        hist, paso0 = ck["historia"], ck["paso"]
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
                         "config": vars(a)}, f)
        os.replace(tmp, a.ckpt)

    # El tramo se redondea al múltiplo de `--cada` para terminar SIEMPRE sobre un checkpoint: si
    # cortara en el medio, esos pasos se perderían igual y la VM los habría gastado al pedo.
    fin = a.pasos if a.tramo <= 0 else min(a.pasos, paso0 + max(a.cada, a.tramo - a.tramo % a.cada))
    if fin < a.pasos:
        print(f"tramo: se corre hasta el paso {fin} y se guarda para continuar despues\n", flush=True)

    for s in range(paso0 + 1, fin + 1):
        ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
            rng, a.batch, nivel=a.nivel, n_hechos=4, n_sesiones=4, p_vieja=a.p_vieja,
            p_nose=a.p_nose)
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
            hist.append({"paso": s, "truncados": float(trunc), "p_nose": float(a.p_nose), **m})
            extra = ("" if a.p_nose == 0 else
                     f" · nose {m['nose']:.4f} (ent {m['nose_ent']:.4f}/rel {m['nose_rel']:.4f})"
                     f" · falsa_abst {m['falsa_abst']:.4f}")
            print(f"    ── eval: vigente {m['vigente']:.4f} · anterior {m['anterior']:.4f}"
                  f" · abstencion {m['abstencion']:.4f} · truncados {trunc:.4f}{extra}", flush=True)
            json.dump({"config": vars(a), "params": n, "hw": hw, "historia": hist},
                      open(a.salida, "w"), indent=1)
            guardar_ckpt(s)
            # Lo del 13-ago no se repite: si se esta truncando, se corta ACA y no despues de 73 min.
            if trunc > 0.01:
                sys.exit(f"ABORTA: truncamiento {trunc:.4f} > 0,01 — se estaria midiendo el padding")

    if a.pesos:
        with open(a.pesos, "wb") as f:
            pickle.dump({"params": jax.device_get(params), "config": vars(a),
                         "vocab": I.ITOS}, f)
        print(f"pesos guardados en {a.pesos} ({os.path.getsize(a.pesos)/1e6:.1f} MB)", flush=True)
    print("\nlisto:", hist[-1] if hist else "sin evaluaciones")


if __name__ == "__main__":
    main()
