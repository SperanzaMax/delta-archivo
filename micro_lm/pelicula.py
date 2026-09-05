"""PELICULA DE PESOS · el micro-LM aprendiendo, cuadro por cuadro · 2026-09-05

Pedido de Maxi: ver EN MOVIMIENTO REAL como se conectan las neuronas mientras se entrena. No es una
animacion ilustrativa: cada cuadro son los pesos de verdad del modelo en ese paso.

Que hace. Corre el entrenamiento con la MISMA perdida (`entrenar.perdida`), el MISMO generador
(`datos.lote`) y el MISMO optimizador que `entrenar.py` (adamw con clip 1.0, warmup + cosine), y cada
`--cada` pasos guarda:

  · una submatriz FIJA de 12x12 de cada matriz del camino de un dato, de punta a punta;
  · los taps de `convq` del bloque 0, que son la VENTANA de la query (INFORME_QUERY_CIEGA_20260901);
  · el `beta` medio de cada bloque, que es la compuerta de escritura de la regla delta;
  · la distribucion de lectura del archivo para UNA muestra fija — el softmax que el modelo usa de
    verdad, capturado desde adentro de `responder`, no reconstruido; y
  · perdida y exactitud del lote.

Los indices de la submatriz se sortean UNA vez y no cambian entre cuadros: si no, el movimiento que se
ve seria el del muestreo y no el del aprendizaje.

LIMITE DECLARADO. Esto no reemplaza a `entrenar.py` ni produce un checkpoint de campania: no aplica el
curriculum de mezcla, corre pocos pasos y su salida es para MIRAR. La trayectoria es la del mismo
sistema con esos hiperparametros; los numeros publicados salen de `entrenar.py`.

Uso:
    python3 pelicula.py --pasos 1000 --cada 20 --salida pelicula.json
"""
import argparse, json, time
import numpy as np
import jax, jax.numpy as jnp, optax

import datos as DAT
import entrenar as E
import idioma as I
import modelo as M

K = 12          # unidades por columna en el dibujo


def submatrices(params, rng, k=K):
    """Indices FIJOS de la submatriz de cada matriz del camino. Se sortean una sola vez."""
    NB = len(params["blocks"])
    rutas = [("emb", ("emb",), "tokens -> d")]
    for i in range(NB):
        rutas += [(f"b{i}.wq", ("blocks", i, "wq"), f"bloque {i} · query"),
                  (f"b{i}.wk", ("blocks", i, "wk"), f"bloque {i} · clave"),
                  (f"b{i}.wv", ("blocks", i, "wv"), f"bloque {i} · valor"),
                  (f"b{i}.m1", ("blocks", i, "m1", "w"), f"bloque {i} · MLP entrada"),
                  (f"b{i}.m2", ("blocks", i, "m2", "w"), f"bloque {i} · MLP salida")]
    rutas += [("arch.kw", ("arch", "kw"), "archivo · escribe la clave"),
              ("arch.vw", ("arch", "vw"), "archivo · escribe el valor"),
              ("arch.qr", ("arch", "qr"), "archivo · forma la consulta"),
              ("arch.wo", ("arch", "wo"), "archivo · devuelve lo leido"),
              ("head", ("head", "w"), "d -> vocabulario")]
    plan = []
    for nombre, ruta, etiqueta in rutas:
        w = params
        for p in ruta:
            w = w[p]
        f = np.sort(rng.choice(w.shape[0], min(k, w.shape[0]), replace=False))
        c = np.sort(rng.choice(w.shape[1], min(k, w.shape[1]), replace=False))
        plan.append({"nombre": nombre, "ruta": ruta, "etiqueta": etiqueta,
                     "forma": list(w.shape), "filas": f, "cols": c})
    return plan


def leer(params, ruta):
    w = params
    for p in ruta:
        w = w[p]
    return w


def atencion_archivo(params, muestra, donde, bloque=0):
    """El softmax de lectura del archivo, capturado DESDE ADENTRO del forward del modelo.

    No se recalcula la formula por afuera: se le pasa a `tronco` la misma clausura `lectura` que usa
    `responder`, con un gancho que se guarda la distribucion. Asi lo que se dibuja es lo que el
    modelo leyo, no una reconstruccion.
    """
    ses, cortes, turnos, mask, cons, pos = muestra
    a = params["arch"]
    archivo = M.escribir(params, ses, cortes)
    ak = archivo @ a["kw"] + M.sello(a, turnos)
    av = archivo @ a["vw"]
    penal = jnp.where(mask, 0.0, -1e9)[:, None, :]
    guardado = {}

    def lectura(h):
        q = h @ a["qr"]
        sim = jnp.einsum("btd,bnd->btn", q, ak) / jnp.sqrt(h.shape[-1]) + penal
        p = jax.nn.softmax(sim, -1)
        guardado["p"] = p
        return jnp.einsum("btn,bnd->btd", p, av) @ a["wo"]

    h = M.tronco(params, cons, lectura, bloque, donde)
    logits = M.ln(params["ln_f"], h) @ params["head"]["w"] + params["head"]["b"]
    p = np.array(guardado["p"][0, int(pos[0])])      # (N,) la lectura en el ultimo token de la query
    return p, np.array(logits[0, int(pos[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pasos", type=int, default=1000)
    ap.add_argument("--cada", type=int, default=20)
    ap.add_argument("--nivel", type=int, default=3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--capas", type=int, default=4)
    ap.add_argument("--kernel-q", type=int, default=5)
    ap.add_argument("--donde", default="lat2")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--semilla", type=int, default=0)
    ap.add_argument("--p-vieja", type=float, default=0.35)
    ap.add_argument("--p-nose", type=float, default=0.2)
    ap.add_argument("--salida", default="pelicula.json")
    a = ap.parse_args()

    M.KQ = a.kernel_q
    params = M.init_params(a.semilla, I.V, D=a.d, NB=a.capas, N_TURNOS=64)
    print(f"parametros {M.contar(params):,} · d={a.d} capas={a.capas} kernel_q={a.kernel_q} "
          f"donde={a.donde}", flush=True)

    warmup = min(500, max(1, a.pasos // 10))
    sched = optax.warmup_cosine_decay_schedule(0.0, a.lr, warmup, a.pasos, a.lr * 0.1)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=0.01))
    state = opt.init(params)

    @jax.jit
    def paso(params, state, ses, cortes, turnos, mask, cons, pos, tgt):
        (l, acc), g = jax.value_and_grad(E.perdida, has_aux=True)(
            params, ses, cortes, turnos, mask, cons, pos, tgt)
        up, state = opt.update(g, state, params)
        return optax.apply_updates(params, up), state, l, acc

    plan = submatrices(params, np.random.default_rng(12345))

    # LA MUESTRA FIJA de la que se mira la lectura del archivo. Se elige una con respuesta (tipo
    # vigente) para que exista una entrada correcta que la atencion pueda encontrar.
    mrng = np.random.default_rng(77)
    while True:
        s, c, t, mk, q, pq, tg, tp, meta, orig, hq = DAT.lote(
            mrng, 8, nivel=a.nivel, p_vieja=0.0, p_nose=0.0, con_meta=True, con_origen=True)
        i = int(np.argmax(tp == DAT.TIPOS["vigente"]))
        if tp[i] == DAT.TIPOS["vigente"] and hq[i] >= 0:
            break
    muestra = (jnp.array(s[i:i+1]), jnp.array(c[i:i+1]), jnp.array(t[i:i+1]),
               jnp.array(mk[i:i+1]), jnp.array(q[i:i+1]), np.array(pq[i:i+1]))
    correctos = [int(x) for x in np.where((orig[i] == hq[i]) & mk[i])[0]]
    respuesta = I.ITOS[int(tg[i])]
    texto_ses = [" ".join(I.ITOS[int(z)] for z in fila if I.ITOS[int(z)] != ".") for fila in s[i]]
    texto_q = " ".join(I.ITOS[int(z)] for z in q[i] if I.ITOS[int(z)] != ".")
    print(f"muestra fija · slots correctos {correctos} · pregunta: {texto_q}", flush=True)

    rng = np.random.default_rng(1000 + a.semilla)
    cuadros, t0 = [], time.time()

    def armar():
        return {
            "config": {"pasos": a.pasos, "cada": a.cada, "nivel": a.nivel, "d": a.d,
                       "capas": a.capas, "kernel_q": a.kernel_q, "donde": a.donde,
                       "batch": a.batch, "lr": a.lr, "semilla": a.semilla, "p_vieja": a.p_vieja,
                       "p_nose": a.p_nose, "params": int(M.contar(params)), "k": K},
            "capas": [{"nombre": it["nombre"], "etiqueta": it["etiqueta"], "forma": it["forma"]}
                      for it in plan],
            "muestra": {"sesiones": texto_ses, "pregunta": texto_q, "correctos": correctos,
                        "slots": int(mk.shape[1]), "ocupados": [int(x) for x in np.where(mk[i])[0]]},
            "respuesta_correcta": respuesta,
            "cuadros": cuadros,
        }

    def guardar():
        with open(a.salida, "w", encoding="utf-8") as f:
            json.dump(armar(), f, separators=(",", ":"))

    def cuadro(n, l, acc):
        pesos = {}
        for it in plan:
            w = np.array(leer(params, it["ruta"]))[np.ix_(it["filas"], it["cols"])]
            pesos[it["nombre"]] = [round(float(x), 4) for x in w.ravel()]
        cq = np.array(params["blocks"][0]["convq"])
        att, lg = atencion_archivo(params, muestra, a.donde)
        cuadros.append({
            "paso": int(n), "perdida": round(float(l), 4), "acc": round(float(acc), 4),
            "pesos": pesos,
            "taps": [round(float(np.abs(x).mean()), 5) for x in cq],
            "beta": [round(float(jax.nn.sigmoid(b["beta"]).mean()), 4) for b in params["blocks"]],
            "atencion": [round(float(x), 5) for x in att],
            "pred": I.ITOS[int(np.argmax(lg))],
        })
        guardar()

    ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
        rng, a.batch, nivel=a.nivel, p_vieja=a.p_vieja, p_nose=a.p_nose)
    l0, acc0 = E.perdida(params, *[jnp.array(x) for x in
                                   (ses, cortes, turnos, mask, cons, pos, tgt)])
    cuadro(0, l0, acc0)

    rng = np.random.default_rng(1000 + a.semilla)
    for n in range(1, a.pasos + 1):
        ses, cortes, turnos, mask, cons, pos, tgt, _ = DAT.lote(
            rng, a.batch, nivel=a.nivel, p_vieja=a.p_vieja, p_nose=a.p_nose)
        params, state, l, acc = paso(params, state, jnp.array(ses), jnp.array(cortes),
                                     jnp.array(turnos), jnp.array(mask), jnp.array(cons),
                                     jnp.array(pos), jnp.array(tgt))
        if n % a.cada == 0 or n == a.pasos:
            cuadro(n, l, acc)
            print(f"  paso {n:5d} · perdida {float(l):6.3f} · acc {float(acc):.4f} · "
                  f"{time.time() - t0:5.1f}s", flush=True)

    guardar()
    print(f"listo · {len(cuadros)} cuadros · {a.salida} · {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
