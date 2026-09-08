"""Aplicar la config de un checkpoint a los globals que deciden la ARQUITECTURA · 2026-09-07

Regla que salio del 6-sep (`RETOMAR.md` §3): todo instrumento que carga un checkpoint tiene que
leer de su config TODO lo que decide la arquitectura. Ese dia le falto a dos: `ser.py` no leia
`kernel_q` y `reloj_o_bandera.py` no leia `sello`. Los dos habrian medido con una arquitectura que
no era la del checkpoint, y el resultado habria sido creible.

El problema de fondo es que cada instrumento repetia las dos lineas a mano, asi que agregar un
tercer global obligaba a acordarse de 17 archivos. Aca esta la lista UNA vez: quien llame a
`aplicar(cfg)` queda cubierto para siempre, incluido lo que se agregue despues.

    import conf_ckpt
    b = pickle.load(open(ruta, "rb"))
    cfg, params = b["config"], b["params"]
    conf_ckpt.aplicar(cfg)

Los defaults son los valores historicos del modulo, no los actuales: un checkpoint viejo no lleva
la clave y tiene que seguir midiendose como se midio.
"""
# `modelo` se importa DENTRO de `aplicar`, no aca arriba: importarlo a nivel de modulo arrastra
# jax, y entonces `auditar_instrumentos.py` —que solo necesita la tabla de abajo— no correria en un
# python sin el entorno. La auditoria tiene que poder correrse siempre, o no se corre nunca.

# global del modulo -> (clave en la config, default historico)
GLOBALS = {
    "KQ": ("kernel_q", 3),      # kernel de `convq` (lat2); 5 desde el 1-sep
    "SELLO": ("sello", "abs"),  # indexacion de `ord`; "rel" desde el 6-sep
}


def aplicar(cfg, verboso=False):
    """Fija en `modelo` todo lo que la config del checkpoint decide. Devuelve lo aplicado."""
    import modelo as M

    puesto = {}
    for nombre, (clave, default) in GLOBALS.items():
        # `or default` y no solo el default de `.get`: hubo configs que guardaron None en `kernel_q`
        # (ver `dilucion.py`), y un None ahi rompe la conv en vez de caer al valor viejo.
        valor = cfg.get(clave, default)
        if valor is None:
            valor = default
        setattr(M, nombre, valor)
        puesto[nombre] = valor
    if verboso:
        print("arquitectura del checkpoint: " + " · ".join(f"{k}={v}" for k, v in puesto.items()))
    return puesto


def descripcion(cfg):
    """Una linea para el encabezado de los instrumentos, con lo que decide la arquitectura."""
    p = {n: cfg.get(c, d) or d for n, (c, d) in GLOBALS.items()}
    return (f"kernel_q={p['KQ']} · sello={p['SELLO']} · donde={cfg.get('donde')} · "
            f"nivel={cfg.get('nivel')} · ses_extra={cfg.get('ses_extra', 0)} · "
            f"pert={cfg.get('pert', False)} · paso {cfg.get('pasos')}")


# ---------------------------------------------------------------------------------------------
# ARGUMENTOS que la config decide, y que `aplicar` NO puede cubrir · 2026-09-08
#
# `aplicar` fija los globals del modulo. Hoy aparecio la otra mitad del mismo defecto y hay que
# nombrarla: `reloj_o_bandera.py` leia bien `sello` y `kernel_q` de la config y aun asi medio las
# tres unidades `rp3` con una arquitectura que no era la suya, porque el bit de pertenencia no es
# un global sino un ARGUMENTO de `modelo.responder`. La funcion `leer()` del instrumento
# reimplementa `responder` a mano para guardar la distribucion de lectura, y al copiarla se quedo
# sin `marca_pert`.
#
# Lo que costo: acierto 0,2969-0,5781 en la sonda contra `vigente` 0,9725-0,9824 en el propio
# entrenamiento de esas mismas unidades. La recuperacion aguantaba y la respuesta se caia, que es
# la firma de medir un modelo sin la entrada de la que aprendio a depender — y se leia como un
# hallazgo sobre el bit de pertenencia.
#
# La regla del RETOMAR §3 queda mas ancha: **todo instrumento que carga un checkpoint tiene que
# reproducir de su config todo lo que decide la arquitectura, sean globals del modulo o argumentos
# de llamada.** Y el corolario, que es el que muerde: **una funcion que reimplementa `responder`
# no hereda sus defaults**; cada vez que `responder` gana un argumento, esa copia queda vieja y en
# silencio.

def pertenece_de(cfg, mask):
    """El argumento `pertenece` que le corresponde a este checkpoint, o None si no lleva bit.

    Replica `entrenar.py:pert_de`: las primeras 4*E_MAX entradas del archivo son el episodio en
    curso. Se pasa el `mask` sólo para tomarle la forma.
    """
    if not cfg.get("pert"):
        return None
    import numpy as np
    import jax.numpy as jnp
    import datos as DAT

    n_pri = 4 * DAT.E_MAX
    m = np.zeros(mask.shape[-1], bool)
    m[:n_pri] = True
    return jnp.array(np.broadcast_to(m, mask.shape))
