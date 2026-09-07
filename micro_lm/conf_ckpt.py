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
