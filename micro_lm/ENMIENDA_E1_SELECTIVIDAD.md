# Enmienda 1 al `PREREG_FRONTERA_E1.md` (SHA `050341b4`)

Fecha: 2026-09-08, escrita **antes** de que termine la primera unidad `at3`. No cambia H1 ni el
diseño; **operacionaliza H2**, que en el prereg quedó como «la magnitud no decae con la distancia» y
así redactada se puede cumplir por la razón equivocada.

## El problema con H2 tal como estaba escrita

Un promedio UNIFORME del pasado también da TV > 0 en las seis distancias y también es plano. Los
números del 4-sep sobre pesos de `lat2` son exactamente eso:

    attn (pesos lat2, kq5)   d1 0,0103 · d2 0,0107 · d3 0,0109 · d4 0,0104 · d5 0,0097 · d6 0,0098

Plano y sin decaimiento, o sea que cumpliría H2 al pie de la letra sin que el modelo haya aprendido
nada. Y hoy quedó medido por qué: el perfil de atención de `attn_causal` es plano de d1 a d23
(0,0213-0,0268) fuera de la propia posición. **Sin selectividad, «ve todo» y «no mira nada» dan la
misma firma.**

## Lo que se agrega

**H2'. Razón de selectividad.** Sobre los pesos de `at3`, con `control_attn.py --donde attn`:

    sel = TV(d = 3) / TV(d = 5)

d = 3 es donde cae la RELACION en la forma canónica del idioma, que es la posición cuya ceguera con
kernel 3 explicó el corte. d = 5 no lleva señal en ninguna forma y es el piso.

- **Se confirma** si `sel >= 1,5` en al menos dos de las tres semillas. El modelo no sólo alcanza la
  posición: la pesa más que al ruido.
- **Se refuta** si `sel` queda en el orden de 1,0 (los pesos de `lat2` dan **1,12**) con `nose_rel`
  igual alto. Eso significaría que el acceso global no es el mecanismo por el que sube `nose_rel`, y
  H1 quedaría cumplida por una causa distinta de la pre-registrada — el resultado más informativo de
  los tres y el que obliga a buscar la causa alternativa antes de escribir nada.

La segunda rama es la que hace que valga la pena escribir esto hoy y no después de ver los números.
