# Enmienda 2 al `PREREG_FRONTERA_E1.md` (SHA `050341b4`)

Fecha: 2026-09-08, escrita con `at3` en el paso 2000 de 26.000 y **antes de que exista ningún dato
de selectividad sobre pesos entrenados con acceso global**. Fija una constante que la enmienda 1
dejó sin fijar y que decide el veredicto.

## Lo que pasó

La enmienda 1 (`61e88855`) puso el umbral de H2' en `sel = TV(d=3)/TV(d=5) >= 1,5`, y tomó como
referencia el **1,12** de `attn` sobre pesos de `kq3_s0` medido el 4-sep con `N = 120`.

Al correr el juez sobre los controles apareció que **`v3_s0` da 1,44 con el mismo N = 120**, o sea
un checkpoint que nunca vio acceso global quedaba a 0,06 del umbral. Eso obligaba a decidir si 1,44
era señal o ruido antes de mirar `at3`.

**Es ruido.** Con `N = 600`, sobre cuatro unidades de control:

| checkpoint | sel con N = 120 | sel con N = 600 |
|---|---|---|
| `v3_s0`  | **1,44** | **1,05** |
| `kq3_s0` | 1,13 | 0,99 |
| `v3_s1`  | — | 0,95 |
| `kq3_s1` | — | 0,94 |

Con N = 600 las cuatro dan **0,94-1,05**, o sea `sel = 1,00` dentro del error: acceso global
presente y **completamente no selectivo**, que es exactamente lo que corresponde a pesos que nunca
entrenaron con él. Con N = 120 la misma cantidad se movía 0,39 en un solo checkpoint, casi el 80 %
del margen que separa la referencia del umbral.

## Lo que se fija

1. **El veredicto de H2' se mide con `N = 600`.** Con N = 120 la sonda no separa 1,0 de 1,5.
2. **La referencia empírica pasa a ser 1,00** (rango observado 0,94-1,05 en cuatro controles), no el
   1,12 que citó la enmienda 1. El umbral **1,5 no se toca**: con la referencia real medida queda
   holgado, que es lo que se quería.
3. Los controles se reportan junto al resultado, medidos con el mismo N y en la misma corrida.

No cambia H1, ni el diseño de la campaña, ni el umbral. Cambia el instrumento con el que se aplica
el criterio, y se declara antes de que el criterio tenga a qué aplicarse.

Datos: `control_attn_ruido_n600.json`. Juez: `selectividad_attn.py`.
