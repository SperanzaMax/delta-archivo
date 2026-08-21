# Estado al cierre del 2026-08-20 (23:30) — para retomar mañana

## Nada quedó corriendo

Rotadores, tramos y watchdogs parados; las dos sesiones de Colab (`tr2_f_2226` en F, `tr2_n_2234` en
N) **paradas a mano** y verificado cuenta por cuenta: las 12 dan «No active sessions». Los
checkpoints viven en la PC y están todos en disco.

## Lo que falta, en una línea

**`t4_s2` es la única unidad sin extender** (sigue en 14000 de 20000). Todo lo demás de
`PREREG_PRESUPUESTO_TOKEN.md` está medido.

## P-2 YA SE CUMPLE, y sin `t4_s2`

El criterio decía: «≥ 3 de 5 pasan → la ventaja de `cabeza` del 18-ago estaba en parte comprada con
presupuesto: el hallazgo queda ACOTADO». Con el instrumento declarado (2048 muestras, rng 77000 +
semilla, regla de decisión por condición):

| unidad | cond | 14000 | 20000 | |
|---|---|---:|---:|---|
| t4_s0 | token | 0,1567 | 0,1296 | mejora, **falla** |
| t4_s1 | token | 0,2081 | **0,0713** | **PASA** |
| t4_s2 | token | 0,1942 | — | **pendiente** |
| s4_s0 | escala | 0,2244 | **0,0485** | **PASA** (`vigente` 0,6150 → 0,8811) |
| s4_s1 | escala | 0,1701 | **0,0722** | **PASA** |
| c4_s0 | cabeza | 0,0835 | 0,0633 | ya pasaba |

**3 de 5 pasan y las 5 fallaban a 14000** (el §2 quedó resuelto antes de mirar nada extendido, para
no repetir el error de R-3 de la réplica). `t4_s2` puede mover el conteo a 4/5 pero **no puede
bajarlo de 3**, así que el desenlace de P-2 no depende de ella.

**Consecuencia, comprometida por adelantado:** el hallazgo del 18-ago —«la cabeza pasa la compuerta
en 4 de 5 unidades donde `token` y `escala` fallan 5 de 5»— **queda ACOTADO: se midió con todas las
unidades a 14000, y con el mismo presupuesto que se le dio a `cabeza` tres de las cinco pasan.**
Lo que sobrevive sin tocar es el margen: `cabeza` llegaba a `falsa_abst` ≈ 0,06-0,08 a 14000, donde
las otras estaban en 0,16-0,22.

**Falta para cerrar el informe:** `t4_s2` extendida y medida, P-1 (Spearman sobre los puntos nuevos,
que necesita las series de `corridas_20260820/`), P-3 (pareado a 20000 contra `cabeza`) y P-4
(`vigente` no cae más de 0,10 — por ahora sube en las cuatro medidas).

## El round-trip quedó cerrado hoy

`INFORME_ROUNDTRIP_20260820.md`: archivado sin interpretar por RT-3 (8/8), con la causa identificada
por los dos diagnósticos —**`err_identidad` vive entero en la colisión de relación**: 0,005-0,014 con
relación única contra 0,38-0,54 con relación repetida, que es el azar entre las dos que empatan— y
con el control de presupuesto (a 20000 baja a 0,18-0,25, o sea **no es techo estructural**).

**La vía que abre, sin correr y sin prereg todavía:** el error dominante ocurre exactamente cuando
**dos entradas del archivo empatan en la clave de lectura**, y eso es objetivo, medible en el paso de
atención y **sin etiquetas** —dos pesos altos y parecidos—. Es lo que el `INFORME_MONITOR` pedía y no
encontraba: separa «anclado en la entrada correcta» de «anclado en cualquier entrada».

## Para arrancar mañana

1. Relanzar sólo `t4_s2`:
   `PREFIJO=t ABST=token P_NOSE=0.4 ./rotar_abst.sh 4:2 20000 3000 500 H K L M N I G`
2. Medirla: `medir_compuerta.py ckpts/t4_s2.pkl` (congelar `.p20000` antes, regla D-1).
3. Escribir `INFORME_PRESUPUESTO_TOKEN.md` con P-1..P-4.
4. Y la decisión de fondo: si se escribe el prereg del **empate de clave**, que es la mejor vía
   abierta para el corte sin etiquetas.
