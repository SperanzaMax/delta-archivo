# El colapso de atención se ve en el paso 2.500

2026-09-08. Evalúa `PREREG_COLAPSO_TEMPRANO.md` (SHA `0e0e7745`), congelado antes de correr las seis
unidades nuevas. **Las tres hipótesis se confirman.**

## 1. El resultado

Nueve unidades `attnp`. Umbral pre-registrado, `nose >= 0,20` en el paso 2.500, que es el **9,6 %**
del presupuesto de 26.000.

| unidad | `nose` en 2.500 | predicción | posiciones efectivas | mecanismo | acuerdo |
|---|---|---|---|---|---|
| `ap3_s8` | 0,0000 | COLAPSA | 1,34 | COLAPSA | sí |
| `ap3_s4` | 0,0107 | COLAPSA | 1,78 | COLAPSA | sí |
| `ap3_s6` | 0,0576 | COLAPSA | 1,33 | COLAPSA | sí |
| `ap3_s1` | 0,1121 | COLAPSA | 2,08 | COLAPSA | sí |
| `ap3_s0` | 0,1231 | COLAPSA | 1,41 | COLAPSA | sí |
| `ap3_s2` | 0,3818 | sana | 20,69 | sana | sí |
| `ap3_s5` | 0,4152 | sana | 11,24 | sana | sí |
| `ap3_s7` | 0,4974 | sana | 8,62 | sana | sí |
| `ap3_s3` | 0,5759 | sana | 9,70 | sana | sí |

**Nueve de nueve.** El pre-registro pedía al menos 8. El hueco entre los dos grupos va de **0,1231 a
0,3818** y el umbral está en el medio, sin ninguna unidad cerca.

**H2 confirmada.** Tasa de colapso **5 de 9 = 0,556**, dentro del 0,4 a 0,8 pre-registrado.

**H3 confirmada** sobre datos existentes, como se declaró. Las tres unidades `attn` tenían `nose` de
0,3259, 0,6967 y 0,3707 en ese paso, las tres del lado sano, y ninguna colapsó.

## 2. La señal conductual llega ANTES que la geométrica, y esto no estaba previsto

`ap3_s5` es el caso que lo muestra y por poco no se ve.

En el paso 2.500 su `nose` era 0,4152, o sea claramente del lado sano, **pero sus posiciones
efectivas eran 4,57**, un valor intermedio entre el rango colapsado (1,33 a 2,08) y el sano (8,62 a
20,69). Medida sólo por el mecanismo, esa unidad era ambigua. Para el paso 4.000 subió a **11,24** y
la ambigüedad desapareció.

**La conducta se define antes que la geometría.** El `nose` del paso 2.500 acertó sobre una unidad
cuya atención todavía no había terminado de formarse. Es post-hoc, con un solo caso, y se anota como
observación y no como resultado, pero apunta a que el umbral podría bajarse todavía más.

## 3. Lo que esto compra

Una unidad bajo 0,20 en el paso 2.500 se descarta y se relanza con otra semilla, en vez de gastar
los 23.500 restantes. **Con una tasa de colapso del 56 % eso decide si el diseño con proyecciones es
usable o no.** Sin el criterio, más de la mitad del cómputo se va en unidades que ya están muertas y
no se sabe.

El ahorro está medido en esta misma campaña. Clasificar nueve unidades costó **24.000 pasos en vez
de 156.000**, que era exactamente la apuesta del §3 del pre-registro.

## 4. Lo que NO dice

**No previene el colapso, lo detecta.** La pregunta de si se puede evitar sigue abierta y sus
candidatos siguen sin probar, temperatura en la atención, congelar las proyecciones los primeros
pasos, penalizar entropía baja.

**Y el umbral es de esta tarea y este banco.** 0,20 sale del hueco observado acá; en otro régimen
habría que recalibrarlo. Lo transferible no es el número sino el método, mirar la métrica que el
loop de entrenamiento ya computa en un paso temprano y ver si separa.

## 5. Archivos

`corridas_20260908/ap3_s*.json` (nueve unidades), `salud_attnp.py`, `PREREG_COLAPSO_TEMPRANO.md`.
