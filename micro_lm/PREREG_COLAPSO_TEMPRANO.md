# Pre-registro · ¿el colapso de atención se ve en el paso 2.500?

2026-09-08, escrito con seis unidades observadas y **antes de correr las seis nuevas**.

## 1. De dónde sale

El escalón 2 (`INFORME_FRONTERA_E2_20260908.md`) dejó que con proyecciones libres la atención de
lectura tiene **dos atractores**, y que el colapso es **permanente**, o sea que las unidades caídas
gastan 26.000 pasos sin salir. Eso vuelve caro el diseño, no por lo que cuesta la unidad buena sino
por lo que cuestan las malas.

Mirando hacia atrás las seis unidades ya corridas apareció esto en el paso 2.500, que es el 9,6 %
del presupuesto.

| | `abstencion` | `nose` | destino |
|---|---|---|---|
| `ap3_s0`, `ap3_s1` (colapsaron) | 0,0723 · 0,0938 | 0,1231 · 0,1121 | muertas |
| `ap3_s2`, `at3_s0/s1/s2` (sanas) | 0,1621 a 0,4688 | 0,3259 a 0,6967 | al techo |

**Seis de seis, con un hueco de 0,20 en `nose` entre los dos grupos.** Es post-hoc y por eso se
pre-registra antes de agregar datos.

**Nota de honestidad sobre el parecido con el paper del atractor.** Ese paper reporta que «cero
contra no cero de respuestas emitidas en el paso 2.500» separa 40 de 40 corridas. La estructura es
la misma —una bifurcación decidida temprano y legible en la evaluación que el loop ya computa— pero
**el mecanismo NO es el mismo**. Allá es un objetivo auto-referencial que colapsa a un detector
constante; acá es la atención que colapsa sobre su propia posición. Se anota como analogía
estructural y no como el mismo fenómeno.

## 2. Hipótesis

**H1.** Un umbral en `nose` en el paso 2.500 separa a las unidades que terminan sanas de las que
colapsan. **Se pre-registra el umbral en 0,20**, que es el punto medio del hueco observado.

Se confirma si clasifica bien **al menos 8 de las 9** unidades `attnp` (las 3 ya corridas más las 6
nuevas). Se refuta si falla en 2 o más.

**H2.** La tasa de colapso de `attnp` está entre **0,4 y 0,8**. Con 3 de 3 observadas dio 0,67, que
es un intervalo de confianza inútil. Con 9 unidades el intervalo baja a algo reportable. Se refuta
si cae fuera de ese rango.

**H3.** `attn` **no colapsa nunca**. Cero de las 3 corridas colapsó y el mecanismo no existe sin
proyecciones. Se refuta si alguna unidad `attn` cae bajo el umbral en el paso 2.500. No se corren
unidades nuevas de `attn` para esto, se usa lo ya medido, así que es una predicción sobre datos
existentes y se declara como tal.

## 3. Diseño, congelado

Seis semillas nuevas, `ap3_s3` a `ap3_s8`, **idénticas a las tres primeras salvo la semilla**.

**Sólo 4.000 pasos por unidad, no 26.000.** Es el punto del pre-registro. Si H1 es cierta, con 4.000
alcanza para clasificar, y eso baja el costo de la campaña de 156.000 pasos a 24.000. Si H1 es
falsa, el ahorro no existía y hay que decirlo.

    PREFIJO=ap SELLO=abs PERT=0 SES_EXTRA=0 KERNEL_Q=5 DONDE=attnp \
      P_NOSE=0.4 ABST=cabeza SEMBRAR=0 HORIZONTE=26000 \
      ./rotar_abst3.sh 3:3,3:4,3:5 4000 2000 500 <cuentas>

**`HORIZONTE=26000` y no 4.000**, a propósito. La curva de learning rate tiene que ser la misma que
la de las tres primeras, o las unidades no son comparables. Se corta en 4.000 y la curva no cambia.

**Clasificación al cierre**: `nose` en el paso 2.500 contra el umbral 0,20, y el veredicto se
contrasta con `salud_attnp.py` sobre los pesos del paso 4.000, donde el colapso ya es visible
(medido, 1,41 y 2,08 posiciones efectivas contra 20,69).

## 4. Para qué sirve si se confirma

Un criterio de reinicio temprano. Una unidad que en el paso 2.500 está bajo 0,20 se descarta y se
relanza con otra semilla, en vez de gastar los 23.500 restantes. Con una tasa de colapso de dos
tercios eso es la diferencia entre que el diseño con proyecciones sea usable o no lo sea.
