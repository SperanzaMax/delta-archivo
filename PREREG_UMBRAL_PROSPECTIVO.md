# PREREG — ¿El techo de `c3_s0` es de calibración? Ahora prospectivo

**Estado:** CONGELADO 2026-08-19 antes de escribir una línea de `sonda_umbral_prospectiva.py` y antes
de mirar un solo número que no esté ya publicado en `INFORME_CABEZA_20260819.md` §4 bis.
**Fecha:** 2026-08-19.

## §1 · Qué se está tratando de cerrar

`INFORME_CABEZA_20260819.md` §4 bis dejó dicho, y dejó dicho que no valía como confirmación:

> `c3_s0` —la única que falla bajo el prereg— pasa con otro corte. Con AUC 0,807 la información para
> decidir está en el logit; lo que está mal puesto es el punto de corte.

Con dos salvedades propias: es **post-hoc** (D-C4) y usa 256 muestras por unidad, ocho veces menos que
la evaluación oficial, con un error medido de 0,065 en `nose` sobre `c3_s1`.

`c3_s0` es **la única unidad que no pasa con `cabeza` en toda la serie**. Si su techo es de
calibración, la afirmación de la campaña pasa de «4 de 5» a «5 de 5 con el corte bien puesto», que es
una afirmación bastante más fuerte y por eso mismo necesita más que un análisis exploratorio.

## §2 · El diseño, y en qué se distingue del post-hoc

Lo que hace inválida la versión del 18-ago no es el split —ya lo tenía— sino que **el umbral se elige
y se juzga sobre lotes generados por el mismo `rng`**, con las mismas sesiones y los mismos hechos
repartidos en dos mitades. Acá las dos muestras son **independientes en los datos**:

| | muestra de AJUSTE | muestra de PRUEBA |
|---|---|---|
| semilla del generador | `90000 + semilla` (la de `evaluar`) | **`77000 + semilla`** (fijada acá, antes de correr) |
| tamaño | 32 lotes × 64 = 2048 | 32 lotes × 64 = 2048 |
| para qué | elegir `a*` | **el único número que se reporta** |

Ocho veces las muestras de la sonda del 18-ago, o sea al nivel de la evaluación oficial.

**El criterio de elección de `a*` no se toca:** entre los cuantiles 0,001 y 0,999 del logit en la
muestra de ajuste, se toma el que maximiza `nose` sujeto a `falsa_abst` ≤ **0,07** (margen, como el
18-ago) y `nose` ≥ 0,50. Se juzga con el criterio real, `falsa_abst` ≤ 0,10 y `nose` ≥ 0,50.

`a*` se imprime **antes** de evaluar la muestra de prueba y queda en el informe. No se elige un umbral
después de ver cómo le fue.

## §3 · Predicciones

- **U-1.** AUC de `c3_s0` en la muestra de prueba ≥ **0,75** (el 18-ago dio 0,807 con 256 muestras).
- **U-2 (la principal).** Con `a*` congelado en la muestra de ajuste, `c3_s0` cumple la compuerta en la
  muestra de prueba: **`falsa_abst` ≤ 0,10 y `nose` ≥ 0,50**.
- **U-3.** `a* > 0`. Con σ>0,5 la unidad falla por abstenerse **de más** (`falsa_abst` 0,1189), así que
  el corte correcto tiene que pedir más evidencia, no menos. Si el `a*` elegido saliera **negativo**,
  el mecanismo no es el que se cree y U-2 no se puede leer como calibración aunque se cumpla.
- **U-4 (las sanas no se rompen).** `c1_s0` y `c2_s0`, que pasan con σ>0,5, siguen pasando con su
  propio `a*`. Si el procedimiento rompe unidades sanas, no sirve como recomendación.

## §4 · Los controles, que van corran o no las predicciones

Son la parte que impide leer U-2 como más de lo que es.

- **C-A · desplazamiento contra forma.** Se mide `a*` también sobre el logit **estandarizado por
  unidad** (`z = (a − μ) / σ`, con μ y σ de la muestra de ajuste). Si en z el corte óptimo cae cerca de
  0 en todas las unidades, lo que está mal no es un umbral fino por unidad sino un **desplazamiento de
  la cabeza**, y el arreglo es un bias, no una calibración. Es un diagnóstico más específico y más
  barato; se reporta como tal.
- **C-B · transferencia.** El `a*` de `c3_s0` se aplica tal cual a `c3_s1` y `c3_s2`. Si transfiere, el
  sesgo pertenece a la **dificultad** (nivel 3) y no a la unidad, y entonces se puede fijar una vez.
  Si no transfiere, cada unidad necesita su corte y eso es mucho menos útil operativamente.
- **C-C · el nulo.** Umbral elegido igual pero sobre `a` **permutado al azar** respecto de las
  etiquetas, 20 repeticiones. Da la tasa de «pasa la compuerta» que se consigue **sin información**,
  con este mismo procedimiento de búsqueda sobre 400 cortes. Si el nulo pasa seguido, el
  procedimiento se pasa a sí mismo y U-2 no dice nada.

## §5 · Qué mata qué

- **U-2 falla** → el resultado del 18-ago era sobreajuste del split, el techo de `c3_s0` vuelve a ser
  indistinguible de capacidad, y la campaña se sostiene como «4 de 5» sin asterisco.
- **U-2 cumple y C-C pasa seguido** → no alcanza; hay que rehacerlo con menos cortes o más muestras.
- **U-2 cumple, C-C limpio y C-A dice desplazamiento** → el resultado es **más** útil que el buscado:
  se corrige con un bias, no con un umbral por unidad.
- **U-2 cumple y U-3 falla** → algo distinto de la calibración está pasando y hay que entenderlo antes
  de escribirlo.

## §6 · Alcance, dicho antes

Esto es **una unidad**, y mide **checkpoints ya entrenados**. No demuestra que entrenar con el umbral
corregido dé un modelo mejor. La afirmación máxima que puede sostener, si todo cumple, es esta y no
más: **la información para decidir cuándo callarse está en el logit de `c3_s0`, y el corte σ>0,5 no es
donde había que leerla.** Entrenar con el corte corregido es otro experimento y otro prereg.
