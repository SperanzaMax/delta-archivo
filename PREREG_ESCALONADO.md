# PRE-REGISTRO · ENTRENAMIENTO ESCALONADO POR CAPACIDAD

Escrito el 2026-08-23 por la mañana, **antes** de lanzar la campaña. El diseño está en
`DISENO_ESCALONADO.md` (22-ago, noche); esto fija los números y lo que cuenta como respuesta.

## 1. La pregunta

Si el presupuesto de muestras se reparte entre tipos de pregunta **según cuánto le falta a cada
uno**, en vez de con una mezcla fija elegida a mano, ¿el modelo aprende mejor, más rápido, o
ninguna de las dos?

Idea de Maxi: *«¿por qué todo tiene que terminar a 4000? Que cada cosa termine cuando le conviene y
el resto continúe hasta su turno.»*

## 2. Por qué la pregunta no es ociosa

`w3_s0` al paso 15750, medido: `vigente` **1,0000** · `anterior` **0,9792** · `nose` **0,7844**
(`nose_ent` 0,9374 / `nose_rel` 0,6090). Las dos primeras están terminadas y se siguen llevando el
**60 %** de las muestras hasta el paso 26000. Lo único que falta es `nose_rel`.

Y la regla propuesta, alimentada con la curva real de esa corrida, habría hecho esto:

| paso | vigente | anterior | nose | → w_vig | w_ant | w_nose |
|---|---|---|---|---|---|---|
| 1500 | 0,7693 | 0,2815 | 0,1521 | 0,298 | 0,336 | 0,366 |
| 6000 | 0,8188 | 0,2511 | 0,5226 | 0,215 | 0,421 | 0,364 |
| 12000 | 0,9017 | 0,4181 | 0,7287 | 0,181 | 0,546 | 0,273 |
| 15000 | 1,0000 | 0,9635 | 0,7864 | 0,171 | 0,439 | 0,390 |
| 15750 | 1,0000 | 0,9792 | 0,7844 | 0,164 | 0,401 | 0,435 |

(la mezcla fija de hoy es 0,390 / 0,210 / 0,400 y no se mueve nunca)

El presupuesto va primero a `anterior` —la capacidad que despega tarde, como ya había medido
E-I3b— y **vuelve** a `nose` recién cuando `anterior` se resuelve. Sube y después baja: un umbral
elegido a mano no produce esa forma.

## 3. La regla, con sus números

`p(tipo) ∝ EMA(error del tipo)`, renormalizado, con:

- **alpha = 0,10** sobre las evaluaciones que ya se corren cada `--cada` pasos. Lenta a propósito:
  el error se mide con 512 muestras y tiene ruido de ±0,02, y además hay realimentación (el
  muestreo cambia el error que decide el muestreo).
- **piso = 0,10** por tipo. Sin piso, una capacidad resuelta deja de verse del todo y se olvida.
- Error inicial 1,0 en los tres → la mezcla arranca uniforme, sin privilegiar a ninguno.

La traducción a las dos palancas que `datos.lote` ya acepta es exacta y biyectiva, así que
`datos.py` —código compartido con las campañas cerradas— **no se toca**:

```
p(nose) = p_nose ;  p(anterior) = (1-p_nose)·p_vieja ;  p(vigente) = (1-p_nose)·(1-p_vieja)
```

## 4. Lo que se corre

Base `donde=pre` (la arquitectura establecida), `abst=cabeza`, `p_nose` de referencia 0,40,
nivel 3, d=128, capas=4, 20000 pasos, `--cada` 250, semillas 0/1/2.

| condición | prefijo | mezcla |
|---|---|---|
| dinámica | `ed` | `--mezcla dinamica` (piso 0,10 · alpha 0,10) |
| fija | `ef` | `--mezcla fija --p-vieja 0.35 --p-nose 0.4` |
| **control** `fijo_promedio` | `ep` | `--mezcla fija` con el promedio que la dinámica terminó usando |

`ef` se corre **fresca y pareada** aunque haya campañas previas con esa configuración. Es la misma
razón por la que la campaña `token` del 17-ago no se reusó como línea de base.

`ep` no se puede lanzar ahora: su mezcla es un **resultado** de `ed`. La corrida dinámica la imprime
al terminar (`CONTROL fijo_promedio: --mezcla fija --p-vieja … --p-nose …`).

## 5. La medición, y la trampa que evita

**La mezcla de entrenamiento se mueve; la de evaluación nunca.** Todas las condiciones se evalúan
con la mezcla de referencia (`p_vieja` 0,35 · `p_nose` 0,40). Si la evaluación siguiera a la mezcla
dinámica, cada condición se estaría midiendo sobre una población de preguntas distinta y el número
cambiaría por **cómo se midió**, no por lo que el modelo aprendió.

El acierto global se calcula como la media de los tres tipos **ponderada por la mezcla de
referencia**, no como un promedio simple: es la misma cantidad en las tres condiciones.

## 6. Predicciones

- **S-0 · bloqueante.** `dinamica` aprende: acierto global ≥ el de `fija` menos 0,02. Si escalonar
  rompe el modelo, no hay experimento.
- **S-1 · principal.** A igual número de pasos (20000), `dinamica` > `fija` en acierto global, en
  al menos **2 de 3** semillas.
- **S-2 · la que le importa a la GPU.** `dinamica` alcanza el acierto final de `fija` en **menos
  pasos**. Es la que convierte la idea en ahorro medible, y la única que la justifica si S-1 empata.
- **S-3 · el control.** `dinamica` > `fijo_promedio`. Si empatan, lo que importaba era la
  proporción y no el escalonamiento, y esto se reporta como **«mejor mezcla»**, no como
  «curriculum». Es la jugada de `barajado` en E-I3, donde la celda que **no** ganó fue la que hizo
  válido el resultado.
- **S-4 · sin olvido.** Ninguna capacidad termina por debajo de donde estaba cuando dejó de
  muestrearse. Es para lo que existe el piso, y hay que verificar que alcance.

## 7. Riesgo declarado

Si una capacidad fuera **imposible** en vez de difícil, absorbería muestreo para siempre y
degradaría a las demás hasta el piso. En esta tarea las tres se alcanzan —está medido—, pero si el
escalonado se lleva a una tarea nueva, ese es el primer control a correr.

## 8. Lo que no se mezcla

Esto **no** se corre sobre `lat`. Las tres unidades `w3_*` siguen con el entrenamiento estándar
para que la pregunta de la query conjunta se responda limpia. Dos cambios en la misma campaña es
exactamente el error de `post`. Corren en paralelo como campañas separadas, cada una con su
control, sobre bases distintas.

## 9. Verificaciones hechas antes de lanzar

- `--mezcla fija` reproduce **bit a bit** la corrida anterior (`vigente` 0,010146103896103896 y
  `anterior` 0,031746031746031744, idénticos): el cambio no toca las campañas en curso.
- Ida y vuelta pesos ↔ probabilidades: identidad en 5 casos, incluido el degenerado `p_nose = 0`.
- La guarda de identidad aborta si un tramo reanuda una corrida `dinamica` sin el flag —el mismo
  agujero que tapó la guarda de `donde` el 22-ago—. Verificado contra un checkpoint real.
- La EMA y el acumulado de la mezcla viajan **dentro del checkpoint**: la campaña corre por tramos
  de 8000 pasos entre cuentas, y sin eso cada tramo reiniciaría la política.
