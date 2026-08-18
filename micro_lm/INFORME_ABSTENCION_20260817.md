# El currículum de abstención funciona, pero sólo con margen de sobra

**2026-08-17** · campaña `x` (p_nose = 0,4) sobre checkpoints de la campaña base ya saturados.
9 modelos entrenados, 4 GPU T4 en paralelo + 1 corrida local.

## El resultado

La compuerta pide **`nose ≥ 0,50` Y `falsa_abst ≤ 0,10`**. Las dos juntas: el modelo que se abstiene
de todo saca `nose = 1,0000` y hay que poder rechazarlo (verificado con `test_metricas_nose.py`,
10/10 comprobaciones el mismo día).

| unidad | margen sobre el atajo | vigente | nose | falsa_abst | compuerta |
|---|---:|---:|---:|---:|---|
| x1_s0 | +0,4094 | 0,9890 | 0,8844 | 0,0095 | **PASA** |
| x2_s0 | +0,4094 | 0,9917 | 0,8635 | **0,0000** | **PASA** |
| x1_s1 | +0,4094 | 0,9370 | 0,9673 | 0,0604 | **PASA** (paso 12250, no cerró) |
| x2_s2 | +0,4071 | 0,9851 | 0,9762 | 0,0163 | **PASA** |
| x3_s2 | +0,2358 | 0,5656 | 0,6379 | 0,2237 | falla |
| x3_s0 | +0,1489 | 0,6203 | 0,5726 | 0,2109 | falla (paso 12500, no cerró) |
| x3_s1 | +0,1870 | 0,6388 | 0,5765 | 0,1757 | falla |
| x4_s1 | +0,1787 | 0,7004 | 0,6891 | 0,1599 | falla |
| x4_s0 | +0,1672 | 0,7047 | 0,7106 | 0,1342 | falla |

**4 de 4 pasan · 5 de 5 fallan.** La separación es total.

## Qué es el «margen»

`vigente` al terminar la campaña base, menos **0,5906**, que es lo que vale la estrategia degenerada
de *no abstenerse nunca* con `p_nose = 0,4`. Es cuánto mejor era el modelo que el atajo **antes** de
que se le introdujera `NOSE`.

## Lo que se puede afirmar

1. **El currículum funciona.** El colapso del 15-ago (falsa_abst 1,0000) no era incapacidad: era el
   punto de introducción equivocado, con el mecanismo rindiendo 0,1296 contra un atajo de 0,4094.
2. **`nose` nunca es el problema.** Está entre 0,57 y 0,98 en los nueve. Todos detectan la ausencia.
   **Lo que falla siempre es `falsa_abst`: se callan de más.**
3. **El criterio operativo del 15-ago hay que corregirlo.** Decía «introducir NOSE cuando `vigente`
   supere la tasa de ausencias (0,4094)». Los nueve modelos la superaban. La vara que separa es el
   margen contra **el atajo**, no contra la tasa.

## Lo que NO se puede afirmar

- **La frontera no está medida.** No hay ningún punto entre +0,2358 y +0,4071. La separación es
  perfecta pero el corte podría estar en cualquier lado de ese hueco.
- **Dentro del grupo que falla, el margen no ordena nada.** x3_s2 tiene el margen más alto del grupo
  (+0,2358) y el *peor* `falsa_abst` (0,2237). Si el margen fuera una variable continua que gradúa
  el resultado, esto no debería pasar. Puede ser umbral y no pendiente — o puede ser que el margen
  no sea la variable correcta y esté correlacionado con otra.
- **Falta el control que puede tumbar todo:** el gemelo de permutación de etiquetas (asignar `NOSE`
  al azar al 40 % de las preguntas, tengan respuesta o no). Hereda la frecuencia sin la señal.
- Dos corridas (x1_s1, x3_s0) quedaron sin cerrar; sus últimos valores ya son estables pero no son
  el punto final.

## Una pista estructural, medida sobre los pesos

El vector del token `NOSE` tiene norma **0,367**, contra **1,011** de «ana» y **1,028** de «beto».
El token de abstención compite en el mismo softmax que los valores con un vector **tres veces más
corto**. Es consistente con el diagnóstico externo: `NOSE` como token entrelaza una decisión binaria
y balanceada («¿está?») con una de 1-entre-100 («¿qué valor?»). **La cabeza de abstención separada
es el próximo paso, y ahora hay un número que la motiva.**
