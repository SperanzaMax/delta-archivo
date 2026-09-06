# PRE-REGISTRO · ¿la tabla `ord` MUESTRA la bandera? · 2026-09-06

Se congela ANTES de mirar un solo peso. Medición sobre checkpoints ya entrenados, en CPU, segundos.

## 1. Por qué

Esta mañana quedó medido que el sello de orden **se comporta** como una bandera y no como un reloj
(`INFORME_RELOJ_O_BANDERA_20260906.md`, prereg `aabb4b20`): con el orden relativo intacto y el
episodio corrido dos lugares, la RECUP cae de 1,0000 a 0,50-0,57 del lado que cruza el umbral,
apilar todas las entradas ajenas en un mismo turno no cambia nada, e invirtiendo los turnos la masa
se va a los turnos altos.

Todo eso es **conducta**. La pregunta ahora es **mecanismo**: si el modelo aprendió una partición
de filas, la tabla `ord` —64 vectores de dimensión 128, uno por turno— tiene que mostrarla. Y si
mostrara en cambio una estructura ordenada y suave, habría una disociación que obligaría a revisar
la lectura conductual.

Es la contraparte de peso del resultado de conducta, y decide con qué reemplazar el sello.

## 2. Qué se mide

Sobre `ord` (64 × 128) de cada checkpoint:

- **Similitud coseno 64 × 64** entre todas las filas.
- **Contraste bloque**: similitud media *dentro* del bloque bajo (turnos 0-23, que el entrenamiento
  vio siempre como ajenos), *dentro* del bloque alto (24-63, el episodio) y *entre* bloques.
- **Estructura de orden dentro de cada bloque**: correlación entre `cos(ord[i], ord[j])` y la
  distancia `|i − j|`. Un reloj da correlación negativa fuerte (turnos cercanos parecidos); una
  bandera da correlación cerca de cero dentro del bloque.
- **Norma por fila**: cuánta señal lleva cada turno.
- **Primera componente principal** proyectada contra el índice de turno: un reloj da una rampa
  monótona; una bandera, un escalón.

Unidades: `lg3_s0/s1/s2` (entrenadas con archivo largo, las que aprendieron a descartar),
`lc3_s0/s1/s2` (control, entrenadas en archivo corto) y `kq3_s0` (el origen del que se sembraron).

## 3. Criterios, comprometidos por adelantado

- **G-1 (partición).** En las tres `lg3`, la similitud media *entre* bloques es menor que *dentro*
  de cada bloque, con una brecha ≥ 0,15. → los pesos muestran la bandera.
- **G-2 (sin reloj dentro del bloque).** En las tres `lg3`, la correlación entre similitud y
  distancia `|i − j|` **dentro del bloque bajo** es débil, |r| ≤ 0,25. → coherente con C-2, que
  mostró que el orden entre las viejas no se usa.
- **G-3 (lo aprendió el entrenamiento largo).** La brecha de G-1 es mayor en las `lg3` que en las
  `lc3` y en `kq3_s0`, en las tres comparaciones. → la partición se formó al ver archivos largos,
  no venía en la siembra.
- **G-4 (el que daría vuelta la lectura).** Si en las `lg3` aparece una rampa monótona en la
  primera componente (|r| ≥ 0,80 contra el índice de turno) **y** G-2 falla, entonces la tabla sí
  codifica orden continuo y la lectura conductual de la mañana necesita revisión.

Si G-1 y G-2 se cumplen pero G-3 no, el resultado es que la partición ya estaba antes: se reporta
así, y cambia el diagnóstico (sería una propiedad de la siembra, no de la campaña).

## 4. Qué se hace con el resultado

- **Bandera confirmada en los pesos:** el reemplazo tiene que quitarle a la fila su significado
  absoluto. Dos candidatos, y esta medición ayuda a elegir: (a) sello **relativo a la consulta**
  (la entrada lleva `turno_actual − turno_entrada`, no `turno_entrada`); (b) **frontera sorteada**
  por muestra en el entrenamiento, para que ninguna fila pueda significar «ajeno». Va con su propio
  pre-registro y su presupuesto de T4.
- **Disociación (G-4):** se para y se revisa la medición conductual antes de tocar la arquitectura.

## 5. Lo que NO puede decir

- Nada causal: son pesos de tres corridas de una campaña. La causa se prueba entrenando con la
  frontera sorteada y volviendo a correr `reloj_o_bandera.py`.
- Nada sobre turnos > 63, que no existen en la tabla.
