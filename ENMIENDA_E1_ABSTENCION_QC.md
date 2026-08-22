# ENMIENDA E-1 al `PREREG_ABSTENCION_QC.md` (2026-08-22, antes de mirar la campania)

Se escribe con la campania de la query conjunta todavia entrenando (paso ~5500 de 26000) y **sin
haber medido una sola unidad `post`**. Lo que la motiva salio de una linea de base sobre `pre`
MADURO, o sea sobre los checkpoints `c3_s0/s1/s2` del 18-ago, que son la arquitectura vieja y no
tienen nada que ver con el contraste de la campania.

## Lo medido

`sonda_abstencion_qc.py`, 1536 muestras por unidad, `c3_s*` a 14000 pasos:

| unidad | `s1` | `s1` estratificado por tamaño | **nulo** | nulo estratificado |
|---|---:|---:|---:|---:|
| `c3_s0` | 0,5467 | 0,5509 | **0,5956** | 0,5967 |
| `c3_s1` | 0,5239 | 0,5218 | **0,5342** | 0,5332 |
| `c3_s2` | 0,5309 | 0,5459 | **0,5877** | 0,5928 |

Dos cosas, y la segunda es la que obliga a la enmienda:

1. **En `pre` maduro no hay señal**, como el cierre del 21-ago predice. `s1` esta en 0,52-0,55.
2. **El nulo NO da 0,50**, da 0,53-0,60. El §2 del pre-registro pedia, en A-3, que el nulo cayera
   entre 0,45 y 0,55; con este instrumento eso **no se cumple ni siquiera cuando no hay señal**.

## Por que el nulo no da 0,50, y por que eso no es un defecto

Primero se probo la explicacion obvia —el **tamaño** del episodio, porque `s1` es un maximo y el
maximo crece con la cantidad de elementos sobre los que se toma—. Se agrego estratificacion por
(posiciones de consulta, entradas validas) y **el nulo no se movio**: 0,5967 contra 0,5956. La
explicacion obvia era falsa y conviene dejarlo escrito, porque el codigo de estratificacion se
escribio para arreglar algo que no era el problema.

Lo que pasa es otra cosa. El nulo reemplaza los scores por gaussianas **de igual media y desvio por
posicion**, asi que por construccion **preserva la ESCALA de los scores del episodio**. Si las
preguntas sin respuesta tienen scores de escala distinta a las que si la tienen, el nulo hereda esa
diferencia. Entonces el nulo no mide «cero», mide exactamente **cuanto AUC se consigue con la escala
sola, sin ninguna estructura fina**. Que valga 0,53-0,60 no lo ensucia: lo vuelve informativo.

Y lo que dice sobre `pre` es mas fuerte que lo que decia el criterio original: **`s1` (0,52-0,55) esta
POR DEBAJO de su propio nulo (0,53-0,60)**. No es que la señal sea chica, es que no hay ninguna, ni
siquiera la que se obtendria mirando solo la escala.

## Que cambia

**A-3 se reescribe.** Deja de ser «el nulo cae entre 0,45 y 0,55» y pasa a ser una comparacion contra
el nulo de la misma unidad:

> **A-3 (nuevo, bloqueante).** La señal cuenta solo si `auc_s1 - auc_nulo >= 0,05` en la misma
> unidad. Un `auc_s1` alto con un nulo igual de alto **no es señal**: es la escala.

**A-1 y A-2 se leen sobre el margen, no sobre el valor crudo.** A-1 pide que
`(auc_s1 - auc_nulo)` sea mayor en `post` que en `pre` por >= 0,05 en al menos 2 de 3 semillas. A-2
mantiene el umbral de utilidad en `auc_s1 >= 0,75`, que es un umbral de magnitud absoluta y no de
contraste, pero solo se evalua si A-3 paso.

## Por que esto no es mover el arco despues de tirar

- Se decidio **antes de medir una sola unidad de la campania**, sobre checkpoints de otra
  arquitectura y de otro dia.
- El criterio nuevo es **mas exigente**, no menos: el original habria dado «nulo sucio, no hay
  resultado» y habria archivado el experimento sin mirarlo; el nuevo obliga a superar un piso que en
  estos datos vale 0,53-0,60 en vez de 0,50.
- Es la leccion del `INFORME_SIN_ETIQUETAS` del 20-ago aplicada de verdad: *lo que da el veredicto es
  cruzarlo con el nulo*. Ahi U-1 «pasaba» en 2 de 8 celdas y eran exactamente las 2 donde el nulo
  tambien pasaba. La forma correcta de usar esa leccion es comparar contra el nulo, no exigirle al
  nulo un valor.

**Cuarta vez en el programa que un criterio propio esta mal calibrado** (las otras: S-4 y el §2 del
monitor v1 el 20-ago, R-3 de la replica). Esta es la primera que se caza con una linea de base
corrida a proposito para eso.
