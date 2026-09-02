# ¿La ley vale fuera de nuestro modelo? · Mamba real, en CPU · congelado ANTES de correr

**2026-09-02.** Sale de la crítica que yo mismo le hice al proyecto hoy: todo está medido en un
vehículo de 3,5 MB y nada se probó en un modelo que no sea nuestro. Ésta es la prueba más barata que
existe para decidirlo, y no necesita entrenar ni una GPU.

## La distinción que se pone a prueba, y es la que hace al hallazgo

En un modelo recurrente todo el mundo asume, con razón, que **el estado** en la posición $t$ depende
de toda la secuencia anterior. Lo que este proyecto sostiene es otra cosa:

> **la QUERY con la que ese estado se lee no depende de toda la secuencia, depende de una ventana.**

En Mamba, `conv1d` (kernel 4) se aplica a $x$ **antes** de calcular $B_t$, $C_t$ y $\Delta_t$. $C_t$ es
el análogo de la query de lectura. Si la ley vale, $C_t$ es función **exacta** de $x[t-3..t]$ y de
nada más.

## La medición

Modelo `state-spaces/mamba-130m-hf`, en CPU, sin entrenar nada. Se toma una secuencia, se cambia
**un token** a distancia $d$ de la última posición, con $d = 1 \dots 8$, y en la **capa 0** se mide,
en la última posición:

1. el movimiento de la **salida de `conv1d`**, que es lo que alimenta a $B$, $C$ y $\Delta$;
2. el movimiento del **estado recurrente** o, en su defecto, de la **salida de la capa**.

## Criterios, escritos antes de mirar

- **R-1 · PRINCIPAL.** El movimiento de la salida de `conv1d` en la última posición es **exactamente
  0,0** para todo $d \ge 4$, y **> 0** para $d \le 3$. Cero exacto, no «chico»: es una identidad
  aritmética de una convolución causal de kernel 4 y si no da, el mecanismo no es el que decimos.
- **R-2 · CONTRASTE, y es lo que hace interesante a R-1.** El movimiento de la **salida de la capa**
  es **> 0 también para $d \ge 4$**. Sin esto, R-1 sería sólo «el modelo no ve tokens lejanos», que es
  falso y no es lo que afirmamos. Con esto, lo que queda demostrado es la **disociación**: el estado ve
  todo y la query ve cuatro tokens.
- **R-3 · GENERALIDAD.** R-1 y R-2 se sostienen en $\ge 4$ de 5 posiciones de prueba distintas y con
  dos textos distintos.

**Qué lo falsa:** que la salida de `conv1d` se mueva con $d \ge 4$ (entonces la ventana no es lo que
creemos), o que la salida de la capa **no** se mueva (entonces no hay disociación que mostrar y el
resultado es trivial).

## Lo que esta prueba NO puede decir

Es una medición de **arquitectura**, no de comportamiento: prueba que la query es local, no que eso
haga fallar al modelo en una tarea. El paso conductual —que una pregunta con la parte discriminante
lejos del final se responda peor— **no** se mide acá y necesita GPU. Y es Mamba, no Gated DeltaNet:
la conv es la misma idea con el mismo kernel por defecto, pero la comprobación es sobre el modelo que
se puede correr en esta máquina.
