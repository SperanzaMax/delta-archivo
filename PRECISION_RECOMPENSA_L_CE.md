# PRECISIÓN a `PREREG_RECOMPENSA_L.md` (SHA `96e750b6`) · **L-2 no es decidible con CE = 1,0**

**2026-08-30, 13:5x.** Se escribe con la campaña **en curso** (paso 2000 de 3000), **antes** de correr
el juez y **antes** de mirar ningún resultado final. Motivo: un chequeo de instrumento que hice
mientras entrenaba muestra que uno de los criterios del pre-registro **no tiene potencia para
decidirse**, y eso hay que decirlo antes y no después.

---

## 1. Qué disparó la sospecha

En los hitos intermedios, `t03_s3` (L=0) y `t53_s3` (L=0,5) daban **el mismo número a cuatro
decimales** en `abstencion` (0,4805), `nose` (0,4658) y `falsa_abst` (0,4836), mientras `vigente` y
`anterior` sí diferían. Dos condiciones que difieren en un parámetro de la pérdida no deberían
coincidir así.

**Primer control: ¿llega `L` al cómputo?** Sí, y se verifica sobre el mismo lote y los mismos pesos:

| L | pérdida | grad medio en q | \|grad\| en q |
|---:|---:|---:|---:|
| 0,00 | 2,72000337 | −0,00353940 | 0,00638886 |
| 0,25 | 2,66930079 | −0,00549252 | 0,00834198 |
| 0,50 | 2,61859846 | −0,00744564 | 0,01029511 |
| 1,00 | 2,51719332 | −0,01135189 | 0,01420136 |

Monótono y en la dirección que predice el §2 del pre-registro. **El instrumento está sano y la
hipótesis del flag perdido queda descartada.**

## 2. La causa real, medida

| término | valor | % de la pérdida |
|---|---:|---:|
| CE del valor (peso 1,0) | **2,1406** | — |
| −E[recompensa], L=0 | 0,1675 | **7,3 %** |
| −E[recompensa], L=0,5 | 0,0781 | **3,5 %** |

Y el gradiente que efectivamente llega a la columna del token `NOSE`:

| | \|grad\| en col. NOSE | \|grad\| medio, resto del vocabulario |
|---|---:|---:|
| L = 0 | 7,09e−06 | 2,49e−05 |
| L = 0,5 | 8,46e−06 | 2,49e−05 |

> **El logit que decide callarse recibe 3,5 veces MENOS gradiente que un token de valor cualquiera.**
> La `ENMIENDA_RECOMPENSA_F` puso CE = 1,0 para que un modelo que se calla no dejara de aprender a
> recuperar. Ese razonamiento era correcto **y con ese peso ahogó la señal que decide la abstención**.

**Y un efecto de segundo orden que no estaba previsto y va en contra del contraste:** subir $L$ no
sólo desplaza el óptimo, **también achica la recompensa en magnitud** (0,1675 → 0,0781), porque el
premio positivo por callarse cancela parte de los castigos. Las dos celdas de L-2 no difieren sólo en
dónde está el óptimo: difieren en cuánta señal hay, y la que debería ganar es la que menos tiene.

## 3. Qué se corrige, y qué NO

**L-2 se declara NO DECIDIBLE en esta campaña.** La diferencia de gradiente entre las dos celdas es de
orden **1,4e−06** sobre un paisaje de 2,5e−05. Un empate ahí **no es evidencia de que $L$ no importe**,
igual que el P1 del 10-ago fue «negativo sin potencia, no equivalencia». Se reportará el número y se
marcará sin adjudicar.

**Lo que SÍ se mantiene, y sin aflojar nada:**

- **L-1** (exactitud global > 0,4065) es una propiedad del modelo resultante y **se juzga igual**.
- **L-3** (abstención estrictamente entre 0,05 y 0,95) **se juzga igual**, y en los hitos intermedios
  ya se está cumpliendo desde el arranque locuaz.
- **L-6** (RECUP no cae más de 0,05) **se juzga igual**, y con una advertencia que el 29 costó una
  hipótesis equivocada: **`vigente` es COMPUESTA**, así que su caída de 0,4576 a ~0,19 **no** dice
  nada por sí sola sobre la recuperación — con `abstencion` en 0,48, la mitad de los aciertos se
  convierte en NOSE por construcción. RECUP se mide aparte, con el argmax y sin la decisión, que es
  lo que `juzgar_L.py` ya hace. **No se lee en ninguna de las dos direcciones sin ese número.**

## 4. La campaña que sigue, y por qué la siembra la habilita

La corrección natural es **bajar `--rec-ce`** para que la recompensa deje de ser el 7 % de la pérdida.
No se elige el valor mirando resultados: se elige igualando el gradiente en la columna de `NOSE` con
el gradiente medio del resto del vocabulario, que es una cantidad **medible antes de entrenar** y sale
del ratio 2,49e−05 / 7,09e−06 ≈ **3,5**.

> **Y hay un argumento para hacerlo ahora que antes no existía: la siembra lo habilita.** El motivo de
> CE = 1,0 era que un modelo que se calla dejaría de aprender a recuperar **desde cero**. Estas
> unidades **no parten de cero**: parten de `b3_s3`/`b3_s6` con RECUP ya en 0,365–0,384. La CE ahí no
> tiene que enseñar la recuperación, sólo **sostenerla**, y eso cuesta mucho menos peso.

**Eso va en un pre-registro propio**, con el valor derivado del ratio medido y no del desenlace de
esta corrida. Elegirlo mirando cómo terminó `t0` sería ajustar sobre la marcha, que es exactamente lo
que el `ESTADO_20260829_NOCHE` dejó prohibido por escrito.

## 5. Lección

Van **seis** defectos de pre-registro este mes. Los cinco anteriores fueron umbrales inalcanzables o
un parámetro mal derivado. **Éste es de una clase nueva: el criterio está bien definido y bien
derivado, y el que no alcanza es el TAMAÑO DE LA SEÑAL.**

> **Regla que deja: antes de correr un contraste entre dos valores de un peso, medir cuánto gradiente
> mueve ese peso comparado con el resto de la pérdida. Un contraste sobre el 3 % de la pérdida no es
> un contraste.**

Sale gratis y son dos líneas de `jax.grad`. Se agrega a la compuerta de la campaña siguiente.
