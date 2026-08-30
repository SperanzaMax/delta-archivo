# INFORME · el atractor mudo se elimina, y el modelo se va al otro extremo

Evalúa `PREREG_PERDIDA_CABEZA.md` (SHA `0f57609d`, congelado 16:02 antes de escribir el instrumento)
con `ENMIENDA_PERDIDA_CABEZA.md` (SHA `fe058151`, congelada 16:37 antes del primer tramo).

11 unidades a 3000 pasos, `SEMBRAR=0`, control `b3_s3`…`b3_s8`. Instrumentos:
`chequeo_perdida_cabeza.py` (compuerta, abrió con 12 pruebas) y `juzgar_perdida.py`, n=8000,
semilla de datos 54321.

---

## 1. Los criterios, contra lo medido

| | criterio | `balance` | `ranking` | |
|---|---|---|---|---|
| **P-0** bloqueante · no-daño | `s4` y `s5` no caen en abstención total | 0 de 2 rotas | 0 de 2 rotas | **CUMPLE** |
| **P-1** PRINCIPAL | ≥3 de 4 mudas emiten respuestas | **4 de 4** | **4 de 4** | **CUMPLE** |
| **P-2** contraste | `ranking` ≥ `balance` | — | — | empatan en P-1 |
| **P-3** mecanicista | AUC > 0,60 | 3 de 6 | 3 de 5 | **NO CUMPLE** |
| **P-4** RIESGO | no salir del silencio inventando | **6 de 6 inventan** | 3 de 5 inventan | **SE DISPARA** |

**P-1 cumple de la forma más contundente posible y P-4 se dispara igual de fuerte.** La celda del §5
del pre-registro para esa combinación está escrita desde antes: *«se cambió mudez por invención; no se
adopta, el resultado es sobre el trade-off y se informa así»*.

## 2. El resultado, en una tabla

| unidad | exactitud | RECUP | AUC | `abst` | `invento` |
|---|---:|---:|---:|---:|---:|
| **control `b3_*` (mudo)** | **0,4065** | 0,3040–0,3964¹ | 0,52–0,58 | 1,0000 | **0,0000** |
| bl3_s8 | 0,3536 | 0,0546 | 0,5524 | 0,8310 | 0,0617 |
| bl3_s4 | 0,3114 | 0,2226 | 0,5652 | 0,5777 | 0,1613 |
| bl3_s5 | 0,3045 | 0,1524 | 0,6565 | 0,5683 | 0,1600 |
| bl3_s7 | 0,2924 | 0,0601 | 0,6331 | 0,6760 | 0,1324 |
| bl3_s3 | 0,2850 | 0,0656 | 0,6238 | 0,6617 | 0,1404 |
| bl3_s6 | 0,2361 | 0,0723 | 0,5935 | 0,5165 | 0,1966 |
| rk3_s7 | 0,4024 | 0,0983 | 0,6110 | 0,9699 | 0,0095 |
| rk3_s8 | 0,3859 | 0,0491 | 0,6656 | 0,9364 | 0,0267 |
| rk3_s6 | 0,3538 | 0,0481 | 0,6620 | 0,8407 | 0,0629 |
| rk3_s4 | 0,3536 | 0,0447 | 0,5963 | 0,8530 | 0,0592 |
| rk3_s3 | 0,2625 | 0,0489 | 0,5599 | 0,6199 | 0,1570 |

¹ medido a 22000–26000 pasos, **no comparable** con las de 3000 (ver §4).

> **Ninguna de las once supera el piso trivial.** Un modelo que no dice nada saca 0,4065; la mejor de
> éstas saca 0,4024. **Las once son peores que el silencio**, y el silencio es lo que veníamos
> tratando como la patología.

## 3. Lo que esto establece, y es un resultado y no un fracaso

**El atractor mudo se elimina por diseño de la pérdida.** Cuatro de cuatro, con dos implementaciones
independientes, y desde el paso 250. El diagnóstico de que la mudez se sostenía en que **la cabeza
podía cobrar el prior** queda confirmado por partida doble: la compuerta lo verificó en la fórmula
—la BCE tiene su mínimo en 1,3900 contra el logit del prior teórico 1,3863— y el entrenamiento lo
confirmó en el banco.

**Y el atractor mudo no era el problema, era la mitad del problema.** Sacarle el pago al silencio no
enseña a acertar, porque estas dos pérdidas tocan **sólo al vigilante**. La cross-entropy del valor y
la de la cabeza son dos términos que no se hablan, así que **nada le dice al modelo que equivocarse
sea peor que callarse**. Se le quitó el único incentivo que tenía y se fue al único lugar que quedaba.

> **La mudez y la invención son los dos extremos de una misma perilla, y ninguna de las dos
> pérdidas la calibra: la mueven de un tope al otro.**

**`ranking` invadió menos que `balance`** (3 de 5 contra 6 de 6) y su mejor unidad es la que llega más
cerca del piso. Pero la lectura es incómoda: `rk3_s7` alcanza 0,4024 **con abstención 0,9699**, o sea
que se acerca al piso **por seguir siendo casi muda**, no por acertar. La correlación es directa: en
las once unidades, cuanto más se abstienen, más cerca del piso quedan. **Eso no es una condición
mejor, es una condición que retrocedió menos.**

## 4. Un defecto del pre-registro, mío, y es el cuarto de la serie

**P-4 no era decidible a 3000 pasos, y se escribió igual.**

Comparar la exactitud de una unidad de 3000 pasos contra el piso es justo —el piso es 0,4065 a
cualquier presupuesto— pero **no dice si la condición sirve**, porque a 3000 pasos ningún modelo de
este banco supera el piso todavía. `b3_s0`, que termina en RECUP 0,9996, tiene `vigente` 0,0222 a esa
altura. Lo mismo con RECUP: las de 3000 pasos dan 0,04–0,22 contra 0,30–0,40 del control, y esa
comparación **no es válida** porque el control tiene 22000 pasos más.

Lo que sí queda establecido sin depender del presupuesto es lo del §3, porque **`invento` sí es
comparable**: el control tiene 0,0000 por construcción, y la invención que aparece acá es nueva.

> Es la **cuarta vez este mes** que un umbral se escribe sin verificar que sea alcanzable con el
> presupuesto de la campaña —E-2 de A5 el 27, K-1 de CALIBRA el 28, F-1 de ATRACTOR esta mañana, y
> ésta—. Las tres anteriores quedaron anotadas «como lección para el próximo pre-registro» y el
> próximo volvió a caer. **`PREREG_RECOMPENSA.md` parte el presupuesto en dos etapas exactamente por
> esto**, y su criterio principal vive en la etapa que puede decidirlo.

## 5. Lo que NO se concluye

- **No se descarta `balance` ni `ranking`.** Se descarta usarlas **solas**. Hacen lo que prometían y
  la mitad que falta no es asunto suyo, es de la pérdida del valor, que no tocan.
- **No se compara con el control en exactitud ni en RECUP a igual presupuesto**, porque no hay control
  a 3000 pasos con checkpoint. Sólo `invento` y `abstencion` son comparables.
- **AUC 0,55–0,67 no es un buen detector**, y P-3 no cumple. Pero es la primera vez que estas unidades
  salen del azar (el control degenerado da 0,52–0,58), y a 3000 pasos no se le puede pedir más.
- **Nada de esto habla de escala.**

## 6. Qué sigue, y ya está corriendo

`PREREG_RECOMPENSA.md` (SHA `f1f7bb66`), de una idea de Maxi, ataca justamente lo que este informe
identifica: **una sola pérdida sobre el resultado final**, que le diga al modelo que acertar paga,
que decir «no sé» cuando de verdad no está paga menos, y que **equivocarse y callarse de más cuestan**.
Los pesos salen de una condición derivada, no elegidos a ojo, y la condición principal **no usa
vigilante** —la probabilidad de abstenerse es la masa que el softmax ya le da a `NOSE`— porque ésa es
la que puede escalar a un modelo grande sin tocarle la arquitectura.

Su compuerta W-0 abrió con 21 chequeos, y uno de ellos deja escrito el riesgo espejo de este informe:
con los pesos elegidos, **si el modelo no logra distinguir, su mejor política es contestar todo**. El
fracaso posible de la campaña que viene es la locuacidad, no la mudez.
