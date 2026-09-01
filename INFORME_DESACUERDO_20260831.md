# El desacuerdo entre búsquedas SEPARA en la dirección correcta, y no supera a la confianza

**2026-08-31, noche.** Idea de Maxi: *«si hace la búsqueda dos veces, si la respuesta es diferente
una de otra, ¿qué sería eso?»*. `desacuerdo_busqueda.py`, criterios D-1 a D-4 escritos **antes** del
dato. Versión **post-hoc y por ruido**, que es la más débil de las tres posibles.

---

## 1. Los números

**`n3_s0`** (sano, RECUP 0,7785), `n=1536`, `K=8` perturbaciones por consulta:

| σ | inestabilidad **con** respuesta | **sin** respuesta | razón | AUC |
|---:|---:|---:|---:|---:|
| 0,05 | 0,0033 | 0,0188 | **5,7×** | 0,5062 |
| 0,10 | 0,0079 | 0,0278 | 3,5× | 0,5191 |
| 0,20 | 0,0210 | 0,0535 | 2,5× | 0,5471 |
| 0,40 | 0,0383 | 0,1009 | 2,6× | **0,6054** |

**El fenómeno existe:** cuando la respuesta **no** está, la búsqueda es entre **2,5 y 5,7 veces** más
inestable. La intuición era correcta.

## 2. Y sin embargo no se adjudica nada, por el control que estaba escrito antes

**D-4 se dispara.** La confianza de salida sobre las mismas muestras da **0,6054** — el mismo número
hasta el cuarto decimal— y la correlación entre las dos llega a **−0,668**. La inestabilidad **no
supera a la confianza**, así que en esta versión **no es una medida nueva**: es otra forma de mirar
lo mismo que ya sabíamos mirar, y que ya sabíamos que no alcanza.

**D-1 tampoco:** 0,6054 queda por debajo del techo del estado (**0,7003**). No rompe el techo.

**Veredicto: D-2, señal parcial, sin adjudicación por D-4.**

*(La coincidencia exacta de los dos AUC en 0,6054 con `n=1536` es casualidad numérica, no identidad:
la correlación es −0,668, no −1. Se aclara para que nadie lea de ahí que son la misma variable.)*

## 3. ★ El hallazgo lateral, y es el más interesante del experimento

**`t03_s3`** (degradado, RECUP 0,3865): inestabilidad media **0,0002 a 0,0043**, o sea **50 veces
menos** que el modelo sano, y AUC **0,4927**, azar. Su confianza da **0,4546**, por debajo del azar.

> **El modelo degradado es casi perfectamente ESTABLE.** Le podés meter ruido grande en la query y
> contesta lo mismo. **La estabilidad no es señal de saber: también es la firma de estar colapsado.**
> Un detector basado en desacuerdo funciona en un modelo que todavía discrimina y se apaga
> exactamente en el régimen donde más falta haría.

Es coherente con el atractor absorbente del 29-ago y con la biestabilidad medida hoy: las unidades
degradadas no dudan, están **rígidas**.

## 4. Lo que este negativo NO cierra, y está declarado desde antes de correr

**D-3 lo dejó escrito: esto no dice nada sobre la versión ENTRENADA.** El precedente medido del
propio proyecto es explícito: el blanco `error` da **0,65 post-hoc** y **1,0000 entrenado**.

Y hay una razón específica para pensar que esta versión subestima: **el ruido gaussiano isotrópico es
la peor forma posible de generar «dos búsquedas distintas»**. Perturba en direcciones arbitrarias del
espacio de queries, la enorme mayoría de las cuales no corresponde a ninguna búsqueda que el modelo
haría. Las dos versiones fuertes de la idea siguen abiertas:

1. **Dos proyecciones de query aprendidas** (`qr1`, `qr2`) con el desacuerdo entrando en la pérdida.
   Ahí las dos búsquedas son ambas plausibles, y el desacuerdo es informativo en vez de ruido.
2. **Buscar por entidad y buscar por relación**, y comparar. Si apuntan a entradas distintas, eso es
   exactamente la **colisión de clave** que ya está identificada como el error dominante.

## 5. Lo que NO dice

- **No mide la idea de Maxi**, mide su versión más barata. La versión entrenada no está probada.
- Un modelo sano y uno degradado, `n=1536`, `K=8`, cuatro sigmas.
- **No compara contra el 0,70 en igualdad de condiciones**: aquel se midió con `n=6144`.
