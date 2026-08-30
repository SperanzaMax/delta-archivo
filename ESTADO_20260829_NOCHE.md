# ESTADO · 29 de agosto, cierre

Para retomar mañana sin releer nada más que este archivo.

---

## 1. EL HALLAZGO DEL DÍA · la abstención perfecta es el conocimiento nulo

`INFORME_BIFURCACION_20260829.md`. Post-hoc sobre los checkpoints del 28.

**Cuatro unidades tienen `nose` 1,0000 e `invento` 0,0000 —perfectas en las dos métricas que el campo
reporta— y exactitud global 0,4065, que es el piso trivial a cuatro decimales.** No son detectores
rotos: son **detectores perfectos de un generador que no sabe nada**, y su calibración es honesta.

- **La cabeza colapsó al prior**, verificado contra la fórmula sin parámetros ajustados: en `b3_s8` la
  predicción da **1,513** y lo medido es **1,507**. AUC 0,522–0,578 (azar), rango del logit 0,85–3,24
  contra 29,17 de una sana.
- **El mecanismo es una carrera.** El blanco `error` es autorreferencial, así que al empezar la
  etiqueta es la constante 1 y la cabeza aprende «me equivoco siempre», **que es verdad**. Hay dos
  puntos fijos y los dos son autoconsistentes.
- **La bifurcación se decide en el paso 2500** y es cero contra no-cero respuestas de 512. **40 de 40**
  en todo el repo, y `b3_s4` se salvó con **una sola muestra**.
- **El atractor es ABSORBENTE**, medido dentro de una unidad: −0,0021 en 4000 pasos (0,4 σ).
  `PREREG_ATRACTOR_MUDO` (2be4a610) + `PRECISION` (5e413fef). La Fase 2 se canceló, 70000 pasos.

## 2. ★ EL CONFOUND, y hay que leerlo antes que nada de lo de arriba

**Sólo existían tres bases de siembra** (`n3_s0/s1/s2`, 12000 pasos previos, RECUP 0,77–0,80), y el
rotador siembra solo cuando la base existe. Entonces:

| grupo | partida | resultado |
|---|---|---|
| s0, s1, s2 | RECUP ≈ 0,78 ya aprendido | **3 de 3 útiles** |
| s3 … s8 | de cero | 2 útiles, **4 mudas** |

**«Nueve unidades, sólo cambia la semilla» es FALSO**, y estaba en el abstract del paper y en
`PREREG_TASA_REGIMEN`. El chequeo del 28 comparó **configs**, y la config no registra la siembra.

**Sobrevive** todo lo medido por unidad, y el mecanismo queda **mejor** explicado (el que llega
sabiendo ya ganó la carrera). **No sobrevive** la tasa «4 de 9», que es 0/3 con base y 4/6 sin base,
ni la comparación 31-contra-9. La exclusividad del blanco autorreferencial **sí** sobrevive, porque
descansa en el control pareado `b3` contra `p3`, que comparte semilla **y** base.

## 3. Las tres campañas de la función de pérdida, todas ideas de Maxi

**A · `balance` y `ranking`** (`0f57609d` + `fe058151`, informe `INFORME_PERDIDA_CABEZA_20260829.md`).
Le sacan a la cabeza el pago por la constante.
**P-1 cumple 4/4 con las dos. P-4 se dispara: ninguna de las 11 supera el piso.** La mejor da 0,4024,
o sea **peores que el silencio**, con invención donde el control tenía 0,0000.
→ **Mudez e invención son los dos topes de una misma perilla**, y estas dos la mueven de un extremo al
otro porque tocan **sólo al vigilante**.

**B · la recompensa acoplada** (`PREREG_RECOMPENSA.md`, `f1f7bb66`). Premio al acierto y al `nose`
correcto, castigo al error y a la falsa abstención, en **una sola pérdida sobre el resultado final**.
**La condición principal NO usa vigilante** —la probabilidad de abstenerse es la masa que el softmax
ya le da a `NOSE`— porque es la que escala a un modelo grande sin tocar la arquitectura.

**C · Etapa 1 con F=1,5 → FRACASO, y la culpa es de una derivación mía.**
Las 8 unidades dieron **`abstencion` 0,0000**. Contestaron todo.
`ENMIENDA_RECOMPENSA_F.md`: la condición se dedujo sobre un **q global** y el modelo elige **q por
muestra**. Hace falta **F < M**; con F=1,5 y M=0,5 el umbral daba **−0,667**, o sea *nunca conviene
callarse*. **Y la compuerta lo tenía impreso** como riesgo cuando era una contradicción de diseño.

> **REGLA NUEVA: si un chequeo dice que el óptimo de la pérdida es un extremo, eso no es un riesgo a
> vigilar. Es un defecto, y se cierra antes de correr.**

## 4. Lo primero que hay que hacer mañana

1. **Mirar la prueba `f23_s3`/`f23_s6`** (F=0,2, CE=1,0), lanzada a las 22:54. Es **exploratoria** y
   no juzga nada; sólo hay que ver si `abstencion` cae en algún lugar **intermedio** en vez de en 0 o
   en 1.

   > **⚠ Predicción anotada ANTES de ver el resultado, para no leerlo mal mañana.** Con F=0,2 el
   > umbral **global** queda en **0,657**, o sea que al modelo le conviene callarse hasta que su
   > acierto supere eso. Es alto: las mudas del control llegan a 0,30–0,40 recién a los 22000 pasos.
   > **Puede que 3000 pasos no alcancen para verlo hablar, y que el silencio no signifique fracaso
   > sino que todavía no llegó al umbral.** Los primeros cuatro hitos (250 a 1000) dieron 1,0000, que
   > es exactamente lo esperado y **no** es una señal en ningún sentido.

   > **RESULTADO, medido a las 23:49 y anotado después de la predicción de arriba.** `f23_s3` llegó
   > a 3000 pasos con **`abstencion` 1,0000 en los DOCE hitos**, incluido el del paso 2500. **La
   > predicción acertó: quedó muda todo el tramo.** `f23_s6` no llegó a correr (sin VM).

   Terminó muda, así que hay dos lecturas y **se distinguen con un experimento, no discutiendo**.
   (a) falta presupuesto, y se ve corriendo **la misma unidad** más pasos.
   (b) F=0,2 dejó el umbral global fuera de alcance, y se ve con un F intermedio.
   Valores ya calculados, umbral por muestra y global: **F=0,3 → 0,133 y 0,590** · **F=0,35 → 0,100 y
   0,557** · **F=0,4 → 0,067 y 0,523**.
   **Y elegir F mirando el resultado es ajustar sobre la marcha**, así que eso va con pre-registro.
2. **Correo institucional.** Nico Censabella (`nicocensabella@frba.utn.edu.ar`) dijo el jueves 27
   «espero mañana tener alguna novedad» y no escribió. **El lunes 31 corresponde el seguimiento**, y
   es el trámite de mayor palanca del proyecto.
3. **Decidir si el paper del atractor se envía.** Está en `preprint/atractor/`, 11 páginas,
   compilando, con el confound declarado como sección propia. Falta elegir destino, y eso depende
   de (2).

## 5. Estado al cerrar

- **Repo pusheado**, `9c65988`. Tres commits hoy y **el último ya se atribuye a la cuenta de GitHub**:
  el email global pasó de `maxi@example.com` al noreply de GitHub. Los 146 viejos **no se tocan**, el
  historial es la evidencia de fecha que cita el paper de gemación.
- **Titular de LinkedIn actualizado** a la opción C. Falta la insignia de verificación, que pide
  documento y la hace Maxi.
- **Mail a gist.science enviado** pidiendo corregir el 4/5 que el propio paper retracta.
- **Post de LinkedIn del atractor listo y SIN publicar**, corregido tras el confound. Necesita el DOI.

### Cosas que quedaron sin correr, y por qué
- 4 de las 12 unidades de la Etapa 1 (`tk3_s3`, `tk3_s6`, `hd3_s4`, `hd3_s5`) no consiguieron VM.
  Colab dio **503 en las trece cuentas durante ocho vueltas**. No es un fallo de diseño, es saturación.
- La prueba de F=0,2 quedó lanzada al cierre; si no terminó, su checkpoint conserva el paso.
