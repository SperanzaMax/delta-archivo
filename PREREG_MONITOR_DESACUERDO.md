# PREREG · El monitor de desacuerdo interno

Congelado el 2026-08-20, después de cerrar `INFORME_SIN_ETIQUETAS_20260820.md` y antes de escribir
la sonda. Sale del §3 de `PLAN_20260820.md` y del plan del modelo que sabe que no sabe.

---

## 1 · Por qué esto y por qué ahora

**Hoy el «no sé» del micro-LM es una salida entrenada a imitar.** La cabeza de abstención aprende a
predecir la etiqueta «esta pregunta no tenía respuesta» a partir de ejemplos, igual que aprende
cualquier otra salida. El plan pedía otra cosa: que el «no sé» fuera **el desacuerdo entre dos
pasadas**, o sea una cantidad interna que el modelo produce por su cuenta.

De las tres piezas del plan, dos están: la **asimetría de tiempo** es E3 de Ligamento y el
**disparador** es la sorpresa de la regla delta de CENTINELA-01. **El monitor nunca se construyó.**

**Y hay una razón nueva, de hoy, para que sea esto y no otra cosa.** El corte sin etiquetas falló, y
el post-hoc dijo exactamente por qué: la información está en el logit —las poblaciones se separan
1,2 σ— pero **no en forma de valle**, así que ningún estimador que mire la *forma de la densidad* la
va a encontrar. Lo que le falta al logit es una **escala absoluta**: `a = 0,3` no significa nada por
sí solo, y por eso el corte necesitaba etiquetas.

**El desacuerdo no tiene ese problema: su cero es interpretable.** «Las K pasadas dieron todas la
misma respuesta» es una afirmación con sentido sin ninguna etiqueta y sin ninguna calibración. Si el
desacuerdo discrimina, el corte sale de la estructura del estadístico y no de un ajuste — que es
justamente lo que hoy no se pudo conseguir.

## 2 · Qué es «dos pasadas» en este modelo

El modelo es determinista: dos pasadas idénticas dan lo mismo y el desacuerdo sería cero por
construcción. La perturbación tiene que ser una **a la que la respuesta correcta sea invariante**, de
modo que el desacuerdo mida inestabilidad del modelo y no un cambio real del problema.

La perturbación elegida, declarada acá y no otra: **permutar el orden de las entradas del archivo**,
llevando `turnos` y `mask` pegados a su entrada. Es invariante por dos razones ya medidas, no
supuestas:

- **E-I3d** mostró que el lector usa el **sello de orden** y no la posición en el tensor: con turnos
  móviles la vigente se sostiene en 0,9644. Si dependiera de la posición, aquella prueba habría caído.
- El archivo se lee por atención sobre las claves, que es **permutación-equivariante** por
  construcción; lo único que rompe la simetría es el sello, y el sello viaja con su entrada.

O sea: **una respuesta anclada en una entrada real del archivo no puede cambiar al permutar.** Si
cambia, el modelo no estaba leyendo evidencia.

## 3 · El estadístico

Para cada consulta se corren **K = 16 pasadas** con permutaciones independientes del archivo, y se
mide sobre las respuestas (argmax de vocabulario, con `NOSE` excluido, igual que en toda la campaña):

**`consistencia` = fracción de las K pasadas que coinciden con la respuesta modal.**

Va de 1/V a 1. **`consistencia = 1` significa acuerdo total**, y ése es el punto que no necesita
calibración. El monitor propuesto es: **abstenerse si `consistencia < 1`** — el corte estructural, sin
un solo parámetro ajustado. Se reporta además la curva completa por si el corte estructural resulta
demasiado exigente.

## 4 · Predicciones

Sobre las **8 unidades de la familia `c` a 14000 pasos**, con las mismas muestras y semillas de
`PREREG_CORTE_SIN_ETIQUETAS.md` (prueba: rng 77000+semilla, 2048 muestras), **reportando por unidad**:

- **M-1 (¿discrimina?).** AUC del desacuerdo separando preguntas sin respuesta de preguntas con
  respuesta **≥ 0,70** en al menos 6 de 8 unidades. Referencia ya medida del logit entrenado: 0,825 en
  la unidad difícil, ≥0,99 en las fáciles. **El desacuerdo no necesita ganarle al logit para valer**:
  el logit se entrenó con etiquetas y esto no.
- **M-2 (la principal, y es la que el logit no pudo).** El corte **estructural** `consistencia < 1`
  pasa la compuerta (`falsa_abst ≤ 0,10` y `nose ≥ 0,50`) en **≥ 6 de 8** unidades. Es el mismo
  criterio con que U-1 sacó 2/8 y σ>0,5 sacó 6/8. **Para que esto valga como avance tiene que llegar a
  6/8 sin ajustar nada.**
- **M-3 (control de que mide lo que decimos, y puede fallar).** Con una perturbación **que sí cambia
  el problema** —quitar del archivo la entrada que la consulta necesita— el desacuerdo tiene que
  **subir** en las preguntas que tenían respuesta: mediana de `consistencia` al menos 0,10 por debajo
  de la del barajado. Si no sube, el estadístico no está midiendo anclaje en evidencia y M-1/M-2 no se
  pueden interpretar.
- **M-4 (nulo).** Con **K pasadas idénticas** (permutación identidad) la consistencia debe dar
  **exactamente 1,000** en todas las muestras y la AUC **0,500**. Es el control que verifica que todo
  el efecto viene de la perturbación y no de ruido numérico del pipeline.

## 5 · Desenlaces, comprometidos por adelantado

- **M-1, M-2 y M-3 cumplen** → **el «no sé» tiene por primera vez un correlato mecánico** en vez de
  ser una etiqueta imitada, y además resuelve el corte sin etiquetas que quedó abierto hoy. Es el
  resultado que el plan del 19 pedía.
- **M-1 cumple y M-2 no** → la señal existe pero el corte estructural no alcanza: el desacuerdo
  sirve como **evidencia** y no como decisión, y hay que decirlo con esas palabras.
- **M-1 falla** → el desacuerdo bajo permutación no distingue, y el monitor tal como está planteado
  no sirve. **No se prueba una segunda perturbación en este experimento.**
- **M-3 falla** → lo medido no es anclaje en evidencia; el resultado se archiva sin interpretar,
  cualquiera sea M-1 y M-2.

## 6 · Lo que no puede decir

- Es un monitor **de inferencia** sobre modelos ya entrenados: no dice nada sobre entrenar contra el
  desacuerdo, que sería el paso siguiente y necesita GPU.
- Idioma cerrado de 242 tokens, `p_nose = 0,4`. Mismo alcance que todo el resto de la campaña.
- **Las 16 pasadas cuestan 16× la inferencia.** El experimento mide si la señal existe, no si el
  costo se justifica; con `K = 2` la consistencia sólo puede valer 0,5 o 1, y esa versión barata se
  reporta aparte como referencia, sin criterio propio.

## 7 · Costo

CPU, sobre los checkpoints que ya están en `micro_lm/ckpts/`. **Cero GPU y ningún entrenamiento.**
Ninguna unidad de las que este experimento lee puede estar entrenándose al mismo tiempo — lección D-1
de hoy —, así que **no se corre sobre `c4_s0` ni `c4_s1` hasta que la réplica cierre.**
