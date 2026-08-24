# DISEÑO · LA ABSTENCIÓN COMO ATRIBUCIÓN — el slot nulo en la memoria

2026-08-24. Decisión de Maxi: **la supervisión vale**, y se arranca por esta vía. Sale del inventario
de los siete intentos de detección y del §5 de `PLAN_FOCO_20260824.md`. Todavía **no** es
pre-registro: esto fija qué se construye y por qué, y qué hay que chequear antes de escribir
predicciones.

---

## 1. El diagnóstico del que sale, en un número

Siete intentos de detectar sin etiquetas, siete resultados en la misma banda: **AUC 0,50-0,67**.
Contra la cabeza supervisada, que da **0,77-0,99**. No es que no encontramos el detector: **la señal
no existe salvo que se la enseñe.**

Y el dato más limpio es el más viejo, del 16-ago, que nadie volvió a mirar:

> **El score del archivo —el matcheo de la consulta contra las claves— da AUC `0,4984` y `0,5022`
> para separar «la respuesta está» de «la respuesta no está».** Azar exacto, dos decimales.

Tiene una causa mecánica que el dictamen del 16-ago ya había escrito y que nunca se atacó:

> *«en `modelo.responder()` la lectura es un softmax sobre las entradas del archivo, o sea suma 1
> siempre. Hoy no existe la posibilidad de que la lectura devuelva "nada": el modelo está obligado a
> leer algo aunque nada matchee.»*

**El archivo se entrenó para recuperar, nunca para decidir pertenencia.** Buscar ahí una señal de
ausencia es buscarla donde nada la puso. Esto no se arregla con un octavo detector: se arregla
haciendo que la ausencia **tenga dónde vivir**.

## 2. Qué se construye

**Un slot nulo aprendido dentro del archivo**, y la abstención leída de la masa de atención que ese
slot recibe.

Concretamente, en `modelo.responder_con_abst`:

- Dos vectores nuevos, `arch["k_nulo"]` y `arch["v_nulo"]`, de dimensión `D` cada uno. Se concatenan
  como una columna más de `ak` y `av`.
- **La columna nula nunca se penaliza**: `penal` sigue tapando las entradas vacías del archivo, pero
  el nulo compite siempre.
- La decisión de abstenerse sale de **la masa de atención del slot nulo**, no de un escalar aparte.

Son **2 × 128 = 256 parámetros** sobre 863.859 (**0,030 %**), y del orden de cinco líneas de código
—el dictamen del 16-ago ya lo había dimensionado así—.

**Por qué esto es atribución y no otra cosa:** el modelo pasa a decir *de dónde* sacó la respuesta, y
«de ningún lado» se vuelve **expresable en el mismo espacio donde ya opera**. No hace falta un
detector externo que interprete su estado: la abstención es una respuesta más sobre la memoria.

## 3. La advertencia que cambia el criterio de éxito

Esto es lo que hace que el diseño no sea ingenuo, y sale de `INFORME_RANK_HECHO_20260816.md`:

> *«el modelo acierta sin que la entrada correcta gane. Incluso entre los aciertos, la entrada del
> hecho preguntado encabeza la lectura sólo la mitad de las veces. El archivo funciona como un banco
> de evidencia parcialmente ordenado, no como un índice que devuelve un registro.»*

Consecuencias directas, y las dos ya estaban escritas en aquel informe:

1. **«Que gane el slot nulo» NO puede ser el criterio de abstención.** Si el modelo acierta con la
   entrada correcta en el puesto 2, exigir que el nulo gane haría abstenerse en casos que hoy se
   contestan bien. El nulo tiene que competir **por masa relativa**, no por victoria.
2. **No se supervisa a qué entrada real apunta.** Forzar un one-hot sobre la entrada correcta pelearía
   contra el mecanismo que hace que el modelo acierte —integrar varias entradas— y estaríamos
   rompiendo lo que funciona para medir lo que no. **Se supervisa sólo la masa del nulo.**

O sea: la pérdida auxiliar le pide al modelo *«cuánta de tu atención va a "nada"»*, y le deja libre
**cómo reparte el resto**.

## 4. Lo que NO es nuevo, y hay que citarlo

El dictamen del 16-ago ya dejó la lista, y conviene tenerla antes de escribir una línea:

- **Pointer Sentinel** (Merity et al., 2016) — el sentinela que decide entre copiar del contexto o
  generar del vocabulario. Es el mismo mecanismo estructural.
- **SQuAD 2.0** — el *no-answer score*, que compite contra los spans. Es la misma idea en QA
  extractivo.
- **OOD por energía** (Liu et al., 2020) — el umbral sobre el score, acá en la interfaz de memoria.
- Predicción selectiva (SelectiveNet, Deep Gamblers), `[IDK]`, R-Tuning.

**El slot nulo no es una invención de este proyecto.** Decirlo al revés sería redescubrir con GPU lo
que está publicado desde 2016.

## 5. Entonces qué sería nuevo

Tres cosas, y las tres son del contraste, no del mecanismo:

1. **El tercer brazo del contraste que el dossier declaró como hueco.** Nadie comparó, dentro del
   mismo modelo con memoria, a igualdad de parámetros y con semillas: abstención como **entrada del
   vocabulario** (`token`, que es `[IDK]`), como **cabeza binaria separada** (`cabeza`, que es
   SelectiveNet) y como **entrada de la memoria** (`slot`, que es pointer sentinel). Los dos primeros
   **ya están corridos y medidos**. Este experimento completa el trípode.
2. **La abstención en una memoria con versiones**, donde la pregunta puede ser por el valor superado.
   Pointer sentinel y SQuAD 2.0 no tienen versionado ni memoria persistente.
3. **La predicción mecánica que lo hace falsable**: si el problema es que el softmax obliga a leer
   algo, entonces darle a dónde no leer tiene que mover **el score del archivo desde el azar**. El
   AUC 0,4984 es la línea base más limpia del proyecto, y esta es la primera intervención que
   apunta directo a ella.

## 6. El diseño experimental, y por qué 2×2 no

El error de `post` fue mover dos cosas a la vez. Acá la tentación sería mover también la supervisión
de atribución positiva. **No se hace**: el primer experimento cambia **una** cosa —dónde vive la
abstención— y reusa todo lo demás.

| condición | dónde vive la abstención | estado |
|---|---|---|
| `token` | entrada del vocabulario (`[IDK]`) | **ya corrida** |
| `escala` | ídem, con el vector renormalizado | **ya corrida** |
| `cabeza` | escalar binario separado | **ya corrida**, es la que gana hoy |
| **`slot`** | **entrada de la memoria** | **lo que se construye** |

El control es `cabeza`, con las mismas semillas, el mismo presupuesto y el mismo generador. La
campaña cuesta 3 unidades.

**Asimetría declarada por adelantado:** `slot` estrena 256 parámetros y `cabeza` estrena 129. No son
iguales. Es una diferencia de 127 parámetros sobre 863.859 (0,015 %), y se declara acá en vez de
descubrirse después — pero si `slot` gana, la objeción «ganó porque tiene más parámetros» hay que
poder contestarla, y para eso está el brazo de control con `d` reducido que se decidirá **sólo si
hace falta**.

## 7. Chequeo de instrumento, ANTES del pre-registro

Por la regla que dejó el monitor v1: lo primero que se verifica de una reparación es que la
reparación **haga algo**, y que lo que hace sea lo que dice. Con pesos al azar, CPU, sin entrenar:

- **A-1** · con el slot agregado, la lectura sigue sumando 1 y la masa del nulo está en el rango
  esperado para pesos al azar (≈ `1/(N+1)`). Si diera 0 o 1, el slot no compite o se come todo.
- **A-2** · **el slot no cambia nada cuando no se lo usa.** Con `k_nulo` puesto en −∞ efectivo, el
  modelo tiene que dar **exactamente** lo mismo que antes de agregarlo. Es la guarda que protege el
  control reusado, igual que K-5 en `lat2`.
- **A-3** · la masa del nulo **responde al contenido**: al tapar del archivo la entrada del hecho
  preguntado, la masa del nulo tiene que **subir**. Con pesos al azar no se espera efecto, así que
  esto se mide sobre un checkpoint **entrenado** (`p3_s0`), que es gratis y está en disco.
- **A-4** · el gradiente llega a `k_nulo` y `v_nulo` (el chequeo que en `lat2` destapó el weight
  decay).

**A-3 es la que importa**: si en un modelo ya entrenado —sin slot en su entrenamiento— la masa del
nulo no reacciona a que la evidencia no esté, entonces el mecanismo depende enteramente de la
supervisión, y eso hay que saberlo **antes** de escribir las predicciones.

## 8. Lo que este experimento NO va a contestar

Se escribe ahora para no confundirlo después:

- **No contesta si el modelo «sabe» sin supervisión.** Maxi decidió que la supervisión vale, así que
  la pregunta es si la abstención se aprende **mejor** cuando vive en la memoria que cuando vive en
  el vocabulario o en una cabeza. Es una pregunta de **dónde**, no de **si**.
- **No cierra el corte sin etiquetas.** Esa línea sigue como estaba.
- **No dice nada sobre escala.** 863.859 parámetros, idioma sintético.
