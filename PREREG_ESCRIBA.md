# PRE-REGISTRO · `escriba` — una cabeza que verifica ANTES de guardar

**Escrito el 2026-08-27.** No se corre hasta que A5 (`--blanco error`) cierre sus tres semillas. Se
congela hoy igualmente, porque el punto de congelarlo es que las predicciones existan antes que los
datos, no antes que la GPU.

---

## 1. De dónde sale, y en palabras de quién

Idea de Maxi, el 27-ago, leyendo el párrafo del trípode sobre el modo de fallo propio de un modelo
con archivo:

> «¿quiere decir que lo que se graba como información se carga en un lugar equivocado o mal
> etiquetado? ¿y si le ponemos una cabeza a ese paso para que verifique eso antes de guardarlo
> definitivo?»

El movimiento estructural es **el mismo que ya ganó** en `INFORME_CABEZA_20260819.md`: sacar una
decisión del softmax compartido y darle su propio lugar, porque dos decisiones de naturaleza distinta
compitiendo por un mismo softmax se estorban. Ahí fueron 129 parámetros, el 0,015 % del modelo, y
bajaron la falsa abstención por un factor de 2 a 2,8×. Lo que Maxi propone es aplicar ese mismo
movimiento **del lado de la escritura**, que es el único de los dos lados que nunca se tocó.

---

## 2. Lo que ya está medido, y que acota la idea antes de empezar

Tres resultados propios recortan el espacio. Van acá arriba justamente para que la condición no se
diseñe contra un problema que ya no existe.

**2.1 · La mitad de la hipótesis que ya está refutada.** «Se graba en el lugar equivocado» **no** es
lo que dicen los datos. `lat2` (`INFORME_LAT2_20260825.md`) tocó la **query**, no la escritura, y
llevó `err_identidad` de 0,0122 / 0,1289 / 0,0762 a **0,0000 en las tres semillas**. Si el hecho
entrara mal al archivo, arreglar cómo se pregunta no lo repararía, y menos hasta cero exacto. La
información entra bien; lo que fallaba era el **direccionamiento en la lectura**.

**2.2 · Y por eso, apuntada a `err_identidad`, la idea no tiene margen.** El error que la cabeza
cazaría ya vale 0,0000. Ninguna cabeza mejora un cero, y una comparación contra un piso no informa.
**Esta es la razón por la que la condición se redirige**, y no un detalle de implementación.

**2.3 · Dónde SÍ quedó margen.** `lat2` arregló identificar la entidad **y pagó con la relación**:
`nose_rel` cayó de 0,9235 a 0,5842 en s0, y su V-3 falló 1/3 por eso. Sumado a lo que el trípode ya
cerró —en una memoria co-entrenada leída por softmax **la ausencia no tiene representación**, con la
masa del slot nulo convergiendo al prior (0,4074 / 0,4046 / 0,4020 contra tasa base 0,4048)— el
agujero abierto no es «a quién pertenece el hecho» sino **«el hecho no está»**.

---

## 3. FASE 0 · medir antes de construir, sin GPU

El proyecto ya pagó por saltear esto una vez y ya lo hizo bien otra: `INFORME_SCORE_ARCHIVO_20260816.md`
midió la señal antes de diseñar la campaña y el resultado bifurcó todo. Se repite el patrón.

**La pregunta de la Fase 0:** en el momento de la escritura, ¿existe en las activaciones una señal
que separe una entrada que va a ser recuperable de una que no? Si no existe, la cabeza tendría que
**crear** esa representación y no sólo leerla, que es un proyecto distinto y mucho más caro.

**Cómo.** Sonda lineal sobre checkpoints **ya entrenados** (`p3_*` y `v3_*`, 26000 pasos, en disco),
CPU, cero GPU. Se extraen las activaciones de `M.escribir` por entrada y se entrena una sonda que
prediga la etiqueta de recuperabilidad, con la etiqueta saliendo del dato sintético. Se reporta AUC
sobre un split que la sonda no vio.

**El control que tiene que poder fallar**, porque en este programa un número limpio escondió un
artefacto siete veces y el 0,4984 del score del archivo obligó a correr `control_score.py`: se
reporta también la AUC de la **misma sonda sobre etiquetas permutadas**. Si la sonda con etiquetas
reales y la sonda con etiquetas permutadas dan lo mismo, lo medido es la capacidad de la sonda y no
una señal del modelo.

### Predicciones de la Fase 0

- **E-0 · BLOQUEANTE.** La sonda con etiquetas permutadas queda en **AUC ≤ 0,55**. Si esto falla, no
  se lee nada más de la Fase 0 y se arregla el instrumento.
- **E-1 · LA QUE DECIDE.** La sonda con etiquetas reales alcanza **AUC ≥ 0,65** en al menos 2 de 3
  semillas. El umbral no es arbitrario: las **siete** vías sin etiquetas del `PLAN_FOCO_20260824.md`
  aterrizaron todas en 0,50-0,67, y una octava dentro de esa banda sería la misma nada con otro
  nombre.

---

## 4. La condición `escriba`, si y sólo si la Fase 0 abre

Cabeza binaria por entrada escrita, entrenada con **etiqueta supervisada** del dato sintético.

**Que haya etiqueta es la única razón por la que esto no es la octava vía muerta.** Las siete que
fallaron buscaban la decisión **sin etiquetas**. La cabeza de abstención funciona porque tiene
blanco. `escriba` tiene blanco. Es más parecida al caso que ganó que a los que perdieron, y esa es
toda su apuesta.

**Variante A, la única que se corre primero.** La cabeza entra **sólo como pérdida auxiliar**. No
modula la escritura, no toca la lectura, no cambia una sola ruta del forward. Separa «¿la señal se
puede aprender?» de «¿sirve para decidir?», que son dos preguntas y en este proyecto mezclarlas ya
costó una campaña.

**Variante B, que NO se corre en esta vuelta.** La cabeza como compuerta que modula lo que se
escribe. Queda declarada acá para que, si más adelante se corre, se vea que no se inventó después de
mirar los resultados de A.

---

## 5. Predicciones de la condición

Instrumentos declarados: `ser.py` (n=2048, semilla 54321) y `diag_relacion.py` (2048 muestras), los
dos leyendo `donde` y la regla de decisión **del checkpoint**. Control pareado `v3_s0/s1/s2` (`lat2`,
26000), idéntico salvo la cabeza.

- **W-0 · BLOQUEANTE.** `escriba` aprende la tarea: acierto ≥ 0,70 en al menos 2 de 3 semillas. Una
  pérdida auxiliar no debería romper nada, y si lo rompe, lo que se cae es más grande que esta idea.
- **W-1 · LA PRINCIPAL.** `nose_rel` sube **≥ 0,05** respecto de su gemela `lat2`, en al menos 2 de 3
  semillas. Es el agujero del §2.3 y es lo único que esta campaña existe para mover.
- **W-2 · CONSERVACIÓN.** `err_identidad` **se mantiene en 0,0000** en las tres. `lat2` lo dejó en
  cero y una pérdida auxiliar no tiene por qué moverlo; si lo mueve, la condición cuesta más de lo
  que trae.
- **W-3 · NO-INTERCAMBIO.** `falsa_abst` ≤ 0,10 en las tres y `acierto` no cae más de 0,05 respecto
  de la gemela. Sin esto, cualquier ganancia en `nose_rel` puede ser el modelo abstiéndose más.

---

## 6. Regla de decisión, comprometida por adelantado

- **E-1 falla** → **la línea se cierra acá y no se entrena nada.** No hay señal de recuperabilidad en
  la escritura para que una cabeza lea, y construirla es otro proyecto. Se reporta como la octava vía
  y se suma a las siete, que es un resultado y no una decepción.
- **E-1 pasa y W-1 falla** → la señal existe en la escritura pero **no se transfiere** a detectar la
  ausencia de la relación. Es un negativo informativo y fuerte: diría que la ausencia sigue sin tener
  dónde vivir aun cuando la escritura sí sabe algo. Se cierra sin probar la Variante B.
- **E-1 y W-1 pasan, W-2 falla** → la cabeza compra detección de ausencia **rompiendo** el
  direccionamiento que `lat2` había arreglado. Se reporta el intercambio y **no** se adopta.
- **W-1, W-2 y W-3 pasan** → recién ahí se escribe el pre-registro de la Variante B. No antes.

---

## 7. Riesgo declarado

**El riesgo principal es la fuga de etiqueta.** La etiqueta de recuperabilidad sale del generador
sintético, y el generador sabe cosas que el modelo no puede saber en el momento de escribir. Una
sonda que dé AUC alta porque la etiqueta filtra información del futuro del episodio mediría el
generador, no el modelo. **Evidencia que separa las dos, declarada antes de correr:** se reporta la
AUC de la sonda alimentada **sólo con la posición de la entrada y la longitud del episodio**, sin
activaciones. Si esa sonda ciega ya alcanza el umbral, E-1 no es interpretable y no se lee.

**El riesgo hermano es el prior**, y tiene nombre propio en este proyecto. El slot nulo del 25-ago
falló aprendiendo la tasa base en vez de la pertenencia, y su masa convergió al prior con tres
decimales de acuerdo. Por eso se reporta, junto a la AUC, **la tasa base de la etiqueta** en el mismo
split. Una cabeza cuya salida media se pegue al prior es el mismo fracaso otra vez, y hay que poder
verlo sin discutirlo.

---

## 8. Lo que este pre-registro NO autoriza

- No autoriza correr nada mientras A5 esté ocupando las cuentas.
- No autoriza la Variante B, que necesita su propio pre-registro y su propia regla de decisión.
- No autoriza reinterpretar la idea como aplicada a `err_identidad`, que está en 0,0000 y donde ya se
  declaró en el §2.2 que no hay margen que medir.
