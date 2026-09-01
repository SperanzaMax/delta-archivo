# Control de PRECISIÓN del desacuerdo · criterios congelados ANTES del dato

**2026-09-01, mañana.** Es el control que el `INFORME_DOS_VECES_20260831.md` declara faltante en su
§4 y que el `PLAN_20260901.md` pone como lo PRIMERO del día:

> «No se comparó la precisión del desacuerdo contra la precisión de la confianza de salida en el mismo
> percentil de cobertura. **Ese es el control que falta**, y es el análogo de D-4 para precisión.»

Sin él, el **0,8980** del 31-ago no está atribuido al desacuerdo: podría ser que marcar por confianza
baja al mismo 9,6 % de cobertura dé lo mismo, y entonces preguntar dos veces no compra nada que una
sola pasada no diera ya. **D-4 se disparó para el AUC** (0,6054 contra 0,6054); esto pregunta si
también se dispara para la precisión, que es la métrica bajo la cual el detector se veía útil.

---

## 1. Definiciones, para que no se muevan después

- **«mal»** = la respuesta emitida no coincide con el objetivo, **incluyendo** las preguntas sin
  respuesta en el archivo (donde cualquier valor es incorrecto por construcción). Es la misma
  definición del 31-ago, con su límite ya declarado: no separa error de ausencia.
- **desacuerdo** = las dos pasadas con ruido independiente devuelven distinto argmax. Cobertura `c*`
  = la fracción que marca, y **no se elige**: sale del mecanismo.
- **confianza** = `max softmax` sobre los logits con `NOSE` enmascarado, la MISMA definición que usó
  `desacuerdo_busqueda.py` en D-4. Se mide en dos variantes y **el control se queda con la mejor**:
  (a) en la pasada **limpia** (1 forward, que es lo que costaría en producción), (b) en la pasada
  ruidosa `r1` (simétrica con el desacuerdo). Darle al control su mejor versión es lo que haría fuerte
  un positivo.
- **igual cobertura** = se marca por confianza el `c*` de preguntas con confianza más baja, con `c*`
  tomado del desacuerdo. Comparar precisiones a coberturas distintas no significa nada.

## 2. Criterios

- **P-1 · PRINCIPAL.** `precisión(desacuerdo) − precisión(confianza @ c*) ≥ +0,05` **y** el IC95
  bootstrap (10.000 remuestreos, pareado por muestra) de esa diferencia **excluye el 0**.
  Si NO se cumple → **el 0,8980 no se atribuye al desacuerdo**: es otra forma de leer la confianza,
  igual que dictaminó D-4 sobre el AUC. La idea de preguntar dos veces queda **sin adjudicar en su
  versión post-hoc**, sin que eso toque la versión entrenada (que sigue sin probar).
- **P-2 · PISO.** `precisión(desacuerdo) > tasa base + 0,10`, con IC95 excluyendo la tasa base.
  Confirma con `n` mayor lo que el 31-ago midió con `n=512`.
- **P-3 · RÉPLICA.** El 0,8980 del 31-ago tiene que caer dentro del IC95 de esta medición.
  Si no cae, **el número de ayer era ruido de `n` chico** y hay que decirlo con esas palabras.
- **P-4 · DESCRIPTIVO, NO ADJUDICA.** Solapamiento (Jaccard) entre el conjunto marcado por desacuerdo
  y el marcado por confianza. Un solapamiento bajo con precisiones parecidas significa que **señalan
  preguntas distintas** y que combinarlos compra cobertura; queda anotado como vía, nunca como
  resultado de esta corrida.

## 3. Riesgo de legibilidad, declarado (lección O-6 del 31-ago)

Si el desacuerdo marca **menos de 30 preguntas**, la precisión no es estimable y el juez tiene que
devolver **NO EVALUABLE**, no un número. Esto es una precondición de P-1, P-2 y P-3 a la vez, y se
dice cuáles: **los tres**. La regla del 31-ago —«cuando un criterio de riesgo protege la legibilidad
de otros, hay que decir CUÁLES, y el juez tiene que devolver NO EVALUABLE en vez de un número»— se
aplica acá por primera vez desde que se escribió.

## 4. Presupuesto

`n = 1536` (contra 512 de ayer), σ=0,4, un modelo (`n3_s0`). Tres forwards por muestra: limpia + dos
ruidosas. Es **menos** trabajo que `desacuerdo_busqueda.py` de ayer (que hizo 33 por muestra) y ya se
midió que eso entra en la PC. Se corre igual un smoke con `n` chico primero, por la regla del §3 del
plan. Sin GPU.

## 5. Lo que esta corrida NO puede decir

Un solo modelo, un solo σ, la versión **post-hoc con ruido** que ya está declarada como la más débil
de las tres. **No mide la idea de Maxi, mide su versión más barata.** Las dos fuertes —dos queries
`qr1`/`qr2` aprendidas con el desacuerdo en la pérdida, y buscar por entidad contra buscar por
relación— siguen intactas gane o pierda esto.
