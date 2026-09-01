# La búsqueda no mira la relación, y la causa es que cae UN TOKEN afuera de la ventana

**2026-09-01.** Pedido de Maxi: probar **entidad contra relación**, y buscar *«algo en el aprendizaje
de búsqueda que podamos modificar»*. Tres mediciones de CPU, sin GPU. La tercera da un diagnóstico
cerrado y una intervención de una línea.

---

## 1. Dos intentos que fallaron, y el segundo por culpa mía

**(a) `entidad_vs_relacion.py` — inválido.** Leía la query desde la **posición** del token de entidad
y la del de relación. El tronco es **causal** (`modelo.py:106`), y en «cual es el `<REL>` de `<ENT>` ?»
la relación viene **antes**: en esa posición el modelo todavía no vio la entidad. No comparaba dos
vistas del mismo ítem — una miraba media pregunta. Dio AUC **0,4049** (invertido) con 84 % de
desacuerdo base. **No es un negativo de la idea, es un instrumento mal construido.**

**(b) Ablación sobre `p3_s0` y `n3_s0` — inválido, y el código lo decía.** Cambiar la entidad o la
relación de la consulta no movía la búsqueda **ni un poco** (TV = 0,0000 exacto). No es un hallazgo:
esas unidades tienen **`donde=pre`**, donde la query es `ln(emb[token]) @ qr`, **función pura del
token de su posición**. La docstring de `tronco` ya lo dice y le atribuye la colisión de clave. **En
`pre` la búsqueda es ciega a la consulta por construcción**, así que la medición no podía dar otra
cosa.

## 2. La medición válida, sobre las familias donde la query sí ve contexto

`v3` (`lat2`) y `w3` (`lat`), que forman la query con una conv causal de kernel 3. n=1536.

| | HAY respuesta | `nose_ent` | `nose_rel` |
|---|---:|---:|---:|
| `v3` · sensibilidad a la **entidad** (TV) | 0,7844 | 0,6305 | 0,7507 |
| `v3` · sensibilidad a la **relación** (TV) | **0,0000** | **0,0000** | **0,0000** |
| `w3` · sensibilidad a la **entidad** (TV) | 0,8452 | 0,6075 | 0,8502 |
| `w3` · sensibilidad a la **relación** (TV) | **0,0000** | **0,0000** | **0,0000** |

**La búsqueda usa la entidad y no mira la relación en absoluto.** Cambiar «el color» por «la altura»
no mueve la lectura ni una milésima, en ninguna de las dos familias ni en ninguno de los tres grupos.

**Y la sensibilidad a la entidad SÍ detecta una de las dos ausencias:** AUC **0,6106** (`v3`) y
**0,6588** (`w3`) contra la ausencia — comparable al 0,66 que dio el término de orden y cerca del
techo 0,7003. Tiene mecanismo: si la entidad no está en el archivo, cambiarla da lo mismo.

**Pero en el caso difícil no sirve: `nose_rel` da 0,5278 y 0,5082, azar.** Ahí la entidad **sí** está
—la búsqueda es tan sensible como en el caso sano (0,7507 · 0,8502)— y lo que falta es la relación,
que la búsqueda **no mira**.

## 3. ★ La causa, y es aritmética

| | distancia a la posición de lectura | dentro del kernel 3 (alcance 2) |
|---|---:|---:|
| **entidad** | **1** (mín 1, máx 1) | **1,0000** |
| **relación** | **3** (mín 3, máx 3) | **0,0000** |

En «cual es la altura de mercado **?**» la conv que forma la query alcanza `?`, `mercado` y `de`. La
relación —`altura`— está a distancia 3. **Queda afuera por un token, siempre, de forma determinista.**

> **`TV_rel = 0,0000` no era aproximado: es exacto porque la relación está fuera de la ventana en el
> 100 % de las consultas.** No es que el modelo aprenda a ignorar la relación: **no la puede ver**.

**Con kernel 5 (alcance 4) queda cubierta en el 100 %.**

## 4. Lo que esto reinterpreta

- **La colisión de clave tiene causa mecánica y trivial.** El proyecto la atribuyó a que la query no
  es conjunta; ahora se sabe **por qué** no lo es: la mitad de la consulta cae fuera del kernel.
- **La campaña de la query conjunta del 22-ago no podía funcionar.** `lat2` se diseñó exactamente para
  «que la query pueda depender de la entidad y de la relación a la vez» y le dio una `convq` **propia
  de kernel 3**. Le alcanzó para la entidad (distancia 1) —y por eso `lat` disolvió `err_identidad` a
  0,0000— y **nunca pudo alcanzar la relación**. Media hipótesis quedó sin probar.
- **`nose_rel` es indetectable en la búsqueda por construcción**, y es el caso que el propio generador
  llama «el que se parece a una alucinación real».

## 5. La intervención que se deriva, y es de una línea

**`convq` con kernel 5** en vez de 3. Es lo mínimo que hace que la query vea entidad **y** relación,
o sea la primera query realmente conjunta del proyecto. Cambia la forma del árbol de parámetros
(`convq` pasa de 3 a 5), así que **no es compatible con los checkpoints** y hay que entrenar desde
cero: es el costo real del experimento.

**Predicción falsable, antes de correrlo:** si la causa es la ventana, con kernel 5 la sensibilidad a
la relación tiene que dejar de ser 0,0000, y el AUC contra `nose_rel` —hoy **0,51, azar**— tiene que
subir. Si sube la sensibilidad pero **no** el AUC, la ventana era necesaria y no suficiente, y el
cuello sigue aguas abajo.

## 6. Lo que no dice

Dos unidades por familia, un nivel, un `p_nose`. La distancia 1/3 es una propiedad **del generador**
(«cual es `<art>` `<sust>` de `<ent>` ?»): en texto real las distancias varían, y ahí el argumento
sería estadístico y no determinista. Lo que no cambia es el mecanismo — **una query formada por una
ventana fija sólo puede condicionarse a lo que entra en la ventana.**
