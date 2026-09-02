# La ventana era el cuello de botella: con kernel 5 la abstención pasa de 0,78 a 0,97

**2026-09-01, noche.** Evalúa `PREREG_KERNEL_Q5.md` (SHA `50c4503d`), congelado antes de correr.
Tres semillas a 26000 pasos contra el control `v3` ya medido, idéntico salvo el kernel de `convq`.

## 1. Los cuatro criterios

| | criterio | resultado | |
|---|---|---|---|
| **K-0** BLOQUEANTE | la sensibilidad a la RELACIÓN deja de ser 0,0000 | **0,2898 · 0,3668 · 0,1898** | **CUMPLE 3/3** |
| **K-1** PRINCIPAL | AUC contra `nose_rel` ≥ 0,60 en ≥2 de 3 | **0,6085 · 0,5774 · 0,6372** | **CUMPLE 2/3** |
| **K-2** UTILIDAD | `nose` ≥ 0,90 y `falsa_abst` ≤ 0,10 en ≥2 de 3 | **3 de 3** (control **0 de 3**) | **CUMPLE** |
| **K-3** NO DAÑO | `vigente` ≥ 0,95 en ≥2 de 3 | **3 de 3** | **CUMPLE** |

## 2. La tabla

| unidad | kernel | vigente | anterior | `nose` | `nose_ent` | **`nose_rel`** | falsa | exactitud |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `kq3_s0` | **5** | 1,0000 | 1,0000 | 0,9697 | 0,9538 | **0,9931** | 0,0000 | **0,9879** |
| `kq3_s1` | **5** | 0,9964 | 1,0000 | 0,9861 | 0,9726 | **1,0000** | 0,0030 | **0,9934** |
| `kq3_s2` | **5** | 1,0000 | 1,0000 | 0,9699 | 0,9356 | **1,0000** | 0,0000 | **0,9880** |
| `v3_s0` | 3 | 1,0000 | 1,0000 | 0,8104 | 0,9851 | **0,6090** | 0,0000 | 0,9242 |
| `v3_s1` | 3 | 1,0000 | 1,0000 | 0,7771 | 1,0000 | **0,5850** | 0,0000 | 0,9108 |
| `v3_s2` | 3 | 0,9927 | 1,0000 | 0,8370 | 0,9414 | **0,7349** | 0,0064 | 0,9326 |

**`nose_rel` va de 0,5850–0,7349 a 0,9931–1,0000, sin un solo solape entre condiciones.** Es el caso
que el propio generador llama «el que se parece a una alucinación real»: la entidad **sí** está en el
archivo, y lo que no está es la relación que se pregunta de ella.

## 3. El mecanismo, verificado y no supuesto

La predicción de K-0 era falsable y se cumplió. Con kernel 3 la sensibilidad de la búsqueda a la
relación era **0,0000 exacto** en las tres celdas y en las dos familias, porque la relación cae a
distancia 3 de la posición de lectura y el kernel alcanza 2. **Está afuera de la ventana en el 100 %
de las consultas.** Con kernel 5 la ventana alcanza 4 y la relación queda cubierta siempre.

**Y la separación viene de la componente correcta.** En el caso difícil, lo que discrimina es la
sensibilidad a la **relación** (0,6085 · 0,5774 · 0,6372) mientras la **entidad** se queda en el azar
(0,5090 en `s0`). Es exactamente lo esperado, porque ahí la entidad está y lo que falta es la relación.
Si el efecto hubiera aparecido en las dos componentes por igual, sería inespecífico y no el mecanismo.

## 4. El costo, que existe y hay que declararlo

**`nose_ent` baja**, de 0,9414–1,0000 a 0,9356–0,9726. El caso fácil, donde la entidad directamente no
está, empeora un poco. El intercambio es muy favorable —se gana entre 0,26 y 0,41 en el caso difícil y
se pierde entre 0,01 y 0,05 en el fácil— pero **no es gratis**, y una lectura que sólo mire `nose`
global lo esconde.

## 5. Lo que este resultado NO dice

- **La distancia fija es una propiedad del generador.** Acá la relación está siempre a exactamente 3
  posiciones, y por eso el efecto es determinista. En texto real esa distancia varía y el mismo
  fenómeno pasaría de ser un cero exacto a una distribución.
- **Aplica a arquitecturas que forman la query con una ventana local**, es decir atención lineal,
  modelos recurrentes y de espacio de estados, y a cualquier modelo que consulte una memoria externa
  desde una capa temprana. Un transformer que forma la query con atención completa no tiene este
  problema puntual.
- **Es un tipo de alucinación, no todos.** Este modelo ya tenía medido `err_fuera` = 0,0000, o sea que
  nunca inventa contenido nuevo. Lo que arreglamos es la atribución equivocada de datos reales.
- **Falta la revisión de literatura** sobre este punto específico, y sin eso no se puede afirmar que
  el hallazgo esté libre. Es lo primero de mañana.

## 6. La lección que sí generaliza

> **La consulta con la que se busca en una memoria tiene que formarse donde ya se vio la pregunta
> completa.** Si se forma antes, el modelo recupera por una parte de la pregunta, ignora el resto, y
> después responde con confianza sobre algo que nunca se le dijo.

Y es verificable en cualquier arquitectura con la misma medición que se usó acá, que es barata y no
necesita entrenar nada: cambiar una parte de la consulta y ver si la búsqueda se mueve. **Si no se
mueve, esa parte es invisible para la búsqueda.**
