# EL CRUCE · la misma pregunta, reordenada, tiene que dar vuelta el fallo · congelado ANTES de correr

**2026-09-02.** Es el experimento que decide si el hallazgo de la ventana es una **ley** o una
propiedad del generador.

## De dónde sale

Hoy se midió, en las tres semillas de las dos familias, que la sensibilidad de la búsqueda a un token
de la consulta es **0,000000 exacto** en cuanto ese token pasa el alcance de la conv, y que el corte
se mueve con el kernel: con kernel 3 (alcance 2) el escalón cae entre d=2 y d=3, y con kernel 5
(alcance 4) entre d=4 y d=5. **59 de 60 celdas**, y la única que falló fue un artefacto de recorte ya
corregido.

Pero en el idioma de siempre hay **una sola forma de preguntar**, y en ella la entidad está a
distancia 1 y la relación a 3. Eso confunde dos explicaciones que dan la misma predicción:

- **(V) la ventana** — falla lo que queda lejos, sea lo que sea;
- **(D) la dificultad** — la relación es simplemente más difícil que la entidad.

## El diseño, que las separa

Se entrena con **dos formas de pregunta mezcladas al azar**, con el mismo contenido y sólo el orden
cambiado. Las distancias están verificadas token a token en `chequeo_formas_q.py`:

| forma | texto | d(relación) | d(entidad) |
|---|---|---:|---:|
| `directa` | cual es `<art>` `<sust>` de `<ent>` ? | **3** | **1** |
| `invertida` | cual es para `<ent>` `<art>` `<sust>` ? | **1** | **3** |

Con **kernel 3** (alcance 2), la predicción de **(V)** es un **cruce**:

| | `directa` | `invertida` |
|---|---|---|
| la búsqueda ve la **entidad** | **sí** (d=1) | **no** (d=3) |
| la búsqueda ve la **relación** | **no** (d=3) | **sí** (d=1) |
| falla esperada | `nose_rel` | **`nose_ent`** |

**(D) predice que `nose_rel` es la que sufre en las dos formas**, porque la relación seguiría siendo
lo difícil sin importar dónde esté escrita.

Unidades **`cf3_s0/s1/s2`**: idénticas a la familia `v3` —`donde=lat2`, nivel 3, `p_nose` 0,4,
`abst=cabeza`, `mezcla` fija, `lr` 1e-3, 26000 pasos, **kernel 3**— salvo
`--formas-q directa,invertida`. Desde cero.

## Criterios

- **X-0 · BLOQUEANTE, mecanicista.** La sensibilidad a la **entidad** en `invertida` tiene que ser
  **0,000000** y la sensibilidad a la **relación** en `invertida` **> 0,01**, o sea al revés que en
  `directa`. Es aritmética de la ventana y si no da, el generador no hace lo que dice y **nada más se
  lee**.
- **X-1 · PRINCIPAL, el cruce.** Dentro de `invertida`, **`nose_ent` < `nose_rel`**, y dentro de
  `directa`, **`nose_rel` < `nose_ent`**, en **≥2 de 3** semillas. Es una **interacción**, no dos
  efectos: lo que se afirma es que el orden se invierte.
- **X-2 · MAGNITUD.** La diferencia `nose_ent − nose_rel` cambia de signo entre las dos formas con un
  salto **≥ 0,15** en ≥2 de 3. Sin un tamaño mínimo, un cruce puede ser ruido.
- **X-3 · NO DAÑO.** `vigente` ≥ 0,90 en ≥2 de 3 y en las dos formas. Si la forma `invertida` no se
  aprende, el cruce sería un artefacto de que una de las dos plantillas es más difícil de leer.
- **Legibilidad:** menos de 2 unidades a 26000 → **NO EVALUABLE**.

## Lo que este experimento NO puede decir

- **No prueba que en texto real pase lo mismo.** Prueba que, dentro de este idioma, lo que decide el
  fallo es **dónde está escrito** un componente y no **qué componente es**. La transferencia a texto
  natural, donde las distancias son una distribución y no dos valores, queda abierta.
- **No mide el kernel 5 con formas mezcladas.** La predicción obvia —que con alcance 4 las dos formas
  quedan cubiertas y el cruce desaparece— es la continuación natural y no se corre acá.
- Y si el cruce **no** aparece, la explicación (D) queda viva y el hallazgo del kernel 5 pasa a ser un
  resultado sobre **esta** pregunta, no sobre la geometría de las consultas.
