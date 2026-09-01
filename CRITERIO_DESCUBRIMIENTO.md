# Cuándo decirle a Maxi «esto es un descubrimiento», y cómo

**2026-09-01.** Pedido explícito de Maxi, y viene de un error mío:

> *«cuando realmente estemos con el descubrimiento quiero que me lo hagas saber de forma clara que
> hemos alcanzado la solución de un problema que hoy no existía, te lo digo porque cuando fue lo de la
> memoria recurrente insertada en el LM y que con eso no olvida nunca más, no me lo hiciste saber tan
> explícitamente y sin vueltas»*.

Tiene razón. E-I3 —el sello de orden que llevó el conflicto de versiones de **0,4570 a 0,9956**— se
reportó como una fila más de una tabla, entre otros seis resultados del día. **Era el resultado
principal del proyecto hasta ese momento y no se dijo así.**

## Las cuatro condiciones, y tienen que cumplirse LAS CUATRO

Un criterio laxo sería peor que no tener criterio: avisar de más entrena a no creerme.

1. **MEDIDO** — número principal, con su control, y el control pudiendo fallar. Nada de un solo
   checkpoint ni de una sola semilla.
2. **REPLICADO** — ≥3 semillas, o réplica en otra condición. Si es bimodal entre semillas, se dice la
   distribución, no la media.
3. **NO OCUPADO** — verificado contra la literatura **el mismo día**, leyendo los trabajos más
   cercanos y preguntándoles explícitamente por nuestras piezas. No alcanza con «no me suena».
4. **RESUELVE EL PROBLEMA DECLARADO**, no un proxy. El problema es el de
   `objetivo-memoria-persistente-llm`: que un LLM no olvide lo que se le dijo, y que diga «eso no lo
   sé» en vez de inventar.

## Cómo se comunica cuando pasa

- **Primera línea del mensaje, sin preámbulo**, diciendo qué problema se resolvió y que no estaba
  resuelto por nadie.
- El número principal **y su control**, en esa misma frase.
- **Qué NO cubre**, inmediatamente después. Un descubrimiento con su límite al lado sigue siendo un
  descubrimiento; sin el límite es una exageración.
- Va **por Telegram además de la terminal**, porque los resultados llegan cuando él no está mirando.

## Lo que NO califica, para que la vara quede clara

- Un negativo bien hecho, por valioso que sea (hoy: el afilado, la reparación del atractor).
- Un mecanismo que funciona en una condición y no replica (hoy: `b3_s0`/`b3_s1` con exactitud
  **1,0000**, que es el modelo que buscamos **pero sale 2 de 9 veces y las dos arrancaron de una base
  preentrenada**).
- Un diagnóstico, por preciso que sea (hoy: la relación fuera de la ventana de la conv). Un
  diagnóstico habilita la solución, no es la solución.

## Estado al 1-sep-2026 · lo que YA califica y no se dijo con estas palabras

**El sello de orden co-entrenado (E-I3, 13-ago) cumple las cuatro.** Medido 0,4570 → **0,9956** con
`barajado` como control —mismos parámetros, sello sin relación con el turno real— que da 0,4768 y
descarta que sea capacidad extra del lector; replicado en 5 semillas **sin un solo solape**; y
verificado hoy contra Co-LMLM (jul-2026) y Trained Persistent Memory (mar-2026), que **no tienen
versionado, ni sello de orden en la clave, ni preguntan por el valor superado**.

**Se le dijo a Maxi el 1-sep, con casi tres semanas de retraso.**

Su límite, que va pegado: es **suficiencia, no inferencia** —la versión vigente es siempre la de turno
mayor, así que no separa «usar el orden» de «preferir lo último»—, y E-I3c midió que a 12000 pasos
**2 de 5 semillas no convergen**.

**La abstención NO califica todavía**, y es lo que falta para el objetivo completo.
