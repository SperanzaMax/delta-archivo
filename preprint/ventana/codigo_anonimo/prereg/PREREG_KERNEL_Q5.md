# Que la query VEA la relación · kernel 5 en `convq` · congelado ANTES de correr

**2026-09-01.** Sale de `INFORME_QUERY_CIEGA_20260901.md`, medido hoy: en «cual es `<art>` `<sust>`
de `<ent>` ?» la **entidad** queda a distancia **1** de la posición de lectura y la **relación** a
distancia **3**, las dos de forma determinista. La conv que forma la query en `lat`/`lat2` tiene
kernel 3 (alcance 2), así que **la relación cae un token afuera de la ventana en el 100 % de las
consultas**, y la sensibilidad de la búsqueda a la relación es **0,0000 exacto** en `v3` y `w3`, en
los tres grupos. No es que el modelo la ignore: **no la puede ver**.

## 1. Diseño

**Tratamiento `kq3_s0/s1/s2`** — idéntico a la familia `v3` (`donde=lat2`, nivel 3, `p_nose` 0,4,
`abst=cabeza`, `mezcla` fija, `lr` 1e-3, 26000 pasos) **salvo `--kernel-q 5`**. Desde cero: `convq`
cambia de forma y no hay checkpoint compatible.

**Control `v3_s0/s1/s2`, ya medido y NO se re-corre.** Misma configuración con kernel 3.
Sus números: RECUP **1,0000** en las tres · `nose` 0,8104 · 0,7771 · (s2) · `falsa_abst` 0,0000 ·
sensibilidad a la relación **0,0000** · AUC contra `nose_rel` **0,4914**.

`convq` arranca en `[1,0,0,0,0]`, o sea `convk(convq,z)==z` exacto (verificado hoy), así que **`lat2`
con kernel 5 sigue conteniendo a `pre` como caso particular** y no puede ser estructuralmente peor.
Costo: **+1.280 parámetros** sobre 865.395 (0,15 %).

## 2. Criterios

- **K-0 · BLOQUEANTE, mecanicista.** La sensibilidad de la búsqueda a la **relación** (distancia TV al
  ablar el token de relación) tiene que dejar de ser 0,0000 en **≥2 de 3** unidades, con media
  **> 0,05**. Si sigue en cero, el kernel no es la causa y **todo lo demás no se lee**: sería que la
  query no usa la ventana aunque la tenga.
- **K-1 · PRINCIPAL.** El AUC de la búsqueda contra `nose_rel` —hoy **0,4914, azar**— sube a
  **≥ 0,60** en ≥2 de 3. Es la pregunta entera: si la query ve la relación, ¿puede notar que **eso**
  que se le pide de esa entidad no está?
- **K-2 · UTILIDAD.** `nose` ≥ **0,90** en ≥2 de 3 manteniendo `falsa_abst` ≤ 0,10. El control da
  `nose` 0,78-0,81 con `falsa_abst` 0,0000, así que esto pregunta si la ventana compra **abstención
  real**, no sólo señal en la sonda.
- **K-3 · NO DAÑO.** RECUP ≥ 0,95 en ≥2 de 3. El control tiene 1,0000: la ventana más ancha no puede
  costar recuperación.
- **Secundario, no adjudica:** los taps aprendidos de `convq`. Si el tap de distancia 3 (el que
  alcanza la relación) queda en ~0, el modelo tuvo la ventana y no la usó, y eso se informa aunque
  K-1 cumpla. Control gratis ya existente: los `convq` de los bloques 1-3 no reciben gradiente, así
  que su movimiento es weight decay puro y separa aprendizaje de decay sin simular nada.

**Riesgo de legibilidad (protege a K-1, K-2 y K-3):** si llegan a 26000 menos de 2 unidades,
**NO EVALUABLE**.

## 3. Lo que no puede decir

Tres semillas, un nivel, una arquitectura. La distancia 1/3 es una propiedad **del generador**; en
texto real las distancias varían y el argumento pasa a ser estadístico. Y si K-1 cumple, **no
distingue** si la ganancia viene de ver la relación o simplemente de tener más contexto: para eso
haría falta un control con kernel 5 donde el tap de la relación esté forzado a cero, que no se corre
acá y queda declarado como el paso siguiente.
