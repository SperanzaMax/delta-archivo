# PREREG · ¿la corrección elíptica se pierde al escribir?

**2026-08-16** · continuación directa de `INFORME_SONDA_VECINO_20260816.md`. CPU, cero GPU.
**Se hashea y ancla ANTES de correr.**

## §1 · La tercera historia

La sonda del vecino dejó un resultado que ninguna de las dos hipótesis en juego explica bien:

- **vecino intacto 0,8301** — el archivo del vecino no está corrupto (descarta «se ligó al vecino»);
- **rescate 0,1049** — el hecho propio tampoco se recupera reformulando la consulta.

Las dos cosas a la vez admiten una explicación que no estaba ni en nuestra tabla ni en la de la
revisión externa: **que la corrección elíptica no se ligue a NADIE.** Que se pierda al escribir, sin
corromper a ningún vecino. El hecho propio quedaría en el archivo con su versión **vieja**, y el
modelo contestaría el valor de otra entidad simplemente porque el suyo nuevo no está en ninguna parte.

Importa porque cambia dónde se arregla: no es un problema de recuperación (nada que abstenerse
resuelva) ni de corrupción del vecino, sino de **la escritura de la corrección**.

## §2 · El test

Sobre los mismos episodios, en los casos donde el modelo comete `err_identidad`, se pregunta por la
versión **ANTERIOR** del hecho propio: `pregunta(rel, ent, "anterior")` → la respuesta correcta es la
v1, la que la corrección reemplazó.

| resultado | interpretación |
|---|---|
| devuelve **v1** bien | el hecho propio SÍ está en el archivo, con su versión vieja → **la corrección se perdió al escribir** |
| tampoco devuelve v1 | el hecho propio no se recupera en absoluto → el problema es de direccionamiento, no de la corrección |

## §3 · Predicciones

- **P-1 (principal).** Entre los casos de `err_identidad`, el acierto sobre la versión **anterior** es
  **≥ 0,50**. Si se cumple, el hecho propio está archivado y lo que falta es la corrección: la falla
  es de **escritura de la revisión**, no de recuperación del hecho.
- **P-2 (control, y puede fallar).** Entre los casos de **acierto**, el acierto sobre la versión
  anterior es **≥ 0,50**. Es el piso de sanidad: si acá tampoco anduviera, la consulta por la anterior
  estaría rota y P-1 no sería interpretable.
- **P-3 (discriminación).** El acierto sobre la anterior es **más alto en los aciertos que en los
  errores**. Si fueran iguales, el estado del hecho propio no tiene relación con que el modelo yerre
  la identidad y esta vía no explica nada.

## §4 · Controles

- Mismo episodio y mismo archivo: sólo cambia el tensor de la consulta, igual que en la sonda del
  vecino.
- Se reporta n de cada celda y se separa por checkpoint; **no se promedian los dos**, porque en la
  sonda del vecino los dos checkpoints se contradijeron y esa fue la información importante.
- Se reporta también qué contesta cuando no acierta la v1: si devuelve la **v2** (la corrección) al
  preguntar por la anterior, el modelo tiene las dos versiones pero **invierte el orden**, que es un
  tercer modo de falla y no debe contarse como «perdida».

## §5 · Límite

Checkpoints con `p_nose = 0`, una semilla por nivel. Mide dónde está el hecho, no cuánto mejoraría al
arreglar la escritura.
