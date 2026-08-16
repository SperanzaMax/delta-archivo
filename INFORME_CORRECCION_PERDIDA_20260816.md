# La corrección no se pierde: el hecho entero no se alcanza

**2026-08-16** · `sonda_correccion.py` · CPU, cero GPU
Pre-registro `PREREG_CORRECCION_PERDIDA.md`, SHA `c51b36b4…`, congelado 21:22 UTC **antes** de correr.

## La hipótesis que se probaba

La sonda del vecino dejó dos hechos que ninguna hipótesis en juego explicaba junta: **vecino intacto
0,83** (no se corrompió a nadie) y **rescate 0,10** (el hecho propio tampoco se recupera). De ahí la
tercera historia: que la corrección elíptica **no se ligue a nadie** y se pierda al escribir, dejando
el hecho propio con su versión vieja.

Predicción de esa historia: al preguntar por la versión **anterior**, el modelo debería devolver la v1
sin problema —porque el hecho sí se archivó— y sólo faltaría la revisión.

## Resultado, n = 4000 por checkpoint

| sobre hechos **revisados** | `n3_s2` (N3) | `n4_s0` (N4) |
|---|---:|---:|
| **casos con `err_identidad`** | n = 428 | n = 379 |
| acierta la **anterior** (v1) | **0,5304** | **0,3193** |
| devuelve la vigente (invierte el orden) | **0,0000** | **0,0000** |
| otra cosa | 0,4696 | 0,6807 |
| **casos con acierto** *(control)* | n = 1494 | n = 1578 |
| acierta la anterior (v1) | **0,9498** | **0,9119** |

| predicción | `n3_s2` | `n4_s0` |
|---|---|---|
| P-1 anterior en errores ≥ 0,50 | 0,5304 ✓ | 0,3193 ✗ |
| P-2 control en aciertos ≥ 0,50 | 0,9498 ✓ | 0,9119 ✓ |
| P-3 aciertos > errores | ✓ | ✓ |

## Lectura

**1. La hipótesis de la corrección perdida queda refutada.** Si la revisión no se hubiera ligado pero
el hecho estuviera archivado, la v1 debería recuperarse tan bien como en los aciertos (0,91-0,95). No
pasa: en los casos fallidos la v1 **también** está degradada (0,53 y 0,32). **No se pierde la
corrección — se pierde el hecho entero, en todas sus versiones.**

**2. P-3 es el resultado, y es fuerte en los dos checkpoints.** Donde el modelo acierta la versión
vigente recupera además la anterior el 91-95 % de las veces; donde falla, sólo el 32-53 %. El
episodio se recupera completo o no se recupera: **el error de identidad no es un fallo por versión,
es un hecho que quedó inalcanzable bajo su propia consulta.**

Junto con la sonda del vecino, el cuadro cierra: **el vecino se recupera bien (0,83), el hecho propio
no se recupera en ninguna versión, y el modelo contesta el valor del vecino.** Eso es competencia de
claves —direccionamiento— y no un problema de escritura de la revisión.

**3. `devuelve la vigente` = 0,0000 EXACTO en los dos checkpoints.** Preguntando por la anterior, el
modelo **nunca** contesta la vigente. Es la confirmación más limpia que tenemos de que el versionado
está resuelto: replica por una vía independiente el `err_version ≤ 0,0078` del informe de SER, y
descarta que el sello de orden se esté usando como mera preferencia por lo último.

## Lo que NO se puede afirmar

**No sabemos si el hecho propio está en el archivo y es inalcanzable, o si nunca se escribió.** Las
dos posibilidades producen exactamente lo mismo desde afuera: no se recupera en ninguna versión. Para
separarlas hay que mirar el archivo directamente y preguntar si existe alguna entrada cuyo contenido
corresponda al hecho propio — medible con el score de matcheo, y es el paso siguiente.

Y otra vez **los dos checkpoints difieren en magnitud** (0,53 vs 0,32) aunque coinciden en el signo,
igual que en la sonda del vecino. Se reporta por checkpoint y no se promedia.

## Consecuencia para la campaña

Se refuerza lo que dijo la sonda del vecino y ahora por dos vías independientes: **el error es de
recuperación, no de escritura** → **convertible en abstención**, sin piso infranqueable. Y aparece un
blanco arquitectónico distinto del `NOSE`: el **direccionamiento** — por qué la clave de un hecho
pierde sistemáticamente contra la de su vecino en ciertos episodios.
