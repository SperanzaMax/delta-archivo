# Informe · Consistencia de ida y vuelta

`PREREG_ROUNDTRIP.md` (SHA `55ba857a…`) · desviaciones en `DESVIACIONES_ROUNDTRIP.md` · datos en
`roundtrip_20260820.json`, `diag_roundtrip_20260820.json`, `diag_relacion_20260820.json` y
`diag_relacion_20000.json`.

## 1 · El veredicto formal: ARCHIVADO SIN INTERPRETAR

Las 8 unidades de la familia `c` a 14000 pasos, 2048 muestras cada una, `p_nose = 0,4`:

| unidad | RT-0a | RT-0b | RT-1 AUC | `f_abst` | `nose` | σ>0,5 `f`/`nose` | RT-3 (ok / sin-resp) | RT-4 | RT-5 (n) | H ok/err |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| c1_s0 | 1,000 | 0,992 | **0,975** | 0,0154 | 0,223 | 0,011 / 0,926 | 0,844 (0,896/0,777) | 0,616 | 1,000 (8) | 0,14 / 1,08 |
| c2_s0 | 1,000 | 0,991 | **0,932** | 0,0161 | 0,349 | 0,005 / 0,846 | 0,790 (0,859/0,693) | 0,540 | 1,000 (7) | 0,33 / 0,75 |
| c3_s0 | 1,000 | 0,241 | 0,502 | 0,7589 | 0,763 | 0,118 / 0,612 | 0,212 (0,204/0,223) | 0,473 | 0,268 (254) | **1,386 / 1,386** |
| c3_s1 | 1,000 | 0,229 | 0,483 | 0,7721 | 0,765 | 0,090 / 0,587 | 0,201 (0,184/0,209) | 0,494 | 0,241 (237) | **1,386 / 1,386** |
| c3_s2 | 1,000 | 0,260 | 0,524 | 0,7492 | 0,765 | 0,094 / 0,553 | 0,196 (0,194/0,209) | 0,491 | 0,238 (235) | **1,386 / 1,386** |
| c4_s0 | 1,000 | 0,394 | 0,554 | 0,6288 | 0,696 | 0,085 / 0,589 | 0,275 (0,289/0,266) | 0,470 | 0,282 (195) | 1,30 / 1,31 |
| c4_s1 | 1,000 | 0,366 | 0,611 | 0,6659 | 0,725 | 0,095 / 0,653 | 0,251 (0,261/0,248) | 0,511 | 0,278 (241) | 1,35 / 1,36 |
| c4_s2 | 1,000 | 0,401 | 0,590 | 0,6332 | 0,695 | 0,193 / 0,670 | 0,284 (0,282/0,301) | 0,496 | 0,295 (224) | 1,30 / 1,33 |

**RT-1 2/8** (pedía 6/8) · **RT-2 0/8** compuerta y 0/8 dominancia · **RT-3 0/8** · RT-4 7/8 (falla en
`c1_s0`) · **RT-5 2/8**, y esos dos son `c1_s0` y `c2_s0` con n = 8 y n = 7 errores, o sea nada.

El §4 tiene una cláusula que domina a las demás: «RT-3 o RT-4 fallan → se archiva sin interpretar,
cualesquiera sean RT-1, RT-2 y RT-5». **RT-3 falla en las 8.** Así que esto no es «el round-trip da
negativo» ni «la lectura de la disyunción queda refutada»: es un experimento cuyos controles dicen
que el instrumento no estaba midiendo lo que la pregunta necesitaba, y el prereg lo previó.

**RT-0 sí pasó**, y por eso hubo experimento: la sustitución mueve los logits en 1,000 de las
muestras (RT-0a) y la vuelta cierra en 0,992 de los aciertos de `c1_s0` (RT-0b, el bloqueante).

**La corrida se rehízo** después de la D-3 y reprodujo los ocho números **bit a bit** (rng
determinista): 0,975 · 0,932 · 0,502 · 0,483 · 0,524 · 0,554 · 0,611 · 0,590.

## 2 · Dónde se rompe, y el número que lo delata

La partición es por nivel, no por semilla. En N1/N2 la vuelta funciona bien (cierra en 0,99 de los
aciertos, AUC 0,93-0,98, posterior concentrada: H = 0,14 y 0,33). En N3 se cae al azar, y la firma es
**H = 1,386 = ln 4 exacto en las tres semillas, idéntico en aciertos y en errores**: la posterior
sobre las candidatas es **uniforme**. Contra rivales que ni siquiera aparecen en el episodio (RT-3),
la vuelta sigue sin preferir la entidad preguntada 4 de cada 5 veces.

O sea: **en N3/N4 el logit del valor emitido casi no depende de la entidad por la que se pregunta**,
que es justo la condición que la vuelta necesita para existir.

## 3 · Los dos diagnósticos, y el segundo da vuelta al primero

Había dos lecturas del mismo número y el prereg no permitía elegir ninguna sin medir:

**(A)** el modelo marginaliza sobre la entidad de origen —la lectura de la disyunción— o **(B)**
sustituir la entidad saca la consulta de distribución y el instrumento deja de leer.

**Diagnóstico 1 (`diag_roundtrip.py`) — descarta (B).** Se mide si **la respuesta** cambia al
sustituir la entidad, 512 muestras por unidad:

| unidad | contesta lo mismo a todas | respuestas distintas | resp = valor de esa otra entidad |
|---|---:|---:|---:|
| c1_s0 | 0,010 | 3,90 | 0,685 |
| c2_s0 | 0,025 | 3,61 | 0,838 |
| c3_s0 | **0,973** | 1,03 | 0,162 |
| c3_s1 | **0,982** | 1,02 | 0,158 |
| c4_s0 | 0,734 | 1,27 | 0,195 |
| c4_s2 | 0,697 | 1,32 | 0,210 |

En N1/N2 el modelo contesta según a quién se le pregunte. En N3 contesta **lo mismo para las cuatro
entidades en el 97-98 % de los episodios**. La entidad no entra en la decisión: (B) queda descartada.

**Diagnóstico 2 (`diag_relacion.py`) — y acá aparece la causa, que no es (A).** La pregunta lleva
**relación + entidad**. Con 4 hechos sobre 6 relaciones, la relación sola casi siempre identifica el
hecho, así que un modelo que se apoya en ella ignora la entidad sin necesidad de promediar nada. La
cuenta que lo hizo sospechoso: `P(la relación preguntada esté repetida) = 1 − (5/6)³ = 0,4213`, y si
ahí elige a ciegas el error sería **0,2107** — contra `err_identidad` medido de 0,19-0,21.

Separando por si la relación preguntada es **única** o está **repetida** en el episodio (1024
muestras, sólo preguntas con respuesta):

| unidad | n única | acierto única | **`err_ident` única** | n repetida | acierto repetida | **`err_ident` repetida** |
|---|---:|---:|---:|---:|---:|---:|
| c1_s0 | 596 | 0,9966 | **0,0000** | 428 | 0,9673 | 0,0164 |
| c2_s0 | 611 | 0,9771 | **0,0033** | 413 | 0,9661 | 0,0169 |
| c3_s0 | 583 | 0,9760 | **0,0137** | 441 | 0,5533 | **0,4376** |
| c3_s1 | 605 | 0,9868 | **0,0066** | 419 | 0,4535 | **0,5442** |
| c3_s2 | 588 | 0,9881 | **0,0051** | 436 | 0,4977 | **0,5000** |
| c4_s0 | 617 | 0,9400 | **0,0097** | 407 | 0,5725 | **0,3833** |
| c4_s1 | 589 | 0,9610 | **0,0119** | 435 | 0,5586 | **0,4230** |
| c4_s2 | 604 | 0,9752 | **0,0116** | 420 | 0,5762 | **0,4000** |

**El error de identidad vive ENTERO en la colisión de relación.** Con relación única el modelo acierta
0,94-0,99 y `err_identidad` es ≈ 0,01. Con relación repetida acierta 0,45-0,58 y erra 0,38-0,54, o
sea **el azar entre las dos candidatas que empatan**. Y `0,42 × 0,44 ≈ 0,185` reconstruye el
`err_identidad` global medido.

**Y N1/N2 muestran que no es una imposibilidad de la tarea:** ahí la relación también se repite el
40 % de las veces y `err_identidad` es 0,016. En los niveles fáciles el modelo **sí** usa la entidad.

## 4 · Control: cuánto de esto es presupuesto

Las mismas tres unidades de nivel 4, a 14000 contra 20000 pasos (los checkpoints que cerró la réplica
de hoy):

| unidad | `err_ident` repetida 14000 → 20000 | acierto repetida 14000 → 20000 | `err_ident` única a 20000 |
|---|---:|---:|---:|
| c4_s0 | 0,3833 → **0,2531** | 0,5725 → 0,7199 | 0,0000 |
| c4_s1 | 0,4230 → **0,2483** | 0,5586 → 0,7333 | 0,0068 |
| c4_s2 | 0,4000 → **0,1833** | 0,5762 → 0,8048 | 0,0000 |

**Con presupuesto el modelo empieza a usar la entidad**: la colisión pasa de casi-azar a 0,18-0,25.
No se resuelve —sigue siendo el error dominante— pero **no es un techo estructural**, es la parte de
la tarea que se aprende último. Es el mismo patrón que E-I3b (preferir-lo-último y usar-el-orden son
dos capacidades y se aprenden en momentos distintos) y el mismo del día de hoy con `falsa_abst`.

## 5 · Qué queda de la lectura de la disyunción

**Su predicción concreta (RT-5) no pudo evaluarse.** «En los errores de identidad la vuelta apunta al
dueño real» presupone que el modelo condiciona en la entidad, y en los niveles donde hay errores de
identidad no lo hace. No hay evidencia a favor ni en contra.

**Pero la estructura del argumento sobrevive, con el eje corrido, y ahora tiene un número exacto.** El
planteo era: la atención promedia entradas que son mutuamente excluyentes, y el promedio de dos
eventos disjuntos no es un evento válido. Lo medido dice que eso pasa **cuando dos entradas empatan
en la clave que el modelo usa para leer** —que resultó ser la relación, no la entidad—, y que cuando
pasa el resultado es exactamente lo que predice una mezcla: **0,50 entre las dos candidatas, en
silencio** (`err_fuera` = 0, sin abstención). La disyunción no se rompe por marginalizar sobre la
entidad; se rompe **por una clave de lectura que no distingue**, y el efecto observable es el que
describía el planteo.

## 6 · La vía que esto abre, y es mejor que las tres cerradas

Las tres vías cerradas (logit, mezcla de gaussianas, desacuerdo) buscaban la señal de «no sé» en la
**salida**. Lo de acá dice dónde está en la **entrada**: el error dominante ocurre exactamente cuando
**dos entradas del archivo empatan en la clave**, y eso es un evento **objetivo, medible en el paso
de lectura y sin una sola etiqueta** —dos pesos de atención altos y parecidos—.

Es, además, lo que el `INFORME_MONITOR` pedía y no encontraba: algo que separe «anclado en la entrada
correcta» de «anclado en cualquier entrada». Un empate en la clave es precisamente el caso en que
«cuál entrada» está indeterminado, y el modelo hoy lo resuelve tirando una moneda sin decirlo.

Queda como propuesta, sin correr, y con su prereg por escribir.

## 7 · Desviaciones

`D-1` (el logit va normalizado; verificado que no cambia nada), `D-2` (mi explicación de por qué
fallaba RT-3 quedó refutada por el desglose que agregué para verificarla) y **`D-3`**, que es la
importante: los diagnósticos importan la sonda para reusar dos funciones, la sonda tenía todo a nivel
de módulo, y **el import la ejecutaba entera y le sobrescribía el JSON a la corrida declarada**.
Arreglado con `main()` + guard; la corrida del prereg se rehízo y coincide bit a bit. La regla del
20-ago se extiende: *un script que se importa no puede tener efectos al importarse.*
