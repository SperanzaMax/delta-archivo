# ESTADO · 27 de agosto, cierre

Para retomar mañana sin releer nada más que este archivo.

---

## 1. Lo que se cerró hoy — tres líneas, y las tres contra la misma pared

### A5 · `--blanco error` — LA VÍA SE CIERRA
`INFORME_A5_BLANCO_ERROR_20260827.md`. Las tres semillas a 26000 contra `p3_*` a 26000.

| | veredicto |
|---|---|
| E-0 bloqueante | CUMPLE 2/3 |
| **E-1 PRINCIPAL** | **NO CUMPLE 1/3** (−0,0137 · −0,1708 · +0,0653) |
| E-2 mecanicista | NO CUMPLE 1/3 |
| E-4 colapso al prior | **sin colapso** en ninguna |

Celda del §6, escrita antes de correr: **«ninguna, sin colapso → el blanco no era el problema y la
vía se cierra»**.

**Lo que sí encontró:** la cabeza con blanco `error` ordena «¿me voy a equivocar?» con AUC **1,0000 y
0,9998**. Sabe. Lo que no hay es cómo convertir eso en menos error.

**Y el resultado de fondo es sobre VARIANZA.** En `s1` el que se rompe es el **control**; en `s2` se
rompe el tratamiento y **le reaparece `err_identidad`** (hasta 0,1215) que `lat2` tenía en 0,0000.
La §3.1 lo acota con Chebyshev: a la escala del efecto buscado la cota da 24,05 y 6,01, o sea **supera
1 y no dice nada**. «No promediar» dejó de ser convención y es aritmética.

### `escriba` (idea de Maxi) — CERRADA EN FASE 0, sin gastar GPU
`INFORME_ESCRIBA_FASE0_20260827.md`. E-1 pedía AUC ≥ 0,65 y dio **0,6392 · 0,5560 · 0,5275**, con
los dos controles sanos. En el vector que se archiva **no hay señal de recuperabilidad**, así que esa
cabeza tendría que *crear* la representación en vez de leerla.

Se buscó la explicación alternativa antes del veredicto (sondear la última escritura mezcla consultas
por versión vigente y anterior) y el control sólo-vigente dio 0,5537 y 0,5215: **no la sostenía**.

### Ausencia de la RELACIÓN — R-1 pasa, y el control la desarma
`INFORME_RELACION_FASE0_20260827.md`. R-1 cumplió 3/3 (0,8054 · 0,8130 · 0,8972)… y **casi toda la
señal vivía en el estado final**. El control post-hoc contra la confianza de salida dio **−0,0003 /
+0,0070 / +0,0302** contra un umbral de 0,03: **dos de tres no aportan nada**.

> Era el modelo mirándose al espejo. **No se recomienda lanzar la condición.**

**Lo que sí queda establecido:** el negativo del 16-ago **no era** artefacto de mezclar `nose_ent`
con `nose_rel`. Controlando por entidad, las señales del archivo siguen en azar (`s_max`
0,4906-0,5100). Ahora está medido en el eje fino además del grueso.

---

## 2. LA CONCLUSIÓN DEL DÍA, que es lo que hay que llevarse

**Tres caminos independientes, la misma pared:**

> **El modelo sabe cuándo no sabe. Lo que falta es convertir eso en la decisión de callarse.**

Es calibración, no capacidad — el mismo techo que el trípode ya había nombrado. Que tres vías
distintas choquen ahí el mismo día vale más que cualquiera de las tres por separado, **y es material
de paper**.

---

## 3. Estado de la infraestructura

**Nada corriendo.** Cero procesos, cero sesiones de Colab, todas las cuentas libres, CPU 36 °C.

**El token de Telegram se ROTÓ.** Estaba hardcodeado en 22 scripts de este repo, que es público, y
pusheado — verificado bajándolo del raw de GitHub. Ahora vive sólo en
`~/.config/avisos/telegram.env` (chmod 600) y los scripts hacen `. micro_lm/tg_token.sh`. El anterior
está **revocado y verificado muerto** (`getMe` → 401), así que lo que quedó en el historial de git es
inofensivo.

**Instrumentos nuevos, todos con su chequeo corrido:** `ser_cobertura.py` (SER a cobertura igualada),
`a5_e2_e4.py` (E-2/E-4), `escriba_fase0.py`, `relacion_fase0.py`, `vigia_b3_15min.sh`, `tg_token.sh`.
Y `ACEL=tpu` en el rotador, más dos bugs arreglados: el `--timeout` de `colab exec` (30 s por defecto,
que en TPU mataba el tramo antes de dar un paso) y `bloqueada()`, que no distinguía el lock propio.

---

## 4. Lo primero que hay que hacer mañana

1. **Pushear 3 commits.** Ya no republica ningún secreto vivo.
2. **Revisar si llegó el DOI** de Research Square (RSID `rs-10839567`, enviado 12:38, screening hasta
   72 h hábiles). Cuando llegue: pegar el link en el **borrador de LinkedIn que ya está guardado** con
   su imagen y su texto alternativo, y publicar.
3. **Decidir la dirección**, que es lo único que quedó abierto de verdad. Las tres líneas del día
   cerraron y **el proyecto no tiene una campaña pendiente**. Las opciones que quedan sobre la mesa:
   - Escribir el hallazgo de calibración como paper. Hay tres resultados independientes que apuntan
     al mismo lugar y dos ya están en informes.
   - Atacar la calibración de frente, que es lo que ninguna de las tres vías hizo.
   - `TELAR-03 Fase 2` (barrido d×n), que sigue pendiente de otra línea.

---

## 5. Lo que sigue esperando a Maxi

**El correo institucional.** Destraba OpenReview (y con él TMLR, ARR y TACL), arXiv, y que su Scholar
sea buscable. **La nota al Rector está redactada desde el 19-ago y sin enviar.** Es, con diferencia,
el trámite de mayor palanca del proyecto y el único que no depende de cómputo.
