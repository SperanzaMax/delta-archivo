# Régimen elíptico — resultados

Prereg `PREREG_ELIPTICA.md` (SHA 299edbd8…), congelado antes de generar los textos.
N = 3000 · 10 semillas × 1000 · k = 5 · ε = 0.3 (sin re-ajustar) · margen 0.02 · IC95 t de Student, 9 gl · encoder `nomic-embed-text` en minúscula

## VIGENTE — métrica principal (§3)

| K | `g_orbita` | `hidratada_0.00` | `hidratada_0.05` | `hidratada_0.10` | `hidratada_0.20` | `hidratada_0.40` | `hidratada_1.00` |
|---|---|---|---|---|---|---|---|
| 1 | 0.9924 | 0.9998 | 0.9508 | 0.9004 | 0.7967 | 0.5907 | 0.0000 |
| 2 | 0.9819 | 0.9975 | 0.9472 | 0.8999 | 0.7993 | 0.6016 | 0.0000 |
| 4 | 0.9200 | 0.9690 | 0.9224 | 0.8774 | 0.7887 | 0.6012 | 0.0000 |
| 8 | 0.4723 | 0.5414 | 0.5368 | 0.5312 | 0.5236 | 0.4884 | 0.0000 |

## COBERTURA — secundaria

| K | `g_orbita` | `hidratada_0.00` | `hidratada_0.05` | `hidratada_0.10` | `hidratada_0.20` | `hidratada_0.40` | `hidratada_1.00` |
|---|---|---|---|---|---|---|---|
| 1 | 0.9924 | 0.9986 | 0.9497 | 0.8995 | 0.7960 | 0.5902 | 0.0000 |
| 2 | 0.9711 | 0.9949 | 0.8988 | 0.8077 | 0.6357 | 0.3541 | 0.0000 |
| 4 | 0.8617 | 0.9445 | 0.8556 | 0.7744 | 0.6208 | 0.3567 | 0.0000 |
| 8 | 0.1955 | 0.2739 | 0.2671 | 0.2615 | 0.2502 | 0.2068 | 0.0000 |

## Coseno de la entrada vigente contra la consulta (P-E3)

| K | `g_orbita` | `hidratada_0.00` | `hidratada_0.05` | `hidratada_0.10` | `hidratada_0.20` | `hidratada_0.40` | `hidratada_1.00` |
|---|---|---|---|---|---|---|---|
| 1 | 0.8176 | 0.8542 | 0.8331 | 0.8115 | 0.7672 | 0.6789 | 0.4251 |
| 2 | 0.8177 | 0.8539 | 0.8323 | 0.8116 | 0.7682 | 0.6832 | 0.4244 |
| 4 | 0.8177 | 0.8538 | 0.8323 | 0.8112 | 0.7704 | 0.6858 | 0.4236 |
| 8 | 0.8177 | 0.8536 | 0.8316 | 0.8098 | 0.7684 | 0.6848 | 0.4237 |

## Veredictos pre-registrados

**P-E0 (control BLOQUEANTE de comparabilidad)** a τ=0 y K=8, `hidratada_0` − `g_orbita` en VIGENTE = **+0.0691** IC95 [+0.0587, +0.0795] · exige ≥ 0 → **reproduce el régimen del 10-ago**

**P-E2 (control de régimen)** VIGENTE de `hidratada_1` (corrección cruda) a K=8 = 0.0000 IC95 [0.0000, 0.0000] · exige < 0,10 → **el régimen elíptico es duro, como el smoke indicaba**

**P-E1 (PRINCIPAL)** — `g_orbita` − `hidratada_τ` en VIGENTE a K=8, por τ:

| τ | dif | IC95 | ¿supera el margen? |
|---|---|---|---|
| 0.00 | -0.0691 | [-0.0795, -0.0587] | no |
| 0.05 | -0.0645 | [-0.0762, -0.0528] | no |
| 0.10 | -0.0589 | [-0.0700, -0.0478] | no |
| 0.20 | -0.0513 | [-0.0657, -0.0369] | no |
| 0.40 | -0.0161 | [-0.0309, -0.0013] | no |
| 1.00 | +0.4723 | [+0.4624, +0.4822] | **sí** |

→ τ* observado (primer punto de la grilla que supera) = **1.00** · exigido ≤ 0.25 · predicción puntual del prereg ≈ 0,075 → **P-E1 NO CONFIRMA (existe cruce pero muy tarde)**

**P-E3 (mecanicista)** pendiente del coseno de la entrada vigente por unidad de τ:
  - `hidratada_τ`: **-0.4291** (exige ≤ −0,30)
  - `g_orbita`: -0.0000 — constante en 0.8177 por construcción (exige |·| ≤ 0,01)
  → P-E3 **CUMPLE**

## Lectura (§5 del prereg)

El cruce existe pero en τ* = 1.00 > 0.25: el mecanismo sólo paga cuando la co-referencia es muy mala. **Negativo práctico**: en un sistema con hidratación razonable no vale la pena.

---

## Exploratorio — dónde cae el cruce (posterior al veredicto, no lo modifica)

P-E1 quedó **NO CONFIRMA** con la grilla congelada. Esta grilla fina se corre **después** y sólo
refina el enunciado práctico; se marca exploratoria porque se eligió habiendo visto el dato.

| τ | `hidratada_τ` | `g_orbita` − `hidratada_τ` | IC95 | supera |
|---|---|---|---|---|
| 0.40 | 0.4884 | −0.0161 | [−0.0309, −0.0013] | no |
| 0.50 | 0.4374 | **+0.0349** | [+0.0201, +0.0497] | sí |
| 0.60 | 0.3712 | +0.1011 | [+0.0860, +0.1162] | sí |
| 0.70 | 0.2888 | +0.1835 | [+0.1726, +0.1944] | sí |
| 0.80 | 0.1959 | +0.2764 | [+0.2663, +0.2865] | sí |
| 0.90 | 0.0981 | +0.3742 | [+0.3652, +0.3832] | sí |
| 0.95 | 0.0492 | +0.4231 | [+0.4145, +0.4317] | sí |

**τ\* ≈ 0,45.** Enunciado práctico: anclar geométricamente sólo conviene si la resolución de
co-referencia falla en **~45 % de las correcciones o más**. Ningún sistema razonable está ahí.

## Por qué falló la predicción puntual, y qué se aprende de eso

El prereg registró **τ\* ≈ 0,075**, del modelo lineal de cosenos. El mecanismo de cosenos salió
**exactamente como se predijo** —P-E3 CUMPLE, pendiente −0,4291 contra −0,426 predicha, `g_orbita`
plano en 0,8177— y aun así el cruce de rendimiento cayó en 0,45, **seis veces más tarde**.

La razón: **el coseno medio no es la distribución**. `hidratada_τ` no degrada uniformemente: cada
entrada queda **perfecta (0,854) o arruinada (0,42)**, nunca en el medio. La media cruza a `g_orbita`
en τ ≈ 0,08, pero el top-k se decide entrada por entrada, y a τ = 0,10 el 90 % de las consultas
todavía tiene su entrada vigente intacta. `g_orbita`, en cambio, paga su peaje en **todas**.

Hay además un efecto de segundo orden que empuja en la misma dirección: al degradarse, las entradas
de las otras revisiones **dejan de competir** por el top-k, lo que ayuda a la entrada sana. Por eso
`hidratada_0.40` da 0,4884 y no el 0,60 × 0,5414 = 0,325 que daría un modelo multiplicativo simple.

**Es la tercera vez en el programa que una media esconde su distribución** (D-012 en E3, con la
bimodalidad 2/8; la meseta falsa de E1 en la auditoría del 27-jul). Vale como regla: cuando una
predicción se deriva de un promedio, chequear la forma de la distribución antes de creerle al número.

## Qué queda establecido

1. **El cierre de la gemación se EXTIENDE, no se revierte.** La objeción de `REVISION_20260811.md`
   era correcta en su premisa —el régimen elíptico es radicalmente distinto, y el argumento «el
   encoder ya la puso donde corresponde» **es falso ahí**— pero la conclusión no cambia: la geometría
   sigue perdiendo contra hidratar el texto, en todo régimen con co-referencia mejor que una moneda.
   Ahora el cierre vale en **dos** regímenes, uno diseñado a propósito para favorecer al mecanismo.
2. **Hallazgo independiente de la gemación, y el más útil de los tres:** la corrección elíptica cruda
   es **literalmente irrecuperable — 0,0000 exacto en las 10 semillas**, no «baja». Un sistema de
   memoria que archive turnos conversacionales **sin resolver co-referencias pierde el 100 % de las
   correcciones**, y lo pierde en silencio: el índice devuelve algo, sólo que nunca lo correcto.
3. **El peaje de la gemación se replicó** en datos nuevos: +0,0691 en VIGENTE a τ=0 (P-E0), con
   coseno 0,8536 vs 0,8177 — el mismo ~0,036 del 10-ago.
