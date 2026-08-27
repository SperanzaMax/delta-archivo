# INFORME · A5, `--blanco error` — la vía se cierra, y deja un resultado sobre VARIANZA

Evalúa `PREREG_BLANCO_ERROR.md` (SHA `d065838f`) + enmienda (`07191f7f`), con sus desviaciones en
`DESVIACIONES_BLANCO_ERROR.md`. Tres unidades `b3_s0/s1/s2` a **26000 pasos**, control pareado
`p3_s0/s1/s2` a 26000, todo a presupuesto igualado.

Instrumentos: `ser_cobertura.py` (n=4000, semilla de datos 54321) para E-0/E-1/E-3, y `a5_e2_e4.py`
(n=4000) para E-2/E-4. Los dos leen `donde` y la regla de decisión **del checkpoint**.

---

## 1. Las predicciones, contra lo medido

| | criterio | s0 | s1 | s2 | veredicto |
|---|---|---|---|---|---|
| **E-0** bloqueante | `vigente` ≥ 0,70 en ≥2/3 | 0,9996 ✓ | 0,9979 ✓ | 0,6762 ✗ | **CUMPLE 2/3** |
| **E-1** PRINCIPAL | Δ SER @0,70 ≤ −0,02 en ≥2/3 | −0,0137 ✗ | **−0,1708** ✓ | +0,0653 ✗ | **NO CUMPLE 1/3** |
| **E-2** mecanicista | AUC sube ≥0,05 en ≥2/3 | +0,0428 ✗ | **+0,2901** ✓ | +0,0151 ✗ | **NO CUMPLE 1/3** |
| **E-3** no-daño | `vigente` no cae >0,05 | ✓ | ✓ | −0,1644 ✗ | cumple 2/3 |
| **E-4** riesgo | colapso al prior | no | no | no | **sin colapso** |

**Celda del §6: «ninguna, sin colapso → el blanco no era el problema y la vía se cierra».** Estaba
escrita antes de correr y se aplica sin discusión.

---

## 2. Lo que la campaña SÍ encontró, y no es lo que fue a buscar

**El blanco `error` produce una cabeza que sabe cuándo el modelo va a equivocarse.** El AUC del logit
sobre «¿me voy a equivocar si contesto?» da **1,0000 · 0,9998 · 0,8132**. En dos de tres semillas es
prácticamente perfecto.

**Lo que falla es que esa información no se convierte en menos error.** E-1 mide justamente eso —SER
a cobertura igualada— y da 1/3. La cabeza sabe y el sistema no lo aprovecha.

Es, otra vez, **calibración y no capacidad**. Es la misma pared contra la que chocaron el trípode
(AUC del logit 0,777-0,998 con techo de calibración) y hoy la Fase 0 de la relación
(`INFORME_RELACION_FASE0_20260827.md`). **Tres vías distintas, la misma pared.**

---

## 3. El resultado de fondo es sobre la VARIANZA, no sobre la media

Es lo que más importa de esta campaña y hay que decirlo así.

| | s0 | s1 | s2 |
|---|---:|---:|---:|
| Δ SER @0,70 | −0,0137 | **−0,1708** | **+0,0653** |
| `acierto` tratamiento | 0,9996 | 0,9979 | **0,6762** |
| `acierto` control | 0,9689 | **0,7918** | 0,8406 |

Promediar las tres daría ≈ −0,04 y un informe que dice «mejora leve». **Eso sería falso.** Lo que
pasó es que la condición **aumentó la dispersión entre semillas**:

- En **s1** el tratamiento es casi perfecto (SER 0,0008) y **el que se rompe es el control** (SER
  0,3090, `nose` 0,5382). La ventaja de −0,1708 se explica tanto por el fallo del control como por el
  acierto del tratamiento.
- En **s2** el tratamiento se rompe él, y de una forma específica: **le reaparece `err_identidad`**
  (0,0425 · 0,0825 · 0,1215 según la cobertura), que `lat2` había puesto en **0,0000**. La condición
  deshizo, en esa semilla, algo que la base ya tenía resuelto.

**`--blanco error` no es mejor ni peor que su control. Es más inestable.** Y una condición con esa
varianza no se adopta aunque su mejor semilla sea espectacular.

---

## 4. Un defecto del propio pre-registro, declarado

**E-2 era matemáticamente imposible de cumplir en s0.** El §5 fijó «sube ≥ 0,05» citando referencias
del control de **0,7068 y 0,8105**. Pero el `p3_s0` real a 26000 llega a **0,9572**, así que el máximo
alcanzable era **+0,0428**: cumplir habría exigido un AUC de 1,0072.

El tratamiento llegó a **1,0000**, o sea al techo, y aun así E-2 «falla» en esa semilla.

**No se usa para rescatar nada** —E-2 falla igual en s2, donde había 0,2019 de margen y sólo subió
0,0151— pero queda anotado para el próximo pre-registro: **un umbral de mejora absoluta necesita
verificar antes cuánto espacio queda hasta el techo**, y las referencias tienen que ser de la unidad
que se va a usar como control, no de otras.

---

## 5. Lo que se cierra y lo que no

**Se cierra:** el blanco móvil `error` como vía para reducir el error silencioso. No se prueba una
variante, no se congela el blanco, no se corre en `--idioma 3`. El §6 lo dice y el §7 fija el
abandono.

**No se cierra, y es lo que queda vivo:** que una cabeza binaria pueda ordenar el eje «me voy a
equivocar» con AUC 1,0000. Eso es información real y medida. El problema es el paso siguiente —
convertirla en decisión— y ése ya tiene nombre en el proyecto desde el trípode.

**Y queda una observación post-hoc, declarada como tal**, para quien escriba la próxima métrica: a
cobertura 0,70 el SER del tratamiento está dominado por `invento` (0,1055 en s0 y s1) con
`err_version` en 0,0008 y `err_identidad` en 0,0000. Forzar cobertura 0,70 cuando sólo el ~60 % de las
preguntas tiene respuesta obliga a fabricar en el 10 % restante, y ese piso se le suma **igual a las
dos condiciones**, comprimiendo cualquier diferencia. La métrica del §4 tenía un piso que nadie
previó al escribirla.
