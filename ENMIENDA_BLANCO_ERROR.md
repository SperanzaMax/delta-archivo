# ENMIENDA · `PREREG_BLANCO_ERROR.md` (SHA `d065838f`)

**2026-08-26, antes de lanzar una sola unidad.**

## E-A1 · Falta la tercera referencia, y baja la expectativa de E-2

**Qué pasó.** El prereg se congeló a las 14:11 UTC citando como referencia de E-2 los valores de la
cabeza actual en **dos** unidades (0,7068 y 0,8105), porque la tercera todavía estaba corriendo.
Terminó a las 14:23 y da **0,9598**.

**Por qué importa, y hay que decirlo antes y no después.** E-2 pide que el AUC suba ≥ 0,05 en ≥ 2/3.
Con las tres referencias, el margen que una **sonda lineal** recupera sobre el mismo estado es:

| | `p3_s0` | `p3_s1` | `p3_s2` |
|---|---:|---:|---:|
| cabeza actual | 0,9598 | 0,7068 | 0,8105 |
| sonda lineal sobre el mismo estado | 0,9696 | 0,7986 | 0,8590 |
| **margen disponible** | **+0,0098** | **+0,0918** | **+0,0485** |

**Sólo una de las tres supera el 0,05 con holgura, y otra está justo en el borde.** Si se toma la
sonda como techo de lo que una cabeza lineal puede extraer del estado, **la evidencia previa predice
que E-2 fallaría**, y `p3_s0` está sencillamente saturada.

**Qué NO se hace: no se afloja el criterio.** Cambiar un umbral después de ver los datos es
exactamente lo que este proyecto no hace. E-2 queda en ≥ 0,05 en ≥ 2/3.

**Qué sí se agrega, que es una lectura declarada por adelantado.** Si E-2 falla, hay dos
explicaciones distintas y no se pueden confundir:

| si además | entonces |
|---|---|
| el AUC sube pero < 0,05, y `p3_s0` está en el techo | **el estado ya tenía la información y la cabeza ya la extraía casi toda.** El blanco no era el cuello de botella |
| el AUC no sube o baja, con E-4 mostrando colapso | el blanco móvil es inentrenable así |

**Y hay una razón real por la que la cabeza podría superar el techo de la sonda**, que se escribe
ahora para que no suene a excusa después: la sonda lee un estado **fijo**, mientras que entrenar con
otro blanco **cambia también el estado**, no sólo cómo se lo lee. El techo de la sonda es un límite
superior para leer, no para aprender.

**La métrica principal sigue siendo E-1** (SER a cobertura igualada), que no tiene este problema de
saturación y es la que responde la pregunta útil.
