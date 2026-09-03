# Enmienda a `PREREG_DISTANCIA_REAL.md` · escrita después de la compuerta y antes de la campaña

**2026-09-03.** La compuerta (1 semilla, 400 pasos, `cerca` y `lejos`) se corrió para ver si había
margen. Dio esto:

| paso | `cerca` `nose_rel` | `lejos` `nose_rel` |
|---:|---:|---:|
| 100 | 0,9500 | 0,8000 |
| 200 | 0,9500 | 0,8000 |
| 300 | 0,9500 | 0,8000 |
| **400** | **0,9500** | **1,0000** |

`vigente` 1,0000 en las dos salvo un eval, así que **G-0 pasa** y con 4 hechos **ya no satura**: la
pared del efecto techo era el diagnóstico equivocado, no una propiedad de la tarea.

## 1. La compuerta es NO EVALUABLE, y se declara así antes de mirar nada más

`evaluar` usaba `n=8` lotes de 16, o sea **~40 ejemplos de `nose_rel`**. El error típico de una
proporción de 0,8 con n=40 es **0,063**, así que la diferencia de 0,15 son **2,4 σ** y los valores se
mueven de a **1/40 = 0,025**. Que `cerca` diera 0,9500 exacto en las cuatro evaluaciones y `lejos`
0,8000 exacto en tres es señal de eso: son 38/40 y 32/40, siempre los mismos ejemplos.

**No se lee ni como positivo ni como negativo.** Se sube a `n=32` (512 ejemplos, ~160 de `nose_rel`,
error ~0,03) y se corre la campaña.

## 2. Lo que la compuerta sí cambió, y hay que declararlo como POST-HOC

El salto de `lejos` de 0,8000 a **1,0000 en el paso 400** apunta a algo que el prereg no previó y que
la §2 del prereg **sí** anticipaba mecánicamente: con 24 capas el modelo tiene margen para pagar el
impuesto de la ventana, así que el efecto no tiene por qué ser un techo.

> **Hipótesis nueva: en un modelo profundo la ventana no decide QUÉ se puede aprender sino CUÁNTO
> CUESTA aprenderlo.**

Eso encaja con el arco del micro-LM sin contradecirlo: allá el vehículo tiene 2 bloques y no hay
capas donde pagar, así que el efecto se ve como techo; acá hay 24 y se vería como demora.

**Es post-hoc: nace de mirar la compuerta.** Por eso se agrega un criterio propio, declarado ahora y
antes de la campaña, en vez de reinterpretar G-1 después:

- **G-1v · VELOCIDAD.** Promedio de `nose_rel` sobre **todas** las evaluaciones de la corrida
  (área bajo la curva, normalizada): **`cerca` − `lejos` ≥ 0,10** en **≥2 de 3** semillas.
- **G-2v.** Lo mismo para **`lejos_dos` − `lejos` ≥ 0,10**.

G-1 y G-2 se siguen leyendo tal como están escritos, **sobre el valor final**. Si G-1 falla y G-1v
cumple, la conclusión es la hipótesis nueva y **se dice que salió de una lectura post-hoc con
criterio congelado después**, no que estaba prevista.

## 3. Cambios de montaje, todos declarados

- `--n-eval 32` (512 ejemplos por evaluación) en vez de 8.
- **800 pasos** en vez de 400, evaluando **cada 100**, para que la curva tenga forma y G-1v se pueda
  calcular sobre 8 puntos y no sobre 4.
- Tres semillas por condición, cuatro condiciones, repartidas en tres cuentas —una semilla completa
  por cuenta— para que ninguna diferencia entre cuentas caiga adentro del contraste.
- Queda disponible y **sin correr** la familia `muylejos` (d=9, donde la atenuación medida en la
  capa 1 es ~1,5 veces la de d=5). Es el plan B si con d=5 las 24 capas alcanzan a pagar el impuesto
  y las cuatro condiciones saturan. **No se corre a menos que eso pase**, y si se corre se dice que
  fue después de ver la campaña.
