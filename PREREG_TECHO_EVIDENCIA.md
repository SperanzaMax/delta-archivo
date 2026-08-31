# PRE-REGISTRO · ¿el 0,70 es el techo del ESTADO o el techo del LECTOR?

**2026-08-31, noche.** Se congela **antes de correr**. CPU, checkpoints ya en disco, cero GPU: corre
mientras las cuatro unidades del slot ocupan las cuentas.

---

## 1. La pregunta, y por qué es la que sigue

El informe de hoy (`INFORME_ORDEN_NOSE_20260831.md` §5.3) dejó el resultado más fuerte de la línea:

> **El techo está en la EVIDENCIA, no en la decisión.** Con la ausencia decodificable a **0,70** y el
> término de orden llegando a **0,66**, no queda margen del lado de la pérdida.

Y su propio §6 declaró el límite: *«No prueba que 0,70 sea el techo real. Es el techo de un lector
**lineal** sobre el estado final de **este** modelo. Una sonda no lineal, o una capa distinta,
podrían dar más.»*

**Esa frase decide la estrategia de toda la línea**, así que se mide antes de construir nada encima.
Si el techo real es 0,85, el cierre del lado de la pérdida está mal dado y hay margen. Si es 0,70 en
toda variante, el diagnóstico se endurece y **la única salida es mejorar la RECUPERACIÓN**.

**Precisión que hay que hacer antes, porque el informe la dijo de más:** `sonda_ausencia_lineal.py`
lee **los LOGITS** (242 features), no «el estado interno». No es lo mismo, y de ahí sale la primera
predicción, que es aritmética y no empírica.

## 2. Los cinco lectores, y qué distingue a cada uno

| | lector | sobre qué | qué agrega |
|---|---|---|---|
| **L1** | ridge lineal | logits (242) | **réplica** del 0,7003 |
| **L2** | ridge lineal | estado final `hn` (128) | el estado ANTES de la proyección de salida |
| **L3** | ridge sobre **random features** (proyección aleatoria + `tanh`, 1024) | `hn` | **NO LINEAL, y sin optimizador**: solución cerrada, nada que converger |
| **L4** | ridge lineal | salida de **cada bloque** (4) | ¿la ausencia vive en una capa intermedia y se pierde después? |
| **L5** | ridge lineal | resumen de la **búsqueda** (`s_max`, brecha top-2, entropía, masa leída) | la evidencia CRUDA, antes de todo el cómputo |

Controles obligatorios en todos, los mismos de `sonda_ausencia_lineal.py`: **NULO** con etiquetas
barajadas (tiene que dar ~0,50) y **TECHO** con «¿el argmax es un nombre?» (tiene que dar ~1,00). Si
un lector falla sus controles, **no se lee su número**.

Dos modelos, los mismos del informe: **`n3_s0`** (base sana, RECUP 0,7885) y **`t03_s3`** (degradada,
RECUP 0,36). Held-out, mitad y mitad, `λ` barrido en cuatro órdenes.

## 3. Predicciones, fijadas ANTES

**T-0 · CONTROL ARITMÉTICO, y va primero porque puede invalidar la comparación.** `logits = hn·W + b`
con `W` de 128×242, así que **si `W` tiene rango 128 el espacio de funciones lineales sobre los
logits es EL MISMO que sobre `hn`**, y L1 y L2 tienen que coincidir dentro de ±0,02. Se mide el rango
de `head.w`. **Si difieren más que eso con rango completo, hay un bug y no se lee nada más.**

**T-1 · PRINCIPAL.** Ningún lector supera **0,73** (el 0,7003 más 0,03). Es decir: **el 0,70 es una
propiedad del ESTADO y no del lector**, y el cierre del informe de hoy queda firme.

**T-2 · LA QUE PUEDE DAR VUELTA LA ESTRATEGIA.** Si **algún** lector supera **0,75** en `n3_s0`, el
0,70 era del LECTOR: hay información de ausencia que el modelo tiene y no usa, y **«no queda margen
del lado de la pérdida» hay que retirarlo**. En ese caso el objetivo pasa a ser un mecanismo que lea
esa información, no que recupere mejor.

**T-3 · MECANICISTA.** L5 (la búsqueda cruda) queda **≤ 0,55**. Ya hay tres mediciones compatibles de
esta tarde: `s_max` 0,5115, sonda sobre el vector completo 0,5065, nulo 0,4803. Si L5 diera alto, la
señal estaría en la búsqueda desde el principio y todo el diagnóstico del día cambia de lugar.

**T-4 · ESCALA CON LA RECUPERACIÓN.** El techo de `n3_s0` (RECUP 0,7885) supera al de `t03_s3`
(RECUP 0,36) en **≥ 0,08**. Es la predicción que sostiene «para saber que algo no está, primero hay
que buscarlo bien»; si los dos techos fueran iguales, esa frase se cae.

## 4. Cómo se lee cada desenlace, escrito ANTES

| desenlace | lectura | qué se hace |
|---|---|---|
| **T-1 cumple** (nada pasa 0,73) | el 0,70 es del estado | el cierre del informe queda firme; **toda la línea pasa a la RECUPERACIÓN** |
| **T-2 se dispara** (algo pasa 0,75) | el 0,70 era del lector | se retira el cierre y se diseña el lector, no el recuperador |
| **sube sólo L3** (no lineal) | la señal está pero es no lineal | la cabeza de abstención necesita profundidad, no otra pérdida |
| **sube sólo L4** (capa intermedia) | la ausencia se pierde aguas abajo | leer la abstención de esa capa, no del estado final |
| **T-4 falla** | el techo no escala con RECUP | se cae el argumento del §5.4 de hoy y hay que reescribirlo |

## 5. Lo que NO contesta

- **No mide un modelo entrenado para esto.** Son sondas sobre checkpoints existentes; un modelo
  entrenado con el blanco adecuado puede tener más (precedente medido: el blanco `error` da 0,65
  post-hoc y **1,0000 entrenado**).
- **No dice cómo usar la señal**, sólo si está.
- **Dos modelos y una tarea.** No dice nada de otros niveles ni de archivos grandes.
