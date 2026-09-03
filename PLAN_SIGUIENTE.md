# Para retomar · escrito la noche del 2-sep con todo apagado

## Estado al cierre

**Nada quedó corriendo.** Colab apagado, catorce sesiones detenidas una por cuenta, locks liberados,
rotadores y vigías muertos.

| | |
|---|---|
| **el micro-LM** | **TERMINADO**, las tres campañas cerradas y juzgadas |
| revisión de literatura | **el hueco está libre**, `REVISION_LITERATURA_VENTANA_20260902.md` |
| validación externa | **Mamba-130M real, 80 celdas de 80**, `INFORME_MODELO_REAL_20260902.md` |
| cuarto preprint | 5 páginas, **en inglés y en español**, compila limpio, citas verificadas |
| PeerJ 138627 | **retirado** por decisión propia, dentro del plazo, con agradecimiento |
| el experimento en modelo real | **bloqueado**, ver §2 |

## 1. Lo que quedó medido, y es el arco entero

Todo con kernel 3 y evaluado en la forma `directa`, donde la búsqueda tiene sensibilidad
**0,000000 exacto** a la relación.

| condición | qué ve la búsqueda | `nose_rel` |
|---|---|---|
| una sola forma | la relación **nunca** | 0,6090 · 0,5850 · 0,7349 |
| dos formas, la relación **nunca** entra | nunca | **0,5370 · 0,5526** |
| dos formas, la relación entra **a veces** | a veces | **1,0000 · 0,9625 · 1,0000** |

Y por el lado de la arquitectura, kernel 5 lo arregla y **kernel 7 no agrega nada**, con los rangos
solapados en las tres semillas. La sonda explica por qué: con alcance 6 la búsqueda ve entidad 0,8317
y relación 0,8238, o sea ve más y no acierta más.

> **La ventana decide qué se puede aprender. Una vez aprendido, se usa incluso donde la ventana no
> llega.**

## 2. LO PRIMERO DE MAÑANA · destrabar el experimento en modelo real

Está atrapado entre dos paredes, las dos **medidas**:

- con **4 hechos** de contexto la tarea **satura**, las dos condiciones dan `nose_rel` 1,0000 y no
  queda margen para medir nada. Es efecto techo, no ausencia de efecto;
- con **16 hechos** rompe el techo pero el paso cuesta **9,7 s en T4** y **279 s en esta PC**, o sea
  11,6 días las seis unidades. Inviable.

**La causa es la misma en los dos casos y tiene arreglo.** Colab **no trae** `mamba-ssm` ni
`causal-conv1d`, así que HF recorre la secuencia token por token en Python, 192 posiciones por cada
una de las 24 capas. Lo que domina el costo **no es el tamaño del modelo sino el largo de la
secuencia**.

**Qué hacer, en este orden.**

1. **Instalar los kernels en la VM** con `colab install`. Compilan desde fuente y tardan entre quince
   y treinta minutos, por eso hoy no se probó. Si andan, el salto es de diez a cincuenta veces y
   resuelve las dos paredes de una sola vez, y encima permite volver al **370m**.
2. Si no compilan en T4, **plan B**: 8 hechos como punto medio, y comprobar si ya rompe el techo.
3. Y no perder tiempo con TPU ni con L4. **La TPU es peor** porque Mamba depende de kernels CUDA que
   ahí no existen, y **L4 y G4 están fuera de cuota** en estas cuentas, verificado con el backend
   rechazando el pedido. En el tier gratuito hay T4 y TPU, nada más.

**Ojo con la disponibilidad.** El 2-sep a la tarde **no había T4 en ninguna de las catorce cuentas**,
probadas a mano una por una. Era falta de disponibilidad global, no problema nuestro. Conviene
intentar temprano.

## 3. Después, lo que le falta al preprint

- El resultado del modelo real, cuando se destrabe.
- La tercera semilla del control ciego, que quedó en 1000 pasos. **No cambia el veredicto**, porque el
  prereg pide dos y cumplen dos, pero cierra la tabla.

## 4. La regla del día, y es la más reutilizable

**Un hallazgo de ARQUITECTURA vale más que uno de ENTRENAMIENTO y hay que buscarlo primero.** Entre el
26-ago y el 1-sep hubo doce intentos de arreglar la abstención por la vía del entrenamiento, todos
negativos o parciales; el primer diagnóstico mecanicista lo resolvió en un día.

**Y el corolario, que hoy sumó tres casos más:** un criterio se escribe sobre la métrica que mide la
intervención **nueva**, nunca sobre la del resultado anterior ni sobre un número supuesto. Antes de
congelar un prereg hay que preguntarse *si la intervención funcionara perfecto, ¿esta métrica se
mueve?* Hoy fallaron por eso A-1, R-1 y el cruce entero.

## 5. Lo que NO hay que hacer

- **No leer `nose_rel` como medida de daño en una ablación.** Premia abstenerse, así que al cegar al
  modelo sube en vez de bajar.
- **No confiar en el kernel nominal.** En mamba-130m y en mamba-370m el tap más viejo vale **cero
  exacto** en todas las capas, 24 de 24 y 48 de 48, así que el alcance real es 2 y no 3. Medirlo.
- **No contar distancias en palabras.** El BPE parte los nombres y las distancias dejan de ser fijas.
  El vocabulario de `modelo_real/vocabulario.json` es todo de un token justamente por eso.

## 6. Operativo

- `/home/maxi/.venv-ligamento/bin/python` para el micro-LM, JAX.
- `/home/maxi/.venv_datasets_pandas/bin/python` para los modelos reales, torch 2.9 y transformers 4.57.
- `micro_lm/estado_todo.sh` imprime el estado de todo en un bloque listo para Telegram.
- `micro_lm/reporte_periodico.sh` manda ese bloque cada 45 minutos, y `micro_lm/avisar_0902.sh` avisa
  con el **juicio ya hecho** cuando cierra una campaña. Los dos hay que relanzarlos a mano.
- `modelo_real/rotar_real.sh <condicion> <semilla> <pasos> [cuentas]` rota cuentas hasta conseguir T4.
- `FORMAS_Q` viaja por los **dos** scripts del pipeline. Sin eso una campaña corre como copia del
  control, y hoy casi pasa.
