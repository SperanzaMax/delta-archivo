# Qué vehículo conviene · TinyLlama vs Qwen3-0.6B vs Qwen3.5-0.8B

2026-09-09. Todo lo de esta nota está **medido con `comparar_vehiculos.py`**, no leído de un paper.

## El hallazgo que cambia la pregunta

Los modelos abiertos de frontera abandonaron la atención completa en todas las capas. Qwen3-Next,
Qwen3.5, Kimi Linear y Kimi K3 corren **3 capas lineales de regla delta por cada 1 de atención
completa**. El mecanismo lineal es Gated DeltaNet, o sea regla delta con compuerta de decaimiento.

Bajado del `config.json` de `Qwen/Qwen3.5-0.8B`, y verificado cargando el modelo:

    24 capas · hidden 1024 · Apache 2.0
    atención completa SOLO en [3, 7, 11, 15, 19, 23]     18 de 24 son lineales
    linear_conv_kernel_dim = 4                            <- la conv corta que forma la query

**La frontera tiene hoy una convolución causal corta de kernel 4 adentro de un tronco de regla
delta.** Es el mecanismo de la ley de la ventana, y el kernel 4 cae entre el 3 que falla (0,6430) y
el 5 que llega al techo (0,9977).

Así que el tronco recurrente del micro-LM dejó de ser la deuda del vehículo y pasó a ser el
parecido.

## Lo medido

| | TinyLlama-1.1B | Qwen3-0.6B | Qwen3.5-0.8B |
|---|---|---|---|
| capas | 22 | 28 | 24 |
| hidden | 2048 | 1024 | 1024 |
| vocabulario | 32.000 | 151.936 | 248.320 |
| arquitectura | **densa** | **densa** | **híbrida** 18 lin + 6 full |
| conv corta | no | no | **kernel 4** |
| `model.model.layers` para el hook | ✅ 22 | ✅ 28 | ✅ 24 |
| `output_hidden_states` | ✅ 23 | ✅ 29 | ✅ 25 |
| piezas del banco de UN token | **161/161** | **161/161** | **161/161** |
| forward de 2 frases (CPU, 2 núcleos) | 0,67 s | 0,34 s | 0,67 s |
| archivo de rango 256 | 2,10 M (0,191 %) | 1,05 M (0,140 %) | 1,05 M (0,139 %) |
| licencia | Apache 2.0 | Apache 2.0 | Apache 2.0 |

## Las tres cosas que decide esto

**1. Los tres montan, y el mismo banco corre en los tres sin tocar una línea.** Las 161 piezas
(60 entidades, 100 valores, la abstención) tokenizan a UN token en los tres tokenizadores. Eso no
estaba garantizado y es lo que abarata todo: `banco_escalado.py --modelo tiny|q3|q35` y listo.
Verificado corriendo dos pasos en cada uno.

**2. El par `q35` contra `q3` es un control de arquitectura casi gratis.** Misma familia, mismo
hidden 1024, misma licencia, ~0,75 B los dos. Uno híbrido con conv corta, el otro denso con
atención completa en las 28. Es la comparación que en el micro-LM cuesta 26.000 pasos por brazo, y
acá viene entrenada.

**3. Y sólo el híbrido abre un experimento que hasta hoy no existía.** La lectura entra en la capa
2, que en `q35` es **lineal**, y la primera de atención completa es la **3**. Entonces se puede
inyectar antes de toda mezcla global (capa 2), en la mezcla misma (capa 3) o justo después (capa 4).
El hallazgo de que el contexto es PRECONDICIÓN y no corrección (0,9998 contra 0,4990) y la ley de
cobertura hacen predicciones distintas ahí. **Va pre-registrado antes de correrse**, como todo lo
demás.

## La decisión

**Hoy no se cambia de vehículo.** TinyLlama tiene el montaje del 6-sep ya validado, con los dos
controles en 0,0000 exacto, y lo que falta es el resultado del banco escalado. Cambiar ahora mezcla
dos cosas que se miden por separado.

**Cuando el banco escalado cierre, el vehículo pasa a ser `q35` con `q3` de control**, no porque
sea más nuevo sino porque es el único que tiene el mecanismo del que hablan las leyes.

TinyLlama queda como tercer brazo, que además es útil: es denso y de otra familia, así que separa
«esto pasa en los densos» de «esto pasa en Qwen».

## Dos avisos operativos

⚠️ **`flash-linear-attention` y `causal-conv1d`.** Sin esas dos librerías el camino lineal de
Qwen3.5 cae al fallback de torch, que avisa por stderr y es lento. Es la misma lección que
`mambapy` el 2-sep: medir el costo por paso ANTES de comprometer una sesión de Colab.

⚠️ **Qwen3.5-0.8B es multimodal.** El texto cuelga de `config.text_config` y hay una torre de
visión al lado que no se usa. `banco_escalado.py` ya lo contempla.
