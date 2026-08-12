# Dos modos de falla, no uno: abstención y atribución silenciosa en las correcciones conversacionales

**Borrador corto — 2026-08-12.** Maximiliano Speranza, investigador independiente.
Material: `INFORME_PISTA_20260812.md`, `INFORME_CURVA_M2_20260812.md`, `BARRIDO_ECO_20260812.md`,
`resultados_none.json`. Repositorio público: https://github.com/SperanzaMax/delta-archivo

---

## 1. El problema

Un sistema de memoria conversacional tiene que absorber correcciones: el usuario dijo algo, más tarde
lo corrige, y el sistema debe saber cuál versión rige. Los bancos actuales (LongMemEval, Memora,
MemoryAgentBench) miden esto **suponiendo que la corrección llega identificada** — que el texto
nombra a la entidad corregida. Las correcciones reales no lo hacen: «no, es Beto» no dice de quién.

Trabajo previo propio midió el caso extremo: correcciones elípticas archivadas sin resolver la
referencia obtienen **0,0000 de recuperación en 10 de 10 semillas**, y fallan en silencio. Este
trabajo mide qué pasa un escalón antes, en el acto de **atribuir** la corrección.

## 2. Instrumento

Conversación sintética con `m` organizaciones activas, cada una con un hecho; una corrección final
que no nombra a ninguna; y la pregunta de a cuál corresponde. La respuesta es **objetivamente
determinable**: un solo hecho es del tipo del valor corregido.

**Compuerta de admisión de sujetos (aporte metodológico).** Sobre el mismo material se hacen dos
preguntas: *«¿cuál de estas entidades tiene un director mencionado?»* (extracción) y *«¿a cuál se
refiere la corrección?»* (resolución). Un modelo entra al banco sólo si su extracción ≥ 0,90. Sin esa
compuerta, no se puede distinguir «la tarea es difícil» de «el sujeto no puede leer».

De cinco modelos locales, **uno solo pasa**: cuatro de ≤2,5 B quedan en el azar en la tarea de
extracción y `qwen2.5-coder` (7 B) da 1,000. El corte es abrupto, sin zona gris.

## 3. Resultado 1: el recall cae por dos causas separables

Sujeto admitido, 20 casos × 3 semillas, con la opción de abstención ofrecida explícitamente:

| `m` | acierto | errores /60 | abstenciones /60 |
|---|---|---|---|
| 1 | 0,667 ± 0,076 | **0** | 19 |
| 4 | 0,450 ± 0,100 | 11 | 19 |
| 8 | 0,250 ± 0,087 | 17 | 25 |

- **La elipsis produce abstención, y no depende de la ambigüedad.** Con `m = 1` hay una sola candidata
  y equivocarse es imposible por construcción; el modelo aun así falla un tercio de las veces, y el
  **100 % de ese fallo es abstenerse**. La forma elíptica bloquea sola. La tasa es 31,7 / 31,7 / 41,7 %
  a lo largo del eje: plana dentro del ruido.
- **La ambigüedad produce error silencioso, y ahí sí escala:** 0 → 11 → 17.

Un banco que reporte sólo `recall` ve una curva que baja y concluye «se pone difícil». Desagregado son
dos fenómenos con causas distintas — y un sistema puede **subir** el recall reduciendo abstenciones y
**empeorar** en lo único que envenena la memoria: los hechos falsos.

## 4. Resultado 2: la política óptima se invierte a lo largo del eje

Contraste pre-registrado, única diferencia la frase que ofrece la salida:

| | Δ acierto al forzar | Δ error al forzar |
|---|---|---|
| `m = 4` | +0,250 | **+0,075** |
| `m = 8` | +0,250 | **+0,250** |

Forzar la atribución rinde **3,3 aciertos por cada hecho falso con 4 entidades activas, y 1 a 1 con 8**.
Como un error silencioso contamina todo lo que se apoye después en él, mientras que una abstención
sólo deja las cosas como estaban, **la política óptima se invierte**: con pocas entidades conviene
forzar; con muchas, permitir la abstención. El cruce está entre 4 y 8.

Dato de magnitud: **con 8 entidades y sin salida, el 45 % de las correcciones se atribuye a la
organización equivocada**, con confianza y sin señal de error.

Nota sobre la abstención: no es incertidumbre calibrada. Al quitar la salida, dos tercios de las
abstenciones previas se convierten en aciertos — el modelo *sabía* la respuesta. Su «no sé» es, en
esa proporción, falso.

## 5. Dos instrumentos invalidados, y por qué se reportan

Ambos fallos fueron detectados y corregidos el mismo día, y son transferibles:

1. **Un control de sanidad que no podía fallar.** La primera versión validaba el instrumento con
   `m = 1 → 1,000`, leído como «el modelo entiende la consigna». Pero con una sola candidata en la
   lista, responder «1» acierta **sin leer**. Con ese control vacío, la caída en `m > 1` admitía dos
   lecturas incompatibles —tarea difícil o sujeto incapaz— y se eligió la equivocada.
2. **Un parser que confundía rechazo con respuesta fuera de dominio.** Diez de once «abstenciones»
   eran el modelo contestando el nombre de la **persona** en vez del de la **organización**, porque la
   pregunta *«which one is being corrected?»* admite leerse como «qué cosa se corrige». La abstención
   debe ser una respuesta **ofrecida y registrada**, no inferida del fracaso del análisis.

## 6. Límites

Un solo sujeto admisible; un solo grado de elipsis (total) y una sola distancia; 60 casos por punto,
insuficiente para diferencias menores a ~0,15; datos sintéticos sin validación humana. La métrica de
error silencioso es una variante **a nivel de índice** de FAMA (Memora, arXiv 2604.20006), del que se
diferencia por medir sin jueces y por desagregar el error de **identidad** del de **versión**.

## 7. Qué queda

Los ejes de grado de elipsis y de distancia, la validación humana, y un segundo sujeto admisible —que
a la escala de modelos locales no existe.
