# Pre-registro — ¿la referencia es recuperable, y bajo qué regla?

**2026-08-12.** Congelado antes de generar el dato. Responde al problema (2) del §10 de
`DISENO_BANCO_ELIPTICO.md`: *la verdad de base por recencia es una convención, no una señal
recuperable*. Sujeto: `albert:v4.0` vía Ollama, temperatura 0, el mismo del chequeo del 11-ago.

## 1. El hecho que motiva esto, y su lectura

El chequeo bloqueante dio **0,150 con azar 0,250** (m=4, d=0) y **0,150 con azar 0,125** (m=8, d=0).
Dos celdas **por debajo del azar**.

Un modelo incapaz de resolver la referencia daría el azar. Dar *menos* que el azar, de forma
consistente en dos celdas, significa que hay una regla de selección operando — sólo que no es la
nuestra. El script del 11-ago registró aciertos y no elecciones, así que la regla es invisible.

**Hipótesis lateral:** el modelo resuelve por **primera mención / centro de atención** (la
predicción de la teoría de *centering* de Grosz & Sidner: el centro preferido es el sujeto de la
mención inicial, no el referente más reciente), mientras nosotros premiamos **recencia**. Si es así,
el problema no es que la referencia sea irrecuperable: es que elegimos mal la verdad de base.

## 2. Condiciones

`m = 4` entidades activas fijo (la celda del contraste). `d ∈ {0, 5}` turnos de relleno.
20 casos por celda, semilla fija, opciones **barajadas** (la lección del 11-ago).

| condición | conversación | corrección | verdad de base |
|---|---|---|---|
| `desnuda` | 4 hechos, todos `director` (valores del mismo pool) | «No, it's X.» | recencia (**convención**) |
| `recencia` | idem | «No, the last one I mentioned — it's X.» | recencia (**explícita**) |
| `tipada` | 4 hechos de tipos con pools **disjuntos**; exactamente **uno** del tipo de X | «No, it's X.» | tipo (**objetiva**) |

`desnuda` es la replicación exacta del chequeo del 11-ago y sirve de ancla.
`recencia` y `tipada` son las dos salidas que el §10 propone, medidas una contra la otra.

## 3. Predicciones (comprometidas antes del dato)

- **P1 (replicación).** `desnuda` cae en [0,05 · 0,45], compatible con el 0,150-0,400 del 11-ago.
  Si sale fuera, el chequeo anterior no era estable y todo lo demás queda en suspenso.
- **P2 (¿sirve el marcador?).** `recencia` − `desnuda` ≥ **+0,25** absoluto. Si no sube, el modelo
  no puede rastrear el orden de mención aunque se lo pidan explícitamente, y **el marcador de
  recencia no salva al eje**.
- **P3 (¿sirve la pista objetiva?).** `tipada` ≥ **0,80**. Si no llega, ni siquiera una referencia
  objetivamente recuperable por contenido es usable a m=4, y entonces el eje `m` no es un eje de
  dificultad graduable: es un piso.
- **P4 (la lateral, y la que decide el diagnóstico).** En `desnuda`, la distribución de elecciones
  **no es uniforme** y su **modo está en la primera entidad mencionada**, con una fracción ≥ 0,40
  sobre la posición 1 de aparición. Si se cumple: el 0,150 no mide incapacidad, mide **otra regla de
  resolución**, y la verdad de base del banco tiene que ser la primera mención o quedar declarada
  como ambigua.

## 4. Qué decide cada resultado

| resultado | consecuencia para ECO |
|---|---|
| P4 se cumple | la verdad de base se redefine; el eje `m` sobrevive con la regla corregida |
| P4 no se cumple y P3 sí | la referencia es recuperable sólo por contenido; el eje `m` se construye sobre `tipada` |
| P2 y P3 fallan juntas | a m=4 nada es recuperable → el eje `m` es un piso, no un eje: **ECO pierde su aporte principal** y hay que rediseñarlo o archivarlo |

## 5. Compromiso

No se reinterpreta P4 después del dato: la fracción sobre posición 1 se lee tal cual, contra 0,25
(uniforme a m=4). Si la distribución resulta multimodal o plana, se declara **sin regla
identificada** y no se busca una tercera lectura en el mismo conjunto de datos.

Script congelado junto a este archivo: `prueba_pista.py`. Salida: `resultados_pista.json`,
con **todas las elecciones crudas** (no sólo aciertos) — es el defecto del 11-ago que esto repara.
