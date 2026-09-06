# PRE-REGISTRO · el sello de orden, ¿es un RELOJ o es una BANDERA? · 2026-09-06

Se congela ANTES de correr nada. Mediciones sobre checkpoints YA entrenados (`lg3_s0/s1/s2`,
campaña `PREREG_ARCHIVO_LARGO` SHA `c769a4ef`), en CPU, sin entrenar y sin gastar Colab.

## 1. Por qué

El 5-sep a la noche la campaña del archivo largo cerró con los cinco criterios, y el resultado más
grande fue que **el sello de orden aprende a descartar lo viejo**: índice de masa 0,17-0,26 con el
sello real contra 0,78-0,81 barajado, RECUP 1,0000 en las tres semillas.

El §3 de `INFORME_ARCHIVO_LARGO_20260905.md` declaró lo que ese control **no** separa: que el modelo
«entienda la recencia» de que **use el sello como parte de la dirección de búsqueda**. En el diseño
las entradas extra llevan turnos por debajo de `TURNO_BASE = 24` por construcción, así que
«preferir turno alto» y «descartar lo viejo» son la misma política.

Hay una segunda razón, y es la que le da urgencia. El punto (a) de lo que quedó abierto es que
`ord` tiene 64 filas y un archivo más largo no se puede sellar. **Agrandar la tabla sólo sirve si el
sello es un reloj.** Si lo que aprendió es una frontera absoluta en 24, o una dirección «andá al
turno más alto», entonces la tabla no es el cuello y agrandarla no arregla nada: el trabajo hay que
hacerlo en la política, no en el tamaño.

## 2. Las dos hipótesis

- **RELOJ.** El modelo condiciona la búsqueda en el **orden relativo** de los turnos. Desplazar todos
  los turnos manteniendo el orden no cambia nada, y el orden **entre** las entradas viejas también se
  usa.
- **BANDERA.** El modelo aprendió una **partición**: turno por encima de cierto valor = del episodio,
  por debajo = ajeno. El valor absoluto importa, y el orden entre las viejas es indistinto.

## 3. Diseño

Cinco condiciones sobre los **mismos lotes**: el archivo se escribe una sola vez por lote (el forward
de escritura no depende de los turnos) y lo único que cambia es el vector `turnos`. Es un diseño
pareado exacto: mismo contenido, misma pregunta, misma entrada correcta.

| condición | turnos del episodio | turnos de las extra | qué introduce |
|---|---|---|---|
| `real` | 24..63 | U(0, 23) | el régimen de entrenamiento — réplica de L-4 |
| `barajado` | turnos mezclados dentro de la muestra | | el control ya publicado |
| `corrido` | 12..51 | U(0, 11) | **orden relativo intacto, frontera movida de 24 a 12** |
| `comprimido` | 24..63 | todas en 23 | **sin orden entre las viejas**, sólo pertenencia al bloque |
| `invertido` | 0..39 | U(40, 63) | **la entrada correcta en turno BAJO** — la recencia genuina |

Métricas por condición, las de `masa_turnos.py`: `indice` (masa en las extra sobre su proporción),
`masa_correcta`, `RECUP` (el argmax de la lectura es la entrada que contesta) y `vigente`
(la respuesta del modelo, que es lo que al final importa).

**Desagregación que decide C-1:** bajo `corrido` los turnos del episodio caen en 12..51, así que la
mitad baja del episodio (12..23) queda **por debajo del umbral 24**. Se reporta RECUP separando las
muestras según el turno de su entrada correcta caiga por debajo o por encima de 24.

3 semillas de checkpoint × 5 condiciones. Lotes y batch fijados en 4 × 16 = 64 muestras por celda,
declarado acá para que no se elija mirando.

## 4. Criterios, comprometidos por adelantado

- **C-0 (sanidad, bloqueante).** `real` replica L-4: índice dentro de [0,12; 0,31] y RECUP ≥ 0,97 en
  las 3 semillas. Si no replica, la medición no adjudica nada y se para.
- **C-1 (frontera absoluta).** Bajo `corrido`, la brecha de RECUP entre las muestras con la entrada
  correcta en turno < 24 y las de turno ≥ 24:
  - **≥ 0,15 en las 3 semillas → BANDERA** (hay umbral absoluto, y cae del lado equivocado).
  - **≤ 0,05 en las 3 semillas → RELOJ** (el orden relativo alcanza).
  - en el medio, o discordante entre semillas, **no adjudica** y se reporta como tal.
- **C-2 (resolución entre las viejas).** `comprimido` contra `real`: si |Δíndice| ≤ 0,05 y
  |ΔRECUP| ≤ 0,03 en las 3 semillas, **el orden entre las entradas viejas NO se usa** — evidencia de
  bandera, independiente de C-1.
- **C-3 (dirección).** Bajo `invertido`, si el índice sube por encima de 0,80 en las 3 semillas
  (o sea: la masa se va a las entradas de turno alto, que ahora son las ajenas), la política es
  **«preferir turno alto»** y no «descartar lo que no viene al caso».
- **C-4 (el que podría dar vuelta todo).** Si bajo `invertido` RECUP se mantiene ≥ 0,90 en las 3
  semillas, el sello **no** está gobernando la recuperación en este régimen y la lectura del 5-sep
  necesita una revisión mayor, porque el control barajado estaría midiendo otra cosa.

## 5. Qué se hace con cada resultado

- **BANDERA (C-1 y/o C-2 y C-3):** agrandar `ord` deja de ser el próximo paso. Lo que sigue es un
  sello que no dependa del valor absoluto — codificación relativa a la consulta, o entrenamiento con
  la frontera sorteada por muestra — y eso va con su propio pre-registro. El resultado del 5-sep no
  se cae: el modelo **sí** aprende a descartar lo viejo en su régimen; lo que se acota es **cómo**.
- **RELOJ:** el tope de 64 filas es efectivamente el cuello, agrandar la tabla es el arreglo, y el
  paso siguiente es medir hasta dónde extrapola a turnos nunca vistos.
- **No adjudica:** se reporta como tal y no se elige la lectura conveniente. Ver
  [[regla-verificar-antes-de-veredicto]].

## 6. Lo que esta medición NO puede decir

- Nada sobre 3280 entradas: mide 300 casilleros, ~161 entradas escritas.
- Nada sobre modelos entrenados de otra forma: son tres checkpoints de la misma campaña.
- No mide extrapolación a turnos > 63, que sigue siendo NaN por `mode="fill"` y es otro experimento.
