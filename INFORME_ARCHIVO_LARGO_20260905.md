# INFORME · el archivo largo era transferencia, y el sello aprende a descartar · 2026-09-05

Campaña `PREREG_ARCHIVO_LARGO.md` (SHA **`c769a4ef`**) con su `ENMIENDA_ARCHIVO_LARGO.md`
(SHA **`0410e957`**). Seis unidades, 2000 pasos, tres semillas × dos condiciones, sembradas desde
`kq3_s0/s1/s2`. **Los cinco criterios cumplen.**

## 1. Los números

| unidad | | paso | archivo largo (300) | archivo corto (40) |
|---|---|---:|---:|---:|
| `lg3_s0` | tratada | 2000 | **1,0000** | 0,9657 |
| `lg3_s1` | tratada | 2000 | **0,9872** | 0,9515 |
| `lg3_s2` | tratada | 2000 | **0,9933** | 0,9970 |
| `lc3_s0` | control | 2000 | 0,0104 | 0,9865 |
| `lc3_s1` | control | 2000 | 0,0333 | 0,9693 |
| `lc3_s2` | control | 2000 | 0,0000 | 0,9884 |

**L-1** (principal) 3/3 · **L-2** (bloqueante) 3/3 · **L-3** (precio) las tres caen ≤ 0,021 contra el
0,05 del criterio · **L-5** RECUP **1,0000** en las tres, contra 0,6667 del origen.

**L-4, el que decide la pregunta de fondo:**

| unidad | índice con el sello real | con turnos barajados | brecha | masa en la correcta | RECUP |
|---|---:|---:|---:|---:|---:|
| `lg3_s0` | **0,1715** | 0,7947 | **+0,6232** | 0,8003 | 1,0000 |
| `lg3_s1` | **0,2352** | 0,8064 | **+0,5713** | 0,7221 | 1,0000 |
| `lg3_s2` | **0,2599** | 0,7833 | **+0,5234** | 0,7092 | 1,0000 |

Línea de base de la mañana, sobre `kq3_s0` sin entrenar en largo: índice **0,8886** y brecha
**0,0001**. El criterio pedía índice ≤ 0,40 y brecha ≥ 0,20: **3/3, con margen**.

## 2. Lo que esto dice

> **El colapso del archivo largo era de TRANSFERENCIA, no un techo del mecanismo.** La curva de la
> mañana —1,0000 con 40 entradas, 0,3008 con 160, 0,0605 con 400— medía un modelo entrenado con 40.
> Con 2000 pasos de exposición, la misma arquitectura resuelve 161 entradas al 0,99, y el control
> entrenado en corto se queda en 0,01-0,03.

> **Y el sello de orden aprende a descartar lo viejo cuando se lo entrena para eso.** A la mañana el
> sello no filtraba por antigüedad y su control lo probaba: barajarlo movía el índice de 0,8886 a
> 0,8885. Después de la campaña, barajarlo lo mueve de 0,17 a 0,79. **Es la misma tabla `ord` y la
> misma arquitectura: lo que cambió es que vio archivos largos durante el entrenamiento.**

Eso contesta R11 —*descartar lo que ya no viene al caso*— que el 5-sep a la mañana había aparecido
como carencia del mecanismo. No era del mecanismo: era del entrenamiento.

## 3. La explicación alternativa, y hasta dónde la descarta el control

**La objeción obvia:** el índice mide masa sobre las entradas extra, y si la lectura se concentra en
una sola entrada correcta, la masa en las extra cae **por construcción**. Con `masa_correcta` en
0,71-0,80, un índice de 0,17-0,26 es aritmética, no necesariamente filtrado.

**Lo que el control contesta:** con los turnos barajados —mismas entradas, mismo contenido, mismas
proporciones (0,9641 contra 0,9628)— el mismo modelo cae a `masa_correcta` 0,21-0,24 y **RECUP de
1,0000 a 0,36-0,56**. Si la concentración viniera del contenido, barajar el sello no la tocaría.
Entonces **el modelo está condicionando la búsqueda en el turno**, y eso es lo que el índice detecta.

**Lo que el control NO separa, y hay que decirlo:** que el modelo «entienda la recencia» de que
**use el sello como parte de la dirección**. En este diseño las entradas extra llevan turnos bajos por
construcción, así que «preferir turno alto» y «descartar lo viejo» son la misma política y el
experimento no puede distinguirlas. Separarlas pide un diseño donde la respuesta correcta a veces esté
en un turno bajo — el mismo tipo de control que E-I3d usó para descartar la tabla de slots.

## 4. Lo que la campaña no dice

- **Nada sobre 3280 entradas.** Mide **161**. La curva del régimen grande sigue siendo de
  transferencia y sin medir entrenada.
- **No separa interferencia de colisión:** las sesiones extra se sortean sin restricción y traen las
  dos.
- **No toca el tope de 64 turnos de `ord`.** Con `TURNO_BASE = 24` los índices caben justo. Un archivo
  que necesite más de 64 turnos distintos sigue sin poder sellarse, y eso no lo arregla ninguna
  campaña: hay que agrandar la tabla.
- **El presupuesto real fue 2000 pasos, no 6000** (enmienda `0410e957`), y el batch efectivo se
  sostuvo con micro-lotes verificados en 1,49e−08.

## 5. Lo que sale de acá

1. **`ord` con más de 64 filas** es ahora el cuello identificado y barato de arreglar.
2. **El régimen grande (3280) entrenado**, que es la pregunta que esta campaña deja abierta con una
   razón para esperar que también sea transferencia.
3. **El control de recencia genuina** del §3, que es el que convierte «usa el sello» en «entiende la
   antigüedad».
