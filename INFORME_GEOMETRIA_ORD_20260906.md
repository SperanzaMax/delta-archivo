# INFORME · la bandera está en los pesos, y `ord` aprendió DOS cosas en la misma tabla · 2026-09-06

Pre-registro `PREREG_GEOMETRIA_ORD.md` (SHA **`bedac0b5`**), instrumento `micro_lm/geometria_ord.py`.
Siete checkpoints, en CPU, segundos. Es la contraparte de peso del resultado conductual de esta
mañana (`INFORME_RELOJ_O_BANDERA_20260906.md`, prereg `aabb4b20`).

## 1. Los números

| unidad | cos dentro 0-23 | cos dentro 24-63 | cos entre | brecha G-1 | r(cos,dist) G-2 | ‖n‖ bajo | ‖n‖ alto |
|---|---:|---:|---:|---:|---:|---:|---:|
| `lg3_s0` | **0,8584** | 0,0372 | −0,1915 | **0,2287** | 0,0324 | **3,1872** | 1,4788 |
| `lg3_s1` | **0,8983** | 0,0445 | −0,2031 | **0,2477** | −0,0447 | **3,5244** | 1,5596 |
| `lg3_s2` | **0,8806** | 0,0353 | −0,1806 | **0,2159** | −0,0121 | **3,6548** | 1,4990 |
| `lc3_s0` | −0,0115 | −0,0002 | −0,0028 | −0,0087 | −0,0125 | 1,2274 | 0,9867 |
| `lc3_s1` | −0,0062 | 0,0051 | 0,0002 | −0,0064 | 0,0006 | 1,1318 | 0,9965 |
| `lc3_s2` | 0,0029 | 0,0029 | −0,0008 | 0,0037 | −0,0249 | 1,2445 | 0,9880 |
| `kq3_s0` | −0,0107 | −0,0002 | −0,0028 | −0,0078 | −0,0137 | 1,1939 | 1,0033 |

## 2. El veredicto

- **G-1 CUMPLE.** Brecha 0,2159-0,2477 en las tres, contra el 0,15 pedido.
- **G-2 CUMPLE.** |r| ≤ 0,045 en las tres: **cero estructura de orden dentro del bloque bajo**.
- **G-3 CUMPLE.** Las tres `lg3` contra las tres `lc3` y el origen `kq3_s0`, sin solape: la
  partición **se formó al ver archivos largos**, no venía en la siembra.
- **G-4 no se activa.** La rampa contra el índice da 0,55-0,57, por debajo del 0,80 — y el control
  del §3 muestra que ni siquiera es una rampa.

> **El modelo no aprendió 64 marcas de turno. Aprendió UNA marca de «ajeno» copiada en las 24 filas
> del bloque bajo** —coseno 0,86-0,90 entre ellas, mediana 0,946 contra su propia dirección media—
> **con más del doble de norma que las del episodio (3,19-3,65 contra 1,48-1,56) y apuntando en
> dirección opuesta.** Eso es literalmente una bandera: un bit, no un reloj.

Y explica sin residuo los tres resultados conductuales de la mañana: apilar todas las ajenas en un
turno no cambia nada **porque las 24 filas ya eran el mismo vector**; correr el episodio dos lugares
lo rompe **porque cae sobre filas que llevan la marca de ajeno con el doble de norma**; e invertir
los turnos manda la masa arriba **porque la marca está en las filas, no en el orden**.

## 3. La explicación alternativa, y lo que apareció al descartarla

**La objeción:** la primera componente correlaciona 0,55 con el índice de turno. ¿No será una rampa,
o sea orden continuo, y no un escalón?

| unidad | r(PC1, índice) | r(PC1, escalón) | r dentro de 0-23 | r dentro de 24-63 |
|---|---:|---:|---:|---:|
| `lg3_s0` | 0,5489 | **0,8587** | −0,0255 | **−0,6787** |
| `lg3_s1` | 0,5503 | **0,8639** | −0,0553 | **−0,6999** |
| `lg3_s2` | 0,5736 | **0,8788** | −0,2321 | **−0,6875** |

Contra el escalón correlaciona 0,86-0,88 y contra el índice 0,55: **es un escalón**, y la
correlación con el índice era su sombra.

**Pero el control trajo algo que el pre-registro no pedía, y hay que decirlo:** dentro del bloque
del episodio la correlación es **−0,68 a −0,70**. Ahí sí hay gradiente. O sea:

> **`ord` aprendió DOS cosas distintas en la misma tabla.** Un **bit de pertenencia** —el escalón
> entre bloques, con el bloque ajeno colapsado a una sola dirección de norma doble— y un
> **gradiente de recencia dentro del episodio**, que es el sello de orden original, el que fue de
> 0,4570 a 0,9956 y tiene DOI `rs-10896018`.

Las dos funciones conviven entrelazadas en una tabla de índices absolutos. Por eso el tope de 64
filas no rompe una cosa sino las dos a la vez.

Esto **no contradice** la lectura de la mañana: la refina. C-2 mostró que el orden entre las viejas
no se usa (y acá el bloque bajo da r ≈ 0), mientras el orden **dentro** del episodio sigue vivo (y
acá da −0,69).

## 4. Qué diseño sale de acá

El reemplazo ya no es una sola cosa. Lo que corresponde es **separar las dos funciones que hoy
comparten la tabla**:

1. **El bit de pertenencia, calculado y no memorizado**: si la entrada es de la conversación en
   curso o de otra. Hoy el modelo lo infiere de en qué fila cayó, y por eso no sobrevive a que las
   filas cambien de rango.
2. **El sello de orden, relativo a la consulta**: `turno_actual − turno_entrada` en vez de
   `turno_entrada`. Preserva el gradiente que ya funciona (−0,69) y lo vuelve invariante al valor
   absoluto — con lo que el tope de 64 pasa a ser una **ventana de recencia** de 64 turnos hacia
   atrás en vez de un techo duro del archivo.

Predicción falsable, para el pre-registro siguiente: entrenado así, `reloj_o_bandera.py` tiene que
dar `corrido` ≈ `real` (hoy la RECUP cae de 1,0000 a 0,64-0,78 y el acierto a 0,02-0,08), y la
geometría tiene que perder el escalón conservando el gradiente.

## 5. Límites

- Son pesos de tres corridas de una campaña; nada causal. La causa se prueba entrenando con el
  sello relativo y volviendo a correr las dos mediciones.
- El gradiente intra-episodio (−0,69) es un hallazgo **no pre-registrado**, encontrado al correr el
  control de G-4. Queda declarado como tal y necesita su propia confirmación.
- Nada sobre turnos > 63: no existen en la tabla.
