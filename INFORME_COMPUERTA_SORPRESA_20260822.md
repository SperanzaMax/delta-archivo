# INFORME · LA COMPUERTA DE LA POLITICA DE ESCRITURA (2026-08-22)

`DISENO_POLITICA_ESCRITURA.md` §4, script `micro_lm/compuerta_sorpresa.py`, modulo
`micro_lm/relleno.py`. Cinco checkpoints ya entrenados, 300 episodios cada uno, **sin gastar GPU**.

Pregunta bloqueante: ¿el residuo comprometido de la regla delta —`beta * ||v - S k||`, la señal de
CENTINELA-01 y la que HOLA usa para decidir escritura— **separa un hecho de un relleno**? Si no
separa, la eviction sorpresa-gated de [[vigia03-capacity-scheduling]] no tiene de donde agarrarse y
se cae antes de la primera unidad.

## Resultado, en una linea

**La sorpresa detecta lo que YA ESTA EN EL ARCHIVO. No detecta lo que no vale la pena archivar.**
Son dos cosas distintas, la idea de VIGIA-03 las trataba como una sola, y solo la primera sobrevive.

## Los numeros

Residuo en la posicion del ultimo token de cada enunciado, que es exactamente donde
`modelo.escribir` toma el vector que archiva. AUC estratificado **por posicion y por largo**, que es
el control que importa (ver §3).

| unidad | hecho | repeticion | charla | **AUC(hecho > repeticion)** | AUC(hecho > charla) |
|---|---:|---:|---:|---:|---:|
| `c1_s0` | 0,6539 | 0,4659 | 0,4959 | **0,9338** | 0,9444 |
| `c3_s0` | 0,7120 | 0,5366 | 0,4655 | **0,8329** | 0,9411 |
| `c3_s1` | 0,8541 | 0,6866 | 0,5447 | **0,7834** | 0,9645 |
| `c3_s2` | 0,8822 | 0,6016 | 0,5969 | **0,9122** | 0,9502 |
| `c4_s0` | 0,8800 | 0,6618 | 0,7942 | **0,8765** | **0,5231** |

- **`hecho > repeticion` es solido: 5 de 5, entre 0,78 y 0,93.** Es ademas el contraste **limpio**:
  las dos clases tienen el mismo largo (6 tokens) y **las mismas palabras exactas**; lo unico que
  cambia es si el archivo ya lo tiene. No hay confound de forma que explicar.
- **`hecho > charla` es fragil**: alto en cuatro unidades y **azar (0,5231) en `c4_s0`**, donde
  ademas la charla tiene media 0,7942 y desvio 0,2309 —alta y dispersa—.

## Por que la charla no sirve como control, y es el modo de falla que estaba declarado

El §4 del diseño anticipo esto textual: *«un token raro puede dar residuo alto por ser raro, no por
ser informativo, y ese seria justamente el modo en que la idea se rompe»*. Es lo que pasa. Los cinco
checkpoints fueron entrenados **sin relleno**, asi que para ellos la charla es fuera de distribucion
y su residuo alto puede venir de la novedad de la FORMA y no de la informatividad del CONTENIDO. En
`c4_s0` —el nivel con parafrasis y sesiones separadas, el de distribucion mas rica— eso alcanza para
borrar la separacion entera.

**Consecuencia de diseño:** la compuerta corrida sobre modelos que nunca vieron relleno **acota por
abajo** lo que pasaria entrenando con relleno (ahi la charla dejaria de ser novedosa), pero no puede
usarse para afirmar que la sorpresa distingue contenido de charla. La afirmacion que sobrevive es la
de redundancia, que no depende de eso.

## Dos errores propios, los dos cazados por el control y no por la lectura

**(1) El primer nulo era el nulo equivocado.** Permutaba etiquetas, o sea destruia toda la estructura
y daba 0,50 pase lo que pase. Es exactamente lo que el `INFORME_SIN_ETIQUETAS` del 20-ago dejo
asentado —«el nulo correcto NO era permutar etiquetas»— y lo escribi igual. Las dos explicaciones
alternativas reales eran concretas: el **largo** (charla 4,3 tokens contra hecho 6,0) y la
**posicion** (el estado `S` se carga y el residuo baja solo a medida que avanza la secuencia). Se
reemplazo por comparar unicamente pares con igual posicion e igual largo.

**(2) El control corregido tenia su propio agujero, y produjo un numero espectacular y falso.** Con
la charla original, el estrato de largo 6 —el UNICO donde hecho y charla se comparan— estaba cubierto
por **un solo armazon de doce** (`antes {n} no esta en {e}`). En `c4_s0` eso dio **AUC 0,1494**, o
sea dado vuelta, y no era una propiedad del modelo sino de esa frase. Con seis armazones largos
agregados, el mismo numero pasa a **0,5231** y el estrato se duplica (n 108 -> 239). **Un control
puede tener un confound propio, y este casi entra al informe como hallazgo.**

Ademas: las repeticiones se insertaban siempre al final del episodio, o sea justo en las posiciones
tardias donde el residuo baja por si solo. El confound habria fabricado el resultado esperado. Se
corrigio a insercion en posicion sorteada despues del original.

**Y el criterio de decision estaba mal escrito por mi**, con un `or` que daba «abre» si cualquiera de
los dos contrastes pasaba: `c4_s0` daba «abre» con la comparacion de charla **invertida**. Es la
cuarta vez en el programa que un criterio propio es mas laxo que los datos disponibles. Ahora decide
`hecho vs repeticion`, que es el contraste limpio, y un C-1e invertido imprime un aviso.

## Que habilita y que no

**Habilita** la campania de la politica de escritura, con la hipotesis reformulada y mas chica:
`sorpresa` deberia ganarle a `fifo` y a `azar` **cuando el episodio tiene repeticiones**, porque ahi
la señal existe y es robusta. Eso ya es la pregunta de la eviction: no gastar rango en lo que el
estado ya predice.

**No habilita** la version amplia —«el modelo sabe que vale la pena guardar»—. Para eso haria falta
que la sorpresa separara contenido de charla, y en la unidad de nivel 4 no lo hace.

**Riesgo que sigue en pie** (§5 del diseño): la sorpresa se mide **al escribir** y lo que importa es
si el hecho se va a **preguntar despues**. Sigue siendo una apuesta sobre la utilidad futura, no una
medicion de ella.

## Para retomar

1. Integrar el relleno a `idioma.py`/`datos.py` —bloqueado hasta que cierre la campania de la query
   conjunta, porque cada tramo re-sube esos archivos (§7 del diseño)—.
2. Pre-registro con la hipotesis reformulada: el eje es **redundancia**, no informatividad.
3. Re-correr esta compuerta sobre un modelo entrenado CON relleno, que es la unica forma de saber si
   la fragilidad de la charla era fuera-de-distribucion o es real.
