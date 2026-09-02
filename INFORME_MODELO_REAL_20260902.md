# La ley vale fuera de nuestro modelo: en Mamba-130M el estado ve todo y la query ve dos tokens

**2026-09-02.** Evalúa `PREREG_MODELO_REAL.md` (SHA `91684b97`), congelado antes de descargar el
modelo. Sale de una crítica que me hice hoy al proyecto entero: todo estaba medido en un vehículo de
3,5 MB y nada se había probado en un modelo que no fuera nuestro.

`state-spaces/mamba-130m-hf`, **129.135.360 parámetros**, en CPU, sin entrenar nada.

## 1. El resultado

| | criterio | resultado | |
|---|---|---|---|
| **R-2** CONTRASTE | la salida de la **capa** se mueve para **toda** distancia | **80 de 80** | **CUMPLE** |
| **R-1** PRINCIPAL | la salida de `conv1d` es 0,0 exacto fuera del alcance | 70 de 80 con el alcance **nominal**, **80 de 80** con el **medido** | ver §3 |

> **La disociación está verificada en un modelo real: el estado ve toda la secuencia y la query con
> la que ese estado se lee ve una ventana de tres tokens.**

Cambiar un solo token a distancia 5, 6, 7 u 8 de la posición de lectura **mueve la salida de la capa**
—entre $1{,}0\times10^{-2}$ y $1{,}0\times10^{-1}$— y deja la salida de `conv1d` en **0,0 exacto**. Y
`conv1d` es lo que alimenta a $B$, $C$ y $\Delta$, o sea a la query con la que el estado se consulta.

Diez posiciones de lectura, dos textos, dos idiomas.

## 2. Por qué esto no es trivial

La objeción obvia es «claro, una convolución causal de kernel 4 no ve más allá de 4, es aritmética».
Correcto, y por eso el criterio que importa es **R-2**, escrito antes: si la salida de la capa
tampoco se moviera, lo único demostrado sería que el modelo ignora tokens lejanos, que es falso.

Lo que queda demostrado es la **disociación**, y es lo que casi nadie tiene presente: en un modelo
recurrente **el estado acumula toda la historia**, pero **la query que lo lee es local**. Que el
modelo «vea» el contexto no implica que el contexto pueda **condicionar la búsqueda**.

## 3. El criterio R-1 falló, y falló por una razón que refuerza el resultado

R-1 pedía movimiento para $d \le 3$, porque el kernel es 4. Se midió movimiento en $d=1$ y $d=2$, y
**0,0 exacto en $d=3$** en las diez posiciones. La causa se buscó en vez de suponerse
(`chequeo_alineamiento_conv.py`):

> **el tap más viejo de la convolución vale CERO EXACTO en las 24 capas del modelo.**

Los otros tres no: en la capa 0 valen $9{,}3\times10^{-1}$, $1{,}4$ y $3{,}5$ de máximo absoluto. Se
verificó además por intervención, cambiando un token y viendo en qué posiciones de salida se mueve la
conv: se mueve en exactamente **tres** posiciones consecutivas, no cuatro.

**El kernel nominal es 4 y la ventana efectiva es 3, o sea alcance 2.**

**No sé la causa y no la voy a inventar.** Puede ser el entrenamiento —aunque un cero exacto en
$24 \times 1536$ pesos no sale de un gradiente— o puede ser un desalineamiento de la conversión al
formato de `transformers`. Lo que sí es sólido es el hecho, y cualquiera lo verifica en tres líneas:

```python
m = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
for c in m.backbone.layers:
    print(c.mixer.conv1d.weight[:, 0, :].abs().max(0).values)   # el primero da 0.
```

**Es el séptimo criterio de este proyecto que no se puede leer como está escrito**, y otra vez por lo
mismo: se escribió sobre un número **supuesto** —el kernel del paper— en vez de sobre el **medido**.
Con el alcance real, R-1 cumple 80 de 80.

## 4. Lo que este resultado sí autoriza a decir, y lo que no

**Sí:** que la localidad de la query no es una particularidad de nuestro micro-LM, sino una propiedad
de una arquitectura desplegada de 129M parámetros, y que convive con un estado que sí ve todo. Y que
el alcance real puede ser **más corto que el que declara la arquitectura**, cosa que conviene medir
antes de confiar en el número nominal.

**No:** que esto haga fallar a Mamba en una tarea. Es una medición de **arquitectura**, no de
**comportamiento**. El paso conductual —que una pregunta con la parte discriminante lejos del final se
responda peor— **no se midió acá** y necesita GPU. Tampoco es Gated DeltaNet, que es el que se cita en
el preprint por su kernel 4 documentado; es Mamba, la misma idea y el mismo default, y es el que se
puede correr en esta máquina.
