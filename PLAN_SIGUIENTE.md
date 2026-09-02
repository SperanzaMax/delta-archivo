# Para retomar · escrito la noche del 1-sep con todo cerrado

## Estado al cierre

**Nada quedó corriendo.** Las tres semillas del kernel 5 llegaron a 26000 y están juzgadas.

| | |
|---|---|
| preprint del sello de orden | **ENVIADO**, `rs-10896018`, en revisión editorial. El DOI llega cuando se postee, hasta 72 h hábiles |
| kernel 5 | **los cuatro criterios CUMPLEN**, `INFORME_KERNEL_Q5_20260901.md` |
| reparación del atractor mudo | cerrada, negativa, 0 de 6 |
| `blanco=error` a presupuesto | corrió, `vigente` 0,20-0,45, no llega al 0,60 de R-1 |
| pool de Colab | **14 cuentas**, se sumó la O el 1-sep |

## 1. LO PRIMERO, y es lo único que separa esto de poder anunciarlo

**Revisar la literatura sobre el hallazgo de la ventana.** El criterio de
`CRITERIO_DESCUBRIMIENTO.md` pide las cuatro condiciones y hoy se cumplen tres: medido con control
que podía fallar y falló (0 de 3), replicado en tres semillas, y resuelve el problema en el vehículo.
**Falta la tercera, que nadie lo haya reportado.**

Qué buscar, concretamente. Si alguien midió que **la ventana con la que se forma la query limita qué
parte de la consulta condiciona la recuperación**, en atención lineal, modelos de espacio de estados
o memoria externa. Términos por donde entrar: *query formation window*, *local convolution query*,
*short convolution* (que es como Mamba y H3 llaman a esa conv), *partial query conditioning*,
*retrieval query receptive field*. Y revisar de frente los papers de la familia delta y Mamba, que es
donde esa conv corta existe por diseño.

**Si está libre, entra directo como cuarto preprint** y es el más accionable de los cuatro, porque da
un diagnóstico barato que cualquiera puede correr sobre su propia arquitectura.

## 2. Después, la medición que hace generalizable el hallazgo

Hoy la distancia entre la relación y la posición de lectura es **fija en 3**, porque el generador
escribe siempre igual. Eso hace el efecto determinista y también acota el alcance.

**La prueba que sigue: variar la distancia.** Generar consultas donde la relación caiga a distancia
2, 3, 5, 8 y ver cómo se degrada. Predicción, la abstención en el caso difícil debería seguir a
«¿entra la relación en la ventana?» y no a la distancia en sí. Si es así, el resultado deja de
depender del generador y pasa a ser una **relación entre ventana y estructura de la consulta**, que
es lo que se puede llevar a texto real.

## 3. Y el barrido que cierra la pregunta de diseño

Kernel 5 fue el mínimo que cubre la relación. **Falta saber si más es mejor o si empeora.** Correr 7
y 9. Hay una hipótesis en contra que vale la pena tener escrita antes: una ventana más ancha mete
palabras irrelevantes en la query y podría **ensuciar** la búsqueda. Hoy hay un indicio compatible
con eso, y es que `nose_ent` bajó un poco al pasar de 3 a 5.

## 4. Lo que NO hay que hacer

- **No leer sólo `nose` global.** Esconde el intercambio entre el caso fácil y el difícil, que es
  donde está toda la información.
- **No anunciar el hallazgo antes del punto 1.** Un criterio laxo entrena a no creer los avisos.
- **No usar `b3` como familia homogénea**, mezcla preentrenadas con de-cero.
- **No intentar reparar unidades mudas.** Cerrado, 0 de 6, y el control tampoco.

## 5. Operativo

- Venv **`/home/maxi/.venv-ligamento/bin/python`**, no hay jax en el sistema.
- `medir_en_colab.sh <CUENTA> <script.py> <ckpts...>` probado, pasa los checkpoints como argumentos.
- Pool con 14 cuentas. La A dio 503 todo el 1-sep, la C y la H anduvieron siempre.
- Alta de cuenta nueva sin argumentos, `~/.colab-pool/alta.sh`, elige letra sola y saca el mail del
  login. Sigue en P.
