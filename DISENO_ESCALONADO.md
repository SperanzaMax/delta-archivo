# DISEÑO · ENTRENAMIENTO ESCALONADO POR CAPACIDAD (2026-08-22, noche)

Idea de Maxi: *«¿por que todo tiene que terminar a 4000? Que cada cosa termine cuando le conviene y
el resto continue hasta su turno.»*

## 1. Por que la idea encaja con lo que ya esta medido

No es una intuicion suelta: el proyecto tiene tres mediciones que la sostienen.

- **E-I3b (13-ago):** *preferir-lo-ultimo y usar-el-orden son DOS capacidades y se aprenden en
  momentos distintos.* La vigente saturaba a 3000 pasos y ANTERIOR recien despegaba a 4000, pasaba
  0,6250 a 5000 y saturaba 0,9583 a 6000.
- **Hoy, en `lat`:** `vigente` 0,861 a 4000 · `nose` 0,407 a 4000 y 0,724 a 8000 · `anterior` sin
  arrancar. Las tres van a ritmos completamente distintos.
- **Hoy, en `pre`:** `vigente` sube hasta 0,99 mientras `anterior` recien pasa 0,6 a los 4000.

Darles 26000 pasos parejos a las tres es arbitrario, y el costo es real: **la GPU es el cuello de
botella de este proyecto** (hoy mismo, 15 vueltas sin T4 en dos unidades).

## 2. La palanca ya existe, y esta clavada

`datos.lote(rng, B, ..., p_vieja=0.35, p_nose=0.0)` decide la mezcla de tipos de pregunta:

| tipo | que mide | hoy |
|---|---|---|
| `vigente` | la version que rige | ~0,39 de las preguntas |
| `anterior` | la version vieja | `p_vieja` = 0,35 de las que tienen respuesta |
| `nose_ent` / `nose_rel` | no esta en el archivo | `p_nose` = 0,40 |

**`p_vieja` no esta en 0,35 por casualidad**: E-I3d encontro que si casi todas las preguntas son por
la vigente, el modelo aprende el atajo de la recencia y nunca aprende a ordenar. Se subio a mano de
0,05 a 0,35 justamente para esto, y quedo escrito en el docstring de `entrenar.py`.

**Lo que propone Maxi es la generalizacion de esa correccion: en vez de elegir la mezcla a mano una
vez, dejar que se mueva sola.**

## 3. La forma auto-regulada, que es mejor que poner umbrales

La version obvia —«cuando `vigente` pase 0,95, bajarle el peso»— necesita umbrales elegidos a dedo, y
este proyecto lleva cuatro criterios propios mal calibrados. Hay una forma sin umbrales:

> **muestrear cada tipo con probabilidad proporcional a su error actual**, estimado con una media
> movil exponencial sobre las evaluaciones que ya se corren cada `--cada` pasos.

`p(tipo) ∝ EMA(error del tipo)`, renormalizado. Cuando `vigente` llega a 0,99, su error tiende a
cero, deja de gastar muestras solo, y el presupuesto se va a `anterior` y a `nose` **sin que nadie
decida nada**. Es exactamente «cada cosa termina cuando le conviene y el resto continua».

**Dos guardas, y las dos son necesarias:**

- **Piso por tipo** (propuesto: 0,10 cada uno). Sin piso, una capacidad resuelta deja de muestrearse
  del todo y se olvida —olvido catastrofico dentro del propio entrenamiento—. El piso la mantiene
  viva a costo bajo.
- **EMA lenta** (propuesto: `alpha = 0,1` sobre las evaluaciones). El error se mide con 512 muestras
  y tiene ruido de ±0,02; sin suavizado, el muestreo perseguiria el ruido y el sistema oscilaria.
  Ademas hay realimentacion —el muestreo cambia el error que decide el muestreo— y una EMA rapida la
  vuelve inestable.

## 4. El control que hace valido al experimento

Aca esta el punto que decide si esto se puede publicar o es una anecdota. Si `dinamico` le gana a
`fijo`, hay **dos** explicaciones y hay que separarlas:

1. el **orden** importa (que es la hipotesis de Maxi: escalonar);
2. o simplemente termino viendo **mas preguntas dificiles en promedio**, y con una mezcla fija igual
   de dificil hubiera dado lo mismo.

**Control obligatorio: `fijo_promedio`** — una condicion con la mezcla CONSTANTE, igualada al
promedio que la condicion dinamica termino usando. Se lee del log de la corrida dinamica y se corre
despues.

Es la misma jugada que `barajado` en E-I3: aquella condicion tenia exactamente los mismos parametros
y el sello sin relacion con el turno real, y fue **la celda que no gano** la que hizo valido el
resultado. Sin `fijo_promedio`, un positivo aca no dice nada sobre escalonar.

## 5. Predicciones (borrador, el pre-registro va aparte)

- **S-0 · bloqueante.** `dinamico` aprende: acierto global >= el de `fijo` menos 0,02. Si escalonar
  rompe el modelo, no hay experimento.
- **S-1 · principal.** A **igual numero de pasos**, `dinamico` > `fijo` en el acierto global, en al
  menos 2 de 3 semillas.
- **S-2 · la que le importa a la GPU.** `dinamico` alcanza el acierto final de `fijo` en **menos
  pasos**. Es la prediccion que convierte la idea en ahorro medible, y la unica que justifica el
  cambio si S-1 sale empatada.
- **S-3 · el control.** `dinamico` > `fijo_promedio`. Si empatan, lo que importaba era la proporcion
  y no el escalonamiento, y la idea se reporta como «mejor mezcla», no como «curriculum».
- **S-4 · sin olvido.** Ninguna capacidad termina **por debajo** de donde estaba cuando dejo de
  muestrearse. Es lo que el piso existe para evitar, y hay que verificar que alcance.

## 6. Lo que NO se mezcla

**Esto no se corre junto con `lat2` (la conv propia).** Son dos cambios independientes y meterlos en
la misma campania repetiria exactamente el error de `post`, que movio dos cosas a la vez y por eso su
experimento no pudo responder nada. Orden: primero `lat2` con el entrenamiento estandar —para que la
pregunta de la query conjunta se responda limpia—, despues el escalonado como mejora ortogonal, sobre
la arquitectura que haya ganado.

Si la GPU lo permite pueden correr **en paralelo como campanias separadas**, cada una con su control:
son preguntas independientes sobre bases distintas. Lo que no puede pasar es que una unidad tenga los
dos cambios.

## 7. Riesgo declarado

La realimentacion puede tener un modo de falla feo: si una capacidad es **imposible** en vez de
dificil, absorbe muestreo para siempre y degrada a las demas hasta el piso. En esta tarea no hay
capacidades imposibles —las tres se alcanzan, esta medido—, pero si el escalonado se lleva a una
tarea nueva, ese es el primer control a correr.
