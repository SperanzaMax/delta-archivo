# INFORME · LA BIMODALIDAD TIENE NOMBRE, Y ES UN ATAJO CON TECHO CALCULABLE (2026-08-22)

Analisis post-hoc sobre datos que ya existian —las tres unidades `pre` de la campania de la mañana,
a 26000 pasos— sin gastar GPU y sin correr nada nuevo. Se declara como **exploratorio**: no habia
pre-registro para esto.

## El problema

Desde E-I3c el proyecto arrastra bimodalidad entre semillas y la venia tratando como ruido o como
falta de convergencia. En la campania de hoy, a igual presupuesto y con todo lo demas igual:

| unidad | acierto | `err_identidad` con relacion repetida |
|---|---:|---:|
| `p3_s0` | 0,9705 | **0,0564** |
| `p3_s1` | 0,7769 | **0,4683** |
| `p3_s2` | 0,8351 | 0,2529 |

## No es falta de presupuesto: `s1` esta plana desde el paso 8000

| `vigente` | 4k | 8k | 12k | 16k | 20k | 24k |
|---|---|---|---|---|---|---|
| `s0` | 0,623 | 0,701 | 0,806 | 0,833 | 0,956 | **0,989** |
| `s1` | 0,573 | **0,761** | 0,745 | 0,779 | 0,722 | **0,740** |
| `s2` | 0,568 | 0,675 | 0,739 | 0,749 | 0,852 | 0,832 |

`s0` sube monotona hasta 0,99. **`s1` llega a 0,761 en el paso 8000 y se queda ahi 16000 pasos mas**,
oscilando entre 0,72 y 0,78 sin direccion. Es un atractor, no una corrida lenta. Darle mas pasos no
la va a mover: es lo que la campania de presupuesto del 21-ago no podia distinguir y aca se ve porque
hay una semilla que si escapa, con el mismo presupuesto.

## Que atractor: el ATAJO DE LA RELACION, y tiene techo calculable

`diag_relacion` a 26000 pasos:

| unidad | `ac_unica` | `ac_repetida` |
|---|---:|---:|
| `p3_s0` | 0,9983 | 0,9425 |
| `p3_s1` | **1,0000** | **0,5317** |
| `p3_s2` | 0,9974 | 0,7471 |

**`s1` acierta el 100 % exacto cuando la relacion es unica en el episodio y tira una moneda cuando se
repite.** Esa es la firma de un modelo que encuentra el hecho **solo por la relacion** y nunca
aprendio a condicionar en la entidad.

Y eso permite calcular por adelantado hasta donde puede llegar sin aprender la entidad:

> **techo del atajo = (1 − P_rep) · 1,0 + P_rep · 0,5**

con `P_rep` = probabilidad de que la relacion preguntada se repita en el episodio.

| unidad | `P_rep` | techo del atajo | acierto real | |
|---|---:|---:|---:|---|
| `p3_s0` | 0,4243 | 0,7878 | **0,9746** | lo supera: aprendio la entidad |
| `p3_s1` | 0,4087 | 0,7957 | 0,8086 | **apenas por encima** — casi todo su desempeño es el atajo |
| `p3_s2` | 0,4268 | 0,7866 | 0,8906 | lo supera a medias |

**La bimodalidad deja de ser «unas semillas convergen y otras no» y pasa a ser «unas semillas escapan
del atajo y otras se quedan».** Con `s1` apenas 0,013 arriba de su techo, practicamente todo lo que
hace se explica sin suponer que use la entidad.

## Por que esto importa mas que el numero

`P_rep ≈ 0,42` **no es un parametro del experimento: es un accidente del generador.** Sale de sortear
4 hechos entre 6 relaciones sin ningun control, y nadie lo eligio. Y sin embargo **fija el techo de
lo que un modelo puede conseguir sin aprender lo que la tarea supuestamente mide**.

Es exactamente el mismo problema que E-I3d encontro con el **atajo de la recencia**: si casi todas las
preguntas son por la version vigente, el modelo aprende «devolve lo ultimo» y nunca aprende a ordenar.
Aquella vez la solucion fue hacer del balance un parametro explicito —`--p-vieja`, hoy en 0,35 y no en
0,05— y quedo escrita en el docstring de `entrenar.py`.

**Falta el analogo para el atajo de la relacion, y es la intervencion que este informe propone:**

> **`--p-colision`** — la fraccion de episodios en los que la relacion preguntada se repite. Hoy vale
> 0,42 por accidente. Subirla a 0,9 baja el techo del atajo de 0,79 a **0,55**, muy por debajo de lo
> alcanzable condicionando en la entidad (~1,0), con lo cual el gradiente deja de tener donde
> estacionarse.

## ★ LA PRUEBA DIRECTA: se sustituye la entidad y se mira si cambia la respuesta

Todo lo anterior es indirecto —surge de correlaciones entre metricas—. `sonda_atajo_relacion.py` lo
prueba de frente y **sin entrenar nada**: en los episodios donde la relacion preguntada se repite, se
arma **la misma consulta cambiando la entidad** por la otra que comparte esa relacion, y se comparan
las dos respuestas. Es una intervencion, no una correlacion.

| unidad | «misma respuesta a las dos entidades» | acierto |
|---|---:|---:|
| `p3_s0` | **0,0983** | 0,9309 |
| `p3_s1` | **0,9742** | 0,5258 |
| `p3_s2` | 0,5434 | 0,7397 |

**`s1` contesta lo mismo a las dos entidades el 97,4 % de las veces.** Las dos consultas son
literalmente indistinguibles para el modelo. `s0` contesta distinto el 90 % de las veces. `s2` queda a
mitad de camino, y eso ya dice que **no es un interruptor: es un continuo.**

Y hay una prediccion cuantitativa que sale sola: si el modelo contesta lo mismo a las dos, **solo una
de las dos respuestas puede ser correcta**, asi que

> acierto = 1 − «misma respuesta» / 2

| unidad | predicho | medido | error |
|---|---:|---:|---:|
| `p3_s0` | 0,9508 | 0,9309 | 0,0200 |
| `p3_s1` | 0,5129 | 0,5258 | 0,0129 |
| `p3_s2` | 0,7283 | 0,7397 | 0,0114 |

**Una sola variable —cuanto mira la entidad— explica el acierto de las tres semillas con error de uno
a dos puntos, sobre un rango que va de 0,53 a 0,93.** No queda margen para que la bimodalidad sea otra
cosa.

## Lo que NO dice este analisis

- Es **post-hoc y sobre tres unidades**. La sustitucion de entidad es una intervencion y por eso el
  diagnostico del atajo es solido; lo que sigue sin probar es que **`P_rep` sea la palanca**. Para eso
  hay que moverlo y ver si el techo se mueve con el, y eso todavia no se corrio.
- No dice que subir `p_colision` **funcione**. Puede pasar que con el atajo cerrado el modelo
  simplemente no aprenda nada y todas las semillas caigan, que es lo que hay que medir.
- La distribucion de entrenamiento y la de evaluacion tendrian que moverse juntas o por separado
  segun lo que se quiera medir, y eso es una decision de diseño que va en el pre-registro.

## Prediccion falsable, para cuando se corra

Si el techo es causal, entrenando con `p_colision = 0,9` y **evaluando en la distribucion de siempre**
(`P_rep ≈ 0,42`), las semillas que hoy se quedan en el atajo deberian superar 0,80, y la dispersion
entre semillas deberia **achicarse**. Si en cambio caen todas, el atajo no era un vicio sino un
escalon necesario, y eso tambien seria un resultado —del tipo de los que este proyecto ya encontro
con el curriculum de E-I3b, donde preferir-lo-ultimo y usar-el-orden resultaron dos capacidades que se
aprenden en momentos distintos—.

## Estado

Sin correr. Requiere tocar el generador (`datos.py`), que esta congelado mientras la campania del
camino lateral rota entre cuentas. Va con pre-registro propio.
