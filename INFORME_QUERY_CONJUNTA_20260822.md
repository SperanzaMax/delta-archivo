# INFORME · LA QUERY CONJUNTA (2026-08-22)

`PREREG_QUERY_CONJUNTA.md` (SHA `d918fec5`) + `ENMIENDA_E1` (sin siembra, horizonte 26000) +
`DESVIACIONES_QUERY_CONJUNTA.md` (D-1). Seis unidades, nivel 3, 26000 pasos, 2048 muestras por
medicion. Las seis llegaron al paso declarado y cada checkpoint declara su propia arquitectura.

## Las cuatro predicciones FALLAN, y `post` es peor en todo

| | `pre` (control) | `post` (query conjunta) |
|---|---|---|
| acierto | 0,9705 · 0,7769 · 0,8351 | 0,3880 · 0,3979 · 0,3675 |
| `err_identidad` | 0,0122 · 0,1289 · 0,0762 | 0,2842 · 0,2783 · 0,2842 |
| `nose` | 0,9119 · 0,5416 · 0,7298 | 0,3715 · 0,3776 · 0,3896 |
| `falsa_abst` | 0,0082 · 0,0041 · 0,0353 | 0,1263 · 0,1272 · 0,1419 |
| `err_identidad` con **relacion unica** | 0,0008 · 0,0000 · 0,0026 | 0,4131 · 0,4013 · 0,4097 |

- **P-1 NO CUMPLE** — `err_identidad` no baja en ninguna semilla; sube.
- **P-2 NO CUMPLE** — 1 de 3.
- **P-3 NO CUMPLE** — `falsa_abst` pasa la compuerta de 0,10 en las tres unidades `post`, y `vigente`
  se derrumba.
- **P-4 NO CUMPLE, y es la que ordena la lectura** — con relacion **unica**, donde no hay ninguna
  colision que disolver, `pre` esta en 0,000-0,003 y `post` en **0,40-0,41**. `post` falla justo
  donde el mecanismo bajo estudio no tiene nada que hacer.

## No es presupuesto, y esta vez se verifico antes de decirlo

La explicacion que este proyecto confundio cuatro veces es la impaciencia. Aca no aplica, y las
curvas lo muestran sin ambiguedad:

| `vigente` | 4000 | 8000 | 12000 | 16000 | 20000 | 24000 |
|---|---|---|---|---|---|---|
| `pre` s0 | 0,6229 | 0,7007 | 0,8062 | 0,8330 | 0,9557 | **0,9888** |
| `post` s0 | 0,3070 | 0,2615 | 0,3399 | 0,3398 | 0,3457 | **0,3632** |
| `post` s1 | 0,2915 | 0,2856 | 0,3428 | 0,3730 | 0,3746 | 0,3412 |
| `post` s2 | 0,3262 | 0,3419 | 0,3383 | 0,3221 | 0,3664 | 0,3443 |

**`post` esta plano desde el paso 4000**: veinte mil pasos mas le dan 0,05, en las tres semillas por
igual. No es una corrida que no convergio, es un techo.

## Lo que este resultado NO autoriza a concluir, y es lo mas importante del informe

El §5 decia que si P-1 fallaba con el instrumento sano, la forma de la query no era la causa de la
colision. **Esa lectura no se puede aplicar**, y la razon estaba escrita en el §6 antes de correr:

> *«Si `post` sale peor en TODO —P-4 incluido—, la lectura correcta es perdida de computo aguas
> abajo, que es el hallazgo de E2-b, y **no** un fracaso de la query conjunta. P-4 es lo que separa
> las dos lecturas, y por eso esta escrito antes.»*

Es exactamente lo que paso. **La condicion `post` cambio dos cosas a la vez** —la forma de la query
*y* el punto donde la lectura entra al computo— y la segunda tuvo un efecto tan grande que se come
cualquier efecto de la primera. El experimento **no fue un test limpio de la hipotesis**: la
hipotesis de la query conjunta queda **SIN PROBAR**, no refutada. Decir otra cosa seria cobrar como
resultado un confound que el propio pre-registro habia anticipado.

Lo que si queda cerrado, por el §5, es **esta posicion de inyeccion**.

## El hallazgo mecanico, que es el resultado real del dia

**La ventana de inyeccion util no es «temprano»: es «antes del primer mixer».**

`post` inyecta media capa mas tarde que `pre` —despues de la conv y del delta-mixer del **mismo**
bloque 0, con 3,5 bloques de 4 todavia por delante— y con eso alcanza para llevar el acierto de
0,97 a 0,39. Para comparar: E2-b habia medido 0,9998 con el acceso en el primer bloque contra 0,4990
en el ultimo, o sea a cinco bloques de distancia. **Aca casi todo ese daño aparece con media capa.**

Esto replica [[hallazgo-contexto-precondicion]] con resolucion mucho mas fina y le pone nombre al
«computo» que importa: no es la profundidad de la red, es **el mixer**. La lectura tiene que entrar
antes de que la regla delta corra, porque lo que el modelo hace con el archivo es alimentarlo al
mecanismo de memoria, no corregir su salida.

**Y de ahi sale un trade-off estructural que no estaba visto:** una query conjunta necesita contexto
ya computado, y la lectura util necesita entrar antes de que el computo ocurra. **En esta
arquitectura las dos cosas son incompatibles por construccion.** Eso convierte el hallazgo del
21-ago —que el modelo consulta token por token— de limitacion accidental en **consecuencia
necesaria**: es la unica forma de que la lectura llegue a tiempo.

## Hallazgo lateral en el brazo de CONTROL, observacional y no pre-registrado

`p3_s0` es el mejor modelo del proyecto hasta hoy: **acierto 0,9705 · `err_identidad` 0,0122 ·
`nose` 0,9119 · `falsa_abst` 0,0082**, y sobre todo **la colision de clave practicamente
desaparecio**: con relacion repetida el error es **0,0564**, contra los 0,38-0,54 que el round-trip
midio el 20-ago. Las tres unidades `pre` pasan la compuerta de la campania de abstencion.

Difiere de las historicas en dos cosas a la vez —26000 pasos **desde cero** y `p_nose = 0,4` desde el
paso 0 en vez de curriculum de dos fases—, asi que **no se puede adjudicar a ninguna de las dos**. Es
consistente con lo que el round-trip ya anticipaba («a 20000 pasos la colision baja a 0,18-0,25; es
lo que se aprende ultimo»), y va mas lejos. Sigue habiendo bimodalidad entre semillas: 0,0564 ·
0,4683 · 0,2529, el patron conocido desde E-I3c.

**Conviene decirlo sin adornos: `err_identidad`, que era el error dominante y el que impedia afirmar
que el modelo no alucina, esta en 0,0122 en una semilla de tres.** No por la query conjunta, sino en
el brazo que existia para servir de control.

## La salida limpia, para el proximo experimento

Separar los dos factores que `post` mezclo: **mantener la inyeccion donde funciona (antes del mixer)
y darle contexto a la query por un camino lateral** —por ejemplo formandola sobre `conv3(ln1(h))` en
vez de `ln1(h)`—. La conv de kernel 3 aporta las dos posiciones anteriores sin pasar por el mixer, y
en el idioma del micro-LM la entidad y la relacion caen a distancia 2 en la forma canonica
(`el director de museo es X`). Eso probaria la query conjunta **sin mover el punto de inyeccion**,
que es lo que este experimento no logro separar.

No es una tercera posicion de inyeccion —es la misma— asi que no cae bajo la prohibicion del §5. Va
con su propio pre-registro y con la compuerta de instrumento que ya existe
(`chequeo_query_conjunta.py`, que mide si la query depende del contexto).
