# PRE-REGISTRO EXPLORATORIO · ¿LA QUERY CONJUNTA REABRE EL CORTE SIN ETIQUETAS? (2026-08-22)

Se escribe y se hashea **mientras la campania de la query conjunta entrena**, antes de mirar un solo
resultado de P-1..P-5. Es un analisis aparte y esta marcado EXPLORATORIO: no toca ni reemplaza al
`PREREG_QUERY_CONJUNTA.md`.

## 1. Por que esto no viola el cierre del 21-ago

Ayer el §5 del `PREREG_EMPATE_CLAVE` cerro «la linea de detectar la abstencion desde una senial
interna», con cuatro vias fallidas: logit, densidad, desacuerdo y empate de clave. Ese cierre se
apoya en un diagnostico comun, y conviene citarlo textual porque es lo que habilita esta nota: **las
cuatro separan estados del modelo, no aciertos de errores.**

Las cuatro se midieron sobre **una sola arquitectura**: la que inyecta la lectura antes del mixer y
tiene, por lo tanto, una query que es funcion pura del token. Cambiar de arquitectura no es probar una
quinta senial sobre el mismo modelo, es preguntar si el cierre era del metodo o del regimen. Tiene
precedente en el proyecto y salio bien: la gemacion se habia cerrado el 10-ago y se reabrio el 11 en
regimen eliptico sin violar su §5, porque lo prohibido era una tercera geometria, no otra
distribucion. Aca lo cerrado es la busqueda de seniales dentro de un modelo dado, y lo que cambia es
el modelo.

## 2. La prediccion, y de donde sale mecanicamente

`SMOKE_EMPATE_20260821.md` midio que la lectura del archivo es **casi uniforme**: razon top2/top1
~0,92 sobre ~6 entradas validas, en las ocho celdas. La causa es la query pura: en la posicion del
token de la relacion matchean TODAS las entradas que comparten esa relacion.

De ahi sale algo que ninguna de las cuatro vias podia aprovechar. Con una query pura, **una pregunta
cuya respuesta no esta en el archivo se ve igual que una cuya respuesta si esta**: en los dos casos
hay un puñado de entradas que matchean la relacion con scores parecidos. Es exactamente lo que el
`INFORME_MONITOR` encontro por otra cara —«el modelo nunca inventa contenido, se ancla en otra entrada
real»— y lo que dejo escrito como el requisito de una via nueva: *separar «anclado en la entrada
CORRECTA» de «anclado en cualquier entrada»*.

Con una query conjunta, la pregunta cambia de forma. Si la query codifica entidad **y** relacion, una
entrada que coincide en las dos tiene score alto y una que coincide solo en la relacion no. Entonces
**la ausencia de la respuesta se vuelve visible en la entrada**: no hay ninguna entrada con score
alto, y eso es una propiedad de la consulta, no una confianza leida de la salida.

- **A-1 · PRINCIPAL.** El AUC de `s1` (score crudo maximo de lectura en la posicion de foco)
  separando preguntas CON respuesta de preguntas SIN respuesta es **mayor en `post` que en `pre`**,
  en al menos 2 de 3 semillas, con una mejora >= 0,05.
- **A-2 · MAGNITUD.** En `post`, ese AUC llega a **>= 0,75** en al menos 2 de 3 semillas. Es el umbral
  que separa «hay senial» de «sirve»: las cuatro vias cerradas daban 0,50-0,67, y la unica que
  funciono —la cabeza supervisada del 18-ago— tiene AUC 0,77-0,99.
- **A-3 · NULO, y es lo que da el veredicto.** El mismo estadistico calculado con el score reemplazado
  por una gaussiana de igual media y desvio **no pasa** (AUC 0,45-0,55). Sin esto no hay resultado:
  el 20-ago U-1 «pasaba» 2/8 y eran exactamente las 2 celdas donde el nulo tambien pasaba, y elegir
  bien el nulo fue lo que permitio el diagnostico.
- **A-4 · ESPECIFICIDAD.** La mejora de A-1 no se explica por que `post` acierte mas: se recalcula el
  AUC **estratificando por acierto**, y la separacion CON/SIN respuesta se sostiene.

## 3. Regla de decision

Si **A-1 o A-3 fallan**, el cierre del 21-ago se mantiene tal cual y se asienta que tambien vale para
la arquitectura de query conjunta. No se prueba una sexta senial.

Si A-1 y A-3 cumplen pero **A-2 no**, hay senial nueva y no alcanza: se reporta como via candidata,
con la magnitud a la vista, y no como corte utilizable.

## 4. Lo que este analisis NO es

No es una prediccion sobre `nose` ni sobre `falsa_abst` —esas ya estan en P-3 del prereg principal y
se miden con la cabeza supervisada, que es otra cosa—. Aca lo unico que se pregunta es si existe una
senial **sin etiquetas** en la entrada. Y si existiera, seguiria faltando el paso que las cuatro vias
nunca dieron: convertir la senial en un corte que no se lleve puestas tantas respuestas buenas como
malas.
