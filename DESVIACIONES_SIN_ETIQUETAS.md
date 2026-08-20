# Desviaciones · `PREREG_CORTE_SIN_ETIQUETAS.md` (SHA 17e0a35e…)

---

## D-1 · Una unidad se midió con el checkpoint equivocado, y la corrida se descartó entera

**Qué pasó.** La primera corrida de `sonda_sin_etiquetas.py` leyó `ckpts/c4_s2.pkl` **mientras el
tramo de Colab del otro experimento del día lo estaba sobrescribiendo**. La unidad medida fue la del
**paso 15000**, no la del 14000 que declara el §3 del prereg. Las otras siete no están afectadas: la
familia `c` no se estaba entrenando salvo `c4_s2`.

**Cómo se detectó.** Al ver que el rotador había bajado un checkpoint nuevo se comparó el `paso`
guardado adentro del `.pkl` contra el mtime del archivo y el minuto en que la sonda había muestreado
esa unidad. `paso 15000`, archivo modificado 15:04, unidad muestreada después.

**Qué se hizo.** La corrida quedó archivada como `corte_sin_etiquetas_20260820_CONTAMINADA.json` —no
se borra: es la evidencia de la desviación— y **se re-corrió completa** apuntando a
`ckpts/c4_s2.pkl.p14000`, la copia que se preservó antes de lanzar el tramo. Se re-corrieron las
ocho unidades, no sólo la afectada, para que el resultado salga de un solo muestreo y no de dos
pegados. La sonda ahora **imprime el checkpoint y su paso** por cada unidad, así el log deja escrito
que lo medido es lo declarado.

**Lo que esto vale como lección, más allá del arreglo.** Dos experimentos del mismo día compartieron
un archivo mutable sin que ninguno de los dos preregs lo mencionara. **Lo único que hizo que fuera
reparable fue la copia `p14000`**, hecha por una razón distinta (que el tramo sobrescribe el
checkpoint y hacía falta el extremo para T-2 del otro prereg). Regla para adelante: **una unidad que
entra en un análisis no puede estar entrenándose al mismo tiempo**, y si el riesgo existe, el
análisis lee de una copia congelada, no del checkpoint vivo.

**Efecto sobre los veredictos.** Ninguno en el sentido: `c4_s2` era la peor unidad por las dos vías
(U-1 no pasaba, σ>0,5 tampoco), y los criterios S-1/S-3/S-5 no se deciden por ella. Se reporta igual
porque el número publicado tiene que salir de la corrida limpia.

---

## D-2 · S-4 estaba mal calibrado, y el defecto es del criterio, no del modelo

**El criterio, escrito por mí:** «σ>0,5 falla en ≥ 6 de 8 unidades. Si no falla, el problema no
existía.» **Observado: falla en 2 de 8.**

**Por qué el criterio estaba mal.** Lo derivé de que el 19-ago σ>0,5 fallaba en las unidades que
miraba el `INFORME_UMBRAL_PROSPECTIVO`, y lo generalicé a que fallaría en casi todas. El dato real
del 19-ago es más chico: `c3_s0` fallaba con 0,1177 y las demás estaban **pegadas al borde de 0,10
por abajo** (0,0900 · 0,0937). Pedir que 6 de 8 fallen era pedir algo que los datos del día anterior
ya desmentían, y yo no fui a buscarlos antes de fijar el número.

**Control que descarta que sea un bug del instrumento:** la sonda reproduce `c3_s0` con σ>0,5 en
**0,1177 / 0,6115**, idéntico dígito por dígito al informe del 19-ago, con muestreo y semillas
independientes de aquella corrida. El muestreo replica; lo que falló fue mi predicción.

**Consecuencia, y no es cosmética:** si el criterio sin calibrar ya pasa en 6 de 8 unidades, entonces
**el piso contra el que hay que comparar U-1 no es «no hay corte» sino «σ>0,5 anda bastante bien»**, y
eso vuelve mucho más exigente —y más honesta— la pregunta del experimento. Es la tercera vez en la
campaña que un criterio escrito por mí falla por pedir de más (F-2 y F-4 el 19-ago).

---

## D-3 · El nulo de S-3 resultó ser el instrumento más informativo, y conviene decir cómo se lee

El §5 lo declaró como control: «U-1 debe pasar en ≤ 1 de 8 bajo el nulo». Lo que devolvió es más
que un control, y **no es post-hoc porque el nulo estaba escrito antes de correr**: la tasa de «pasa»
bajo el nulo **por unidad** dice cuánta compuerta se gana sin información, y resultó
**99-100/100 en las dos unidades fáciles** contra 0-12/100 en las seis difíciles.

Eso convierte el resultado principal en algo que la tabla de S-1 sola no muestra: **las dos únicas
unidades donde U-1 pasa son exactamente las dos donde pasa cualquier corte**. El nulo no se limitó a
no ensuciar el resultado, explicó el resultado.
