# PRE-REGISTRO · LA POLITICA DE ESCRITURA (2026-08-22, tarde · sin correr)

Se escribe con la campania del camino lateral entrenando. **Todavia no se puede lanzar**: falta
integrar `relleno.py` al generador, y los cinco archivos del generador estan congelados mientras haya
una campania rotando entre cuentas (§7 de `DISENO_POLITICA_ESCRITURA.md`). Queda hasheado y listo.

Es el frente de «que no olvide nunca lo que le dije», y la ultima idea grande del brazo interno que
nunca se corrio: la **eviction sorpresa-gated** de [[vigia03-capacity-scheduling]].

## 1. La hipotesis, ya corregida por su compuerta

La formulacion original de VIGIA-03 era «echar primero lo de baja sorpresa, porque no vale la pena
gastar rango en lo predecible». La compuerta de esta mañana
(`INFORME_COMPUERTA_SORPRESA_20260822.md`) la acota, y conviene arrancar de la version acotada:

- **`hecho > repeticion` es solido: 0,78-0,93 en 5 de 5 checkpoints**, y es el contraste limpio
  (mismo largo, las mismas palabras, y lo unico que cambia es si el archivo ya lo tiene).
- **`hecho > charla` es fragil**: alto en cuatro unidades y **azar (0,5231) en `c4_s0`**.

**Frase que ordena el diseño: la sorpresa detecta lo que YA ESTA EN EL ARCHIVO, no lo que no vale la
pena archivar.** Asi que la hipotesis que se prueba es la de **redundancia**, no la de
informatividad. Es mas chica que la original y es la que tiene señal medida detras.

## 2. Diseño

**Presion de capacidad.** El archivo se acota a `N_ARCH = 12` entradas cuando el episodio produce 24
enunciados (8 hechos + 8 repeticiones/parafrasis + 8 charlas). Hoy no hay cota y por eso la pregunta
«que guardar» no existe: entra todo.

**Cuatro politicas, a IGUAL presupuesto de entradas:**

| | que retiene |
|---|---|
| `todo` (techo) | sin cota — no es una politica, es el limite superior contra el que se mide el costo de tener que elegir |
| `fifo` | las ultimas `N`, o sea recencia pura |
| `azar` | `N` al azar — el piso que cualquier politica tiene que superar para existir |
| `sorpresa` | las `N` de mayor residuo comprometido `beta·‖v − S k‖` al momento de escribir |

`azar` no es decorativo: es el control que hace que el experimento pueda perder. Sin el, `sorpresa`
le gana a `fifo` y no se sabria si es por elegir bien o por no elegir por recencia.

**Config**, la de la campania de hoy: nivel 3, `d=128`, `capas=4`, `batch=64`, `lr=1e-3`,
`p_nose=0.4`, `--abst cabeza`, 26000 pasos con horizonte 26000, sin siembra, 3 semillas.

## 3. Predicciones

- **E-0 · BLOQUEANTE, primero.** Con el archivo acotado, `todo` sigue aprendiendo la tarea
  (acierto >= 0,70) y `azar` **cae** respecto de `todo` en al menos 0,10. Si `azar` no cae, la cota
  no genero presion de capacidad y **ninguna politica puede mostrar nada**: se arregla `N_ARCH` y se
  vuelve a correr, sin interpretar nada del resto. Es la leccion del control vacio `m=1` del 12-ago,
  puesta adelante.

- **E-1 · PRINCIPAL.** `sorpresa` > `azar` en acierto, en al menos 2 de 3 semillas, con diferencia
  >= 0,05.

- **E-2 · CONTRA LA RECENCIA.** `sorpresa` > `fifo`, en al menos 2 de 3 semillas. Es la comparacion
  que le importa a VIGIA-03: la recencia es lo que hace hoy cualquier cache, y superarla es lo que
  justifica gastar en medir sorpresa.

- **E-3 · MECANICISTA.** Lo que `sorpresa` retiene es **lo que despues se pregunta**: la fraccion de
  hechos preguntados que quedaron en el archivo es mayor en `sorpresa` que en `fifo` y que en `azar`.
  Separa «acierta mas» de «acierta mas POR HABER GUARDADO LO QUE HACIA FALTA».

- **E-4 · COSTO DE LA COTA.** `sorpresa` recupera al menos la mitad de lo que `azar` perdio respecto
  de `todo`. Fija la magnitud en unidades interpretables en vez de en diferencias sueltas.

## 4. Regla de decision

- **E-0 falla** → no hay experimento, se corrige la cota y se vuelve a correr. No se interpreta nada.
- **E-1 falla** → la eviction sorpresa-gated **no funciona en este regimen** y la linea se cierra:
  no se prueba una segunda señal de escritura. Seria el negativo que VIGIA-03 pide desde el 21-jul
  («si no hay diferencia, matar la idea, como VIGIA-02»).
- **E-1 pasa y E-2 no** → la sorpresa sirve, pero no mas que la recencia, que es gratis. Se reporta
  asi y **no** como exito: el argumento entero de VIGIA-03 es que la recencia desperdicia rango.
- **E-1 y E-2 pasan y E-3 no** → mejora sin mecanismo, se reporta sin adjudicarsela a la retencion.

## 5. Riesgos declarados

**(a) La sorpresa se mide al ESCRIBIR y lo que importa es si el hecho se va a PREGUNTAR despues.** Es
una apuesta sobre la utilidad futura, no una medicion de ella. Si `sorpresa` gana, lo mostrado es que
la informatividad local predice la utilidad futura **en esta tarea**, donde lo que se pregunta son
hechos. Es mas chico que «el modelo sabe que guardar».

**(b) La charla es fuera de distribucion para los checkpoints de hoy**, y por eso la compuerta solo
autoriza el eje de redundancia. Entrenando **con** relleno desde el paso 0 la charla deja de ser
novedosa; eso puede mejorar o empeorar la señal, y no se predice. Se reporta como observacion.

**(c) Bimodalidad entre semillas.** Medida hoy en `pre`: `ident_rep` vale 0,0564 · 0,4683 · 0,2529 a
igual presupuesto. Con tres semillas, un 2 de 3 puede ser bimodalidad. Todo se reporta **pareado por
semilla**, nunca por media (regla de E-I3c).

## 6. Orden de ejecucion

1. integrar `relleno.py` a `idioma.py` / `datos.py`, con `chequeo_padding.py` verde y el chequeo de
   `relleno.chequeo()` pasando;
2. re-correr la compuerta de sorpresa sobre un modelo entrenado **con** relleno (riesgo b);
3. recien ahi la campania.

El paso 1 esta bloqueado hasta que la campania del camino lateral cierre.
