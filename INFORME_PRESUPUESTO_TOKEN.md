# INFORME · EL MISMO PRESUPUESTO PARA `token` Y `escala` (2026-08-20/21)

Pre-registro `PREREG_PRESUPUESTO_TOKEN.md` (SHA `a63b8e80…`), congelado antes de lanzar. Cierra lo
que la réplica de `c4` dejó pendiente: cuantificar el confound del §4 del `INFORME_FRONTERA` yendo a
las unidades que **sí fallaban** la compuerta a 14000.

Instrumento declarado: `medir_compuerta.py`, 2048 muestras, rng 77000 + semilla, `p_nose` 0,4, y
**regla de decisión por condición** — `token` y `escala` deciden por el token NOSE del softmax,
`cabeza` por su salida binaria. Medir `token` con el criterio de `cabeza` daría un número sin
sentido: ahí la cabeza nunca entró en la pérdida.

Verificado antes de lanzar: las cinco unidades tienen `horizonte 20000` en su config original, así
que extender **no toca la curva de lr** y la mejora no es confundible con un cambio de tasa.

## Resultado

`falsa_abst` con el instrumento declarado, 14000 → 20000:

| unidad | cond | 14000 | 20000 | compuerta |
|---|---|---:|---:|---|
| `t4_s0` | token | 0,1567 | 0,1296 | mejora, **falla** |
| `t4_s1` | token | 0,2081 | **0,0713** | **PASA** |
| `t4_s2` | token | 0,1942 | **0,0458** | **PASA** |
| `s4_s0` | escala | 0,2244 | **0,0485** | **PASA** |
| `s4_s1` | escala | 0,1701 | **0,0722** | **PASA** |

Las cinco fallaban a 14000, y eso quedó resuelto **antes** de mirar nada extendido, para no repetir
el error de R-3 de la réplica, que pidió «pasar habiéndola fallado» sin mirar que dos unidades ya
pasaban.

## P-1 · CUMPLE, y por las dos vías

Sobre los puntos nuevos de la serie de entrenamiento (`analizar_presupuesto.py`, 13 puntos por
unidad entre 14000 y 20000):

| unidad | `falsa_abst` inicio → fin | Spearman(paso, `falsa_abst`) |
|---|---|---:|
| `t4_s0` | 0,1342 → 0,1220 | −0,5879 |
| `t4_s1` | 0,1713 → 0,0960 | −0,4396 |
| `t4_s2` | 0,1419 → 0,0407 | −0,7527 |
| `s4_s0` | 0,2436 → 0,0365 | −0,8571 |
| `s4_s1` | 0,1691 → 0,0721 | −0,6099 |

**Baja en 5 de 5 y el Spearman es negativo en 5 de 5**, contra ≥ 4 de 5 pedidos por cada mitad. Es la
réplica de R-1/R-2 en la otra condición.

Las series son la eval interna del tramo, con 512 muestras, y por eso sirven para la **tendencia**,
no para el nivel: los extremos de la tabla de arriba se miden aparte con 2048 y son otros números.
No se mezclan en la misma columna. La potencia de P-1 sale de la **longitud** de la serie, no de la
precisión de cada punto — que es exactamente la distinción que el 19-ago faltó y produjo la falsa
alarma de degradación de `c4_s2`.

## P-2 · CUMPLE con 4 de 5

El criterio decía: **≥ 3 de 5 pasan → la ventaja de `cabeza` estaba en parte comprada con
presupuesto**. Pasan `t4_s1`, `t4_s2`, `s4_s0` y `s4_s1`; la única que no cruza es `t4_s0`. El
desenlace estaba resuelto con 3 antes de medir la quinta, así que `t4_s2` no lo decidió: lo reforzó.

**Consecuencia, comprometida por adelantado y escrita con las palabras del prereg: el hallazgo del
18-ago queda ACOTADO.** «La cabeza pasa la compuerta en 4 de 5 unidades donde `token` y `escala`
fallan 5 de 5» se midió con **todas las unidades a 14000**, y con el mismo presupuesto que se le dio
a `cabeza`, cuatro de las cinco pasan.

Lo que sobrevive sin tocar es el **margen**: a 14000 `cabeza` estaba en `falsa_abst` 0,06-0,08 donde
las otras dos condiciones estaban en 0,16-0,22. Y P-3 lo confirma a 20000, que es donde recién ahora
se puede comparar sin que el presupuesto ensucie la comparación.

## P-3 · CUMPLE 3 de 3, y es lo que le pone el límite al acotamiento de P-2

El contraste que el 18-ago se hizo a 14000 y que nunca se había hecho con las dos condiciones
igualadas. `falsa_abst` a 20000, mismo instrumento, mismas semillas:

| semilla | `cabeza` | `token` | |
|---|---:|---:|---|
| s0 | **0,0633** | 0,1296 | cabeza |
| s1 | **0,0421** | 0,0713 | cabeza |
| s2 | **0,0166** | 0,0458 | cabeza |

**A igual presupuesto `cabeza` sigue teniendo menor `falsa_abst` en las tres semillas**, con una
razón de 2× a 2,8×. Medición reproducible: `c4_s0` dio 0,0633 hoy y 0,0633 ayer, bit a bit.

**Leído junto con P-2, el resultado tiene dos mitades y hay que decir las dos:**

- lo que **cae** es la forma en que el hallazgo estaba enunciado — «`cabeza` pasa la compuerta donde
  las otras fallan» ya no es cierto, porque con 6000 pasos más cuatro de cinco también la pasan;
- lo que **queda en pie** es la comparación pareada — a igual presupuesto la cabeza sigue por debajo
  en las 3 de 3.

O sea: el presupuesto explica el **cruce de la compuerta**, no la **ventaja**. La compuerta era un
umbral, y un umbral convierte una diferencia continua en un sí/no que depende de dónde esté puesto;
lo que no depende del umbral es que la diferencia sigue ahí con el mismo signo.

## P-4 · CUMPLE 5 de 5, y por el lado fuerte

`vigente` no sólo no cae los 0,10 que el control tolera: **sube en las cinco unidades**, hasta +0,32
en `s4_s0` (0,6025 → 0,9234). Las tres cosas —`falsa_abst` abajo, `nose` arriba, `vigente` arriba—
se mueven a la vez, así que **no hubo intercambio**: el modelo no compró abstención a costa de
responder peor.

## Lo que este informe no puede decir

- **No dice que en nivel 4 se falle por presupuesto.** Sigue siendo cierto lo de la réplica: para
  cuantificar el confound del §4 del `INFORME_FRONTERA` hacían falta las unidades que sí fallaban, y
  ahora se sabe que tres de cinco lo cruzan con 6000 pasos más. El confound queda **acotado**, no
  eliminado: `t4_s0` mejora y aun así falla.
- Una semilla por celda en `escala`, tres en `token`. La bimodalidad entre semillas es parte del
  fenómeno en este brazo (E-I3c) y con dos unidades no se separa dificultad de no-convergencia.
- Nada de esto habla de la **calidad** de la abstención, sólo de su tasa de falsos. El techo de
  calibración medido el 18-ago (AUC 0,77-0,99) no lo toca este experimento.
