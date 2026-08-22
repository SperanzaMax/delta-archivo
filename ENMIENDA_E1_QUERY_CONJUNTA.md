# ENMIENDA E-1 al `PREREG_QUERY_CONJUNTA.md` (2026-08-22, ANTES de lanzar)

Se escribe y se hashea antes de que corra un solo paso. No hay ningun resultado a la vista.

## Que se descubrio

Al cablear la infra para lanzar la campania aparecio que `tramo_abst.sh` **siembra**: la primera vez,
una unidad de la campania de abstencion arranca copiando el checkpoint BASE `n{nivel}_s{semilla}.pkl`.
Verificado en disco: `n3_s0/s1/s2` estan en el **paso 12000**, con `p_nose = 0.0`. O sea las unidades
historicas `c3_s*` no son «14000 pasos», son **12000 de base mas 14000 de fase = 26000 pasos
efectivos**, y la abstencion entro como segunda fase de un curriculum.

El §2 del pre-registro fijaba 20000 pasos por unidad. Con siembra o sin ella, ese numero estaba mal
elegido, y de las dos maneras:

- **Sembrando**, el brazo `post` heredaria pesos entrenados en la arquitectura `pre` —un encoder ya
  adaptado a una query que es funcion pura del token— y tendria que desaprenderlo. Es un confound
  severo y ademas el chequeo de identidad del checkpoint lo aborta, con razon.
- **Sin sembrar**, 20000 desde cero es MENOS presupuesto que los 26000 efectivos de las historicas, y
  encima las unidades tienen que aprender el nivel 3 y la abstencion a la vez en vez de en dos fases.
  Un negativo ahi seria de nuevo el error que este proyecto ya pago cuatro veces.

## Que cambia

1. **Sin siembra.** Las seis unidades arrancan desde cero. El flag `SEMBRAR=0` se agrega a
   `tramo_abst.sh` para poder decirlo explicitamente en vez de depender de que el archivo base no
   exista. El contraste que interesa es de ARQUITECTURA, y para eso las dos ramas tienen que partir
   del mismo lugar, que es ninguno.
2. **Horizonte y pasos: 26000**, no 20000. Iguala el presupuesto efectivo de las unidades historicas.
3. **`p_nose = 0.4` desde el paso 0** en las dos condiciones, sin curriculum de dos fases. Cambia la
   dinamica respecto de lo historico, pero **la cambia igual para `pre` y para `post`**, que es lo
   unico que el contraste necesita.

## Que NO cambia

Las cinco predicciones del §4, la regla de decision del §5 y el riesgo declarado del §6 quedan tal
cual. Ninguna depende del numero de pasos ni del punto de partida: todas comparan `post` contra `pre`
medidos con el mismo instrumento y el mismo presupuesto.

## Limite que queda asentado

Los numeros de esta campania **no son comparables con los de `c3_s*`** (otro curriculum, otro punto de
partida). Cualquier lectura contra el `err_identidad` historico de 0,19-0,21 es de contexto, no de
comparacion. El contraste valido es interno, `post` contra `pre`, dentro de esta campania.
