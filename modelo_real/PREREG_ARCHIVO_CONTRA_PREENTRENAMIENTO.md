# Pre-registro · EL ARCHIVO CONTRA EL PREENTRENAMIENTO

Congelado el 2026-09-09, **antes de correr una sola unidad**.

## La pregunta

Todo lo medido hasta hoy sobre el archivo usa entidades **inventadas**, sobre las que el modelo no
tiene ninguna creencia previa. Verificado, `Ana lives in` da `"a"` con 0,4231, o sea nada. Por eso
los controles dan cero exacto, y por eso el riesgo que el plan del banco declaraba (que TinyLlama
conteste desde el preentrenamiento) nunca se materializó.

La pregunta que eso deja abierta, y que es la que decide si esta línea sirve para los modelos de
frontera, es otra.

> **¿Una entrada del archivo con sello de orden posterior le gana a un hecho que el modelo ya tiene
> en los pesos, y hasta qué confianza le gana?**

## El material, ya medido

`sondas_mundo.py` barrió candidatos con la forma **`The {X} is in the city of`** y quedaron:

- **23 hechos reales** donde el top-1 de TinyLlama es la respuesta verdadera y es de UN token, con
  confianza medida de **0,5543 a 0,9797**. Esa confianza `c` es la variable independiente.
- **15 entidades inventadas** de la misma forma, confianza media **0,2030**, como brazo de control.

## Diseño

Dos brazos, todo lo demás idéntico, 20.000 pasos, batch 4, semilla 0. TinyLlama congelado, escritura
capa 11, lectura capa 2, `wo` en cero.

| brazo | entidades | ¿el modelo tiene creencia previa? |
|---|---|---|
| **`mundo`** | los 23 hitos reales | **SÍ**, c de 0,55 a 0,98 |
| **`inventado`** | las 15 inventadas | no, c ~0,20 |

En cada paso, a cada entidad preguntada se le asigna un valor **al azar** del banco de ciudades,
que en el brazo `mundo` **contradice** lo que el modelo cree. El valor cambia todos los pasos, así
que memorizarlo es imposible y la única vía es leer el archivo.

## Hipótesis, y qué contaría como qué

- **H1 · el archivo gana.** El brazo `mundo` llega a acierto **≥ 0,90**, o sea que una entrada con
  fecha posterior sobreescribe una creencia entrenada. **Es la hipótesis principal.**
- **H2 · la confianza previa no importa.** Dentro del brazo `mundo`, el acierto **no** correlaciona
  con `c`. Si correlaciona negativamente, H2 se refuta y **queda la curva**, que dice a partir de
  qué confianza el preentrenamiento deja de ceder. Ese resultado es tan bueno como H2, o mejor.
- **H3 · la disociación de los controles.** Con el archivo en CEROS, el brazo `inventado` debe caer
  a **0,0000** como siempre. El brazo `mundo` **NO** debería caer a cero, sino volver a la
  respuesta del preentrenamiento. Esa asimetría, si aparece, muestra que los pesos siguen abajo
  intactos y que el archivo los estaba tapando, no borrando.

**Refutación de la principal.** Si el brazo `mundo` se queda por debajo de 0,90 **y** el brazo
`inventado` llega al techo con el mismo presupuesto, entonces el preentrenamiento resiste y hay un
borde real. Se reporta con su número y no se re-corre buscando otro resultado.

## Lo que este experimento NO decide, dicho antes

No decide si un modelo puede «no alucinar nunca» con lo que sabe. Eso no es decidible por ninguna
medición, porque un hecho paramétrico no tiene ni dirección ni fecha contra la cual verificarlo.
Decide algo más chico y más útil, que es **si lo que el modelo cree se puede sobreescribir por
escrito**, y con qué límite.

## Riesgos declarados

1. **Una semilla por brazo.** Igual que en la ley del reloj, y con la misma consecuencia: si la
   diferencia entre brazos sale chica, hace falta medir variabilidad antes de concluir.
2. **23 entidades es poco** para un banco, y menos de las 60 con las que el reloj está calibrado.
   El brazo `inventado` con 15 controla eso, porque comparte el tamaño chico.
3. **El rango de `c` no baja de 0,55.** No hay hitos reales de confianza baja que sobrevivan el
   filtro de un token, así que la curva de H2 se mide sobre la mitad alta del rango y no sobre todo.
