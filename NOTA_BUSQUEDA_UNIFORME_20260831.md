# La búsqueda es SIEMPRE EL MISMO ACTO, y además a media máquina

**2026-08-31, noche.** Medición directa de la intuición de Maxi: *«¿y si le enseñamos a buscar
diferente, que no todo sea lo mismo?»*. `n3_s0`, 3072 consultas, `p_nose` 0,4. CPU.

**Es una NOTA, no un pre-registro:** no hay criterio fijado de antemano, es descriptiva y así se
informa. Lo que la hace publicable dentro del proyecto es que el número tiene un patrón de referencia
—el rango que la búsqueda PODRÍA usar— y ese patrón no se eligió después.

---

## 1. El número

| rasgo de la búsqueda | media | desvío | CV | con respuesta | sin respuesta | brecha |
|---|---:|---:|---:|---:|---:|---:|
| entropía de la lectura | 1,7687 | 0,1675 | **0,095** | 1,7717 | 1,7642 | **−0,0075** |
| masa en la entrada ganadora | 0,2061 | 0,0316 | 0,154 | 0,2054 | 0,2072 | **+0,0018** |
| brecha top-2 de los scores | 0,0731 | 0,0633 | 0,866 | 0,0717 | 0,0751 | +0,0034 |
| norma de lo leído | 19,6451 | 2,6959 | 0,137 | 19,6497 | 19,6383 | −0,0114 |

**La brecha entre buscar algo que está y algo que no está vale 0,04 σ en la entropía y 0,06 σ en la
masa ganadora.** Indistinguible.

## 2. Y el hallazgo que no se buscaba: la búsqueda no usa el rango que tiene

Con 40 entradas la entropía de la lectura puede ir de **0** (mira una sola) a **3,689** (reparte
entre todas). La observada vive entre **1,386 y 2,060** (p5-p95), una banda estrecha en el medio, y
la masa en la ganadora está siempre cerca de **0,206**: reparte entre unas cinco entradas, **siempre**.

> **No es que a veces busque afilado y a veces difuso. Busca siempre igual, y siempre a media
> máquina.**

## 3. Por qué esto importa, y qué corrige de la lectura de hoy

El informe de esta tarde midió que la evidencia de ausencia en la búsqueda cruda es **azar**
(`s_max` 0,5115, sonda sobre el vector completo 0,5065, y hoy L5 0,4986 con nulo 0,4904) y lo leyó
como «la señal no está ahí». **Con este número la lectura se precisa: no está ahí porque la búsqueda
no tiene con qué ponerla.**

`modelo.py:293` calcula `sim = q·k / sqrt(d)` y el divisor es una **constante**. El modelo no tiene
ningún grado de libertad para buscar con más o menos decisión según la consulta. La forma de su
lectura la fija la arquitectura, no la pregunta. **Un mecanismo sin ese grado de libertad no puede
producir una señal que dependa de la consulta, y la ausencia es exactamente eso.**

**El slot nulo NO arregla esto**, y conviene decirlo porque las dos ideas salieron de la misma frase
de Maxi la misma noche: el slot agrega **un destino** para la búsqueda («ninguna»); no agrega **modos
de buscar**. Las cuatro unidades que corren desde las 18:47 prueban lo primero, no lo segundo.

## 4. El diseño que se deriva, y el precedente que lo sostiene

**Afilado por consulta:** un `β(x)` aprendido que multiplique los scores antes del softmax. Con `β`
alto la búsqueda es decisiva, con `β` bajo es difusa. Es el cambio más chico que crea el grado de
libertad, no toca el resto de la arquitectura, y **el propio modelo decide cuándo usarlo**.

**El precedente del proyecto es el argumento fuerte:** el blanco `error` da **0,65 medido post-hoc y
1,0000 entrenado**. Lo que se entrena, se crea. Una búsqueda con un grado de libertad puede
**fabricar** la señal de ausencia en vez de tener que extraerla de donde hoy no está.

**Otras tres formas de «que no todo sea lo mismo», por si el afilado no alcanza:** descomponer la
consulta en entidad × relación —que ya se hizo y funcionó: es `lat2`, `err_identidad` 0,0000—;
presupuesto variable, o sea buscar dos veces si la primera no encontró; y cabezas de lectura con
roles distintos.

## 5. Lo que NO dice

- **No dice que un `β(x)` vaya a funcionar.** Dice que hoy el grado de libertad no existe.
- **Un modelo, un nivel, un régimen.** `n3_s0`, nivel 3, `p_nose` 0,4.
- **No es un pre-registro** y no adjudica ningún criterio.
