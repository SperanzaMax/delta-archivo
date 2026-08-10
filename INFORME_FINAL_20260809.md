# Informe — tarea de hechos versionados (2026-08-09, madrugada)

## Resumen en una línea

El experimento se detuvo **dos veces**, por dos razones distintas, ambas preregistradas. No hay
resultado sobre las cuatro condiciones: **ninguna se analizó**.

---

## 1. Qué se hizo

1. Se diseñó una tarea de **memoria versionada de hechos** comparable con VersionRAG: cada entidad
   tiene un atributo cuyo valor se revisa (v1 → v2), y se consulta por el valor.
2. Se **pre-registró** con hash antes de generar un solo embedding: 4 condiciones (`sin`,
   `sobrescritura`, `duplicados`, `gemacion`), predicciones P1–P4, margen de 0.02, ε=0.30, k=5,
   10 semillas, y una **compuerta de identificabilidad** (AUC > 0.95) con criterio de detención.
3. Se registró la desviación D1 (semillas de la parte generativa) **antes** de ver resultados.

## 2. Primer intento — `gemma:2b`

9.000 embeddings en T4, 1369 s, 0 fallos.

**Compuerta: AUC 0.8937 → NO PASA.** cos(consulta, su hecho) 0.8351 vs cos(consulta, otro) 0.7779:
separación de 0.057. El experimento se detuvo sin analizar ninguna condición.

**Exploratorio declarado:** corregir anisotropía rescata el AUC (centrado 0.9670; all-but-the-top
k=1 0.9708) **pero no el top-1** (0.09 → 0.13). Diagnóstico: `gemma:2b` es un modelo generativo;
sus embeddings no están optimizados para recuperación.

**Consecuencia sobre R12:** aquellos 1.000 perfectos usaban wikitext, textos naturalmente diversos.
R12 midió un régimen fácil, no la capacidad general del espacio.

## 3. Segundo intento — `nomic-embed-text` (enmienda E1, con hash)

9.000 embeddings en 244 s (27 ms c/u, 5,6× más rápido).

**Compuerta: PASA.** Recalculado localmente sobre los datos bajados:

| métrica | valor |
|---|---|
| AUC | **0.9928** |
| top-1 del hecho correcto | **0.708** sobre 3.000 |
| top-10 / top-100 | 0.803 / 0.989 |
| rango mediano | **0** |

El encoder dedicado identifica entidades muy bien — exactamente lo que fallaba con Gemma.

## 4. El hallazgo que detiene todo

**Los embeddings de v1 y v2 son idénticos bit a bit en 3.000/3.000 casos:**

```
max|E1 - E2| por entidad: media 0.00e+00   max 0.00e+00
vectores bit a bit idénticos: 3000/3000
cos(v1, v2) = 1.000000
```

Dos textos distintos —"The director of X is **Ana Ruiz**" y "…is **Beto Lima**"— producen el mismo
vector. **No es una propiedad del modelo: es un defecto de procesamiento**, casi con seguridad
truncamiento del texto (la entidad va al principio y sobrevive; el valor va al final y se pierde).
Lo confirma que sí distingue entidades entre sí: cos(v1_i, v1_j) = 0.7016 para i≠j.

**Por qué esto detiene el experimento:** si v1 y v2 ocupan exactamente el mismo punto, **ninguna
geometría puede distinguir la versión vigente de la anterior**. La tarea pierde toda señal, y las
cuatro condiciones darían números sin significado. Es el mismo tipo de fallo silencioso que en R13:
correrlo igual habría producido tablas presentables y vacías.

## 5. Discrepancia no resuelta

El script en la VM reportó `top-1 = 0.0200`; el mismo cálculo sobre los datos bajados da **0.708**.
La versión local es la verificable (los datos están en disco y el cálculo es reproducible), pero
**la causa de la discrepancia no está identificada** y queda anotada como pendiente. No afecta la
conclusión —el bloqueo es el punto 4— pero no debe darse por resuelta.

## 6. Estado

- **Ninguna de las cuatro condiciones fue analizada.** No hay resultado sobre P1–P4.
- Prereg, desviación D1 y enmienda E1 congelados con hash y timestamp.
- Datos guardados: `hechos.npz` (Gemma, 72 MB), `hechos_nomic.npz` (nomic, 9000×768).
- Colab: sin sesiones activas.

## 7. Qué haría falta para retomar

1. **Arreglar el truncamiento**: verificar el límite de tokens de `nomic-embed-text` en Ollama y
   acortar los textos, o poner el valor al principio de la oración. Es un chequeo de 5 minutos que
   debería ser parte de la compuerta.
2. **Ampliar la compuerta**: el umbral por AUC fue insuficiente en el primer intento (AUC 0.97 con
   top-1 de 0.13). La compuerta debería exigir **top-1 y rango**, no solo AUC — es la lección de R7
   aplicada al propio criterio de admisión.
3. **Agregar un chequeo de discriminación de valores**: verificar que el embedding distingue v1 de
   v2 *antes* de montar nada. Ese chequeo no existía y es el que habría atajado esto en el minuto
   uno.

## 8. Lectura honesta

Dos noches, dos detenciones, cero condiciones analizadas. Pero las detenciones fueron **por diseño
y a tiempo**: en ambos casos el mecanismo de compuerta hizo lo que debía, y el costo fue horas de
GPU en lugar de un resultado inventado. La lección aprovechable es que las compuertas estaban
**incompletas** —miraban identificación pero no discriminación de valores, y usaban AUC donde
hacía falta rango— y eso ya está corregido para el próximo intento.
