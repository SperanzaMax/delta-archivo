# SER · el error no es de versión, es de identidad

**2026-08-15** · micro-LM, idioma v2, checkpoints a 12 000 pasos · `ser.py`

## Qué se midió y por qué faltaba

La §6 del diseño pide medir el error silencioso **desagregado por tipo**, y hasta hoy no se hacía:
las corridas reportaban `vigente` y `anterior` y nada más. La desagregación es el corte propio
frente a **FAMA** (arXiv 2604.20006), que penaliza el reuso de memoria invalidada sin separar dos
fallas que son de mecanismos distintos:

- **error de versión** — contesta otra versión *del hecho que se preguntó*. Encontró el hecho y se
  equivocó de momento. Es la falla del orden temporal.
- **error de identidad** — contesta el valor de *otra entidad*. Ni siquiera fue al hecho correcto.
  Es la falla del direccionamiento.

`SER` cuenta sólo los errores contestados **con seguridad**: si el modelo se abstiene no entra. Esa
es la tesis del proyecto — un error avisado cuesta una respuesta, uno silencioso cuesta la confianza
en todas las demás.

Implementación aditiva: `idioma.episodio(con_meta=True)` y `datos.lote(con_meta=True)` devuelven los
hechos del episodio con todas sus versiones. **Verificado bit a bit contra un hash de referencia:**
el generador no cambió, las corridas siguen siendo reproducibles desde su semilla.

## Resultado

| nivel | acierto | SER | err_versión | err_identidad | err_fuera |
|---|---:|---:|---:|---:|---:|
| N1 · plantilla fija | 0,9980 | 0,0020 | 0,0020 | 0,0000 | 0,0000 |
| N2 · paráfrasis | 1,0000 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| N3 · corrección elíptica | 0,7754 | 0,2246 | 0,0020 | **0,2227** | 0,0000 |
| N4 · multi-sesión | 0,7598 | 0,2402 | 0,0078 | **0,2324** | 0,0000 |

Tres lecturas, en orden de solidez:

**1. El versionado está resuelto.** `err_versión ≤ 0,0078` en los cuatro niveles. Es el problema que
la línea viene persiguiendo desde R1 («agrupa perfecto pero no ordena»), que E-I2 midió en 0,4576
—azar entre la vieja y la nueva— y que E-I3 reparó con el sello de orden. Acá aparece cerrado en un
modelo entrenado de punta a punta.

**Control obligatorio, porque el reparto no significa nada sin él:** con pocos valores en juego,
«el valor de otra entidad» se acierta por azar más seguido que «otra versión del mismo hecho»,
simplemente porque hay más entidades ajenas que versiones propias. Se calculó el reparto esperado
eligiendo uniformemente entre los valores presentes en el archivo:

```
esperado por azar    versión 0,0741  ·  identidad 0,9259
observado            versión 0,0279  ·  identidad 0,9721
```

Los errores de versión son **2,7× menos frecuentes que por azar**. El descarte no es un artefacto
del conteo.

**2. Lo que rompe es la identificación, y aparece exactamente en N3.** N1 y N2 no tienen un solo
error de identidad; N3 salta a 0,2227 y N4 se queda ahí. N3 es el nivel que introduce la
**corrección elíptica** (`no , es beto`, sin nombrar la entidad): la corrección no dice de quién
habla, así que hay que inferir a qué hecho pegarla — y es ahí donde el modelo se equivoca de dueño.
La falla cae donde el mecanismo predice que caiga.

**3. El modelo nunca inventa contenido: `err_fuera = 0,0000` en los cuatro niveles.** Toda respuesta
equivocada es un valor que **está** en el archivo, puesto en la entidad que no era. La alucinación
acá no es fabricar un dato, es **atribuir mal uno real**. Es una forma más difícil de detectar desde
afuera, porque cualquier verificación de «¿este dato existe?» la da por buena.

## Por qué importa para el arco de la línea

El preprint publicado (DOI 10.21203/rs.3.rs-10669947/v1) muestra que un índice **no paramétrico**
sobre encoder congelado da **0,0000 exacto en 10/10 semillas** con correcciones elípticas crudas: se
pierde el 100 % de las correcciones, en silencio. El §5 del diseño lo dice al presentar N3: «que un
modelo entrenado desde cero con el archivo adentro lo resuelva sería el argumento que hoy no
tenemos».

Ese argumento ahora existe: **0,7754 contra 0,0000**, y el resto del error está caracterizado en vez
de ser una bolsa de fallos.

## Lo que este informe NO dice

- **No mide abstención.** Los checkpoints se entrenaron con `p_nose = 0,0`, así que el modelo nunca
  tuvo una pregunta sin respuesta y `NOSE` no era una opción aprendible. La columna de abstención da
  0,0000 por construcción, no por resultado. Eso lo mide la campaña `x`, en cola.
- **N3 y N4 tienen una sola semilla completa cada uno.** Con la bimodalidad ya medida (nivel 2 dio
  0,8028 en una semilla y 1,0000 en las otras dos), un solo valor no distingue dificultad de semilla
  trabada. Los números de N3 y N4 son provisorios hasta tener las tres.
- **No separa las dos causas posibles del error de identidad**: puede ser que la corrección elíptica
  se pegue al hecho equivocado *al escribir*, o que la consulta recupere el hecho equivocado *al
  leer*. Distinguirlas es el experimento siguiente y encaja con `NOTA_FOCO.md`.
