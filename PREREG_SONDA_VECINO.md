# PREREG · el error de identidad, ¿se escribe mal o se lee mal?

**2026-08-16** · Fase 0, ítem 3. CPU, checkpoints existentes, cero GPU.
**Se hashea y ancla ANTES de correr el script.**

## §1 · Por qué decide el plan entero

`err_identidad` es el 93 % de lo que falla (0,2227 en N3, 0,2324 en N4): el modelo trae un valor
**real** del archivo y lo pega a la entidad equivocada. La revisión externa separó tres clases de
alucinación y sólo dos son atacables con abstención:

- **clase 2 · confusión al leer** — dos claves compiten bajo la consulta. Tiene señal interna →
  **convertible** en abstención.
- **clase 3 · memoria falsa por escritura corrupta** — la corrección elíptica se ligó al vecino **al
  escribir**. Al leer, ese hecho falso tiene matcheo alto, margen alto y entropía baja: es
  mecánicamente **idéntico** a uno verdadero → **ninguna abstención lo detecta, por diseño**.

Dato que lo motiva: el umbral de confianza apagó el **48 %** de los errores de identidad. La mitad
con señal es clase 2. **La otra mitad podría ser clase 3, y entonces ninguna campaña de abstención la
va a tocar nunca.** Sin este número, el criterio de éxito de la campaña está mal calibrado.

## §2 · El diseño: preguntar por el vecino

Sobre episodios de N3 donde el modelo comete `err_identidad`:

1. Se identifica el **vecino** = el hecho de `otros` cuyas versiones contienen el valor que el modelo
   contestó.
2. Sobre **el mismo episodio y el mismo archivo** (no se regenera nada) se hacen dos consultas
   nuevas, **no ambiguas**, que nombran la entidad explícitamente:
   - `q_propia` — por el hecho que se preguntó originalmente;
   - `q_vecino` — por el vecino.

Lectura, comprometida de antemano:

| resultado de `q_vecino` | interpretación | clase |
|---|---|---|
| devuelve el **valor propio del vecino** | el archivo está intacto; el error apareció sólo bajo la consulta original | **2 · lectura** |
| devuelve el **valor nuevo del hecho propio** (el de la corrección) | la corrección se ligó al vecino al escribir: el archivo contiene un hecho falso | **3 · escritura** |

## §3 · Predicciones

- **P-1.** La consulta no ambigua sobre el hecho propio **recupera** una parte de los errores:
  acierto de `q_propia` ≥ 0,30 sobre los casos que fallaban con la consulta original. Si diera ~0, el
  hecho propio no está recuperable de ninguna forma y el problema no es de consulta.
- **P-2 (principal).** La escritura contribuye de forma no trivial: **fracción de vecinos corruptos
  ≥ 0,25**. Si se cumple, hay un piso de error que la abstención **no puede** bajar y hay que atacar
  por el lado de la ligadura al escribir.
- **P-3 (control, y puede fallar).** En los casos donde el modelo **acertó**, el vecino debe estar
  intacto: fracción de vecinos corruptos ≤ 0,10. Si también salieran corruptos ahí, la sonda estaría
  midiendo un artefacto de la consulta y no el estado del archivo, y P-2 no sería interpretable.

## §4 · Controles

- **La sonda no regenera el episodio ni el archivo**: se reusan `ses`, `cortes`, `turnos`, `mask`
  exactos de la muestra, y sólo cambia el tensor de la consulta. Si se regenerara, se estaría midiendo
  otro episodio.
- **La condición de «corrupto» exige que el valor devuelto sea el de la corrección del hecho propio**,
  no cualquier error: un vecino que devuelve un tercer valor cuenta como `otro`, y se reporta aparte.
- **Se reporta n de cada celda.** Con `err_identidad` ≈ 0,22, de 4000 muestras salen ~880 casos.

## §5 · Límite

Checkpoints entrenados con `p_nose = 0` y una sola semilla por nivel. Mide **dónde** falla la
ligadura, no cuánto mejoraría al arreglarla.
