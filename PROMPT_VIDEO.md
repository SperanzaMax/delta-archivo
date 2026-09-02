# Prompt para generar un video corto del micro-LM · 2026-09-02

Pensado para Sora, Veo, Kling o Runway. **En inglés**, porque todos rinden bastante mejor así.
Abajo va la versión en español por si querés editarla.

**Antes de usarlo, tres avisos prácticos.**

1. Los generadores de video **escriben mal el texto**. Por eso el prompt pide poquísimas palabras en
   pantalla y números cortos. Si igual salen deformados, generá sin texto y agregá los rótulos
   después con cualquier editor.
2. Conviene generarlo **escena por escena**, de 5 a 8 segundos cada una, y pegarlas. Un prompt largo
   de una sola vez sale peor.
3. Formato sugerido **16:9** para LinkedIn, o 9:16 si lo querés vertical.

---

## Escena 1 · La memoria que no se pisa (6 s)

> A dark studio space, deep charcoal background, soft volumetric light. Rows of small translucent
> glass tablets float in mid-air in a shallow arc, each glowing faint cyan, like index cards made of
> light. A single tablet brightens as a new one materializes just beside it, slightly forward,
> connected by a thin luminous thread — the old one dims to a cool blue but does not disappear. Slow
> dolly-in, shallow depth of field, cinematic, photoreal, no text. Calm and precise, not flashy.

**Qué muestra.** La gemación. Al revisar un dato no se sobrescribe, se deposita una versión nueva al
lado y la vieja sigue existiendo.

## Escena 2 · Preguntar y que responda la versión que rige (6 s)

> Same floating archive of glass tablets. A warm amber pulse travels from the foreground into the
> arc of tablets, touches several, and one single tablet — the newest of a connected pair — flares
> bright white while all the others stay dim. The camera pushes past the dim ones toward the lit
> one. Volumetric light, slow motion, photoreal, cinematic, no text.

**Qué muestra.** El sello de orden. La búsqueda encuentra el hecho y además sabe **cuál versión**
rige. Es el resultado que pasó de 0,4570 a 0,9956.

## Escena 3 · La ventana que no alcanza (7 s) · **la escena clave**

> A row of seven small glowing cubes floats in a straight line against black, like words on an
> invisible sentence. A narrow cone of light, a spotlight from above, illuminates only the last
> three cubes on the right. The fourth cube from the right sits just outside the edge of the light,
> completely dark, one single step beyond the beam. Slow lateral camera move revealing the sharp
> boundary between lit and unlit. The dark cube pulses faintly, unseen. Photoreal, high contrast,
> cinematic, no text.

**Qué muestra.** El hallazgo. La consulta se arma con una ventana corta, y la palabra que dice **qué**
se está preguntando cae **un solo paso afuera**. El modelo no la ignora, no la puede ver.

## Escena 4 · La ventana se ensancha (6 s)

> Same row of seven glowing cubes. The cone of light slowly widens, and the previously dark cube is
> swallowed by the beam and ignites bright gold. The moment it lights up, the whole row snaps into a
> coherent alignment and a soft pulse travels down the line. Satisfying, precise, photoreal,
> cinematic, no text.

**Qué muestra.** El arreglo, que cuesta 1.280 parámetros sobre 865.395.

## Escena 5 · Decir «no sé» (7 s) · **el cierre**

> The floating archive again. An amber query pulse enters, sweeps across every glass tablet, and
> finds nothing — each tablet it touches stays dim. The pulse slows, hesitates, and instead of
> lighting a wrong tablet it simply dissolves into a soft steady blue glow in the empty space where
> an answer would have been. Restrained, quiet, no explosion. Photoreal, cinematic, shallow depth of
> field, no text.

**Qué muestra.** Lo que buscábamos. Cuando la respuesta no está, el modelo **no enciende la
equivocada**. Es `nose_rel` pasando de 0,59 a 1,00 con `falsa_abst` en 0,0000.

---

## Rótulos, para agregar en edición y no en el generador

- Escena 2 → **0,4570 → 0,9956**
- Escena 3 → **un token afuera**
- Escena 4 → **+1.280 parámetros**
- Escena 5 → **0,59 → 1,00**
- Cierre → **micro-LM · 3,5 MB · entrenado desde cero**

## Sobre qué se puede afirmar

Cuidado con titularlo «eliminamos las alucinaciones». Lo honesto y que igual suena fuerte es
**«un modelo de 3,5 MB que no olvida lo que se le dijo y sabe cuándo no lo sabe»**. Todo lo del
video pasó de verdad y está medido. La palabra «eliminar» aplica a un tipo de alucinación, la
atribución equivocada desde una memoria, y en este vehículo.

---

## Versión en español, por si preferís generarlo así

**Escena 1.** Estudio oscuro, fondo gris carbón, luz volumétrica suave. Filas de pequeñas tabletas de
vidrio translúcido flotando en arco, brillando en cian tenue. Una se ilumina mientras otra nueva se
materializa al lado, apenas adelante, unida por un hilo luminoso. La vieja se atenúa a azul frío pero
no desaparece. Cámara acercándose lento, poca profundidad de campo, fotorrealista, sin texto.

**Escena 2.** El mismo archivo flotante. Un pulso ámbar entra desde el frente, toca varias tabletas y
una sola, la más nueva de un par conectado, estalla en blanco brillante mientras el resto queda
apagado. La cámara pasa entre las apagadas hacia la encendida. Cámara lenta, fotorrealista, sin texto.

**Escena 3.** Siete cubos luminosos en línea recta sobre negro, como palabras de una oración
invisible. Un cono de luz cenital ilumina sólo los tres últimos de la derecha. El cuarto queda justo
afuera del borde, completamente oscuro, a un paso del haz. Movimiento lateral lento que revela el
límite nítido entre lo iluminado y lo oscuro. Alto contraste, fotorrealista, sin texto.

**Escena 4.** Los mismos siete cubos. El cono de luz se ensancha despacio y el cubo oscuro queda
adentro y se enciende en dorado. En ese instante toda la fila se alinea y un pulso suave la recorre.
Fotorrealista, sin texto.

**Escena 5.** El archivo otra vez. Un pulso ámbar entra, barre todas las tabletas y no encuentra nada,
ninguna se enciende. El pulso desacelera, duda, y en vez de encender una equivocada se disuelve en un
resplandor azul quieto en el lugar vacío donde habría ido la respuesta. Contenido, silencioso, sin
explosión. Fotorrealista, sin texto.
