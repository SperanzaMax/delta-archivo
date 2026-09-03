# Envío a TMLR del paper de la ventana

**Archivo a subir:** `ventana_tmlr.pdf` (7 páginas). TMLR sube **el PDF** a OpenReview, no el fuente.

**Anonimato verificado en las tres capas**, no supuesto:
1. el encabezado sale «Under review as submission to TMLR / Anonymous authors / Paper under
   double-blind review»;
2. `pdftotext` no devuelve ninguna coincidencia de nombre, ORCID, mail ni dominio institucional;
3. **los metadatos del PDF están vacíos** (`Author:`, `Title:`, `Subject:`, `Keywords:`). Ésta es la
   que se filtra siempre y la que nadie mira.

⚠️ Antes de cada recompilado hay que repetir el chequeo 3: `pdfinfo` puede volver a poblar `Author`
si alguien agrega `\hypersetup{pdfauthor=...}` o si se compila desde un editor que lo inyecta.

---

## Title

```
The Query Cannot See the Question: A Short Convolution's Reach Decides Which Part of a Query Conditions Retrieval
```

## Abstract

El del PDF, en texto plano. OpenReview acepta `$...$` para matemática simple; sacar `\textbf`,
`\emph` y `\texttt`.

## Keywords

```
state space models, linear attention, associative recall, abstention, selective prediction,
mechanistic interpretability, retrieval, pre-registration
```

## Lo que hay que declarar en el formulario

⚠️ **CORRECCIÓN del 3-sep, verificada contra la FAQ y el author guide.** La primera versión de este
archivo decía que había que declarar el enlace de arXiv en el envío. **Es al revés.**

- **Preprint en arXiv: SE PUEDE, antes o durante la revisión, y con tu nombre.** Textual de la FAQ:
  *«You can still submit, even if a preprint of your work already is available online somewhere.»*
- **Pero NO se linkea desde el envío.** El author guide es explícito: el doble ciego del envío se
  mantiene *«by not linking to another version that includes the authors' names»*. Así que **no hay
  que pegar la URL de arXiv en ningún campo del formulario**, ni en el PDF, ni en el material
  suplementario.
- **La carga recae en el revisor, no en vos.** A los revisores se les pide que *«no busquen
  activamente»* la identidad de los autores, y la FAQ contempla el hallazgo accidental: *«You may
  come across it by chance. If this happens, and you believe that it will influence your judgment,
  please contact the action editor.»* Que el título sea googleable no es un problema tuyo.
- **Conflictos de interés:** la Facultad Regional Buenos Aires de la UTN, que es lo que ya quedó
  cargado en el perfil de OpenReview. El sistema lo toma de ahí.
- **Material suplementario, si se manda: también anonimizado**, hasta 100 MB, en PDF o ZIP. Para
  este paper no hace falta, pero si alguna vez se manda el código hay que limpiarlo igual que el
  manuscrito.

## Formato, verificado contra la fuente

- **Sin límite de páginas.** Textual: *«Submissions may be any length, but a paper's length should be
  justified by its content»*, con la advertencia de que los muy largos demoran la revisión. Con 7
  páginas no hay problema.
- **La plantilla tiene que ser la oficial**, de
  `github.com/JmlrOrg/tmlr-style-file`. La copia que se usó acá se comparó contra la descarga
  oficial y es **idéntica byte a byte** (`md5 e302463e460a0173d5bee936bc95a625`).
- **Todos los autores tienen que tener el perfil de OpenReview completo y activo**, con afiliaciones,
  conflictos e historial de publicaciones. Ya está.

## Los dos criterios con los que lo van a juzgar, textuales

1. *«Are the claims made in the submission supported by accurate and convincing evidence?»*
2. *«Would some individuals in TMLR's audience be interested in the findings of this paper, and does
   the paper communicate those findings clearly to its intended audience?»*

Y lo que **explícitamente NO** evalúan: novedad (*«novelty of the studied method is not a necessary
criterion for acceptance»*), significancia o impacto (*«we explicitly avoid these terms»*), y estado
del arte (*«should not be rejected because it isn't achieving a new state-of-the-art»*).

**Esto es lo que hace de este paper el candidato correcto.** El criterio 1 premia claims acotados con
evidencia, que es exactamente su forma, y el negativo del Resultado 6 no resta bajo ninguno de los
dos criterios — al contrario, es evidencia de que los claims están delimitados.

## Candidatos a Action Editor

Elegir de la lista viva de AEs de TMLR (cambia seguido, así que hay que mirarla al momento de
enviar) a alguien cuyo perfil incluya **modelos de espacio de estados / atención lineal** o
**interpretabilidad mecanicista**. No elegir por prestigio general: el criterio de TMLR es si los
claims están sostenidos, y eso lo juzga mejor quien conoce la arquitectura.

---

## Por qué este paper y no los otros dos, con el número que lo decide

La cuota es la *Generalized Harmonic Quota Rule* con N₁ = 2: firmando solo, **dos envíos al año**. Y
textual de la política: *«Budget is spent for all submissions by an author, including those which are
desk rejected.»* **Un desk reject cuesta medio año**, así que el primer envío tiene que ser el más
difícil de rebotar de entrada.

| candidato | dónde vive la evidencia |
|---|---|
| **ventana** (éste) | micro-LM **+ `mamba-130m` y `mamba-370m`**, modelos que no son nuestros |
| E1 de Ligamento | sólo el micro-LM |
| CENTINELA | sólo simulación propia |

Y este paper trae **un negativo pre-registrado adentro** (Resultado 6: el efecto de acceso no se
traduce en falla conductual en un modelo de 24 capas), con la explicación de por qué el negativo es
coherente con el positivo. Un editor que busque motivos para rebotarlo se encuentra la objeción ya
contestada en el texto, que es exactamente lo contrario de lo que produce un desk reject.
