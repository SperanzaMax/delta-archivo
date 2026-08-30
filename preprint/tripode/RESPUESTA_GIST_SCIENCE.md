# Respuesta a Gist.Science — pedido de corrección

**2026-08-29.** Contestar al mail de `mail@email.gist.science` del 29-ago 04:19, que ofrece corregir
la página si algo está mal (*«If anything's inaccurate, just reply and we'll fix it»*).

Página: <https://gist.science/paper/rs/rs-10839567>

---

## Lo verificado antes de contestar

Se comprobaron **19 afirmaciones numéricas** de la página contra `tripode_en.tex`. **Todas correctas**:
863.730 parámetros · factor 2–2,8× · 129 parámetros (0,015 %) · 256 del slot · gap +0,1465 · masas
0,4074 / 0,4046 / 0,4020 contra base 0,4048 · barrido de 400 puntos · Youden entre +0,038 y +0,078 ·
AUC 0,777–0,998 · siete intentos en AUC 0,50–0,67 · 27 unidades de entrenamiento.

Y respeta el límite principal: dice explícitamente que esto **no** significa *«the model knows when it
does not know»*.

## El único error, y por qué importa

La página afirma:

> *«On difficult training units, the head passed the pre-registered gate in 4/5 cases, while the token
> and renormalized token passed 0/5.»*

Esa tabla **está** en el paper (§ *Budget: what the gate was measuring, and what survives*), pero el
paper **la retracta explícitamente unas líneas después**:

> *«We therefore retract the framing, in the pre-registration's own words: "cabeza passes the gate
> where the others fail" is not true once the others get 6,000 more steps. What survives is the paired
> comparison at matched budget.»*

O sea que la página presenta como hallazgo justo la afirmación que el paper desarma. Es el pasaje más
honesto del trabajo —una retractación de su propio encuadre, hecha con las palabras del
pre-registro— y es el que quedó mal citado.

**No se responde «looks good» hasta que esto se corrija.**

---

## Texto del mail

```
Subject: Re: Your paper on Research Square "Where Abstention Lives..." - explained in plain language

Hi Luc,

Thank you for putting this together, and for offering to fix inaccuracies.

I checked the page against the manuscript and the numbers are accurate throughout
(the 863,730 parameters, the 2-2.8x factor, the +0.1465 gap, the slot's 0.4074 /
0.4046 / 0.4020 against a 0.4048 base rate, the 400-point sweep, Youden's J, the
AUC ranges). I also appreciate that the page keeps the paper's main restraint
intact, that none of this supports "the model knows when it does not know".

There is one passage I would ask you to change, because the paper explicitly
retracts it. Under "What They Found", the page states this.

  "On difficult training units, the head passed the pre-registered gate in 4/5
   cases, while the token and renormalized token passed 0/5."

That table is in the paper, but it appears there in order to be withdrawn. A few
lines later, in the section on budget, the manuscript says this.

  "We therefore retract the framing, in the pre-registration's own words:
   'cabeza passes the gate where the others fail' is not true once the others get
   6,000 more steps. What survives is the paired comparison at matched budget."

The 4/5 vs 0/5 result does not survive giving the competing conditions an equal
budget. What the paper claims instead is the paired comparison at matched budget,
where the separate head reduces false abstention by a factor of 2 to 2.8x against the
vocabulary token in 3 of 3 seeds.

Would you replace that sentence with the matched-budget comparison, or keep it
with the retraction attached? Either works for me; what I would like to avoid is
the gate result standing on its own, since a reader would come away with a
stronger claim than the paper makes.

Everything else on the page looks accurate to me, and once this is adjusted I am
happy for it to be marked as author-reviewed.

Best regards,
Maximiliano Speranza
ORCID 0009-0005-0413-8554
```

---

## Nota sobre el estilo

El único `:` que queda en el cuerpo está **dentro de la cita textual del paper**, y se deja porque
alterar una cita para que suene a él la falsearía. Todo lo demás va sin dos puntos y sin raya larga.

## Nota

**No se pide retirar la página ni se discute el servicio.** El pedido es una corrección puntual y
verificable, con la cita del propio paper que la sostiene. Si contestan corrigiendo, se responde el
«looks good» que piden y queda marcada como author-reviewed, que suma un poco de visibilidad — y la
visibilidad es cuello de botella conocido del proyecto mientras el perfil de Scholar siga sin
verificar.
