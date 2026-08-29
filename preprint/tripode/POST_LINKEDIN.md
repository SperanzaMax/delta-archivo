# Post de LinkedIn — trípode de abstención

DOI a pegar: https://doi.org/10.21203/rs.3.rs-10839567/v1
Imagen: `IMG para Link.png`

---

## Texto

Un modelo de lenguaje puede saber que no sabe algo y contestar igual.

Acabo de publicar un preprint que compara cuatro maneras distintas de preguntarle a un modelo chico si
debería abstenerse. Cuatro interfaces, pre-registradas antes de correr nada, sobre un micro modelo de
3,5 MB entrenado desde cero con memoria versionada adentro.

Gana la cabeza dedicada. Ordena las preguntas que va a errar con un AUC de 0,9998, casi perfecto.

Lo interesante es lo que eso implica. La señal está. El modelo distingue lo que sabe de lo que no
sabe con una precisión que no deja lugar a dudas. Lo que no aparece es la manera de convertir esa
señal en la decisión de callarse. Después de este trabajo probé tres vías independientes para cerrar
ese salto y las tres chocaron contra la misma pared el mismo día.

O sea que el problema no es de capacidad. Es de calibración.

Me parece que ahí hay algo que vale la pena mirar de cerca, porque cambia la pregunta. No se trata de
enseñarle al modelo a reconocer su propia ignorancia, eso ya lo hace. Se trata de que actúe en
consecuencia.

El preprint está acá, con los pre-registros y los datos.

https://doi.org/10.21203/rs.3.rs-10839567/v1

#MachineLearning #AI #Research #LLM

---

## Texto alternativo de la imagen

Gráfico de barras que compara cuatro interfaces de abstención en un modelo de lenguaje pequeño. La
condición "cabeza" alcanza un AUC de 0,9998 y supera a las otras tres.
