# El token del bot estaba en el repo, y el repo es público

**27-ago-2026.** Hallazgo fuera de plan, encontrado mirando si convenía pushear los commits del 26.

## Lo que se verificó, no lo que se supuso

`https://github.com/SperanzaMax/delta-archivo` responde **200 sin autenticación**, y

```
curl -s https://raw.githubusercontent.com/.../main/micro_lm/avisar_telegram.sh | grep <token>
```

lo devuelve. No es una sospecha por leer el `.git/config`: el secreto se bajó de internet.

**Alcance:** 22 scripts trackeados, todos con la misma línea `TOKEN="8723956710:AA…"`, más el
`chat_id` en 18 de ellos. Y está **en el historial de commits**, así que sacarlo del árbol de hoy no
lo borra de GitHub.

**Qué habilita:** mandar mensajes haciéndose pasar por `@Albertagente_bot`, y leer los mensajes que
le lleguen al bot con `getUpdates`. No da acceso a la máquina ni a las cuentas de Colab.

## Lo que se hizo hoy

El secreto salió del repo a `~/.config/avisos/telegram.env` (chmod 600), y los 22 scripts ahora
sourcean `micro_lm/tg_token.sh`, que **no contiene el token** y sólo sabe dónde buscarlo.

Dos decisiones de diseño que no son de estilo:

1. **El helper no aborta si falta el archivo.** Estos scripts orquestan campañas de 26000 pasos por
   unidad; quedarse sin canal de aviso no puede tirar abajo una corrida de horas. Avisa por stderr y
   deja `TOKEN` vacío, que es el comportamiento que ya tenían con la red caída. Verificado en los dos
   caminos: con el archivo (envío real, `ok:true`) y sin él bajo `set -euo pipefail` (sobrevive).
2. **Se conservaron los nombres `TOKEN` y `CHAT`.** Así el cambio es una línea por script y no hubo
   que releer la lógica de aviso de veintidós archivos.

**Se editó con `sed -i` y no con un editor, a propósito:** `sed -i` reemplaza por *rename* atómico,
así que el `rotar_abst3.sh` que estaba corriendo en ese momento conservó su inodo abierto y la
campaña A5 no se tocó. Verificado: `/proc/183276/fd/255` apuntaba al inodo viejo, marcado
`(deleted)`. Editar en el lugar un script que bash está leyendo por bloques es la forma conocida de
corromper una corrida larga.

## Lo que FALTA, y es de Maxi

1. **Rotar el token en BotFather.** Mientras no se rote, el que está publicado sigue siendo válido:
   lo de hoy evita que se filtre de nuevo, no cancela lo ya filtrado. Cuando se rote, se cambia
   `~/.config/avisos/telegram.env` y nada más.
2. **Decidir qué hacer con el historial.** Limpiarlo es reescribir la historia de un repo público
   (`git filter-repo` + push forzado), y eso rompe cualquier clon existente. Rotar el token vuelve
   inofensivo lo que quedó, así que el orden sensato es **rotar primero** y recién después decidir si
   el historial vale el costo.

Mientras tanto **no se pushea nada**: subir estos commits sin rotar sería volver a publicar el mismo
secreto en el mismo lugar.
