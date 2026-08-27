#!/usr/bin/env bash
# NO tiene el token adentro, y esa es toda su razon de ser.
#
# El 27-ago se verifico que el token del bot estaba escrito a mano en 22 scripts de este repo, que
# es PUBLICO en GitHub: un `curl` al raw lo bajaba sin autenticacion. Sacarlo del archivo de hoy no
# alcanza —queda en el historial de commits— pero es la mitad que si depende de nosotros, y deja el
# repo en un estado donde rotar el token es cambiar UN archivo y no veintidos.
#
# El secreto vive fuera del repo, en $TG_ENV (por defecto ~/.config/avisos/telegram.env, chmod 600):
#
#     TG_TOKEN=<token del bot>
#     TG_CHAT=<chat_id>
#
# Se SOURCEA, no se ejecuta:
#
#     . "$(dirname "${BASH_SOURCE[0]}")/tg_token.sh"
#
# y deja definidas TOKEN y CHAT, que son los nombres que los scripts ya usaban. Se eligio respetar
# esos nombres justamente para que el cambio en cada script sea una sola linea y no haya que releer
# la logica de aviso de veintidos archivos.
#
# NO ABORTA SI FALTA EL ARCHIVO, y esto es a proposito. Estos scripts orquestan campanias de 26000
# pasos por unidad; quedarse sin canal de aviso no puede ser motivo para tirar abajo una corrida de
# horas. Avisa por stderr y deja TOKEN vacio: `mandar()` falla sola y en silencio, que es exactamente
# lo que ya hacia cuando la red estaba caida. Un `${TG_TOKEN:?}` habria sido mas prolijo de leer y
# habria matado la primera campania que corriera en una maquina sin configurar.
#
# Los `:-` tambien son necesarios y no decorativos: casi todos los scripts que sourcean esto corren
# con `set -u`, donde leer una variable no definida termina el proceso.
: "${TG_ENV:=$HOME/.config/avisos/telegram.env}"
if [ -r "$TG_ENV" ]; then
  . "$TG_ENV"
else
  echo "!! no se puede leer $TG_ENV — los avisos de Telegram quedan mudos (ver micro_lm/tg_token.sh)" >&2
fi
TOKEN="${TG_TOKEN:-}"
CHAT="${TG_CHAT:-}"
