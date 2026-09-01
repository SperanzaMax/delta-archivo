#!/usr/bin/env bash
# Cambia el token del bot de avisos DESPUES de revocarlo en BotFather.
#
#   Uso:  ./rotar_token.sh 8723956710:EL_TOKEN_NUEVO_QUE_TE_DIO_BOTFATHER
#
# Hace tres cosas y en este orden: valida que el token nuevo FUNCIONE contra la API de Telegram
# antes de tocar nada, guarda una copia del archivo viejo, y recien despues escribe. Si el token
# no anda, no toca el archivo: mas vale seguir con el viejo que quedarse sin canal de avisos con
# 26 scripts que lo usan.
set -uo pipefail
NUEVO="${1:?falta el token nuevo. Uso: ./rotar_token.sh 8723956710:AA...}"
ENV="${TG_ENV:-$HOME/.config/avisos/telegram.env}"

echo "== 1. validando el token nuevo contra la API"
OK=$(curl -s "https://api.telegram.org/bot$NUEVO/getMe" | grep -o '"ok":true')
[ -z "$OK" ] && { echo "   ** el token nuevo NO responde. No se toco nada. **"; exit 1; }
echo "   responde OK"

echo "== 2. copia de seguridad"
cp -a "$ENV" "$ENV.bak-$(date +%Y%m%d%H%M)" && echo "   guardada"

echo "== 3. escribiendo el token nuevo"
sed -i "s|^TG_TOKEN=.*|TG_TOKEN=$NUEVO|" "$ENV"
chmod 600 "$ENV"

echo "== 4. probando un aviso real"
. ./tg_token.sh
R=$(curl -s -X POST "https://api.telegram.org/bot$TOKEN/sendMessage" -d chat_id="$CHAT" \
    --data-urlencode "text=Token rotado correctamente. Este mensaje salio con el token NUEVO.")
echo "$R" | grep -q '"ok":true' && echo "   aviso enviado: LISTO" || { echo "   ** el aviso fallo **"; echo "$R" | head -c 200; }
