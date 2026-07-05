#!/bin/bash
# Rotate the oMLX API key (leaked 2026-06 — see memory/saksham-remote-access.md),
# restart the server, re-open the teacher tunnel, and verify end-to-end.
# Run manually: bash scripts/rotate_omlx_and_open_tunnel.sh
# The new key is never printed; it lands in ~/.zshrc as OMLX_API_KEY.
set -euo pipefail

NEW_KEY=$(openssl rand -hex 24)
STAMP=$(date +%Y%m%d-%H%M%S)

cp ~/.omlx/settings.json ~/.omlx/settings.json.bak-"$STAMP"
cp ~/.zshrc ~/.zshrc.bak-"$STAMP"

OMLX_NEW_KEY="$NEW_KEY" python3 <<'PY'
import json, os, re

p = os.path.expanduser('~/.omlx/settings.json')
d = json.load(open(p))
d['auth']['api_key'] = os.environ['OMLX_NEW_KEY']
json.dump(d, open(p, 'w'), indent=2)
print('settings.json: api_key rotated')

z = os.path.expanduser('~/.zshrc')
s = open(z).read()
s2, n = re.subn(
    r'(OMLX_API_KEY=)["\x27]?[^"\x27\s]+["\x27]?',
    r'\g<1>"' + os.environ['OMLX_NEW_KEY'] + '"',
    s,
)
open(z, 'w').write(s2)
print(f'zshrc: {n} OMLX_API_KEY line(s) updated')
PY

echo "--- restarting oMLX server"
(cd /Applications/oMLX.app/Contents/Resources && ./Python/cpython-3.11/bin/python3 -m omlx.cli restart)

echo "--- verifying local auth (want 401 then 200)"
echo "old/blank key -> $(curl -s -o /dev/null -w '%{http_code}' --retry 15 --retry-connrefused --retry-delay 2 -H 'Authorization: Bearer stale-key' http://127.0.0.1:8000/v1/models)"
echo "new key       -> $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $NEW_KEY" http://127.0.0.1:8000/v1/models)"

echo "--- starting teacher tunnel (teacher.elcl.systems -> 127.0.0.1:8000)"
if ! pgrep -f "cloudflared.*teacher-config" >/dev/null; then
  nohup /opt/homebrew/bin/cloudflared tunnel --config ~/.cloudflared/teacher-config.yml run \
    > ~/.cloudflared/teacher-tunnel-"$STAMP".log 2>&1 &
  echo "tunnel started, pid $!"
else
  echo "tunnel already running"
fi

echo "--- verifying public endpoint (want 401 then 200; retries while tunnel connects)"
echo "no key  -> $(curl -s -o /dev/null -w '%{http_code}' --retry 10 --retry-delay 3 --retry-all-errors https://teacher.elcl.systems/v1/models)"
echo "new key -> $(curl -s -o /dev/null -w '%{http_code}' -H "Authorization: Bearer $NEW_KEY" https://teacher.elcl.systems/v1/models)"

echo
echo "DONE. Send Saksham (via DM, not the channel):"
echo "  OMLX_URL=https://teacher.elcl.systems/v1"
echo "  key: the new OMLX_API_KEY value in ~/.zshrc (grep OMLX_API_KEY ~/.zshrc)"
