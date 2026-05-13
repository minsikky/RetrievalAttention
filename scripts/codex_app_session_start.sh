#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  codex_app_session_start.sh --name NAME [options]

Options:
  --name NAME              Session name. Metadata file: .codex/app_sessions/NAME.json
  --listen URL             App-server listen URL. Default: ws://127.0.0.1:8788
  --cwd DIR                Thread cwd. Default: current directory.
  --approval POLICY        Default approval policy. Default: never
  --sandbox MODE           Default sandbox. Default: workspace-write
  --developer-instructions TEXT
EOF
}

name=""
listen_url="ws://127.0.0.1:8788"
cwd="$(pwd)"
approval="never"
sandbox="workspace-write"
developer_instructions=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --name) name="${2:-}"; shift 2 ;;
    --listen) listen_url="${2:-}"; shift 2 ;;
    --cwd) cwd="${2:-}"; shift 2 ;;
    --approval) approval="${2:-}"; shift 2 ;;
    --sandbox) sandbox="${2:-}"; shift 2 ;;
    --developer-instructions) developer_instructions="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${name}" ]]; then
  echo "--name is required" >&2
  exit 2
fi

mkdir -p .codex/app_sessions
server_log=".codex/app_sessions/${name}.server.log"
server_err=".codex/app_sessions/${name}.server.err"
server_env=".codex/app_sessions/${name}.server.env"
session_file=".codex/app_sessions/${name}.json"

if command -v setsid >/dev/null 2>&1; then
  setsid codex app-server --listen "${listen_url}" > "${server_log}" 2> "${server_err}" < /dev/null &
else
  nohup codex app-server --listen "${listen_url}" > "${server_log}" 2> "${server_err}" < /dev/null &
fi
server_pid=$!

python3 - <<'PY' "${listen_url}"
import sys, time, urllib.request
url = sys.argv[1]
ready = url.replace("ws://", "http://").replace("wss://", "https://")
if ready.endswith("/"):
    ready = ready[:-1]
ready = ready + "/readyz"
deadline = time.time() + 15.0
last = None
while time.time() < deadline:
    try:
        with urllib.request.urlopen(ready, timeout=1.0) as r:
            if 200 <= getattr(r, "status", 200) < 300:
                sys.exit(0)
    except Exception as exc:
        last = exc
        time.sleep(0.2)
raise SystemExit(f"app-server not ready: {last}")
PY

cat > "${server_env}" <<EOF
name=${name}
listen_url=${listen_url}
server_pid=${server_pid}
server_log=${server_log}
server_err=${server_err}
session_file=${session_file}
cwd=${cwd}
EOF

create_cmd=(
  node scripts/codex_app_session_ctl.mjs create
  --session-file "${session_file}"
  --url "${listen_url}"
  --cwd "${cwd}"
  --approval "${approval}"
  --sandbox "${sandbox}"
)
if [[ -n "${developer_instructions}" ]]; then
  create_cmd+=(--developer-instructions "${developer_instructions}")
fi

"${create_cmd[@]}"

echo "server_env=${server_env}"
echo "session_file=${session_file}"
