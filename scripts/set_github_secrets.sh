#!/usr/bin/env bash
# Set the six GitHub Actions secrets that .github/workflows/operate.yml needs,
# reading every value from the local files that already hold it:
#   .env               MLFLOW_TRACKING_URI / _USERNAME / _PASSWORD, DATABASE_URL
#   .dvc/config.local  the Cloudflare R2 key pair (AWS_ACCESS_KEY_ID / _SECRET_ACCESS_KEY)
# Values are piped straight into `gh secret set`; nothing is printed.
#
# Usage:
#   scripts/set_github_secrets.sh            set the secrets, then list them
#   scripts/set_github_secrets.sh --verify   ...and trigger the read-only checks
#
# Only the standalone repo needs them: the fork's default branch is the NVIDIA
# mirror, which has no operate.yml, and scheduled workflows run from there.
set -euo pipefail

REPO="${REPO:-oluwadunni1/fraud-detection-mlops}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v gh >/dev/null || { echo "gh (GitHub CLI) is not installed" >&2; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh is not logged in: run 'gh auth login'" >&2; exit 1; }
[ -f .env ] || { echo "missing $ROOT/.env" >&2; exit 1; }
[ -f .dvc/config.local ] || { echo "missing $ROOT/.dvc/config.local" >&2; exit 1; }
[ -x .venv/bin/python ] || { echo "missing .venv (needed to parse .env safely)" >&2; exit 1; }

# Read one key from .env with python-dotenv -- never `source .env` (CLAUDE.md decision 10).
env_value() {
  .venv/bin/python -c "
import sys
from dotenv import dotenv_values
sys.stdout.write(dotenv_values('.env').get('$1') or '')"
}

# Read one key from .dvc/config.local (INI: `    access_key_id = ...`), quotes stripped.
dvc_value() {
  awk -F'=' -v k="$1" '
    { key=$1; gsub(/[ \t]/, "", key) }
    key==k { v=substr($0, index($0, "=")+1); gsub(/^[ \t"]+|[ \t"]+$/, "", v); printf "%s", v; exit }
  ' .dvc/config.local
}

set_secret() {   # name value-producing-command...
  local name="$1"; shift
  local value; value="$("$@")"
  if [ -z "$value" ]; then
    echo "  $name: EMPTY in the local file -- not set" >&2
    return 1
  fi
  printf '%s' "$value" | gh secret set "$name" -R "$REPO"
  echo "  $name: set"
}

echo "Setting Actions secrets on $REPO"
failed=0
for k in MLFLOW_TRACKING_URI MLFLOW_TRACKING_USERNAME MLFLOW_TRACKING_PASSWORD DATABASE_URL; do
  set_secret "$k" env_value "$k" || failed=1
done
set_secret AWS_ACCESS_KEY_ID     dvc_value access_key_id     || failed=1
set_secret AWS_SECRET_ACCESS_KEY dvc_value secret_access_key || failed=1

echo
gh secret list -R "$REPO"
[ "$failed" -eq 0 ] || { echo "some secrets were not set (see above)" >&2; exit 1; }

if [ "${1:-}" = "--verify" ]; then
  echo
  echo "Triggering read-only checks (none of them moves an alias):"
  gh workflow run operate.yml -R "$REPO" -f action=keepalive && echo "  keepalive  -> checks DATABASE_URL"
  gh workflow run operate.yml -R "$REPO" -f action=promote   && echo "  promote    -> dry run; checks R2 + DagsHub (expect 'no gain': v3 is already champion)"
  echo
  echo "Watch them with:  gh run list -R $REPO --workflow operate.yml --limit 3"
  echo "Inspect a run:    gh run view <id> -R $REPO --log-failed"
fi
