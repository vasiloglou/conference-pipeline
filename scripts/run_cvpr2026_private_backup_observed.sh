#!/usr/bin/env sh
# This entrypoint is deliberately an observation boundary, never a backup runner.
set -eu
set +x

if [ "$#" -ne 2 ] || [ "$1" != "--observation-only" ] || [ "$2" != "--expect-absent" ]; then
    exit 64
fi

case "$0" in
    */*) SCRIPT_DIRECTORY=${0%/*} ;;
    *) SCRIPT_DIRECTORY=. ;;
esac
CDPATH= cd -- "$SCRIPT_DIRECTORY/.."
REPO_ROOT=$PWD

[ -f "$REPO_ROOT/.env" ] || exit 65

unset GDRIVE_BACKUP_ROOT
. "$REPO_ROOT/.env"

if [ -n "${GDRIVE_BACKUP_ROOT:-}" ]; then
    backup_root_present=true
else
    backup_root_present=false
fi
unset GDRIVE_BACKUP_ROOT

exec "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/src/conference_pipeline/backup_preflight_observation.py" \
    --observation-only \
    --expect-absent \
    --backup-root-present "$backup_root_present"
