#!/usr/bin/env bash
#
# dev-sync.sh — copy this working tree into the installed /watch plugin cache so
# local edits are picked up by Claude Code without publishing a release.
#
# The install path is resolved from ~/.claude/plugins/installed_plugins.json, so
# it follows version bumps automatically. Override it by passing a path as $1 or
# setting WATCH_INSTALL_PATH. Pass --dry-run to preview without writing.
#
# Usage:
#   ./dev-sync.sh                 # sync into the resolved install path
#   ./dev-sync.sh --dry-run       # show what would change
#   ./dev-sync.sh /some/other/dir # sync into an explicit path
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PLUGIN_KEY="watch@pachecoke-claude-video"
INSTALLED_JSON="${HOME}/.claude/plugins/installed_plugins.json"

# Resolve a working Python interpreter. On Windows, bare `python3` is the
# Microsoft Store alias that is on PATH but fails, so test each candidate by
# actually running it rather than trusting `command -v`.
PY=""
for cand in python3 python py; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c '' >/dev/null 2>&1; then
    PY="$cand"; break
  fi
done
if [[ -z "$PY" ]]; then
  echo "error: no working Python interpreter found (tried python3, python, py)" >&2
  exit 1
fi

DRY_RUN=()
DEST=""
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=(--dry-run --itemize-changes) ;;
    -*) echo "unknown flag: $arg" >&2; exit 2 ;;
    *) DEST="$arg" ;;
  esac
done

# Resolve the destination install path if not given explicitly.
if [[ -z "$DEST" ]]; then
  DEST="${WATCH_INSTALL_PATH:-}"
fi
if [[ -z "$DEST" ]]; then
  if [[ ! -f "$INSTALLED_JSON" ]]; then
    echo "error: $INSTALLED_JSON not found; pass an install path explicitly" >&2
    exit 1
  fi
  DEST="$(PLUGIN_KEY="$PLUGIN_KEY" "$PY" - "$INSTALLED_JSON" <<'PY'
import json, os, sys
data = json.load(open(sys.argv[1]))
key = os.environ["PLUGIN_KEY"]
records = data.get("plugins", {}).get(key, [])
# Prefer a record whose installPath actually exists on disk.
paths = [r.get("installPath") for r in records if r.get("installPath")]
for p in paths:
    if os.path.isdir(p):
        print(p); break
else:
    print(paths[0] if paths else "")
PY
)"
fi

if [[ -z "$DEST" ]]; then
  echo "error: could not resolve an install path for '$PLUGIN_KEY'" >&2
  echo "       install the plugin first, or pass a path: scripts/dev-sync.sh /path/to/install" >&2
  exit 1
fi
if [[ ! -d "$DEST" ]]; then
  echo "error: install path does not exist: $DEST" >&2
  exit 1
fi

echo "source: $REPO_ROOT"
echo "dest:   $DEST"
echo

# Mirror shipped files only. Dev-only artifacts and runtime state are excluded.
# No delete: the cache holds runtime state (.in_use/) we must not touch.
if command -v rsync >/dev/null 2>&1; then
  rsync -a ${DRY_RUN[@]+"${DRY_RUN[@]}"} \
    --exclude '.git/' \
    --exclude '.venv/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '.DS_Store' \
    --exclude '.in_use/' \
    --exclude 'tests/' \
    --exclude 'docs/' \
    --exclude 'dist/' \
    --exclude 'V2_*.md' \
    --exclude 'dev-sync.sh' \
    "$REPO_ROOT/" "$DEST/"
else
  # No rsync (e.g. Git Bash on Windows): copy with Python, same excludes, no delete.
  DRY="0"
  if [[ ${#DRY_RUN[@]} -gt 0 ]]; then DRY="1"; fi
  SRC="$REPO_ROOT" DST="$DEST" DRY="$DRY" "$PY" - <<'PYCOPY'
import fnmatch, os, shutil
src, dst, dry = os.environ["SRC"], os.environ["DST"], os.environ["DRY"] == "1"
patterns = ['.git', '.venv', '.pytest_cache', '__pycache__', '*.pyc',
            '.DS_Store', '.in_use', 'tests', 'docs', 'dist', 'V2_*.md', 'dev-sync.sh']
def excluded(name):
    return any(fnmatch.fnmatch(name, p) for p in patterns)
for root, dirs, files in os.walk(src):
    dirs[:] = [d for d in dirs if not excluded(d)]
    rel = os.path.relpath(root, src)
    for f in files:
        if excluded(f):
            continue
        r = f if rel == "." else os.path.join(rel, f)
        d = os.path.join(dst, r)
        if dry:
            print("would copy", r)
        else:
            os.makedirs(os.path.dirname(d), exist_ok=True)
            shutil.copy2(os.path.join(root, f), d)
PYCOPY
fi

echo "done."
