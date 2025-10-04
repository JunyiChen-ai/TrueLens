#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/.." && pwd)
cd "$root"

usage() {
  cat <<EOF
Usage: tools/data_toggle.sh <materialize|restore|status>

materialize  - If data is a symlink, back it up to data.symlink_backup,
               create a real data/ dir, and copy only vids subfolders:
               FakeSV/vids, FakeTT/vids, FVC/vids from the original target.
restore      - Remove temporary real data/ dir and restore the original symlink.
status       - Print current data entry status.

Notes:
- If DATA_ROOT is set, use it as the source of datasets.
- Otherwise, when materializing from a symlink, the symlink's target is used.
EOF
}

status() {
  if [ -L data ]; then
    echo "data is a symlink -> $(readlink -f data)"
  elif [ -d data ]; then
    echo "data is a real directory"
  else
    echo "data missing"
  fi
  [ -e data.symlink_backup ] && echo "backup exists: data.symlink_backup -> $(readlink -f data.symlink_backup || echo file)"
}

materialize() {
  local src_root="${DATA_ROOT:-}"
  if [ -L data ]; then
    echo "Backing up symlink 'data' to 'data.symlink_backup'"
    mv data data.symlink_backup
    if [ -z "${src_root}" ]; then
      src_root=$(readlink -f data.symlink_backup)
    fi
  fi
  if [ -z "${src_root}" ]; then
    echo "ERROR: DATA_ROOT not set and no symlink backup to infer source." >&2
    exit 1
  fi
  echo "Using source root: ${src_root}"
  mkdir -p data/FakeSV/vids data/FakeTT/vids data/FVC/vids
  for ds in FakeSV FakeTT FVC; do
    if [ -d "${src_root}/${ds}/vids" ]; then
      echo "Copying ${ds}/vids ..."
      cp -a "${src_root}/${ds}/vids/." "data/${ds}/vids/"
    else
      echo "WARNING: Missing ${src_root}/${ds}/vids" >&2
    fi
  done
  echo "Materialization done."
}

restore() {
  if [ -d data ] && [ ! -L data ]; then
    echo "Removing temporary real data directory..."
    rm -rf data
  fi
  if [ -e data.symlink_backup ]; then
    echo "Restoring symlink from data.symlink_backup -> data"
    mv data.symlink_backup data
  else
    echo "No data.symlink_backup found; nothing to restore." >&2
  fi
  status
}

cmd=${1:-}
case "$cmd" in
  materialize) materialize ;;
  restore) restore ;;
  status) status ;;
  *) usage; exit 1 ;;
esac

