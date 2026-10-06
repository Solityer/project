#!/usr/bin/bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
if [[ "$#" -ne 0 ]]; then
    printf 'usage: %s\n' "$0" >&2
    exit 2
fi

# The formal chain uses a fixed environment independent of login-shell setup.
unset BASH_ENV ENV CDPATH PYTHONHOME PYTHONPATH PYTHONSTARTUP
unset LD_PRELOAD LD_AUDIT LD_LIBRARY_PATH LD_RUN_PATH
unset DYLD_INSERT_LIBRARIES DYLD_LIBRARY_PATH
export HOME=/home/pzh
export PATH=/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export TZ=UTC
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
umask 077

launch_nonce="$(/usr/bin/python3 -c 'import secrets; print(secrets.token_hex(32))')"
COMPARISON_FORMAL_LAUNCH_NONCE="$launch_nonce" \
    /usr/bin/python3 "$script_dir/formal_runner.py" --_launch "$launch_nonce"
