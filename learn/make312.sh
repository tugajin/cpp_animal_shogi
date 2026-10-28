#!/bin/sh
# gamelibs を Python 3.12 (.venv) 向けにビルドする。
# システムの Python 3.12 にはヘッダ (Python.h) が無く apt でも入れられないため、
# uv の standalone Python 3.12 のヘッダを借りてコンパイルする。
# 拡張モジュールは libpython にリンクしないので、実行時は .venv の 3.12.11 でそのまま動く
# (同じ 3.12 系ならパッチ版違いのヘッダでビルドしても ABI は互換)。
# 実行場所: learn/
set -e
cd "$(dirname "$0")"

VENV_PY=../.venv/bin/python

# uv が無ければ .venv に入れる
if [ ! -x ../.venv/bin/uv ]; then
    "$VENV_PY" -m pip install uv
fi

# standalone Python 3.12 を取得し、そのヘッダ位置を特定する
# 注意: `uv python find` は有効化中の .venv を返してしまうため、インストール先を直接探す
../.venv/bin/uv python install 3.12
UV_BIN=$(ls -d ~/.local/share/uv/python/cpython-3.12.*/bin/python3.12 | sort -V | tail -n 1)
UV_ROOT=$(dirname "$(dirname "$UV_BIN")")
UV_INC="$UV_ROOT/include/python3.12"
echo "uv python: $UV_BIN"
ls "$UV_INC/Python.h"

PYBIND_INC=$("$VENV_PY" -c "import pybind11; print(pybind11.get_include())")
EXT_SUFFIX=$("$VENV_PY" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -O3 -Wall -shared -std=c++17 -fPIC -DNDEBUG \
    -I"$PYBIND_INC" -I"$UV_INC" \
    gamelibs.cpp -o "gamelibs$EXT_SUFFIX"

ls -l gamelibs*.so