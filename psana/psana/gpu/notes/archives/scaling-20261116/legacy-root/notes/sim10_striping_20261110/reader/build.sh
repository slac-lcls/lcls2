#!/bin/bash
set -euo pipefail
source_dir=$(cd -- "$(dirname -- "$0")" && pwd)
build_dir=${1:?new build directory}
mkdir "$build_dir"
g++ -O3 -std=c++17 -pthread -Wall -Wextra -Werror "$source_dir/reader.cpp" -o "$build_dir/reader"
g++ -O3 -std=c++17 -pthread -Wall -Wextra -Werror -DREADER_TESTING "$source_dir/reader.cpp" -o "$build_dir/reader-test"
g++ --version > "$build_dir/compiler.txt"
