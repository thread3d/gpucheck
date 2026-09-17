#!/bin/bash
# Builds gpucheck. Uses the Command Line Tools toolchain directly, so no Xcode
# project or SwiftPM resolution is required. Metal shaders are compiled at
# runtime by the app, so the `metal` command line tool is not needed either.
set -euo pipefail

cd "$(dirname "$0")"

BUILD_DIR=".build"
MODULE_CACHE="$BUILD_DIR/modulecache"
mkdir -p "$MODULE_CACHE"

# The DSH file sandbox may block the default module cache under /var/folders,
# so the cache is kept inside the project directory.
swiftc \
  -O \
  -swift-version 5 \
  -module-cache-path "$MODULE_CACHE" \
  -framework Metal \
  -framework MetalPerformanceShaders \
  -framework OpenCL \
  -o "$BUILD_DIR/gpucheck" \
  Sources/*.swift

echo "built $BUILD_DIR/gpucheck"
