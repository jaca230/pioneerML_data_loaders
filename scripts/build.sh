#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
BASE_DIR=$(realpath "${SCRIPT_DIR}/..")
BUILD_DIR="${BASE_DIR}/build"
CLEANUP_SCRIPT="${SCRIPT_DIR}/cleanup.sh"

OVERWRITE=false
JOBS_ARG="-j"  # Use all processors by default

show_help() {
    echo "Usage: ./build.sh [OPTIONS]"
    echo
    echo "Options:" 
    echo "  -o, --overwrite           Remove existing build directory before building"
    echo "  -j, --jobs <number>       Specify number of processors (default: all available)"
    echo "  -h, --help                Display this help message"
}

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -o|--overwrite)
            OVERWRITE=true
            shift
            ;;
        -j|--jobs)
            if [[ -n "${2:-}" && "$2" != -* ]]; then
                JOBS_ARG="-j$2"
                shift 2
            else
                JOBS_ARG="-j"
                shift
            fi
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "[build.sh, ERROR] Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

if [ "$OVERWRITE" = true ]; then
    echo "[build.sh] Cleaning previous build with: $CLEANUP_SCRIPT"
    if [ -x "$CLEANUP_SCRIPT" ]; then
        "$CLEANUP_SCRIPT"
    else
        rm -rf "$BUILD_DIR"
    fi
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

PYBIND11_CMAKE_DIR=""
if command -v python &>/dev/null; then
    if PYBIND11_CMAKE_DIR=$(python -m pybind11 --cmakedir 2>/dev/null); then
        echo "[build.sh] Detected pybind11 CMake dir: $PYBIND11_CMAKE_DIR"
    else
        echo "[build.sh, ERROR] pybind11 not found in current Python environment."
        echo "Install with: pip install pybind11  (or add to your env setup) "
        exit 1
    fi
else
    echo "[build.sh, ERROR] 'python' not found on PATH"
    exit 1
fi

echo "[build.sh] Running cmake in: $BUILD_DIR"
cmake -Dpybind11_DIR="$PYBIND11_CMAKE_DIR" "$BASE_DIR"

echo "[build.sh] Building with make $JOBS_ARG"
make $JOBS_ARG

echo "[build.sh] Build complete."
echo "[build.sh] Binaries: $BUILD_DIR/bin (if any)"
echo "[build.sh] Libraries: $BUILD_DIR/lib"
