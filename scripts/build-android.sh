#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NDK="${ANDROID_NDK_HOME:-${ANDROID_NDK_ROOT:-${ANDROID_NDK_LATEST_HOME:-}}}"
if [[ -z "$NDK" || ! -d "$NDK" ]]; then
    echo "ERROR: ANDROID_NDK_HOME (or ANDROID_NDK_ROOT) must point to Android NDK r27c." >&2
    exit 1
fi

HOST_TAG="linux-x86_64"
TOOLCHAIN="$NDK/toolchains/llvm/prebuilt/$HOST_TAG"
CXX="$TOOLCHAIN/bin/aarch64-linux-android24-clang++"
READELF="$TOOLCHAIN/bin/llvm-readelf"
OBJDUMP="$TOOLCHAIN/bin/llvm-objdump"
MODEL="$ROOT/src/incbin/model.qnne"
BUILD="$ROOT/build-android-arm64"
DIST="$ROOT/dist"
OUTPUT="$DIST/Quirky-Android-ARM64-NEON"
EXPECTED_MODEL_SIZE=3496154
EXPECTED_MODEL_SHA256="4ef706e9479b161ab0d7a461872c9e3d7b0be06211c99ac08e8b66b26ac1ce50"

for tool in "$CXX" "$READELF" "$OBJDUMP"; do
    if [[ ! -x "$tool" ]]; then
        echo "ERROR: required NDK tool not found: $tool" >&2
        exit 1
    fi
done
command -v cmake >/dev/null 2>&1 || { echo "ERROR: cmake is required." >&2; exit 1; }
command -v ninja >/dev/null 2>&1 || { echo "ERROR: ninja is required." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 is required." >&2; exit 1; }
command -v file >/dev/null 2>&1 || { echo "ERROR: file is required." >&2; exit 1; }
command -v sha256sum >/dev/null 2>&1 || { echo "ERROR: sha256sum is required." >&2; exit 1; }

if [[ ! -f "$MODEL" ]]; then
    echo "ERROR: embedded Quirky NNUE model is missing: $MODEL" >&2
    exit 1
fi
MODEL_SIZE="$(wc -c < "$MODEL" | tr -d ' ')"
MODEL_SHA256="$(sha256sum "$MODEL" | awk '{print $1}')"
if [[ "$MODEL_SIZE" != "$EXPECTED_MODEL_SIZE" || "$MODEL_SHA256" != "$EXPECTED_MODEL_SHA256" ]]; then
    echo "ERROR: src/incbin/model.qnne does not match the supplied original/Android-port model." >&2
    echo "Expected size: $EXPECTED_MODEL_SIZE" >&2
    echo "Actual size:   $MODEL_SIZE" >&2
    echo "Expected SHA:  $EXPECTED_MODEL_SHA256" >&2
    echo "Actual SHA:    $MODEL_SHA256" >&2
    exit 1
fi

echo "=== Quirky embedded NNUE ==="
echo "Size:   $MODEL_SIZE bytes"
echo "SHA256: $MODEL_SHA256"

echo "=== Verify NEON arithmetic mapping ==="
python3 "$ROOT/scripts/verify-neon-arithmetic.py"

rm -rf "$BUILD" "$DIST"
mkdir -p "$DIST"

CXX_FLAGS="-march=armv8-a+simd -fno-strict-aliasing"
RELEASE_FLAGS="-O3 -DNDEBUG -march=armv8-a+simd -fno-strict-aliasing"

cmake -S "$ROOT" -B "$BUILD" -G Ninja \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-24 \
    -DANDROID_STL=c++_static \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_FLAGS="$CXX_FLAGS" \
    -DCMAKE_CXX_FLAGS_RELEASE="$RELEASE_FLAGS" \
    -DNO_AVX2=ON

cmake --build "$BUILD" --target Quirky --parallel "${BUILD_JOBS:-2}"

BIN="$BUILD/Quirky"
if [[ ! -s "$BIN" ]]; then
    echo "ERROR: CMake did not produce the Quirky executable." >&2
    exit 1
fi

"$READELF" -h "$BIN" > "$DIST/elf-header.txt"
if ! grep -Eiq 'Machine:[[:space:]]+AArch64' "$DIST/elf-header.txt"; then
    echo "ERROR: Quirky output is not an AArch64 ELF executable." >&2
    cat "$DIST/elf-header.txt" >&2
    exit 1
fi

"$OBJDUMP" -d "$BIN" > "$DIST/disassembly.txt"
if ! grep -Eiq '[[:space:]]add[[:space:]]+v[0-9]+\.8h' "$DIST/disassembly.txt"; then
    echo "ERROR: final executable has no expected NEON signed-16-bit vector ADD." >&2
    exit 1
fi
if ! grep -Eiq '[[:space:]]sub[[:space:]]+v[0-9]+\.8h' "$DIST/disassembly.txt"; then
    echo "ERROR: final executable has no expected NEON signed-16-bit vector SUB." >&2
    exit 1
fi

echo "PASS: final executable contains NEON 8x16-bit ADD and SUB instructions."

"$READELF" -d "$BIN" > "$DIST/dynamic-section.txt"
if grep -Fq 'libc++_shared.so' "$DIST/dynamic-section.txt"; then
    echo "ERROR: executable depends on libc++_shared.so." >&2
    exit 1
fi

install -m 755 "$BIN" "$OUTPUT"
sha256sum "$OUTPUT" > "$OUTPUT.sha256"
cp "$MODEL" "$DIST/model.qnne.source-check"  # audit copy; not part of the engine artifact

file "$OUTPUT"
cat "$OUTPUT.sha256"
echo "PASS: Android ARM64 NEON executable created: $OUTPUT"
