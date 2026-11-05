# Quirky Android build scripts

This folder contains build-time helpers. It is not copied into the Android executable and does not need to be installed on the phone.

## `build-android.sh`

Run from the repository root or call it by path. It:

1. Checks that the original embedded `src/incbin/model.qnne` is present and matches the source archive's size/SHA-256.
2. Configures an Android ARM64 build using NDK r27c, API 24, static C++ runtime, and baseline ARMv8-A NEON.
3. Builds only the UCI engine target (`Quirky`), not `model_sampler`.
4. Verifies that the output is AArch64, that the linked executable contains signed-16-bit NEON ADD/SUB instructions, and that it does not need `libc++_shared.so`.
5. Runs a deterministic arithmetic-equivalence check for the AVX2/NEON NNUE lane operations.
6. Produces `dist/Quirky-Android-ARM64-NEON` and its SHA-256 file.

`verify-neon-arithmetic.py` is an arithmetic-level host check for signedness, lane ordering, saturation, wrapping and shifts. It complements (but does not replace) the final-binary NEON disassembly check.

Required environment variable: `ANDROID_NDK_HOME` (or `ANDROID_NDK_ROOT`) pointing to NDK r27c. Required host tools: CMake, Ninja, Python 3, `file`, `sha256sum` and the NDK LLVM tools.

The GitHub Actions workflow downloads NDK r27c and invokes this script. The `scripts` directory is build tooling only; keep it in the source repository for reproducible builds.
