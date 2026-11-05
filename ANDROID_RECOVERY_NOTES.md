# Quirky Android ARM64 NEON — stack recovery and source comparison

## Objective

Keep the original Quirky search and NNUE strength while eliminating stack exhaustion and rare long-search out-of-bounds failures. No pruning, reduction, move-ordering, evaluation constants, or NNUE weights were changed.

## Source comparison

Compared `Quirky-main(original).zip` with `Quirky-main (android-port).zip` and the later `Quirky-ARM64-NEON-source-fixed.zip`.

- The original and Android-port `src/incbin/model.qnne` are byte-identical: 3,496,154 bytes, SHA-256 `4ef706e9479b161ab0d7a461872c9e3d7b0be06211c99ac08e8b66b26ac1ce50`.
- The Android-port adds `src/eval/layers_neon.h`, switches `layers.h` to NEON on AArch64, and keeps the original AVX2 implementation for non-ARM builds.
- The Android-port adds `src/util/large_stack_thread.h`, uses a larger-stack launcher/search thread, and moves `Searcher` from a local stack object to `std::make_unique`.
- `src/util/bit.h` bypasses x86 BMI2 intrinsics on AArch64 and uses the existing portable bit-deposit/extract implementation.
- `src/search/searcher/searcher.cpp` adds the missing `<algorithm>` include. The other search/evaluation control logic remains unchanged.
- The CMake target detection accepts both AArch64 processor names and Android's `arm64-v8a` ABI, so an Android toolchain cannot accidentally receive x86 SSE/AVX compiler flags.
- The later `Quirky-ARM64-NEON-source-fixed.zip` contains the same engine code as the supplied Android-port archive; it adds documentation only.

## Stack-crash root cause and correction

The Android port already moved `Searcher` itself to the heap with `std::make_unique`, but a second, less visible stack allocation remained in the constructor:

```cpp
// Removed: this constructs a 2.8 MiB temporary on the worker stack.
global_context_.history_table = HistoryTable();
```

`GlobalContext::history_table` is already default-constructed in place before the constructor body runs. `HistoryTable::HistoryTable()` initializes all of its members: killer moves, counter moves, simple history, capture history, and continuation history. The assignment therefore performed a redundant second initialization by constructing a full-size temporary and copying it into the already initialized member.

GCC 14 `-fstack-usage` measured the `Searcher::Searcher` frame at **2,800,688 bytes** before removing that line. After removal, the same function measures **48 bytes**. The history table is still fully initialized once, with the same contents; only the redundant temporary and copy have been removed. This preserves search heuristics and strength while eliminating a major stack spike at the start of each search.

The existing Android change that heap-allocates `Searcher` is retained. This also avoids placing the large `Searcher` object itself on the launcher stack.

## Additional intermittent crash risk found

The existing Android port sized:

- `Position::MAX_BUFFER_SIZE` at 256 evaluator states; and
- `Searcher::local_context_` at 255 entries, indexed 0 through 254.

But `Searcher::Search()` can receive deeper `idepth` values because singular extensions increase the actual ply count, and quiescence search has its own recursive check/evasion continuation. There was no terminal guard before `local_context_[idepth]` or before `Position::buffer_[++buffer_head_]`. A rare long/tactical search could therefore index beyond these fixed arrays. This is a separate risk from stack exhaustion and can explain intermittent tournament crashes.

The recovery changes:

1. Increase the evaluator-state history buffer from 256 to 512 slots. Keep the original nominal maximum search depth at 127; this is safety headroom, not a strength/depth setting change.
2. Stop the regular search safely before `idepth` reaches the local-context sentinel.
3. Stop quiescence recursion safely when the evaluator-state history buffer is exhausted. This also puts a hard ceiling on repeated check/evasion qsearch chains.
4. Guard both `Position::MakeMove` overloads before incrementing the buffer head.
5. Keep the `Searcher` on the heap.
6. Increase the Android pthread stack reservation to 32 MiB for both launcher and search threads. The stack is virtual address space and is committed as touched; it is not a 32 MiB physical allocation per thread.
7. Keep non-Android platforms on the standard `std::thread` implementation so the Android stack helper does not regress Windows or desktop builds.

The depth/buffer guard only affects pathological branches beyond the engine's previous fixed-array capacity; normal search, heuristics and evaluation are unchanged.

## NEON arithmetic review

The NEON layer preserves the original model dimensions/layout and signed/unsigned interpretation:

- Feature accumulator add/sub use wrapping signed-16-bit lane arithmetic, matching AVX2 `_mm256_add_epi16` / `_mm256_sub_epi16`.
- The sparse first linear layer implements the original unsigned activation × signed int8 weight pairwise dot. Activations are clipped to 0..127, so treating them as signed bytes has identical values. Pair sums stay within int16 for one `maddubs` pair; `Addx2` uses wrapping int16 addition exactly like `_mm256_add_epi16`.
- `vpaddlq_s16` reproduces the pairwise signed-16-to-int32 accumulation of `_mm256_madd_epi16`.
- The precise linear layer uses signed int16 widening products and sums them in int32 lanes.
- ReLU shifts, clamps and saturating narrow operations match the original AVX2 pack/clamp sequence.

The AArch64 build verifies that the final executable contains 8x16-bit NEON ADD and SUB instructions. No Dotprod/SDOT kernel is claimed; this model uses signed-16 and int8-layer arithmetic, and the port uses NEON only.

## Network integrity

`src/incbin/model.qnne` is embedded by the existing CMake `embed.py` step. The build checks its original archive size and hash before compilation. It is never downloaded, replaced, or regenerated from another model.

- Size: 3,496,154 bytes
- SHA-256: `4ef706e9479b161ab0d7a461872c9e3d7b0be06211c99ac08e8b66b26ac1ce50`

## Android build

- Architecture: AArch64 / ARM64
- ISA: baseline ARMv8-A NEON (`-march=armv8-a+simd`)
- NDK: r27c 27.2.12479018
- Android API: 24
- C++ standard: C++20
- Runtime: static C++ runtime; no `libc++_shared.so`
- Output: one UCI executable only
- Workflow: `.github/workflows/android.yml`, manual `workflow_dispatch`

## Verification status

Completed locally:

- Archive comparison, network hash comparison, stack-usage inspection, and fixed-array bounds review.
- GCC 14 `-fstack-usage`: `Searcher::Searcher` reduced from 2,800,688 bytes to 48 bytes after removing the redundant `HistoryTable()` temporary; the `HistoryTable` member still initializes itself in place.
- 5,000 deterministic arithmetic-level AVX2/NEON equivalence rounds for activation signedness, pair sums, wrapping, clipping and shifts.
- Modified C++ translation units compiled with GCC C++20, `-Wall -Wextra -Wpedantic -Werror`; host `LargeStackThread` fallback smoke test passed.
- Modified source paths and CMake target reviewed.
- Shell script syntax and workflow YAML parse passed.

Still required:

- Actual NDK r27c GitHub Actions build.
- Galaxy A16 UCI runtime test.
- Long-game/tournament soak test with the same time control and NNUE.
- Fixed-node A/B versus the original Android binary to confirm the NEON output and move decisions remain equivalent on-device.

Do not mark Quirky successful until the NDK build and phone tests pass.
