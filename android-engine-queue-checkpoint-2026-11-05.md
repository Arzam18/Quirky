# Android ARM64 Engine Queue Checkpoint — 2026-11-05 (Quirky active)

| Order | Engine | Status | Notes |
|---:|---|---|---|
| 1 | Askaig | Successful | Android ARM64 NEON build and device UCI runtime confirmed. |
| 2 | Chessnix | Successful | Android ARM64 build and device runtime confirmed. |
| 3 | NARC Genesys | Successful | Android runtime confirmed; approximately 1M NPS. Signed-int16 NNUE path; no Dotprod kernel claimed. |
| 4 | Pea | Successful | Android build and runtime confirmed. |
| 5 | RukChess | Successful | Rebased Android ARM64 NEON build and device UCI runtime confirmed. |
| 6 | Solo | Successful | Android ARM64 NEON build and device UCI runtime confirmed. |
| 7 | Peacekeeper | Build successful; A/B device test pending | User confirmed the latest Android build succeeds. User is comparing it with the older strong build for fixed-time strength and long-game segfault stability. |
| 8 | **Quirky** | **Active — recovery package prepared; GitHub Actions pending** | Original and Android-port source compared. Preserves model and NEON implementation; fixes larger stack and recursive per-ply buffer bounds. |
| 9 | Kociolek | Queued | Source archive `NNUE-Chess-engine-main.zip`; audit network files when active. |

## Quirky source comparison

- Original archive: `Quirky-main(original).zip`.
- Android port archive: `Quirky-main (android-port).zip`.
- A later `Quirky-ARM64-NEON-source-fixed.zip` has the same engine code as the supplied Android-port archive; it adds documentation.
- Original and Android-port `src/incbin/model.qnne` are byte-identical: 3,496,154 bytes, SHA-256 `4ef706e9479b161ab0d7a461872c9e3d7b0be06211c99ac08e8b66b26ac1ce50`.
- Android port changes CMake architecture flags, adds the AArch64 NEON NNUE layer and larger-stack thread helper, routes the launcher and search worker through the helper, heap-allocates `Searcher`, and selects portable bit operations on ARM. The original search heuristics and NNUE model data are not changed.

## Quirky crash analysis and recovery

- The Android port already used `std::make_unique<Searcher>`, but `Searcher::Searcher` still created a redundant `HistoryTable()` temporary on the stack in `global_context_.history_table = HistoryTable();`. `GlobalContext` constructs that member in place before the constructor body, and `HistoryTable::HistoryTable()` initializes all of its arrays. Removing the redundant assignment preserves the same fully initialized history state while reducing GCC 14 `-fstack-usage` for `Searcher::Searcher` from 2,800,688 bytes to 48 bytes.
- The supplied Android port's custom pthread stack was 16 MiB. Recovery raises it to 32 MiB for both launcher and search threads, while keeping desktop/Windows builds on `std::thread`.
- A second latent bounds risk was found: `Position::MAX_BUFFER_SIZE` was 256 while singular extensions and qsearch can add plies beyond the nominal search depth; `local_context_` also had 255 entries indexed 0..254 without an entry guard. The recovery increases the evaluator state history buffer to 512, retains the original nominal search depth cap of 127, guards before context indexing, guards qsearch when the history buffer is full, and guards both `Position::MakeMove` overloads before incrementing the state pointer.
- This is a separate protection from stack sizing. It addresses a plausible intermittent long-game/tournament crash, but actual device validation is still required to confirm whether it was the observed failure.
- The NEON NNUE operations were reviewed against the AVX2 source. A 5,000-round deterministic arithmetic emulation passes for signedness, pairwise sums, 16-bit wrapping, clipping, shifts and precise-linear accumulation. A separate AArch64 Clang test emits `add vN.8h`, `sub vN.8h`, and `smull`. No Dotprod claim is made.

## Quirky build configuration

- NDK r27c `27.2.12479018`, Android API 24, AArch64.
- ARMv8-A baseline NEON (`-march=armv8-a+simd`), C++20.
- Embedded model stays in `src/incbin/model.qnne`; no network replacement/download.
- Builds only the `Quirky` UCI executable; does not build `model_sampler`.
- `workflow_dispatch` only.
- Output includes ELF AArch64, NEON ADD/SUB verification and no `libc++_shared.so` check.

## Local validation status

Passed: source-archive diff, model hash comparison, stack-usage inspection, code review, 5,000-round NEON arithmetic emulation, isolated AArch64 intrinsic cross-compilation, shell syntax, YAML parse, host compilation of the modified C++ translation units, and host thread-wrapper smoke test.

Not yet verified: complete Android NDK build, final Android executable disassembly, Galaxy A16 UCI runtime, long-game soak test, and fixed-node/fixed-time strength comparison. Do not mark Quirky successful until GitHub Actions and device tests pass.

## Next sequence

1. Upload the Quirky modified-files-only ZIP to the current Android-port repository.
2. Run `Build Quirky for Android ARM64 NEON`.
3. Test `uci`, `isready`, and a bounded search on the Galaxy A16.
4. Run long-game/tournament soak tests against the previous Android binary. Compare fixed-node search before comparing fixed-time match results.
5. After Quirky passes, move to Kociolek.
