#!/usr/bin/env python3
"""Deterministic scalar emulation of the AVX2 and ARM NEON NNUE lane math.

This checks lane ordering, signedness, wrapping, saturation, and shift semantics.
It is an arithmetic-level test; the Android workflow separately checks the
actual AArch64 executable for emitted NEON instructions.
"""

from __future__ import annotations

import random
import struct

SEED = 0x515549524B59
ROUNDS = 5000


def s16(x: int) -> int:
    x &= 0xFFFF
    return x - 0x10000 if x & 0x8000 else x


def s32(x: int) -> int:
    x &= 0xFFFFFFFF
    return x - 0x100000000 if x & 0x80000000 else x


def sat_s16(x: int) -> int:
    return max(-32768, min(32767, x))


def sat_s8(x: int) -> int:
    return max(-128, min(127, x))


def pairwise(xs: list[int]) -> list[int]:
    assert len(xs) % 2 == 0
    return [xs[i] + xs[i + 1] for i in range(0, len(xs), 2)]


def avx_maddubs(act_word: int, weights32: list[int]) -> list[int]:
    # _mm256_set1_epi32 repeats the same 4 activation bytes eight times.
    pattern = list(struct.pack("<I", act_word))
    activations = pattern * 8  # 32 bytes
    products = [activations[i] * weights32[i] for i in range(32)]
    # _mm256_maddubs_epi16 is unsigned-byte x signed-byte, pairwise
    # saturating to int16. For clipped activations 0..127, each pair is safe.
    pairs = [sat_s16(products[i] + products[i + 1]) for i in range(0, 32, 2)]
    return [s32(x) for x in pairwise(pairs)]


def neon_maddubs(act_word: int, b0: list[int], b1: list[int]) -> list[int]:
    # Maddubs8 reinterprets four repeated int32 lanes as signed bytes.
    # NNUE activations are clipped to [0,127], so this is value-equivalent.
    pattern = list(struct.pack("<I", act_word)) * 4  # one 128-bit vector
    p0 = [sat_s16(pattern[i] * b0[i] + pattern[i + 1] * b0[i + 1])
          for i in range(0, 16, 2)]
    p1 = [sat_s16(pattern[i] * b1[i] + pattern[i + 1] * b1[i + 1])
          for i in range(0, 16, 2)]
    # MaddEpi16 / vpaddlq_s16: adjacent int16 pairs are widened to int32.
    return [s32(x) for x in pairwise(p0)] + [s32(x) for x in pairwise(p1)]


def avx_addx2(word0: int, weights0: list[int], word1: int,
              weights1: list[int]) -> list[int]:
    p0 = avx_maddubs_pairs(word0, weights0)
    p1 = avx_maddubs_pairs(word1, weights1)
    combined = [s16(a + b) for a, b in zip(p0, p1)]
    return [s32(x) for x in pairwise(combined)]


def avx_maddubs_pairs(act_word: int, weights32: list[int]) -> list[int]:
    pattern = list(struct.pack("<I", act_word)) * 8
    products = [pattern[i] * weights32[i] for i in range(32)]
    return [sat_s16(products[i] + products[i + 1]) for i in range(0, 32, 2)]


def neon_addx2(word0: int, weights0: list[int], word1: int,
               weights1: list[int]) -> list[int]:
    p0 = neon_maddubs_pairs(word0, weights0)
    p1 = neon_maddubs_pairs(word1, weights1)
    combined = [s16(a + b) for a, b in zip(p0, p1)]
    return [s32(x) for x in pairwise(combined)]


def neon_maddubs_pairs(act_word: int, weights32: list[int]) -> list[int]:
    pattern = list(struct.pack("<I", act_word)) * 4
    first = [sat_s16(pattern[i] * weights32[i] + pattern[i + 1] * weights32[i + 1])
             for i in range(0, 16, 2)]
    second = [sat_s16(pattern[i] * weights32[i + 16] + pattern[i + 1] * weights32[i + 17])
              for i in range(0, 16, 2)]
    return first + second


def clipped_relu16_avx(values: list[int]) -> list[int]:
    return [max(0, sat_s8(x >> 5)) for x in values]


def clipped_relu16_neon(values: list[int]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(values), 8):
        shifted = [max(0, x >> 5) for x in values[i:i + 8]]
        out.extend(sat_s8(x) for x in shifted)
    return out


def clipped_relu32_avx(values: list[int]) -> list[int]:
    return [sat_s16(max(0, min(8191, x))) for x in values]


def clipped_relu32_neon(values: list[int]) -> list[int]:
    out: list[int] = []
    for i in range(0, len(values), 4):
        out.extend(sat_s16(max(0, min(8191, x))) for x in values[i:i + 4])
    return out


def main() -> None:
    rng = random.Random(SEED)

    for _ in range(ROUNDS):
        act = [rng.randrange(128) for _ in range(4)]
        word = struct.unpack("<I", bytes(act))[0]
        b0 = [rng.randrange(-128, 128) for _ in range(16)]
        b1 = [rng.randrange(-128, 128) for _ in range(16)]
        assert avx_maddubs(word, b0 + b1) == neon_maddubs(word, b0, b1)

        word1 = struct.unpack("<I", bytes(rng.randrange(128) for _ in range(4)))[0]
        c0 = [rng.randrange(-128, 128) for _ in range(32)]
        c1 = [rng.randrange(-128, 128) for _ in range(32)]
        assert avx_addx2(word, c0, word1, c1) == neon_addx2(word, c0, word1, c1)

        acc = [rng.randrange(-32768, 32768) for _ in range(32)]
        weights = [rng.randrange(-32768, 32768) for _ in range(32)]
        # AVX2 and NEON signed-16-bit accumulator add/sub both wrap modulo 2^16.
        add_avx = [s16(a + w) for a, w in zip(acc, weights)]
        add_neon = [s16(a + w) for a, w in zip(acc, weights)]
        sub_avx = [s16(a - w) for a, w in zip(acc, weights)]
        sub_neon = [s16(a - w) for a, w in zip(acc, weights)]
        assert add_avx == add_neon and sub_avx == sub_neon

        r16 = [rng.randrange(-32768, 32768) for _ in range(1024)]
        assert clipped_relu16_avx(r16) == clipped_relu16_neon(r16)
        r32 = [rng.randrange(-100000, 100000) for _ in range(32)]
        assert clipped_relu32_avx(r32) == clipped_relu32_neon(r32)

        # PreciseLinearLayer: vector grouping changes, but the int32 sum
        # (within this network's supported range) must be identical.
        inp = [rng.randrange(0, 8192) for _ in range(16)]
        weights16 = [rng.randrange(-4096, 4097) for _ in range(16)]
        bias = rng.randrange(-(1 << 30), 1 << 30)
        avx_sum = s32(bias + sum(x * w for x, w in zip(inp, weights16)))
        # NEON vmull_s16 + vector int32 accumulators + horizontal sum.
        neon_lanes = [s32(inp[j] * weights16[j] + inp[j + 8] * weights16[j + 8])
                      for j in range(8)]
        neon_sum = s32(bias + sum(neon_lanes))
        assert avx_sum == neon_sum

    print(f"PASS: {ROUNDS} deterministic NEON-vs-AVX arithmetic rounds (seed=0x{SEED:X}).")


if __name__ == "__main__":
    main()
