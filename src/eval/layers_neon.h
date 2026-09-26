#ifndef QUIRKY_SRC_EVAL_LAYERS_NEON_H
#define QUIRKY_SRC_EVAL_LAYERS_NEON_H

// ARM64 NEON implementation of Quirky's NNUE layers.
//
// This file keeps the original NNUE model layout and arithmetic, but
// replaces the original AVX2 intrinsics with ARM64 NEON operations.

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <arm_neon.h>

#include "core/board/types.h"
#include "core/util.h"
#include "model_weights.h"
#include "util/bit.h"
#include "util/io.h"

namespace q_eval {

static constexpr int WEIGHT_SCALE = 64;
static constexpr int ACTIVATION_SCALE = 127;
static constexpr int OUTPUT_SCALE = 64 * 64;
static constexpr int PRECISE_WEIGHT_SCALE = 64;

static constexpr uint8_t FEATURE_ADDITIONAL_PRECISION = 5;
static constexpr uint8_t LINEAR_ADDITIONAL_PRECISION = 3;

struct ModelReader {
  public:
    template <std::integral T>
    T ReadWeight(int scale) {
        float weight = MODEL_WEIGHTS[index_++];
        int64_t final_weight = std::round(weight * scale);
        if (final_weight > static_cast<int64_t>(std::numeric_limits<T>::max()) ||
            final_weight < static_cast<int64_t>(std::numeric_limits<T>::min())) {
            q_util::ExitWithError("Model weights are out of range");
        }
        return static_cast<T>(final_weight);
    }

  private:
    size_t index_ = 0;
};

template <size_t INPUT_SIZE, size_t OUTPUT_SIZE>
struct FeatureLayer {
  public:
    void Initialize(ModelReader& reader) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            for (size_t j = 0; j < OUTPUT_SIZE / 2; j++) {
                weights_[i][j] =
                    reader.ReadWeight<int16_t>(ACTIVATION_SCALE *
                                               (1 << FEATURE_ADDITIONAL_PRECISION));

                const q_core::cell_t cell =
                    i / q_core::BOARD_SIZE + 1;
                const q_core::coord_t coord =
                    i % q_core::BOARD_SIZE;

                size_t pos =
                    (static_cast<size_t>(q_core::FlipCellColor(cell)) - 1) *
                        q_core::BOARD_SIZE +
                    q_core::FlipCoord(coord);

                weights_[pos][j + OUTPUT_SIZE / 2] = weights_[i][j];
            }
        }

        for (size_t i = 0; i < OUTPUT_SIZE / 2; i++) {
            biases_[i] =
                reader.ReadWeight<int16_t>(
                    ACTIVATION_SCALE *
                    (1 << FEATURE_ADDITIONAL_PRECISION));

            biases_[i + OUTPUT_SIZE / 2] = biases_[i];
        }
    }

    void GetResultOnEmptyBoard(int16_t* output) {
        std::copy(biases_.begin(), biases_.end(), output);
    }

    void Update(int16_t* __restrict input,
                size_t position,
                int8_t delta) {
        const int16_t* __restrict weights =
            weights_[position].data();

        for (size_t i = 0; i < OUTPUT_SIZE; i += 8) {
            int16x8_t a = vld1q_s16(input + i);
            int16x8_t b = vld1q_s16(weights + i);

            if (delta == -1) {
                a = vsubq_s16(a, b);
            } else {
                a = vaddq_s16(a, b);
            }

            vst1q_s16(input + i, a);
        }
    }

    void Add(int16_t* input, size_t position) {
        const int16_t* weights =
            weights_[position].data();

        for (size_t i = 0; i < OUTPUT_SIZE; i += 8) {
            const int16x8_t a =
                vld1q_s16(input + i);
            const int16x8_t b =
                vld1q_s16(weights + i);

            vst1q_s16(
                input + i,
                vaddq_s16(a, b));
        }
    }

    void SubAdd(int16_t* input,
                size_t position_first,
                size_t position_second) {
        const int16_t* first =
            weights_[position_first].data();

        const int16_t* second =
            weights_[position_second].data();

        for (size_t i = 0; i < OUTPUT_SIZE; i += 8) {
            int16x8_t a =
                vld1q_s16(input + i);

            const int16x8_t b =
                vld1q_s16(first + i);

            const int16x8_t c =
                vld1q_s16(second + i);

            a = vsubq_s16(a, b);
            a = vaddq_s16(a, c);

            vst1q_s16(input + i, a);
        }
    }

    void SubSubAdd(int16_t* __restrict input,
                   size_t position_first,
                   size_t position_second,
                   size_t position_third) {
        const int16_t* first =
            weights_[position_first].data();

        const int16_t* second =
            weights_[position_second].data();

        const int16_t* third =
            weights_[position_third].data();

        for (size_t i = 0; i < OUTPUT_SIZE; i += 8) {
            int16x8_t a =
                vld1q_s16(input + i);

            const int16x8_t b =
                vld1q_s16(first + i);

            const int16x8_t c =
                vld1q_s16(second + i);

            const int16x8_t d =
                vld1q_s16(third + i);

            a = vsubq_s16(a, b);
            a = vsubq_s16(a, c);
            a = vaddq_s16(a, d);

            vst1q_s16(input + i, a);
        }
    }

  private:
    alignas(64)
    std::array<std::array<int16_t, OUTPUT_SIZE>, INPUT_SIZE> weights_;

    alignas(64)
    std::array<int16_t, OUTPUT_SIZE> biases_;
};

// An ARM NEON representation of the original AVX2 __m256i used by the
// sparse linear layer: eight signed 32-bit lanes.
struct NeonI32x8 {
    int32x4_t lo;
    int32x4_t hi;
};

// An ARM NEON representation of the original AVX2 __m256i used by
// _mm256_maddubs_epi16: sixteen signed 16-bit pair sums.
struct NeonI16x16 {
    int16x8_t lo;
    int16x8_t hi;
};

// Exact numerical equivalent of _mm256_maddubs_epi16 for this NNUE.
//
// The first operand is an int32 containing four clipped-ReLU bytes.
// _mm256_set1_epi32 replicates those four bytes across the whole
// 256-bit register. The NNUE activations are in [0, 127], so treating
// those bytes as signed int8 gives the same numerical product as the
// original unsigned-byte operation.
//
// Each output consists of four byte products. The original instruction
// first sums pairs into signed int16 lanes. Those pair sums cannot
// overflow int16 for activation values [0,127] and int8 weights.
static inline NeonI16x16 Maddubs8(int32x4_t a,
                                   const int8_t* b) {
    const int8x16_t aa =
        vreinterpretq_s8_s32(a);

    const int8x16_t b0 =
        vld1q_s8(b);

    const int8x16_t b1 =
        vld1q_s8(b + 16);

    // Each vmull produces the byte products for one 128-bit half.
    // vpadd then performs the adjacent pair sums that
    // _mm256_maddubs_epi16 produces.
    const int16x8_t p0 =
        vmull_s8(
            vget_low_s8(aa),
            vget_low_s8(b0));

    const int16x8_t p1 =
        vmull_s8(
            vget_high_s8(aa),
            vget_high_s8(b0));

    const int16x8_t q0 =
        vmull_s8(
            vget_low_s8(aa),
            vget_low_s8(b1));

    const int16x8_t q1 =
        vmull_s8(
            vget_high_s8(aa),
            vget_high_s8(b1));

    const int16x4_t p0_pairs =
        vpadd_s16(
            vget_low_s16(p0),
            vget_high_s16(p0));

    const int16x4_t p1_pairs =
        vpadd_s16(
            vget_low_s16(p1),
            vget_high_s16(p1));

    const int16x4_t q0_pairs =
        vpadd_s16(
            vget_low_s16(q0),
            vget_high_s16(q0));

    const int16x4_t q1_pairs =
        vpadd_s16(
            vget_low_s16(q1),
            vget_high_s16(q1));

    return {
        vcombine_s16(p0_pairs, p1_pairs),
        vcombine_s16(q0_pairs, q1_pairs)
    };
}

static inline NeonI32x8 MaddEpi16(const NeonI16x16& a) {
    return {
        vpaddlq_s16(a.lo),
        vpaddlq_s16(a.hi)
    };
}

static inline NeonI32x8 AddI32x8(const NeonI32x8& a,
                                  const NeonI32x8& b) {
    return {
        vaddq_s32(a.lo, b.lo),
        vaddq_s32(a.hi, b.hi)
    };
}

template <size_t INPUT_SIZE, size_t OUTPUT_SIZE>
struct LinearLayer {
  public:
    void Initialize(ModelReader& reader) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            for (size_t j = 0; j < OUTPUT_SIZE; j++) {
                weights_[GetWeightIndex(
                    static_cast<int>(j * INPUT_SIZE + i))] =
                    reader.ReadWeight<int8_t>(
                        WEIGHT_SCALE *
                        (1 << LINEAR_ADDITIONAL_PRECISION));
            }
        }

        for (size_t i = 0; i < OUTPUT_SIZE; i++) {
            biases_[i] =
                reader.ReadWeight<int32_t>(
                    ACTIVATION_SCALE *
                    WEIGHT_SCALE *
                    (1 << LINEAR_ADDITIONAL_PRECISION));
        }

        for (size_t i = 0; i < 256; i++) {
            for (size_t j = 0; j < 8; j++) {
                lookup_indices_[i][j] = 0;
            }

            uint64_t j = i;
            uint64_t k = 0;

            while (j > 0) {
                lookup_indices_[i][k++] =
                    q_util::ExtractLowestBit(j);
            }
        }
    }

    void Process(const int8_t* src, int32_t* dest) {
        constexpr size_t SPARSE_CHUNK_SIZE = 4;
        constexpr size_t IN_WIDTH = 4;
        constexpr size_t NUM_CHUNKS =
            INPUT_SIZE / SPARSE_CHUNK_SIZE;
        constexpr size_t OUT_CC =
            OUTPUT_SIZE / 8;

        const int32_t* in32 =
            reinterpret_cast<const int32_t*>(src);

        std::array<uint16_t, NUM_CHUNKS> nnz{};
        const size_t count =
            FindNNZ(nnz.data(), in32, NUM_CHUNKS);

        std::array<NeonI32x8, OUT_CC> regs{};

        for (size_t i = 0; i < OUT_CC; i++) {
            regs[i].lo =
                vld1q_s32(biases_.data() + i * 8);
            regs[i].hi =
                vld1q_s32(biases_.data() + i * 8 + 4);
        }

        size_t i = 0;

        for (; i + 1 < count; i += 2) {
            const uint16_t i0 =
                nnz[i + 0];
            const uint16_t i1 =
                nnz[i + 1];

            const int32x4_t f0 =
                vdupq_n_s32(in32[i0]);

            const int32x4_t f1 =
                vdupq_n_s32(in32[i1]);

            const int8_t* c0 =
                &weights_[static_cast<size_t>(i0) *
                          OUTPUT_SIZE *
                          SPARSE_CHUNK_SIZE];

            const int8_t* c1 =
                &weights_[static_cast<size_t>(i1) *
                          OUTPUT_SIZE *
                          SPARSE_CHUNK_SIZE];

            for (size_t j = 0; j < OUT_CC; j++) {
                Addx2(
                    &regs[j],
                    f0,
                    c0 + j * 32,
                    f1,
                    c1 + j * 32);
            }
        }

        if (i < count) {
            const uint16_t i0 =
                nnz[i];

            const int32x4_t f0 =
                vdupq_n_s32(in32[i0]);

            const int8_t* c0 =
                &weights_[static_cast<size_t>(i0) *
                          OUTPUT_SIZE *
                          SPARSE_CHUNK_SIZE];

            for (size_t j = 0; j < OUT_CC; j++) {
                Add(
                    &regs[j],
                    f0,
                    c0 + j * 32);
            }
        }

        for (size_t i = 0; i < OUT_CC; i++) {
            vst1q_s32(
                dest + i * 8,
                vshrq_n_s32(
                    regs[i].lo,
                    LINEAR_ADDITIONAL_PRECISION));

            vst1q_s32(
                dest + i * 8 + 4,
                vshrq_n_s32(
                    regs[i].hi,
                    LINEAR_ADDITIONAL_PRECISION));
        }
    }

  private:
    int GetWeightIndex(int idx) {
        return ((idx / 4) %
                    (INPUT_SIZE / 4) *
                    OUTPUT_SIZE * 4) +
               (idx / INPUT_SIZE * 4) +
               (idx % 4);
    }

    static void Add(NeonI32x8* acc,
                    int32x4_t a,
                    const int8_t* b) {
        const NeonI16x16 p =
            Maddubs8(
                a,
                vld1q_s8(b),
                vld1q_s8(b + 16));

        const NeonI32x8 result =
            MaddEpi16(p);

        *acc =
            AddI32x8(*acc, result);
    }

    static void Addx2(NeonI32x8* acc,
                      int32x4_t a0,
                      const int8_t* b0,
                      int32x4_t a1,
                      const int8_t* b1) {
        const NeonI16x16 p0 =
            Maddubs8(
                a0,
                vld1q_s8(b0),
                vld1q_s8(b0 + 16));

        const NeonI16x16 p1 =
            Maddubs8(
                a1,
                vld1q_s8(b1),
                vld1q_s8(b1 + 16));

        const NeonI16x16 combined = {
            vaddq_s16(p0.lo, p1.lo),
            vaddq_s16(p0.hi, p1.hi)
        };

        const NeonI32x8 result =
            MaddEpi16(combined);

        *acc =
            AddI32x8(*acc, result);
    }

    size_t FindNNZ(uint16_t* dest,
                   const int32_t* inputs,
                   const size_t chunks) {
        // The original AVX2 implementation builds an 8-bit mask for
        // each group of eight int32 values and expands it through a
        // lookup table. A scalar scan preserves exactly the same
        // ordering and is only used to discover sparse input rows.
        size_t count = 0;

        for (size_t i = 0; i < chunks; i++) {
            if (inputs[i] > 0) {
                dest[count++] =
                    static_cast<uint16_t>(i);
            }
        }

        return count;
    }

    alignas(64)
    std::array<int8_t,
               INPUT_SIZE * OUTPUT_SIZE> weights_;

    alignas(64)
    std::array<int32_t, OUTPUT_SIZE> biases_;

    alignas(64)
    std::array<std::array<uint16_t, 8>, 256>
        lookup_indices_;
};

template <size_t INPUT_SIZE, size_t OUTPUT_SIZE>
struct PreciseLinearLayer {
  public:
    void Initialize(ModelReader& reader) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            for (size_t j = 0; j < OUTPUT_SIZE; j++) {
                weights_[j * INPUT_SIZE + i] =
                    reader.ReadWeight<int16_t>(
                        WEIGHT_SCALE *
                        PRECISE_WEIGHT_SCALE);
            }
        }

        for (size_t i = 0; i < OUTPUT_SIZE; i++) {
            biases_[i] =
                reader.ReadWeight<int32_t>(
                    ACTIVATION_SCALE *
                    WEIGHT_SCALE *
                    WEIGHT_SCALE *
                    PRECISE_WEIGHT_SCALE);
        }
    }

    void Process(const int16_t* input,
                 int32_t* output) {
        constexpr int REGISTER_WIDTH = 8;

        constexpr int NUMBER_OF_INPUT_CHUNKS =
            INPUT_SIZE / REGISTER_WIDTH;

        constexpr int NUMBER_OF_OUTPUT_CHUNKS =
            OUTPUT_SIZE / 4;

        for (int i = 0;
             i < NUMBER_OF_OUTPUT_CHUNKS;
             i++) {
            const size_t offset0 =
                (i * 4 + 0) * INPUT_SIZE;
            const size_t offset1 =
                (i * 4 + 1) * INPUT_SIZE;
            const size_t offset2 =
                (i * 4 + 2) * INPUT_SIZE;
            const size_t offset3 =
                (i * 4 + 3) * INPUT_SIZE;

            int32x4_t sum0_lo =
                vdupq_n_s32(0);
            int32x4_t sum0_hi =
                vdupq_n_s32(0);

            int32x4_t sum1_lo =
                vdupq_n_s32(0);
            int32x4_t sum1_hi =
                vdupq_n_s32(0);

            int32x4_t sum2_lo =
                vdupq_n_s32(0);
            int32x4_t sum2_hi =
                vdupq_n_s32(0);

            int32x4_t sum3_lo =
                vdupq_n_s32(0);
            int32x4_t sum3_hi =
                vdupq_n_s32(0);

            for (int j = 0;
                 j < NUMBER_OF_INPUT_CHUNKS;
                 j++) {
                const int16x8_t in =
                    vld1q_s16(
                        input +
                        j * REGISTER_WIDTH);

                const int16x8_t w0 =
                    vld1q_s16(
                        weights_.data() +
                        offset0 +
                        j * REGISTER_WIDTH);

                const int16x8_t w1 =
                    vld1q_s16(
                        weights_.data() +
                        offset1 +
                        j * REGISTER_WIDTH);

                const int16x8_t w2 =
                    vld1q_s16(
                        weights_.data() +
                        offset2 +
                        j * REGISTER_WIDTH);

                const int16x8_t w3 =
                    vld1q_s16(
                        weights_.data() +
                        offset3 +
                        j * REGISTER_WIDTH);

                const int32x4_t p0lo =
                    vmull_s16(
                        vget_low_s16(in),
                        vget_low_s16(w0));

                const int32x4_t p0hi =
                    vmull_s16(
                        vget_high_s16(in),
                        vget_high_s16(w0));

                const int32x4_t p1lo =
                    vmull_s16(
                        vget_low_s16(in),
                        vget_low_s16(w1));

                const int32x4_t p1hi =
                    vmull_s16(
                        vget_high_s16(in),
                        vget_high_s16(w1));

                const int32x4_t p2lo =
                    vmull_s16(
                        vget_low_s16(in),
                        vget_low_s16(w2));

                const int32x4_t p2hi =
                    vmull_s16(
                        vget_high_s16(in),
                        vget_high_s16(w2));

                const int32x4_t p3lo =
                    vmull_s16(
                        vget_low_s16(in),
                        vget_low_s16(w3));

                const int32x4_t p3hi =
                    vmull_s16(
                        vget_high_s16(in),
                        vget_high_s16(w3));

                sum0_lo =
                    vaddq_s32(sum0_lo, p0lo);
                sum0_hi =
                    vaddq_s32(sum0_hi, p0hi);

                sum1_lo =
                    vaddq_s32(sum1_lo, p1lo);
                sum1_hi =
                    vaddq_s32(sum1_hi, p1hi);

                sum2_lo =
                    vaddq_s32(sum2_lo, p2lo);
                sum2_hi =
                    vaddq_s32(sum2_hi, p2hi);

                sum3_lo =
                    vaddq_s32(sum3_lo, p3lo);
                sum3_hi =
                    vaddq_s32(sum3_hi, p3hi);
            }

            int32x4_t outval = {
                vaddvq_s32(sum0_lo) +
                    vaddvq_s32(sum0_hi),
                vaddvq_s32(sum1_lo) +
                    vaddvq_s32(sum1_hi),
                vaddvq_s32(sum2_lo) +
                    vaddvq_s32(sum2_hi),
                vaddvq_s32(sum3_lo) +
                    vaddvq_s32(sum3_hi)
            };

            outval =
                vaddq_s32(
                    outval,
                    vld1q_s32(
                        biases_.data() + i * 4));

            outval =
                vshrq_n_s32(
                    outval,
                    q_util::GetHighestBit(
                        static_cast<uint32_t>(
                            WEIGHT_SCALE *
                            PRECISE_WEIGHT_SCALE)));

            vst1q_s32(
                output + i * 4,
                outval);
        }
    }

  private:
    alignas(64)
    std::array<int16_t,
               INPUT_SIZE * OUTPUT_SIZE> weights_;

    alignas(64)
    std::array<int32_t, OUTPUT_SIZE> biases_;
};

template <size_t INPUT_SIZE>
struct OutputLayer {
  public:
    void Initialize(ModelReader& reader) {
        for (size_t i = 0; i < INPUT_SIZE; i++) {
            weights_[i] =
                reader.ReadWeight<int16_t>(
                    WEIGHT_SCALE *
                    OUTPUT_SCALE /
                    ACTIVATION_SCALE);
        }

        bias_ =
            reader.ReadWeight<int32_t>(
                WEIGHT_SCALE *
                OUTPUT_SCALE);
    }

    int32_t Process(const int16_t* input) {
        int32_t ans = 0;

        for (uint16_t i = 0;
             i < INPUT_SIZE;
             i++) {
            ans +=
                static_cast<int32_t>(input[i]) *
                static_cast<int32_t>(weights_[i]);
        }

        ans += bias_;
        return ans;
    }

  private:
    alignas(64)
    std::array<int16_t, INPUT_SIZE> weights_;

    int32_t bias_;
};

inline void ClippedReLU16(int size,
                          int8_t* output,
                          const int16_t* input) {
    const int16x8_t zero =
        vdupq_n_s16(0);

    for (int i = 0; i < size; i += 16) {
        int16x8_t a =
            vld1q_s16(input + i);

        int16x8_t b =
            vld1q_s16(input + i + 8);

        a =
            vshrq_n_s16(
                a,
                FEATURE_ADDITIONAL_PRECISION);

        b =
            vshrq_n_s16(
                b,
                FEATURE_ADDITIONAL_PRECISION);

        a =
            vmaxq_s16(a, zero);

        b =
            vmaxq_s16(b, zero);

        // The original _mm256_packs_epi16 saturates to signed int8,
        // then maxes against zero. vqmovn_s16 is the equivalent
        // saturating narrowing operation after the clamp.
        const int8x8_t a8 =
            vqmovn_s16(a);

        const int8x8_t b8 =
            vqmovn_s16(b);

        vst1q_s8(
            output + i,
            vcombine_s8(a8, b8));
    }
}

inline void ClippedReLU32(int size,
                          int16_t* output,
                          const int32_t* input) {
    const int32x4_t zero =
        vdupq_n_s32(0);

    const int32x4_t upper =
        vdupq_n_s32(
            32768 *
                WEIGHT_SCALE /
                256 -
            1);

    for (int i = 0; i < size; i += 8) {
        int32x4_t a =
            vld1q_s32(input + i);

        int32x4_t b =
            vld1q_s32(input + i + 4);

        a =
            vmaxq_s32(a, zero);

        b =
            vmaxq_s32(b, zero);

        a =
            vminq_s32(a, upper);

        b =
            vminq_s32(b, upper);

        const int16x4_t a16 =
            vqmovn_s32(a);

        const int16x4_t b16 =
            vqmovn_s32(b);

        vst1q_s16(
            output + i,
            vcombine_s16(a16, b16));
    }
}

}  // namespace q_eval

#endif  // QUIRKY_SRC_EVAL_LAYERS_NEON_H
