/// \file q4nx_host.hpp
/// \brief Host-side reader for the Q4NX weight container.
/// \note Independent of q4_npu_eXpress.dll: this un-tiles a Q4NX tensor into a
///       plain row-major bf16 matrix so a host engine can use it directly.
///
/// The layout is not guessed. It was derived by inverting q4nx-build's
/// `_pack_q4nx` and cross-checked two ways: against an independent reading
/// solved from a published model file (LLMNpuTest/tools/q4nx.py), and against
/// FastFlowLM's own shipped models -- 2374 tensors across 13 models and 6
/// architecture families reproduce byte-for-byte
/// (q4nx-build/tools/validate_against_flm.py).
///
/// One tile covers 32 output rows x 256 input columns in 5120 bytes:
///
///     [ 512 B  d  as 256 bf16 ][ 512 B  m  as 256 bf16 ][ 4096 B nibbles ]
///
/// Tiles are row-major over (rows/32) x (cols/256). Inside a tile, with
/// R = row in tile (0..31), c = column in tile (0..255) and kb = c / 32:
///
///     d, m index          = kb * 32 + R
///     nibble byte offset  = (R / 16) * 2048 + c * 8 + ((R % 16) / 2)
///     low nibble when (R % 16) is even
///
/// Q4_1 semantics throughout: w = code * d + m, codes 0..15.
#pragma once

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "buffer.hpp"
#include "typedef.hpp"

namespace q4nx_host {

constexpr size_t ROW_BLOCK = 32;    ///< output rows per tile
constexpr size_t COL_BLOCK = 256;   ///< input columns per tile
constexpr size_t GROUP = 32;        ///< weights per quantization group
constexpr size_t META_BYTES = 512;  ///< 256 bf16 entries per metadata plane
constexpr size_t TILE_BYTES = 2 * META_BYTES + ROW_BLOCK * COL_BLOCK / 2;  // 5120

/// \brief Reinterpret two bytes as bf16, then widen to float.
inline float bf16_to_float(const uint8_t* p) {
    uint32_t bits = (static_cast<uint32_t>(p[1]) << 24) | (static_cast<uint32_t>(p[0]) << 16);
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

/// \brief Is this tensor a tiled Q4NX matrix, or a plain bf16 passthrough?
/// \note The embedding table and every norm are stored unquantized.
inline bool is_tiled(size_t byte_size, size_t rows, size_t cols) {
    if (rows % ROW_BLOCK || cols % COL_BLOCK) return false;
    return byte_size == (rows / ROW_BLOCK) * (cols / COL_BLOCK) * TILE_BYTES;
}

/// \brief Un-tile a Q4NX tensor into row-major bf16.
/// \param packed the raw tensor bytes as stored in model.q4nx
/// \param rows output rows (N), \param cols input columns (K)
/// \param out  receives rows*cols bf16 values, row-major
inline void dequantize(const uint8_t* packed, size_t packed_bytes,
                       size_t rows, size_t cols, bf16* out) {
    if (!is_tiled(packed_bytes, rows, cols)) {
        throw std::runtime_error("q4nx_host: tensor is not a " +
                                 std::to_string(rows) + "x" + std::to_string(cols) +
                                 " tile grid");
    }
    const size_t col_tiles = cols / COL_BLOCK;

    for (size_t row = 0; row < rows; ++row) {
        const size_t row_tile = row / ROW_BLOCK;
        const size_t R = row % ROW_BLOCK;
        // Where this row's nibbles start inside a tile, and which half-block.
        const size_t g = R / 16;
        const size_t rem = R % 16;
        const size_t r_off = rem / 2;
        const unsigned shift = (rem % 2) ? 4u : 0u;   // b = 1 is the high nibble

        for (size_t col_tile = 0; col_tile < col_tiles; ++col_tile) {
            const uint8_t* tile = packed + (row_tile * col_tiles + col_tile) * TILE_BYTES;
            const uint8_t* dplane = tile;
            const uint8_t* mplane = tile + META_BYTES;
            const uint8_t* codes = tile + 2 * META_BYTES;

            bf16* dst = out + row * cols + col_tile * COL_BLOCK;
            for (size_t c = 0; c < COL_BLOCK; ++c) {
                const size_t kb = c / GROUP;
                const size_t meta = kb * ROW_BLOCK + R;
                const float d = bf16_to_float(dplane + 2 * meta);
                const float m = bf16_to_float(mplane + 2 * meta);
                const uint8_t byte = codes[g * 2048 + c * 8 + r_off];
                const float code = static_cast<float>((byte >> shift) & 0x0F);
                dst[c] = static_cast<bf16>(code * d + m);
            }
        }
    }
}

}  // namespace q4nx_host
