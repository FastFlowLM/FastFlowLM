/// \file dequant.hpp
/// \brief dequant class
/// \author FastFlowLM Team
/// \date 2025-06-24
/// \version 0.9.10
/// \note This is a header file for the dequant class
#pragma once
#include "lm_config.hpp"
#include "npu_utils/npu_instr_utils.hpp"

/// \brief dequant class
/// \note This is a class for the dequant layer
class Dequant{
public:
    Dequant(){}

    /// \brief Constructor
    /// \param config the configuration
    /// \param xclbin_name the xclbin name
    /// \param npu the npu manager
    Dequant(LM_Config& config);
    ~Dequant();

    typedef enum: int {
        Q4_1 = 0,
        Q8_0 = 1
    } quant_block_t;

    void reorder_cpy(u8 *dst, buffer<u8> &src,
        quant_block_t quant_block_type,
        const int quant_matrix_row,
        const int quant_matrix_col,
        const int vertical_blocks=2,
        const int vetrical_block_interleave_byte_size=-1);
    void generate_dequant_q4_1_seq(npu_sequence* seq, const uint32_t D_in, const uint32_t D_out, const uint32_t weight_offset, int mode);
    void generate_dequant_q80_packed_in_q4nx_seq(npu_sequence* seq, const uint32_t D_in, const uint32_t D_out, const uint32_t weight_offset, int mode);
private:
    struct Impl;
    Impl* _impl;

};

