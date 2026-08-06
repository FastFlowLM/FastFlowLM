/// \file npu_utils.hpp
/// \brief Backend dispatcher for the NPU device/app management classes.
/// \note  The NPU dispatch layer genuinely differs between the two runtime
///        backends (XRT builds an ELF via aiebu + xrt::module/elf/ext::kernel;
///        HRX builds an XADX executable directly via hrx_amdxdna). Rather than
///        interleave the two implementations, each lives in its own variant
///        header and this file includes exactly one, chosen by the FLM_USE_HRX
///        build flag (0 = XRT default, 1 = HRX). This keeps each backend's code
///        clean and lets the XRT variant stay a verbatim copy of the upstream
///        implementation.
#pragma once

#if defined(FLM_USE_HRX)
#include "npu_utils_hrx.hpp"
#else
#include "npu_utils_xrt.hpp"
#endif