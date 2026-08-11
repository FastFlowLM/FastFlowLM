/// \file modeling_qwen3_6_moe_image.cpp
/// \brief Qwen3_6_MOE image processing implementation
/// \author FastFlowLM Team
/// \date 2026-01-23
/// \version 0.9.28
/// \note This is a source file for the Qwen3_6_MOE image processing functionality

#include "AutoModel/modeling_qwen3_6_moe.hpp"

qwen3_6_moe_image_t Qwen3_6_MOE::load_image(const std::string& filename) {
    qwen3_6_moe_image_t empty_result{};
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image(filename, decoded)) {
        header_print("ERROR", "Qwen3_6_MOE failed to load image: " << filename);
        return empty_result;
    }

    if (this->image_pre_resize > 0) {
        int max_height;
        switch(this->image_pre_resize) {
            case 1:
                max_height = 480;
                break;
            case 2:
                max_height = 720;
                break;
            case 3:
                max_height = 1080;
                break;
            case 4:
                max_height = 1440;
                break;
            case 5:
                max_height = 2160;
                break;
            case 6:
                max_height = 2880;
                break;
            case 7:
                max_height = 3240;
                break;
            case 8:
                max_height = 4320;
                break;
            default:
                max_height = decoded.height; // no resizing
                break;
        }

        if (decoded.height > max_height) {
            image_data_t resized_image;
            float ratio = static_cast<float>(max_height) / static_cast<float>(decoded.height);
            int target_width = static_cast<int>(static_cast<float>(decoded.width) * ratio);
            int target_height = max_height;
            header_print_r("FLM", "Qwen3_6_MOE resizing image from (" + std::to_string(decoded.width) + ", " + std::to_string(decoded.height) + ") to (" + std::to_string(target_width) + ", " + std::to_string(target_height) + ")\n");
            if (image_reader_.resize_image(decoded, target_width, target_height, resized_image)) {
                image_reader_.recycle(decoded);
                decoded = std::move(resized_image);
            }
        }
    }

    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered)) {
        header_print("ERROR", "Qwen3_6_MOE failed to reorder image (HWC->CHW): " << filename);
        image_reader_.recycle(decoded);
        return empty_result;
    }

    image_reader_.recycle(decoded);

    qwen3_6_moe_image_t result{};
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}

qwen3_6_moe_image_t Qwen3_6_MOE::load_image_base64(const std::string& base64_string) {
    qwen3_6_moe_image_t empty_result{};
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image_base64(base64_string, decoded)) {
        header_print("ERROR", "Qwen3_6_MOE failed to decode base64 image");
        return empty_result;
    }

    if (this->image_pre_resize > 0) {
        int max_height;
        switch(this->image_pre_resize) {
            case 1:
                max_height = 480;
                break;
            case 2:
                max_height = 720;
                break;
            case 3:
                max_height = 1080;
                break;
            case 4:
                max_height = 1440;
                break;
            case 5:
                max_height = 2160;
                break;
            case 6:
                max_height = 2880;
                break;
            case 7:
                max_height = 3240;
                break;
            case 8:
                max_height = 4320;
                break;
            default:
                max_height = decoded.height; // no resizing
                break;
        }

        if (decoded.height > max_height) {
            image_data_t resized_image;
            float ratio = static_cast<float>(max_height) / static_cast<float>(decoded.height);
            int target_width = static_cast<int>(static_cast<float>(decoded.width) * ratio);
            int target_height = max_height;
            header_print_r("FLM", "Qwen3_6_MOE resizing image from (" + std::to_string(decoded.width) + ", " + std::to_string(decoded.height) + ") to (" + std::to_string(target_width) + ", " + std::to_string(target_height) + ")");
            if (image_reader_.resize_image(decoded, target_width, target_height, resized_image)) {
                image_reader_.recycle(decoded);
                decoded = std::move(resized_image);
            }
        }
    }
    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered)) {
        header_print("ERROR", "Qwen3_6_MOE failed to reorder base64 image (HWC->CHW)");
        image_reader_.recycle(decoded);
        return empty_result;
    }

    image_reader_.recycle(decoded);

    qwen3_6_moe_image_t result{};
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}


void Qwen3_6_MOE::smart_resize(
    int height,
    int width,
    int& h_bar,
    int& w_bar,
    int factor,
    int min_pixels,
    int max_pixels
) {
    if (height <= 0 || width <= 0) {
        header_print("ERROR", "Qwen3_6_MOE smart_resize got invalid image size ("
            << width << "x" << height << "); likely a failed image load");
        h_bar = 0;
        w_bar = 0;
        return;
    }
    double aspect_ratio = static_cast<double>(std::max(height, width)) /
                          static_cast<double>(std::min(height, width));
    if (aspect_ratio > 200.0) {
        std::cerr << "absolute aspect ratio must be smaller than 200, got " +
            std::to_string(aspect_ratio);
    }

    h_bar = static_cast<int>(std::round(static_cast<double>(height) / factor)) * factor;
    w_bar = static_cast<int>(std::round(static_cast<double>(width) / factor)) * factor;

    long long total_pixels = static_cast<long long>(h_bar) * w_bar;

    if (total_pixels > max_pixels) {
        double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(factor,
                         static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor,
                         static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (total_pixels < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) /
                                (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
}


///@brief: preprocess the image for Qwen3_6_MOE model
///@note: Converts uint8 image to BF16 format, data is already in (3, H, W) CHW layout
///@param: image: the image to preprocess (already in CHW format)
///@return: the preprocessed image in BF16 format
void Qwen3_6_MOE::preprocess_image(qwen3_6_moe_image_t& image, std::vector<bf16> &pixel_values) {
    const int width = image.width;
    const int height = image.height;
    const int channels = 3; // RGB
    int resized_height = 0;
    int resized_width = 0;
    // do the automatically resizing in here

    if (width <= 0 || height <= 0) {
        header_print("ERROR", "Qwen3_6_MOE preprocess_image skipped: invalid image size ("
            << width << "x" << height << "); check the image path/data");
        image.width_resized = 0;
        image.height_resized = 0;
        image.grid_h = 0;
        image.grid_w = 0;
        image._data.free();
        return;
    }

    qwen3_6_moe_npu* lm_engine_qwen3_6_ptr = reinterpret_cast<qwen3_6_moe_npu*>(this->lm_engine.get());
    smart_resize(
        height, width,
        resized_height, resized_width,

        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_PATCH_SIZE * lm_engine_qwen3_6_ptr->QWEN3_6_MOE_IMAGE_MERGE_SIZE,
        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_SHORTEST_EDGE,
        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_LONGEST_EDGE
    );

    if (resized_height <= 0 || resized_width <= 0) {
        header_print("ERROR", "Qwen3_6_MOE preprocess_image skipped: smart_resize produced ("
            << resized_width << "x" << resized_height << ")");
        image.width_resized = 0;
        image.height_resized = 0;
        image.grid_h = 0;
        image.grid_w = 0;
        image._data.free();
        return;
    }
    // std::cout << "resized_height "<< resized_height << " resized_width " << resized_width <<std::endl;

    // Cache size calculations for efficiency
    const uint32_t single_frame_size = resized_height * resized_width * channels;
    const uint32_t total_patch_size = single_frame_size * lm_engine_qwen3_6_ptr->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE;
    const uint32_t grid_h = resized_height / lm_engine_qwen3_6_ptr->QWEN3_6_MOE_PATCH_SIZE;
    const uint32_t grid_w = resized_width / lm_engine_qwen3_6_ptr->QWEN3_6_MOE_PATCH_SIZE;

    // Pre-allocate final buffer to avoid reallocation
    const uint32_t prev_pixel_values_size = pixel_values.size();
    pixel_values.resize(prev_pixel_values_size + total_patch_size);

    // Use non-optimized path for consistent results across platforms
    auto resize_image =  imgproc::avx512::resize_bicubic_antialias_rgb_planar_avx512(
        image._data.data(), width, height, resized_width, resized_height, true
    );

    // Reuse scratch buffer across calls to avoid repeated allocations
    static thread_local std::vector<float> patch_vector_scratch;
    if (patch_vector_scratch.size() < total_patch_size) {
        patch_vector_scratch.resize(total_patch_size);
    }

    // Apply rescale and normalization to first frame
    imgproc::avx512::rescale_and_normalize_avx512(
        resize_image.data(), patch_vector_scratch.data(),
        resized_width, resized_height, channels,
        true, lm_engine_qwen3_6_ptr->QWEN3_6_MOE_VISION_RESCALE_FACTOR,
        true, lm_engine_qwen3_6_ptr->QWEN3_6_MOE_VISION_RESCALE_IMAGE_MEAN, lm_engine_qwen3_6_ptr->QWEN3_6_MOE_VISION_RESCALE_IMAGE_STD
    );

    // Replicate first frame for temporal patches (optimized for QWEN3_6_MOE_TEMPORAL_PATCH_SIZE = 2)
    // This is more efficient than a loop for the common case
    if  (lm_engine_qwen3_6_ptr->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE == 2) {
        memcpy(
            patch_vector_scratch.data() + single_frame_size,
            patch_vector_scratch.data(),
            single_frame_size * sizeof(float)
        );
    } else {
        // Generic loop for other TEMPORAL_PATCH_SIZE values
        for(unsigned l = 1; l < lm_engine_qwen3_6_ptr->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE; l++){
            memcpy(
                patch_vector_scratch.data() + l * single_frame_size,
                patch_vector_scratch.data(),
                single_frame_size * sizeof(float)
            );
        }
    }

    // Reorder patches directly into pre-allocated pixel_values
    imgproc::reorder_patches_inplace(
        patch_vector_scratch.data(),
        pixel_values.data() + prev_pixel_values_size,
        1, 1, // something special for image
        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_TEMPORAL_PATCH_SIZE,
        channels,
        grid_h, grid_w,
        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_MERGE_SIZE,
        lm_engine_qwen3_6_ptr->QWEN3_6_MOE_PATCH_SIZE
    );


    image.width_resized = resized_width;
    image.height_resized = resized_height;
    image.grid_h = grid_h;
    image.grid_w = grid_w;
    image._data.free(); // free the data
    // release the original uint8_t data now, since is no longer useful to us
}
