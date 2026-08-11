/// \file modeling_qwen3_5_omni_image.cpp
/// \brief Qwen3_5_Omni image (vision) preprocessing.
/// \author FastFlowLM Team
/// \note Vision math is the Qwen3.5-VL family pipeline; the payload it fills
///       (qwen3_5_omni_image_payload_t) is gemma4e-shaped (per-image vectors).

#include "AutoModel/modeling_qwen3_5_omni.hpp"

qwen3_5_omni_image_t Qwen3_5_Omni::load_image(const std::string& filename) {
    qwen3_5_omni_image_t empty_result{};
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image(filename, decoded)) {
        header_print("ERROR", "Qwen3_5_Omni failed to load image: " << filename);
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
            header_print_r("FLM", "Qwen3_5-Omni resizing image from (" + std::to_string(decoded.width) + ", " + std::to_string(decoded.height) + ") to (" + std::to_string(target_width) + ", " + std::to_string(target_height) + ")\n");
            if (image_reader_.resize_image(decoded, target_width, target_height, resized_image)) {
                image_reader_.recycle(decoded);
                decoded = std::move(resized_image);
            }
        }
    }
    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered)) {
        header_print("ERROR", "Qwen3_5_Omni failed to reorder image (HWC->CHW): " << filename);
        image_reader_.recycle(decoded);
        return empty_result;
    }
    image_reader_.recycle(decoded);

    qwen3_5_omni_image_t result{};
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}

qwen3_5_omni_image_t Qwen3_5_Omni::load_image_base64(const std::string& base64_string) {
    qwen3_5_omni_image_t empty_result{};
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image_base64(base64_string, decoded)) {
        header_print("ERROR", "Qwen3_5_Omni failed to decode base64 image");
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
            header_print_r("FLM", "Qwen3_5-Omni resizing image from (" + std::to_string(decoded.width) + ", " + std::to_string(decoded.height) + ") to (" + std::to_string(target_width) + ", " + std::to_string(target_height) + ")\n");
            if (image_reader_.resize_image(decoded, target_width, target_height, resized_image)) {
                image_reader_.recycle(decoded);
                decoded = std::move(resized_image);
            }
        }
    }    
    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered)) {
        header_print("ERROR", "Qwen3_5_Omni failed to reorder base64 image (HWC->CHW)");
        image_reader_.recycle(decoded);
        return empty_result;
    }
    image_reader_.recycle(decoded);

    qwen3_5_omni_image_t result{};
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}

void Qwen3_5_Omni::smart_resize(
    int height, int width,
    int& h_bar, int& w_bar,
    int factor, int min_pixels, int max_pixels
) {
    if (height <= 0 || width <= 0) {
        header_print("ERROR", "Qwen3_5_Omni smart_resize got invalid image size ("
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
        h_bar = std::max(factor, static_cast<int>(std::floor(height / beta / factor)) * factor);
        w_bar = std::max(factor, static_cast<int>(std::floor(width / beta / factor)) * factor);
    } else if (total_pixels < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<int>(std::ceil(width * beta / factor)) * factor;
    }
}

///@brief Preprocess one image: resize -> rescale/normalize -> patchify (bf16, CHW).
///@note Appends the flattened patch data onto pixel_values and reports the grid
///      pair, valid patch size, and soft-token count via reference params.
void Qwen3_5_Omni::preprocess_image(
    qwen3_5_omni_image_t& image,
    std::vector<bf16>& pixel_values,
    std::vector<int>& image_grid_pair,
    uint32_t& valid_patch_size,
    uint32_t& num_soft_tokens
) {
    const int width = image.width;
    const int height = image.height;
    const int channels = 3; // RGB
    int resized_height = 0;
    int resized_width = 0;

    if (width <= 0 || height <= 0) {
        header_print("ERROR", "Qwen3_5_Omni preprocess_image skipped: invalid image size ("
            << width << "x" << height << "); check the image path/data");
        image.width_resized = 0;
        image.height_resized = 0;
        valid_patch_size = 0;
        num_soft_tokens = 0;
        image._data.free();
        return;
    }

    smart_resize(
        height, width,
        resized_height, resized_width,
        static_cast<int>(this->patch_size * this->merge_size),
        static_cast<int>(this->shortest_edge),
        static_cast<int>(this->longest_edge)
    );

    if (resized_height <= 0 || resized_width <= 0) {
        header_print("ERROR", "Qwen3_5_Omni preprocess_image skipped: smart_resize produced ("
            << resized_width << "x" << resized_height << ")");
        image.width_resized = 0;
        image.height_resized = 0;
        valid_patch_size = 0;
        num_soft_tokens = 0;
        image._data.free();
        return;
    }

    const uint32_t single_frame_size = resized_height * resized_width * channels;
    const uint32_t total_patch_size = single_frame_size * this->temporal_patch_size;
    const uint32_t grid_h = resized_height / this->patch_size;
    const uint32_t grid_w = resized_width / this->patch_size;

    const uint32_t prev_pixel_values_size = pixel_values.size();
    pixel_values.resize(prev_pixel_values_size + total_patch_size);

    auto resize_image = imgproc::avx512::resize_bicubic_antialias_rgb_planar_avx512(
        image._data.data(), width, height, resized_width, resized_height, true
    );

    static thread_local std::vector<float> patch_vector_scratch;
    if (patch_vector_scratch.size() < total_patch_size) {
        patch_vector_scratch.resize(total_patch_size);
    }

    imgproc::avx512::rescale_and_normalize_avx512(
        resize_image.data(), patch_vector_scratch.data(),
        resized_width, resized_height, channels,
        true, this->vision_rescale_factor,
        true, this->vision_image_mean, this->vision_image_std
    );

    // Replicate the first frame across the temporal patch dimension.
    if (this->temporal_patch_size == 2) {
        memcpy(patch_vector_scratch.data() + single_frame_size,
               patch_vector_scratch.data(),
               single_frame_size * sizeof(float));
    } else {
        for (unsigned l = 1; l < this->temporal_patch_size; l++) {
            memcpy(patch_vector_scratch.data() + l * single_frame_size,
                   patch_vector_scratch.data(),
                   single_frame_size * sizeof(float));
        }
    }

    imgproc::reorder_patches_inplace(
        patch_vector_scratch.data(),
        pixel_values.data() + prev_pixel_values_size,
        1, 1, // something special for image
        this->temporal_patch_size,
        channels,
        grid_h, grid_w,
        this->merge_size,
        this->patch_size
    );

    image.width_resized = resized_width;
    image.height_resized = resized_height;
    image_grid_pair = { static_cast<int>(grid_h), static_cast<int>(grid_w) };
    
    num_soft_tokens = (grid_h * grid_w) / (this->merge_size * this->merge_size);
    image._data.free();
}
