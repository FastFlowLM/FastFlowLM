/// \file modeling_Qwen3VL_image.cpp
/// \brief Gemma4e image processing implementation
/// \author FastFlowLM Team
/// \date 2025-09-01
/// \version 0.9.24
/// \note This is a source file for the Gemma4e image processing functionality
#include "AutoModel/modeling_gemma4_12b.hpp"
#include "error_measure.hpp"
#include <utility>
#include <algorithm>
#include <numeric>
#include <climits>
#include <cmath>

/// \brief Merge k×k groups of small patches into larger model patches.
/// Mirrors the Python `patches_merge` (single image, no batch dim).
/// \param patches           (L, D) row-major, D = patch_size*patch_size*3
/// \param positions_xy      (L, 2) row-major, integer (x, y) per patch
/// \param length            number of output patches; requires L == length * k*k
/// \param merged_patches    out: (length, k*k*D) = (length, (k*patch_size)^2 * 3)
/// \param merged_positions  out: (length, 2)
static void patches_merge(
    const std::vector<float> &patches,
    const std::vector<int> &positions_xy,
    int length,
    std::vector<float> &merged_patches,
    std::vector<int> &merged_positions)
{
    const int L = static_cast<int>(positions_xy.size() / 2);
    const int D = static_cast<int>(patches.size() / L);

    // patch_size = isqrt(D / 3)
    const int patch_size = static_cast<int>(std::lround(std::sqrt(D / 3.0)));
    if (patch_size * patch_size * 3 != D) {
        std::cerr << "patches_merge: D=" << D << " is not patch_size*patch_size*3" << std::endl;
        exit(-1);
    }

    // k = isqrt(L / length)
    const int k = static_cast<int>(std::lround(std::sqrt(static_cast<double>(L) / length)));
    if (k * k * length != L) {
        std::cerr << "patches_merge: cannot merge L=" << L << " into length=" << length << std::endl;
        exit(-1);
    }

    // floor division matching torch rounding_mode="floor"
    auto floor_div = [](int a, int b) {
        int q = a / b, r = a % b;
        if ((r != 0) && ((r < 0) != (b < 0))) --q;
        return q;
    };

    // max_x = max(x) + 1
    int max_x = 0;
    for (int i = 0; i < L; ++i) max_x = std::max(max_x, positions_xy[i * 2 + 0]);
    max_x += 1;

    // target ordering per patch (groups patches into kernel-contiguous order)
    std::vector<long long> target_ordering(L);
    for (int i = 0; i < L; ++i) {
        int x = positions_xy[i * 2 + 0];
        int y = positions_xy[i * 2 + 1];
        int kx = floor_div(x, k);
        int ky = floor_div(y, k);
        long long num_from_tl = (long long)k * k * kx + (long long)k * max_x * ky;
        int wx = ((x % k) + k) % k;
        int wy = ((y % k) + k) % k;
        long long within = (long long)wx + (long long)wy * k;
        target_ordering[i] = within + num_from_tl;
    }

    // perm = argsort(target_ordering); kernel_ordered[i] = patches[perm[i]]
    std::vector<int> perm(L);
    std::iota(perm.begin(), perm.end(), 0);
    std::stable_sort(perm.begin(), perm.end(),
        [&](int a, int b) { return target_ordering[a] < target_ordering[b]; });

    // merge patches: within kernel j -> (a = y-within, b = x-within);
    // merged spatial coords are (a*patch_size + p, b*patch_size + q)
    const int merged_dim = k * k * D; // (k*patch_size)^2 * 3
    const int side = k * patch_size;  // merged spatial side length
    merged_patches.assign(static_cast<size_t>(length) * merged_dim, 0.0f);

    for (int l = 0; l < length; ++l) {
        float *dst = &merged_patches[static_cast<size_t>(l) * merged_dim];
        for (int j = 0; j < k * k; ++j) {
            int a = j / k; // y within kernel
            int b = j % k; // x within kernel
            int src_row = perm[l * k * k + j];
            const float *src = &patches[static_cast<size_t>(src_row) * D];
            for (int p = 0; p < patch_size; ++p) {
                for (int q = 0; q < patch_size; ++q) {
                    for (int c = 0; c < 3; ++c) {
                        int out_row = a * patch_size + p;
                        int out_col = b * patch_size + q;
                        int out_idx = (out_row * side + out_col) * 3 + c;
                        int in_idx = (p * patch_size + q) * 3 + c;
                        dst[out_idx] = src[in_idx];
                    }
                }
            }
        }
    }

    // merged positions: min over the kernel of floor(pos / k)
    merged_positions.assign(static_cast<size_t>(length) * 2, 0);
    for (int l = 0; l < length; ++l) {
        int min_x = INT_MAX, min_y = INT_MAX;
        for (int j = 0; j < k * k; ++j) {
            int src_row = perm[l * k * k + j];
            min_x = std::min(min_x, floor_div(positions_xy[src_row * 2 + 0], k));
            min_y = std::min(min_y, floor_div(positions_xy[src_row * 2 + 1], k));
        }
        merged_positions[l * 2 + 0] = min_x;
        merged_positions[l * 2 + 1] = min_y;
    }
}
gemma4_12b_image_t Gemma4_12B::load_image(const std::string &filename)
{
    gemma4_12b_image_t empty_result;
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image(filename, decoded))
    {
        return empty_result;
    }

    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered))
    {
        image_reader_.recycle(decoded);
        return empty_result;
    }

    image_reader_.recycle(decoded);

    gemma4_12b_image_t result;
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}

gemma4_12b_image_t Gemma4_12B::load_image_base64(const std::string &base64_string)
{
    gemma4_12b_image_t empty_result;
    image_data_t decoded;
    image_data_t reordered;
    if (!image_reader_.load_image_base64(base64_string, decoded))
    {
        return empty_result;
    }

    if (!image_reader_.reorder_hwc_to_chw(decoded, reordered))
    {
        image_reader_.recycle(decoded);
        return empty_result;
    }

    image_reader_.recycle(decoded);

    gemma4_12b_image_t result;
    result.width = reordered.width;
    result.height = reordered.height;
    result._data = std::move(reordered.pixels);
    image_reader_.recycle(reordered);
    return result;
}






std::vector<uint8_t> Gemma4_12B::aspect_ratio_preserving_resize(
    gemma4_12b_image_t &image,
    int patch_size,
    int max_patches,
    int pooling_kernel_size)
{

    int height = image.height;
    int width = image.width;
    int channels =3 ;



    int target_height = height;
    int target_width = width;
    {
        // the get_aspect_ratio_preserving_size() in python
        int total_px = height * width;
        int target_px = max_patches * (patch_size * patch_size);
        double factor = std::sqrt(static_cast<double>(target_px) / total_px);
        double ideal_height = factor * height;
        double ideal_width = factor * width;
        int side_mult = pooling_kernel_size * patch_size;

        target_height = static_cast<int>(std::floor(ideal_height / side_mult)) * side_mult;
        target_width = static_cast<int>(std::floor(ideal_width / side_mult)) * side_mult;

        if (target_height == 0 && target_width == 0) {
            std::cerr << "Attempting to resize to a 0 x 0 image."<< std::endl;
            exit(-1);
        }

        int max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;

        if (target_height == 0) {
            target_height = side_mult;
            target_width = std::min(
                static_cast<int>(std::floor(static_cast<double>(width) / height)) * side_mult,
                max_side_length
            );
        } else if (target_width == 0) {
            target_width = side_mult;
            target_height = std::min(
                static_cast<int>(std::floor(static_cast<double>(height) / width)) * side_mult,
                max_side_length
            );
        }

        if (target_height * target_width > target_px) {
            
            std::cerr << "Resized image exceeds max_patches"<< std::endl;
            exit(-1);
        }
    }

    if(target_height == height && target_width == width){
        image.width_resized = target_width;
        image.height_resized = target_height;

        std::vector<uint8_t> result(static_cast<size_t>(channels) * height * width);
        memcpy(result.data(), image._data.data(), static_cast<size_t>(channels) * height * width);

        image._data.free();
        return result;


    }else{
        // This is the point where we will need to do the 

        // Trigger the resize, which is the cubic 
        auto resized_image = imgproc::resize_bicubic_antialias_rgb_planar_optimized(

            image._data.data(),
            width, height,
            target_width, target_height,
            true
        );


        image.width_resized = target_width;
        image.height_resized = target_height;
        image._data.free();
        return resized_image;
    
    }
    







}

///@brief: preprocess the image for Gemma4_12B model
///@note: Converts uint8 image to BF16 format, data is already in (3, H, W) CHW layout
///@param: image: the image to preprocess (already in CHW format)
///@return: the preprocessed image in BF16 format
void Gemma4_12B::preprocess_image(
    gemma4_12b_image_t &image,
    std::pair<int, int> & patch_element_per_patch,
    uint32_t & valid_patch_size, // the unpadded size per image
    std::vector<bf16> &pixel_values,
    std::vector<int> &image_grid_pairs, // [num_of_position_id][x, y]
    uint32_t &num_soft_tokens

)
{

    //std::cout << "hit preprocess_image, image size: " << image.width << "x" << image.height << ", pixel count: " << (image.width * image.height) << std::endl;
    gemma4_12b_npu *lm_engine_gemma4e_ptr = reinterpret_cast<gemma4_12b_npu *>(this->lm_engine.get());
    int max_patches = this->image_softtoken_budget * lm_engine_gemma4e_ptr->GEMMA4_12B_vision_pooling_kernel_size * lm_engine_gemma4e_ptr->GEMMA4_12B_vision_pooling_kernel_size;
 
    // first, do_resize
    std::vector<uint8_t> resized_image_data = aspect_ratio_preserving_resize(
        image,
        lm_engine_gemma4e_ptr->GEMMA4_12B_vision_patch_size,
        max_patches,
        lm_engine_gemma4e_ptr->GEMMA4_12B_vision_pooling_kernel_size
    );

    // step 2, rescale and normaliuze
    std::vector<float> rescaled_and_normalized_bufer(
        static_cast<size_t>(image.width_resized) * image.height_resized * 3
    );


    imgproc::rescale_and_normalize_optimized(
        resized_image_data.data(),
        rescaled_and_normalized_bufer.data(),
        image.width_resized, image.height_resized, 3,
        true, lm_engine_gemma4e_ptr->GEMMA4_12B_vision_rescale_factor,
        false, lm_engine_gemma4e_ptr->GEMMA4_12B_vision_image_mean, lm_engine_gemma4e_ptr->GEMMA4_12B_vision_image_std
    );
   
    auto patch_height = image.height_resized / lm_engine_gemma4e_ptr->GEMMA4_12B_vision_patch_size;
    auto patch_width = image.width_resized / lm_engine_gemma4e_ptr->GEMMA4_12B_vision_patch_size;
    int num_patches = patch_height * patch_width;
    int patch_size = lm_engine_gemma4e_ptr->GEMMA4_12B_vision_patch_size;
    int num_channels = 3;
  

    std::vector<float> teacher_patches(static_cast<size_t>(num_patches) * patch_size * patch_size * num_channels);

    {
        // the convert_image_to_parchtes operation in python

        for (int ph = 0; ph < patch_height; ++ph) {
            for (int pw = 0; pw < patch_width; ++pw) {
                int patch_idx = ph * patch_width + pw;
                for (int c = 0; c < num_channels; ++c) {
                    for (int y = 0; y < patch_size; ++y) {
                        for (int x = 0; x < patch_size; ++x) {
                            int src_y = ph * patch_size + y;
                            int src_x = pw * patch_size + x;
                            int src_idx = (c * image.height_resized + src_y) * image.width_resized + src_x;
                            
                            int dst_idx = ((((ph * patch_width) + pw) * patch_size + y) * patch_size + x) * num_channels + c;
                            teacher_patches[dst_idx] = rescaled_and_normalized_bufer[src_idx];
                        }
                    }
                }
            }
        }


    }


    // std::cout << "num_patches " << num_patches << "elements_per_patch" << elements_per_patch<< std::endl;


    // // debug, compare with reference tensor to see the error
    // SafeTensors referencetensor("/scratch/shdu/transformerExplorer/gemma4_unified_image_debug.safetensors");



// {
//     buffer<float> reference_teacher_patches;
//     referencetensor.load_weights(
//         reference_teacher_patches,
//         "teacher_patches"
//     );

//     print_error_metrics<float, float>(
//         reference_teacher_patches.data(), teacher_patches.data(),
//         1,
//         num_patches, elements_per_patch,
//         num_patches,elements_per_patch
//     );
// }
    // step 4

  
  
    std::vector<int> teacher_position;
    teacher_position.resize(static_cast<size_t>(num_patches) * 2); // now is teacher_positions
    for (int ph = 0; ph < patch_height; ++ph) {
        for (int pw = 0; pw < patch_width; ++pw) {
            int patch_idx = ph * patch_width + pw;
            teacher_position[patch_idx * 2 + 0] = pw; // x 
            teacher_position[patch_idx * 2 + 1] = ph; // y
        }
    }
    int num_model_patches = num_patches / (lm_engine_gemma4e_ptr->GEMMA4_12B_vision_pooling_kernel_size*lm_engine_gemma4e_ptr->GEMMA4_12B_vision_pooling_kernel_size);
    
// {
//     buffer<int64_t> teacher_positions_ref;
//     referencetensor.load_weights(
//         teacher_positions_ref,
//         "teacher_positions"
//     );

//     print_error_metrics<int64_t, int>(
//         teacher_positions_ref.data(), teacher_position.data(),
//         1,
//         num_patches, 2,
//         num_patches,2
//     );
// }

    // Step 5: Merge k×k teacher patches into model patches via patches_merge.
    //   merged_patches   -> [num_model_patches, k*k*elements_per_patch]  (becomes pixel_values)
    //   merged_positions -> [num_model_patches, 2]                        (becomes image_grid_pairs)
    std::vector<float> merged_patches;
    std::vector<int> merged_positions;
    patches_merge(teacher_patches, teacher_position, num_model_patches, merged_patches, merged_positions);
    size_t elements_per_patch = (merged_patches.size() / num_model_patches);
    // std::cout << "num_model_patches " << num_model_patches
    //           << " merged elements_per_patch " << elements_per_patch<< std::endl;

    // // debug: compare merged results with reference tensors (adjust tensor names as needed)
    // {
    //     int merged_dim = static_cast<int>(merged_patches.size() / num_model_patches);
    //     buffer<float> merged_patches_ref;
    //     referencetensor.load_weights(merged_patches_ref, "merged_patches_after_patches_merge");
    //     print_error_metrics<float, float>(
    //         merged_patches_ref.data(), merged_patches.data(),
    //         1, num_model_patches, merged_dim, num_model_patches, merged_dim);
    
    //     buffer<int64_t> merged_positions_ref;
    //     referencetensor.load_weights(merged_positions_ref, "merged_positions_after_patches_merge");
    //     print_error_metrics<int64_t, int>(
    //         merged_positions_ref.data(), merged_positions.data(),
    //         1, num_model_patches, 2, num_model_patches, 2);
    // }



    // Step 6: pad_along_first_dim — pad merged patches (0) and positions (-1)
    // from num_model_patches up to image_softtoken_budget (max_soft_tokens).
    const size_t max_soft_tokens = static_cast<size_t>(this->image_softtoken_budget);

    // pixel_values: [max_soft_tokens, elements_per_patch], padding rows = 0, float -> bf16
    pixel_values.assign(max_soft_tokens * elements_per_patch, static_cast<bf16>(0.0f));
    for (size_t i = 0; i < merged_patches.size(); ++i) {
        pixel_values[i] = static_cast<bf16>(merged_patches[i]);
    }

    // image_grid_pairs: [max_soft_tokens, 2], padding rows = -1
    image_grid_pairs.assign(max_soft_tokens * 2, -1);
    memcpy(image_grid_pairs.data(), merged_positions.data(), merged_positions.size() * sizeof(int));


    num_soft_tokens = num_model_patches;  // but actual text has to handled start and end token
    valid_patch_size = static_cast<uint32_t>(num_model_patches);





    // // debug: compare merged results with reference tensors (adjust tensor names as needed)
    // {
    //     int merged_dim = static_cast<int>(merged_patches.size() / num_model_patches);
    //     buffer<float> merged_patches_ref;
    //     referencetensor.load_weights(merged_patches_ref, "merged_patches_after_patches_merge");
    //     print_error_metrics<float, float>(
    //         merged_patches_ref.data(), merged_patches.data(),
    //         1, num_model_patches, merged_dim, num_model_patches, merged_dim);
    
    //     buffer<int64_t> merged_positions_ref;
    //     referencetensor.load_weights(merged_positions_ref, "merged_positions_after_patches_merge");
    //     print_error_metrics<int64_t, int>(
    //         merged_positions_ref.data(), merged_positions.data(),
    //         1, num_model_patches, 2, num_model_patches, 2);
    // }





    // // store the logical 2D shape of pixel_values: (max_patches, elements_per_patch)
    // // std::cout << "padding of " << padding_length << " patches is applied." << std::endl;
    // // std::cout << "image_grid_pairs size: " << image_grid_pairs.size() << ", expected: " << max_patches * 2 << std::endl;
    patch_element_per_patch.first = image_softtoken_budget;
    patch_element_per_patch.second =  elements_per_patch ;



    

}
