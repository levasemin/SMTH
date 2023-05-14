#include <exception>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>

#include <iostream>
#include <string>
#include <utility>
#include <vector>
#include <filesystem>
#include <deque>
#include <immintrin.h>


void make_filter_intrinsic(uint32_t *array, float *kernel, int picture_h, int picture_w, int kernel_h, int kernel_w)
{
    int max_x = picture_w - 1;
    int max_y = picture_h - 1;
    std::deque<uint32_t *> saved_rows;
    
    for (int i = 0; i < kernel_h / 2 + 1; i++)
    {
        uint32_t *row_copy = new uint32_t[picture_w + kernel_w - 1];

        std::fill(row_copy, row_copy + kernel_w / 2, array[0]);
        std::copy(array, array + picture_w, row_copy + kernel_w / 2);
        std::fill(row_copy + picture_w + kernel_w / 2, row_copy + picture_w + kernel_w - 1, array[picture_w - 1]);
        
        saved_rows.push_back(row_copy);
    }
    
    for (int i = 0; i < kernel_h / 2 + 1; i++)
    {
        uint32_t *row_copy = new uint32_t[picture_w + kernel_w - 1];
        std::fill(row_copy, row_copy + kernel_w / 2, array[i * picture_w]);
        std::copy(array + picture_w * i, array + picture_w * (i + 1), row_copy + kernel_w / 2);
        std::fill(row_copy + picture_w + kernel_w / 2, row_copy + picture_w + kernel_w - 1, array[(i + 1) * picture_w - 1]);
        saved_rows.push_back(row_copy);
    }

    for (int y = 0; y < picture_h; y++)
    {
        uint32_t *current_save = saved_rows.front();
        saved_rows.pop_front();
        int y_ = y + 4 < picture_h ? y + 3 : picture_h - 2;

        std::fill(current_save, current_save + kernel_w / 2, array[(y_ + 1) * picture_w]);
        std::copy(array + (y_ + 1) * picture_w, array + (y_ + 2) * picture_w, current_save + kernel_w / 2);
        std::fill(current_save + picture_w + kernel_w / 2, current_save + picture_w + kernel_w - 1 ,array[(y_ + 2) * picture_w - 1]);

        saved_rows.push_back(current_save);
        
        for (int x = 0; x < picture_w; x++)
        {   
            array[y * picture_w + x] = 0xFF000000;
            
            for (int k = 0; k < 3; k++)
            {
                auto color_result = _mm256_set1_ps(0);
                for (int i = 0; i < kernel_h; i++)
                {
                    __m256i picture_part = _mm256_maskload_epi32((const int *)saved_rows[i] + x, _mm256_set1_epi32(0xFFFFFFFF));
                    __m256i one_color = _mm256_srli_epi32(picture_part, k * 8);
                    __m256i mask = _mm256_set1_epi32(0xFF);
                    one_color = _mm256_and_si256(one_color, mask);
                    
                    auto one_color_ps = _mm256_setr_ps(((int *)&one_color)[0], ((int *)&one_color)[1],
                                                               ((int *)&one_color)[2], ((int *)&one_color)[3],
                                                               ((int *)&one_color)[4], ((int *)&one_color)[5],
                                                               ((int *)&one_color)[6], ((int *)&one_color)[7]);
                                                               
                    __m256 kernel_part =  _mm256_setr_ps(kernel[i * kernel_w],      kernel[i * kernel_w + 1],
                                                         kernel[i * kernel_w + 2],  kernel[i * kernel_w + 3],
                                                         kernel[i * kernel_w + 4],  kernel[i * kernel_w + 5],
                                                         kernel[i * kernel_w + 6],  0); 

                    color_result = _mm256_add_ps(color_result, _mm256_mul_ps(one_color_ps, kernel_part));
                    auto mul = _mm256_mul_ps(one_color_ps, kernel_part);
                }

                float color = 0;

                for (int i = 0; i < kernel_w; i++)
                {
                    color += ((float *)&color_result)[i]; 
                }
                
                array[y * picture_w + x] += uint32_t(uint8_t(color > 0 ? color : 0)) << k * 8; 
            }
        }
    }
}

pybind11::array_t<uint8_t> make_filter(pybind11::array_t<uint8_t> &&arr, pybind11::array_t<float> &&kernel, int picture_x, int picture_y, int kernel_x, int kernel_y)
{
    make_filter_intrinsic((uint32_t *)arr.mutable_data(), (float *)kernel.mutable_data(), picture_y, picture_x, kernel_y, kernel_x);
    std::vector<long> shape;
    shape.push_back(picture_y);
    shape.push_back(picture_x);
    shape.push_back(4); 
    pybind11::array_t<uint8_t> new_arr(pybind11::detail::any_container<long>(std::move(shape)), (uint8_t *)arr.mutable_data());

    return new_arr;
}

PYBIND11_MODULE(Filter, m) {
    m.doc() = "pybind11 Filter plugin"; // optional module docstring
    m.def("make_filter", &make_filter, "A function which multiplies two numbers");
}