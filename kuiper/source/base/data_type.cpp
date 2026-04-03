#include "base/data_type.h"
#include <stdexcept>
#include <cstring>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace base {

// ========== FP16 位操作辅助函数 (CPU 端) ==========

namespace {
  // IEEE 754 FP32 位表示
  union FP32Bits {
    float f;
    uint32_t u;
  };

  // FP32 转 FP16 位表示 (CPU 端实现)
  inline uint16_t float_to_float16_bits(float f) {
    FP32Bits f32;
    f32.f = f;

    uint32_t f32_bits = f32.u;
    uint32_t sign = (f32_bits >> 31) & 0x1;
    uint32_t exponent = (f32_bits >> 23) & 0xFF;
    uint32_t mantissa = f32_bits & 0x7FFFFF;

    // 处理特殊情况
    if (exponent == 255) {
      // 无穷大或 NaN
      return static_cast<uint16_t>((sign << 15) | 0x7C00 |
                                   ((mantissa != 0) ? 0x0200 : 0));
    }

    if (exponent == 0) {
      // 零或次正规数
      return static_cast<uint16_t>(sign << 15);
    }

    // 调整指数偏移 (127 -> 15)
    int32_t new_exponent = static_cast<int32_t>(exponent) - 127 + 15;

    if (new_exponent >= 31) {
      // 上溢
      return static_cast<uint16_t>((sign << 15) | 0x7C00);
    } else if (new_exponent <= 0) {
      // 下溢到零
      return static_cast<uint16_t>(sign << 15);
    } else {
      // 正常数
      return static_cast<uint16_t>((sign << 15) |
                                   (new_exponent << 10) |
                                   (mantissa >> 13));
    }
  }

  // FP16 位表示转 FP32 (CPU 端实现)
  inline float float16_bits_to_float(uint16_t f16_bits) {
    uint32_t sign = (f16_bits >> 15) & 0x1;
    uint32_t exponent = (f16_bits >> 10) & 0x1F;
    uint32_t mantissa = f16_bits & 0x3FF;

    FP32Bits f32;

    if (exponent == 0) {
      if (mantissa == 0) {
        // 零
        f32.u = sign << 31;
      } else {
        // 次正规数
        f32.u = (sign << 31) | mantissa;
      }
    } else if (exponent == 31) {
      // 无穷大或 NaN
      f32.u = (sign << 31) | 0x7F800000 |
              (mantissa ? (mantissa << 13) : 0);
    } else {
      // 正常数
      f32.u = (sign << 31) |
              ((exponent + 112) << 23) |
              (mantissa << 13);
    }

    return f32.f;
  }
}

// ========== FP32 ↔ FP16 转换实现 ==========

void DataTypeConverter::fp32_to_fp16(const float* src, float16_t* dst, size_t size) {
  if (!src || !dst) {
    throw std::invalid_argument("src or dst is null");
  }

  for (size_t i = 0; i < size; ++i) {
    dst[i] = float_to_float16_bits(src[i]);
  }
}

void DataTypeConverter::fp16_to_fp32(const float16_t* src, float* dst, size_t size) {
  if (!src || !dst) {
    throw std::invalid_argument("src or dst is null");
  }

  for (size_t i = 0; i < size; ++i) {
    dst[i] = float16_bits_to_float(src[i]);
  }
}

std::vector<float16_t> DataTypeConverter::fp32_to_fp16(const std::vector<float>& src) {
  std::vector<float16_t> dst(src.size());
  fp32_to_fp16(src.data(), dst.data(), src.size());
  return dst;
}

std::vector<float> DataTypeConverter::fp16_to_fp32(const std::vector<float16_t>& src) {
  std::vector<float> dst(src.size());
  fp16_to_fp32(src.data(), dst.data(), src.size());
  return dst;
}

// ========== INT8 → FP16 转换实现 ==========

void DataTypeConverter::int8_to_fp16(const int8_t* int8_weights,
                                     const float* scales,
                                     int group_size,
                                     size_t size,
                                     float16_t* dst) {
  if (!int8_weights || !scales || !dst) {
    throw std::invalid_argument("int8_weights, scales, or dst is null");
  }
  if (group_size <= 0) {
    throw std::invalid_argument("group_size must be positive");
  }

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < size; ++i) {
    int group_idx = static_cast<int>(i) / group_size;
    // INT8 → FP32 → FP16
    float dequantized = static_cast<float>(int8_weights[i]) * scales[group_idx];
    dst[i] = float_to_float16_bits(dequantized);
  }
}

std::vector<float16_t> DataTypeConverter::int8_to_fp16(
    const std::vector<int8_t>& int8_weights,
    const std::vector<float>& scales,
    int group_size) {

  if (scales.size() != (int8_weights.size() + group_size - 1) / group_size) {
    throw std::invalid_argument("scales size does not match int8_weights size and group_size");
  }

  std::vector<float16_t> fp16_weights(int8_weights.size());
  int8_to_fp16(int8_weights.data(), scales.data(), group_size,
               int8_weights.size(), fp16_weights.data());
  return fp16_weights;
}

void DataTypeConverter::int8_to_fp16(const int8_t* int8_weights,
                                     const float* scales,
                                     const float* zero_points,
                                     int group_size,
                                     size_t size,
                                     float16_t* dst) {
  if (!int8_weights || !scales || !zero_points || !dst) {
    throw std::invalid_argument("input pointer is null");
  }
  if (group_size <= 0) {
    throw std::invalid_argument("group_size must be positive");
  }

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (size_t i = 0; i < size; ++i) {
    int group_idx = static_cast<int>(i) / group_size;
    // INT8 → FP32 (带 zero_point) → FP16
    float dequantized = (static_cast<float>(int8_weights[i]) - zero_points[group_idx]) *
                       scales[group_idx];
    dst[i] = float_to_float16_bits(dequantized);
  }
}

// ========== GPU 端转换存根 (实际实现在 data_type.cu) ==========

void DataTypeConverter::fp32_to_fp16_gpu(const float*, float16_t*, size_t, cudaStream_t) {
  throw std::runtime_error("GPU conversion not available: must be compiled with CUDA");
}

void DataTypeConverter::fp16_to_fp32_gpu(const float16_t*, float*, size_t, cudaStream_t) {
  throw std::runtime_error("GPU conversion not available: must be compiled with CUDA");
}

void DataTypeConverter::int8_to_fp16_gpu(const int8_t*, const float*, int, size_t,
                                         float16_t*, cudaStream_t) {
  throw std::runtime_error("GPU conversion not available: must be compiled with CUDA");
}

// ========== 工具函数实现 ==========

// 注意: supports_fp16(), supports_fp16_tensor_core(), get_gpu_architecture()
// 这三个函数在 data_type.cu 中实现，以支持 GPU 能力检测

float DataTypeConverter::verify_fp32_to_fp16_precision(const float* fp32_data,
                                                       const float16_t* fp16_data,
                                                       size_t size) {
  if (!fp32_data || !fp16_data) {
    throw std::invalid_argument("input pointer is null");
  }

  float max_error = 0.0f;
  float sum_error = 0.0f;

  for (size_t i = 0; i < size; ++i) {
    float fp32_value = fp32_data[i];
    float converted = float16_bits_to_float(fp16_data[i]);
    float error = std::abs(fp32_value - converted);
    max_error = std::max(max_error, error);
    sum_error += error;
  }

  return max_error;
}

float DataTypeConverter::verify_int8_to_fp16_precision(const int8_t* int8_weights,
                                                       const float* scales,
                                                       int group_size,
                                                       const float16_t* fp16_weights,
                                                       size_t size) {
  if (!int8_weights || !scales || !fp16_weights) {
    throw std::invalid_argument("input pointer is null");
  }

  float max_error = 0.0f;
  float sum_error = 0.0f;

  for (size_t i = 0; i < size; ++i) {
    int group_idx = static_cast<int>(i) / group_size;
    float expected = static_cast<float>(int8_weights[i]) * scales[group_idx];
    float actual = float16_bits_to_float(fp16_weights[i]);
    float error = std::abs(expected - actual);
    max_error = std::max(max_error, error);
    sum_error += error;
  }

  return max_error;
}

}  // namespace base
