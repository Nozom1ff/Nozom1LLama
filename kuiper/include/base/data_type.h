#ifndef KUIPER_INCLUDE_BASE_DATA_TYPE_H_
#define KUIPER_INCLUDE_BASE_DATA_TYPE_H_

#include "base.h"
#include <vector>

// CUDA Stream 类型声明
// 使用条件编译避免与 CUDA 头文件冲突
#ifndef __CUDA_RUNTIME_H__
typedef void* cudaStream_t;
#endif

namespace base {

// ========== FP16 数据类型转换工具类 ==========
// 支持 FP32 ↔ FP16 转换和 INT8 → FP16 转换
// 注意：CPU 端使用位操作模拟，CUDA 端使用内置函数

class DataTypeConverter {
public:
  // ========== FP32 ↔ FP16 转换 (非量化版本) ==========

  /**
   * @brief FP32 转 FP16 (CPU 端)
   * @param src FP32 源数据
   * @param dst FP16 目标缓冲区
   * @param size 元素数量
   */
  static void fp32_to_fp16(const float* src, float16_t* dst, size_t size);

  /**
   * @brief FP16 转 FP32 (CPU 端)
   * @param src FP16 源数据
   * @param dst FP32 目标缓冲区
   * @param size 元素数量
   */
  static void fp16_to_fp32(const float16_t* src, float* dst, size_t size);

  /**
   * @brief FP32 转 FP16 (返回 vector)
   */
  static std::vector<float16_t> fp32_to_fp16(const std::vector<float>& src);

  /**
   * @brief FP16 转 FP32 (返回 vector)
   */
  static std::vector<float> fp16_to_fp32(const std::vector<float16_t>& src);

  // ========== INT8 → FP16 转换 (量化版本) ==========

  /**
   * @brief INT8 权重反量化到 FP16
   * @param int8_weights INT8 权重
   * @param scales 每组的缩放因子 (FP32)
   * @param group_size 量化组大小
   * @param size 权重元素数量
   * @param dst FP16 目标缓冲区
   *
   * 公式: fp16[i] = (float)int8[i] * scales[i / group_size]
   */
  static void int8_to_fp16(const int8_t* int8_weights,
                           const float* scales,
                           int group_size,
                           size_t size,
                           float16_t* dst);

  /**
   * @brief INT8 权重反量化到 FP16 (返回 vector)
   */
  static std::vector<float16_t> int8_to_fp16(
      const std::vector<int8_t>& int8_weights,
      const std::vector<float>& scales,
      int group_size);

  /**
   * @brief INT8 权重反量化到 FP16 (使用 zero_point)
   */
  static void int8_to_fp16(const int8_t* int8_weights,
                           const float* scales,
                           const float* zero_points,
                           int group_size,
                           size_t size,
                           float16_t* dst);

  // ========== GPU 端转换 ==========

  /**
   * @brief GPU 端 FP32 转 FP16
   * @param src_device 设备端 FP32 源数据
   * @param dst_device 设备端 FP16 目标缓冲区
   * @param size 元素数量
   * @param stream CUDA 流
   */
  static void fp32_to_fp16_gpu(const float* src_device,
                               float16_t* dst_device,
                               size_t size,
                               cudaStream_t stream = nullptr);

  /**
   * @brief GPU 端 FP16 转 FP32
   */
  static void fp16_to_fp32_gpu(const float16_t* src_device,
                               float* dst_device,
                               size_t size,
                               cudaStream_t stream = nullptr);

  /**
   * @brief GPU 端 INT8 反量化到 FP16
   */
  static void int8_to_fp16_gpu(const int8_t* int8_weights_device,
                               const float* scales_device,
                               int group_size,
                               size_t size,
                               float16_t* dst_device,
                               cudaStream_t stream = nullptr);

  // ========== 工具函数 ==========

  /**
   * @brief 检查 GPU 是否支持 FP16
   */
  static bool supports_fp16();

  /**
   * @brief 检查 GPU 是否支持 Tensor Core (FP16)
   */
  static bool supports_fp16_tensor_core();

  /**
   * @brief 获取 GPU 架构信息
   */
  static std::string get_gpu_architecture();

  /**
   * @brief 验证 FP16 转换精度
   * @return 最大误差
   */
  static float verify_fp32_to_fp16_precision(const float* fp32_data,
                                            const float16_t* fp16_data,
                                            size_t size);

  /**
   * @brief 验证 INT8 → FP16 转换精度
   */
  static float verify_int8_to_fp16_precision(const int8_t* int8_weights,
                                            const float* scales,
                                            int group_size,
                                            const float16_t* fp16_weights,
                                            size_t size);
};

// ========== 内联辅助函数 (单值转换) ==========

namespace detail {
  // FP32 → FP16 位操作实现 (CPU 端)
  inline uint16_t float_to_float16_bits(float f) {
    union { float f; uint32_t u; } f32;
    f32.f = f;

    uint32_t sign = (f32.u >> 31) & 0x1;
    uint32_t exponent = (f32.u >> 23) & 0xFF;
    uint32_t mantissa = f32.u & 0x7FFFFF;

    if (exponent == 255) {
      // 无穷大或 NaN
      return static_cast<uint16_t>((sign << 15) | 0x7C00 |
                                   ((mantissa != 0) ? 0x0200 : 0));
    }
    if (exponent == 0) {
      // 零
      return static_cast<uint16_t>(sign << 15);
    }

    // 调整指数偏移 (127 -> 15)
    int32_t new_exponent = static_cast<int32_t>(exponent) - 127 + 15;

    if (new_exponent >= 31) {
      return static_cast<uint16_t>((sign << 15) | 0x7C00);  // 无穷大
    } else if (new_exponent <= 0) {
      return static_cast<uint16_t>(sign << 15);  // 零
    } else {
      return static_cast<uint16_t>((sign << 15) |
                                   (new_exponent << 10) |
                                   (mantissa >> 13));
    }
  }

  // FP16 位表示转 FP32 (CPU 端)
  inline float float16_bits_to_float(uint16_t f16_bits) {
    uint32_t sign = (f16_bits >> 15) & 0x1;
    uint32_t exponent = (f16_bits >> 10) & 0x1F;
    uint32_t mantissa = f16_bits & 0x3FF;

    union { float f; uint32_t u; } f32;

    if (exponent == 0) {
      if (mantissa == 0) {
        f32.u = sign << 31;  // 零
      } else {
        f32.u = (sign << 31) | mantissa;  // 次正规数
      }
    } else if (exponent == 31) {
      f32.u = (sign << 31) | 0x7F800000 |
              (mantissa ? (mantissa << 13) : 0);  // 无穷大/NaN
    } else {
      f32.u = (sign << 31) |
              ((exponent + 112) << 23) |
              (mantissa << 13);
    }

    return f32.f;
  }
}

/**
 * @brief 快速 FP32 转 FP16 (单值) - CPU 端版本
 */
inline float16_t fp32_to_fp16(float value) {
  return detail::float_to_float16_bits(value);
}

/**
 * @brief 快速 FP16 转 FP32 (单值) - CPU 端版本
 */
inline float fp16_to_fp32(float16_t value) {
  return detail::float16_bits_to_float(value);
}

/**
 * @brief INT8 转 FP32 再转 FP16 (单值，带 scale) - CPU 端版本
 */
inline float16_t int8_to_fp16(int8_t value, float scale) {
  return fp32_to_fp16(static_cast<float>(value) * scale);
}

}  // namespace base

#endif  // KUIPER_INCLUDE_BASE_DATA_TYPE_H_
