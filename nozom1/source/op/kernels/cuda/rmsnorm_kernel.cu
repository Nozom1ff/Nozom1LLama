#include <cub/block/block_reduce.cuh>
#include "rmsnorm_kernel.cuh"

// FIXME 这里的做法也是用一个核处理一个最后一维度的向量 ，按最后一维归一化
namespace kernel
{
/**
 * 计算多维输入 input = (dim1, dim2) 在dim2维度上的rmsnorm
 */
// TODO 其实应该展平？
static __global__ void row_rmsnorm_f32_dim(float *input,
                                           float *weight,
                                           float *output,
                                           int dim_size,
                                           int size,
                                           float eps)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    if (bid >= dim_size)
    {
        return;
    }
    // NOTE size就是最后一维的长度，也就是归一化后每个张量的长度
    float *block_input  = input + bid * size;
    float *block_output = output + bid * size;
    // 向量化访存
    const int pack_num = size / 4;
    const int pack_off = pack_num * 4;  // 打包剩下的，最多三个
    // “控制循环次数”
    float sum          = 0.f;
    float4 *input_pack = reinterpret_cast<float4 *>(block_input);
    // 256个线程轮番轰炸，size=n*blockDim.x + (0~3)
    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 input_float4 = *(input_pack + i);
        sum += input_float4.x * input_float4.x;
        sum += input_float4.y * input_float4.y;
        sum += input_float4.z * input_float4.z;
        sum += input_float4.w * input_float4.w;
    }
    // NOTE “Grid-Stride Loop” 设计范式；无论剩余元素多少，步长不能改—— 步长的作用是 “划分线程的遍历范围”，而非
    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        sum += block_input[i] * block_input[i];
    }
    using BlockReduce = cub::BlockReduce<float, 128>;
    // NOTE CUB 内部用于共享内存的临时存储类型，需声明为 __shared__,还需要用typename修饰
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;
    // 倒数
    const float scale   = rsqrtf(sum / static_cast<float>(size) + eps);
    float4 *weight_pack = reinterpret_cast<float4 *>(weight);
    float4 *output_pack = reinterpret_cast<float4 *>(block_output);

    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 in_float4  = *(input_pack + i);
        float4 wei_float4 = *(weight_pack + i);
        *(output_pack + i) = make_float4(scale * in_float4.x * wei_float4.x,
                                         scale * in_float4.y * wei_float4.y,
                                         scale * in_float4.z * wei_float4.z,
                                         scale * in_float4.w * wei_float4.w);
    }
    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        block_output[i] = weight[i] * block_input[i] * scale;
    }
}

// NOTE 这里只用一个BLOCK，意味着把所有tensor看作一维展开
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(float *in, float *wei, float *out, int size, float eps)
{
    const int tid = threadIdx.x;

    constexpr int pack_size = 4;
    const int pack_num      = size / pack_size;
    const int pack_off      = pack_size * pack_num;

    float sum       = 0.0f;
    float4 *in_pack = reinterpret_cast<float4 *>(in);
    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 in_float4 = *(in_pack + i);
        sum += in_float4.x * in_float4.x;
        sum += in_float4.y * in_float4.y;
        sum += in_float4.z * in_float4.z;
        sum += in_float4.w * in_float4.w;
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        sum += in[i] * in[i];
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0)
    {
        shared_val = sum;
    }
    __syncthreads();
    sum               = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    float4 *wei_pack = reinterpret_cast<float4 *>(wei);
    float4 *out_pack = reinterpret_cast<float4 *>(out);
    for (int i = tid; i < pack_num; i += blockDim.x)
    {
        float4 in_float4  = *(in_pack + i);
        float4 wei_float4 = *(wei_pack + i);
        *(out_pack + i)   = make_float4(scale * in_float4.x * wei_float4.x,
                                      scale * in_float4.y * wei_float4.y,
                                      scale * in_float4.z * wei_float4.z,
                                      scale * in_float4.w * wei_float4.w);
    }

    for (int i = pack_off + tid; i < size; i += blockDim.x)
    {
        out[i] = wei[i] * in[i] * scale;
    }
}

void rmsnorm_kernel_cu(const tensor::Tensor &input,
                       const tensor::Tensor &weight,
                       const tensor::Tensor &output,
                       void *stream)
{
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());
    CHECK(input.device_type() == base::DeviceType::kCUDA && weight.device_type() == base::DeviceType::kCUDA &&
          output.device_type() == base::DeviceType::kCUDA);

#if defined(QWEN2_SUPPORT) || defined(QWEN3_SUPPORT)
    const float eps = 1e-6f;
#else
    const float eps = 1e-5f;
#endif
    const int32_t size        = static_cast<int32_t>(input.size());
    float *in_ptr             = const_cast<float *>(input.ptr<float>());
    float *wei_ptr            = const_cast<float *>(weight.ptr<float>());
    float *out_ptr            = const_cast<float *>(output.ptr<float>());
    constexpr int threads_num = 128;
    if (stream)
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<128><<<1, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
    else
    {
        row_rmsnorm_f32<128><<<1, threads_num>>>(in_ptr, wei_ptr, out_ptr, size, eps);
    }
}

void rmsnorm_kernel_cu_dim(const tensor::Tensor &input,
                           const tensor::Tensor &weight,
                           const tensor::Tensor &output,
                           int32_t dim,
                           void *stream)
{
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == base::DeviceType::kCUDA && weight.device_type() == base::DeviceType::kCUDA &&
          output.device_type() == base::DeviceType::kCUDA);

    const float eps          = 1e-6f;
    const int32_t total_size = static_cast<int32_t>(input.size());
    const int32_t size       = input.get_dim(input.dims_size() - 1);
    const int32_t dim_size   = total_size / size;

    float *in_ptr             = const_cast<float *>(input.ptr<float>());
    float *wei_ptr            = const_cast<float *>(weight.ptr<float>());
    float *out_ptr            = const_cast<float *>(output.ptr<float>());
    constexpr int threads_num = 128;
    if (stream)
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32_dim<<<dim_size, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
    else
    {
        row_rmsnorm_f32_dim<<<dim_size, threads_num>>>(in_ptr, wei_ptr, out_ptr, dim_size, size, eps);
    }
}

}  // namespace kernel