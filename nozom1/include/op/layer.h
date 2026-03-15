#ifndef __NOZOM1_INCLUDE_OP_LAYER_H__
#define __NOZOM1_INCLUDE_OP_LAYER_H__
#include <base/cuda_config.h>
#include <string>
#include <vector>
#include "base/base.h"
#include "tensor/tensor.h"

namespace op
{
class Layer;
enum class LayerType : uint8_t
{
    kLayerUnknown   = 0,
    kLayerLinear    = 1,
    kLayerEncode    = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm   = 4,
    kLayerMatmul    = 5,
    kLayerRoPe      = 6,
    kLayerMHA       = 7,
    kLayerSoftmax   = 8,
    kLayerAdd       = 9,
    kLayerSwiGLU    = 10,
};

class BaseLayer
{
public:
    explicit BaseLayer(base::DeviceType device_type,
                       LayerType layer_type,
                       base::DataType data_type,
                       std::string layer_name = "");

    base::DataType data_type() const;

    LayerType layer_type() const;

    virtual base::Status init() = 0;

    // 1. 主接口：const 引用传递（零拷贝，性能最优）
    virtual base::Status forward(const std::vector<tensor::Tensor> &inputs,
                                 const std::vector<tensor::Tensor> &outputs) = 0;

    // 2. 无参版本：做实际工作（子类重写）
    virtual base::Status forward() = 0;

    virtual void set_input(int32_t idx, const tensor::Tensor &input) = 0;

    virtual void set_output(int32_t idx, const tensor::Tensor &input) = 0;

    // 批量设置输入输出
    virtual void set_input(const std::vector<tensor::Tensor> &inputs) = 0;

    virtual void set_output(const std::vector<tensor::Tensor> &outputs) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual base::Status check() const = 0;

    virtual tensor::Tensor &get_input(int32_t idx) = 0;

    virtual tensor::Tensor &get_output(int32_t idx) = 0;

    virtual const tensor::Tensor &get_input(int32_t idx) const = 0;

    virtual const tensor::Tensor &get_output(int32_t idx) const = 0;

    virtual base::Status set_weight(int32_t idx, const tensor::Tensor &weight);

    virtual base::Status set_weight(int32_t idx,
                                    const std::vector<int32_t> &dims,
                                    const void *weight_ptr,
                                    base::DeviceType device_type = base::DeviceType::kUnknown);


    const std::string &get_layer_name() const;

    void set_layer_name(const std::string &layer_name);

    void set_data_type(base::DataType data_type);

    base::DeviceType device_type() const;

    void set_device_type(base::DeviceType device_type);

protected:
    std::string layer_name_;
    LayerType layer_type_         = LayerType::kLayerUnknown;
    base::DataType data_type_     = base::DataType::kUnknown;
    base::DeviceType device_type_ = base::DeviceType::kUnknown;
};

class Layer : public BaseLayer
{
public:
    explicit Layer(base::DeviceType device_type,
                   LayerType layer_type,
                   base::DataType data_type,
                   std::string layer_name = "");

    base::Status init() override;

    base::Status check_tensor(const tensor::Tensor &tensor,
                              base::DeviceType device_type,
                              base::DataType data_type) const;

    base::Status check_tensor_with_dim(const tensor::Tensor &tensor,
                                       base::DeviceType device_type,
                                       base::DataType data_type,
                                       ...) const;

    base::Status check() const override;

    base::Status forward(const std::vector<tensor::Tensor> &inputs,
                         const std::vector<tensor::Tensor> &outputs) override;

    void set_input(int32_t idx, const tensor::Tensor &input) override;

    void set_output(int32_t idx, const tensor::Tensor &output) override;

    void set_input(const std::vector<tensor::Tensor> &inputs) override;

    void set_output(const std::vector<tensor::Tensor> &outputs) override;

    const tensor::Tensor &get_input(int32_t idx) const override;

    const tensor::Tensor &get_output(int32_t idx) const override;

    tensor::Tensor &get_input(int32_t idx) override;

    tensor::Tensor &get_output(int32_t idx) override;

    size_t input_size() const override;

    size_t output_size() const override;

    void reset_input_size(size_t size);

    void reset_output_size(size_t size);

    virtual void to_cuda();

    void set_cuda_config(std::shared_ptr<kernel::CudaConfig> config);

    std::shared_ptr<kernel::CudaConfig> cuda_config() const;

    // 便捷重载：1 个输入，1 个输出（零开销）
    inline base::Status forward(const tensor::Tensor &input, const tensor::Tensor &output)
    {
        inputs_  = {input};
        outputs_ = {output};
        return forward();
    }

    // 无参版本：做实际工作（子类重写）
    virtual base::Status forward() = 0;

protected:
    std::vector<tensor::Tensor> inputs_;
    std::vector<tensor::Tensor> outputs_;
    std::shared_ptr<kernel::CudaConfig> cuda_config_;
};

class LayerParam : public Layer
{
public:
    explicit LayerParam(base::DeviceType device_type,
                        LayerType layer_type,
                        bool is_quant_layer    = false,
                        std::string layer_name = "");

    size_t weight_size() const;

    void reset_weight_size(size_t size);

    tensor::Tensor &get_weight(int32_t idx);

    const tensor::Tensor &get_weight(int32_t idx) const;

    void to_cuda() override;

    base::Status set_weight(int32_t idx, const tensor::Tensor &weight) override;

    base::Status set_weight(int32_t idx,
                            const std::vector<int32_t> &dims,
                            const void *weight_ptr,
                            base::DeviceType device_type = base::DeviceType::kUnknown) override;

    void set_scales(const tensor::Tensor &scales);

    void set_group_size(int32_t group_size);

    int32_t get_scale_num() const;

protected:
    int32_t group_size_  = 0;
    bool is_quant_layer_ = false;
    tensor::Tensor scales_;
    std::vector<tensor::Tensor> weights_;
};
}  // namespace op
#endif