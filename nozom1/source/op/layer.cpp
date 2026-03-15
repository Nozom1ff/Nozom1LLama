#include "op/layer.h"
#include <base/cuda_config.h>
#include <glog/logging.h>
#include <cstdarg>
#include <numeric>
#include <utility>

namespace op
{

BaseLayer::BaseLayer(base::DeviceType device_type,
                     LayerType layer_type,
                     base::DataType data_type,
                     std::string layer_name)
    : device_type_(device_type),
      layer_type_(layer_type),
      data_type_(data_type),
      layer_name_(std::move(layer_name))
{}

base::DataType BaseLayer::data_type() const
{
    return data_type_;
}

LayerType BaseLayer::layer_type() const
{
    return layer_type_;
}

base::Status BaseLayer::set_weight(int32_t idx, const tensor::Tensor &weight)
{
    return base::error::FunctionNotImplement();
}

base::Status BaseLayer::set_weight(int32_t idx,
                                   const std::vector<int32_t> &dims,
                                   const void *weight_ptr,
                                   base::DeviceType device_type)
{
    return base::error::FunctionNotImplement();
}

const std::string &BaseLayer::get_layer_name() const
{
    return layer_name_;
}

void BaseLayer::set_layer_name(const std::string &layer_name)
{
    layer_name_ = layer_name;
}

void BaseLayer::set_data_type(base::DataType data_type)
{
    data_type_ = data_type;
}

base::DeviceType BaseLayer::device_type() const
{
    return device_type_;
}

void BaseLayer::set_device_type(base::DeviceType device_type)
{
    device_type_ = device_type;
}

Layer::Layer(base::DeviceType device_type, LayerType layer_type, base::DataType data_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, data_type, std::move(layer_name))
{}

base::Status Layer::init()
{
    return base::error::Success();
}

base::Status Layer::check_tensor(const tensor::Tensor &tensor,
                                 base::DeviceType device_type,
                                 base::DataType data_type) const
{
    if (tensor.is_empty())
    {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type)
    {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type)
    {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }
    return base::error::Success();
}

base::Status Layer::check_tensor_with_dim(const tensor::Tensor &tensor,
                                          base::DeviceType device_type,
                                          base::DataType data_type,
                                          ...) const
{
    std::va_list args;
    if (tensor.is_empty())
    {
        return base::error::InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type)
    {
        return base::error::InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type)
    {
        return base::error::InvalidArgument("The tensor has a wrong data type.");
    }

    va_start(args, data_type);
    int32_t dims = tensor.dims_size();
    for (int32_t i = 0; i < dims; ++i)
    {
        int32_t dim = va_arg(args, int32_t);
        if (dim != tensor.get_dim(i))
        {
            return base::error::InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
        }
    }
    va_end(args);
    return base::error::Success();
}

void Layer::set_input(int32_t idx, const tensor::Tensor &input)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
}

void Layer::set_input(const std::vector<tensor::Tensor> &inputs)
{
    this->inputs_ = inputs;
}

void Layer::set_output(int32_t idx, const tensor::Tensor &output)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
}

void Layer::set_output(const std::vector<tensor::Tensor> &outputs)
{
    this->outputs_ = outputs;
}

const tensor::Tensor &Layer::get_input(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

tensor::Tensor &Layer::get_input(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

tensor::Tensor &Layer::get_output(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

base::Status Layer::check() const
{
    return base::error::FunctionNotImplement("The check function is not implement yet");
}

const tensor::Tensor &Layer::get_output(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size)
{
    inputs_.resize(size);
}

void Layer::reset_output_size(size_t size)
{
    outputs_.resize(size);
}

void Layer::to_cuda()
{
    for (auto &input : inputs_)
    {
        if (!input.is_empty())
        {
            input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    for (auto &output : outputs_)
    {
        if (!output.is_empty())
        {
            output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}

void Layer::set_cuda_config(std::shared_ptr<kernel::CudaConfig> config)
{
    if (!config)
    {
        return;
    }
    this->cuda_config_ = config;
}

std::shared_ptr<kernel::CudaConfig> Layer::cuda_config() const
{
    return cuda_config_;
}

size_t Layer::input_size() const
{
    return inputs_.size();
}

size_t Layer::output_size() const
{
    return outputs_.size();
}

base::Status Layer::forward(const std::vector<tensor::Tensor> &inputs, const std::vector<tensor::Tensor> &outputs)
{
    // 直接赋值给成员变量（零拷贝，只拷贝 shared_ptr）
    inputs_  = inputs;
    outputs_ = outputs;
    // 调用无参版本做实际工作
    return forward();
}

LayerParam::LayerParam(base::DeviceType device_type, LayerType layer_type, bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, base::DataType::kUnknown, std::move(layer_name)),
      is_quant_layer_(is_quant_layer)
{}

base::Status LayerParam::set_weight(int32_t idx, const tensor::Tensor &weight)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK(weight.data_type() == base::DataType::kTypeFp32);
    if (!weight.is_empty())
    {
        CHECK(weight.device_type() == device_type_);
    }
    weights_.at(idx) = weight;
    return base::error::Success();
}

const tensor::Tensor &LayerParam::get_weight(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

void LayerParam::to_cuda()
{
    Layer::to_cuda();
    for (auto &weight : weights_)
    {
        weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
    if (!scales_.is_empty())
    {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
}

base::Status LayerParam::set_weight(int32_t idx,
                                    const std::vector<int32_t> &dims,
                                    const void *weight_ptr,
                                    base::DeviceType device_type)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK_NE(weight_ptr, nullptr);

    size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
    std::shared_ptr<base::Buffer> buffer =
        std::make_shared<base::Buffer>(size, nullptr, const_cast<void *>(weight_ptr), true);
    if (device_type != base::DeviceType::kUnknown)
    {
        buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_)
    {
        tensor::Tensor weight(base::DataType::kTypeFp32, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        weights_.at(idx) = weight;
    }
    else
    {
        // is quant layer
        tensor::Tensor weight(base::DataType::kTypeInt8, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        weights_.at(idx) = weight;

        const int32_t weight_size = static_cast<int32_t>(weight.size());
        CHECK(weight_size % group_size_ == 0);

        int32_t scale_nums = weight_size / group_size_;

        // 创建 scales tensor
        std::vector<int32_t> scale_dims = {scale_nums};
        tensor::Tensor scales_tensor(base::DataType::kTypeFp32, scale_dims);
        scales_tensor.set_device_type(device_type);

        // 创建 scales buffer
        size_t scales_byte_size                     = scale_nums * sizeof(float);
        std::shared_ptr<base::Buffer> scales_buffer = std::make_shared<base::Buffer>(
            scales_byte_size, nullptr, reinterpret_cast<void *>((int8_t *)weight_ptr + weight_size), true);
        scales_buffer->set_device_type(device_type);
        CHECK(scales_tensor.assign(scales_buffer));

        scales_ = scales_tensor;
    }

    return base::error::Success();
}

void LayerParam::set_scales(const tensor::Tensor &scales)
{
    CHECK(!scales.is_empty());
    this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size)
{
    this->group_size_ = group_size;
}

int32_t LayerParam::get_scale_num() const
{
    CHECK(!scales_.is_empty());
    return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size)
{
    weights_.resize(size);
}

size_t LayerParam::weight_size() const
{
    return weights_.size();
}

tensor::Tensor &LayerParam::get_weight(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

}  // namespace op
