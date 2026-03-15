#ifndef __NOZOM1_INCLUDE_OP_ADD_H__
#define __NOZOM1_INCLUDE_OP_ADD_H__
#include "base/base.h"
#include "layer.h"

namespace op
{
class VecAddLayer : public Layer
{
public:
    // NOTE 修改，作者版本硬编码fp32
    explicit VecAddLayer(base::DeviceType device_type, base::DataType data_type = base::DataType::kTypeFp32);

    base::Status check() const override;

    base::Status forward() override;
};
}  // namespace op
#endif