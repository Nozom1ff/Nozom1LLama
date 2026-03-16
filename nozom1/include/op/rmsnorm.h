#ifndef __NOZOM1_INCLUDE_OP_RMSNORM_H__
#define __NOZOM1_INCLUDE_OP_RMSNORM_H__
#include "layer.h"

namespace op
{
class RmsNormLayer : public LayerParam
{
public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim, base::DataType data_type);

    base::Status check() const override;

    base::Status forward() override;

private:
    int32_t dim_ = 0;
};
}  // namespace op
#endif  // KUIPER_INCLUDE_OP_RMSNORM_H_
