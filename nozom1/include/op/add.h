#ifndef __NOZOM1_INCLUDE_OP_ADD_H__
#define __NOZOM1_INCLUDE_OP_ADD_H__
#include "base/base.h"
#include "layer.h"

namespace op
{
class VecAddLayer : public Layer
{
public:
    explicit VecAddLayer(base::DeviceType device_type);

    base::Status check() const override;

    base::Status forward() override;
};
}  // namespace op
#endif