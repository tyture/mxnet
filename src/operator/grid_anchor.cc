/*!
 * Copyright (c) 2016 by Contributors
 * \file gird_anchor.cc
 * \brief generate grid anchors cpu impl
 * \author Joshua Zhang
*/

#include "./grid_anchor-inl.h"

namespace mshadow {
template<typename DType>
inline void GridAnchorForward(const Tensor<cpu, 3, DType> &out,
                              int in_width, int in_height) {
  float step_x = 1.f / in_width;
  float step_y = 1.f / in_height;
  for (int r = 0; r < in_height; ++r) {
    float center_y = (r + 0.5) * step_y;
    for (int c = 0; c < in_width; ++c) {
      float center_x = (c + 0.5) * step_x;
      out[0][r][c] = center_x;
      out[1][r][c] = center_y;
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(GridAnchorParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GridAnchorOp<cpu, DType>(param);
  });
  return op;
}

Operator* GridAnchorProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(GridAnchorParam);

MXNET_REGISTER_OP_PROPERTY(GridAnchor, GridAnchorProp)
.add_argument("data", "Symbol", "Input data.")
.add_arguments(GridAnchorParam::__FIELDS__())
.describe("Generate grid anchors from input data shape.");

}  // namespace op
}  // namespace mxnet
