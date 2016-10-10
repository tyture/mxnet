/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor-inl.h
 * \brief generate grid anchors
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_GRID_ANCHOR_INL_H
#define MXNET_OPERATOR_GRID_ANCHOR_INL_H
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <valarray>
#include "./operator_common.h"


namespace mxnet {
namespace op {

namespace gridanchor_enum {
enum GridAnchorOpInputs {kData};
enum GridAnchorOpOutputs {kOut};
}  // namespace gridanchor_enum

struct GridAnchorParam : public dmlc::Parameter<GridAnchorParam> {
  int reserve;
  DMLC_DECLARE_PARAMETER(GridAnchorParam) {
    DMLC_DECLARE_FIELD(reserve).set_default(0)
    .describe("Place-holder for future parameters.");
  }
};  // struct GridAnchorParam

template<typename xpu, typename DType>
class GridAnchorOp : public Operator {
 public:
  explicit GridAnchorOp(GridAnchorParam param) {
    param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(static_cast<int>(in_data.size()), 1);
    CHECK_GE(in_data[gridanchor_enum::kData].ndim(), 4)
      << "Spatial data is required, i.e. width and height";
    int in_height = in_data[gridanchor_enum::kData].size(2);
    CHECK_GT(in_height, 0);
    int in_width = in_data[gridanchor_enum::kData].size(3);
    CHECK_GT(in_width, 0);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // since input sizes are same in each batch, we could share the info
    // i.e. output batch is 1
    Shape<3> oshape = Shape3(2, in_height, in_width);
    Tensor<xpu, 3, DType> out = out_data[gridanchor_enum::kOut]
      .get_with_shape<xpu, 3, DType>(oshape, s);
    GridAnchorForward(out, in_width, in_height);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_states) {
    using namespace mshadow;
    using namespace mshadow::expr;
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // this layer does not pass gradients
    Tensor<xpu, 2, DType> grad = in_grad[gridanchor_enum::kData]
      .FlatTo2D<xpu, DType>(s);
    grad = 0.f;
  }

 private:
  GridAnchorParam param_;
};  // class GridAnchorOp

template<typename xpu>
Operator *CreateOp(GridAnchorParam, int dtype);

#if DMLC_USE_CXX11
class GridAnchorProp: public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 1) << "Inputs: [data]" << in_shape->size();
    TShape dshape = in_shape->at(gridanchor_enum::kData);
    CHECK_EQ(dshape.ndim(), 4) << "Input data should be 4D: batch-channel-y-x";
    int in_height = dshape[2];
    CHECK_GT(in_height, 0) << "Input height should > 0";
    int in_width = dshape[3];
    CHECK_GT(in_width, 0) << "Input width should > 0";
    TShape oshape = dshape;
    oshape[0] = 1;  // share outputs as single batch
    oshape[1] = 2;  // x and y
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GridAnchorProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "GridAnchor";
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  GridAnchorParam param_;
};  // class GridAnchorProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_GRID_ANCHOR_INL_H
