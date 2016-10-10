/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor_target-inl.h
 * \brief training target computation for grid anchors detection
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_GRID_ANCHOR_TARGET_INL_H_
#define MXNET_OPERATOR_GRID_ANCHOR_TARGET_INL_H_
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

namespace gridtarget_enum {
enum GridAnchorTargetOpInputs {kAnchor, kLabel, kClsPred};
enum GridAnchorTargetOpOutputs {kBox, kBoxMask, kCls};
enum GridAnchorTargetOpResource {kTempSpace};
}  // namespace gridtarget_enum

struct GridAnchorTargetParam : public dmlc::Parameter<GridAnchorTargetParam> {
  float ignore_label;
  float negative_mining_ratio;
  int minimum_negative_samples;
  DMLC_DECLARE_PARAMETER(GridAnchorTargetParam) {
    DMLC_DECLARE_FIELD(ignore_label).set_default(-1.0f)
    .describe("Label for ignored anchors.");
    DMLC_DECLARE_FIELD(negative_mining_ratio).set_default(-1.0f)
    .describe("Max negative to positive samples ratio, use -1 to disable mining");
    DMLC_DECLARE_FIELD(minimum_negative_samples).set_default(0)
    .describe("Minimum number of negative samples.");
  }
};  // struct GridAnchorTargetParam

template<typename xpu, typename DType>
class GridAnchorTargetOp : public Operator {
 public:
  explicit GridAnchorTargetOp(GridAnchorTargetParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    CHECK_EQ(in_data.size(), 3);
    CHECK_EQ(out_data.size(), 3);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    // get all tensors with spatially compressed to 1d
    TShape ashape = in_data[gridtarget_enum::kAnchor].shape_;
    Shape<3> tshape = Shape3(ashape[0], 2, ashape.ProdShape(2, ashape.ndim()));
    Tensor<xpu, 3, DType> anchors = in_data[gridtarget_enum::kAnchor]
      .get_with_shape<xpu, 3, DType>(tshape, s);
    Tensor<xpu, 3, DType> labels = in_data[gridtarget_enum::kLabel]
      .get<xpu, 3, DType>(s);
    tshape[1] = in_data[gridtarget_enum::kClsPred].size(1);
    Tensor<xpu, 3, DType> cls_preds = in_data[gridtarget_enum::kClsPred]
      .get_with_shape<xpu, 3, DType>(tshape, s);
    tshape[1] = in_data[gridtarget_enum::kBox].size(1);
    Tensor<xpu, 3, DType> box_target = out_data[gridtarget_enum::kBox]
      .get_with_shape<xpu, 3, DType>(tshape, s);
    tshape[1] = in_data[gridtarget_enum::kBoxMask].size(1);
    Tensor<xpu, 3, DType> box_mask = out_data[gridtarget_enum::kBoxMask]
      .get_with_shape<xpu, 3, DType>(tshape, s);
    tshape[1] = in_data[gridtarget_enum::kCls].size(1);
    Tensor<xpu, 3, DType> cls_target = out_data[gridtarget_enum::kCls]
      .get_with_shape<xpu, 3, DType>(tshape, s);
    tshape[1] = 4;
    Tensor<xpu, 3, DType> temp_space = ctx.requested[gridtarget_enum::kTempSpace]
      .get_space_typed<xpu, 3, DType>(tshape, s);

    // default values
    box_target = 0.f;
    box_mask = 0.f;
    cls_target = param_.ignore_label - 1;  // initial indicator
    temp_space = 0.f;

    GridAnchorTargetForward(box_target, box_mask, cls_target,
                          anchors, labels, cls_preds, temp_space,
                          param_.ignore_label,
                          param_.negative_mining_ratio,
                          param_.minimum_negative_samples);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  Tensor<xpu, 2, DType> grad = in_grad[gridtarget_enum::kClsPred].FlatTo2D<xpu, DType>(s);
  grad = 0.f;
}

 private:
  GridAnchorTargetParam param_;
};  // class GridAnchorTargetOp

template<typename xpu>
Operator* CreateOp(GridAnchorTargetParam param, int dtype);

#if DMLC_USE_CXX11
class GridAnchorTargetProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"anchor", "label", "cls_pred"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"box_target", "box_mask", "cls_target"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Input: [anchor, label, clsPred]";
    TShape ashape = in_shape->at(gridtarget_enum::kAnchor);
    CHECK_EQ(ashape.ndim(), 4) << "Anchor should be [1-2-height-width] tensor";
    CHECK_EQ(ashape[0], 1) << "Anchors are shared across batches, first dim=1";
    CHECK_EQ(ashape[1], 2) << "Number boxes should > 0";
    CHECK_GE(ashape[2], 1) << "Height should > 0";
    CHECK_GE(ashape[3], 1) << "Width should > 0";
    TShape lshape = in_shape->at(gridtarget_enum::kLabel);
    CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-5] tensor";
    CHECK_GT(lshape[1], 0) << "Padded label should > 0";
    CHECK_EQ(lshape[2], 5) << "Label should be [batch-num_labels-5] tensor";
    TShape pshape = in_shape->at(gridtarget_enum::kClsPred);
    CHECK_GE(pshape.ndim(), ashape.ndim()) << "Class pred dim should == anchor dim";
    CHECK_GE(pshape[1], 2) << "Class number must >= 1";
    CHECK_EQ(pshape[2], ashape[2]) << "Anchor/Prediction height mismatch";
    CHECK_EQ(pshape[3], ashape[3]) << "Anchor/Prediction width mismatch";
    TShape box_shape = ashape;
    box_shape[1] = 4;  // delta_x, delta_y, width, height
    TShape mask_shape = box_shape;
    TShape label_shape = ashape;
    label_shape[1] = 1;  // class label
    out_shape->clear();
    out_shape->push_back(box_shape);
    out_shape->push_back(mask_shape);
    out_shape->push_back(label_shape);
    return true;
  }

  OperatorProperty* Copy() const override {
    GridAnchorTargetProp* GridAnchorTarget_sym = new GridAnchorTargetProp();
    GridAnchorTarget_sym->param_ = this->param_;
    return GridAnchorTarget_sym;
  }

  std::string TypeString() const override {
    return "GridAnchorTarget";
  }

  //  decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {};
  }

  std::vector<ResourceRequest> ForwardResource(
       const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not implemented";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                              std::vector<int> *in_type) const override;

 private:
  GridAnchorTargetParam param_;
};  // class GridAnchorTargetProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_GRID_ANCHOR_TARGET_INL_H_
