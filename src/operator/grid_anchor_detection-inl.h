/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor_detection-inl.h
 * \brief post-process grid anchor predictions
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_GRID_ANCHOR_DETECTION_INL_H_
#define MXNET_OPERATOR_GRID_ANCHOR_DETECTION_INL_H_
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
namespace griddet_enum {
enum GridAnchorDetectionOpInputs {kClsProb, kBoxPred, kAnchor};
enum GridAnchorDetectionOpOutputs {kOut};
enum GridAnchorDetectionOpResource {kTempSpace};
}  // namespace griddet_enum

struct GridAnchorDetectionParam : public dmlc::Parameter<GridAnchorDetectionParam> {
  bool clip;
  float threshold;
  float nms_threshold;
  bool force_suppress;
  DMLC_DECLARE_PARAMETER(GridAnchorDetectionParam) {
    DMLC_DECLARE_FIELD(clip).set_default(true)
    .describe("Clip out-of-boundary boxes.");
    DMLC_DECLARE_FIELD(threshold).set_default(0.01f)
    .describe("Threshold to be a positive prediction.");
    DMLC_DECLARE_FIELD(nms_threshold).set_default(0.5f)
    .describe("Non-maximum suppression threshold.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
    .describe("Suppress all detections regardless of class_id.");
  }
};  // struct GridAnchorDetectionParam

template<typename xpu, typename DType>
class GridAnchorDetectionOp : public Operator {
 public:
  explicit GridAnchorDetectionOp(GridAnchorDetectionParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     using namespace mshadow;
     using namespace mshadow::expr;
     CHECK_EQ(in_data.size(), 3) << "Input: [cls_prob, box_pred, anchor]";
     CHECK_EQ(out_data.size(), 1);

     Stream<xpu> *s = ctx.get_stream<xpu>();
     TShape cshape = in_data[griddet_enum::kClsProb].shape_;
     Shape<3> tshape = Shape3(cshape[0], cshape[1],
       cshape.ProdShape(2, cshape.ndim()));
     Tensor<xpu, 3, DType> cls_prob = in_data[griddet_enum::kClsProb]
       .get_with_shape<xpu, 3, DType>(tshape, s);
     tshape[1] = in_data[griddet_enum::kBoxPred].shape_[1];
     Tensor<xpu, 3, DType> box_pred = in_data[griddet_enum::kBoxPred]
       .get_with_shape<xpu, 3, DType>(tshape, s);
     tshape[0] = 1;
     tshape[1] = in_data[griddet_enum::kAnchor].shape_[1];
     Tensor<xpu, 3, DType> anchors = in_data[griddet_enum::kAnchor]
       .get_with_shape<xpu, 3, DType>(tshape, s);
     Tensor<xpu, 3, DType> out = out_data[griddet_enum::kOut]
       .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> temp_space = ctx.requested[griddet_enum::kTempSpace]
       .get_space_typed<xpu, 3, DType>(out.shape_, s);

     GridAnchorDetectionForward(out, cls_prob, box_pred, anchors,
       param_.threshold, param_.clip);
     GridAnchorNonMaximumSuppression(out, temp_space, param_.nms_threshold,
       param_.force_suppress);
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
}

 private:
  GridAnchorDetectionParam param_;
};  // class GridAnchorDetectionOp

template<typename xpu>
Operator *CreateOp(GridAnchorDetectionParam, int dtype);

#if DMLC_USE_CXX11
class GridAnchorDetectionProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"cls_prob", "box_pred", "anchor"};
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 3) << "Inputs: [cls_prob, box_pred, anchor]";
    TShape cshape = in_shape->at(griddet_enum::kClsProb);
    TShape bshape = in_shape->at(griddet_enum::kBoxPred);
    TShape ashape = in_shape->at(griddet_enum::kAnchor);
    CHECK_GE(cshape[1], 2) << "Number of classes must > 1";
    CHECK_EQ(cshape.ndim(), bshape.ndim());
    CHECK_EQ(bshape.ndim(), ashape.ndim());
    for (index_t i = 0; i < cshape.ndim(); ++i) {
      if (i == 1) continue;
      CHECK_EQ(cshape[i], bshape[i]) << "Provided: " << cshape << ", " << bshape;
      CHECK_EQ(bshape[i], ashape[i]) << "Provided: " << bshape << ", " << ashape;
    }
    TShape oshape = TShape(3);
    oshape[0] = cshape[0];
    oshape[1] = ashape.ProdShape(2, ashape.ndim());  // num spatial
    oshape[2] = 6;  // [id, prob, xmin, ymin, xmax, ymax]
    out_shape->clear();
    out_shape->push_back(oshape);
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new GridAnchorDetectionProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "GridAnchorDetection";
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
  GridAnchorDetectionParam param_;
};  // class GridAnchorDetectionProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namesapce mxnet

#endif  // MXNET_OPERATOR_GRID_ANCHOR_DETECTION_INL_H_
