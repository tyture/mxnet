/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor_detection-inl.h
 * \brief post-process grid anchor predictions cpu impl
 * \author Joshua Zhang
*/
#include "./grid_anchor_detection-inl.h"
#include <algorithm>
#include <cmath>

namespace mshadow {
namespace griddet_util {
template<typename DType>
struct SortElemDescend {
  DType value;
  int index;

  SortElemDescend(DType v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};  // struct SortElemDescend

template<typename DType>
inline DType CalculateOverlap(const DType *a, const DType *b) {
  DType w = std::max(DType(0), std::min(a[2], b[2]) - std::max(a[0], b[0]));
  DType h = std::max(DType(0), std::min(a[3], b[3]) - std::max(a[1], b[1]));
  DType i = w * h;
  DType u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i;
  return u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
inline DType Clip(DType value, DType lower, DType upper) {
  if (value < lower) value = lower;
  if (value > upper) value = upper;
  return value;
}
}

template<typename DType>
inline void GridAnchorDetectionForward(const Tensor<cpu, 3, DType> &out,
                                     const Tensor<cpu, 4, DType> &cls_prob,
                                     const Tensor<cpu, 3, DType> &box_pred,
                                     const Tensor<cpu, 2, DType> &anchors,
                                     float threshold, bool clip,
                                     float size_norm) {
  using namespace griddet_util;
  index_t num_classes = cls_prob.size(1);
  index_t num_spatial = cls_prob.size(3);
  index_t num_anchors = cls_prob.size(2);
  index_t count = 0;
  for (index_t nbatch = 0; nbatch < cls_prob.size(0); ++nbatch) {
    for (index_t i = 0; i < num_spatial; ++i) {
      DType anchor_x = anchors[0][i];
      DType anchor_y = anchors[1][i];
      for (index_t n = 0; n < num_anchors; ++n) {
        DType score = -1;
        int id = 0;
        for (int j = 1; j < num_classes; ++j) {
          DType temp = cls_prob[nbatch][j][n][i];
          if (temp > score) {
            score = temp;
            id = j;
          }
        }
        if (id > 0 && score < threshold) {
          id = 0;
        }
        // [id, prob, xmin, ymin, xmax, ymax]
        out[nbatch][count][0] = id - 1;  // remove background, restore original id
        out[nbatch][count][1] = (id == 0 ? DType(-1) : score);
        DType xmin = anchor_x + box_pred[nbatch][n * 4][i] * size_norm;
        DType ymin = anchor_y + box_pred[nbatch][n * 4 + 1][i] * size_norm;
        DType xmax = anchor_x + box_pred[nbatch][n * 4 + 2][i] * size_norm;
        DType ymax = anchor_y + box_pred[nbatch][n * 4 + 3][i] * size_norm;
        DType lower = 0;
        DType upper = 1;
        out[nbatch][count][2] = clip? Clip(xmin, lower, upper) : xmin;
        out[nbatch][count][3] = clip? Clip(ymin, lower, upper) : ymin;
        out[nbatch][count][4] = clip? Clip(xmax, lower, upper) : xmax;
        out[nbatch][count][5] = clip? Clip(ymax, lower, upper) : ymax;
        ++count;
      }
    }
  }
}

template<typename DType>
inline void GridAnchorNonMaximumSuppression(const Tensor<cpu, 3, DType> &out,
                                  const Tensor<cpu, 3, DType> &temp_space,
                                  float nms_threshold, bool force_suppress) {
  using namespace griddet_util;
  Copy(temp_space, out, out.stream_);
  index_t num_anchors = out.size(1);
  for (index_t nbatch = 0; nbatch < out.size(0); ++nbatch) {
    DType *pout = out.dptr_ + nbatch * num_anchors * 6;
    // sort confidence in descend order
    std::vector<SortElemDescend<DType>> sorter;
    sorter.reserve(num_anchors);
    for (index_t i = 0; i < num_anchors; ++i) {
      DType id = pout[i * 6];
      if (id >= 0) {
        sorter.push_back(SortElemDescend<DType>(pout[i * 6 + 1], i));
      } else {
        sorter.push_back(SortElemDescend<DType>(DType(0), i));
      }
    }
    std::stable_sort(sorter.begin(), sorter.end());
    // re-order output
    DType *ptemp = temp_space.dptr_ + nbatch * num_anchors * 6;
    for (index_t i = 0; i < sorter.size(); ++i) {
      for (index_t j = 0; j < 6; ++j) {
        pout[i * 6 + j] = ptemp[sorter[i].index * 6 + j];
      }
    }
    // apply nms
    for (index_t i = 0; i < num_anchors; ++i) {
      index_t offset_i = i * 6;
      if (pout[offset_i] < 0) continue;  // skip eliminated
      for (index_t j = i + 1; j < num_anchors; ++j) {
        index_t offset_j = j * 6;
        if (pout[offset_j] < 0) continue;  // skip eliminated
        if (force_suppress || (pout[offset_i] == pout[offset_j])) {
          // when foce_suppress == true or class_id equals
          DType iou = CalculateOverlap(pout + offset_i + 2, pout + offset_j + 2);
          if (iou >= nms_threshold) {
            pout[offset_j] = -1;
          }
        }
      }
    }
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(GridAnchorDetectionParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GridAnchorDetectionOp<cpu, DType>(param);
  });
  return op;
}

Operator* GridAnchorDetectionProp::CreateOperatorEx(Context ctx,
                                                  std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(GridAnchorDetectionParam);
MXNET_REGISTER_OP_PROPERTY(GridAnchorDetection, GridAnchorDetectionProp)
.describe("Convert grid anchor detection predictions.")
.add_argument("cls_prob", "Symbol", "Class probabilities.")
.add_argument("box_pred", "Symbol", "Box regression predictions.")
.add_argument("anchors", "Symbol", "Multibox prior anchor boxes")
.add_arguments(GridAnchorDetectionParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
