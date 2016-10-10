/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor_target.cc
 * \brief training target computation for grid anchors detection, cpu impl
 * \author Joshua Zhang
*/
#include "./grid_anchor_target-inl.h"
#include <cmath>
#include <algorithm>

namespace mshadow {
struct SortElemDescend {
  float value;
  int index;

  SortElemDescend(float v, int i) {
    value = v;
    index = i;
  }

  bool operator<(const SortElemDescend &other) const {
    return value > other.value;
  }
};

template<typename DType>
inline void GridAnchorTargetForward(const Tensor<cpu, 3, DType> &box_target,
                           const Tensor<cpu, 3, DType> &box_mask,
                           const Tensor<cpu, 3, DType> &cls_target,
                           const Tensor<cpu, 3, DType> &anchors,
                           const Tensor<cpu, 3, DType> &labels,
                           const Tensor<cpu, 3, DType> &cls_preds,
                           const Tensor<cpu, 3, DType> &temp_space,
                           float ignore_label,
                           float negative_mining_ratio,
                           int minimum_negative_samples) {
  for (index_t nbatch = 0; nbatch < labels.size(0); ++nbatch) {
    index_t num_valid_gt = 0;
    for (index_t i = 0; i < labels.size(1); ++i) {
      if (static_cast<float>(labels[nbatch][i][0]) == -1.0f) {
        // the rest should be all padding labels with value -1
        CHECK_EQ(static_cast<float>(labels[nbatch][i][1]), -1.0f);
        CHECK_EQ(static_cast<float>(labels[nbatch][i][2]), -1.0f);
        CHECK_EQ(static_cast<float>(labels[nbatch][i][3]), -1.0f);
        CHECK_EQ(static_cast<float>(labels[nbatch][i][4]), -1.0f);
        break;
      }
      ++num_valid_gt;
    }  // end iterate labels

    DType init_value = static_cast<DType>(ignore_label - 1);
    int num_pos = 0;
    int num_ignore = 0;
    for (index_t i = 0; i < num_valid_gt; ++i) {
      for (index_t j = 0; j < anchors.size(2); ++j) {
        DType anchor_x = anchors[nbatch][0][j];
        DType anchor_y = anchors[nbatch][1][j];
        DType cls_id = labels[nbatch][i][0];
        DType gt_xmin = labels[nbatch][i][1];
        DType gt_ymin = labels[nbatch][i][2];
        DType gt_xmax = labels[nbatch][i][3];
        DType gt_ymax = labels[nbatch][i][4];
        if ((anchor_x > gt_xmax) && (anchor_x < gt_xmax)
            && (anchor_y > gt_ymin) && (anchor_y < gt_ymax)) {
          if (cls_target[nbatch][0][j] == init_value) {
            // not marked, good to be a positive grid
            DType gt_x = (gt_xmin + gt_xmax) / 2;
            DType gt_y = (gt_ymin + gt_ymax) / 2;
            DType gt_w = gt_xmax - gt_xmin;
            DType gt_h = gt_ymax - gt_ymin;
            cls_target[nbatch][0][j] = cls_id + 1;  // 0 reserved for background
            box_target[nbatch][0][j] = gt_x - anchor_x;  // x
            box_target[nbatch][1][j] = gt_y - anchor_y;  // y
            box_target[nbatch][2][j] = sqrtf(gt_w);  // width
            box_target[nbatch][3][j] = sqrtf(gt_h);  // height
            box_mask[nbatch][0][j] = 1;
            box_mask[nbatch][1][j] = 1;
            box_mask[nbatch][2][j] = 1;
            box_mask[nbatch][3][j] = 1;
            ++num_pos;
          } else if (cls_target[nbatch][0][j] > 0) {
            // already marked by other label
            // this region belong to multiple objects, mark as don't care
            cls_target[nbatch][0][j] = ignore_label;
            box_target[nbatch][0][j] = 0;
            box_target[nbatch][1][j] = 0;
            box_target[nbatch][2][j] = 0;
            box_target[nbatch][3][j] = 0;
            box_mask[nbatch][0][j] = 0;
            box_mask[nbatch][1][j] = 0;
            box_mask[nbatch][2][j] = 0;
            box_mask[nbatch][3][j] = 0;
            --num_pos;
            ++num_ignore;
          }
        }
      }  // end iterate spatial
    }  // end iterate labels

    if (negative_mining_ratio > 0) {
      int num_neg = static_cast<int>(anchors.size(2)) - num_pos - num_ignore;
      int num_tmp = static_cast<int>(negative_mining_ratio * num_pos);
      if (num_tmp < minimum_negative_samples) {
        num_tmp = minimum_negative_samples;
      }
      if (num_tmp > num_neg) {
        num_tmp = num_neg;
      }
      num_neg = num_tmp;

      std::vector<SortElemDescend> sorter;
      sorter.reserve(static_cast<int>(anchors.size(2)) - num_pos - num_ignore);
      for (index_t j = 0; j < anchors.size(2); ++j) {
        // check status of each anchor
        if (cls_target[nbatch][0][j] == init_value) {
          // calculate class predictions
          DType max_val = cls_preds[nbatch][0][j];
          DType max_val_pos = cls_preds[nbatch][1][j];
          for (index_t k = 2; k < cls_preds.size(1); ++k) {
            DType tmp = cls_preds[nbatch][k][j];
            if (tmp > max_val_pos) max_val_pos = tmp;
          }
          if (max_val_pos > max_val) max_val = max_val_pos;
          DType sum = 0.f;
          for (index_t k = 0; k < cls_preds.size(1); ++k) {
            DType tmp = cls_preds[nbatch][k][j];
            sum += std::exp(tmp - max_val);
          }
          max_val_pos = std::exp(max_val_pos - max_val) / sum;
          sorter.push_back(SortElemDescend(max_val_pos, j));
        }
      }
      CHECK_EQ(sorter.size(), anchors.size(2) - num_pos - num_ignore);
      std::stable_sort(sorter.begin(), sorter.end());
      for (index_t k = 0; k < num_neg; ++k) {
        cls_target[nbatch][0][sorter[k].index] = 0;  // 0 as background
      }
      for (index_t j = 0; j < anchors.size(2); ++j) {
        // mark rest as ignored
        if (cls_target[nbatch][0][j] == init_value) {
          cls_target[nbatch][0][j] = ignore_label;
        }
      }
    } else {
      // mark all as negative sample
      for (index_t j = 0; j < anchors.size(2); ++j) {
        // check status of each anchor
        if (cls_target[nbatch][0][j] == init_value) {
          cls_target[nbatch][0][j] = 0;  // 0 as background
        }
      }
    }
  }  // end iterate batch
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<cpu>(GridAnchorTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GridAnchorTargetOp<cpu, DType>(param);
  });
  return op;
}

Operator* GridAnchorTargetProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                       std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  CHECK(InferType(in_type, &out_type, &aux_type));
  DO_BIND_DISPATCH(CreateOp, param_, in_type->at(0));
}

DMLC_REGISTER_PARAMETER(GridAnchorTargetParam);
MXNET_REGISTER_OP_PROPERTY(GridAnchorTarget, GridAnchorTargetProp)
.describe("Compute GridAnchor training targets")
.add_argument("anchor", "Symbol", "Generated anchor points.")
.add_argument("label", "Symbol", "Object detection labels.")
.add_argument("cls_pred", "Symbol", "Class predictions.")
.add_arguments(GridAnchorTargetParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
