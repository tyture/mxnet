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
namespace gridtarget_util {
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
inline DType Distance(DType x1, DType y1, DType x2, DType y2) {
  DType dx = x1 - x2;
  DType dy = y1 - y2;
  return sqrt(dx * dx + dy * dy);
}

template<typename DType>
inline DType DistanceToCenter(DType x, DType y, DType left, DType top,
                              DType right, DType bottom) {
  DType x2 = (left + right) / 2;
  DType y2 = (top + bottom) / 2;
  return Distance(x, y, x2, y2);
}

template<typename DType>
inline DType CalculateOverlap(DType xmin1, DType ymin1, DType xmax1, DType ymax1,
                              DType xmin2, DType ymin2, DType xmax2, DType ymax2) {
  DType ix = std::max(DType(0), std::min(xmax1, xmax2) - std::max(xmin1, xmin2));
  DType iy = std::max(DType(0), std::min(ymax1, ymax2) - std::max(ymin1, ymin2));
  DType inter = ix * iy;
  if (inter <= 0) return 0;
  DType uni = (xmax1 - xmin1) * (ymax1 - ymin1) + (xmax2 - xmin2) * (ymax2 - ymin2);
  uni -= inter;
  return inter / uni;
                              }
}  // namespace gridtarget_util

template<typename DType>
inline void GridAnchorTargetForward(const Tensor<cpu, 3, DType> &box_target,
                           const Tensor<cpu, 3, DType> &box_mask,
                           const Tensor<cpu, 4, DType> &cls_target,
                           const Tensor<cpu, 2, DType> &anchors,
                           const Tensor<cpu, 3, DType> &labels,
                           const Tensor<cpu, 4, DType> &cls_preds,
                           const Tensor<cpu, 3, DType> &temp_space,
                           float ignore_label,
                           float negative_mining_ratio,
                           int minimum_negative_samples,
                           float size_norm, float core_area,
                           float buffer_area, bool absolute_area) {
  using namespace gridtarget_util;
  CHECK_GE(core_area, 0);
  CHECK_LE(core_area, 1);
  if (buffer_area < core_area) buffer_area = core_area;
  CHECK_LE(buffer_area, 1);
  int num_spatial = static_cast<int>(anchors.size(1));
  int num_anchors = static_cast<int>(cls_preds.size(2));
  int num_classes = static_cast<int>(cls_preds.size(1));
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
      DType cls_id = labels[nbatch][i][0];
      DType gt_xmin = labels[nbatch][i][1];
      DType gt_ymin = labels[nbatch][i][2];
      DType gt_xmax = labels[nbatch][i][3];
      DType gt_ymax = labels[nbatch][i][4];
      for (index_t j = 0; j < num_spatial; ++j) {
        DType anchor_x = anchors[0][j];
        DType anchor_y = anchors[1][j];

        if (anchor_x < gt_xmin || anchor_x > gt_xmax ||
            anchor_y < gt_ymin || anchor_y > gt_ymax) {
          continue;
        }

        // calculate decision areas
        float base_size = 1.f;
        if (!absolute_area) {
          base_size = std::min(gt_xmax - gt_xmin, gt_ymax - gt_ymin);
        }
        float core_size = base_size * core_area;
        float buffer_size = base_size * buffer_area;
        DType dist = DistanceToCenter(anchor_x, anchor_y, gt_xmin, gt_ymin,
          gt_xmax, gt_ymax);

        if (dist < buffer_size) {
          DType best_iou = -1;
          int best_pos = -1;
          for (int k = 0; k < num_anchors; ++k) {
            // find the best matching anchor at this position
            DType anchor_xmin = anchors[2 + k * 4][j];
            DType anchor_ymin = anchors[3 + k * 4][j];
            DType anchor_xmax = anchors[4 + k * 4][j];
            DType anchor_ymax = anchors[5 + k * 4][j];
            DType iou = CalculateOverlap(anchor_xmin, anchor_ymin, anchor_xmax,
              anchor_ymax, gt_xmin, gt_ymin, gt_xmax, gt_ymax);
            if (iou > best_iou) {
              best_iou = iou;
              best_pos = k;
            }
          }
          CHECK_GE(best_pos, 0);
          if (cls_target[nbatch][0][best_pos][j] > 0) {
            // already marked as positive class, means conflict, mark as ignore
            cls_target[nbatch][0][best_pos][j] = ignore_label;
            box_target[nbatch][best_pos * 4][j] = 0;  // left
            box_target[nbatch][best_pos * 4 + 1][j] = 0;  // top
            box_target[nbatch][best_pos * 4 + 2][j] = 0;  // right
            box_target[nbatch][best_pos * 4 + 3][j] = 0;  // bottom
            box_mask[nbatch][best_pos * 4][j] = 0;
            box_mask[nbatch][best_pos * 4 + 1][j] = 0;
            box_mask[nbatch][best_pos * 4 + 2][j] = 0;
            box_mask[nbatch][best_pos * 4 + 3][j] = 0;
            --num_pos;
            ++num_ignore;
          } else if (cls_target[nbatch][0][best_pos][j] == init_value) {
            if (dist < core_size) {
              // mark as positive
              cls_target[nbatch][0][best_pos][j] = cls_id + 1;  // 0 reserved for background
              box_target[nbatch][best_pos * 4][j] = (gt_xmin - anchor_x) / size_norm;  // left
              box_target[nbatch][best_pos * 4 + 1][j] = (gt_ymin - anchor_y) / size_norm;  // top
              box_target[nbatch][best_pos * 4 + 2][j] = (gt_xmax - anchor_x) / size_norm;  // right
              box_target[nbatch][best_pos * 4 + 3][j] = (gt_ymax - anchor_y) / size_norm;  // bottom
              box_mask[nbatch][best_pos * 4][j] = 1;
              box_mask[nbatch][best_pos * 4 + 1][j] = 1;
              box_mask[nbatch][best_pos * 4 + 2][j] = 1;
              box_mask[nbatch][best_pos * 4 + 3][j] = 1;
              ++num_pos;
            } else {
              // mark as ignore
              cls_target[nbatch][0][best_pos][j] = ignore_label;
              box_target[nbatch][best_pos * 4][j] = 0;  // left
              box_target[nbatch][best_pos * 4 + 1][j] = 0;  // top
              box_target[nbatch][best_pos * 4 + 2][j] = 0;  // right
              box_target[nbatch][best_pos * 4 + 3][j] = 0;  // bottom
              box_mask[nbatch][best_pos * 4][j] = 0;
              box_mask[nbatch][best_pos * 4 + 1][j] = 0;
              box_mask[nbatch][best_pos * 4 + 2][j] = 0;
              box_mask[nbatch][best_pos * 4 + 3][j] = 0;
              ++num_ignore;
            }
          }
        }
      }  // end iterate spatial
    }  // end iterate labels

    if (negative_mining_ratio > 0) {
      int num_neg = num_anchors * num_spatial - num_pos - num_ignore;
      int num_tmp = static_cast<int>(negative_mining_ratio * num_pos);
      if (num_tmp < minimum_negative_samples) {
        num_tmp = minimum_negative_samples;
      }
      if (num_tmp > num_neg) {
        num_tmp = num_neg;
      }
      num_neg = num_tmp;

      std::vector<SortElemDescend> sorter;
      sorter.reserve(num_anchors * num_spatial - num_pos - num_ignore);
      for (int j = 0; j < num_spatial; ++j) {
        // check status of each anchor
        for (int n = 0; n < num_anchors; ++n) {
          if (cls_target[nbatch][0][n][j] == init_value) {
            // calculate class predictions
            DType max_val = cls_preds[nbatch][0][n][j];
            DType max_val_pos = cls_preds[nbatch][1][n][j];
            for (int k = 2; k < num_classes; ++k) {
              DType tmp = cls_preds[nbatch][k][n][j];
              if (tmp > max_val_pos) max_val_pos = tmp;
            }
            if (max_val_pos > max_val) max_val = max_val_pos;
            DType sum = 0.f;
            for (int k = 0; k < num_classes; ++k) {
              DType tmp = cls_preds[nbatch][k][n][j];
              sum += std::exp(tmp - max_val);
            }
            max_val_pos = std::exp(max_val_pos - max_val) / sum;
            sorter.push_back(SortElemDescend(max_val_pos, j * num_anchors + n));
          }
        }
      }
      CHECK_EQ(sorter.size(), num_anchors * num_spatial - num_pos - num_ignore);
      std::stable_sort(sorter.begin(), sorter.end());
      for (int k = 0; k < num_neg; ++k) {
        int idx = sorter[k].index;
        int x = idx % num_anchors;
        int y = idx / num_anchors;
        cls_target[nbatch][0][x][y] = 0;  // 0 as background
      }
      for (index_t j = 0; j < anchors.size(2); ++j) {
        for (int n = 0; n < num_anchors; ++n) {
          // mark rest as ignored
          if (cls_target[nbatch][0][n][j] == init_value) {
            cls_target[nbatch][0][n][j] = ignore_label;
          }
        }
      }
    } else {
      // mark all as negative sample
      for (index_t j = 0; j < anchors.size(2); ++j) {
        for (int n = 0; n < num_anchors; ++n) {
          // mark rest as ignored
          if (cls_target[nbatch][0][n][j] == init_value) {
            cls_target[nbatch][0][n][j] = ignore_label;
          }
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
