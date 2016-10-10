/*!
 * Copyright (c) 2016 by Contributors
 * \file grid_anchor_target.cc
 * \brief training target computation for grid anchors detection, cuda impl
 * \author Joshua Zhang
*/
#include "./grid_anchor_target-inl.h"

#define WARPS_PER_BLOCK 16
#define THREADS_PER_WARP 32

#define GRID_ANCHOR_TARGET_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void GridFindMatches(DType *cls_target, DType *box_target,
                            DType *box_mask, const DType *anchors,
                            const DType *labels, float ignore_label,
                            int num_batches, int num_labels, int num_spatial) {
  const float init_value = ignore_label - 1;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_spatial) return;
  int b = index / num_spatial;
  int l = index % num_spatial;
  const DType *p_label = labels + b * num_labels * 5;
  const DType *p_anchor = anchors + b * num_spatial * 2;
  DType *p_cls_target = cls_target + b * num_spatial;
  DType *p_box_target = box_target + b * num_spatial * 4;
  DType *p_box_mask = box_mask + b * num_spatial * 4;
  DType anchor_x = p_anchor[l];
  DType anchor_y = p_anchor[l + num_spatial];
  for (int i = 0; i < num_labels; ++i) {
    if (p_label[i * 5] == DType(-1.f)) {
      break;
    }
    DType cls_id = p_label[i * 5];
    DType gt_xmin = p_label[i * 5 + 1];
    DType gt_ymin = p_label[i * 5 + 2];
    DType gt_xmax = p_label[i * 5 + 3];
    DType gt_ymax = p_label[i * 5 + 4];
    if ((anchor_x > gt_xmax) && (anchor_x < gt_xmax)
        && (anchor_y > gt_ymin) && (anchor_y < gt_ymax)) {
      if (p_cls_target[l] == init_value) {
        // not marked, good to be a positive grid
        DType gt_x = (gt_xmin + gt_xmax) / 2;
        DType gt_y = (gt_ymin + gt_ymax) / 2;
        DType gt_w = gt_xmax - gt_xmin;
        DType gt_h = gt_ymax - gt_ymin;
        p_cls_target[l] = cls_id + 1;  // 0 reserved for background
        p_box_target[l] = gt_x - anchor_x;  // x
        p_box_target[l + num_spatial] = gt_y - anchor_y;  // y
        p_box_target[l + 2 * num_spatial] = sqrt(gt_w);  // width
        p_box_target[l + 3 * num_spatial] = sqrt(gt_h);  // height
        p_box_mask[l] = 1;
        p_box_mask[l + num_spatial] = 1;
        p_box_mask[l + 2 * num_spatial] = 1;
        p_box_mask[l + 3 * num_spatial] = 1;
      } else if (p_cls_target[l] > 0) {
        // already marked by other label
        // this region belong to multiple objects, mark as don't care
        p_cls_target[l] = ignore_label;
        p_box_target[l] = 0;
        p_box_target[l + num_spatial] = 0;
        p_box_target[l + 2 * num_spatial] = 0;
        p_box_target[l + 3 * num_spatial] = 0;
        p_box_mask[l] = 0;
        p_box_mask[l + num_spatial] = 0;
        p_box_mask[l + 2 * num_spatial] = 0;
        p_box_mask[l + 3 * num_spatial] = 0;
      }
    }
  }
}

template<typename DType>
__global__ void GridNegativeMining(DType *cls_target, DType *temp_space,
                                   const DType *cls_preds, float ignore_label,
                                   float negative_mining_ratio,
                                   int minimum_negative_samples,
                                   int num_batches, int num_spatial,
                                   int num_classes) {
  int nbatch = blockIdx.x;
  cls_target += nbatch * num_spatial;
  temp_space += nbatch * num_spatial * 4;
  cls_preds += nbatch * num_classes * num_spatial;
  const int num_threads = WARPS_PER_BLOCK * THREADS_PER_WARP;
  __shared__ int num_neg;
  __shared__ int count;

  if (threadIdx.x == 0) {
    count = 0;
    // check number of negatives to assign
    int num_pos = 0;
    int num_ignore = 0;
    for (int i = 0; i < num_spatial; ++i) {
      if (cls_target[i] > 0) {
        ++num_pos;
      } else if (cls_target[i] == ignore_label) {
        ++num_ignore;
      }
    }
    num_neg = num_pos * negative_mining_ratio;
    if (num_neg < minimum_negative_samples) {
      num_neg = minimum_negative_samples;
    }
    if (num_neg > (num_spatial - num_pos - num_ignore)) {
      num_neg = num_spatial - num_pos - num_ignore;
    }
  }
  __syncthreads();

  DType init_value = ignore_label - 1;
  for (int i = threadIdx.x; i < num_spatial; i += num_threads) {
    if (cls_target[i] == init_value) {
      // calculate class prodictions
      DType max_val = cls_preds[i];
      DType max_val_pos = cls_preds[i + num_spatial];
      for (int k = 2; k < num_classes; ++k) {
        DType tmp = cls_preds[i + k * num_spatial];
        if (tmp > max_val_pos) max_val_pos = tmp;
      }
      if (max_val_pos > max_val) max_val = max_val_pos;
      DType sum = 0.f;
      for (int k = 0; k < num_classes; ++k) {
        DType tmp = cls_preds[i + k * num_spatial];
        sum += exp(tmp - max_val);
      }
      max_val_pos = exp(max_val_pos - max_val) / sum;
      // use buffer to store temporal score and index
      temp_space[count] = max_val_pos;  // score
      temp_space[count + num_spatial] = i;  // index
      ++count;
    }
  }
  __syncthreads();

  // merge sort
  DType *index_src = temp_space + num_spatial * 2;
  DType *index_dst = temp_space + num_spatial * 3;
  DType *src = index_src;
  DType *dst = index_dst;
  for (int i = threadIdx.x; i < count; i += num_threads) {
    index_src[i] = i;
  }
  __syncthreads();

  for (int width = 2; width < (count << 1); width <<= 1) {
    int slices = (count - 1) / (num_threads * width) + 1;
    int start = width * threadIdx.x * slices;
    for (int slice = 0; slice < slices; ++slice) {
      if (start >= count) break;
      int middle = start + (width >> 1);
      if (count < middle) middle = count;
      int end = start + width;
      if (count < end) end = count;
      int i = start;
      int j = middle;
      for (int k = start; k < end; ++k) {
        int idx_i = static_cast<int>(src[i]);
        int idx_j = static_cast<int>(src[j]);
        if (i < middle && (j >= end || temp_space[idx_i] > temp_space[idx_j])) {
          dst[k] = src[i];
          ++i;
        } else {
          dst[k] = src[j];
          ++j;
        }
      }
      start += width;
    }
    __syncthreads();
    // swap src/dst
    src = src == index_src? index_dst : index_src;
    dst = dst == index_src? index_dst : index_src;
  }
  __syncthreads();

  for (int i = threadIdx.x; i < count; i += num_threads) {
    int idx = static_cast<int>(temp_space[num_spatial + static_cast<int>(src[i])]);
    if (i < num_neg) {
      cls_target[idx] = 0;
    } else {
      cls_target[idx] = ignore_label;
    }
  }
}

template<typename DType>
__global__ void GridUseAllNegatives(DType *cls_target, float ignore_label,
                                int num_batches, int num_spatial) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_batches * num_spatial) return;
  if (cls_target[idx] == DType(ignore_label - 1)) {
    cls_target[idx] = 0;
  }
}
}  // namespace cuda

template<typename DType>
inline void GridAnchorTargetForward(const Tensor<gpu, 3, DType> &box_target,
                           const Tensor<gpu, 3, DType> &box_mask,
                           const Tensor<gpu, 3, DType> &cls_target,
                           const Tensor<gpu, 3, DType> &anchors,
                           const Tensor<gpu, 3, DType> &labels,
                           const Tensor<gpu, 3, DType> &cls_preds,
                           const Tensor<gpu, 3, DType> &temp_space,
                           float ignore_label,
                           float negative_mining_ratio,
                           int minimum_negative_samples) {
  // checks
  CHECK_EQ(anchors.CheckContiguous(), true);
  CHECK_EQ(labels.CheckContiguous(), true);
  CHECK_EQ(cls_preds.CheckContiguous(), true);
  CHECK_EQ(box_target.CheckContiguous(), true);
  CHECK_EQ(box_mask.CheckContiguous(), true);
  CHECK_EQ(cls_target.CheckContiguous(), true);
  int num_batches = labels.size(0);
  int num_labels = labels.size(1);
  int num_spatial = anchors.size(2);
  int num_classes = cls_preds.size(1);
  CHECK_GE(num_batches, 1);
  CHECK_GE(num_labels, 1);
  CHECK_GE(num_spatial, 1);

  const int num_threads = THREADS_PER_WARP * WARPS_PER_BLOCK;
  int num_blocks;

  // find anchors to ground-truths matches
  num_blocks = (num_batches * num_spatial - 1) / num_threads + 1;
  cuda::GridFindMatches<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
    box_target.dptr_, box_mask.dptr_, anchors.dptr_, labels.dptr_, ignore_label,
    num_batches, num_labels, num_spatial);
  GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());

  // assign negative targets
  if (negative_mining_ratio > 0) {
    num_blocks = num_batches;  // use one block for each batch
    cuda::GridNegativeMining<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
      temp_space.dptr_, cls_preds.dptr_, ignore_label, negative_mining_ratio,
      minimum_negative_samples, num_batches, num_spatial, num_classes);
    GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  } else {
    num_blocks = (num_batches * num_spatial - 1) / num_threads + 1;
    cuda::GridUseAllNegatives<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
      ignore_label, num_batches, num_spatial);
    GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  }
}
}  // namespace mshadow

namespace mxnet {
namespace op {
template<>
Operator *CreateOp<gpu>(GridAnchorTargetParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new GridAnchorTargetOp<gpu, DType>(param);
  });
  return op;
}
}  // namespace op
}  // namespace mxnet
