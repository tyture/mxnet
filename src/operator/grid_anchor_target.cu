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
__device__ void Distance(DType *dist, DType x1, DType y1, DType x2, DType y2) {
  DType dx = x1 - x2;
  DType dy = y1 - y2;
  *dist = sqrt(dx * dx + dy * dy);
}

template<typename DType>
__device__ void DistanceToCenter(DType *dist, DType x, DType y, DType left, DType top,
                              DType right, DType bottom) {
  DType x2 = (left + right) / 2;
  DType y2 = (top + bottom) / 2;
  Distance(dist, x, y, x2, y2);
}

template<typename DType>
__device__ void CalculateOverlap(DType *iou, const DType *a, const DType *b,
                                 int stride_a, int stride_b) {
  // DType al = a[0];
  // DType at = a[stride_a];
  // DType ar = a[stride_a * 2];
  // DType ab = a[stride_a * 3];
  // DType bl = b[0];
  // DType bt = b[stride_b];
  // DType br = b[stride_b * 2];
  // DType bb = b[stride_b * 3];
  // if (threadIdx.x == 0) printf("%f, %f, %f, %f,---, %f, %f, %f, %f\n", al, at, ar, ab, bl, bt, br, bb);
  DType w = max(DType(0), min(a[2 * stride_a], b[2 * stride_b]) - max(a[0], b[0]));
  DType h = max(DType(0), min(a[3 * stride_a], b[3 * stride_b]) - max(a[stride_a], b[stride_b]));
  DType i = w * h;
  DType u = (a[2 * stride_a] - a[0]) * (a[3 * stride_a] - a[stride_a]) +
    (b[2 * stride_b] - b[0]) * (b[3 * stride_b] - b[stride_b]) - i;
  (*iou) =  u <= 0.f ? static_cast<DType>(0) : static_cast<DType>(i / u);
}

template<typename DType>
__global__ void GridFindMatches(DType *cls_target, DType *box_target,
                            DType *box_mask, const DType *anchors,
                            const DType *labels, float ignore_label,
                            int num_batches, int num_labels, int num_spatial,
                            int num_anchors, float size_norm, float core_area,
                            float buffer_area, bool absolute_area) {
  const float init_value = ignore_label - 1;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= num_batches * num_spatial) return;
  int b = index / num_spatial;
  int l = index % num_spatial;
  const DType *p_label = labels + b * num_labels * 5;
  DType *p_cls_target = cls_target + b * num_spatial * num_anchors + l;
  DType *p_box_target = box_target + b * num_spatial * 4 * num_anchors + l;
  DType *p_box_mask = box_mask + b * num_spatial * 4 * num_anchors + l;
  DType anchor_x = anchors[l];
  DType anchor_y = anchors[l + num_spatial];
  const DType *p_anchor = anchors + l + 2 * num_spatial;

  for (int i = 0; i < num_labels; ++i) {
    if (p_label[i * 5] == DType(-1.f)) {
      break;
    }
    DType cls_id = p_label[i * 5];
    DType gt_xmin = p_label[i * 5 + 1];
    DType gt_ymin = p_label[i * 5 + 2];
    DType gt_xmax = p_label[i * 5 + 3];
    DType gt_ymax = p_label[i * 5 + 4];

    if (anchor_x < gt_xmin || anchor_x > gt_xmax ||
        anchor_y < gt_ymin || anchor_y > gt_ymax) {
      continue;
    }

    // calculate decision areas
    float base_size = 1.f;
    if (!absolute_area) {
      float gt_width = gt_xmax - gt_xmin;
      float gt_height = gt_ymax - gt_ymin;
      base_size = gt_width < gt_height ? gt_width : gt_height;
    }
    float core_size = base_size * core_area;
    float buffer_size = base_size * buffer_area;
    DType dist;
    DistanceToCenter(&dist, anchor_x, anchor_y, gt_xmin, gt_ymin,
      gt_xmax, gt_ymax);

    if (dist < buffer_size) {
      DType best_iou = -1;
      int best_pos = -1;
      for (int j = 0; j < num_anchors; ++j) {
        DType iou;
        CalculateOverlap(&iou, p_anchor + j * 4 * num_spatial, p_label + i * 5 + 1, num_spatial, 1);
        if (iou > best_iou) {
          best_iou = iou;
          best_pos = j;
        }
        // if (threadIdx.x == 0) printf("best pos:%d\n", best_pos);
      }
      if (p_cls_target[best_pos * num_spatial] > 0) {
        // already marked as positive class, means conflict, mark as ignore
        p_cls_target[best_pos * num_spatial] = ignore_label;
        p_box_target[best_pos * num_spatial * 4] = 0;
        p_box_target[best_pos * num_spatial * 4 + num_spatial] = 0;
        p_box_target[best_pos * num_spatial * 4 + 2 * num_spatial] = 0;
        p_box_target[best_pos * num_spatial * 4 + 3 * num_spatial] = 0;
        p_box_mask[best_pos * num_spatial * 4] = 0;
        p_box_mask[best_pos * num_spatial * 4 + num_spatial] = 0;
        p_box_mask[best_pos * num_spatial * 4 + 2 * num_spatial] = 0;
        p_box_mask[best_pos * num_spatial * 4 + 3 * num_spatial] = 0;
      } else if (p_cls_target[best_pos * num_spatial] == init_value) {
        if (dist < core_size) {
          // mark as positive
          p_cls_target[best_pos * num_spatial] = cls_id + 1;  // 0 reserved
          p_box_target[best_pos * num_spatial * 4] = (gt_xmin - anchor_x) / size_norm;  // left
          p_box_target[best_pos * num_spatial * 4 + num_spatial] = (gt_ymin - anchor_y) / size_norm;  // top
          p_box_target[best_pos * num_spatial * 4 + 2 * num_spatial] = (gt_xmax - anchor_x) / size_norm;  // right
          p_box_target[best_pos * num_spatial * 4 + 3 * num_spatial] = (gt_ymax - anchor_y) / size_norm;  // bottom
          p_box_mask[best_pos * num_spatial * 4] = 1;
          p_box_mask[best_pos * num_spatial * 4 + num_spatial] = 1;
          p_box_mask[best_pos * num_spatial * 4 + 2 * num_spatial] = 1;
          p_box_mask[best_pos * num_spatial * 4 + 3 * num_spatial] = 1;
        } else {
          p_cls_target[best_pos * num_spatial] = ignore_label;
          p_box_target[best_pos * num_spatial * 4] = 0;
          p_box_target[best_pos * num_spatial * 4 + num_spatial] = 0;
          p_box_target[best_pos * num_spatial * 4 + 2 * num_spatial] = 0;
          p_box_target[best_pos * num_spatial * 4 + 3 * num_spatial] = 0;
          p_box_mask[best_pos * num_spatial * 4] = 0;
          p_box_mask[best_pos * num_spatial * 4 + num_spatial] = 0;
          p_box_mask[best_pos * num_spatial * 4 + 2 * num_spatial] = 0;
          p_box_mask[best_pos * num_spatial * 4 + 3 * num_spatial] = 0;
        }
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
                                   int num_anchors, int num_classes) {
  int nbatch = blockIdx.x;
  cls_target += nbatch * num_spatial * num_anchors;
  temp_space += nbatch * num_spatial * num_anchors * 3;
  cls_preds += nbatch * num_classes * num_spatial * num_anchors;
  const int num_threads = WARPS_PER_BLOCK * THREADS_PER_WARP;
  __shared__ int num_neg;

  if (threadIdx.x == 0) {
    // check number of negatives to assign
    int num_pos = 0;
    int num_ignore = 0;
    for (int i = 0; i < num_spatial * num_anchors; ++i) {
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
    // printf("Pos: %d, ignore; %d, neg: %d\n", num_pos, num_ignore, num_neg);
  }
  __syncthreads();

  // DType init_value = ignore_label - 1;
  for (int i = threadIdx.x; i < num_spatial; i += num_threads) {
    for (int j = 0; j < num_anchors; ++j) {
      int offset = i + j * num_spatial;
      if (cls_target[offset] < ignore_label) {
        // calculate class predictions
        int stride = num_anchors * num_spatial;
        DType max_val = cls_preds[offset];
        DType max_val_pos = cls_preds[offset + stride];
        for (int k = 2; k < num_classes; ++k) {
          DType tmp = cls_preds[offset + k * stride];
          if (tmp > max_val_pos) max_val_pos = tmp;
        }
        if (max_val_pos > max_val) max_val = max_val_pos;
        DType sum = 0.f;
        for (int k = 0; k < num_classes; ++k) {
          DType tmp = cls_preds[offset + k * stride];
          sum += exp(tmp - max_val);
        }
        max_val_pos = exp(max_val_pos - max_val) / sum;
        // use buffer to store temporal score
        temp_space[offset] = max_val_pos;  // score
      } else {
        temp_space[offset] = -1;
      }
    }
  }
  __syncthreads();

  // merge sort
  int count = num_spatial * num_anchors;
  DType *index_src = temp_space + count;
  DType *index_dst = temp_space + count * 2;
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
    int idx = static_cast<int>(src[i]);
    if (i < num_neg) {
      cls_target[idx] = 0;
    } else {
      if (cls_target[idx] < 0) {
        cls_target[idx] = ignore_label;
      }
    }
  }
}

template<typename DType>
__global__ void GridUseAllNegatives(DType *cls_target, float ignore_label,
                                int num) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num) return;
  if (cls_target[idx] == DType(ignore_label - 1)) {
    cls_target[idx] = 0;
  }
}

template<typename DType>
__global__ void PrintOutput(DType *ptr, int num) {
  for (int i = 0; i < num; ++i) {
    if (float(ptr[i]) == -1) continue;
    printf("%d: %f\n", i, float(ptr[i]));
  }
}
}  // namespace cuda

template<typename DType>
inline void GridAnchorTargetForward(const Tensor<gpu, 3, DType> &box_target,
                           const Tensor<gpu, 3, DType> &box_mask,
                           const Tensor<gpu, 4, DType> &cls_target,
                           const Tensor<gpu, 2, DType> &anchors,
                           const Tensor<gpu, 3, DType> &labels,
                           const Tensor<gpu, 4, DType> &cls_preds,
                           const Tensor<gpu, 3, DType> &temp_space,
                           float ignore_label,
                           float negative_mining_ratio,
                           int minimum_negative_samples,
                           float size_norm, float core_area,
                           float buffer_area, bool absolute_area) {
  // checks
  CHECK_EQ(anchors.CheckContiguous(), true);
  CHECK_EQ(labels.CheckContiguous(), true);
  CHECK_EQ(cls_preds.CheckContiguous(), true);
  CHECK_EQ(box_target.CheckContiguous(), true);
  CHECK_EQ(box_mask.CheckContiguous(), true);
  CHECK_EQ(cls_target.CheckContiguous(), true);
  int num_batches = labels.size(0);
  int num_labels = labels.size(1);
  int num_spatial = anchors.size(1);
  int num_anchors = cls_preds.size(2);
  CHECK_EQ((anchors.size(0) - 2) % 4, 0);
  int num_classes = cls_preds.size(1);
  CHECK_GE(num_batches, 1);
  CHECK_GE(num_labels, 1);
  CHECK_GE(num_spatial, 1);
  CHECK_GE(core_area, 0);
  CHECK_LE(core_area, 1);
  if (buffer_area < core_area) buffer_area = core_area;
  CHECK_LE(buffer_area, 1);

  const int num_threads = THREADS_PER_WARP * WARPS_PER_BLOCK;
  int num_blocks;

  // find anchors to ground-truths matches
  num_blocks = (num_batches * num_spatial - 1) / num_threads + 1;
  cuda::GridFindMatches<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
    box_target.dptr_, box_mask.dptr_, anchors.dptr_, labels.dptr_, ignore_label,
    num_batches, num_labels, num_spatial, num_anchors, size_norm, core_area,
    buffer_area, absolute_area);
  GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());

  // assign negative targets
  if (negative_mining_ratio > 0) {
    num_blocks = num_batches;  // use one block for each batch
    cuda::GridNegativeMining<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
      temp_space.dptr_, cls_preds.dptr_, ignore_label, negative_mining_ratio,
      minimum_negative_samples, num_batches, num_spatial, num_anchors, num_classes);
    GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  } else {
    num_blocks = (num_batches * num_spatial * num_anchors - 1) / num_threads + 1;
    cuda::GridUseAllNegatives<DType><<<num_blocks, num_threads>>>(cls_target.dptr_,
      ignore_label, num_batches * num_spatial * num_anchors);
    GRID_ANCHOR_TARGET_CUDA_CHECK(cudaPeekAtLastError());
  }

  // cuda::PrintOutput<DType><<<1,1>>>(cls_target.dptr_, num_spatial);
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
