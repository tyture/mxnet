/*!
 * Copyright (c) 2017 by Contributors
 * \file yolo_output-inl.h
 * \brief yolo-v2 output layer
 * \author Joshua Zhang
*/
#ifndef MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
#define MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <mxnet/base.h>
#include <nnvm/tuple.h>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include "../operator_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"

namespace mxnet {
namespace op {
namespace yoloout_enum {
enum YoloOutputOpInputs {kData, kLabel};
enum YoloOutputOpOutputs {kOut, kTemp, kCopy};
enum YoloOutputOpAuxiliary {kCounter};
enum YoloOutputOpResource {kTempSpace};
}  // namespace yoloout_enum

struct YoloOutputParam : public dmlc::Parameter<YoloOutputParam> {
  int num_class;
  int num_anchor;
  float overlap_thresh;
  float object_grad_scale;
  float background_grad_scale;
  float class_grad_scale;
  float coord_grad_scale;
  nnvm::Tuple<float> anchors;
  int warmup_samples;
  float warmup_grad_scale;
  float nms_threshold;
  int nms_topk;
  bool force_suppress;
  DMLC_DECLARE_PARAMETER(YoloOutputParam) {
    DMLC_DECLARE_FIELD(num_class).set_lower_bound(1)
    .describe("Number of object classes.");
    DMLC_DECLARE_FIELD(num_anchor).set_default(5)
    .set_lower_bound(1)
    .describe("Number of anchors.");
    DMLC_DECLARE_FIELD(overlap_thresh).set_default(0.6)
    .describe("Positive overlap threshold.");
    DMLC_DECLARE_FIELD(object_grad_scale).set_default(1.0)
    .describe("Gradient scale for positive objects.");
    DMLC_DECLARE_FIELD(background_grad_scale).set_default(1.0)
    .describe("Gradient scale for background.");
    DMLC_DECLARE_FIELD(class_grad_scale).set_default(1.0)
    .describe("Gradient scale for positive objects.");
    DMLC_DECLARE_FIELD(coord_grad_scale).set_default(1.0)
    .describe("Gradient scale for box offsets.");
    DMLC_DECLARE_FIELD(anchors)
    .set_default({1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52})
    .describe("Predefined anchor box widths and heights.");
    DMLC_DECLARE_FIELD(warmup_samples).set_default(12800)
    .describe("Number of images to warm up towards averaging position for box "
    "predictions when start a new training. ");
    DMLC_DECLARE_FIELD(warmup_grad_scale).set_default(0.01)
    .describe("Gradient scale for non-critical anchors during warm-up stage.");
    DMLC_DECLARE_FIELD(nms_threshold).set_default(0.5f)
    .describe("Non-maximum suppression threshold.");
    DMLC_DECLARE_FIELD(force_suppress).set_default(false)
    .describe("Suppress all detections regardless of class_id.");
    DMLC_DECLARE_FIELD(nms_topk).set_default(-1)
    .describe("Keep maximum top k detections before nms, -1 for no limit.");
  }
};  // struct YoloOutputParam

template<typename DType>
MSHADOW_XINLINE DType Intersect(DType l1, DType r1, DType l2, DType r2) {
  DType left = l1 > l2 ? l1 : l2;
  DType right = r1 < r2 ? r1 : r2;
  DType w = right - left;
  return w > 0 ? w : DType(0);
}

template<typename DType>
MSHADOW_XINLINE DType Area(DType l1, DType t1, DType r1, DType b1) {
  DType width = r1 - l1;
  DType height = b1 - t1;
  if (width <= 0 || height <= 0) return DType(0);
  return width * height;
}

template<typename DType>
MSHADOW_XINLINE DType IOU(DType l1, DType t1, DType r1, DType b1,
  DType l2, DType t2, DType r2, DType b2) {
  DType inter_area = Intersect(l1, r1, l2, r2) * Intersect(t1, b1, t2, b2);
  if (inter_area <= 0) return DType(0);
  DType area1 = Area(l1, t1, r1, b1);
  DType area2 = Area(l2, t2, r2, b2);
  return inter_area / (area1 + area2 - inter_area);
}

// compute intersection-over-union overlap between two boxes
struct calc_overlap {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out,
      const DType* L1, const DType* T1, const DType* R1, const DType* B1,
      const DType* L2, const DType* T2, const DType* R2, const DType* B2) {
    out[i] = IOU(L1[i], T1[i], R1[i], B1[i], L2[i], T2[i], R2[i], B2[i]);
  }
};

struct clip_zero_one {
  template<typename DType>
  MSHADOW_XINLINE static DType Map(DType a) {
    if (a < 0.f) return DType(0.f);
    if (a > 1.f) return DType(1.f);
    return DType(a);
  }
};  // struct clip_zero_one

// create index mask for labels
// struct index_mask {
//   template<typename DType>
//   MSHADOW_XINLINE static void Map(int i, DType* out,
//       const DType* x, const DType* y, const index_t width, const index_t height,
//       const int stride, const DType on_value) {
//     if (x[i] < 0 || y[i] < 0) return;
//     int depth = width * height * stride;
//     int offset = i * depth;
//     int start = static_cast<int>(y[i] * width + x[i]) * stride;
//     for (int j = 0; j < stride; ++j) {
//       int pos = start + j;
//       if (pos >= 0 && pos < depth) {
//         out[offset + pos] = on_value;
//       }
//     }
//   }
// };

// find best anchor box per ground-truth, and calculate grad
struct box_grad {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* grad, DType* out_label,
      const DType* label, const DType* anchor, const DType* pred,
      const index_t label_width, const index_t label_offset,
      const index_t pred_width, const index_t pred_offset,
      const index_t grad_width, const index_t grad_offset,
      const index_t num_anchor, const index_t num_label,
      const index_t width, const index_t height,
      const float box_scale, const float object_scale) {
    for (int n = 0; n < num_label; ++n) {
      int offset = (i * num_label + n) * label_width;
      DType class_id = label[offset];
      if (class_id < 0) return;  // padded label
      offset += label_offset;
      // ground-truth
      DType gl = label[offset];
      DType gt = label[offset + 1];
      DType gr = label[offset + 2];
      DType gb = label[offset + 3];
      DType gx = (gl + gr) / 2;
      DType gy = (gt + gb) / 2;
      DType gw = gr - gl;
      DType gh = gb - gt;
      if (gx < 0 || gy < 0 || gx > 1 || gy > 1) continue;  // invalid gt
      if (gw <= 0 || gh <= 0 || gw > 1 || gh > 1) continue ;  // invalid gt
      // specific block region only where gt center located
      int col = static_cast<int>(gx * width);
      int row = static_cast<int>(gy * height);
      int best_anchor = 0;
      DType best_ovp = 0;
      // find best anchor
      for (int j = 0; j < num_anchor; ++j) {
        DType aw = anchor[j * 2] / width;
        DType ah = anchor[j * 2 + 1] / height;
        if (aw < 0 || ah < 0) continue;  // invalid param
        DType minw = gw < aw ? gw : aw;
        DType minh = gh < ah ? gh : ah;
        DType ovp = minw * minh;
        ovp = ovp / (gw * gh + aw * ah - ovp);
        if (ovp > best_ovp) {
          best_ovp = ovp;
          best_anchor = j;
        }
      }
      // box prediction and box grad
      // if (i >= 0) {
      //   printf("Batch: %d, Label: %d, Best anchor: %d, row:%d, col:%d, class: %d\n",
      //     i, n, best_anchor, row, col, int(class_id));
      // }
      offset = (i * width * height * num_anchor + row * width * num_anchor +
        col * num_anchor + best_anchor) * pred_width + pred_offset;
      DType px = pred[offset];
      DType py = pred[offset + 1];
      DType pw = pred[offset + 2];
      DType ph = pred[offset + 3];
      int out_offset = (i * width * height * num_anchor + row * width * num_anchor +
        col * num_anchor + best_anchor) * grad_width + grad_offset;
      DType aw = anchor[best_anchor * 2];
      DType ah = anchor[best_anchor * 2 + 1];
      DType scale = box_scale * (2 - gw * gh);
      grad[out_offset] = scale * (px - gx * width + col);  // x
      grad[out_offset + 1] = scale * (py - gy * height + row); // y
      grad[out_offset + 2] = scale * (pw - logf(gw * width / aw));  // w
      grad[out_offset + 3] = scale * (ph - logf(gh * height / ah)); // h
      // if (fabs(grad[out_offset + 3] > 5)) {
      //   printf("best_anchor: %d, gw:%f, gh:%f, scale:%f, tx:%f, ty:%f, tw:%f, th:%f, x:%f, y:%f, w:%f, h:%f, dx:%f, dy:%f, dw:%f, dh:%f\n",
      //     best_anchor, gw, gh, scale, gx * width - col, gy * height - row, log(gw * width / aw), log(gh * height / ah),
      //     px, py, pw, ph, grad[out_offset], grad[out_offset+1], grad[out_offset+2], grad[out_offset+3]);
      // }


      // object grad
      px = (px + col) / width;
      py = (py + row) / height;
      pw = expf(pw) * anchor[best_anchor * 2] / width;
      ph = expf(ph) * anchor[best_anchor * 2 + 1] / height;
      DType iou = IOU(px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2, gl, gt, gr, gb);
      --out_offset;  // layout : num_class + 1 + 4
      --offset;
      grad[out_offset] = object_scale * (pred[offset] - iou);

      // class target
      offset = i * width * height * num_anchor + row * width * num_anchor +
        col * num_anchor + best_anchor;
      out_label[offset] = class_id;
    }

  }
};

/*!
   * \brief Implementation of the non-maximum suppression operation
   *
   * \param i the launched thread index
   * \param index sorted index in descending order
   * \param input the input of nms op
   * \param k nms topk number
   * \param ref compare reference position
   * \param num number of input boxes in each batch
   * \param stride input stride, usually 6 (id-score-x1-y1-x2-y2)
   * \param offset_box box offset, usually 2
   * \param thresh nms threshold
   * \param force force suppress regardless of class id
   * \param offset_id class id offset, used when force == false, usually 0
   * \tparam DType the data type
   */
struct nms_impl {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* index, const DType* input,
                                  int k, int ref, int num, int stride,
                                  int offset_box, int offset_id,
                                  float thresh, bool force) {
    int b = i / k;  // batch
    int pos = i % k + ref + 1;  // position
    if (index[b * num + ref] < 0) return;  // reference has been suppressed
    if (index[b * num + pos] < 0) return;  // self been suppressed
    int ref_offset = static_cast<int>(index[b * num + ref]) * stride + offset_box;
    int pos_offset = static_cast<int>(index[b * num + pos]) * stride + offset_box;
    if (!force) {
      int ref_id = static_cast<int>(input[ref_offset - offset_box + offset_id]);
      int pos_id = static_cast<int>(input[pos_offset - offset_box + offset_id]);
      if (ref_id != pos_id) return;  // different class
    }
    DType refl = input[ref_offset];
    DType reft = input[ref_offset + 1];
    DType refr = input[ref_offset + 2];
    DType refb = input[ref_offset + 3];
    DType pl = input[pos_offset];
    DType pt = input[pos_offset + 1];
    DType pr = input[pos_offset + 2];
    DType pb = input[pos_offset + 3];
    DType iou = IOU(refl, reft, refr, refb, pl, pt, pr, pb);
    if (iou > thresh) {
      // printf("batch: %d, ref: %d, pos: %d, iou: %f\n", b, ref, pos, iou);
      index[b * num + pos] = -1;
    }
  }
};

struct nms_assign {
  template<typename DType>
  MSHADOW_XINLINE static void Map(int i, DType* out, const DType* input,
                                  const DType *index, int k, int num, int stride) {
    int count = 0;
    for (int j = 0; j < k; ++j) {
      int location = static_cast<int>(index[i * num + j]);
      if (location >= 0) {
        // copy to output
        // printf("Batch: %d, Pos: %d, index: %d, %d\n", i, j, location, location % num);
        int out_location = (i * num + count) * stride;
        int in_location = location * stride;
        for (int s = 0; s < stride; ++s) {
          out[out_location + s] = input[in_location + s];
        }
        // if (i == 0) {
        //   printf("pos: %d, score: %f\n", j, float(out[out_location + 1]));
        // }
        ++count;
      }
    }
    // printf("batch: %d, count: %d\n", i, count);
  }
};

template<typename xpu, typename DType>
class YoloOutputOp : public Operator {
 public:
  explicit YoloOutputOp(YoloOutputParam param) {
    this->param_ = param;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
     using namespace mshadow;
     using namespace mshadow::expr;
     using namespace mxnet_op;
     CHECK_EQ(in_data.size(), 2U) << "YoloOutput Input: [data, label]";
     CHECK_EQ(out_data.size(), 3U) << "YoloOutput Output: [output, temp, copy]";
     Stream<xpu> *s = ctx.get_stream<xpu>();
     Tensor<xpu, 4, DType> data = in_data[yoloout_enum::kData]
      .get<xpu, 4, DType>(s);
     Tensor<xpu, 3, DType> out = out_data[yoloout_enum::kOut]
      .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> temp = out_data[yoloout_enum::kTemp]
      .get<xpu, 3, DType>(s);
     Tensor<xpu, 3, DType> out_copy = out_data[yoloout_enum::kCopy]
      .get<xpu, 3, DType>(s);
     Shape<3> tshape = temp.shape_;
     int nc = param_.num_class;
     Shape<3> softmax_shape = Shape3(tshape[0], nc, tshape[1]);
     Shape<1> bias_shape = Shape1(param_.num_anchor * 2);
     Shape<3> anchors_shape = Shape3(tshape[0], tshape[1], 2);
     index_t temp_space_size = 2 * softmax_shape.Size() + bias_shape.Size()
      + anchors_shape.Size();
     Tensor<xpu, 1, DType> temp_buffer = ctx.requested[yoloout_enum::kTempSpace]
      .get_space_typed<xpu, 1, DType>(Shape1(temp_space_size), s);
     CHECK_EQ(temp_buffer.CheckContiguous(), true);
     Tensor<xpu, 3, DType> softmax_in(temp_buffer.dptr_, softmax_shape, s);
     Tensor<xpu, 3, DType> softmax_out(softmax_in.dptr_ + softmax_in.MSize(),
      softmax_shape, s);
     Tensor<xpu, 1, DType> xpu_bias(softmax_out.dptr_ + softmax_out.MSize(),
      bias_shape, s);
     Tensor<xpu, 3, DType> temp_anchors(xpu_bias.dptr_ + xpu_bias.MSize(),
      anchors_shape, s);
     ScalarExp<DType> in_w = ScalarExp<DType>(1.0 / data.shape_[3]);
     ScalarExp<DType> in_h = ScalarExp<DType>(1.0 / data.shape_[2]);

     // change the order of dimensions;
     temp = reshape(transpose(data, Shape4(0, 2, 3, 1)), temp.shape_);
     softmax_in = transpose(slice<2>(temp, 0, nc), Shape3(0, 2, 1));
     Softmax(softmax_out, softmax_in);
     slice<2>(temp, 0, nc) = transpose(softmax_out, Shape3(0, 2, 1));
     // class id to output
     slice<2>(out, 0, 1) = transpose(reduce_keepdim<red::maximum, true>(
       softmax_out, 1), Shape3(0, 2, 1));
     // apply logistic to score, x, y
     slice<2>(temp, nc, nc + 3) = F<mshadow_op::sigmoid>(slice<2>(temp, nc, nc + 3));
     // scores to output
     slice<2>(out, 1, 2) = F<mshadow_op::identity>(slice<2>(
      temp, nc, nc + 1));
     // x = (logistic(pred[0]) + i) / in_w
     tshape[2] = 1;
     slice<2>(out, 2, 3) = in_w * (slice<2>(temp, nc + 1, nc + 2) +
      reshape(broadcast_with_axis(repmat(range<DType>(
      0, data.shape_[3], 1, param_.num_anchor), data.shape_[2]), -1,
      data.shape_[0]), tshape));
     // y = (logistic(pred[1]) + j) / in_h
     slice<2>(out, 3, 4) = in_h * (slice<2>(temp, nc + 2, nc + 3) +
      reshape(broadcast_with_axis(range<DType>(0,
      data.shape_[2], 1, data.shape_[3] * param_.num_anchor), -1, data.shape_[0]),
      tshape));
     // anchors
     nnvm::Tuple<DType> anchors(param_.anchors.begin(), param_.anchors.end());
     Tensor<cpu, 1, DType> cpu_bias(anchors.begin(), Shape1(anchors.ndim()));
    //  Tensor<xpu, 1, DType> xpu_bias = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_space_typed<xpu, 1, DType>(cpu_bias.shape_, s);
     Copy(xpu_bias, cpu_bias, s);
     temp_anchors = reshape(repmat(xpu_bias, data.shape_[0] * data.shape_[2] * data.shape_[3]),
      temp_anchors.shape_);
    //  Tensor<cpu, 2, DType> debug_anchors = ctx.requested[yoloout_enum::kTempSpace]
    //   .get_host_space_typed<2, DType>(Shape2(tshape[1], 2));
    //  Copy(debug_anchors, temp_anchors[0], s);
    //  for (int ii = 0; ii < tshape[1]; ++ii) {
    //    LOG(INFO) << "anchor " << ii << ": " << debug_anchors[ii][0] << ", " << debug_anchors[ii][1];
    //  }
    //  CHECK_EQ(1, 0);
     // w = exp(pred[2]) * anchor[w] / in_w
     slice<2>(out, 4, 5) = in_w * F<mshadow_op::exp>(slice<2>(temp, nc + 3, nc + 4)) *
      slice<2>(temp_anchors, 0, 1);
     // h = exp(pred[3]) * anchor[y] / in_h
     slice<2>(out, 5, 6) = in_h * F<mshadow_op::exp>(slice<2>(temp, nc + 4, nc + 5)) *
      slice<2>(temp_anchors, 1, 2);

     // convert output from x, y, w, h to xmin, ymin, xmax, ymax format
     slice<2>(out, 2, 3) -= ScalarExp<DType>(0.5) * slice<2>(out, 4, 5);
     slice<2>(out, 3, 4) -= ScalarExp<DType>(0.5) * slice<2>(out, 5, 6);
     slice<2>(out, 4, 5) += slice<2>(out, 2, 3);
     slice<2>(out, 5, 6) += slice<2>(out, 3, 4);

     // make copy for backward
     out_copy = F<mshadow_op::identity>(out);

     // clip to boundaries
     slice<2>(out, 2, 6) = F<clip_zero_one>(slice<2>(out, 2, 6));

     // apply nms
     if (param_.nms_threshold > 0 && param_.nms_threshold < 1) {
       int keep = param_.nms_topk < 0 ? out.shape_[1] : param_.nms_topk;
       keep = keep > out.shape_[1] ? out.shape_[1] : keep;
       if (keep > 0) {
         // descend sort by score
         int num_batch = out.shape_[0];
         int num_elem = out.shape_[1];
         Shape<1> sort_index_shape = Shape1(num_batch * num_elem);
         Shape<3> buffer_shape = Shape3(num_batch, num_elem, out.shape_[2]);
         index_t nms_ws_size = 3 * sort_index_shape.Size() + buffer_shape.Size();
         Tensor<xpu, 1, DType> nms_workspace = ctx.requested[yoloout_enum::kTempSpace]
          .get_space_typed<xpu, 1, DType>(Shape1(nms_ws_size), s);
         CHECK_EQ(nms_workspace.CheckContiguous(), true);
         Tensor<xpu, 1, DType> sorted_index(nms_workspace.dptr_, sort_index_shape, s);
         Tensor<xpu, 1, DType> scores(sorted_index.dptr_ + sorted_index.MSize(),
          sort_index_shape, s);
         Tensor<xpu, 1, DType> batch_id(scores.dptr_ + scores.MSize(),
          sort_index_shape, s);
         Tensor<xpu, 3, DType> buffer(batch_id.dptr_ + batch_id.MSize(),
          buffer_shape, s);

         // copy score to buffer, sort accordingly to get sorted index
         scores = reshape(slice<2>(out, 1, 2), scores.shape_);
         sorted_index = range<DType>(0, num_batch * num_elem);
         mshadow::SortByKey(scores, sorted_index, false);
         batch_id = F<mshadow_op::floor>(sorted_index / ScalarExp<DType>(num_elem));
         mshadow::SortByKey(batch_id, scores, true);
         batch_id = F<mshadow_op::floor>(sorted_index / ScalarExp<DType>(num_elem));
         mshadow::SortByKey(batch_id, sorted_index, true);

         // go through each box as reference, suppress if overlap > threshold
         // sorted_index with -1 is marked as suppressed
         for (int ref = 0; ref < keep; ++ref) {
           int num_worker = keep - ref - 1;
           if (num_worker < 1) continue;
           Kernel<nms_impl, xpu>::Launch(s, num_batch * num_worker, sorted_index.dptr_,
             out.dptr_, num_worker, ref, num_elem, out.shape_[2], 2, 0,
             param_.nms_threshold, param_.force_suppress);
         }

         // store the result
         buffer = F<mshadow_op::identity>(out);
         out = -1;
         Kernel<nms_assign, xpu>::Launch(s, num_batch, out.dptr_,
          buffer.dptr_, sorted_index.dptr_, keep, num_elem, out.shape_[2]);
       }
     }

      // Tensor<cpu, 3, DType> debug_bias = ctx.requested[yoloout_enum::kTempSpace]
      //  .get_host_space_typed<3, DType>(temp.shape_);
      // Copy(debug_bias, temp, s);
      // for (int i = 0; i < 845; ++i) {
      //   LOG(INFO) << i << ": " << debug_bias[0][i][0 + 20] << ", " << debug_bias[0][i][21];
      // }

      // Tensor<cpu, 2, DType> debug = ctx.requested[yoloout_enum::kTempSpace]
      //  .get_host_space_typed<2, DType>(Shape2(temp.shape_[1], 6));
      // Copy(debug, out[0], s);
      // for (int ii = 0; ii < debug.shape_[0]; ++ii) {
      //   LOG(INFO) << ii << ": " << debug[ii][0] << ", " << debug[ii][1] << ", "
      //    << debug[ii][2] << ", " << debug[ii][3] << ", " << debug[ii][4] << ", "
      //    << debug[ii][5];
      // }
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
    CHECK_EQ(in_data.size(), 2U) << "YoloOutput Input: [data, label]";
    CHECK_EQ(out_data.size(), 3U) << "YoloOutput Output: [output, temp, copy]";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 3, DType> label = in_data[yoloout_enum::kLabel].get<xpu, 3, DType>(s);
    Tensor<xpu, 4, DType> grad = in_grad[yoloout_enum::kData].get<xpu, 4, DType>(s);
    Tensor<xpu, 3, DType> temp_out = out_data[yoloout_enum::kTemp].get<xpu, 3, DType>(s);
    Tensor<xpu, 3, DType> out_copy = out_data[yoloout_enum::kCopy].get<xpu, 3, DType>(s);
    Tensor<xpu, 1> counter = aux_states[yoloout_enum::kCounter].get<xpu, 1, real_t>(s);
    index_t num_batch = label.shape_[0];
    index_t num_label = label.shape_[1];
    index_t label_width = label.shape_[2];
    index_t pred_width = temp_out.shape_[2];
    index_t num_box = temp_out.shape_[1];
    index_t in_width = grad.shape_[3];
    index_t in_height = grad.shape_[2];
    index_t nc = param_.num_class;
    index_t grad_width = nc + 5;
    index_t num_anchor = param_.num_anchor;
    const DType ignore_label = static_cast<DType>(-1);
    // LOG(INFO) << "Label size: " << num_label;

    // temp space
    Shape<2> label_shape = Shape2(num_batch, num_box);
    Shape<3> softmax_shape = Shape3(num_batch, nc, num_box);
    Shape<4> overlaps_shape = Shape4(9, num_batch, num_box, num_label);
    Shape<3> grad_shape = Shape3(num_batch, num_box, grad_width);
    Shape<1> anchor_shape = Shape1(num_anchor * 2);
    // Shape<4> label_index_shape = Shape4(2, num_batch, num_label, 1);
    // Shape<3> temp_index_mask_shape = Shape3(num_batch, num_label, num_box);
    size_t temp_size_total = label_shape.Size() + 2 * softmax_shape.Size() +
     overlaps_shape.Size() + grad_shape.Size() + anchor_shape.Size();
    // LOG(INFO) << "Total size: " << temp_size_total;
    Tensor<xpu, 1, DType> temp_space = ctx.requested[yoloout_enum::kTempSpace]
     .get_space_typed<xpu, 1, DType>(Shape1(temp_size_total), s);
    CHECK_EQ(temp_space.CheckContiguous(), true);
    // LOG(INFO) << "Total dptr: " << temp_space.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 2, DType> temp_label(temp_space.dptr_, label_shape, s);
    // LOG(INFO) << "Label dptr: " << temp_label.dptr_ << ", " << label_shape.Size();
    Tensor<xpu, 3, DType> temp_softmax(temp_label.dptr_ + temp_label.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmax dptr: " << temp_softmax.dptr_ << ", " << softmax_shape.Size();
    Tensor<xpu, 3, DType> temp_softmax_grad(temp_softmax.dptr_ + temp_softmax.MSize(),
     softmax_shape, s);
    // LOG(INFO) << "softmaxgrad dptr: " << temp_softmax_grad.dptr_ << ", " << softmax_shape.Size();
    // [0]-[7] for x1, y1, w1, h1, x2, y2, w2, h2, [8] for overlap
    Tensor<xpu, 4, DType> buffer(temp_softmax_grad.dptr_ + temp_softmax_grad.MSize(),
     overlaps_shape, s);
    // LOG(INFO) << "overlap dptr: " << buffer.dptr_ << ", " << overlaps_shape.Size();
    Tensor<xpu, 3, DType> temp_grad(buffer.dptr_ + buffer.MSize(),
     grad_shape, s);
    Tensor<xpu, 1, DType> xpu_bias(temp_grad.dptr_ + temp_grad.MSize(),
     anchor_shape, s);

    Shape<3> tshape = Shape3(num_batch, num_box, num_label);
    for (int i = 0; i < 4; ++i) {
      // gt_x1, gt_y1, gt_x2, gt_y2
      buffer[i] = reshape(broadcast_with_axis(slice<2>(label, i + 1, i + 2), 0,
       num_box), tshape);
      // o_x1, o_y1, o_x2, o_y2
      buffer[i + 4] = reshape(broadcast_with_axis(slice<2>(out_copy, i + 2,
       i + 3), 1, num_label), tshape);
    }
    mxnet_op::Kernel<calc_overlap, xpu>::Launch(s, tshape.Size(), buffer[8].dptr_,
     buffer[0].dptr_, buffer[1].dptr_, buffer[2].dptr_, buffer[3].dptr_,
     buffer[4].dptr_, buffer[5].dptr_, buffer[6].dptr_, buffer[7].dptr_);

    // Tensor<cpu, 2, DType> debug = ctx.requested[yoloout_enum::kTempSpace]
    //  .get_host_space_typed<2, DType>(Shape2(num_box, num_label));
    // Copy(debug, buffer[8][0], s);
    // for (int ii = 0; ii < num_box; ++ii) {
    //   for (int jj = 0; jj < num_label; ++jj) {
    //     if (debug[ii][jj] > 0) {
    //       LOG(INFO) << ii << ", " << jj << ": " << debug[ii][jj];
    //     }
    //   }
    // }
    // grad = 0;
    // return;

    // objectness grad
    temp_grad = DType(0);
    slice<2>(temp_grad, nc, nc + 1) = ScalarExp<DType>(param_.background_grad_scale) *
     slice<2>(temp_out, nc, nc + 1);
    // mask out when iou > thresh
    slice<2>(temp_grad, nc, nc + 1) *= F<mshadow_op::lt>(
     reduce_keepdim<red::maximum, false>(buffer[8], 2),
     ScalarExp<DType>(param_.overlap_thresh));

    // optional warm up for initial training stage
    Tensor<cpu, 1, real_t> cpu_counter = ctx.requested[yoloout_enum::kTempSpace]
     .get_host_space_typed<1, real_t>(Shape1(1));
    Copy(cpu_counter, counter, s);
    if (cpu_counter.dptr_[0] < 0) {
      cpu_counter.dptr_[0] = 0;
      counter = 0;
    }
    if (cpu_counter.dptr_[0] < param_.warmup_samples) {
      const ScalarExp<DType> init_scale = ScalarExp<DType>(param_.warmup_grad_scale);
      slice<2>(temp_grad, nc + 1, nc + 2) = init_scale * (
        slice<2>(temp_out, nc + 1, nc + 2) - ScalarExp<DType>(0.5));
      slice<2>(temp_grad, nc + 2, nc + 3) = init_scale * (
        slice<2>(temp_out, nc + 2, nc + 3) - ScalarExp<DType>(0.5));
      slice<2>(temp_grad, nc + 3, nc + 4) = init_scale *
        slice<2>(temp_out, nc + 3, nc + 4);
      slice<2>(temp_grad, nc + 4, nc + 5) = init_scale *
        slice<2>(temp_out, nc + 4, nc + 5);
    }
    if (cpu_counter.dptr_[0] < 3e38) {
      counter += num_batch;
    }

    // find best match for each ground-truth, and calculate grad for box pred
    nnvm::Tuple<DType> anchors(param_.anchors.begin(), param_.anchors.end());
    Tensor<cpu, 1, DType> cpu_bias(anchors.begin(), Shape1(anchors.ndim()));
    Copy(xpu_bias, cpu_bias, s);
    temp_label = ignore_label;  // assign default as ignored
    mxnet_op::Kernel<box_grad, xpu>::Launch(s, num_batch,
     temp_grad.dptr_, temp_label.dptr_, label.dptr_, xpu_bias.dptr_, temp_out.dptr_,
     label_width, 1, pred_width, nc + 1, grad_width, nc + 1,
     num_anchor, num_label, in_width, in_height, param_.coord_grad_scale,
     param_.object_grad_scale);

    // softmax loss
    temp_softmax = transpose(slice<2>(temp_out, 0, nc), Shape3(0, 2, 1));
    SoftmaxGrad(temp_softmax_grad, temp_softmax, temp_label, ignore_label);
    slice<2>(temp_grad, 0, nc) = transpose(temp_softmax_grad, Shape3(0, 2, 1))
      * ScalarExp<DType>(param_.class_grad_scale);

    // apply logistic grad to score, x, y
    slice<2>(temp_grad, nc, nc + 3) *= F<mshadow_op::sigmoid_grad>(
      slice<2>(temp_out, nc, nc + 3));

    // transpose grad to data shape
    grad = transpose(reshape(temp_grad, Shape4(num_batch, in_height,
      in_width, num_anchor * grad_width)), Shape4(0, 3, 1, 2));

    // Tensor<cpu, 3, DType> debug = ctx.requested[yoloout_enum::kTempSpace]
    //  .get_host_space_typed<3, DType>(Shape3(125, in_height, in_width));
    // Copy(debug, grad[0], s);
    // for (int ii = 0; ii < in_height; ++ii) {
    //   for (int jj = 0; jj < in_width; ++jj) {
    //     for (int kk = 0; kk < 5; ++kk) {
    //       std::stringstream ss;
    //       ss << ii << "-" << jj << "-" << kk << ": ";
    //       for (int nn = 0; nn < 25; ++nn) {
    //         ss << "(" << nn << ")" << debug[kk * 25  + nn][jj][ii];
    //       }
    //       if (debug[kk * 25][jj][ii] > 1e10 || debug[kk * 25][jj][ii] < -1e10) {
    //         LOG(INFO) << ss.str();
    //       }
    //     }
    //   }
    // }

    // Tensor<cpu, 2, DType> debug = ctx.requested[yoloout_enum::kTempSpace]
    //  .get_host_space_typed<2, DType>(Shape2(845, 25));
    // Copy(debug, temp_grad[0], s);
    // for (int ii = 0; ii < 845; ++ii) {
    //   if (fabs(debug[ii].dptr_[0]) < 1e-10 ) continue;
    //   std::stringstream ss;
    //   ss << ii << ": ";
    //   for (int jj = 0; jj < 25; ++jj) {
    //     ss << jj << "-" << debug[ii].dptr_[jj] << ", ";
    //   }
    //   LOG(INFO) << ss.str();
    // }

    // store loss in temp_output for metric, [0]:softmax, [1]:object, [2]:box
    // temp_out = 0;
    // Tensor<xpu, 2, DType> metric_out = out_data[yoloout_enum::kTemp]
    //  .get_with_shape<xpu, 2, DType>(Shape2(temp_out.shape_.Size() / num_batch,
    //  num_batch), s);
    // metric_out[0] = sumall_except_dim<0>(F<mshadow_op::abs>(slice<2>(temp_grad, 0, nc)));
    // metric_out[1] = sumall_except_dim<0>(F<mshadow_op::abs>(slice<2>(temp_grad, nc, nc + 1)));
    // metric_out[2] = sumall_except_dim<0>(F<mshadow_op::abs>(slice<2>(temp_grad, nc + 1, nc + 5)));
    // Tensor<cpu, 2, DType> loss_debug = ctx.requested[yoloout_enum::kTempSpace]
    //  .get_host_space_typed<2, DType>(Shape2(3, num_batch));
    // Copy(loss_debug[0], metric_out[0], s);
    // Copy(loss_debug[1], metric_out[1], s);
    // Copy(loss_debug[2], metric_out[2], s);
    // DType l1 = 0;
    // DType l2 = 0;
    // DType l3 = 0;
    // for (int ii = 0; ii < num_batch; ++ii) {
    //   l1 += loss_debug[0][ii];
    //   l2 += loss_debug[1][ii];
    //   l3 += loss_debug[2][ii];
    // }
    // LOG(INFO) << l1 << ", " << l2 << ", " << l3;
  }

 private:
  YoloOutputParam param_;
};  // class YoloOutputOp

template<typename xpu>
Operator *CreateOp(YoloOutputParam, int dtype);

#if DMLC_USE_CXX11
class YoloOutputProp : public OperatorProperty {
 public:
  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  std::vector<std::string> ListArguments() const override {
    return {"data", "label"};
  }

  std::vector<std::string> ListOutputs() const override {
    return {"output", "temp", "copy"};
  }

  std::vector<std::string> ListAuxiliaryStates() const override {
    return {"beta"};
  }

  int NumVisibleOutputs() const override {
    return 1;
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    CHECK_EQ(in_shape->size(), 2U) << "Input:[data, label]";
    const TShape &dshape = in_shape->at(0);
    if (dshape.ndim() == 0) return false;
    if (dshape.ndim() != 4) throw InferShapeError("4-dim data required",
      yoloout_enum::kData);

    // data shape
    CHECK_EQ(param_.anchors.ndim() % 2, 0);
    CHECK_EQ(param_.num_anchor, param_.anchors.ndim() / 2) << "anchor number mismatch";
    int num_channel = param_.num_anchor * (param_.num_class + 1 + 4);
    TShape data_shape = Shape4(dshape[0], num_channel, dshape[2], dshape[3]);
    SHAPE_ASSIGN_CHECK(*in_shape, yoloout_enum::kData, data_shape);
    // label shape
    TShape lshape = in_shape->at(yoloout_enum::kLabel);
    if (lshape.ndim() > 0) {
      CHECK_EQ(lshape.ndim(), 3) << "Label should be [batch-num_labels-(>=5)] tensor";
      CHECK_GT(lshape[1], 0) << "Padded label should > 0";
      CHECK_GE(lshape[2], 5) << "Label width must >=5";
    } else {
      lshape = Shape3(dshape[0], 2, 5);
    }
    SHAPE_ASSIGN_CHECK(*in_shape, yoloout_enum::kLabel, lshape);
    // output shape
    TShape oshape = Shape3(dshape[0], param_.num_anchor * dshape[2] * dshape[3], 6);
    out_shape->clear();
    out_shape->push_back(oshape);
    out_shape->push_back(Shape3(dshape[0], param_.num_anchor * dshape[2] * dshape[3],
      param_.num_class + 4 + 1));
    out_shape->push_back(oshape);
    aux_shape->clear();
    aux_shape->push_back(Shape1(1));
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1U);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    aux_type->clear();
    aux_type->push_back(mshadow::kFloat32);
    out_type->clear();
    out_type->push_back(dtype);  // out
    out_type->push_back(dtype);  // temp
    out_type->push_back(dtype);  // out copy
    return true;
  }

  OperatorProperty* Copy() const override {
    auto ptr = new YoloOutputProp();
    ptr->param_ = param_;
    return ptr;
  }

  std::string TypeString() const override {
    return "_contrib_YoloOutput";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {in_data[yoloout_enum::kLabel], out_data[yoloout_enum::kTemp],
     out_data[yoloout_enum::kCopy]};
  }

  std::vector<ResourceRequest> ForwardResource(
       const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  std::vector<ResourceRequest> BackwardResource(
      const std::vector<TShape> &in_shape) const override {
    return {ResourceRequest::kTempSpace};
  }

  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  YoloOutputParam param_;
};  // YoloOutputProp
#endif  // DMLC_USE_CXX11
}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_CONTRIB_YOLO_OUTPUT_INL_H_
