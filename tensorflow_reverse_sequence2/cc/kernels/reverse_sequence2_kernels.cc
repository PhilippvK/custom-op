/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/array_ops.cc.
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/reverse_sequence_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <memory>
#include <vector>

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Tlen>
void CheckErrors(OpKernelContext *context, int batch_dim, int seq_dim) {
  const Tensor &input = context->input(0);
  const Tensor &seq_lengths = context->input(1);

  auto seq_lens_t = seq_lengths.vec<Tlen>();

  std::vector<Tlen> seq_lens_vec(seq_lens_t.size());

  // Copy seq_len info down for validity checks
  context->eigen_device<Device>().memcpyDeviceToHost(
      seq_lens_vec.data(), seq_lens_t.data(), sizeof(Tlen) * seq_lens_t.size());

  OP_REQUIRES(context, batch_dim != seq_dim,
              errors::InvalidArgument("batch_dim == seq_dim == ", seq_dim));
  OP_REQUIRES(context, seq_dim < input.dims(),
              errors::InvalidArgument("seq_dim must be < input rank", " ( ",
                                      seq_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, batch_dim < input.dims(),
              errors::InvalidArgument("batch_dim must be < input rank", " ( ",
                                      batch_dim, " vs. ", input.dims(), ")"));
  OP_REQUIRES(context, seq_lengths.NumElements() == input.dim_size(batch_dim),
              errors::InvalidArgument("Length of seq_lengths != input.dims(",
                                      batch_dim, "), ", "(",
                                      seq_lengths.NumElements(), " vs. ",
                                      input.dim_size(batch_dim), ")"));

  for (size_t d = 0; d < seq_lens_vec.size(); ++d) {
    OP_REQUIRES(context, seq_lens_vec[d] >= 0,
                errors::InvalidArgument("seq_lens(", d, ") < 0"));
    OP_REQUIRES(context, seq_lens_vec[d] <= input.dim_size(seq_dim),
                errors::InvalidArgument("seq_lens(", d, ") > input.dims(",
                                        seq_dim, ")"));
  }
}

template <typename Device, typename T, typename Tlen>
class ReverseSequence2Op : public OpKernel {
public:
  explicit ReverseSequence2Op(OpKernelConstruction *context)
      : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("batch_dim", &batch_dim_));
    OP_REQUIRES_OK(context, context->GetAttr("seq_dim", &seq_dim_));
  }

  void Compute(OpKernelContext *context) override {
    const Tensor &input = context->input(0);
    const Tensor &seq_lengths = context->input(1);

    // Preliminary validation of sizes.
    OP_REQUIRES(context, TensorShapeUtils::IsVector(seq_lengths.shape()),
                errors::InvalidArgument("seq_lengths must be 1-dim, not ",
                                        seq_lengths.dims()));

    auto seq_lens_t = seq_lengths.vec<Tlen>();

    CheckErrors<Device, Tlen>(context, batch_dim_, seq_dim_);
    if (!context->status().ok())
      return;

    const int input_dims = input.dims();

    Tensor *output = nullptr;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, input.shape(), &output));

#define HANDLE_DIM(NDIM)                                                       \
  case NDIM:                                                                   \
    functor::ReverseSequence<Device, T, Tlen, NDIM>::Compute(                  \
        context->eigen_device<Device>(), input.tensor<T, NDIM>(), batch_dim_,  \
        seq_dim_, seq_lens_t, output->tensor<T, NDIM>());                      \
    break;

    switch (input_dims) {
      HANDLE_DIM(2);
      HANDLE_DIM(3);
      HANDLE_DIM(4);
      HANDLE_DIM(5);

    default:
      OP_REQUIRES(
          context, false,
          errors::InvalidArgument(
              "ReverseSequence2Op : Unhandled input dimensions: ", input_dims));
    }
  }

private:
  int32 batch_dim_;
  int32 seq_dim_;

  TF_DISALLOW_COPY_AND_ASSIGN(ReverseSequence2Op);
};

#define REGISTER_REVERSE_SEQUENCE2(type, len_type)                              \
  REGISTER_KERNEL_BUILDER(Name("ReverseSequence2")                              \
                              .Device(DEVICE_CPU)                              \
                              .TypeConstraint<type>("T")                       \
                              .TypeConstraint<len_type>("Tlen"),               \
                          ReverseSequence2Op<CPUDevice, type, len_type>);

#define REGISTER_REVERSE_SEQUENCE2_LEN(type)                                    \
  REGISTER_REVERSE_SEQUENCE2(type, int32);                                      \
  REGISTER_REVERSE_SEQUENCE2(type, int64);

TF_CALL_NUMBER_TYPES(REGISTER_REVERSE_SEQUENCE2_LEN);
TF_CALL_bool(REGISTER_REVERSE_SEQUENCE2_LEN);

} // namespace tensorflow
