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

//#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/util/bcast.h"

// using namespace tensorflow;

namespace tensorflow {
namespace {

template <class T>
void ScalarAddition(OpKernelContext *context, const T *full_input,
                    int64 num_elements, T scalar_input, T *output) {
  for (int i = 0; i < num_elements; ++i) {
    output[i] = full_input[i] + scalar_input;
  }
}

template <class T>
void VectorAddition(OpKernelContext *context, const T *x_data, const T *y_data,
                    int64 num_elements, T *output) {
  for (int i = 0; i < num_elements; ++i) {
    output[i] = x_data[i] + y_data[i];
  }
}

template <class T>
void VectorTensorAddition(const T *vector_data, int64 vector_num_elements,
                          const T *tensor_data, int64 tensor_num_elements,
                          T *output) {
  for (int i = 0; i < tensor_num_elements; ++i) {
    const int64 vector_i = i % vector_num_elements;
    output[i] = vector_data[vector_i] + tensor_data[i];
  }
}

} // namespace

template <class T> class Add2Op : public OpKernel {
public:
  explicit Add2Op(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // Grab the input tensors
    const Tensor &x = context->input(0);
    const Tensor &y = context->input(1);

    BCast bcast(BCast::FromShape(x.shape()), BCast::FromShape(y.shape()));
    if (!bcast.IsValid()) {
      context->SetStatus(errors::InvalidArgument(
          "Incompatible shapes: ", x.shape().DebugString(), " vs. ",
          y.shape().DebugString()));
      return;
    }

    // Create an output tensor
    Tensor *z;
    OP_REQUIRES_OK(context, context->allocate_output(
                                0, BCast::ToShape(bcast.output_shape()), &z));

    const T *x_data = x.flat<T>().data();
    const T *y_data = y.flat<T>().data();
    T *z_data = z->flat<T>().data();

    // TODO: fix
    const int ndims = bcast.x_reshape().size();
    if (ndims <= 1) {
      if (x.NumElements() == 1) {
        ScalarAddition<T>(context, y_data, y.NumElements(), x_data[0], z_data);
      } else if (y.NumElements() == 1) {
        ScalarAddition<T>(context, x_data, x.NumElements(), y_data[0], z_data);
      } else {
        VectorAddition<T>(context, x_data, y_data, x.NumElements(), z_data);
      }
    } else if (ndims == 2) {
      const T *vector_data;
      int64 vector_num_elements;
      const T *tensor_data;
      int64 tensor_num_elements;
      if (x.NumElements() < y.NumElements()) {
        vector_data = x_data;
        vector_num_elements = x.NumElements();
        tensor_data = y_data;
        tensor_num_elements = y.NumElements();
      } else {
        vector_data = y_data;
        vector_num_elements = y.NumElements();
        tensor_data = x_data;
        tensor_num_elements = x.NumElements();
      }
      VectorTensorAddition<T>(vector_data, vector_num_elements, tensor_data,
                              tensor_num_elements, z_data);
    } else {
      LOG(INFO) << "ndims=" << ndims;
      LOG(INFO) << "bcast.x_reshape()="
                << TensorShape(bcast.x_reshape()).DebugString();
      LOG(INFO) << "bcast.y_reshape()="
                << TensorShape(bcast.y_reshape()).DebugString();
      LOG(INFO) << "bcast.x_bcast()="
                << TensorShape(bcast.x_bcast()).DebugString();
      LOG(INFO) << "bcast.y_bcast()="
                << TensorShape(bcast.y_bcast()).DebugString();

      context->SetStatus(errors::Unimplemented(
          "Broadcast between ", context->input(0).shape().DebugString(),
          " and ", context->input(1).shape().DebugString(),
          " is not supported yet."));
      return;
    }
  }
};

REGISTER_KERNEL_BUILDER(
    Name("Add2").Device(DEVICE_CPU).TypeConstraint<int8>("T"), Add2Op<int8>);
REGISTER_KERNEL_BUILDER(
    Name("Add2").Device(DEVICE_CPU).TypeConstraint<uint8>("T"), Add2Op<uint8>);
REGISTER_KERNEL_BUILDER(
    Name("Add2").Device(DEVICE_CPU).TypeConstraint<int32>("T"), Add2Op<int32>);
REGISTER_KERNEL_BUILDER(
    Name("Add2").Device(DEVICE_CPU).TypeConstraint<uint32>("T"),
    Add2Op<uint32>);
REGISTER_KERNEL_BUILDER(
    Name("Add2").Device(DEVICE_CPU).TypeConstraint<float>("T"),
    Add2Op<float>);

} // namespace tensorflow
