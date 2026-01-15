#include <cstdint>
#include <fcntl.h>
#include <memory>
#include <numeric>
#include <tuple>

#include "glog/logging.h"

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
std::shared_ptr<Tensor> MatmulForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法前向计算
    // REF:
    // =================================== 作业 ===================================
    
    if (input->Dims().size()==2 && other->Dims().size()==2) {
        auto output = std::make_shared<Tensor>(
            std::vector<int64_t>{input->Dims()[0], other->Dims()[1]}, DataType::kFLOAT32);
        output->EigenMatrix() = input->EigenMatrix() * other->EigenMatrix();
        return output;            
    }else {
        const auto &input_dims = input->Dims();
        const auto &other_dims = other->Dims();        

        // 计算输出维度,假设batch维度无需广播处理（纬度数一致且相等）
        CHECK_EQ(input_dims.size(), other_dims.size());
        size_t max_rank = input_dims.size();
        std::vector<int64_t> output_dims(max_rank);
        output_dims[max_rank - 2] = input_dims[input_dims.size() - 2];
        output_dims[max_rank - 1] = other_dims[other_dims.size() - 1];
        
        for (int i = 0; i < max_rank - 2; ++i) {
            int64_t dim1 = input_dims[max_rank - 3 - i];
            int64_t dim2 = other_dims[other_dims.size() - 3 - i];
            CHECK_EQ(dim1, dim2);
            output_dims[max_rank - 3 - i] = dim1;
        }

        auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);
        
        const int64_t M = input_dims[input_dims.size() - 2];
        const int64_t K = input_dims[input_dims.size() - 1];
        const int64_t N = other_dims[other_dims.size() - 1];

        const int64_t batch_size = std::accumulate(output_dims.begin(), output_dims.end() - 2, 1, std::multiplies<int64_t>{});

        float* in_ptr = input->EigenMatrix().data();
        float* other_ptr = other->EigenMatrix().data();
        float* out_ptr = output->EigenMatrix().data();
        for (int i = 0; i < batch_size; ++i) {
             long in_off = i * M * K;
             long other_off = i * K * N;
             long out_off = i * M * N;
             
             Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_in(
                in_ptr + in_off, M, K);
             Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_other(
                other_ptr + other_off, K, N);
             Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_out(
                out_ptr + out_off, M, N);
             
             mat_out.noalias() = mat_in * mat_other;
        }
        return output;
    }
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
MatmulBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
               const std::shared_ptr<Tensor> &grad_output) {
    // =================================== 作业 ===================================
    // TODO：实现CPU上的矩阵乘法反向传播
    // REF:
    // =================================== 作业 ===================================
    /*
    对input[i][j],grad_input[i][j] = sum_k(grad_output[i][k] * other[j][k])
    所有grad_input = grad_output * other^T
    对other[i][j],grad_other[i][j] = sum_k(input[k][i] * grad_output[k][j])
    所有grad_other = input^T * grad_output
    */
    auto grad_input = std::make_shared<Tensor>(input->Dims(), DataType::kFLOAT32);
    auto grad_other = std::make_shared<Tensor>(other->Dims(), DataType::kFLOAT32);
    if (input->Dims().size()==2 && other->Dims().size()==2 && grad_output->Dims().size()==2) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * other->EigenMatrix().transpose();
        grad_other->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    } else {
        // bmm反向传播
        const auto &input_dims = input->Dims();
        const auto &other_dims = other->Dims();
        const auto &grad_dims = grad_output->Dims();

        const int64_t batch_size = std::accumulate(grad_dims.begin(), grad_dims.end() - 2, 1, std::multiplies<int64_t>{});

        // 初始化梯度为0
        grad_input->EigenMatrix().setZero();
        grad_other->EigenMatrix().setZero();

        float* in_ptr = input->EigenMatrix().data();
        float* other_ptr = other->EigenMatrix().data();
        float* g_out_ptr = grad_output->EigenMatrix().data();
        float* g_in_ptr = grad_input->EigenMatrix().data();
        float* g_other_ptr = grad_other->EigenMatrix().data();

        const int64_t M = input_dims[input_dims.size() - 2];
        const int64_t K = input_dims[input_dims.size() - 1];
        const int64_t N = other_dims[other_dims.size() - 1];

        for (int i = 0; i < batch_size; ++i) {
            long in_off = i * M * K;
            long other_off = i * K * N;
            long out_off = i * M * N;

            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_in(
               in_ptr + in_off, M, K);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_other(
               other_ptr + other_off, K, N);
            Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_g_out(
               g_out_ptr + out_off, M, N);
            
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_g_in(
               g_in_ptr + in_off, M, K);
            Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> mat_g_other(
               g_other_ptr + other_off, K, N);
            
            mat_g_in.noalias() += mat_g_out * mat_other.transpose();
            mat_g_other.noalias() += mat_in.transpose() * mat_g_out;
        }
    }
    return {grad_input, grad_other};
}

std::shared_ptr<Tensor> LinearForward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight,
                                      bool transpose, const std::shared_ptr<Tensor> &bias) {
    /*
    transpose:  output = input * weight^T + bias
    output[*, out_features] = input[*, in_features] * weight[out_features, in_features]^T + bias[out_features]

    !transpose: output = input * weight + bias
    output[*, out_features] = input[*, in_features] * weight[in_features, out_features] + bias[out_features]
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    const int out_features = weight_dims[transpose ? 0 : 1];

    if (bias) {
        const auto &bias_dims = bias->Dims();
        CHECK_EQ(bias_dims.size(), 1);
        CHECK_EQ(bias_dims[0], out_features);
    }

    auto output_dims = input_dims;
    *output_dims.rbegin() = out_features;
    auto output = std::make_shared<Tensor>(output_dims, DataType::kFLOAT32);

    if (transpose) {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix().transpose();
    } else {
        output->EigenMatrix() = input->EigenMatrix() * weight->EigenMatrix();
    }

    if (bias) {
        output->EigenMatrix().rowwise() += bias->EigenVector();
    }

    return output;
}

std::tuple<std::shared_ptr<Tensor>, std::shared_ptr<Tensor>, std::shared_ptr<Tensor>>
LinearBackward(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &weight, bool transpose,
               int64_t out_features, const std::shared_ptr<Tensor> &grad_output, const bool bias) {
    /*
    transpose: grad_input = grad_output * weight
    grad_input[*, in_features] = grad_output[*, out_features] * weight[out_features, in_features]
    grad_weight[out_features, in_features] = grad_output[*, out_features]^T * input[*, in_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)

    !transpose: grad_input = grad_output * weight^T
    grad_input[*, in_features] = grad_output[_, out_features] * weight[in_features, out_features]^T
    grad_weight[in_features, out_features] = input[*, in_features]^T * grad_output[*, out_features]
    grad_bias[out_features] = grad_output[*, out_features].sum(axis=0)
    */

    const auto &input_dims = input->Dims();
    CHECK_GE(input_dims.size(), 2);
    const int64_t bs = std::accumulate(input_dims.rbegin() + 1, input_dims.rend(), 1, std::multiplies<int64_t>{});
    const int64_t in_features = *input_dims.rbegin();

    const auto &weight_dims = weight->Dims();
    CHECK_EQ(weight_dims.size(), 2);
    CHECK_EQ(in_features, weight_dims[transpose ? 1 : 0]);
    CHECK_EQ(out_features, weight_dims[transpose ? 0 : 1]);

    auto grad_input = std::make_shared<Tensor>(input_dims, DataType::kFLOAT32);
    auto grad_weight = std::make_shared<Tensor>(weight_dims, DataType::kFLOAT32);
    std::shared_ptr<Tensor> grad_bias = nullptr;
    if (bias) {
        grad_bias = std::make_shared<Tensor>(std::vector<int64_t>{out_features}, DataType::kFLOAT32);
    }

    if (transpose) {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix();
        grad_weight->EigenMatrix() = grad_output->EigenMatrix().transpose() * input->EigenMatrix();
    } else {
        grad_input->EigenMatrix() = grad_output->EigenMatrix() * weight->EigenMatrix().transpose();
        grad_weight->EigenMatrix() = input->EigenMatrix().transpose() * grad_output->EigenMatrix();
    }
    if (bias) {
        grad_bias->EigenVector() = grad_output->EigenMatrix().colwise().sum();
    }

    return {grad_input, grad_weight, grad_bias};
}
} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_LINEAR_KERNEL(kernel_name)                                                                        \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_LINEAR_KERNEL(MatmulForward)
REGISTER_CPU_LINEAR_KERNEL(MatmulBackward)
REGISTER_CPU_LINEAR_KERNEL(LinearForward)
REGISTER_CPU_LINEAR_KERNEL(LinearBackward)

#undef REGISTER_CPU_LINEAR_KERNEL
