#include <cstddef>
#include <memory>

#include "infini_train/include/dispatcher.h"
#include "infini_train/include/tensor.h"

namespace infini_train::kernels::cpu {
void AccumulateGrad(const std::shared_ptr<Tensor> &gradient, float rate, const std::shared_ptr<Tensor> &tensor) {
    for (int64_t idx = 0; idx < gradient->NumElements(); ++idx) {
        static_cast<float *>(tensor->DataPtr())[idx] += rate * static_cast<const float *>(gradient->DataPtr())[idx];
    }
}

void AdamAccumulateGrad(const std::shared_ptr<Tensor> &grad, const std::shared_ptr<Tensor> &param,
                        const std::shared_ptr<Tensor> &m, const std::shared_ptr<Tensor> &v, float learning_rate,
                        float beta1, float beta2, float eps, int64_t t) {
    // =================================== 作业 ===================================
    // TODO：实现Adam优化器的梯度累积和参数更新
    // REF:
    // =================================== 作业 ===================================
    // Adam优化器更新公式：(t 更新由 Adam::Step 控制)
    // m = beta1 * m + (1 - beta1) * grad
    // v = beta2 * v + (1 - beta2) * grad^2
    // m_hat = m / (1 - beta1^t)
    // v_hat = v / (1 - beta2^t)
    // param = param - learning_rate * m_hat / (sqrt(v_hat) + eps)
    for(int64_t idx = 0 ; idx < grad->NumElements(); ++idx) {
        float g = static_cast<const float*>(grad->DataPtr())[idx];
        
        // 更新一阶动量m
        float m_new = beta1 * static_cast<float*>(m->DataPtr())[idx] + (1 - beta1) * g;
        static_cast<float*>(m->DataPtr())[idx] = m_new;
        
        // 更新二阶动量v
        float v_new = beta2 * static_cast<float*>(v->DataPtr())[idx] + (1 - beta2) * g * g;
        static_cast<float*>(v->DataPtr())[idx] = v_new;
        
        // 计算偏差修正后的动量
        float m_hat = m_new / (1 - std::pow(beta1, t));
        float v_hat = v_new / (1 - std::pow(beta2, t));
        
        static_cast<float*>(param->DataPtr())[idx] -= learning_rate * m_hat / (std::sqrt(v_hat) + eps);
    }

}

} // namespace infini_train::kernels::cpu

#define REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(kernel_name)                                                               \
    REGISTER_KERNEL(infini_train::DeviceType::kCPU, kernel_name, infini_train::kernels::cpu::kernel_name)

REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AccumulateGrad)
REGISTER_CPU_ACCUMULATE_GRAD_KERNEL(AdamAccumulateGrad)

#undef REGISTER_CPU_ACCUMULATE_GRAD_KERNEL
