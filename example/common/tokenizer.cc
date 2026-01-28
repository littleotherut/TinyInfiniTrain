#include "example/common/tokenizer.h"

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "glog/logging.h"

namespace infini_train {

constexpr uint32_t kGpt2Eot = 50256;
constexpr uint32_t kLLaMA3Eot = 128001;
constexpr uint64_t kRandomU32Multiplier = 0x2545F4914F6CDD1Dull;
constexpr float kF32Divisor = 16777216.0f; // 2^24
constexpr uint64_t kRngState = 1337;

using Version = Tokenizer::Version;

const std::unordered_map<uint32_t, uint32_t> kEotMap = {
    {20240328, kGpt2Eot},   // GPT-2
    {20240801, kLLaMA3Eot}, // LLaMA-3
};

const std::unordered_map<uint32_t, std::vector<uint32_t>> kPromptMap = {
    // e.g. "The meaning of life is"
    // ref: https://tiktokenizer.vercel.app/
    {20240328, std::vector<uint32_t>{464, 3616, 286, 1204, 318}}, // GPT-2
    {20240801, std::vector<uint32_t>{791, 7438, 315, 2324, 374}}, // LLaMA-3
};

std::vector<uint8_t> ReadSeveralBytesFromIfstream(size_t num_bytes, std::ifstream *ifs) {
    std::vector<uint8_t> result(num_bytes);
    ifs->read(reinterpret_cast<char *>(result.data()), num_bytes);
    return result;
}

template <typename T> T BytesToType(const std::vector<uint8_t> &bytes, size_t offset) {
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable.");
    T value;
    std::memcpy(&value, &bytes[offset], sizeof(T));
    return value;
}

unsigned int RandomU32(uint64_t &state) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (state * kRandomU32Multiplier) >> 32;
}

float RandomF32(uint64_t &state) { // random float32 in [0,1)
    return (RandomU32(state) >> 8) / kF32Divisor;
}

int SampleMult(float *probabilities, int n, float coin) {
    // sample index from probabilities (they must sum to 1!)
    // coin is a random number in [0, 1), usually from RandomF32()
    float cdf = 0.0f;
    for (int i = 0; i < n; i++) {
        cdf += probabilities[i];
        if (coin < cdf) {
            return i;
        }
    }
    return n - 1; // in case of rounding errors
}

Tokenizer::Tokenizer(const std::string &filepath) {
    /* ===================================== 作业 =====================================
    TODO：实现Tokenizer二进制文件加载

    文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | VOCAB TABLE                           |
    | magic(4B) | version(4B) | vocab_size(4B) | reserved(1012B) | token词表数据       |
    ----------------------------------------------------------------------------------
    ===================================== 作业 ===================================== */
    std::ifstream ifs(filepath, std::ios::binary);
    // 读取header (1024 bytes = 256 * 4 bytes)
    auto header_bytes = ReadSeveralBytesFromIfstream(1024, &ifs);
    magic_number_ = BytesToType<uint32_t>(header_bytes, 0);
    uint32_t version = BytesToType<uint32_t>(header_bytes, 4);
    vocab_size_ = BytesToType<uint32_t>(header_bytes, 8);

    CHECK(kEotMap.count(magic_number_)) << "Unknown magic number: " << magic_number_;
    eot_token_ = kEotMap.at(magic_number_);

    // 读取词表数据 (1字节长度 + token字符串)
    token_table_.resize(vocab_size_);
    for (uint32_t i = 0; i < vocab_size_; ++i) {
        uint8_t token_len;
        ifs.read(reinterpret_cast<char *>(&token_len), 1);
        std::string token(token_len, '\0');
        ifs.read(&token[0], token_len);
        token_table_[i] = token;
    }
}

std::string Tokenizer::Decode(uint32_t token_id) const {
    /* ===================================== 作业 =====================================
    TODO：实现token_id到文本的转换
    功能描述：根据token_id返回对应的文本片段
    ===================================== 作业 ===================================== */
    CHECK_LT(token_id, vocab_size_) << "Token ID out of range: " << token_id;
    return token_table_[token_id];
}

void Tokenizer::GenerateText(infini_train::nn::Module &model, uint32_t batch_size, uint32_t sequence_length,
                             uint32_t text_length, Device device) const {
    LOG(INFO) << "Skip";
    return;
    std::vector<int64_t> dims;
    dims.assign({batch_size, sequence_length});
    // x_tensor (FLAGS_batch_size, FLAGS_sequence_length) eq:(4, 64)
    infini_train::Tensor x_tensor = infini_train::Tensor(dims, DataType::kINT64);
    int64_t *x_buff = static_cast<int64_t *>(x_tensor.DataPtr());
    for (int i = 0; i < batch_size * sequence_length; ++i) { x_buff[i] = eot_token_; }

    // Give some contexts: "The meaning of life is "
    auto prompt = kPromptMap.at(magic_number_);
    auto prompt_len = prompt.size();
    for (int i = 0; i < prompt_len; ++i) { x_buff[i] = prompt[i]; }
    std::cout << "The meaning of life is";

    auto x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
    uint64_t kRngState = 1337;
    LOG(INFO) << "start generate text:";
    for (int t = prompt_len; t < text_length; t++) {
        /* ===================================== 作业 =====================================
        TODO：实现单步文本生成逻辑
        HINT：调用model.Forward推理获取logits，根据推理结果进行随机采样，调用Decode获取文本结果
        ===================================== 作业 ===================================== */
        auto output = model.Forward({x}); // GPT2/net.cc
        auto logits = output[0]; // (bs, seq_len, vocab_size)
        auto logits_dims = logits->Dims();

        int64_t vocab_size = logits_dims[2];
        
        // 优化：在Device端进行Slice和Softmax，减少D2H数据传输量
        std::vector<int64_t> starts = {0, static_cast<int64_t>(t - 1), 0};
        std::vector<int64_t> ends = {static_cast<int64_t>(batch_size), static_cast<int64_t>(t), vocab_size};
        std::vector<int64_t> steps = {1, 1, 1};

        auto sliced_logits = nn::function::Slice(logits, starts, ends, steps);
        auto probs = nn::function::Softmax(sliced_logits, -1);

        auto probs_cpu = probs->To(Device(DeviceType::kCPU, 0));
        float* probs_ptr = static_cast<float*>(probs_cpu.DataPtr());

        for (uint32_t b = 0; b < batch_size; ++b) {
            float* batch_probs = probs_ptr + b * vocab_size;

            // sample
            float coin = RandomF32(kRngState);
            int sampled_token_id = SampleMult(batch_probs, vocab_size, coin);
            x_buff[b * sequence_length + t] = sampled_token_id;

            // decode and print 
            std::string token_str = Decode(static_cast<uint32_t>(sampled_token_id));
            std::cout << token_str;
        }
        // 将更新后的 CPU 数据同步到 GPU
        x = std::make_shared<infini_train::Tensor>(x_tensor.To(device));
        break;
    }
    std::cout << std::endl;
}
} // namespace infini_train
