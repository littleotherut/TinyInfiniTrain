#include "example/common/tiny_shakespeare_dataset.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "glog/logging.h"

#include "infini_train/include/tensor.h"

namespace {
using DataType = infini_train::DataType;
using TinyShakespeareType = TinyShakespeareDataset::TinyShakespeareType;
using TinyShakespeareFile = TinyShakespeareDataset::TinyShakespeareFile;

const std::unordered_map<int, TinyShakespeareType> kTypeMap = {
    {20240520, TinyShakespeareType::kUINT16}, // GPT-2
    {20240801, TinyShakespeareType::kUINT32}, // LLaMA 3
};

const std::unordered_map<TinyShakespeareType, size_t> kTypeToSize = {
    {TinyShakespeareType::kUINT16, 2},
    {TinyShakespeareType::kUINT32, 4},
};

const std::unordered_map<TinyShakespeareType, DataType> kTypeToDataType = {
    {TinyShakespeareType::kUINT16, DataType::kUINT16},
    {TinyShakespeareType::kUINT32, DataType::kINT32},
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

TinyShakespeareFile ReadTinyShakespeareFile(const std::string &path, size_t sequence_length) {
    /* =================================== 作业 ===================================
       TODO：实现二进制数据集文件解析
       文件格式说明：
    ----------------------------------------------------------------------------------
    | HEADER (1024 bytes)                     | DATA (tokens)                        |
    | magic(4B) | version(4B) | num_toks(4B) | reserved(1012B) | token数据           |
    ----------------------------------------------------------------------------------
       =================================== 作业 =================================== */
    std::ifstream ifs(path, std::ios::binary);
    auto magic_bits = ReadSeveralBytesFromIfstream(4, &ifs);
    int32_t magic = BytesToType<int32_t>(magic_bits, 0);
    auto version_bits = ReadSeveralBytesFromIfstream(4, &ifs);
    int32_t version = BytesToType<int32_t>(version_bits, 0);
    auto num_toks_bits = ReadSeveralBytesFromIfstream(4, &ifs);
    int32_t num_toks = BytesToType<int32_t>(num_toks_bits, 0);
    auto dataset_type_iter = kTypeMap.find(magic);
    if (dataset_type_iter == kTypeMap.end()) {
        LOG(FATAL) << "不支持的TinyShakespeare数据集类型，magic: " << magic;
    }
    TinyShakespeareType dataset_type = dataset_type_iter->second;
    ReadSeveralBytesFromIfstream(1012, &ifs); // reserved, unused
    
    // 读取原始数据到临时缓冲区
    int64_t type_size = kTypeToSize.at(dataset_type);
    int64_t total_size = static_cast<int64_t>(num_toks) * type_size;
    std::vector<uint8_t> raw_data(total_size);
    ifs.read(reinterpret_cast<char *>(raw_data.data()), total_size);
    
    // 创建 INT64 类型的 Tensor 并转换数据
    infini_train::Tensor tensor({num_toks}, DataType::kINT64);
    int64_t *dst = static_cast<int64_t *>(tensor.DataPtr());
    
    if (dataset_type == TinyShakespeareType::kUINT16) {
        const uint16_t *src = reinterpret_cast<const uint16_t *>(raw_data.data());
        for (int32_t i = 0; i < num_toks; ++i) {
            dst[i] = static_cast<int64_t>(src[i]);
        }
    } else {  // kUINT32
        const uint32_t *src = reinterpret_cast<const uint32_t *>(raw_data.data());
        for (int32_t i = 0; i < num_toks; ++i) {
            dst[i] = static_cast<int64_t>(src[i]);
        }
    }

    TinyShakespeareFile file;
    file.type = dataset_type;
    file.dims = {static_cast<int64_t>(num_toks / sequence_length), static_cast<int64_t>(sequence_length)};
    file.tensor = tensor;
    return file;
}
} // namespace

TinyShakespeareDataset::TinyShakespeareDataset(const std::string &filepath, size_t sequence_length)
    : text_file_(ReadTinyShakespeareFile(filepath, sequence_length)),
      sequence_length_(sequence_length),
      sequence_size_in_bytes_(sequence_length * sizeof(int64_t)),  // 数据已转换为 INT64
      num_samples_(text_file_.dims[0] - 1) {
    // =================================== 作业 ===================================
    // TODO：初始化数据集实例
    // HINT: 调用ReadTinyShakespeareFile加载数据文件
    // =================================== 作业 ===================================
}

std::pair<std::shared_ptr<infini_train::Tensor>, std::shared_ptr<infini_train::Tensor>>
TinyShakespeareDataset::operator[](size_t idx) const {
    CHECK_LT(idx, text_file_.dims[0] - 1);
    std::vector<int64_t> dims = std::vector<int64_t>(text_file_.dims.begin() + 1, text_file_.dims.end());
    // x: (seq_len), y: (seq_len) -> stack -> (bs, seq_len) (bs, seq_len)
    return {std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_, dims),
            std::make_shared<infini_train::Tensor>(text_file_.tensor, idx * sequence_size_in_bytes_ + sizeof(int64_t),
                                                   dims)};
}

size_t TinyShakespeareDataset::Size() const { return num_samples_; }
