/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// THE GOLDEN DUMP MANIFEST READER: the line-based manifest.txt plus its raw
// little-endian .bin tensors, as golden/generate_dumps.py writes them.
//
// One definition, shared by every runner that consumes the dumps -- the
// CAModel/NPU kernel runners of tests/npu/test_golden_dump.cpp and the CPU
// oracle comparison of tests/cpu/test_golden_decode_oracle.cpp. The format is
// the PyTorch golden's side of the comparison; two readers that drifted would
// let a runner pass against a manifest the other would refuse.
//
// Self-contained C++17, toolkit-free.

#ifndef GOLDEN_DUMP_MANIFEST_HPP
#define GOLDEN_DUMP_MANIFEST_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tq_dump {

inline void require(bool ok, const std::string& what)
{
    if (!ok) {
        std::fprintf(stderr, "CHECK FAILED: %s\n", what.c_str());
        throw std::runtime_error(what);
    }
}

struct Tensor {
    std::string dtype;
    std::size_t count = 0;
    std::string file;
    std::vector<std::uint8_t> bytes;

    std::size_t element_bytes() const
    {
        if (dtype == "u8" || dtype == "i8") return 1;
        if (dtype == "u16" || dtype == "bf16" || dtype == "f16") return 2;
        if (dtype == "f32" || dtype == "i32") return 4;
        throw std::runtime_error("unknown dump dtype " + dtype);
    }
    std::size_t size_bytes() const { return count * element_bytes(); }
};

// A sparsely stored tensor: the engine-scale caches of the real decode cases
// (6496 blocks is ~400 MB a plane, almost all of it guard fill) are dumped as
// a fill word plus the byte segments that carry live data. The reader
// materialises the full image, so every consumer downstream is unchanged.
struct TensorSource {
    bool sparse = false;
    std::uint32_t fill = 0;
    std::string index_file;   // <u64 offset, u64 bytes> little-endian pairs
    std::string data_file;    // the segments' payloads, concatenated
};

inline void fill_pattern(std::uint8_t* at, std::size_t bytes, std::uint32_t word)
{
    for (std::size_t i = 0; i + sizeof(word) <= bytes; i += sizeof(word))
        std::memcpy(at + i, &word, sizeof(word));
}

struct Manifest {
    std::string kernel;
    std::map<std::string, long long> geom;
    std::map<std::string, double> scalars;
    std::map<std::string, Tensor> tensors;
    std::map<std::string, TensorSource> sources;

    long long geo(const std::string& key) const
    {
        const auto it = geom.find(key);
        require(it != geom.end(), "dump has no geom " + key);
        return it->second;
    }
    double scalar(const std::string& key) const
    {
        const auto it = scalars.find(key);
        require(it != scalars.end(), "dump has no float " + key);
        return it->second;
    }
    const Tensor& tensor(const std::string& name) const
    {
        const auto it = tensors.find(name);
        require(it != tensors.end(), "dump has no tensor " + name);
        return it->second;
    }
};

inline std::vector<std::uint8_t> read_file(const std::string& path)
{
    std::ifstream bin(path, std::ios::binary);
    require(bin.good(), "cannot open " + path);
    bin.seekg(0, std::ios::end);
    const std::size_t bytes = static_cast<std::size_t>(bin.tellg());
    bin.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> out(bytes);
    if (bytes != 0) {
        bin.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
        require(bin.good(), "short read on " + path);
    }
    return out;
}

inline void materialise_sparse(const std::string& case_dir, Tensor& tensor,
                               const TensorSource& source)
{
    const std::size_t total = tensor.size_bytes();
    tensor.bytes.assign(total, 0);
    fill_pattern(tensor.bytes.data(), total, source.fill);
    const std::vector<std::uint8_t> index = read_file(case_dir + "/" + source.index_file);
    require(index.size() % (2 * sizeof(std::uint64_t)) == 0,
            source.index_file + ": the sparse index is not a whole number of segments");
    const std::vector<std::uint8_t> payload = read_file(case_dir + "/" + source.data_file);
    std::size_t payload_at = 0;
    const std::size_t segments = index.size() / (2 * sizeof(std::uint64_t));
    for (std::size_t s = 0; s < segments; ++s) {
        std::uint64_t offset = 0, bytes = 0;
        std::memcpy(&offset, index.data() + s * 16, 8);
        std::memcpy(&bytes, index.data() + s * 16 + 8, 8);
        require(offset + bytes <= total, source.index_file + ": a segment runs off the tensor");
        require(payload_at + bytes <= payload.size(),
                source.data_file + ": payload shorter than the segments describe");
        std::memcpy(tensor.bytes.data() + static_cast<std::size_t>(offset),
                    payload.data() + payload_at, static_cast<std::size_t>(bytes));
        payload_at += static_cast<std::size_t>(bytes);
    }
    require(payload_at == payload.size(), source.data_file + ": payload has unclaimed bytes");
}

inline Manifest load_manifest(const std::string& case_dir)
{
    std::ifstream stream(case_dir + "/manifest.txt");
    require(stream.good(), "cannot open " + case_dir + "/manifest.txt");
    Manifest manifest;
    std::string line;
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream fields(line);
        std::string kind;
        fields >> kind;
        if (kind == "kernel") {
            fields >> manifest.kernel;
        } else if (kind == "geom") {
            std::string key;
            long long value = 0;
            fields >> key >> value;
            manifest.geom[key] = value;
        } else if (kind == "float") {
            std::string key, value;
            fields >> key >> value;
            manifest.scalars[key] = std::stod(value);
        } else if (kind == "tensor") {
            std::string name;
            Tensor tensor;
            fields >> name >> tensor.dtype >> tensor.count >> tensor.file;
            manifest.tensors[name] = tensor;
        } else if (kind == "sparse_tensor") {
            std::string name, fill_hex;
            Tensor tensor;
            TensorSource source;
            std::size_t segments = 0;
            fields >> name >> tensor.dtype >> tensor.count >> fill_hex >> segments >>
                source.index_file >> source.data_file;
            source.sparse = true;
            source.fill = static_cast<std::uint32_t>(std::stoul(fill_hex, nullptr, 16));
            tensor.file = source.data_file;
            manifest.tensors[name] = tensor;
            manifest.sources[name] = source;
        } else {
            require(false, "unparseable manifest line: " + line);
        }
    }
    require(!manifest.kernel.empty(), "manifest names no kernel");
    for (auto& [name, tensor] : manifest.tensors) {
        const auto source = manifest.sources.find(name);
        if (source != manifest.sources.end() && source->second.sparse) {
            materialise_sparse(case_dir, tensor, source->second);
            continue;
        }
        const std::string path = case_dir + "/" + tensor.file;
        tensor.bytes = read_file(path);
        require(tensor.bytes.size() == tensor.size_bytes(),
                path + ": size drift between manifest and file");
    }
    return manifest;
}

template <class T>
std::vector<T> typed(const Tensor& tensor)
{
    require(tensor.bytes.size() % sizeof(T) == 0, "dump tensor is not a whole number of elements");
    std::vector<T> out(tensor.bytes.size() / sizeof(T));
    std::memcpy(out.data(), tensor.bytes.data(), tensor.bytes.size());
    return out;
}

}  // namespace tq_dump

#endif  // GOLDEN_DUMP_MANIFEST_HPP
