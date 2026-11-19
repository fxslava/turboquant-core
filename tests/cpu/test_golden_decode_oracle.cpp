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

// THE GOLDEN-ON-GOLDEN COMPARISON: the plain C golden oracle
// (tests/common/turboquant_golden_oracle.hpp) against the PyTorch golden of
// vllm-ascend's turboquant_v1.py, over the decode dumps of golden/dumps/.
//
// Why this gate exists. The kernel runners (tests/npu/test_golden_dump.cpp)
// hold the KERNELS to the PyTorch golden. This gate holds the C++ ORACLE to
// it: the oracle is the reference the device-side e2e fixtures and any future
// host-side runner are written against, so its decode half must be shown to
// reproduce the PyTorch golden's attention on the real decode cases before it
// can be trusted to judge anything else. The two sides are independent
// re-statements of the same contract -- the oracle reads the NZ cache through
// the production address functions of turboquant_layout.h and rotates with a
// dense Pi built from the LCG pi_signs; the dump's expected images come from
// golden/turboquant_golden.py's decode_reference, the standalone copy of
// turboquant_v1.py's golden decode. Both compute in double, so agreement is
// bit-exact or a couple of fp32 ULP, and anything worse is a real divergence
// in one of the two statements, not tolerance.
//
// Three checks per case:
//   1. pi_signs: the oracle's LCG vector bit-equals the dump's (the signs the
//      real kernels ingest).
//   2. query_rot: the oracle's dense-Pi rotation of the raw query. BIT-EXACT
//      on the integer fixtures (every butterfly intermediate is an exact
//      integer by design), a few fp32 ULP on the real cases' random query.
//   3. expected_stage0/1: the oracle's fp64 decode (raw query in, rotated and
//      model-basis contexts out) against the dump's fp64 reference, per
//      channel on --ulp-budget OR --rel-budget of the row max.
//
//   --dump-root DIR   where golden/dumps lives (default "golden/dumps")
//   --case NAME       one case directory instead of the default list
//   --ulp-budget N    fp32 ULP budget for the stage outputs (default 4)
//   --rel-budget R    row-max-relative budget for the same (default 1e-5)
//   --query-rot-rel R row-max-relative budget for query_rot on non-integer
//                     fixtures (default 1e-5; integer fixtures are always held
//                     bit-exact)

#include "../common/golden_dump_manifest.hpp"
#include "../common/turboquant_golden_oracle.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace golden_oracle {

void require(bool ok, const std::string& what)
{
    if (!ok) {
        std::fprintf(stderr, "CHECK FAILED: %s\n", what.c_str());
        throw std::runtime_error(what);
    }
}

using tq_dump::Manifest;
using tq_dump::Tensor;
using tq_dump::load_manifest;
using tq_dump::typed;

// The ordered-integer ULP distance between two fp32 words.
int fp32_ulp(std::uint32_t a, std::uint32_t b)
{
    const auto order = [](std::uint32_t w) -> long long {
        return (w & 0x80000000u) ? -static_cast<long long>(w & 0x7FFFFFFFu)
                                 : static_cast<long long>(w & 0x7FFFFFFFu);
    };
    const long long d = order(a) - order(b);
    return static_cast<int>(d < 0 ? -d : d);
}

struct FloatDiff {
    int worst_ulp = 0;
    double worst_abs = 0.0;
    double worst_row_rel = 0.0;
    double scale_ratio = 0.0;
    std::size_t over_budget = 0;
};

// One output image of [rows, head_size] doubles against the dump's fp32
// reference. A channel passes on the fp32 ULP budget or on the row-relative
// one -- both are fp64 references rounded to fp32, so the ULP half carries
// almost everything and the relative half covers the rare channel sitting on
// an fp32 rounding boundary through a libm exp.
FloatDiff compare_stage(const char* what, const double* got, const Tensor& expected, int rows,
                        int width, int ulp_budget, double rel_budget)
{
    const std::vector<float> want = typed<float>(expected);
    require(want.size() == static_cast<std::size_t>(rows) * width,
            std::string(what) + ": the reference image does not match the geometry");
    FloatDiff diff;
    double dot = 0.0, norm = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        const float got_f = static_cast<float>(got[i]);
        const std::uint32_t got_bits = tq_test::float_bits(got_f);
        const std::uint32_t want_bits = tq_test::float_bits(want[i]);
        const int ulp = fp32_ulp(got_bits, want_bits);
        const double absolute = std::fabs(static_cast<double>(got_f) - want[i]);
        diff.worst_ulp = std::max(diff.worst_ulp, ulp);
        diff.worst_abs = std::max(diff.worst_abs, absolute);
        dot += static_cast<double>(got_f) * want[i];
        norm += static_cast<double>(want[i]) * want[i];
        double row_max = 0.0;
        const std::size_t row = i / width * width;
        for (int c = 0; c < width; ++c)
            row_max = std::max(row_max, std::fabs(static_cast<double>(want[row + c])));
        const double row_rel = row_max > 0.0 ? absolute / row_max : absolute;
        diff.worst_row_rel = std::max(diff.worst_row_rel, row_rel);
        if (ulp <= ulp_budget || row_rel <= rel_budget) continue;
        if (diff.over_budget < 4)
            std::fprintf(stderr,
                         "  %s channel %zu: oracle %.9g (0x%08x) golden %.9g (0x%08x), "
                         "%d fp32 ULP, %.4g of the row max\n",
                         what, i, got_f, got_bits, static_cast<double>(want[i]), want_bits, ulp,
                         row_rel);
        ++diff.over_budget;
    }
    diff.scale_ratio = norm > 0.0 ? dot / norm : 1.0;
    return diff;
}

int run_case(const std::string& case_dir, int ulp_budget, double rel_budget, double query_rot_rel)
{
    const Manifest dump = load_manifest(case_dir);
    require(dump.kernel == "decode", case_dir + " is not a decode dump");
    tq_golden::DecodeGeometry g;
    g.requests = static_cast<int>(dump.geo("requests"));
    g.kv_heads = static_cast<int>(dump.geo("num_kv_heads"));
    const int num_heads = static_cast<int>(dump.geo("num_heads"));
    require(num_heads % g.kv_heads == 0, "num_heads is not a whole GQA group");
    g.group_heads = num_heads / g.kv_heads;
    g.head_size = static_cast<int>(dump.geo("head_size"));
    g.page_tokens = static_cast<int>(dump.geo("block_size"));
    g.max_pages = static_cast<int>(dump.geo("max_blocks"));
    g.score_scale = static_cast<float>(dump.scalar("scale"));
    const long long integer_fixture = dump.geo("integer_fixture");
    const double inv_sqrt_len = dump.scalar("inv_sqrt_len");

    std::printf("decode oracle %s: context %lld, %d KV heads x %d = %d heads, page %d, "
                "table width %d\n",
                case_dir.c_str(), dump.geo("context"), g.kv_heads, g.group_heads, num_heads,
                g.page_tokens, g.max_pages);
    int failures = 0;

    // 1. The LCG pi_signs, bit for bit: the rotation's sign diagonal is the
    // one operand neither side may restate differently.
    const std::vector<float> want_signs = typed<float>(dump.tensor("pi_signs"));
    const std::vector<float> signs = tq_golden::kernel_pi_signs(g.head_size);
    require(signs.size() == want_signs.size(), "pi_signs length drift");
    bool signs_exact = true;
    for (std::size_t c = 0; c < signs.size(); ++c)
        if (tq_test::float_bits(signs[c]) != tq_test::float_bits(want_signs[c])) {
            std::fprintf(stderr, "  pi_signs[%zu]: oracle %+.1f golden %+.1f\n", c,
                         static_cast<double>(signs[c]), static_cast<double>(want_signs[c]));
            signs_exact = false;
        }
    std::printf("  pi_signs: %s (%zu entries)\n", signs_exact ? "bit-exact" : "DIFFER",
                signs.size());
    if (!signs_exact) ++failures;

    const tq_golden::DensePi pi(signs);
    require(std::fabs(pi.inv_sqrt_len - inv_sqrt_len) == 0.0,
            "inv_sqrt_len drift between the oracle and the dump");

    const std::vector<std::int32_t> lengths = typed<std::int32_t>(dump.tensor("context_lens"));
    const std::vector<std::int32_t> table = typed<std::int32_t>(dump.tensor("block_table"));
    require(static_cast<std::size_t>(g.requests) * g.max_pages <= table.size(),
            "the block table is short for the geometry");

    // The raw query, widened from its bf16 storage exactly as the device
    // widens it.
    const std::vector<std::uint16_t> raw_bits = typed<std::uint16_t>(dump.tensor("raw_query"));
    require(raw_bits.size() ==
                static_cast<std::size_t>(g.requests) * num_heads * g.head_size,
            "the raw query does not match the geometry");
    std::vector<float> query(raw_bits.size());
    for (std::size_t i = 0; i < raw_bits.size(); ++i)
        query[i] = tq_test::bf16_to_float(raw_bits[i]);

    // 2. query_rot: the dense-Pi ingress rotation of the raw query. The
    // integer fixtures make every intermediate exact, so there the comparison
    // is bit-exact. A random query makes the oracle's fp64 dense product and
    // the golden's fp32 dense matmul two summation orders of the same value --
    // a few fp32 ULP of the ROW MAX by construction, and arbitrarily many ULP
    // of a near-zero channel, which is why the random half is gated on the
    // row-relative measure the vllm-ascend unit test uses for the same check.
    const std::vector<float> want_rot = typed<float>(dump.tensor("query_rot"));
    require(want_rot.size() == query.size(), "query_rot does not match the geometry");
    int rot_worst_ulp = 0;
    double rot_worst_row_rel = 0.0;
    bool rot_bit_exact = true;
    {
        std::vector<double> x(static_cast<std::size_t>(g.head_size));
        std::vector<double> y(static_cast<std::size_t>(g.head_size));
        const std::size_t rows = query.size() / g.head_size;
        for (std::size_t row = 0; row < rows; ++row) {
            double row_max = 0.0;
            for (int c = 0; c < g.head_size; ++c) {
                x[static_cast<std::size_t>(c)] = query[row * g.head_size + c];
                row_max = std::max(row_max,
                                   std::fabs(static_cast<double>(want_rot[row * g.head_size + c])));
            }
            pi.rotate_query(x.data(), y.data());
            for (int c = 0; c < g.head_size; ++c) {
                const float got = static_cast<float>(y[static_cast<std::size_t>(c)]);
                const int ulp = fp32_ulp(tq_test::float_bits(got),
                                         tq_test::float_bits(want_rot[row * g.head_size + c]));
                rot_worst_ulp = std::max(rot_worst_ulp, ulp);
                if (row_max > 0.0)
                    rot_worst_row_rel =
                        std::max(rot_worst_row_rel,
                                 std::fabs(static_cast<double>(got) -
                                           want_rot[row * g.head_size + c]) /
                                     row_max);
                if (ulp != 0) rot_bit_exact = false;
            }
        }
    }
    if (integer_fixture) {
        std::printf("  query_rot: %s (integer fixture, held bit-exact)\n",
                    rot_bit_exact ? "bit-exact" : "DIFFERS -- the rotation is wrong, not rounded");
        if (!rot_bit_exact) ++failures;
    } else {
        std::printf("  query_rot: worst %d fp32 ULP, worst %.3g of row max (budget %.3g, "
                    "random fixture)\n",
                    rot_worst_ulp, rot_worst_row_rel, query_rot_rel);
        if (rot_worst_row_rel > query_rot_rel) ++failures;
    }

    // 3. The decode itself, from the raw query to both output stages.
    const tq_golden::DecodeOutputs out = tq_golden::decode_reference(
        g, pi, query.data(), dump.tensor("key_cache").bytes.data(),
        dump.tensor("value_cache").bytes.data(), typed<float>(dump.tensor("scale_cache")).data(),
        table.data(), lengths.data());
    const FloatDiff stage0 =
        compare_stage("stage0 (rotated)", out.stage0.data(), dump.tensor("expected_stage0"),
                      g.requests * num_heads, g.head_size, ulp_budget, rel_budget);
    const FloatDiff stage1 =
        compare_stage("stage1 (model basis)", out.stage1.data(), dump.tensor("expected_stage1"),
                      g.requests * num_heads, g.head_size, ulp_budget, rel_budget);
    const struct {
        const char* name;
        const FloatDiff* diff;
    } legs[] = {
        {"stage0 (rotated)", &stage0},
        {"stage1 (model basis)", &stage1},
    };
    for (const auto& leg : legs) {
        std::printf("    %-22s worst %d fp32 ULP (budget %d), worst abs %.3e, worst %.3g of "
                    "row max (budget %.3g), scale ratio %.8f\n",
                    leg.name, leg.diff->worst_ulp, ulp_budget, leg.diff->worst_abs,
                    leg.diff->worst_row_rel, rel_budget, leg.diff->scale_ratio);
        if (leg.diff->over_budget != 0) {
            std::fprintf(stderr, "    %zu channels over budget\n", leg.diff->over_budget);
            ++failures;
        }
    }

    require(failures == 0, "the C++ golden oracle diverges from the PyTorch golden");
    std::printf("PASS %s: the C++ golden oracle reproduces the PyTorch golden\n",
                case_dir.c_str());
    return 0;
}

}  // namespace golden_oracle

int main(int argc, char** argv)
{
    std::string dump_root = "golden/dumps";
    std::string case_override;
    int ulp_budget = 4;
    double rel_budget = 1e-5;
    double query_rot_rel = 1e-5;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--dump-root" && i + 1 < argc) {
            dump_root = argv[++i];
        } else if (arg == "--case" && i + 1 < argc) {
            case_override = argv[++i];
        } else if (arg == "--ulp-budget" && i + 1 < argc) {
            const auto got =
                std::from_chars(argv[i + 1], argv[i + 1] + std::strlen(argv[i + 1]), ulp_budget);
            if (got.ec != std::errc() || got.ptr != argv[i + 1] + std::strlen(argv[i + 1]) ||
                ulp_budget < 0) {
                std::fprintf(stderr, "--ulp-budget needs a non-negative integer\n");
                return 2;
            }
            ++i;
        } else if (arg == "--rel-budget" && i + 1 < argc) {
            rel_budget = std::stod(argv[++i]);
            if (!(rel_budget >= 0.0)) {
                std::fprintf(stderr, "--rel-budget needs a non-negative number\n");
                return 2;
            }
        } else if (arg == "--query-rot-rel" && i + 1 < argc) {
            query_rot_rel = std::stod(argv[++i]);
            if (!(query_rot_rel >= 0.0)) {
                std::fprintf(stderr, "--query-rot-rel needs a non-negative number\n");
                return 2;
            }
        } else {
            std::fprintf(stderr,
                         "usage: %s [--dump-root DIR] [--case DIR] [--ulp-budget N] "
                         "[--rel-budget R] [--query-rot-rel R]\n",
                         argv[0]);
            return 2;
        }
    }

    // The integer fixtures, the serving geometry on them, and the real serving
    // decode unit test's four contexts.
    const std::vector<std::string> default_cases = {
        "decode_ctx128",       "decode_ctx200",       "decode_serving_ctx272",
        "decode_serving_ctx316", "decode_real_ctx64", "decode_real_ctx128",
        "decode_real_ctx200",  "decode_real_ctx1055",
    };
    const std::vector<std::string> cases =
        case_override.empty() ? default_cases : std::vector<std::string>{case_override};
    int failed = 0;
    for (const std::string& name : cases) {
        const std::string dir = case_override.empty() ? dump_root + "/" + name : case_override;
        try {
            golden_oracle::run_case(dir, ulp_budget, rel_budget, query_rot_rel);
        } catch (const std::exception& error) {
            std::fprintf(stderr, "FAIL %s: %s\n", dir.c_str(), error.what());
            ++failed;
        }
    }
    if (failed != 0) {
        std::fprintf(stderr, "golden decode oracle comparison: %d of %zu case(s) FAILED\n",
                     failed, cases.size());
        return 1;
    }
    std::printf("PASS: the C++ golden oracle agrees with the PyTorch golden on all %zu cases\n",
                cases.size());
    return 0;
}
