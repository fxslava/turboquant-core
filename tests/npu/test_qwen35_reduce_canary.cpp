#include "../common/camodel_session.hpp"
#include "../common/turboquant_test_common.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

extern "C" void launch_qwen35_reduce_canary(void*, void*, void*, std::uint32_t);

int main(int argc, char** argv)
{
    try {
        if (argc != 2 || (std::string(argv[1]) != "1" && std::string(argv[1]) != "2"))
            throw std::invalid_argument("usage: test_qwen35_reduce_canary_sim 1|2");
        camodel::Session session;
        for (std::uint32_t splits : {1u, 2u}) {
            if (std::string(argv[1]) != std::to_string(splits)) continue;
            std::vector<float> workspace(6 * splits * 272, 0);
            std::vector<double> expected(6 * 256, 0);
            for (std::size_t head = 0; head < 6; ++head) {
                double denominator = 0;
                for (std::uint32_t split = 0; split < splits; ++split) {
                    const auto offset = (head * splits + split) * 272;
                    const float maximum = static_cast<float>(split) * 0.5f;
                    const float mass = 32.0f + static_cast<float>(head * 3 + split * 7);
                    workspace[offset + 256] = maximum;
                    workspace[offset + 264] = mass;
                    const double weight = std::exp(static_cast<double>(maximum));
                    denominator += weight * mass;
                    for (std::size_t col = 0; col < 256; ++col) {
                        const float value = 0.5f + static_cast<float>((head + col + split * 5) % 13) * 0.125f;
                        workspace[offset + col] = value * mass;
                        expected[head * 256 + col] += weight * workspace[offset + col];
                    }
                }
                for (std::size_t col = 0; col < 256; ++col) expected[head * 256 + col] /= denominator;
            }
            camodel::Buffer device_workspace(workspace.size() * sizeof(float));
            camodel::Buffer output(expected.size() * sizeof(std::uint16_t));
            device_workspace.upload(workspace);
            camodel::check(aclrtMemset(output.ptr, output.bytes, 0xff, output.bytes), "poison canary output");
            launch_qwen35_reduce_canary(session.stream, device_workspace.ptr, output.ptr, splits);
            camodel::check(aclrtSynchronizeStream(session.stream), "reduction canary synchronize");
            std::vector<std::uint16_t> actual(expected.size());
            output.download(actual);
            double error2 = 0, reference2 = 0, maximum = 0;
            for (std::size_t i = 0; i < actual.size(); ++i) {
                const double got = tq_test::bf16_to_float(actual[i]);
                if (!std::isfinite(got)) throw std::runtime_error("non-finite reduction result");
                const double error = got - expected[i];
                error2 += error * error;
                reference2 += expected[i] * expected[i];
                maximum = std::max(maximum, std::abs(error));
            }
            const double relative = std::sqrt(error2 / reference2);
            std::printf("reduce canary: heads=6 dim=256 splits=%u per_block=3 relative_l2=%.9g max_abs=%.9g\n",
                splits, relative, maximum);
            std::fflush(stdout);
            if (relative >= 0.01) throw std::runtime_error("reduction golden mismatch");
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL reduction canary: %s\n", error.what());
        return 1;
    }
}
