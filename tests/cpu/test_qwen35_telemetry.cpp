#include "tests/common/qwen35_decode_benchmark.hpp"
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
namespace b = qwen35_bench;
struct Partial {
    double maximum = -std::numeric_limits<double>::infinity(), mass = 0;
    int position = 0;
    bool improved = false;
    std::array<double, 256> acc{};
};
int main()
{
    const b::Geometry g{128, 1, 2048, 2};
    const b::Plan plan(g);
    const b::Fixture f(g);
    tq_golden::DecodeGeometry dg;
    dg.group_heads = 6;
    dg.page_tokens = 128;
    dg.max_pages = 16;
    tq_golden::DensePi pi(f.signs);
    std::array<std::array<Partial, 2>, 6> parts{};
    for (int h = 0; h < 6; ++h) {
        std::array<double, 256> raw{}, q{};
        for (int c = 0; c < 256; ++c) raw[c] = f.query[h * 256 + c];
        pi.rotate_query(raw.data(), q.data());
        for (int s = 0; s < 2; ++s) {
            auto& p = parts[h][s];
            std::array<double, 1024> scores{};
            double before_last = p.maximum;
            for (int r = 0; r < 1024; ++r) {
                const int row = s * 1024 + r;
                const int slot = plan.table[row / 128] * 128 + row % 128;
                for (int c = 0; c < 256; ++c)
                    scores[r] += q[c] * tq_golden::dequantise_nz(dg, f.key.data(), f.scales.data(), 0, slot, 0, c, 0);
                scores[r] *= 0.0625;
                if (r < 960) before_last = std::max(before_last, scores[r]);
                if (scores[r] > p.maximum) { p.maximum = scores[r]; p.position = row; }
            }
            p.improved = p.maximum > before_last;
            for (int r = 0; r < 1024; ++r) {
                const int row = s * 1024 + r;
                const int slot = plan.table[row / 128] * 128 + row % 128;
                const double w = std::exp(scores[r] - p.maximum);
                p.mass += w;
                for (int c = 0; c < 256; ++c)
                    p.acc[c] += w * tq_golden::dequantise_nz(dg, f.value.data(), f.scales.data(), 1, slot, 0, c, 0);
            }
        }
    }
    const auto expected = f.expected(plan);
    double bad_error = 0, good_error = 0, norm = 0;
    for (int h = 0; h < 6; ++h) {
        double good_max = std::max(parts[h][0].maximum, parts[h][1].maximum);
        auto bad = parts[h];
        for (int s = 0; s < 2; ++s) {
            const int first = (h / 3) * 3;
            if (parts[first][s].improved) bad[s].maximum = bad[s].position;
            if (parts[first + 1][s].improved) bad[s].mass = bad[s].position;
        }
        const double bad_max = std::max(bad[0].maximum, bad[1].maximum);
        double good_mass = 0, bad_mass = 0, good_mean = 0, bad_mean = 0, ref_mean = 0;
        for (int s = 0; s < 2; ++s) {
            good_mass += std::exp(parts[h][s].maximum - good_max) * parts[h][s].mass;
            bad_mass += std::exp(bad[s].maximum - bad_max) * bad[s].mass;
        }
        for (int c = 0; c < 256; ++c) {
            double good = 0, wrong = 0;
            for (int s = 0; s < 2; ++s) {
                good += std::exp(parts[h][s].maximum - good_max) * parts[h][s].acc[c];
                wrong += std::exp(bad[s].maximum - bad_max) * bad[s].acc[c];
            }
            good /= good_mass;
            wrong /= bad_mass;
            const double ref = expected[h * 256 + c];
            good_error += (good - ref) * (good - ref);
            bad_error += (wrong - ref) * (wrong - ref);
            norm += ref * ref;
            good_mean += good; bad_mean += wrong; ref_mean += ref;
        }
        std::printf("head=%d old=%.9g fixed=%.9g golden=%.9g last_improved=%d,%d\n", h,
            bad_mean / 256, good_mean / 256, ref_mean / 256, int(parts[h][0].improved), int(parts[h][1].improved));
    }
    const double old_relative = std::sqrt(bad_error / norm);
    const double fixed_relative = std::sqrt(good_error / norm);
    std::printf("old_relative_l2=%.9g fixed_relative_l2=%.9g\n", old_relative, fixed_relative);
    if (old_relative < 0.3 || fixed_relative >= 1e-12) return 1;
    return 0;
}
