#include "../common/qwen35_decode_benchmark.hpp"
#include "../common/camodel_session.hpp"

#include <acl/acl.h>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <string>

namespace {
namespace bench = qwen35_bench;
using camodel::check;

struct Session {
    int device;
    aclrtStream stream = nullptr;
    explicit Session(int id) : device(id)
    {
        check(aclInit(nullptr), "aclInit");
        try {
            check(aclrtSetDevice(device), "aclrtSetDevice");
            check(aclrtCreateStream(&stream), "aclrtCreateStream");
        } catch (...) {
            aclrtResetDevice(device);
            aclFinalize();
            throw;
        }
    }
    ~Session()
    {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(device);
        aclFinalize();
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
};

struct Events {
    std::vector<aclrtEvent> events;
    explicit Events(std::size_t count)
    {
        events.reserve(count);
        try {
            for (std::size_t i = 0; i < count; ++i) {
                aclrtEvent event = nullptr;
                check(aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE), "create timeline event");
                events.push_back(event);
            }
        } catch (...) {
            for (auto event : events) aclrtDestroyEvent(event);
            throw;
        }
    }
    ~Events() { for (auto event : events) aclrtDestroyEvent(event); }
    Events(const Events&) = delete;
    Events& operator=(const Events&) = delete;
};

void verify_partials(const bench::Plan& plan, const camodel::Buffer& workspace)
{
    const auto g = plan.geometry;
    if (g.splits == 1) return;
    std::vector<float> partials(g.workspace_bytes() / sizeof(float));
    workspace.download(partials);
    for (std::uint32_t token = 0; token < g.batch; ++token)
        for (std::uint32_t head = 0; head < 6; ++head)
            for (std::uint32_t split = 0; split < g.splits; ++split) {
                const auto descriptor = (std::size_t(token) * g.splits + split) * bench::tiling::kDecodeTilingWords;
                const bool live = plan.descriptors[descriptor + bench::tiling::kTilingWordNumTiles] != 0;
                const auto offset = ((std::size_t(token) * 6 + head) * g.splits + split) *
                    (256 + bench::tq::kPartialTail);
                const float sum = partials[offset + 256 + bench::tq::kPartialSumLane];
                if (!std::isfinite(sum) || (live ? sum <= 0 : sum != 0))
                    throw std::runtime_error("Split-K partial does not match the live page partition");
            }
}

void verify(const std::vector<std::uint16_t>& actual, const std::vector<double>& expected)
{
    double error2 = 0, reference2 = 0, max_abs = 0;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const double got = tq_test::bf16_to_float(actual[i]);
        if (!std::isfinite(got)) throw std::runtime_error("non-finite benchmark output");
        const double error = got - expected[i];
        error2 += error * error;
        reference2 += expected[i] * expected[i];
        max_abs = std::max(max_abs, std::abs(error));
    }
    const double relative_l2 = std::sqrt(error2 / std::max(reference2, 1e-30));
    std::fprintf(stderr, "oracle: relative_l2=%.6g max_abs=%.6g\n", relative_l2, max_abs);
    if (relative_l2 > 0.025 || max_abs > 0.06)
        throw std::runtime_error("2K host golden sanity check failed");
}

void print_static()
{
    std::puts("block_size,batch_size,seq_len,split_k,status,reason,live_splits,fp16_bytes,tq_bytes,"
              "workspace_bytes,kv_read_bytes,tiling_bytes");
    for (auto block : bench::kBlocks) for (auto batch : bench::kBatches)
        for (auto context : bench::kContexts) for (auto splits : bench::kSplits) {
            const bench::Geometry g{block, batch, context, splits};
            const bench::Plan p(g);
            std::printf("%u,%u,%u,%u,%s,%s,%u,%llu,%llu,%llu,%llu,%zu\n", block, batch,
                context, splits, p.valid() ? "VALID" : "UNSUPPORTED", p.reason.c_str(), p.live_splits,
                static_cast<unsigned long long>(g.baseline_bytes()),
                static_cast<unsigned long long>(g.cache_bytes()),
                static_cast<unsigned long long>(g.workspace_bytes()),
                static_cast<unsigned long long>(g.kv_read_bytes()), p.descriptors.size() * sizeof(std::uint32_t));
        }
}

void run_geometry(Session& session, std::uint32_t block, std::uint32_t batch,
    std::uint32_t context, bool validation_only)
{
    const bench::Geometry base{block, batch, context, 1};
    const bench::Plan first(base);
    if (!first.valid()) throw std::runtime_error(first.reason);
    const bench::Fixture fixture(base);
    const auto expected = validation_only ? fixture.expected(first) : std::vector<double>{};
    camodel::Buffer key(fixture.key.size()), value(fixture.value.size());
    camodel::Buffer scales(fixture.scales.size() * sizeof(float));
    camodel::Buffer raw(fixture.raw.size() * sizeof(std::uint16_t));
    camodel::Buffer query(fixture.query.size() * sizeof(float));
    camodel::Buffer signs(fixture.signs.size() * sizeof(float));
    camodel::Buffer table(first.table.size() * sizeof(std::int32_t));
    camodel::Buffer lengths(first.lengths.size() * sizeof(std::int32_t));
    camodel::Buffer mode(64), unused(32), output(fixture.raw.size() * sizeof(std::uint16_t));
    key.upload(fixture.key);
    value.upload(fixture.value);
    scales.upload(fixture.scales);
    raw.upload(fixture.raw);
    signs.upload(fixture.signs);
    table.upload(first.table);
    lengths.upload(first.lengths);
    check(aclrtMemset(mode.ptr, mode.bytes, 0, mode.bytes), "clear mode tables");
    check(aclrtMemset(unused.ptr, unused.bytes, 0, unused.bytes), "clear unused basis inputs");
    double best_mean = std::numeric_limits<double>::infinity();
    std::uint32_t best_split = 0;
    for (auto splits : bench::kSplits) {
        const bench::Geometry g{block, batch, context, splits};
        const bench::Plan p(g);
        if (!p.valid()) {
            std::printf("%u,%u,%u,%u,UNSUPPORTED,%s,,,,,,,\n", block, batch, context, splits, p.reason.c_str());
            continue;
        }
        camodel::Buffer workspace(g.workspace_bytes());
        camodel::Buffer descriptors(p.descriptors.size() * sizeof(std::uint32_t));
        descriptors.upload(p.descriptors);
        check(aclrtMemset(workspace.ptr, workspace.bytes, 0, workspace.bytes), "clear workspace");
        check(aclrtMemset(output.ptr, output.bytes, 0xff, output.bytes), "poison output");
        const bench::fd::Buffers decode{query.ptr, key.ptr, value.ptr, scales.ptr, table.ptr,
            lengths.ptr, mode.ptr, workspace.ptr, output.ptr, descriptors.ptr};
        const bench::fd::RawQueryBuffers buffers{raw.ptr, signs.ptr, unused.ptr, unused.ptr, unused.ptr, decode};
        const auto launch = [&] { launch_fused_decode_opt_raw_query(session.stream, &p.config, &buffers); };
        if (!expected.empty()) {
            launch();
            check(aclrtSynchronizeStream(session.stream), "golden synchronize");
            std::vector<std::uint16_t> actual(fixture.raw.size());
            output.download(actual);
            verify(actual, expected);
            verify_partials(p, workspace);
        }
        if (validation_only) continue;
        for (int i = 0; i < 10; ++i) launch();
        check(aclrtSynchronizeStream(session.stream), "warmup synchronize");
        Events timeline(100);
        for (std::size_t i = 0; i < 50; ++i) {
            check(aclrtRecordEvent(timeline.events[2 * i], session.stream), "record start");
            launch();
            check(aclrtRecordEvent(timeline.events[2 * i + 1], session.stream), "record stop");
        }
        check(aclrtSynchronizeEvent(timeline.events.back()), "timing synchronize");
        std::vector<double> samples;
        for (std::size_t i = 0; i < 50; ++i) {
            float ms = 0;
            check(aclrtEventElapsedTime(&ms, timeline.events[2 * i], timeline.events[2 * i + 1]), "elapsed time");
            if (!std::isfinite(ms) || ms <= 0) throw std::runtime_error("invalid device timing sample");
            samples.push_back(static_cast<double>(ms) * 1000);
        }
        std::sort(samples.begin(), samples.end());
        const double mean = std::accumulate(samples.begin(), samples.end(), 0.0) / samples.size();
        const double median = (samples[24] + samples[25]) * 0.5;
        const double p99 = samples[49];
        if (mean < best_mean) {
            best_mean = mean;
            best_split = splits;
        }
        std::printf("%u,%u,%u,%u,MEASURED,,%u,%.6f,%.6f,%.6f,%.6f,%llu,%.6f\n",
            block, batch, context, splits, p.live_splits, samples.front(), median, mean, p99,
            static_cast<unsigned long long>(g.kv_read_bytes()), static_cast<double>(g.kv_read_bytes()) / mean / 1000);
        std::fflush(stdout);
    }
    if (best_split != 0)
        std::fprintf(stderr, "best measured split: block=%u batch=%u context=%u split=%u mean_us=%.6f\n",
            block, batch, context, best_split, best_mean);
}
}

int main(int argc, char** argv)
{
    try {
        if (argc == 1 || (argc == 2 && std::string(argv[1]) == "--static")) {
            print_static();
            return 0;
        }
        if (argc != 4 || std::string(argv[1]) != "--run" || std::string(argv[2]) != "--device")
            throw std::invalid_argument("usage: bench_qwen35_decode [--static | --run --device ID]");
        const std::string id(argv[3]);
        std::size_t used = 0;
        const int device = std::stoi(id, &used);
        if (used != id.size() || device < 0) throw std::invalid_argument("invalid device ID");
        Session session(device);
        const char* soc = aclrtGetSocName();
        if (!soc || std::string(soc).find("Ascend950PR") != 0)
            throw std::runtime_error("benchmark requires Ascend950PR hardware");
        std::fprintf(stderr, "device=%d soc=%s warmup=10 samples=50 clock=ACL_EVENT_TIME_LINE output_stage=0 fused_context_limit=0\n", device, soc);
        for (auto block : bench::kBlocks) for (auto batch : bench::kBatches)
            run_geometry(session, block, batch, 2048, true);
        std::fprintf(stderr, "all 16 short-context golden checks passed; starting timing sweep\n");
        std::puts("block_size,batch_size,seq_len,split_k,status,reason,live_splits,min_us,p50_us,mean_us,p99_us,kv_read_bytes,effective_kv_GB_s");
        for (auto block : bench::kBlocks) for (auto batch : bench::kBatches)
            for (auto context : bench::kContexts) run_geometry(session, block, batch, context, false);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "benchmark failed: %s\n", error.what());
        return 1;
    }
}
