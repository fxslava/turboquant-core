#include "tests/common/turboquant_production.hpp"
#include "tests/common/camodel_session.hpp"
#include "tests/common/turboquant_scatter_oracle.hpp"
#ifdef TQ_HAS_E8_KERNELS
#include "src/turboquant/common/e8_component.hpp"
#endif
#include "src/turboquant/common/turboquant_launch.h"

#include <acl/acl.h>
#include <memory>

namespace {
using namespace tq_production;
using camodel::check;

class Session {
public:
    aclrtStream stream = nullptr;

    explicit Session(unsigned id) : device_(static_cast<int>(id))
    {
        check(aclInit(nullptr), "aclInit");
        try {
            check(aclrtSetDevice(device_), "aclrtSetDevice");
            ready_ = true;
            const char* name = aclrtGetSocName();
            require(name && std::string_view(name).find("Ascend950PR") == 0, "expected Ascend950PR device");
            check(aclrtCreateStream(&stream), "aclrtCreateStream");
        } catch (...) {
            if (ready_) aclrtResetDevice(device_);
            aclFinalize();
            throw;
        }
    }

    ~Session()
    {
        aclrtDestroyStream(stream);
        aclrtResetDevice(device_);
        aclFinalize();
    }

    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;

    void synchronize() const { check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream"); }

private:
    int device_;
    bool ready_ = false;
};

class Gm {
public:
    explicit Gm(std::uint64_t bytes) : bytes_(bytes)
    {
        require(bytes && bytes <= std::numeric_limits<std::size_t>::max() - 2 * kGuard, "invalid GM allocation size");
        check(aclrtMalloc(&base_, bytes + 2 * kGuard, ACL_MEM_MALLOC_NORMAL_ONLY), "aclrtMalloc");
        try {
            reset();
            check(aclrtMemset(base_, kGuard, kCanary, kGuard), "leading GM canary");
            check(aclrtMemset(ptr(bytes), kGuard, kCanary, kGuard), "trailing GM canary");
        } catch (...) {
            aclrtFree(base_);
            throw;
        }
    }

    ~Gm() { aclrtFree(base_); }
    Gm(const Gm&) = delete;
    Gm& operator=(const Gm&) = delete;

    std::uint8_t* ptr(std::uint64_t offset = 0) const
    {
        return static_cast<std::uint8_t*>(base_) + kGuard + offset;
    }

    template<class T> T* as(std::uint64_t offset = 0) const
    {
        return reinterpret_cast<T*>(ptr(offset));
    }

    void reset(std::uint8_t fill = 0)
    {
        check(aclrtMemset(ptr(), bytes_, fill, bytes_), "payload init");
    }

    template<class T>
    void upload(const T* data, std::uint64_t size, std::uint64_t offset = 0) const
    {
        require(offset <= bytes_ && size <= (bytes_ - offset) / sizeof(T), "H2D copy exceeds payload");
        check(aclrtMemcpy(ptr(offset), bytes_ - offset, data, size * sizeof(T), ACL_MEMCPY_HOST_TO_DEVICE), "H2D");
    }

    template<class T>
    void upload(const std::vector<T>& data) const
    {
        upload(data.data(), data.size());
    }

    template<class T>
    void download(T* data, std::uint64_t size, std::uint64_t offset = 0) const
    {
        require(offset <= bytes_ && size <= (bytes_ - offset) / sizeof(T), "D2H copy exceeds payload");
        check(aclrtMemcpy(data, size * sizeof(T), ptr(offset), size * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST), "D2H");
    }

    template<class T>
    void download(std::vector<T>& data) const
    {
        download(data.data(), data.size());
    }

    void verify() const
    {
        std::array<std::uint8_t, kGuard> bytes{};
        for (const void* address : {static_cast<const void*>(base_), static_cast<const void*>(ptr(bytes_))}) {
            check(aclrtMemcpy(bytes.data(), bytes.size(), address, bytes.size(), ACL_MEMCPY_DEVICE_TO_HOST), "GM canary D2H");
            require(std::all_of(bytes.begin(), bytes.end(), [](auto x) { return x == kCanary; }), "GM guard overwritten");
        }
    }

private:
    static constexpr std::uint64_t kGuard = 256;
    static constexpr std::uint8_t kCanary = 0xa7;
    void* base_ = nullptr;
    std::uint64_t bytes_;
};

class CapturedTimeline {
public:
    explicit CapturedTimeline(aclrtStream stream) : stream_(stream)
    {
        check(aclmdlRICaptureBegin(stream_, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL), "ACL graph capture begin");
        capturing_ = true;
    }
    ~CapturedTimeline()
    {
        if (capturing_) aclmdlRICaptureEnd(stream_, &model_);
        if (executed_) aclrtSynchronizeStream(stream_);
        if (model_) aclmdlRIDestroy(model_);
    }
    CapturedTimeline(const CapturedTimeline&) = delete;
    CapturedTimeline& operator=(const CapturedTimeline&) = delete;
    void execute()
    {
        const auto status = aclmdlRICaptureEnd(stream_, &model_);
        capturing_ = false;
        check(status, "ACL graph capture end");
        executed_ = true;
        check(aclmdlRIExecuteAsync(model_, stream_), "ACL graph execute");
    }
private:
    aclrtStream stream_;
    aclmdlRI model_ = nullptr;
    bool capturing_ = false, executed_ = false;
};

class Event {
public:
    Event() { check(aclrtCreateEventWithFlag(&event_, ACL_EVENT_TIME_LINE), "aclrtCreateEventWithFlag"); }
    ~Event() { aclrtDestroyEvent(event_); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
    aclrtEvent get() const { return event_; }

private:
    aclrtEvent event_ = nullptr;
};

template<class Launch>
void time_case(const Session& session, const Filters& f, std::uint64_t bytes, Launch launch)
{
#ifdef TQ_PRODUCTION_SIM
    throw std::runtime_error("CAModel timings are not NPU performance; use --verify or the accuracy runner");
#endif
    for (unsigned i = 0; i < std::max(1u, f.warmup); ++i) launch();
    session.synchronize();

    const auto events = std::make_unique<Event[]>(std::uint64_t{2} * f.iterations);
    std::unique_ptr<CapturedTimeline> graph;
    if (f.timing == "graph") graph = std::make_unique<CapturedTimeline>(session.stream);
    for (unsigned i = 0; i < f.iterations; ++i) {
        const auto record = [&](aclrtEvent event) {
            check(graph ? aclrtRecordEventWithFlag(event, session.stream, ACL_EVENT_RECORD_EXTERNAL)
                        : aclrtRecordEvent(event, session.stream), "record device timeline event");
        };
        record(events[std::uint64_t{2} * i].get());
        launch();
        record(events[std::uint64_t{2} * i + 1].get());
    }
    if (graph) graph->execute();
    session.synchronize();
    std::vector<double> samples;
    samples.reserve(f.iterations);
    for (unsigned i = 0; i < f.iterations; ++i) {
        float ms = 0;
        check(aclrtSynchronizeEvent(events[std::uint64_t{2} * i + 1].get()), "aclrtSynchronizeEvent");
        check(aclrtEventElapsedTime(&ms, events[std::uint64_t{2} * i].get(),
            events[std::uint64_t{2} * i + 1].get()), "aclrtEventElapsedTime");
        samples.push_back(1000.0 * ms);
        require(std::isfinite(ms) && ms > 0, "device event has no positive finite elapsed time");
    }
    std::printf(" timing_source=aclrtEventElapsedTime timing_scope=%s samples=%u warmup=%u", graph ? "captured_device_stream" : "eager_device_stream", f.iterations, std::max(1u, f.warmup));
    const auto timing = summarize(samples);
    std::printf(" device_p50_us=%.3f device_p95_us=%.3f device_mean_us=%.3f effective_bytes=%llu device_effective_GB_s=%.3f",
        timing.p50, timing.p95, timing.mean, static_cast<unsigned long long>(bytes), bytes / timing.p50 / 1000);
    if (f.samples) {
        std::printf(" device_samples_us=");
        for (unsigned i = 0; i < samples.size(); ++i) std::printf("%s%.6f", i ? "," : "", samples[i]);
    }
}

void verify_attention(const Plan& plan, Mode mode, const std::vector<std::uint16_t>& actual)
{
    const auto m = metrics(actual, attention_reference(plan, mode, false));
    print_accuracy(m);
    const double limit = mode == Mode::e8 ? 0.025 : 0.075;
    require(m.relative_l2 < limit && m.cosine > 1 - limit * limit,
        "BF16 context differs from CPU attention");
    constexpr double kE8MinimumCosine = 0.99999;
    if (mode == Mode::e8) require(m.cosine >= kE8MinimumCosine, "E8 cosine must be at least 0.99999");
}

void verify_all(std::initializer_list<const Gm*> buffers)
{
    for (const auto* buffer : buffers) if (buffer) buffer->verify();
}

constexpr std::uint32_t kLm4RowBytes = kDim / 2;

void upload_lm4(const Plan& plan, const Gm& key, const Gm& value, const Gm& scales)
{
    const auto scale_stride = layout::NzScaleSlotFloats(plan.p.kv_heads);
    const auto packed_page = std::uint64_t{plan.block} * plan.p.kv_heads * kLm4RowBytes;
    const auto scale_page = std::uint64_t{plan.block} * scale_stride;
    constexpr std::uint32_t repeats = 64;
    const auto pages_per_batch = kPatternPages * repeats;
    std::vector<std::uint8_t> k(packed_page * pages_per_batch), v(k.size());
    std::vector<float> s(scale_page * pages_per_batch);
    for (unsigned page = 0; page < kPatternPages; ++page)
        for (unsigned row = 0; row < plan.block; ++row) {
            const auto slot = std::uint64_t{page} * plan.block + row;
            for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv)
                for (unsigned lane = 0; lane < 2; ++lane) {
                    const auto r = static_cast<std::uint32_t>(slot);
                    (s[layout::NzTiledScaleIndex(slot, kv, lane, plan.p.kv_heads)]) = row_scale(r, kv, lane);
                    auto& plane = lane ? v : k;
                    for (unsigned c = 0; c < kLm4RowBytes; ++c) {
                        const unsigned low = (lm_code(Mode::lm4, r, kv, lane, c) + 8) & 15;
                        const unsigned high = (lm_code(Mode::lm4, r, kv, lane, c + kLm4RowBytes) + 8) & 15;
                        plane[layout::NzTiledPackedByte(slot, kv, c, plan.p.kv_heads, kLm4RowBytes)] =
                            static_cast<std::uint8_t>(low | high << 4);
                    }
                }
        }

    for (unsigned r = 1; r < repeats; ++r) {
        std::copy_n(k.data(), packed_page * kPatternPages, k.data() + packed_page * kPatternPages * r);
        std::copy_n(v.data(), packed_page * kPatternPages, v.data() + packed_page * kPatternPages * r);
        std::copy_n(s.data(), scale_page * kPatternPages, s.data() + scale_page * kPatternPages * r);
    }
    for (unsigned p = 0; p < plan.pages; p += pages_per_batch) {
        const auto n = std::min(plan.pages - p, pages_per_batch);
        key.upload(k.data(), n * packed_page, p * packed_page);
        value.upload(v.data(), n * packed_page, p * packed_page);
        scales.upload(s.data(), n * scale_page, p * scale_page * sizeof(float));
    }
}

void lm4_decode(const Session& session, const Filters& f, const Plan& plan)
{
    const auto c = plan.native_launch();
    const auto query_elements = std::uint64_t{c.tokens} * c.heads * kDim;
    const auto packed_bytes = plan.slots() * c.kv_heads * kLm4RowBytes;
    Gm key(packed_bytes), value(packed_bytes);
    Gm scales(plan.slots() * layout::NzScaleSlotFloats(c.kv_heads) * sizeof(float));
    upload_lm4(plan, key, value, scales);
    std::vector<float> query(query_elements);
    for (unsigned row = 0; row < c.tokens; ++row)
        for (unsigned head = 0; head < c.heads; ++head)
            query[(std::uint64_t{row} * c.heads + head) * kDim + query_channel(row, head)] = query_value(row, head);
    alignas(tiling::TurboQuantDecodeTilingData) TilingWords words;
    const auto word_count = tiling::decode_tiling_words(c.tokens, c.splits, c.grid);
    const auto status = tiling::build_decode_tiling(tiling::decode_tiling_request(c),
        plan.lengths.data(), plan.table.data(), words.data());
    require(status == tiling::DecodeTilingStatus::kOk, "LM4 native tiling rejected");
    Gm db(word_count * sizeof(std::uint32_t)), q(query_elements * sizeof(float));
    Gm table(plan.table.size() * sizeof(std::int32_t));
    Gm lengths(std::uint64_t{32} * sizeof(std::int32_t));
    Gm lut(sizeof(kLm4Centroids)), output(query_elements * sizeof(std::uint16_t));
    Gm work(std::uint64_t{c.tokens} * c.heads * c.splits * (kDim + layout::kPartialTail) * sizeof(float));
    lengths.upload(plan.lengths);
    table.upload(plan.table);
    lut.upload(kLm4Centroids, std::size(kLm4Centroids));
    q.upload(query);
    db.upload(words.data(), word_count);
    output.reset(0xff);
    const fd::Buffers b{q.ptr(), key.ptr(), value.ptr(), scales.ptr(), table.ptr(), lengths.ptr(), lut.ptr(),
        work.ptr(), output.ptr(), db.ptr()};
    const auto launch = [&] { launch_fused_decode_opt(session.stream, &c, &b); };
    if (f.verify) {
        launch();
        session.synchronize();
        std::vector<std::uint16_t> actual(query_elements);
        output.download(actual);
        verify_attention(plan, Mode::lm4, actual);
    } else time_case(session, f, plan.kv_bytes(Mode::lm4, false), launch);
    verify_all({&key, &value, &scales, &lengths, &table, &lut, &q, &db, &work, &output});
}

#ifdef TQ_HAS_E8_KERNELS
constexpr unsigned kE8QueryFloats = e8_decode::kPaddedHeads * kDim;
constexpr unsigned kE8OutputElements = e8_decode::kHeads * kDim;

void upload_e8(const Plan& plan, const Gm& key, const Gm& value, bool verify)
{
    const auto tiles = plan.block / kTileRows;
    const auto packets_per_page = std::uint64_t{tiles} * plan.p.kv_heads;
    constexpr unsigned repeats = 64;
    constexpr unsigned batch = kPatternPages * repeats;
    std::vector<Packet> k(packets_per_page * batch), v(k.size());
    for (unsigned page = 0; page < kPatternPages; ++page)
        for (unsigned t = 0; t < tiles; ++t)
            for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv) {
                const auto first_row = page * plan.block + t * kTileRows;
                const auto idx = page * packets_per_page + std::uint64_t{t} * plan.p.kv_heads + kv;
                k[idx] = packet_fixture(first_row, kv, 0);
                v[idx] = packet_fixture(first_row, kv, 1);
                if (verify) {
                    verify_packet(k[idx], first_row, kv, 0);
                    verify_packet(v[idx], first_row, kv, 1);
                }
            }
    for (unsigned r = 1; r < repeats; ++r) {
        std::copy_n(k.data(), packets_per_page * kPatternPages, k.data() + packets_per_page * kPatternPages * r);
        std::copy_n(v.data(), packets_per_page * kPatternPages, v.data() + packets_per_page * kPatternPages * r);
    }
    for (unsigned p = 0; p < plan.pages; p += batch) {
        const auto count = std::min(plan.pages - p, batch) * packets_per_page;
        const auto offset = p * packets_per_page * sizeof(Packet);
        key.upload(k.data(), count, offset);
        value.upload(v.data(), count, offset);
    }
}

void verify_e8_witness(const Plan& plan, const Gm& witness, unsigned seq, unsigned kv, unsigned plane, std::uint64_t offset = 0)
{
    const auto tiles = ceil_div(plan.lengths[seq], kTileRows);
    std::vector<std::uint16_t> reconstructed(std::uint64_t{tiles} * kTileRows * kDim);
    witness.download(reconstructed.data(), reconstructed.size(), offset);
    std::array<float, kDim> expected{};
    for (unsigned row = 0; row < tiles * kTileRows; ++row) {
        if (row < static_cast<unsigned>(plan.lengths[seq])) reconstruct(Mode::e8, plan.pattern_row(seq, row), kv, plane, expected.data());
        else expected.fill(0);
        for (unsigned c = 0; c < kDim; ++c)
            require(reconstructed[std::uint64_t{row} * kDim + c] == tq_test::float_to_bf16_rne(expected[c]),
                "E8 fused dequant witness differs from CPU golden");
    }
}

void e8_decode_case(const Session& session, const Filters& f, const Plan& plan)
{
    const auto tiles_per_page = plan.block / kTileRows;
    const auto packet_bytes = plan.slots() / kTileRows * plan.p.kv_heads * sizeof(Packet);
    Gm key(packet_bytes), value(packet_bytes);
    upload_e8(plan, key, value, f.verify);
    const auto max_tiles = ceil_div(plan.p.context, kTileRows);
    const auto groups = ceil_div(plan.p.group, e8_decode::kHeads);
    const auto lanes = plan.p.sequences * plan.p.kv_heads;
    const auto jobs = std::uint64_t{lanes} * groups;
    const auto grid = lanes * plan.p.splits;
    unsigned max_split_tiles = 0;
    for (auto length : plan.lengths)
        for (unsigned split = 0; split < plan.p.splits; ++split)
            max_split_tiles = std::max(max_split_tiles, e8_decode::split_range(length, plan.block, plan.p.splits, split).tiles);
    const auto stride = std::uint64_t{ceil_div(max_split_tiles, 4)} * 4;
    std::vector<std::uint64_t> offset_map(grid * stride);
    std::vector<e8_decode::Task> task_map(grid);
    std::vector<float> query(jobs * kE8QueryFloats);
    for (unsigned seq = 0; seq < plan.p.sequences; ++seq)
        for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv) {
            const auto lane = seq * plan.p.kv_heads + kv;
            for (unsigned split = 0; split < plan.p.splits; ++split) {
                const auto range = e8_decode::split_range(plan.lengths[seq], plan.block, plan.p.splits, split);
                const auto task = lane * plan.p.splits + split;
                task_map[task] = {task * stride, std::uint64_t{lane} * max_tiles + range.first_tile,
                                  std::uint64_t{lane} * groups * max_tiles + range.first_tile,
                                  range.tiles, range.last_rows, lane, 0};
                for (unsigned tile = 0; tile < range.tiles; ++tile) {
                    const auto logical_tile = range.first_tile + tile;
                    const auto page = plan.physical(seq, logical_tile / tiles_per_page);
                    offset_map[task * stride + tile] = packet_offset(page, logical_tile % tiles_per_page,
                                                                     kv, tiles_per_page, plan.p.kv_heads);
                }
            }
            for (unsigned gh = 0; gh < plan.p.group; ++gh) {
                const auto job = std::uint64_t{lane} * groups + gh / e8_decode::kHeads;
                const auto head = kv * plan.p.group + gh;
                query[job * kE8QueryFloats + std::uint64_t{gh % e8_decode::kHeads} * kDim + query_channel(seq, head)] = query_value(seq, head);
            }
        }
    Gm offsets(offset_map.size() * sizeof(std::uint64_t)), tasks(task_map.size() * sizeof(e8_decode::Task));
    Gm q(query.size() * sizeof(float)), output(jobs * kE8OutputElements * sizeof(std::uint16_t));
    const auto workspace_bytes = e8_decode::workspace_floats(lanes, groups, plan.p.splits) * sizeof(float);
    std::unique_ptr<Gm> work, kw, vw, sw;
    if (workspace_bytes) work = std::make_unique<Gm>(workspace_bytes);
    if (f.verify) {
        const auto witness_bytes = std::uint64_t{lanes} * max_tiles * kTileRows * kDim * sizeof(std::uint16_t);
        kw = std::make_unique<Gm>(witness_bytes);
        vw = std::make_unique<Gm>(witness_bytes);
        sw = std::make_unique<Gm>(jobs * max_tiles * kTileRows * e8_decode::kHeads * sizeof(float));
    }
    offsets.upload(offset_map);
    tasks.upload(task_map);
    q.upload(query);
    output.reset(0xff);
    const e8_decode::Launch params{max_split_tiles, 64, kAttentionScale, lanes, groups, plan.p.splits, max_tiles};
    const e8_decode::Buffers buffers{key.ptr(), value.ptr(), offsets.ptr(), q.ptr(), output.ptr(),
        kw ? kw->ptr() : nullptr, vw ? vw->ptr() : nullptr, sw ? sw->ptr() : nullptr,
        tasks.ptr(), work ? work->ptr() : nullptr};
    const auto launch = [&] { launch_e8_fused_decode_fp32(session.stream, &params, &buffers); };
    std::printf(" launches=%u reader_launches=1 reduce_launches=%u grid=%u query_groups=%u workspace_bytes=%llu witnesses=%s physical_query_heads=%llu",
        1 + unsigned(plan.p.splits > 1), unsigned(plan.p.splits > 1), grid, groups,
        static_cast<unsigned long long>(workspace_bytes), f.verify ? "on" : "off",
        static_cast<unsigned long long>(jobs * e8_decode::kHeads));
    if (f.verify) {
        launch();
        session.synchronize();
        for (unsigned seq = 0; seq < plan.p.sequences; ++seq)
            for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv) {
                const auto lane = seq * plan.p.kv_heads + kv;
                const auto offset = std::uint64_t{lane} * max_tiles * kTileRows * kDim * sizeof(std::uint16_t);
                verify_e8_witness(plan, *kw, seq, kv, 0, offset);
                verify_e8_witness(plan, *vw, seq, kv, 1, offset);
            }
        std::vector<std::uint16_t> padded(jobs * kE8OutputElements);
        output.download(padded);
        std::vector<std::uint16_t> actual(std::uint64_t{plan.p.sequences} * plan.heads() * kDim);
        for (unsigned lane = 0; lane < lanes; ++lane)
            for (unsigned h = 0; h < plan.p.group; ++h) {
                const auto source = (std::uint64_t{lane} * groups + h / e8_decode::kHeads) * kE8OutputElements + std::uint64_t{h % e8_decode::kHeads} * kDim;
                std::copy_n(padded.data() + source, kDim, actual.data() + (std::uint64_t{lane} * plan.p.group + h) * kDim);
            }
        verify_attention(plan, Mode::e8, actual);
    } else time_case(session, f, plan.kv_bytes(Mode::e8, false), launch);
    verify_all({&key, &value, &offsets, &tasks, &q, &output, work.get(), kw.get(), vw.get(), sw.get()});
}

void e8_dequant(const Session& session, const Filters& f, const Plan& plan)
{
    constexpr unsigned rows_per_launch = 16;
    const auto tiles = plan.block / kTileRows;
    const auto packets = std::uint64_t{2} * plan.p.kv_heads * tiles;
    const auto elements = packets * kTileRows * kDim;
    const auto index = [&](unsigned plane, unsigned kv, unsigned tile) {
        return (std::uint64_t{plane} * plan.p.kv_heads + kv) * tiles + tile;
    };
    std::vector<Packet> input(packets);
    for (unsigned plane = 0; plane < 2; ++plane)
        for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv)
            for (unsigned tile = 0; tile < tiles; ++tile) {
                auto& frame = input[index(plane, kv, tile)];
                frame = packet_fixture(tile * kTileRows, kv, plane);
                if (f.verify) verify_packet(frame, tile * kTileRows, kv, plane);
            }
    Gm data(packets * sizeof(Packet)), bf16(elements * sizeof(std::uint16_t)), int8(elements * sizeof(std::int8_t));
    data.upload(input);
    bf16.reset(0xff);
    const auto launch = [&] {
        for (std::uint64_t frame = 0; frame < packets; ++frame)
            for (std::uint64_t row = 0; row < kTileRows; row += rows_per_launch) {
                const auto base = frame * sizeof(Packet);
                const auto output = (frame * kTileRows + row) * kDim;
                launch_e8_parallel_component(session.stream, data.as<std::uint16_t>(base + row * packet::kLabelRowBytes),
                    data.as<float>(base + packet::kScaleOffset + row * sizeof(float)),
                    bf16.as<std::uint16_t>(output * sizeof(std::uint16_t)), int8.as<std::int8_t>(output));
            }
    };
    std::printf(" dequant_rows_per_launch=%u launches=%llu",
        rows_per_launch, static_cast<unsigned long long>(packets * kTileRows / rows_per_launch));
    if (f.verify) {
        launch();
        session.synchronize();
        std::vector<std::uint16_t> actual(elements);
        std::vector<double> expected(elements);
        bf16.download(actual);
        std::array<float, kDim> row{};
        for (unsigned plane = 0; plane < 2; ++plane)
            for (unsigned kv = 0; kv < plan.p.kv_heads; ++kv)
                for (unsigned tile = 0; tile < tiles; ++tile)
                    for (unsigned r = 0; r < kTileRows; ++r) {
                        reconstruct(Mode::e8, tile * kTileRows + r, kv, plane, row.data());
                        const auto offset = (index(plane, kv, tile) * kTileRows + r) * kDim;
                        std::copy(row.begin(), row.end(), expected.data() + offset);
                    }
        const auto m = metrics(actual, expected);
        print_accuracy(m);
        require(m.max_ulp == 0, "E8 dequant BF16 witness must be bit-exact");
    } else time_case(session, f, packets * sizeof(Packet) +
        elements * (sizeof(std::uint16_t) + sizeof(std::int8_t)), launch);
    verify_all({&data, &bf16, &int8});
}

#endif // TQ_HAS_E8_KERNELS

void lm4_ingress(const Session& session, const Filters& f, const Plan& plan)
{
    const auto heads = plan.p.kv_heads;
    const auto batch = plan.p.sequences;
    const auto tiles_per_page = plan.block / kTileRows;
    constexpr unsigned sampled_pages = 2;
    const auto pages = batch * sampled_pages;
    const auto slot_count = std::uint64_t{pages} * plan.block;
    const auto tokens = batch * (plan.block + (plan.p.ragged ? 47 : plan.block));
    const auto input_elements = std::uint64_t{tokens} * heads * kDim;
    const auto packed_bytes = slot_count * heads * kLm4RowBytes;
    const auto scale_words = slot_count * layout::NzScaleSlotFloats(heads);
    const auto pi = tq_golden::kernel_pi_signs(kDim);
    std::vector<std::uint16_t> key(input_elements), value(input_elements);
    std::vector<std::int32_t> slots(tokens), lengths(std::uint64_t{batch} + 1);
    const unsigned rows = tokens / batch;
    for (unsigned seq = 0; seq < batch; ++seq) {
        lengths[seq + 1] = static_cast<std::int32_t>((seq + 1) * rows);
        for (unsigned r = 0; r < rows; ++r) {
            const auto token = std::uint64_t{seq} * rows + r;
            const auto physical_page = pages - 1 - (seq * sampled_pages + r / plan.block);
            slots[token] = static_cast<std::int32_t>(std::uint64_t{physical_page} * plan.block + r % plan.block);
            for (unsigned h = 0; h < heads; ++h)
                for (unsigned c = 0; c < kDim; ++c) {
                    const auto index = (token * heads + h) * kDim + c;
                    const float k = static_cast<int>(sample(r, h, 0, c) % 23) - 11;
                    const float v = static_cast<int>(sample(r, h, 1, c) % 19) - 9;
                    key[index] = tq_test::float_to_bf16_rne(k);
                    value[index] = tq_test::float_to_bf16_rne(v);
                }
        }
    }
    Gm k(input_elements * sizeof(std::uint16_t)), v(input_elements * sizeof(std::uint16_t));
    Gm s(slots.size() * sizeof(std::int32_t)), signs(pi.size() * sizeof(float));
    Gm cu(std::uint64_t{ceil_div(batch + 1, 8)} * 8 * sizeof(std::int32_t));
    Gm kc(packed_bytes), vc(packed_bytes), sc(scale_words * sizeof(float));
    k.upload(key);
    v.upload(value);
    s.upload(slots);
    signs.upload(pi);
    cu.upload(lengths);
    constexpr unsigned sentinel = 0x55;
    kc.reset(sentinel);
    vc.reset(sentinel);
    sc.reset(sentinel);
    const auto launch = [&] {
        launch_turboquant_reshape_scatter_aiv(session.stream, std::min(batch * heads, kNpuGrid),
            k.as<std::uint16_t>(), v.as<std::uint16_t>(), kc.as<std::int8_t>(), vc.as<std::int8_t>(),
            sc.as<float>(), s.as<std::int32_t>(), signs.as<float>(), cu.as<std::int32_t>(),
            batch, heads, kDim, plan.block, pages, 0, batch * heads, kAttentionScale, 1);
    };
    std::printf(" ingress_tokens=%u pages_per_sequence=%u tiles_per_page=%u", tokens, sampled_pages, tiles_per_page);
    if (f.verify) {
        launch();
        session.synchronize();
        std::vector<std::uint8_t> actual_k(packed_bytes), actual_v(packed_bytes);
        std::vector<float> actual_s(scale_words);
        kc.download(actual_k);
        vc.download(actual_v);
        sc.download(actual_s);
        std::vector<std::uint8_t> expected_k(packed_bytes, sentinel), expected_v(packed_bytes, sentinel);
        std::vector<std::uint32_t> expected_s(scale_words, 0x55555555u);
        std::vector<float> row(kDim);
        tq_test::Shape shape;
        shape.head_size = kDim;
        for (unsigned t = 0; t < tokens; ++t)
            for (unsigned h = 0; h < heads; ++h)
                for (unsigned plane = 0; plane < 2; ++plane) {
                    const auto& input = plane ? value : key;
                    auto& expected = plane ? expected_v : expected_k;
                    for (unsigned c = 0; c < kDim; ++c)
                        row[c] = tq_test::bf16_to_float(input[(std::uint64_t{t} * heads + h) * kDim + c]);
                    const auto golden = tq_scatter::quantise_row_lloyd_max(shape, row, pi);
                    expected_s[layout::NzTiledScaleIndex(slots[t], h, plane, heads)] = tq_test::float_bits(golden.scale);
                    for (unsigned c = 0; c < kLm4RowBytes; ++c) {
                        const auto offset = layout::NzTiledPackedByte(slots[t], h, c, heads, kLm4RowBytes);
                        expected[offset] = static_cast<std::uint8_t>((golden.q[c] & 15) | ((golden.q[c + kLm4RowBytes] & 15) << 4));
                    }
                }
        require(actual_k == expected_k && actual_v == expected_v, "Lloyd-Max scatter codes or untouched cache changed");
        for (std::uint64_t i = 0; i < scale_words; ++i)
            require(tq_test::float_bits(actual_s[i]) == expected_s[i], "Lloyd-Max scatter scale or untouched lane changed");
        std::printf(" packed_byte_mismatches=0 scale_mismatches=0");
    } else {
        time_case(session, f, std::uint64_t{tokens} * heads *
            (2 * kDim * sizeof(std::uint16_t) + 2 * kLm4RowBytes + 2 * sizeof(float)), launch);
    }
    verify_all({&k, &v, &s, &signs, &cu, &kc, &vc, &sc});
}

}

int main(int argc, char** argv) try
{
    Filters f(argc, argv);
#ifdef TQ_PRODUCTION_ACCURACY
    f.verify = true;
#endif
    if (f.help) { f.usage(argv[0]); return 0; }
#ifdef TQ_PRODUCTION_SIM
    require(f.list || f.verify, "CAModel timing is not NPU performance; use --verify or the accuracy runner");
#endif
    std::unique_ptr<Session> session;
    unsigned executed = 0;
    f.visit([&](const Plan& plan, Mode mode, std::string_view stage) {
        print_case(plan, mode, stage);
        auto reason = plan.device_limit(mode, stage);
#ifndef TQ_HAS_E8_KERNELS
        if (mode == Mode::e8) reason = "E8 kernels disabled at build time";
#endif
        if (reason) {
            std::printf(" status=UNSUPPORTED reason=%s\n", reason);
            return;
        }
        if (f.list) {
            std::puts(" status=READY");
            return;
        }
        if (!session) session = std::make_unique<Session>(f.device);
        if (stage == "ingress") lm4_ingress(*session, f, plan);
        else if (mode == Mode::lm4) lm4_decode(*session, f, plan);
#ifdef TQ_HAS_E8_KERNELS
        else if (stage == "dequant") e8_dequant(*session, f, plan);
        else e8_decode_case(*session, f, plan);
#else
        else throw std::logic_error("unavailable codec reached device dispatch");
#endif
        ++executed;
        std::printf(" status=%s guards=PASS\n", f.verify ? "PASS" : "MEASURED");
        std::fflush(stdout);
    });
    require(f.list || executed > 0, "no physical cases matched the filters");
    return 0;
}
catch (const std::exception& error) {
    std::fprintf(stderr, "TurboQuant NPU case failed: %s\n", error.what());
    return 1;
}
