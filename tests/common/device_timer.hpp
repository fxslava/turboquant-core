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

#pragma once
// DEVICE-SIDE kernel timing for the `--perf` mode of the runners in tests/npu.
//
// WHAT IT MEASURES, AND WHY THAT SHAPE
//
// One sample is the interval between two ACL_EVENT_TIME_LINE events RECORDED ON THE STREAM
// either side of one launch. The timestamps are taken by the device as the stream retires each
// marker, so a sample is the kernel's own residency plus the stream's dispatch of it -- not a
// host round trip, and not a number the host's scheduler can inflate.
//
// The timed loop issues every iteration back to back and synchronises ONCE, at the end, rather
// than per iteration. Stream order already serialises the launches, so the per-launch intervals
// are unaffected, and not returning to the host between them is what makes the samples a steady
// state rather than a sequence of cold starts with a host bubble in front of each. That is also
// why the events are allocated up front, one pair per iteration: reading an elapsed time is a
// host call, and doing it inside the loop would put the bubble back.
//
// WARM-UP IS NOT OPTIONAL. The first launch of a kernel pays for its binary registration and
// for whatever the allocator and the L2 did not have resident, and on this operator the first
// sample runs several times the steady one. Warm-up launches are issued and discarded before the
// first event is recorded.
//
// WHEN THE DEVICE CANNOT TIME ITSELF -- a CAModel driver that does not implement timeline events
// -- this falls back to the host clock around the whole batch and SAYS SO, in the struct
// (`device_timed == false`) and in every line it prints. A host-clocked number includes the
// launch path and is not comparable with a device-clocked one; the two must never be read as the
// same measurement, which is the whole reason the flag is carried rather than inferred.
//
// WHAT IT IS NOT. A sample is a wall interval, so it says how long the launch took and nothing
// about WHY: no per-pipe occupancy, no MTE2/Cube overlap, no bank-conflict census. The seamless
// chunk pipeline's overlap claim needs a trace with intersecting intervals and is not something
// this file can establish -- see "What is verified, and what is not" in README.md. Use msprof
// for that; use this to tell whether a change moved the number at all.

#include <acl/acl.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace camodel {

// Timeline events are not reusable across recordings without a reset, so the loop allocates one
// pair per iteration. This caps that allocation; a measurement needs tens of samples, not
// thousands, and a runner that asks for more is told the number it will actually get.
constexpr int kMaxPerfIters = 512;
constexpr int kDefaultPerfIters = 20;
constexpr int kDefaultPerfWarmup = 3;

struct PerfStats {
    int samples = 0;
    double min_ms = 0.0;
    double median_ms = 0.0;
    double mean_ms = 0.0;
    double max_ms = 0.0;
    // True when every sample came from a pair of ACL_EVENT_TIME_LINE events on the stream.
    // False when the batch was timed on the host clock instead, which includes the launch path.
    bool device_timed = false;

    double min_us() const { return min_ms * 1000.0; }
    double median_us() const { return median_ms * 1000.0; }
    double mean_us() const { return mean_ms * 1000.0; }
    double max_us() const { return max_ms * 1000.0; }
    const char* clock_name() const { return device_timed ? "device" : "host"; }
};

namespace detail {

inline PerfStats summarise(std::vector<double>& samples, bool device_timed)
{
    PerfStats stats;
    stats.device_timed = device_timed;
    if (samples.empty()) return stats;
    std::sort(samples.begin(), samples.end());
    stats.samples = static_cast<int>(samples.size());
    stats.min_ms = samples.front();
    stats.max_ms = samples.back();
    const std::size_t mid = samples.size() / 2;
    stats.median_ms = (samples.size() % 2 == 1) ? samples[mid]
                                                : 0.5 * (samples[mid - 1] + samples[mid]);
    double total = 0.0;
    for (const double sample : samples) total += sample;
    stats.mean_ms = total / static_cast<double>(samples.size());
    return stats;
}

// Every timeline event this batch needs, or an empty vector if the driver will not give them.
// All-or-nothing on purpose: a partial set would silently mix device-clocked and host-clocked
// samples into one distribution.
inline std::vector<aclrtEvent> create_timeline_events(int count)
{
    std::vector<aclrtEvent> events;
    events.reserve(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
        aclrtEvent event = nullptr;
        if (aclrtCreateEventWithFlag(&event, ACL_EVENT_TIME_LINE) != ACL_SUCCESS) {
            for (aclrtEvent made : events) aclrtDestroyEvent(made);
            return {};
        }
        events.push_back(event);
    }
    return events;
}

inline void destroy_events(std::vector<aclrtEvent>& events)
{
    for (aclrtEvent event : events) aclrtDestroyEvent(event);
    events.clear();
}

// The host-clock fallback: one interval around the whole batch, divided by the iteration count.
// There is no distribution to report -- min, median and max are all the mean -- because the
// per-iteration boundaries are exactly what the host cannot see without a sync per iteration,
// and a sync per iteration would be measuring the sync.
template <class Launch>
PerfStats measure_on_host(aclrtStream stream, int warmup, int iters, Launch&& launch)
{
    for (int i = 0; i < warmup; ++i) launch();
    check(aclrtSynchronizeStream(stream), "perf warm-up synchronize");

    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) launch();
    check(aclrtSynchronizeStream(stream), "perf batch synchronize");
    const auto end = std::chrono::steady_clock::now();

    const double batch_ms = std::chrono::duration<double, std::milli>(end - begin).count();
    const double each = batch_ms / static_cast<double>(iters);
    std::vector<double> samples(static_cast<std::size_t>(iters), each);
    return summarise(samples, false);
}

}  // namespace detail

// Time `iters` launches of `launch` on `stream`, after `warmup` discarded ones.
//
// `launch` takes no arguments and must enqueue exactly one kernel launch on `stream` and nothing
// else -- no upload, no download, no synchronise. Anything else it does lands inside the
// measured interval.
//
// Every launch is the SAME launch, with the same inputs and the same output buffers, so a
// measured operator must be idempotent in its own arguments: all three TurboQuant operators are
// (both writers recompute the identical cache bytes, and the decode rewrites its output and its
// Split-K workspace from inputs it only reads). Call this AFTER the correctness and canary
// checks all the same, so a perf re-launch can never be what a validation pass observed.
template <class Launch>
PerfStats measure_launch(aclrtStream stream, int warmup, int iters, Launch&& launch)
{
    if (iters < 1) iters = 1;
    if (iters > kMaxPerfIters) iters = kMaxPerfIters;
    if (warmup < 0) warmup = 0;

    // Warm up and drain FIRST, so a launch failure is reported by `check` as itself rather than
    // leaking the event pairs on the way out.
    for (int i = 0; i < warmup; ++i) launch();
    check(aclrtSynchronizeStream(stream), "perf warm-up synchronize");

    std::vector<aclrtEvent> events = detail::create_timeline_events(2 * iters);
    if (events.empty()) return detail::measure_on_host(stream, 0, iters, launch);

    // Issue the whole batch without returning to the host: stream order serialises the launches,
    // so each pair of markers brackets one of them and nothing between them is host time.
    bool recorded = true;
    for (int i = 0; i < iters && recorded; ++i) {
        recorded = aclrtRecordEvent(events[2 * static_cast<std::size_t>(i)], stream) == ACL_SUCCESS;
        launch();
        recorded = recorded &&
                   aclrtRecordEvent(events[2 * static_cast<std::size_t>(i) + 1], stream) == ACL_SUCCESS;
    }
    const bool drained = aclrtSynchronizeStream(stream) == ACL_SUCCESS;

    std::vector<double> samples;
    if (recorded && drained) {
        samples.reserve(static_cast<std::size_t>(iters));
        for (int i = 0; i < iters; ++i) {
            float ms = 0.0f;
            if (aclrtEventElapsedTime(&ms, events[2 * static_cast<std::size_t>(i)],
                                      events[2 * static_cast<std::size_t>(i) + 1]) != ACL_SUCCESS) {
                samples.clear();
                break;
            }
            samples.push_back(static_cast<double>(ms));
        }
    }
    detail::destroy_events(events);

    // The launches already ran; only the clock failed. Re-measure on the host rather than report
    // a distribution assembled from whichever elapsed-time calls happened to succeed.
    if (samples.empty()) return detail::measure_on_host(stream, 0, iters, launch);
    return detail::summarise(samples, true);
}

// One line per measured configuration.
//
// `items` and `unit` are the operator's own work unit -- tokens for a writer, query vectors for
// the decode -- and `bytes` the traffic one launch moves, so the rate columns are in the terms
// the kernel is budgeted in rather than in milliseconds alone. Pass 0 for either to drop its
// column.
inline void print_perf_line(const char* label, const PerfStats& stats, double items,
                            const char* unit, double bytes)
{
    std::printf("  perf %-28s %7.1f us median  (min %7.1f  mean %7.1f  max %7.1f)  "
                "%d %s-clocked samples",
                label, stats.median_us(), stats.min_us(), stats.mean_us(), stats.max_us(),
                stats.samples, stats.clock_name());
    if (items > 0.0 && stats.median_ms > 0.0)
        std::printf("  %.3f M%s/s", items / stats.median_ms / 1.0e3, unit);
    if (bytes > 0.0 && stats.median_ms > 0.0)
        std::printf("  %.2f GB/s", bytes / stats.median_ms / 1.0e6);
    std::printf("\n");
}

// The one paragraph a perf run has to carry with its numbers, so a median cannot be quoted
// without what it is a median of.
inline void print_perf_preamble(int warmup, int iters, bool any_host_clocked)
{
    std::printf("PERF: %d warm-up launches discarded, then %d timed launches issued back to back "
                "with one synchronise at the end. A sample is the interval between two "
                "ACL_EVENT_TIME_LINE markers recorded on the stream either side of ONE launch, "
                "so it is the kernel's device residency plus the stream's dispatch of it.%s "
                "No per-pipe occupancy and no overlap claim: use msprof for those.\n",
                warmup, iters,
                any_host_clocked
                    ? " SOME CONFIGURATIONS FELL BACK TO THE HOST CLOCK (marked host-clocked "
                      "below): the driver would not give timeline events, and a host-clocked "
                      "number includes the launch path and is NOT comparable with a "
                      "device-clocked one."
                    : "");
}

}  // namespace camodel
