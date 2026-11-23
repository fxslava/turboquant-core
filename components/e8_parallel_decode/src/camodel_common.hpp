#pragma once

#include <acl/acl.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace camodel {
constexpr std::size_t kGuardBytes = 64;
constexpr std::uint32_t kCanary = 0xDEADBEEF;

inline void check(aclError result, const char* call)
{
    if (result != ACL_SUCCESS) {
        std::fprintf(stderr, "%s failed: %d\n", call, static_cast<int>(result));
        throw std::runtime_error(call);
    }
}

struct Session {
    aclrtStream stream = nullptr;
    Session()
    {
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(0), "aclrtSetDevice");
        check(aclrtCreateStream(&stream), "aclrtCreateStream");
    }
    ~Session()
    {
        if (stream) aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
    }
};

struct GuardedDeviceBuffer {
    std::vector<std::uint32_t> host;
    void* device = nullptr;
    std::size_t payload_bytes;
    explicit GuardedDeviceBuffer(std::size_t bytes)
        : host((bytes + 2 * kGuardBytes + 3) / 4, kCanary),
          payload_bytes(bytes)
    {
        if (bytes % 32 != 0) throw std::runtime_error("unaligned GM payload");
        check(aclrtMalloc(&device, host.size() * 4, ACL_MEM_MALLOC_NORMAL_ONLY),
              "aclrtMalloc");
        upload();
    }
    ~GuardedDeviceBuffer() { if (device) aclrtFree(device); }
    void* payload() const
    {
        return static_cast<std::uint8_t*>(device) + kGuardBytes;
    }
    void* host_payload() { return reinterpret_cast<std::uint8_t*>(host.data()) + kGuardBytes; }
    void upload()
    {
        check(aclrtMemcpy(device, host.size() * 4, host.data(), host.size() * 4,
                          ACL_MEMCPY_HOST_TO_DEVICE), "H2D guarded buffer");
    }
    void download()
    {
        check(aclrtMemcpy(host.data(), host.size() * 4, device, host.size() * 4,
                          ACL_MEMCPY_DEVICE_TO_HOST), "D2H guarded buffer");
    }
    void assert_guards() const
    {
        const std::size_t start = kGuardBytes / 4;
        const std::size_t end = (kGuardBytes + payload_bytes) / 4;
        for (std::size_t i = 0; i < start; ++i)
            if (host[i] != kCanary) throw std::runtime_error("head GM canary corrupted");
        for (std::size_t i = end; i < host.size(); ++i)
            if (host[i] != kCanary) throw std::runtime_error("tail GM canary corrupted");
    }
};
}  // namespace camodel
