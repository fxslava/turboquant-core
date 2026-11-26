#include <cstddef>
#include <cstdint>

extern "C" std::uint32_t __real_RegisterAscendBinary(const char*, std::size_t, std::uint32_t, void**);
extern "C" int UnregisterAscendBinary(void*);

namespace {
struct RegisteredBinary {
    void** slot;

    ~RegisteredBinary()
    {
        // The generated launch stub may already have released and cleared this slot.
        if (void* handle = *slot) {
            *slot = nullptr;
            UnregisterAscendBinary(handle);
        }
    }
};
}

extern "C" std::uint32_t __wrap_RegisterAscendBinary(
    const char* image, std::size_t bytes, std::uint32_t type, void** slot)
{
    const auto status = __real_RegisterAscendBinary(image, bytes, type, slot);
    if (status == 0 && slot != nullptr && *slot != nullptr) {
        static const RegisteredBinary registered{slot};
    }
    return status;
}
