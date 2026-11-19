#include "../common/camodel_session.hpp"

extern "C" void launch_qwen35_tail_canary(void*, void*);

int main()
{
    try {
        camodel::Session session;
        camodel::Buffer output(192 * sizeof(float));
        camodel::check(aclrtMemset(output.ptr, output.bytes, 0xff, output.bytes), "poison tail output");
        launch_qwen35_tail_canary(session.stream, output.ptr);
        camodel::check(aclrtSynchronizeStream(session.stream), "tail canary synchronize");
        std::vector<float> actual(192);
        output.download(actual);
        bool reproduced = false;
        for (std::size_t head = 0; head < 6; ++head) {
            const auto old = head * 16, fixed = 96 + old;
            reproduced |= actual[old] != 1.25f || actual[old + 8] != 64.0f;
            if (actual[fixed] != 1.25f || actual[fixed + 8] != 64.0f ||
                actual[fixed + 1] != 37.0f + static_cast<float>(head % 3) || actual[fixed + 2] != 2.0f)
                throw std::runtime_error("telemetry publication corrupted a fixed partial tail");
        }
        if (!reproduced) throw std::runtime_error("old predicate alias did not reproduce");
        std::puts("PASS tail canary: old alias corrupts mass; separate predicate preserves every max/sum");
        std::fflush(stdout);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL tail canary: %s\n", error.what());
        return 1;
    }
}