// From: https://stackoverflow.com/questions/26003413/stdchrono-or-boostchrono-support-for-clock-monotonic-coarse
#ifndef Pds_FastMonotonicClock_hh
#define Pds_FastMonotonicClock_hh

#include <cassert>
#include <chrono>
#include <time.h>

#define MAYBE_UNUSED __attribute__((unused))


namespace Pds {
    class fast_monotonic_clock {
    public:
        using duration = std::chrono::nanoseconds;
        using rep = duration::rep;
        using period = duration::period;
        using time_point = std::chrono::time_point<fast_monotonic_clock>;

        static constexpr bool is_steady = true;

        static time_point now(clockid_t clkId = clock_id()) noexcept;

        static duration get_resolution(clockid_t clkId = clock_id()) noexcept;

    private:
        static clockid_t clock_id();
        static clockid_t test_coarse_clock();
        static duration convert(const timespec&);
    };
};

inline clockid_t Pds::fast_monotonic_clock::test_coarse_clock() {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC_COARSE, &t) == 0) {
        return CLOCK_MONOTONIC_COARSE;
    } else {
        return CLOCK_MONOTONIC;
    }
}

inline clockid_t Pds::fast_monotonic_clock::clock_id() {
    static clockid_t the_clock = test_coarse_clock();
    return the_clock;
}

inline auto Pds::fast_monotonic_clock::convert(const timespec& t) -> duration {
    return std::chrono::seconds(t.tv_sec) + std::chrono::nanoseconds(t.tv_nsec);
}

inline auto Pds::fast_monotonic_clock::now(clockid_t clkId) noexcept -> time_point {
    struct timespec t;
    MAYBE_UNUSED const auto result = clock_gettime(clkId, &t);
    assert(result == 0);
    return time_point{convert(t)};
}

inline auto Pds::fast_monotonic_clock::get_resolution(clockid_t clkId) noexcept -> duration {
    struct timespec t;
    MAYBE_UNUSED const auto result = clock_getres(clkId, &t);
    assert(result == 0);
    return convert(t);
}

#undef MAYBE_UNUSED

#endif
