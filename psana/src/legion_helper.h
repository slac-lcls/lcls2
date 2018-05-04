#ifndef LEGION_HELPER_H
#define LEGION_HELPER_H

#ifndef PSANA_USE_LEGION
#error legion_helper.h requires PSANA_USE_LEGION
#endif

#include <stddef.h>

#include <legion.h>

class LegionArray {
public:
    LegionArray();
    LegionArray(size_t bytes);
    LegionArray(const LegionArray &array) = delete; // not supported
    ~LegionArray();

    LegionArray &operator=(LegionArray &&array); // consumes array

    operator bool() const;

    char *get_pointer();

private:
    Legion::LogicalRegionT<1> region;
    Legion::PhysicalRegion physical;
};

#endif
