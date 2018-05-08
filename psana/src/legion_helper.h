#ifndef LEGION_HELPER_H
#define LEGION_HELPER_H

#ifndef PSANA_USE_LEGION
#error legion_helper.h requires PSANA_USE_LEGION
#endif

#include <stddef.h>

#include <vector>

#include <legion.h>

// Helper for creating a logical region
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

    template <class C, typename ... Ts>
    friend class LegionTask;
};

// Helper for creating a task
template <class C, typename ... Ts>
class LegionTask {
public:
    LegionTask();
    LegionTask(Ts ... ts);
    void add_array(const LegionArray &array);
    void launch();

protected:
    std::vector<LegionArray> arrays; // valid inside run()

private:
    Legion::TaskLauncher launcher;

protected:
    static Legion::TaskID register_task(const char *task_name);
public:
    static void task_wrapper(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *runtime);
};

#include "legion_helper.inl"

#endif
