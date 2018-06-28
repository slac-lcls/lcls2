#include "legion_helper.hh"

#include <legion.h>

using namespace Legion;

LegionArray::LegionArray()
{
}

LegionArray::LegionArray(size_t bytes)
{
    Runtime *runtime = Runtime::get_runtime();
    Context ctx = Runtime::get_context();

    IndexSpaceT<1> ispace = runtime->create_index_space(ctx, Rect<1>(0, bytes-1));
    FieldSpace fspace = runtime->create_field_space(ctx);
    FieldAllocator falloc = runtime->create_field_allocator(ctx, fspace);
    falloc.allocate_field(1, FID_X);
    region = runtime->create_logical_region(ctx, ispace, fspace);
}

LegionArray::~LegionArray()
{
    if (region != LogicalRegion::NO_REGION) {
        Runtime *runtime = Runtime::get_runtime();
        Context ctx = Runtime::get_context();

        if (physical.is_mapped()) {
            runtime->unmap_region(ctx, physical);
        }

        physical = PhysicalRegion();
        runtime->destroy_logical_region(ctx, region);
    }
}

LegionArray &LegionArray::operator=(LegionArray &&array)
{
    if (region != LogicalRegion::NO_REGION) {
        Runtime *runtime = Runtime::get_runtime();
        Context ctx = Runtime::get_context();

        if (physical.is_mapped()) {
            runtime->unmap_region(ctx, physical);
        }
        runtime->destroy_logical_region(ctx, region);
    }

    region = array.region;
    physical = array.physical;

    array.region = LogicalRegion::NO_REGION;
    array.physical = PhysicalRegion();
}

LegionArray::operator bool() const
{
    return region != LogicalRegion::NO_REGION;
}

char *LegionArray::get_pointer()
{
    assert(region != LogicalRegion::NO_REGION);

    if (!physical.is_mapped()) {
        Runtime *runtime = Runtime::get_runtime();
        Context ctx = Runtime::get_context();

        InlineLauncher launcher(RegionRequirement(region, READ_WRITE, EXCLUSIVE, region));
        launcher.add_field(FID_X);
        physical = runtime->map_region(ctx, launcher);
        physical.wait_until_valid();
    }

    UnsafeFieldAccessor<char,1,coord_t,Realm::AffineAccessor<char,1,coord_t> > acc(physical, FID_X);
    return acc.ptr(0);
}
