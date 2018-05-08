enum FieldIDs {
  FID_X = 101,
};

template <class C, typename ... Ts>
LegionTask<C, Ts ...>::LegionTask()
{
    launcher = Legion::TaskLauncher(C::task_id, Legion::TaskArgument());
}

template <class C, typename ... Ts>
void LegionTask<C, Ts ...>::add_args(const void *args, size_t arglen)
{
    launcher.argument = Legion::TaskArgument(args, arglen);
}

template <class C, typename ... Ts>
void LegionTask<C, Ts ...>::add_array(const LegionArray &array)
{
    launcher.add_region_requirement(
        Legion::RegionRequirement(array.region, READ_WRITE, EXCLUSIVE, array.region))
        .add_field(FID_X);
}

template <class C, typename ... Ts>
void LegionTask<C, Ts ...>::launch()
{
    Legion::Runtime *runtime = Legion::Runtime::get_runtime();
    Legion::Context ctx = Legion::Runtime::get_context(); 

    runtime->execute_task(ctx, launcher);
}

template <class C, typename ... Ts>
Legion::TaskID LegionTask<C, Ts ...>::register_task(const char *task_name)
{
    Legion::Runtime *runtime = Legion::Runtime::get_runtime();

    // Get a task ID
    static Legion::TaskID first_id = 0;
    static Legion::TaskID next_id_offset = 0;
    if (!first_id) {
        first_id = runtime->generate_library_task_ids("psana", 1<<16);
    }
    Legion::TaskID task_id = first_id + next_id_offset++;

    // Register task
    Legion::TaskVariantRegistrar registrar(task_id, task_name, false /*global*/);
    registrar.add_constraint(Legion::ProcessorConstraint(Legion::Processor::IO_PROC));
    runtime->register_task_variant<C::task_wrapper>(registrar);
    runtime->attach_name(task_id, task_name);

    return task_id;
}

template <class C, typename ... Ts>
void LegionTask<C, Ts ...>::task_wrapper(const Legion::Task *task,
                                         const std::vector<Legion::PhysicalRegion> &regions,
                                         Legion::Context ctx, Legion::Runtime *runtime)
{
    assert(task->arglen == sizeof(C::Args));
    typename C::Args *args = (typename C::Args *)task->args;

    C c;
    c.run(*args);
}
