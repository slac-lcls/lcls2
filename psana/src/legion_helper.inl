enum FieldIDs {
  FID_X = 101,
};

template <typename ... Ts>
struct phantom
{
};

template<typename T1>
size_t get_serialized_size(phantom<T1> t1)
{
    return sizeof(T1);
}

template<typename T1, typename ... Ts>
size_t get_serialized_size(phantom<T1, Ts ...> ts)
{
    return sizeof(T1) + get_serialized_size(phantom<Ts ...>());
}

template<typename ... Ts>
size_t get_serialized_size()
{
    return get_serialized_size(phantom<Ts ...>());
}

template<typename T1>
void serialize(char *buffer, T1 t1)
{
    *((T1 *)buffer) = t1;
}

template<typename T1, typename ... Ts>
void serialize(char *buffer, T1 t1, Ts ... ts)
{
    *((T1 *)buffer) = t1;
    buffer += sizeof(T1);
    serialize(buffer, ts ...);
}

template<class C, typename ... Us>
void deserialize_and_run_(const char *buffer, phantom<>, Us ... us)
{
    C c;
    c.run(us ...);
}

template<class C, typename T1, typename ... Ts, typename ... Us>
void deserialize_and_run_(const char *buffer, phantom<T1, Ts ...>, Us ... us)
{
    T1 t1 = *((T1 *)buffer);
    buffer += sizeof(T1);
    deserialize_and_run_<C>(buffer, phantom<Ts ...>(), us ..., t1);
}

template<class C, typename T1, typename ... Ts>
void deserialize_and_run(const char *buffer)
{
    T1 t1 = *((T1 *)buffer);
    buffer += sizeof(T1);
    deserialize_and_run_<C>(buffer, phantom<Ts ...>(), t1);
}

template <class C, typename ... Ts>
LegionTask<C, Ts ...>::LegionTask()
{
}

template <class C, typename ... Ts>
LegionTask<C, Ts ...>::LegionTask(Ts ... ts)
{
    size_t buf_size = get_serialized_size<Ts ...>();
    char *buffer = (char *)malloc(buf_size);
    assert(buffer);
    serialize(buffer, ts ...);

    launcher = Legion::TaskLauncher(C::task_id, Legion::TaskArgument(buffer, buf_size));
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
    assert(task->arglen == get_serialize_size<Ts ...>());

    deserialize_and_run<C, Ts ...>((const char *)task->args);
}
