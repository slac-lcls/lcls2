add_library(utils SHARED
    src/Logger.cc
    src/DirFileIterator.cc
    src/trim.cc
)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../..>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    Logger.hh
    Utils.hh
    MacTimeFix.hh
    DirFileIterator.hh
    ctest_utils.hh
    SysLog.hh
    NDArrayGenerators.hh
    trim.hh
    DESTINATION include/psalg/utils
)

install(TARGETS utils
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
