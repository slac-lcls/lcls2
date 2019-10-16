add_library(utils SHARED
    src/Logger.cc
    src/DirFileIterator.cc
)

target_include_directories(utils PUBLIC
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

install(FILES
    Logger.hh
    Utils.hh
    MacTimeFix.hh
    DirFileIterator.hh
    ctest_utils.hh
    DESTINATION include/psalg/utils
)

install(TARGETS utils
    EXPORT psalgTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
