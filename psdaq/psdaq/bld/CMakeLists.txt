set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_library(bld SHARED
    HpsEventIterator.cc
    Server.cc
    Client.cc
)

target_include_directories(bld PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(bld
    Threads::Threads
)

install(TARGETS bld
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
