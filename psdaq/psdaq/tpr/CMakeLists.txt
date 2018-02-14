add_library(tpr SHARED
    Module.cc
    Client.cc
)

target_link_libraries(tpr
  xtcdata
)

install(TARGETS tpr
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
