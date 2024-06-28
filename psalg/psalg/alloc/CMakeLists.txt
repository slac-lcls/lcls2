add_library(alloc INTERFACE
            AllocArray.hh
            Allocator.hh)

target_link_libraries(alloc INTERFACE
    xtcdata::xtc
)

########################################
# This command specifies the include directories that should be propagated to targets
# that link against alloc
# This is useful since we want to link psana to alloc

# INTERFACE just means the target has no source code but it does have headers and (possibly) links to libraries. Here we link to xtc

# $<INSTALL_INTERFACE:include --> This just tells CMake that headers provided by alloc "should be" installed into the include directory relative to CMAKE_INSTALL_PREFIX. "Should be" in the context of CMake's generator expressions ($<INSTALL_INTERFACE:include>) indicates where headers or files are expected to be placed during installation, not an immediate command for CMake to execute. 
target_include_directories(alloc INTERFACE
    $<INSTALL_INTERFACE:include>
)

########################################
# Here we are installing the targets. In this case alloc. 
# We are not installing the export set here. That is when we do 
# install(EXPORT psalgTargets
#    FILE psalgTargets.cmake
#    NAMESPACE psalg::
#    DESTINATION lib/cmake/psalg)

# We are installing the target alloc, stating that the target is part of the 
# psalgTargets export set that will be installed in another install(...).
# DESTINATION include/psalg/alloc: Specifies the destination directory where the 
# header files should be installed relative to CMAKE_INSTALL_PREFIX.
# This destination part may be redundant due to above.

install(TARGETS alloc
    EXPORT psalgTargets
    INCLUDES DESTINATION include/psalg/alloc
)

########################################
# install(FILES ...) 
#copies the actual header files to the installation directory specified 
#(include/psalg/alloc), ensuring they are available alongside the installed target
#(alloc).

install(FILES
        AllocArray.hh
        Allocator.hh
        DESTINATION include/psalg/alloc)