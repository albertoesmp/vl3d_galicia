# ---   CMAKE EXECUTABLES   --- #
# ----------------------------- #
# EXECUTABLE: main_test
# Copy sources list
set(exe_sources ${sources})
# Remove sources for python modules
list(FILTER exe_sources EXCLUDE REGEX ".vl3d/cpp/src/module/.")
list(FILTER exe_sources EXCLUDE REGEX ".vl3d/cpp/include/module/.")
message("exe_sources: ${exe_sources}")
# Add main_test executable to run VL3D++ tests
add_executable(main_test ${exe_sources})
# Link target libraries for main_test executable
if(WIN32 OR MSVC)  # Windows compilation
    target_link_libraries(main_test ${VL3DPP_TARGET_LIBRARIES})
else()  # Linux compilation
    target_link_libraries(main_test ${VL3DPP_TARGET_LIBRARIES})
endif()
