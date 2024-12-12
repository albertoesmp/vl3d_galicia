# ---   CMAKE BUILDING  --- #
# ------------------------- #
# Register vl3dpp as a library
ADD_LIBRARY(vl3dpp ${sources})

# List of all target libraries
set(
    VL3DPP_TARGET_LIBRARIES
    Python3::Python
    ${ARMADILLO_LIBRARIES}
    blas
    lapack
    Eigen3::Eigen
    ${PCL_LIBRARIES}
)
message("VL3D++ libraries: ${VL3DPP_TARGET_LIBRARIES}")

# LINK TARGET LIBRARIES
if(WIN32 OR MSVC)  # Windows compilation
    target_link_libraries(vl3dpp ${VL3DPP_TARGET_LIBRARIES})
else()  # Linux compilation
    target_link_libraries(vl3dpp ${VL3DPP_TARGET_LIBRARIES})
endif()

# BUILD PYBIND11 MODULES
pybind11_add_module(pyvl3dpp MODULE src/module/vl3dpp.cpp)
target_link_libraries(pyvl3dpp PRIVATE vl3dpp)
install(TARGETS pyvl3dpp DESTINATION .)
