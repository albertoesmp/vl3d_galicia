# ---   CMAKE PYBIND11   --- #
# -------------------------- #

# Add PyBind11
find_package(Python3 COMPONENTS Development NumPy)
set(Pybind11_Python3_INCLUDE_DIRS ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
include_directories(${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
message("PyBind11_Python3_INCLUDE_DIRS: ${Pybind11_Python3_INCLUDE_DIRS}")
# Try to find PyBind11
find_package(pybind11 QUIET)

# If not found, add it as a subdirectory
if(NOT ${pybind11_FOUND})
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11)
endif()

# Prioritize custom PyBind11 install, if any
set(PyBind11_INSTALL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/pybind11/install/")
if(EXISTS "${PyBind11_INSTALL_DIR}")
    message("Found \"${PyBind11_INSTALL_DIR}\"")
    set(PyBind11_INSTALL_INCLUDE "${PyBind11_INSTALL_DIR}include")
    list(INSERT pybind11_INCLUDE_DIRS 0 "${PyBind11_INSTALL_INCLUDE}")
    set(pybind11_INCLUDE_DIR "${PyBind11_INSTALL_INCLUDE}")
    include_directories(${PyBind11_INSTALL_INCLUDE})
endif()

# Report
if(${pybind11_FOUND})
    message("PyBind11 FOUND!  :)")
else()
    message("PyBind11 NOT FOUND!  :(")
endif()
message("PyBind11_INCLUDE_DIRS: ${pybind11_INCLUDE_DIRS}")
message("PyBind11_INCLUDE_DIR: ${pybind11_INCLUDE_DIR}")
message("PyBind11_LIBRARIES: ${pybind11_LIBRARIES}")
