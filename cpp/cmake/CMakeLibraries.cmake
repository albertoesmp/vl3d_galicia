# ---   CMAKE LIBRARIES   --- #
# --------------------------- #

# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
    message(
        "WARNING! VL3D++ BUILT WITHOUT OpenMP\n"
        "Consequently, a lot of shared memory parallelism will be missing. "
        "Suboptimal performance on many-cores CPUs can be expected."
    )
endif()




# Armadillo
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo)  # Use armadillo from lib
    # Include from lib directory
    #add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo)
    set(_ARMADILLO_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/armadillo/)
    message("Loading armadillo from \"${_ARMADILLO_LIB_DIR}\".")
    set(ARMADILLO_INCLUDE_DIRS ${_ARMADILLO_LIB_DIR}/include)
    set(ARMADILLO_LIBRARIES ${_ARMADILLO_LIB_DIR}libarmadillo.so)
else()  # Try to find already installed armadillo
    message("Loading armadillo from system.")
    find_package(Armadillo REQUIRED)
endif()

# Armadillo with custom LAPACK
if(LAPACK_LIB)
    # Add custom lapack library to armadillo if specified
    message("Armadillo is using custom LAPACK at \"${LAPACK_LIB}\"")
    set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARIES} ${LAPACK_LIB})
endif()

# Include directories and target link libraries
include_directories(${ARMADILLO_INCLUDE_DIRS})
link_directories(${ARMADILLO_LIBRARIES})

# Report included directories and libraries
message("Armadillo include: " ${ARMADILLO_INCLUDE_DIRS})
message("Armadillo libraries: " ${ARMADILLO_LIBRARIES})





# Carma
if(${CARMA_DEBUG})  # Enable CARMA debug if requested
    OPTION(ENABLE_CARMA_EXTRA_DEBUG "Enable CARMA_EXTRA_DEBUG" ON)
    OPTION(ENABLE_ARMA_EXTRA_DEBUG "Enable ARMA_EXTRA_DEBUG" ON)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma)  # Use carma from lib
    # Include from lib directory
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/)
    set(carma_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/include)
    set(carma_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/carma/include)
else()  # Try to find already installed Carma
    find_package(Carma CONFIG REQUIRED)
endif()

# Include directories
include_directories(${carma_INCLUDE_DIR})
include_directories(${carma_INCLUDE_DIRS})

# Report included directories and libraries
message("Carma found: " ${carma_FOUND})
message("Carma include dir: " ${carma_INCLUDE_DIR})
message("Carma include dirs: " ${carma_INCLUDE_DIRS})





# PCL (PointCloudLibrary)
set(PCL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib/pcl/")
message("PCL directory: ${PCL_DIR}")
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
if(EXISTS ${PCL_DIR}install/)
    # Include from from lib directory
    message("Loading PCL from \"${PCL_DIR}\".")
    set(PCL_INCLUDE_DIRS ${PCL_DIR}install/include/)
    file(GLOB PCL_LIBRARIES "${PCL_DIR}install/lib/*.so")
else()
    message("Loading PCL from system.")
    find_package(PCL 1.13 REQUIRED)
endif()
message("PCL include: ${PCL_INCLUDE_DIRS}")
message("PCL library dirs: ${PCL_LIBRARY_DIRS}")
message("PCL libraries: ${PCL_LIBRARIES}")
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

