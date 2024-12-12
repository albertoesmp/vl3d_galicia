# ---   CMAKE SOURCES   --- #
# ------------------------- #
# Register files
file(GLOB_RECURSE sources CONFIGURE_DEPENDS
    src/*.cpp
    src/*.tpp
    include/*.hpp
)
message("sources: ${sources}")

# VL3D++ include directories
set(VL3DPP_INCLUDE_DIRECTORIES
    # Header includes
    "include/"
    # Template-based includes (i.e., .tpp)
    "src/"
)

include_directories(${VL3DPP_INCLUDE_DIRECTORIES})