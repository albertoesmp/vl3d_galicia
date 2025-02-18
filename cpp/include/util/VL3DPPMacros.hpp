/**
 * @author Alberto M. Esmoris Pena
 * @brief Useful macros for the VL3D++ software.
 */

// Start if to prevent multiple inclusions
#ifndef VL3DPP_MACROS_HPP_
#define VL3DPP_MACROS_HPP_




// Macro to flag methods as used (even if they are not) with GCC and CLang
#if defined(__GNUC__) || defined(__clang__)
#define VL3DPP_USED_ __attribute__((used))
#else
#define VL3DPP_USED_
#endif


// Macro to govern the default scheduling policy for OpenMP
#define VL3DPP_OMP_SCHEDULE dynamic
// Macro to govern the default chunk sizes for OpenMP
#define VL3DPP_OMP_CHUNK_SIZE 256
#define VL3DPP_OMP_CHUNK_SIZE_SMALL 32
// Macro to govern the default scheduling+chunk size combination for OpenMP
#define VL3DPP_OMP_SCHEDULE_CHUNKED VL3DPP_OMP_SCHEDULE,VL3DPP_OMP_CHUNK_SIZE
#define VL3DPP_OMP_SCHEDULE_CHUNKED_SMALL VL3DPP_OMP_SCHEDULE,VL3DPP_OMP_CHUNK_SIZE_SMALL



// End if to prevent multiple inclusions
#endif