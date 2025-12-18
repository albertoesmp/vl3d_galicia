#ifndef VL3DPP_CUSTOM_OPENMP_
#define VL3DPP_CUSTOM_OPENMP_

// ***   INCLUDES   *** //
// ******************** //
#include <armadillo>

namespace vl3dpp::util{


// ***  CUSTOM ARGOPT REDUCTIONS FOR OPENMP  *** //
// ********************************************* //
/**
 * @brief Templated structure to represent the value and the index for argopt
 * like reductions in OpenMP (e.g., argmax).
 */
template <typename ValueType, typename IndexType>
struct VL3DPP_OMP_Opt {
    ValueType x; // The value
    IndexType i; // The index
};

// OpenMP argmax reduction
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<float, size_t> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<double, size_t> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<float, arma::uword> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<double, arma::uword> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<float, int> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmax : VL3DPP_OMP_Opt<double, int> : \
    omp_out = omp_out.x > omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))

// OpenMP argmin reduction
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<float, size_t> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<double, size_t> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<float, arma::uword> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<double, arma::uword> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<float, int> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))
#pragma omp declare reduction(argmin : VL3DPP_OMP_Opt<double, int> : \
    omp_out = omp_out.x < omp_in.x ? omp_out : omp_in \
) initializer(omp_priv=(omp_orig))


}


#endif
