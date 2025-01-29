#pragma once

// ***   INCLUDES   *** //
// ******************** //
#include <cmath>

namespace vl3dpp { namespace util {

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Header-only class providing util functions to work with multiple
 *  threads.
 */
class MultithreadingUtils {
public:
    // ***   STATIC METHODS  *** //
    // ************************* //
    /**
     * @brief Find the correct chunk size for a given parallel computation.
     *
     * If the number of chunks \f$\left\lceil\frac{m}{n}\right\rceil\f$ is
     * greater than or equal to the number of available threads, then the
     * given chunk size is already correct, i.e, \f$n' = n\f$. Otherwise, the
     * correct chunk size is said to be:
     *
     * \f[
     *  \left\lceil\frac{m}{t}\right\rceil
     * \f]
     *
     * @param nElems The total number of elements \f$m\f$.
     * @param chunkSize The original number of elements per chunk \f$n\f$.
     * @param nthreads The number of available threads \f$t\f$.
     * @return The corrected chunk size \f$n'\f$.
     */
    static int correctChunkSize(
        int const nElems,
        int const chunkSize,
        int const nthreads
    ){
        int const nChunks = std::ceil(nElems / chunkSize);
        if(nChunks < nthreads){
            return std::ceil((double)nElems / (double)nthreads);
        }
        return chunkSize;
    }

    /**
     * @brief Compute the chunk size for a static workload distribution such
     *  that all the elements are distributed at once along the chunks.
     *
     * For example, let us say that we have 1000 elements and 10 threads.
     * That means that we will have 10 chunks of 100 elements each, i.e.,
     * chunk size 100. The formula to compute the chunk size is:
     *
     * \f[
     *  \left\lceil\frac{m}{t}\right\rceil
     * \f]
     *
     * @param nElems The number of elements \f$m\f$.
     * @param nthreads The number of available threads \f$t\f$.
     * @return The chunk size for a full static parallelization
     */
    static int computeFullStaticChunkSize(
        int const nElems,
        int const nthreads
    ){
        return (int) std::ceil((double)nElems / (double)nthreads);
    }


};

}}