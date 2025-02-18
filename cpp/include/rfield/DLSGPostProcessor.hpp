#ifndef VL3DPP_DL_SG_POST_PROCESSOR_
#define VL3DPP_DL_SG_POST_PROCESSOR_

// ***   INCLUDES   *** //
// ******************** //
#include <adt/grid/SparseGrid.hpp>
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/MultithreadingUtils.hpp>

#include <thread>
#include <armadillo>
#include <omp.h>

#include <map>
#include <vector>
#include <string>
#include <sstream>

using vl3dpp::adt::grid::SparseGrid;
using vl3dpp::util::MultithreadingUtils;

using std::map;
using std::string;
using std::vector;

namespace vl3dpp::rfield {

// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 * @brief Class representing a sparse grid post-processor for deep learning
 *  models.
 *
 * NOTE that it can also be used for hierarchical sparse grids because its
 * post-processing is based on the first sparse grid in the hierarchy.
 *
 * @tparam XDecimalType The type of decimal number for the coordinates
 *  (structure space).
 * @tparam FDecimalType The type of decimal number for the features (typically
 *  cell-wise and point-wise probabilities).
 * @tparam IndexType The type of integer number used for cell and point-wise
 *  indexing.
 */
template <
    typename XDecimalType,
    typename FDecimalType,
    typename IndexType
>
class DLSGPostProcessor{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @see DLFPSPostProcessor::reductionType
     * @see DLSGPostProcessor::reduceFunction
     * TODO Rethink : Implement reduceFunction
     * TODO Rethink : Perhaps only mean will be supported and with different
     *  logic as to DLFPSPostProcessor
     */
    string reductionType;
    /**
     * @see DLFPSPostProcessor::minClipValue
     */
    FDecimalType minClipValue;
    /**
     * The cell size for each sparse grid.
     * @see adt::grid::SparseGrid::size
     */
    XDecimalType cellSize;
    /**
     * @brief The min vertex for each sparse grid.
     * @see adt::grid::SparseGrid::A
     */
    vector<arma::Row<XDecimalType>> const &A;
    /**
     * @brief The number of partitions along each axis for each sparse grid.
     * @see adt::grid::SparseGrid::n
     */
    vector<arma::Col<IndexType>> const &n;
    /**
     * @brief The submanifold map for aech sparse grid.
     * @see adt::grid::SparseGrid::h
     */
    vector<map<IndexType, IndexType>> const &h;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a hierarchical sparse grid (HSG) post-processor for deep
     *  learning.
     * @param[in] minClipValue Ignored for now, because only mean reduction is
     *  currently supported.
     * @see DLSGPostProcessor::cellSize
     * @see DLSGPostProcessor::A
     * @see DLSGPostProcessor::n
     * @see DLSGPostProcessor::h
     * @see DLSGPostProcessor::reductionType
     * @see DLSGPostProcessor::nthreads
     */
    DLSGPostProcessor(
        XDecimalType const cellSize,
        vector<arma::Row<XDecimalType>> const &A,
        vector<arma::Col<IndexType>> const &n,
        vector<map<IndexType, IndexType>> const &h,
        string const &reductionType,
        FDecimalType const minClipValue,
        int const nthreads
    );
    virtual ~DLSGPostProcessor() = default;

    // ***  POST-PROCESSING METHODS  *** //
    // ********************************* //
    /**
     * @brief Post-process the given atch of probabilities in the receptive
     *  fields back to the \f$m \in \mathbb{Z}_{>0}\f$ points in the original
     *  space.
     *
     * TODO Rethink : If only mean is supported, comment it here
     *
     * @param X The structure space of the original input point cloud.
     * @param zBatch The batch representing the probabilities in the receptive
     *  fields.
     * @return The point-wise probabilities in the original space, i.e., outside
     *  the receptive fields.
     */
    arma::Mat<FDecimalType> operator()(
        arma::Mat<XDecimalType> const &X,
        std::vector<arma::Mat<FDecimalType>> const &zBatch
    ) const;
};

#include <rfield/DLSGPostProcessor.tpp>

}

#endif
