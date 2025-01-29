#ifndef VL3DPP_GRID_MESHER_
#define VL3DPP_GRID_MESHER_

// ***   INCLUDES   *** //
// ******************** //
#include <util/VL3DPPException.hpp>

#include <armadillo>


namespace vl3dpp::alg{


// ***   CLASS   *** //
// ***************** //
/**
 * @author Alberto M. Esmoris Pena
 * @brief Class to generate grids of linearly spaced nodes in spaces of
 *  arbitrary dimensionality. It supports different node spacing on each axis.
 * @tparam XDecimalType The type for the decimal numbers representing the
 *  coordinates of the points.
 */
template <typename XDecimalType>
class GridMesher{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The axis-wise length of each cell
     *  \f$\pmb{l} \in \mathbb{R}^{n_x}\f$.
     *
     * More concretely, \f$l_k\f$ gives the separation between nodes along the
     *  \f$k\f$-axis.
     */
    arma::Col<XDecimalType> cellSize;
    /**
     * @brief Whether to add one extra node per axis to the grid (to extend
     *  the covered space) or not.
     */
    bool extraNodes;


public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Instantiate a GridMesher.
     * @see GridMesher::cellSize
     */
    GridMesher(arma::Col<XDecimalType> const &cellSize, bool const extraNodes);
    virtual ~GridMesher() = default;


    // ***   GRID METHODS  *** //
    // *********************** //
    /**
     * @brief Compute the nodes of the grid distributed along the axis-aligned
     *  bounding box of the given point cloud.
     * @param X The structure space representing a point cloud.
     * @return The point-wise (one row, one point) coordinates of the grid's
     *  nodes.
     * @see GridMesher::computeNodes
     */
    arma::Mat<XDecimalType> computeNodes(
        arma::Mat<XDecimalType> const &X
    ) const;
    /**
     * @brief Compute the nodes of the grid that has \f$\pmb{a}\f$ as min
     *  vertex and \f$\pmb{b}\f$ as max vertex.
     *
     * For example, any node in a \f$n_x\f$-dimensional grid can be defined as:
     *
     * \f[
     *  \pmb{g}_{i_1,\ldots,i_{n_x}} = \left[\begin{array}{c}
     *      \vdots \\
     *      a_k + i_k l_k \\
     *      \vdots
     *  \end{array}\right]
     * \f]
     *
     * Where \f$a_k\f$ is the \f$k\f$-th component of the min vertex,
     * \f$b_k\f$ is the \f$k\f$-th component of the max vertex, and
     * \f$l_k\f$ is the cell size along the \f$k\f$-th axis.
     *
     * @param a The min vertex of the grid.
     * @param b The max vertex of the grid.
     * @return The nodes of the requested grid.
     * @see GridMesher::cellSize
     */
    arma::Mat<XDecimalType> computeNodes(
        arma::Col<XDecimalType> const &a,
        arma::Col<XDecimalType> const &b
    ) const;
    /**
     * @brief Specialization of GridMesher::computeNodes for 2D grids.
     * @see GridMesher::computeNodes
     */
    arma::Mat<XDecimalType> computeNodes2D(
        arma::Col<XDecimalType> const &a,
        arma::Col<XDecimalType> const &b
    ) const;
    /**
     * @brief Specialization of GridMesher::computeNodes for 3D grids.
     * @see GridMesher::computeNodes
     */
    arma::Mat<XDecimalType> computeNodes3D(
        arma::Col<XDecimalType> const &a,
        arma::Col<XDecimalType> const &b
    ) const;
    /**
     * @brief General implementation of GridMesher::computeNodes for grids
     *  with an arbitrary dimensionality.
     * @see GridMesher::computeNodes
     */
    arma::Mat<XDecimalType> computeNodesND(
        arma::Col<XDecimalType> const &a,
        arma::Col<XDecimalType> const &b
    ) const;


};


#include <alg/GridMesher.tpp>


}

#endif
