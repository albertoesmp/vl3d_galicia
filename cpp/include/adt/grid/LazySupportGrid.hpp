#ifndef VL3DPP_SUPPORT_GRID_YIELDER_
#define VL3DPP_SUPPORT_GRID_YIELDER_


// ***   INCLUDES   *** //
// ******************** //
#include <armadillo>



namespace vl3dpp::adt::grid{

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Lazy support grid class.
 *
 * The lazy support grid is a deferred grid generator that can be used on a
 * thread-safe way to obtain the centroids of a grid. The dimensionality of the
 * grid is determined by the dimensionality of the given min and max points for
 * its construction.
 *
 * @tparam DecimalType The decimal type for the coordinates of the grid nodes.
 */
template <typename DecimalType=float>
class LazySupportGrid {
protected:
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief Half the length of each axis for each cell.
     */
    arma::Col<DecimalType> halfLength;
    /**
     * @brief The minimum point of the grid.
     */
    arma::Col<DecimalType> xmin;
    /**
     * @brief The maximum point of the grid.
     */
    arma::Col<DecimalType> xmax;

    // ***  ATTRIBUTES : STATE  *** //
    // **************************** //
    /**
     * @brief The number of cells along the rows parallel to each axis.
     *
     * Note that n[i] gives the number of cells along the row parallel to the
     * \f$i\f$-th axis.
     */
    arma::Col<arma::uword> n;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Build a LazySupportGrid from given extreme points.
     * @see LazySupportGrid::halfLength
     * @see LazySupportGrid::xmin
     * @see LazySupportGrid::xmax
     */
    LazySupportGrid(
        arma::Col<DecimalType> const &halfLength,
        arma::Col<DecimalType> const &xmin,
        arma::Col<DecimalType> const &xmax
    );
    /**
     * @brief Build a LazySupportGrid from the given structure space matrix.
     * @param halfLength
     * @param X The structure space matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times n_x}\f$ representing the position
     *  of \f$m > 0\f$ points in \f$n_x\f$-dimensional Euclidean space.
     * @see LazySupportGrid::LazySupportGrid(arma::Col<DecimalType> const &, arma::Col<DecimalType> const &, arma::Col<DecimalType> const &)
     */
    LazySupportGrid(
        arma::Col<DecimalType> const &halfLength,
        arma::Mat<DecimalType> const &X
    );
    /**
     * @brief Build a LazySupportGrid from a structure space matrix.
     * @see LazySupportGrid::LazySupportGrid(arma::Col<DecimalType> const &, arma::Col<DecimalType> const &, arma::Col<DecimalType> const &)
     */
    virtual ~LazySupportGrid() = default;


    // ***   GRID METHODS   *** //
    // ************************ //
    /**
     * @brief Obtain the centroid of the \f$\phi\f$-th cell.
     *
     * Note that when only a single index is given, it will be translated to
     * a vector with as many indices as the space dimensionality using an one
     * to one (injective) map.
     *
     * Let \f$\phi\f$ be the index and \f$n_k\f$ be the number of partitions
     * along the \f$k\f$-th axis. For then, the index for the first axis
     * will be given by \f$i_1 = \phi \bmod n_1\f$ and, subsequently, the
     * index for any axis \f$j > \phi\f$ will be given by:
     *
     * \f[
     *  i_j = \left\lfloor
     *      \frac{\phi}{\prod_{k=1}^{j-1}{n_k}}
     *  \right\rfloor \bmod n_j
     * \f]
     *
     * @param phi The single index identifying the cell.
     * @return The coordinates of the centroid at the \f$i\f$-th cell.
     * @see LazySupportGrid::getCentroid(arma::Col<arma::uword> const) const
     */
    arma::Col<DecimalType> getCentroid(size_t const phi) const;
    /**
     * @brief Obtain the centroid corresponding to the given axis-wise
     *  indices.
     *
     * For example, for a 3D grid with
     * \f$\vec{i} \in \mathbb{Z}^{3}_{\geq 0}\f$
     * assuming lexicographic order \f$x > y > z\f$ it will be that \f$i_1\f$
     * gives the index along the \f$x\f$-axis, \f$i_2\f$ gives the index along
     * the \f$y\f$-axis, and \f$i_3\f$ gives the index along the \f$z\f$-axis.
     *
     * The center of the cell represented by the given indices can be
     * calculated as follows
     *
     * \f[
     *  \pmb{x}_{*} + \left((2 \vec{i} + \vec{1}) \odot \vec{r} \right)
     * \f]
     *
     * where \f$\vec{i}\f$ is the vector of indices, \f$\vec{r}\f$ is the
     * vector with the axis-wise half lengths, \f$\pmb{x}_{*}\f$ is the min
     * point of the grid, and \f$\odot\f$ represents the Hadamard product.
     *
     * @param i The vector of axis-wise indices identifying the cell.
     * @return The coordinates of the centroid at the \f$i\f$-th cell.
     */
    arma::Col<DecimalType> getCentroid(arma::Col<arma::uword> const i) const;
    /**
     * @brief Obtain the total number of cells in the support grid.
     *
     * \f[
     *  \prod_{k} n_k
     * \f]
     *
     * Where \f$n_k\f$ is the number of partitions along the \f$k\f$-th axis.
     *
     * @return The number of cells in the support grid.
     */
    arma::uword getNumCells() const;
    /**
     * @brief Find the index of the cell corresponding to the given point.
     * @param x The point whose associated cell must be found.
     * @return The index of the cell that contains the given point.
     */
    arma::uword findCellIndex(arma::Col<DecimalType> const &x) const;
};

}

#include <adt/grid/LazySupportGrid.tpp>

#endif
