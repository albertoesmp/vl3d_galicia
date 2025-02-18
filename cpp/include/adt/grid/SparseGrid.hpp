#ifndef VL3DPP_SPARSE_GRID_
#define VL3DPP_SPARSE_GRID_


// ***   INCLUDES   *** //
// ******************** //
#include <util/VL3DPPException.hpp>
#include <util/VL3DPPMacros.hpp>
#include <util/TimeWatcher.hpp>
#include <util/logging/GlobalLogger.hpp>

#include <armadillo>
#include <omp.h>

#include <map>
#include <string>
#include <vector>
#include <thread>
#include <sstream>
#include <cmath>
#include <algorithm>


namespace vl3dpp::adt::grid{

using vl3dpp::util::TimeWatcher;
using std::map;
using std::string;
using std::vector;

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Sparse grid.
 *
 * The sparse grid is made of regular voxels, i.e., each voxel has a given size
 * \f$l \in \mathbb{R}_{>0}\f$ that is the same for each side/edge. Thus, the
 * volume of each voxel is \f$$l^{n_x}\f$, where \f$n_x\f$ is the
 * dimensionality of the structure space. The sparse grid only represents in
 * memory active cells/voxels, i.e., those that have at least one point inside
 * them.
 *
 * @tparam XDecimalType The type of decimal number for the coordinates (
 *  structure space).
 * @tparam IndexType The type of integer number used for cell and point-wise
 *  indexing.
 */
template <typename XDecimalType=float, typename IndexType=int>
class SparseGrid{
protected:
    // ***  ATTRIBUTES : SPECIFICATION  *** //
    // ************************************ //
    /**
     * @brief The edge length for each voxel (\f$l \in \mathbb{R}_{>0}\f$).
     */
    XDecimalType size;
    /**
     * @brief The number of padding cells \f$n_p\f$. For each axis (i.e.,
     *  dimension) \f$n_p\f$ extra partitions will be added before the first
     *  partition and also \f$n_p\f$ extra partitions after the last one.
     *
     * For example, a 3D grid with \f$20 \times 20 \times 5\f$ cells with
     * \f$n_p = 2\f$ will become a grid with \f$24 \times 24 \times 9\f$ cells.
     */
    IndexType paddingCells;
    /**
     * @brief How many threads must be used.
     *
     * If 0, then the number of threads depends on the current configuration.
     * If -1, then all available threads will be used.
     */
    int nthreads;
    /**
     * @brief Whether to write/print the execution time of the fit method
     *  when building a hierarchical sparse grid (true) or not (false, default).
     */
    bool logTime;

    // ***  ATTRIBUTES : STATE  *** //
    // **************************** //
    /**
     * @brief Key-value map (to be used as static hash table) that maps the
     *  cell index (key) to the corresponding active cell index (value).
     *
     * Note that cell indices will be sparse as they represent only the active
     * cells in the grid. However, point indices must be dense and sequential
     * because they represent points as rows in matrices. In general, this
     * map can be seen as encoding a function between a finite domain and a
     * finite image such that \f$h(g_i) = i\f$, where
     * \f$g_i \in \mathbb{Z}_{\geq0}\f$ gives the index of the \f$i\f$-th active
     * cell.
     *
     * Iterating over the keys of \f$h\f$ gives the indices of all the active
     * cells in the sparse grid.
     */
    map<IndexType, IndexType> h;
    /**
     * @brief The min point of the grid \f$A \in \mathbb{R}^{n_x}\f$.
     */
    arma::Row<XDecimalType> A;
    /**
     * @brief The number of partitions along each axis
     *  \f$\pmb{n} \in \mathbb{Z}_{\geq 0}^{n_x}\f$.
     *
     * Note that \f$n_d\f$ gives the number of partitions along the \f$d\f$-th
     *  axis (also can be read as \f$d\f$-th dimension). For example, for
     *  \f$n_x = 3\f$, \f$n_1\f$ gives the number of partitions along the
     *  \f$x\f$-axis, \f$n_2\f$ for the \f$y\f$-axis, and \f$n_3\f$ for the
     *  \f$z\f$-axis.
     */
    arma::Col<IndexType> n;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Initialize a SparseGrid. Note that this constructs the object, but
     *  it does not generate the grid. Once a SparseGrid is built, it must be
     *  fit to a structure space to represent it by calling the
     *  SparseGrid::fit method.
     * @see SparseGrid::fit
     */
    SparseGrid(
        XDecimalType const size,
        IndexType const paddingCells,
        int const nthreads
    );
    /**
     * @brief Alternative initialization for SparseGrid.
     *
     * This constructor is
     * designed to build a SparseGrid that supports calling the
     * SparseGrid::indexFromCoordinates method. Thus, it only contains the
     * data that is needed for that operation. It is useful, for example, for
     * post-processing tasks (see vl3dpp::rfield::DLSGPostProcessor for an
     * example).
     *
     * @see SparseGrid::size
     * @see SparseGrid::A
     * @see SparseGrid::n
     * @see rfield::vl3dpp::rfield::DLSGPostProcessor
     */
    SparseGrid(
        XDecimalType const size,
        arma::Row<XDecimalType> const &A,
        arma::Col<IndexType> const &n
    );
    /**
     * @brief Alternative initialization for SparseGrid.
     *
     * This constructor is designed to build a SparseGrid that supports
     * encode/reduce operations. This, it only contains the data that is needed
     * for those operations. It is useful, for example, for receptive
     * field-based reductions (e.g., to see how the reference labels are
     * distributed on a particular receptive field).
     *
     * @param hk The vector of keys to build the map SparseGrid::h
     * @param hv The vector of values to build the map SparseGrid::v
     *
     * @see SparseGrid::size
     * @see SparseGrid::A
     * @see SparseGrid::n
     * @see SparseGrid::hk
     * @see SparseGird::hv
     * @see SparseGrid::encodeMatrix
     * @see SparseGrid::encodeVector
     */
    SparseGrid(
        XDecimalType const size,
        arma::Row<XDecimalType> const &A,
        arma::Col<IndexType> const &n,
        arma::Col<IndexType> const &hk,
        arma::Col<IndexType> const &hv
    );
    virtual ~SparseGrid() = default;


    // ***  MAIN METHODS  *** //
    // ********************** //
    /**
     * @brief Fit the sparse grid to the given structure space.
     *
     * After calling SparseGrid::fit the SparseGrid::h map will represent a
     *  transformation between the indices of the active cells and their
     *  corresponding sequential point-wise indices.
     *
     * @param[in] X The structure space (i.e., matrix of point-wise coordinates)
     *  that must be represented by the sparse grid.
     * @see SparseGrid::h
     */
    void fit(arma::Mat<XDecimalType> const &X);

    /**
     * @brief Encode the given point-wise features.
     *
     * Encode the feature space \f$\pmb{F} \in \mathbb{R}^{m \times n_f}\f$
     * into the grid. The features on the \f$R\f$ active cells will be then
     * represented by a matrix \f$\pmb{Y} \in \mathbb{R}^{R \times n_f}\f$
     * where \f$h(g_i) = i\f$ means the vector of features representing the
     * active cell with index \f$g_i\f$ is given by the \f$i\f$-th row of
     * \f$\pmb{Y}\f$, i.e, \f$\pmb{y}_{i*}\f$.
     *
     * @tparam FType The numeric type used to represent the features.
     * @param X The structure space representing the points. NOTE that X
     *  must be the same matrix that was used to fit the sparse grid.
     *  TODO Rethink : Does X really need to be the same as for fit?
     * @param F The point-wise features, the \f$i\f$-th row \f$\pmb{f}_{i*}\f$
     *  gives the features vector representing the \f$i\f$-th point.
     * @param reduceStrategy The strategy to be used to reduce a feature for a
     *  given cell when there are two or more points inside that cell.
     *  Supported strategies are:
     *  <ul>
     *      <li>mean</li>
     *      <li>max</li>
     *      <li>min</li>
     *      <li>mode</li>
     *  </ul>
     * @return The encoded features matrix.
     * @see SparseGrid::reduceMean
     * @see SparseGrid::reduceMax
     * @see SparseGrid::reduceMin
     * @see SparseGrid::reduceMode
     * @see SparseGrid::encodeVector
     */
    template <typename FType>
    arma::Mat<FType> encodeMatrix(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<FType> const &F,
        string const &reduceStrategy
    );
    /**
     * @brief Encode the given point-wise features similar to
     *  SparseGrid::encodeMatrix but considering a vector instead of a matrix.
     * @see SparseGrid::encodeMatrix
     */
    template <typename FType>
    arma::Col<FType> encodeVector(
        arma::Mat<XDecimalType> const &X,
        arma::Col<FType> const &F,
        string const &reduceStrategy
    );

    // ***  GETTERs and SETTERs  *** //
    // ***************************** //
    /**
     * @brief Obtain the number of cells in the grid.
     * @return Number of cells in the grid.
     */
    inline IndexType getNumCells() const
    {return arma::prod(n);}

    /**
     * @brief Obtain the number of active cells in the grid (i.e, non-empty).
     * @return Number of active cells in the grid.
     */
    inline IndexType getNumActiveCells() const
    {return h.size();}

    /**
     * @brief Obtain the number of padding cells (the same for each
     *  axis/dimension).
     * @return Number of padding cells.
     * @see SparseGrid::paddingCells
     */
    inline IndexType getPaddingCells() const
    {return paddingCells;}
    /**
     * @brief Set the number of padding cells (the same for each
     *  axis/dimension).
     *
     * Note that updating the number of padding cells will only lead to a
     *  sparse grid with such padding after calling the fit method.
     *
     * @param paddingCells The new number of padding cells.
     * @return Number of padding cells.
     * @see SparseGrid::paddingCells.
     * @see SparseGrid::fit
     */
    inline void setPaddingCells(IndexType const paddingCells)
    {this->paddingCells = paddingCells;}
    /**
     * @brief Obtain the number of partitions along the given axis.
     * @param dim The dimension/axis whose number of partitions must be
     *  obtained.
     * @return Number of partitions along given axis.
     * @see SparseGrid::n
     */
    inline IndexType getNumAxisPartitions(arma::uword const dim) const
    {return n[dim];}
    /**
     * @brief Obtain a read-only (const) reference to the vector that encodes
     *  the number of partitions per axis.
     * @return Read-only (const) reference to the number of partitions per
     *  axis.
     */
    inline arma::Col<IndexType> const & getNumAxisPartitions() const
    {return n;}
    /**
     * @brief Obtain a reference to the map that encodes the active cells.
     * @return Reference to the map that encodes the active cells.
     * @see SparseGrid::h
     */
    inline map<IndexType, IndexType> & getMap()
    {return h;}
    /**
     * @brief Obtain the cell size.
     * @return The cell size.
     * @see SparseGrid::size
     */
    inline XDecimalType getCellSize() const
    {return size;}
    /**
     * @brief Obtain the number of max threads for parallel executions.
     * @return Number of max threads for parallel executions.
     * @see SparseGrid::nthreads
     */
    inline int getNumThreads() const
    {return nthreads;}
    /**
     * @brief Obtain the min vertex/point of the sparse grid.
     * @return Min vertex of the sparse grid.
     * @see SparseGrid::A
     */
    inline arma::Row<XDecimalType> getMinVertex() const
    {return A;}
    /**
     * @brief Check whether the execution time of the
     *  SparseGrid::fit method must be logged or not.
     * @return True if the execution time of SparseGrid::fit must
     *  be logged, false otherwise.
     * @see SparseGrid::logTime
     */
    inline bool getLogTime() const
    {return logTime;}
    /**
     * @brief Set whether the execution time of the
     *  SparseGrid::fit method must be logged or not.
     * @param logTime True to enable the logging of the execution time,
     *  false otherwise.
     * @see SparseGrid::logTime
     */
    inline void setLogTime(bool const logTime)
    {this->logTime = logTime;}

protected:
    // ***  REDUCE METHODS  *** //
    // ************************ //
    /**
     * @brief Reduce the given vector of values to a single one obtained by
     *  computing the mean.
     * @param[in] f The input values (features).
     * @return The mean of the input values.
     * @see SparseGrid::encodeMatrix
     */
    template <typename FType>
    static FType reduceMean(vector<FType> const &f);
    /**
     * @brief Reduce the given vector of values to a single one obtained by
     *  computing the max.
     * @param[in] f The input values (features).
     * @return The max of the input values.
     * @see SparseGrid::encodeMatrix
     */
    template <typename FType>
    static FType reduceMax(vector<FType> const &f);
    /**
     * @brief Reduce the given vector of values to a single one obtained by
     *  computing the min.
     * @param[in] f The input values (features).
     * @return The min of the input values.
     * @see SparseGrid::encodeMatrix
     */
    template <typename FType>
    static FType reduceMin(vector<FType> const &f);
    /**
     * @brief Reduce the given vector of values to a single one obtained by
     *  computing the mode.
     * @param[in] f The input values (features, typically integer labels).
     * @return The mode of the input values.
     * @see SparseGrid::encodeMatrix
     */
    template <typename FType>
    static FType reduceMode(vector<FType> const &f);

    // ***  UTIL METHODS  *** //
    // ********************** //
public:
    /**
     * @brief Obtain the index of the cell to which the given point belongs.
     *
     * For a given point \f$\pmb{x} \in \mathbb{R}^{n_x}\f$ on an
     * \f$n_x\f$-dimensional space, its cell index \f$g\f$ can be computed as:
     *
     * \f[
     *  g = \sum_{d=1}^{n_x} \min \; \left\{
     *      n_d - 1,\,
     *      \left\lfloor\frac{x_d-A_d}{l}\right\rfloor
     *  \right\} \prod_{k=d+1}^{n_x}{n_k}
     * \f]
     *
     * Where \f$A_d\f$ is the \f$d\f$-th component of the grid's min vertex,
     * \f$l \in \mathbb{R}_{>0}\f$ is the edge length for each voxel, and
     * \f$x_{d}\f$ is the \f$d\f$-th component of the input point.
     *
     * @param xi The point whose corresponding cell index must be computed.
     * @return The cell index for the given point.
     */
    IndexType indexFromCoordinates(arma::Row<XDecimalType> const &xi);
protected:
    /**
     * @brief Common logic for the SparseGird::encodeMatrix and
     *  SparseGrid::encodeVector methods.
     *
     * The common logic of preparing an encoding operation includes determining
     *  the reduce function, and finding the indices of the points inside each
     *  cell (i.e., the point-wise neighborhoods of the active cells).
     *
     * @tparam FType See SparseGrid::encodeMatrix
     * @param[in] X See SparseGrid::encodeMatrix
     * @param[in] F See SparseGrid::encodeMatrix
     * @param[in] reduceStrategy See SparseGrid::encodeMatrix
     * @param[out] reduce The pointer to the reduce function will be passed
     *  as output through this argument.
     * @return The indices of the points in active cells such that I[i] gives
     *  the indices of the points in the i-th active cell of the grid.
     * @see SparseGrid::encodeMatrix
     * @see SparseGrid::encodeVector
     */
    template <typename FType>
    vector<vector<IndexType>> prepareEncoding(
        arma::Mat<XDecimalType> const &X,
        arma::Mat<FType> const &F,
        string const &reduceStrategy,
        FType (**reduce) (vector<FType> const &f)
    );
public:
    /**
     * @brief Report the configuration of the sparse grid through the given
     *  output stream.
     * @param out The output stream where the report must be outputted.
     */
    void report(std::ostream & out);
};


#include <adt/grid/SparseGrid.tpp>


}

#endif
