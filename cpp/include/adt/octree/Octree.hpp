#ifndef VL3DPP_OCTREE_
#define VL3DPP_OCTREE_


// ***   INCLUDES   *** //
// ******************** //
#include <util/VL3DPPMacros.hpp>
#include <armadillo>
#include <pcl/point_cloud.h>
#include <pcl/octree/octree_search.h>
#include <vector>

namespace vl3dpp::test{
    class OctreeTest;
}

namespace vl3dpp::adt::octree {

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief Octree class.
 *
 * Class that represents a typical octree. It can be used to significantly
 * speedup spatial queries.
 *
 * @tparam IndexType The type for the indices and also some integer values.
 * @tparam DecimalType The type for decimal numbers.
 */
template <typename IndexType=int, typename DecimalType=float>
class Octree {
protected:
    friend class vl3dpp::test::OctreeTest;
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The 3D point cloud on top of which the 3D octree is built.
     * @see Octree::octree3D
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D = nullptr;
    /**
     * @brief The 2D point cloud (3D point cloud with all points having the
     *  same \f$z\f$ coordinate) on top of which the 2D octree is built.
     * @see Octree::octree2D
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud2D = nullptr;
    /**
     * @brief The underlying octree for 3D spatial queries.
     * @see Octree::pcloud3D
     */
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> * octree3D = nullptr;
    /**
     * @brief The underyling octree for 2D spatial queries, where all the
     *  points are considered to have an equal \f$z\f$ coordinate.
     * @see Octree::pcloud2D
     */
    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> * octree2D = nullptr;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Instantiate an Octree from the given structure space matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times 3}\f$.
     *
     * When the max node value is used, the length of the longest axis of the
     * bounding box containing the point cloud \f$d\f$ will be considered.
     * For then, assuming a 3D structure space, the number of nodes will be
     * given by \f$2^{3n}\f$ where \f$n\f$ represents the number of partitions
     * along the longest axis. Therefore, the resolution \f$r\f$ will be
     * \f$r = d/n\f$ with
     * \f$n = \left\lceil\frac{\log_2(N)}{3}\right\rceil\f$,
     * where \f$N\f$ is the max nodes argument.
     *
     * @param X The structure space matrix of \f$m\f$ points in 3D.
     * @param make2D Whether to build an Octree for 2D spatial queries (true)
     *  or not (false).
     * @param make3D Whether to build an Octree for 3D spatial queries (true)
     *  or not (false).
     * @param directResolution Whether to use the given resolution and ignore
     *  the max nodes and expected point/node ratio (true) or not (false).
     * @param forceEpnRatio Whether to consider the expected point/node ratio
     *  when computing the resolution from the max nodes (true) or not (false).
     * @param resolution The resolution for the octree to be built, i.e.,
     *  the voxel size for the leaf octants.
     * @param maxNodes The maximum number of nodes the Octree can have. Note
     *  that the number of nodes can be smaller but it will never be bigger.
     * @param epnRatio The expected point/node ratio. When considered, the
     *  number of max nodes will be truncated (if necessary) to satisfy
     *  that the number of points per node is not NECESSARILY smaller than
     *  this ratio. Note that nodes with less points than epnRatio are allowed,
     *  depending on the point distribution. However, the resolution will not
     *  be set to a value for which this happens necessarily due to max
     *  nodes greater than the number of points divided by the epnRatio.
     * @see Octree::octree2D
     * @see Octree::octree3D
     */
    Octree(
        arma::Mat<DecimalType> const &X,
        bool const make2D=false,
        bool const make3D=true,
        bool const directResolution=true,
        bool const forceEpnRatio=true,
        DecimalType const resolution=1.0f,
        arma::uword const maxNodes=1000000000,
        arma::uword const epnRatio=8
    );
    /**
     * @brief Instantiate an Octree using the given point clouds.
     *
     * @param pcloud3D The 3D point cloud for the Octree.
     * @param pcloud2D The 2D point cloud for the Octree.
     * @param resolution The resolution for the octree to be built, i.e.,
     *  the voxel size for the leaf octants.
     * @see Octree::pcloud2D
     * @see Octree::pcloud3D
     * @see Octree::octree2D
     * @see Octree::octree3D
     */
    Octree(
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D=nullptr,
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud2D=nullptr,
        DecimalType const resolution=1.0f
    );
    virtual ~Octree();


    // ***  ASSIGNMENT OPERATOR  *** //
    // ***************************** //
    /**
     * @brief Cannibalize the rhs Octree, i.e., take the ownership of its
     *  resources.
     * @param ot The Octree to be cannibalized.
     * @return Reference to the cannibal Octree (this).
     */
    Octree & operator=(Octree &&ot);


    // ***   BUILDING METHODS   *** //
    // **************************** //
    /**
     * @param X The structure space matrix of \f$m\f$ points in 3D.
     * @param make2D Whether to build an Octree for 2D spatial queries (true)
     *  or not (false).
     * @param make3D Whether to build an Octree for 3D spatial queries (true)
     *  or not (false).
     * @param resolution The resolution for the octree to be built, i.e.,
     *  the voxel size for the leaf octants.
     * @see Octree::buildOctree2D
     * @see Octree::buildOctree3D
     */
    void buildOctree(
        arma::Mat<DecimalType> const &X,
        bool const make2D,
        bool const make3D,
        DecimalType const resolution
    );
    /**
     * @brief Build the Octree for 2D spatial queries.
     * @param resolution Voxel size for leaf octants.
     * @see Octree::buildOctree
     * @see Octree::octree2D
     */
    void buildOctree2D(DecimalType const resolution);
    /**
     * @brief Build the Octree for 3D spatial queries.
     * @param resolution Voxel size for leaf octants.
     * @see Octree::buildOctree
     * @see Octree::octree3D
     */
    void buildOctree3D(DecimalType const resolution);


    // ***   KNN METHODS   *** //
    // *********************** //
    /**
     * @see Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findKnn(
        PointType const &x, IndexType const k
    ) VL3DPP_USED_ ;
    /**
     * @brief Find the K nearest neighbors of point
     *  \f$\pmb{x} \in \mathbb{R}^{3}\f$.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{k_1*}, \ldots, \pmb{x}_{k_K*} :
     *      \Vert{\pmb{x}_{k_1*}-\pmb{x}}\Vert \leq \ldots \leq
     *          \Vert{\pmb{x}_{k_K*}-\pmb{x}}\Vert
     *  \right\}
     * \f]
     *
     * Where \f$\pmb{x}_{k_i*} \in \mathbb{R}^{3}\f$ is the \f$i\f$-th point
     * in the point cloud that is closest to \f$\pmb{x}\f$.
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] k How many nearest neighbors must be found.
     * @param[out] n Output vector to be filled with the indices of the
     *  \f$K\f$ nearest neighbors.
     * @param[out] d Output vector to be filled with the 3D distances with
     *  respect to each of the \f$K\f$ nearest neighbors.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findKnn(
        PointType const &x,
        IndexType const k,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    arma::Mat<IndexType> findKnn(
        arma::Mat<DecimalType> const &X, IndexType const k
    );
    /**
     * @see Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findKnn(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        arma::Mat<IndexType> &N,
        arma::Mat<DecimalType> &D
    ) VL3DPP_USED_ ;


    // ***   2D KNN METHODS   *** //
    // ************************** //
    /**
     * @see Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findKnn2D(
        PointType const &x, IndexType const k
    ) VL3DPP_USED_ ;
        /**
         * @brief Find the K nearest neighbors of point
         *  \f$\pmb{x} \in \mathbb{R}^{2}\f$.
         *
         * The neighborhood definition is:
         *
         * \f[
         *  \mathcal{N}(\pmb{x}) = \left\{
         *      \pmb{x}_{k_1*}, \ldots, \pmb{x}_{k_K*} :
         *      \Vert{\pmb{x}_{k_1*}-\pmb{x}}\Vert \leq \ldots \leq
         *          \Vert{\pmb{x}_{k_K*}-\pmb{x}}\Vert
         *  \right\}
         * \f]
         *
         * Where \f$\pmb{x}_{k_i*} \in \mathbb{R}^{2}\f$ is the \f$i\f$-th
         * point in the point cloud that is closest to \f$\pmb{x}\f$.
         *
         * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
         * @param[in] k How many nearest neighbors must be found.
         * @param[out] n Output vector to be filled with the indices of the
         *  \f$K\f$ nearest neighbors.
         * @param[out] d Output vector to be filled with the 2D distances with
         *  respect to each of the \f$K\f$ nearest neighbors.
         */
    template <typename PointType=arma::Col<DecimalType>>
    void findKnn2D(
        PointType const &x,
        IndexType const k,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    arma::Mat<IndexType> findKnn2D(
        arma::Mat<DecimalType> const &X, IndexType const k
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findKnn2D(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        arma::Mat<IndexType> &N,
        arma::Mat<DecimalType> &D
    ) VL3DPP_USED_ ;


    // ***   BOUNDED KNN METHODS   *** //
    // ******************************* //
    /**
     * @see Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBoundedKnn(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @brief Find the K nearest neighbors of point
     *  \f$\pmb{x} \in \mathbb{R}^{3}\f$ bounded by the max squared
     *  distance threshold \f$\tau^2\in\mathbb{R}_{>0}\f$.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{k_1*}, \ldots, \pmb{x}_{k_K*} :
     *      \Vert{\pmb{x}_{k_1*}-\pmb{x}}\Vert \leq \ldots \leq
     *          \Vert{\pmb{x}_{k_K*}-\pmb{x}}\Vert ,\,
     *      i=1,\ldots,K,\,
     *      \Vert{\pmb{x}_{k_i*}-\pmb{x}}\Vert^2 \leq \tau^2
     *  \right\}
     * \f]
     *
     * Where \f$\pmb{x}_{k_i*} \in \mathbb{R}^{3}\f$ is the \f$i\f$-th point
     * in the point cloud that is closest to \f$\pmb{x}\f$, subject to
     * being closer than the given distance \f$\tau^2\f$.
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] k How many nearest neighbors must be found.
     * @param[in] maxSquaredDistance The max squared distance threshold
     *  \f$\tau^2\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  \f$K\f$ bounded nearest neighbors.
     * @param[out] d Output vector to be filled with the 3D distances with
     *  respect to each of the \f$K\f$ bounded nearest neighbors.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findBoundedKnn(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_;
    /**
     * @see Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findBoundedKnn(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findBoundedKnn(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance,
        std::vector<arma::Col<IndexType>> &N,
        std::vector<arma::Col<DecimalType>> &D
    ) VL3DPP_USED_ ;


    // ***   BOUNDED 2D KNN METHODS   *** //
    // ********************************** //
    /**
     * @see Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBoundedKnn2D(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @brief Find the K nearest neighbors of point
     *  \f$\pmb{x} \in \mathbb{R}^{2}\f$ bounded by the max squared
     *  distance threshold \f$\tau^2\in\mathbb{R}_{>0}\f$.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{k_i*}, \ldots, \pmb{x}_{k_K*} :
     *      \Vert{\pmb{x}_{k_1}-\pmb{x}}\Vert \leq \ldots \leq
     *          \Vert{\pmb{x}_{k_K*}-\pmb{x}}\Vert ,\,
     *      i=1,\ldots,K,\,
     *      \Vert{\pmb{x}_{k_i*}-\pmb{x}}\Vert^2 \leq \tau^2
     *  \right\}
     * \f]
     *
     * Where \f$\pmb{x}_{k_i*} \in \mathbb{R}^{2}\f$ is the \f$i\f$-th point
     * in the point cloud that is closest to \f$\pmb{x}\f$, subject to
     * being closer than the given distance \f$\tau^2\f$. Note that \f$K\f$
     * might be smaller than requested if there are not enough neighbors
     * that satisfy the max squared distance constraint.
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] k How many nearest neighbors must be found.
     * @param[in] maxSquaredDistance The max squared distance threshold
     *  \f$\tau^2\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  \f$K\f$ bounded nearest neighbors.
     * @param[out] d Output vector to be filled with the 2D distances with
     *  respect to each of the \f$K\f$ bounded nearest neighbors.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findBoundedKnn2D(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findBoundedKnn2D(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findBoundedKnn2D(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance,
        std::vector<arma::Col<IndexType>> &N,
        std::vector<arma::Col<DecimalType>> &D
    ) VL3DPP_USED_ ;


    // ***  SPHERICAL NEIGHBORHOOD METHODS  *** //
    // **************************************** //
    /**
     * @see Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findSphere(
        PointType const &x,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors on the 3D ball of radius
     *  \f$r\in\mathbb{R}_{>0}\f$ centered at
     *  \f$\pmb{x} \in \mathbb{R}^{3}\f$.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m,
     *      \Vert{\pmb{x}_{i*} - \pmb{x}}\Vert \leq r
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] r The radius \f$r\f$ of the spherical neighborhood.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the given sphere.
     * @param[out] d Output vector to be filled with the distances with
     *  respect to each neighbor in the spherical neighborhood.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findSphere(
        PointType const &x,
        DecimalType const r,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findSphere(
        arma::Mat<DecimalType> const &X,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findSphere(
        arma::Mat<DecimalType> const &X,
        DecimalType const r,
        std::vector<arma::Col<IndexType>> &N,
        std::vector<arma::Col<DecimalType>> &D
    ) VL3DPP_USED_ ;


    // ***  CYLINDRICAL NEIGHBORHOOD METHODS  *** //
    // ****************************************** //
    /**
     * @see Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findCylinder(
        PointType const &x,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors inside the 2D circle of radius
     *  \f$r\in\mathbb{R}_{>0}\f$ centered at
     *  \f$\pmb{x} \in \mathbb{R}^{2}\f$. Note that considering all the
     *  neighbors inside a 2D circle embedded in a 3D space yields a
     *  cylindrical neighborhood as the circle is propagated along the axis
     *  of the third coordinate.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m,
     *      \Vert{\pmb{x}_{i*} - \pmb{x}}\Vert \leq r
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] r The radius \f$r\f$ of the cylindrical neighborhood.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the cylindrical neighborhoods.
     * @param[out] d Output vector to be filled with the 2D distances with
     *  respect to each neighbor in the cylindrical neighborhood.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findCylinder(
        PointType const &x,
        DecimalType const r,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findCylinder(
        arma::Mat<DecimalType> const &X,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    void findCylinder(
        arma::Mat<DecimalType> const &X,
        DecimalType const r,
        std::vector<arma::Col<IndexType>> &N,
        std::vector<arma::Col<DecimalType>> &D
    ) VL3DPP_USED_ ;


    // ***  BOUNDED CYLINDRICAL NEIGHBORHOOD METHODS  *** //
    // ************************************************** //
    /**
     * @see Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    arma::Col<IndexType> findBoundedCylinder(
        PointType const &x,
        BoundaryCoordinateType const &z,
        DecimalType const r,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors inside the 2D circle of radius
     *  \f$r\in\mathbb{R}_{>0}\f$ centered at
     *  \f$\pmb{x} \in \mathbb{R}^{2}\f$. Note that considering all the
     *  neighbors inside a 2D circle embedded in a 3D space yields a
     *  cylindrical neighborhood as the circle is propagated along the axis
     *  of the third coordinate. Moreover, there is a vertical value
     *  \f$z\f$ associated to the point \f$\pmb{x}\f$. Any neighbor inside
     *  the 2D circle must also have its vertical value \f$z_{i}\f$
     *  satisfying \f$z-z_* \leq z_i \leq z+z^*\f$, where
     *  \f$z_*, z^* \in \mathbb{R}_{>0}\f$ are the lower and upper vertical
     *  thresholds, respectively.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m,
     *      \Vert{\pmb{x}_{i*} - \pmb{x}}\Vert \leq r ,\,
     *      z-z_* \leq z_i \leq z+z^*
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] z The point-wise coordinates for the boundary check.
     * @param[in] r The radius \f$r\f$ of the cylindrical neighborhood.
     * @param[in] zmin The min vertical threshold \f$z_*\f$.
     * @param[in] zmax The max vertical threshold \f$z^*\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the bounded cylinder.
     * @param[out] d Output vector to be filled with the 2D distances with
     *  respect to each neighbor in the bounded cylindrical neighborhood.
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    void findBoundedCylinder(
        PointType const &x,
        BoundaryCoordinateType const &z,
        DecimalType const r,
        DecimalType const zmin,
        DecimalType const zmax,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    std::vector<arma::Col<IndexType>> findBoundedCylinder(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        DecimalType const r,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    void findBoundedCylinder(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        DecimalType const r,
        DecimalType const zmin,
        DecimalType const zmax,
        std::vector<arma::Col<IndexType>> &N,
        std::vector<arma::Col<DecimalType>> &D
    ) VL3DPP_USED_ ;


    // ***  BOX NEIGHBORHOOD METHODS  *** //
    // ********************************** //
    /**
     * @see Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBox(
        PointType const &x,
        PointType const halfLength
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors inside the given 3D box with axis-wise
     * half sizes \f$\pmb{s} = (s_x, s_y, s_z) \leq \pmb{0}\f$ centered at
     * \f$\pmb{x} \in \mathbb{R}^{3}\f$.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m ,\,
     *      \pmb{x} - \pmb{s} \leq \pmb{x}_{i*} \leq \pmb{x} + \pmb{s}
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] halfLength The half size for each axis \f$\pmb{s}\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the box.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findBox(
        PointType const &x,
        PointType const halfLength,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    std::vector<arma::Col<IndexType>> findBox(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    void findBox(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const halfLength,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;


    // ***  RECTANGULAR NEIGHBORHOOD METHODS  *** //
    // ****************************************** //
    /**
     * @see Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findRectangle(
        PointType const &x,
        PointType const halfLength
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors inside the given 2D rectangle with
     *  axis-wise half sizes \f$\pmb{s} = (s_x, s_y) \leq \pmb{0}\f$ centered
     *  at \f$\pmb{x} \in \mathbb{R}^{2}\f$. Note that considering all the
     *  neighbors inside a 2D rectangle embedded in a 3D space yields a
     *  3D neighborhood consisting of the rectangle propagated along the axis
     *  of the third variable.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m ,\,
     *      \pmb{x} - \pmb{s} \leq \pmb{x}_{i*} \leq \pmb{x} + \pmb{s}
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] halfLength The half size for each axis \f$\pmb{s}\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the rectangle.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findRectangle(
        PointType const &x,
        PointType const halfLength,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    std::vector<arma::Col<IndexType>> findRectangle(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    void findRectangle(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const halfLength,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;


    // ***  BOUNDED RECTANGULAR NEIGHBORHOOD METHODS  *** //
    // ************************************************** //
    /**
     * @see Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    arma::Col<IndexType> findBoundedRectangle(
        PointType const &x,
        BoundaryCoordinateType const &z,
        PointType const halfLength,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors inside the given 2D rectangle with
     *  axis-wise half sizes \f$\pmb{s} = (s_x, s_y) \leq \pmb{0}\f$ centered
     *  at \f$\pmb{x} \in \mathbb{R}^{2}\f$. Note that considering all the
     *  neighbors inside a 2D rectangle embedded in a 3D space yields a
     *  3D neighborhood consisting of the rectangle propagated along the axis
     *  of the third variable. Moreover, there is a vertical value \f$z\f$
     *  associated to the point \f$\pmb{x}\f$. Any neighbor inside the 2D
     *  rectangle must also have its vertical value \f$z_i\f$ satisfying
     *  \f$z-z_* \leq z_i \leq z + z^*\f$, where
     *  \f$z_*, z^* \in \mathbb{R}_{>0}\f$ are the lower and upper vertical
     *  thresholds, respectively.
     *
     * The neighborhood definition is:
     *
     * \f[
     *  \mathcal{N}(\pmb{x}) = \left\{
     *      \pmb{x}_{i*} :
     *      1 \leq i \leq m ,\,
     *      \pmb{x} - \pmb{s} \leq \pmb{x}_{i*} \leq \pmb{x} + \pmb{s} ,\,
     *      z-z_* \leq z_i \leq z+z^*
     *  \right\}
     * \f]
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[in] z The point-wise coordinates for the boundary check.
     * @param[in] halfLength The half size for each axis \f$\pmb{s}\f$.
     * @param[in] zmin The min vertical threshold \f$z_*\f$.
     * @param[in] zmax The max vertical threshold \f$z^*\f$.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the rectangle.
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    void findBoundedRectangle(
        PointType const &x,
        BoundaryCoordinateType const &z,
        PointType const halfLength,
        DecimalType const zmin,
        DecimalType const zmax,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <typename BoundaryCoordinateType=arma::subview_col<DecimalType>>
    std::vector<arma::Col<IndexType>> findBoundedRectangle(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const halfLength,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <typename BoundaryCoordinateType=arma::subview_col<DecimalType>>
    void findBoundedRectangle(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const halfLength,
        DecimalType const zmin,
        DecimalType const zmax,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;


    // ***  OCTANT NEIGHBORHOOD METHODS  *** //
    // ************************************* //
    /**
     * @see Octree::findOctant(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findOctant(
        PointType const &x
    ) VL3DPP_USED_ ;
    /**
     * @brief Find all the neighbors that lie in the same octree's octant as
     *  the given \f$\pmb{x} \in \mathbb{R}^{3}\f$.
     *
     * @param[in] x The point \f$\pmb{x}\f$ whose neighborhood must be found.
     * @param[out] n Output vector to be filled with the indices of the
     *  neighbors inside the same octant as \f$\pmb{x}\f$.
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findOctant(
        PointType const &x, arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findOctant(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&)
     */
    std::vector<arma::Col<IndexType>> findOctant(
        arma::Mat<DecimalType> const &X
    ) VL3DPP_USED_ ;
    /**
     * @see Octree::findOctant(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&)
     */
    void findOctant(
        arma::Mat<DecimalType> const &X,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;

};

}

#include <adt/octree/Octree.tpp>

#endif