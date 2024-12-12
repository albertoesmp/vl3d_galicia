#ifndef VL3DPP_KDTREE_
#define VL3DPP_KDTREE_

// ***   INCLUDES   *** //
// ******************** //
#include <util/VL3DPPMacros.hpp>
#include <math/MathConstants.hpp>
#include <armadillo>
#include <pcl/point_cloud.h>
#include <pcl/search/kdtree.h>
#include <vector>

namespace vl3dpp::test{
    class KDTreeTest;
}

namespace vl3dpp::adt::kdtree {

/**
 * @author Alberto M. Esmoris Pena
 * @version 1.0
 *
 * @brief KDTree class.
 *
 * Class that represents a typical K-dimensional tree. It can be used to
 * significantly speedup spatial queries.
 *
 * @tparam IndexType The type for the indices and also some integer values.
 * @tparam DecimalType The type for decimal numbers.
 */
template <typename IndexType=int, typename DecimalType=float>
class KDTree{
protected:
    friend class vl3dpp::test::KDTreeTest;
    // ***   ATTRIBUTES   *** //
    // ********************** //
    /**
     * @brief The 3D point cloud on top of which the 3D KDTree is built.
     * @see KDTree::kdt3D
     */
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D = nullptr;
    /**
     * @brief The 2D point cloud on top of which the 2D KDTree is built.
     * @see KDTree::kdt2D
     */
    pcl::PointCloud<pcl::PointXY>::Ptr pcloud2D = nullptr;
    /**
     * @brief The underlying KDTree for 3D spatial queries.
     * @see KDTree::pcloud3D
     */
    pcl::search::KdTree<pcl::PointXYZ> * kdt3D = nullptr;
    /**
     * @brief The underlying KDTree for 2D spatial queries.
     * @see KDTree::pcloud2D
     */
    pcl::search::KdTree<pcl::PointXY> * kdt2D = nullptr;

public:
    // ***  CONSTRUCTION / DESTRUCTION  *** //
    // ************************************ //
    /**
     * @brief Instantiate a KDTree from the given point-wise matrix
     *  \f$\pmb{X} \in \mathbb{R}^{m \times n}\f$ where the rows are points
     *  in an \f$n\f$-dimensional space.
     *
     * @param X The point-wise matrix of \f$m\f$ points (i.e., rows).
     * @param make2D Whether to build a KDTree for 2D spatial queries (true)
     *  or not (false).
     * @param make3D Whether to build a KDTree for 3D spatial queries (true)
     *  or not (false).
     * @see KDTree::kdt2D
     * @see KDTree::kdt3D
     */
    KDTree(
        arma::Mat<DecimalType> const &X,
        bool const make2D=false,
        bool const make3D=true
    );
    /**
     * @brief Instantiate a KDTree using the given point clouds.
     *
     * @param pcloud3D The 3D point cloud for the KDTree.
     * @param pcloud2D The 2D point cloud for the KDTree.
     * @see KDTree::pcloud2D
     * @see KDTree::pcloud3D
     * @see KDTree::kdt2D
     * @see KDTree::kdt3D
     */
    KDTree(
        pcl::PointCloud<pcl::PointXYZ>::Ptr pcloud3D=nullptr,
        pcl::PointCloud<pcl::PointXY>::Ptr pcloud2D=nullptr
    );
    virtual ~KDTree();

    // ***  ASSIGNMENT OPERATOR  *** //
    // ***************************** //
    /**
     * @brief Cannibalize the rhs KDTree, i.e., take the ownership of its
     *  resources.
     * @param kdt The KDTree to be cannibalized.
     * @return Reference to the cannibal KDTree (this).
     */
    KDTree & operator=(KDTree &&kdt);

    // ***   BUILDING METHODS   *** //
    // **************************** //
    /**
     * @brief Build the KDTree for 2D spatial queries.
     * @see KDTree::kdt2D
     */
    void buildKDTree2D();
    /**
     * @brief Build the KDTree for 3D spatial queries.
     * @see KDTree::kdt3D
     */
    void buildKDTree3D();

    // ***   KNN METHODS   *** //
    // *********************** //
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findKnn(
        PointType const &x, IndexType const k
    ) VL3DPP_USED_;
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findKnn(
        PointType const &x,
        IndexType const k,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    arma::Mat<IndexType> findKnn(
        arma::Mat<DecimalType> const &X, IndexType const k
    );
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findKnn2D(
        PointType const &x, IndexType const k
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findKnn2D(
        PointType const &x,
        IndexType const k,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    arma::Mat<IndexType> findKnn2D(
        arma::Mat<DecimalType> const &X, IndexType const k
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findKnn2D(arma::Col<DecimalType> const &, IndexType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBoundedKnn(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findBoundedKnn(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBoundedKnn2D(
        PointType const &x,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findBoundedKnn2D(
        arma::Mat<DecimalType> const &X,
        IndexType const k,
        DecimalType const maxSquaredDistance
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedKnn2D(arma::Col<DecimalType> const &, IndexType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findSphere(
        PointType const &x,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findSphere(
        PointType const &x,
        DecimalType const r,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findSphere(
        arma::Mat<DecimalType> const &X,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findSphere(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findCylinder(
        PointType const &x,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findCylinder(
        PointType const &x,
        DecimalType const r,
        arma::Col<IndexType> &n,
        arma::Col<DecimalType> &d
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    std::vector<arma::Col<IndexType>> findCylinder(
        arma::Mat<DecimalType> const &X,
        DecimalType const r
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findCylinder(arma::Col<DecimalType> const &, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
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
     * @see vl3dpp::adt::octree::Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename BoundaryCoordinateType=arma::subview_col<DecimalType>>
    std::vector<arma::Col<IndexType>> findBoundedCylinder(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        DecimalType const r,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedCylinder(arma::Col<DecimalType> const &, DecimalType const, DecimalType const, DecimalType const, arma::Col<IndexType>&, arma::Col<DecimalType>&)
     */
    template <typename BoundaryCoordinateType=arma::subview_col<DecimalType>>
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
     * @see vl3dpp::adt::octree::Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findBox(
        PointType const &x,
        arma::Col<DecimalType> const &halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findBox(
        PointType const &x,
        arma::Col<DecimalType> const &halfLength,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    std::vector<arma::Col<IndexType>> findBox(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const &halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBox(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    void findBox(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const &halfLength,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;


    // ***  RECTANGULAR NEIGHBORHOOD METHODS  *** //
    // ****************************************** //
    /**
     * @see vl3dpp::adt::octree::Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    arma::Col<IndexType> findRectangle(
        PointType const &x,
        arma::Col<DecimalType> const &halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    template <typename PointType=arma::Col<DecimalType>>
    void findRectangle(
        PointType const &x,
        arma::Col<DecimalType> const &halfLength,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    std::vector<arma::Col<IndexType>> findRectangle(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const &halfLength
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, arma::Col<IndexType>&)
     */
    void findRectangle(
        arma::Mat<DecimalType> const &X,
        arma::Col<DecimalType> const &halfLength,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;


    // ***  BOUNDED RECTANGULAR NEIGHBORHOOD METHODS  *** //
    // ************************************************** //
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    arma::Col<IndexType> findBoundedRectangle(
        PointType const &x,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const &halfLength,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    void findBoundedRectangle(
        PointType const &x,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const &halfLength,
        DecimalType const zmin,
        DecimalType const zmax,
        arma::Col<IndexType> &n
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    std::vector<arma::Col<IndexType>> findBoundedRectangle(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const &halfLength,
        DecimalType const zmin,
        DecimalType const zmax
    ) VL3DPP_USED_ ;
    /**
     * @see vl3dpp::adt::octree::Octree::findBoundedRectangle(arma::Col<DecimalType> const &, arma::Col<DecimalType> const, DecimalType const, DecimalType const, arma::Col<IndexType>&)
     */
    template <
        typename PointType=arma::Col<DecimalType>,
        typename BoundaryCoordinateType=arma::subview_col<DecimalType>
    >
    void findBoundedRectangle(
        arma::Mat<DecimalType> const &X,
        BoundaryCoordinateType const &z,
        arma::Col<DecimalType> const &halfLength,
        DecimalType const zmin,
        DecimalType const zmax,
        std::vector<arma::Col<IndexType>> &N
    ) VL3DPP_USED_ ;

protected:

};

}

#include <adt/kdtree/KDTree.tpp>

#endif