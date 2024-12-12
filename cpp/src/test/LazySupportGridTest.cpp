#include <test/LazySupportGridTest.hpp>
#include <util/VL3DPPException.hpp>

using namespace vl3dpp::adt::grid;
using namespace vl3dpp::test;

// ***  CONSTRUCTION / DESTRUCTION  *** //
// ************************************ //
LazySupportGridTest::LazySupportGridTest(double const eps) :
    BaseTest("Lazy support grid test"),
    eps(eps)
{}


// ***   R U N   *** //
// ***************** //
bool LazySupportGridTest::run(){
    // Initialize structure space for 2D test
    arma::Mat<double> X2D({
        {0.0, 0.0},
        {6.0, 4.0}
    });
    // LazySupportGrid for X2D
    LazySupportGrid<double> grid2D(
        arma::Col<double>({1.0, 1.0}),
        arma::min(X2D, 0).as_col(),
        arma::max(X2D, 0).as_col()
    );
    // Expected results for 2D test
    arma::Mat<double> G2D({
        {1, 1},
        {3, 1},
        {5, 1},
        {1, 3},
        {3, 3},
        {5, 3}
    });
    // Validate 2D test
    if(!validateGrid(grid2D, G2D)) return false;

    // Initialize structure space for 3D test
    arma::Mat<double> X3D({
        {-2.5, 0.0, -1.0},
        {2.5, 5.0, 9.0}
    });
    // LazySupportGrid for X3D
    LazySupportGrid<double> grid3D(
        arma::Col<double>({1.0, 1.0, 2.0}),
        arma::min(X3D, 0).as_col(),
        arma::max(X3D, 0).as_col()
    );
    // Expected results for 3D test
    arma::Mat<double> G3D({
        {-1.5, 1.0, 1.0},
        {0.5, 1.0, 1.0},
        {2.5, 1.0, 1.0},
        {-1.5, 3.0, 1.0},
        {0.5, 3.0, 1.0},
        {2.5, 3.0, 1.0},
        {-1.5, 5.0, 1.0},
        {0.5, 5.0, 1.0},
        {2.5, 5.0, 1.0},
        {-1.5, 1.0, 5.0},
        {0.5, 1.0, 5.0},
        {2.5, 1.0, 5.0},
        {-1.5, 3.0, 5.0},
        {0.5, 3.0, 5.0},
        {2.5, 3.0, 5.0},
        {-1.5, 5.0, 5.0},
        {0.5, 5.0, 5.0},
        {2.5, 5.0, 5.0},
        {-1.5, 1.0, 9.0},
        {0.5, 1.0, 9.0},
        {2.5, 1.0, 9.0},
        {-1.5, 3.0, 9.0},
        {0.5, 3.0, 9.0},
        {2.5, 3.0, 9.0},
        {-1.5, 5.0, 9.0},
        {0.5, 5.0, 9.0},
        {2.5, 5.0, 9.0}
    });
    // Validate 3D test
    if(!validateGrid(grid3D, G3D)) return false;

    // Find index test on 2D Grid
    LazySupportGrid<double> grid2D2(
         arma::Col<double>({1.0, 1.0}),
         arma::Col<double>({0, 0}),
         arma::Col<double>({8, 6})
    );
    if(grid2D2.findCellIndex({0.0, 0.0}) != 0) return false;
    if(grid2D2.findCellIndex({1.0, 0.0}) != 0) return false;
    if(grid2D2.findCellIndex({2.0, 0.0}) != 1) return false;
    if(grid2D2.findCellIndex({2.5, 0.0}) != 1) return false;
    if(grid2D2.findCellIndex({5.5, 0.0}) != 2) return false;
    if(grid2D2.findCellIndex({8.0, 0.0}) != 3) return false;
    if(grid2D2.findCellIndex({0.0, 2.0}) != 4) return false;
    if(grid2D2.findCellIndex({2.0, 2.0}) != 5) return false;
    if(grid2D2.findCellIndex({8.0, 6.0}) != 11) return false;

    // If this point is reached, then all test passed
    return true;
}


// ***  UTIL METHODS  *** //
// ********************** //
bool LazySupportGridTest::validateGrid(
    LazySupportGrid<double> const &grid,
    arma::Mat<double> const &Y
){
    for(size_t i = 0 ; i < grid.getNumCells() ; ++i) {
        arma::Col<double> gi = grid.getCentroid(i);
        arma::Col<double> yi = Y.row(i).as_col();
        if(arma::any(arma::abs(gi-yi) > eps)) return false;
    }
    return true;
}
