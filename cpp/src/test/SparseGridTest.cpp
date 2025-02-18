#include <test/SparseGridTest.hpp>

using vl3dpp::test::SparseGridTest;

// ***   R U N   *** //
// ***************** //
bool SparseGridTest::run(){
    // Compute 2D test
    SparseGrid<float, arma::uword> sg2D(1, 0, 1);
    arma::Mat<float> const X2D = {
        {-2, -2},       // alpha
        {-1, -1},       // A
        {1.1, 0.4},     // B
        {1.7, 0.6},     // C
        {3.3, -0.5},    // D
        {4.2, 1.1},     // E
        {5.5, 0.3},     // F
        {6.2, -1.1},    // G
        {6.5, -1.7},    // H
        {7.7, 3},       // I
        {8.1, 1.5},     // J
        {8.1, -0.5},    // K
        {8.6, -0.1},    // L
        {8.6, -0.9},    // M
        {10, 3}         // beta
    };
    arma::Mat<float> const F2D = {
        {-1.1, 1.1},    // alpha
        {-1.2, 0.5},    // A
        {-1.3, 0.3},    // B
        {-1.4, 1.2},    // C
        {0.5, 0.5},     // D
        {0.6, -0.1},    // E
        {0.0, 0.1},     // F
        {1.3, 0.0},     // G
        {1.3, 1.3},     // H
        {0.8, 0.2},     // I
        {0.7, 0.3},     // J
        {0.9, -0.2},    // K
        {-0.3, -0.4},   // L
        {0.1, 0.5},     // M
        {0.9, 0.3}      // beta
    };
    arma::Col<float> const &f2D = F2D.col(0);
    arma::Mat<arma::s32> C2D = {
        {0, 0},     // alpha
        {1, 0},     // A
        {2, 1},     // B
        {2, 1},     // C
        {3, 2},     // D
        {4, 2},     // E
        {0, 2},     // F
        {1, 3},     // G
        {1, 3},     // H
        {2, 4},     // I
        {3, 4},     // J
        {4, 0},     // K
        {0, 0},     // L
        {0, 0},     // M
        {4, 1}      // beta
    };
    arma::Col<arma::s32> const &c2D = C2D.col(0);
    sg2D.fit(X2D);
    arma::Mat<float> Y2Dmean = sg2D.encodeMatrix(X2D, F2D, "mean");
    arma::Mat<float> Y2Dmax = sg2D.encodeMatrix(X2D, F2D, "max");
    arma::Mat<float> Y2Dmin = sg2D.encodeMatrix(X2D, F2D, "min");
    arma::Mat<arma::s32> C2Dmode = sg2D.encodeMatrix(X2D, C2D, "mode");
    arma::Col<float> y2Dmean = sg2D.encodeVector(X2D, f2D, "mean");
    arma::Col<float> y2Dmax = sg2D.encodeVector(X2D, f2D, "max");
    arma::Col<float> y2Dmin = sg2D.encodeVector(X2D, f2D, "min");
    arma::Col<arma::s32> c2Dmode = sg2D.encodeVector(X2D, c2D, "mode");
    arma::Mat<float> const refY2Dmean = {
        {-1.1, 1.1},    // alpha
        {-1.2, 0.5},    // A
        {-1.35, 0.75},  // B,C
        {0.5, 0.5},     // D
        {0.6, -0.1},    // E
        {0.0, 0.1},     // F
        {1.3, 0.65},    // G,H
        {0.8, 0.2},     // I
        {0.7, 0.3},     // J
        {0.2333333, -0.0333333},     // K,L,M
        {0.9, 0.3}      // beta
    };
    arma::Mat<float> const refY2Dmax = {
        {-1.1, 1.1},    // alpha
        {-1.2, 0.5},    // A
        {-1.3, 1.2},    // B,C
        {0.5, 0.5},     // D
        {0.6, -0.1},    // E
        {0.0, 0.1},     // F
        {1.3, 1.3},     // G,H
        {0.8, 0.2},     // I
        {0.7, 0.3},     // J
        {0.9, 0.5},     // K,L,M
        {0.9, 0.3}      // beta
    };
    arma::Mat<float> const refY2Dmin = {
        {-1.1, 1.1},    // alpha
        {-1.2, 0.5},    // A
        {-1.4, 0.3},    // B,C
        {0.5, 0.5},     // D
        {0.6, -0.1},    // E
        {0.0, 0.1},     // F
        {1.3, 0.0},     // G,H
        {0.8, 0.2},     // I
        {0.7, 0.3},     // J
        {-0.3, -0.4},   // K,L,M
        {0.9, 0.3}      // beta
    };
    arma::Mat<arma::s32> const refC2Dmode = {
        {0, 0},     // alpha
        {1, 0},     // A
        {2, 1},     // B,C
        {3, 2},     // D
        {4, 2},     // E
        {0, 2},     // F
        {1, 3},     // G,H
        {2, 4},     // I
        {3, 4},     // J
        {0, 0},     // K,L,M
        {4, 1}      // beta
    };
    if(!arma::approx_equal(refY2Dmean, Y2Dmean, "absdiff", eps)) return false;
    if(!arma::approx_equal(refY2Dmax, Y2Dmax, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY2Dmin, Y2Dmin, "absdiff", 0)) return false;
    if(!arma::approx_equal(refC2Dmode, C2Dmode, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY2Dmean.col(0), y2Dmean, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY2Dmax.col(0), y2Dmax, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY2Dmin.col(0), y2Dmin, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refC2Dmode.col(0), c2Dmode, "absdiff", 0)){
        return false;
    }
    if(sg2D.getNumCells() != 60) return false;
    if(sg2D.getNumActiveCells() != 11) return false;
    if(sg2D.getNumAxisPartitions(0) != 12) return false;
    if(sg2D.getNumAxisPartitions(1) != 5) return false;

    // Compute 3D test
    SparseGrid<float, arma::uword> sg3D(1, 0, 1);
    arma::Mat<float> const X3D = {
        {0, 0, 0},          // alpha
        {2, 0, 1},          // A
        {2.1, 0.1, 2.2},    // B
        {2.5, 0.5, 3.5},    // C
        {2.2, 1.2, 2.3},    // D
        {3.3, 0.3, 1.3},    // E
        {3.4, 0.4, 1.4},    // F
        {3.3, 0.3, 2.3},    // G
        {3.6, 0.6, 2.5},    // H
        {3.2, 1.2, 2.5},    // I
        {4.2, 0.3, 1.1},    // J
        {4.2, 0.3, 1.2},    // K
        {4.1, 0.6, 1.3},    // L
        {4.2, 0.5, 2.1},    // M
        {4.2, 0.6, 2.1},    // N
        {4.2, 1.1, 2.1},    // O
        {4.3, 1.3, 2.2},    // P
        {11.5, 0.5, 0.5},   // Q
        {11.5, 1.1, 1.2},   // R
        {11.6, 1.3, 1.3},   // S
        {14, 2, 4}          // beta
    };
    arma::Mat<float> const F3D = {
        {-1.1, 1.1},    // alpha
        {-1.2, 0.5},    // A
        {-1.3, 0.3},    // B
        {-1.4, 1.2},    // C
        {0.5, 0.5},     // D
        {0.6, -0.1},    // E
        {0.0, 0.1},     // F
        {1.3, 0.0},     // G
        {1.3, 1.3},     // H
        {0.8, 0.2},     // I
        {0.7, 0.3},     // J
        {0.9, -0.2},    // K
        {-0.3, -0.4},   // L
        {0.1, 0.5},     // M
        {-1, 0},        // N
        {1, 0},         // O
        {-1, -1},       // P
        {1, 1},         // Q
        {1, -1},        // R
        {0.3, 0.3},     // S
        {0.9, 0.3}      // beta
    };
    arma::Col<float> const &f3D = F3D.col(0);
    arma::Mat<arma::s32> C3D = {
        {0, 0},     // alpha
        {1, 0},     // A
        {2, 1},     // B
        {2, 1},     // C
        {3, 2},     // D
        {4, 2},     // E
        {4, 2},     // F
        {1, 3},     // G
        {1, 3},     // H
        {2, 4},     // I
        {3, 4},     // J
        {4, 0},     // K
        {3, 0},     // L
        {0, 0},     // M
        {0, 0},     // N
        {3, 2},     // O
        {3, 2},     // P
        {2, 3},     // Q
        {3, 3},     // R
        {3, 3},     // S
        {4, 1}      // beta
    };
    arma::Col<arma::s32> const &c3D = C3D.col(0);
    sg3D.fit(X3D);
    arma::Mat<float> Y3Dmean = sg3D.encodeMatrix(X3D, F3D, "mean");
    arma::Mat<float> Y3Dmax = sg3D.encodeMatrix(X3D, F3D, "max");
    arma::Mat<float> Y3Dmin = sg3D.encodeMatrix(X3D, F3D, "min");
    arma::Mat<arma::s32> C3Dmode = sg3D.encodeMatrix(X3D, C3D, "mode");
    arma::Col<float> y3Dmean = sg3D.encodeVector(X3D, f3D, "mean");
    arma::Col<float> y3Dmax = sg3D.encodeVector(X3D, f3D, "max");
    arma::Col<float> y3Dmin = sg3D.encodeVector(X3D, f3D, "min");
    arma::Col<arma::s32> c3Dmode = sg3D.encodeVector(X3D, c3D, "mode");
    arma::Mat<float> const refY3Dmean = {
        {-1.1, 1.1},        // alpha
        {-1.2, 0.5},        // A
        {-1.3, 0.3},        // B
        {-1.4, 1.2},        // C
        {0.5, 0.5},         // D
        {0.3, 0},           // E, F
        {1.3, 0.65},        // G, H
        {0.8, 0.2},         // I
        {0.4333333, -0.1},  // J, K, L
        {-0.45, 0.25},      // M, N
        {0, -0.5},          // O, P
        {1, 1},             // Q
        {0.65, -0.35},      // R, S
        {0.9, 0.3}          // beta
    };
    arma::Mat<float> const refY3Dmax = {
        {-1.1, 1.1},        // alpha
        {-1.2, 0.5},        // A
        {-1.3, 0.3},        // B
        {-1.4, 1.2},        // C
        {0.5, 0.5},         // D
        {0.6, 0.1},         // E, F
        {1.3, 1.3},         // G, H
        {0.8, 0.2},         // I
        {0.9, 0.3},         // J, K, L
        {0.1, 0.5},         // M, N
        {1, 0},             // O, P
        {1, 1},             // Q
        {1, 0.3},           // R, S
        {0.9, 0.3}          // beta
    };
    arma::Mat<float> const refY3Dmin = {
        {-1.1, 1.1},        // alpha
        {-1.2, 0.5},        // A
        {-1.3, 0.3},        // B
        {-1.4, 1.2},        // C
        {0.5, 0.5},         // D
        {0, -0.1},          // E, F
        {1.3, 0},           // G, H
        {0.8, 0.2},         // I
        {-0.3, -0.4},       // J, K, L
        {-1, 0},            // M, N
        {-1, -1},           // O, P
        {1, 1},             // Q
        {0.3, -1},          // R, S
        {0.9, 0.3}          // beta
    };
    arma::Mat<arma::s32> const refC3Dmode = {
        {0, 0},    // alpha
        {1, 0},    // A
        {2, 1},    // B
        {2, 1},    // C
        {3, 2},    // D
        {4, 2},    // E, F
        {1, 3},    // G, H
        {2, 4},    // I
        {3, 0},    // J, K, L
        {0, 0},    // M, N
        {3, 2},    // O, P
        {2, 3},    // Q
        {3, 3},    // R, S
        {4, 1}     // beta
    };
    if(!arma::approx_equal(refY3Dmean, Y3Dmean, "absdiff", eps)) return false;
    if(!arma::approx_equal(refY3Dmax, Y3Dmax, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY3Dmin, Y3Dmin, "absdiff", 0)) return false;
    if(!arma::approx_equal(refC3Dmode, C3Dmode, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY3Dmean.col(0), y3Dmean, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY3Dmax.col(0), y3Dmax, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY3Dmin.col(0), y3Dmin, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refC3Dmode.col(0), c3Dmode, "absdiff", 0)){
        return false;
    }
    if(sg3D.getNumCells() != 112) return false;
    if(sg3D.getNumActiveCells() != 14) return false;
    if(sg3D.getNumAxisPartitions(0) != 14) return false;
    if(sg3D.getNumAxisPartitions(1) != 2) return false;
    if(sg3D.getNumAxisPartitions(2) != 4) return false;

    // Compute 3D test with padding
    SparseGrid<float, arma::uword> sg3DP(1, 3, 1);
    sg3DP.fit(X3D);
    Y3Dmean = sg3DP.encodeMatrix(X3D, F3D, "mean");
    Y3Dmax = sg3DP.encodeMatrix(X3D, F3D, "max");
    Y3Dmin = sg3DP.encodeMatrix(X3D, F3D, "min");
    C3Dmode = sg3DP.encodeMatrix(X3D, C3D, "mode");
    y3Dmean = sg3DP.encodeVector(X3D, f3D, "mean");
    y3Dmax = sg3DP.encodeVector(X3D, f3D, "max");
    y3Dmin = sg3DP.encodeVector(X3D, f3D, "min");
    c3Dmode = sg3DP.encodeVector(X3D, c3D, "mode");
    if(!arma::approx_equal(refY3Dmean, Y3Dmean, "absdiff", eps)) return false;
    if(!arma::approx_equal(refY3Dmax, Y3Dmax, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY3Dmin, Y3Dmin, "absdiff", 0)) return false;
    if(!arma::approx_equal(refC3Dmode, C3Dmode, "absdiff", 0)) return false;
    if(!arma::approx_equal(refY3Dmean.col(0), y3Dmean, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY3Dmax.col(0), y3Dmax, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refY3Dmin.col(0), y3Dmin, "absdiff", eps)){
        return false;
    }
    if(!arma::approx_equal(refC3Dmode.col(0), c3Dmode, "absdiff", 0)){
        return false;
    }
    if(sg3DP.getNumCells() != 1600) return false;
    if(sg3DP.getNumActiveCells() != 14) return false;
    if(sg3DP.getNumAxisPartitions(0) != 20) return false;
    if(sg3DP.getNumAxisPartitions(1) != 8) return false;
    if(sg3DP.getNumAxisPartitions(2) != 10) return false;

    // On all tests passed
    return true;
}