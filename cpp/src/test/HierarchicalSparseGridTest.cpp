#include <test/HierarchicalSparseGridTest.hpp>

using vl3dpp::test::HierarchicalSparseGridTest;

// ***   R U N   *** //
// ***************** //
bool HierarchicalSparseGridTest::run(){
    if(!simple2DTestA()) return false;
    if(!simple2DTestB()) return false;
    if(!simple2DTestC()) return false;
    if(!simple2DTestD()) return false;
    if(!simple2DTestE()) return false;
    if(!simple2DTestF()) return false;
    if(!simple2DTestG()) return false;
    if(!simple2DTestH()) return false;
    if(!simple2DTestI()) return false;
    if(!simple2DTestJ()) return false;
    if(!simple2DTestK()) return false;
    if(!simple3DTestA()) return false;

    // On all tests passed
    return true;
}

// ***   TESTS   *** //
// ***************** //
bool HierarchicalSparseGridTest::simple2DTestA(){
    // Reference values
    std::map<int, int> const h0Ref{
        {9, 0},
        {10, 1},
        {11, 2},
        {12, 3},
        {13, 4},
        {14, 5},
        {17, 6},
        {18, 7},
        {19, 8},
        {20, 9},
        {21, 10},
        {22, 11},
        {25, 12},
        {26, 13},
        {27, 14},
        {28, 15},
        {29, 16},
        {30, 17},
        {33, 18},
        {34, 19},
        {35, 20},
        {36, 21},
        {37, 22},
        {38, 23},
        {41, 24},
        {42, 25},
        {43, 26},
        {44, 27},
        {45, 28},
        {46, 29},
        {49, 30},
        {50, 31},
        {51, 32},
        {52, 33},
        {53, 34},
        {54, 35},
        {57, 36},
        {58, 37},
        {59, 38},
        {60, 39},
        {61, 40},
        {62, 41},
        {65, 42},
        {66, 43},
        {67, 44},
        {68, 45},
        {69, 46},
        {70, 47},
        {73, 48},
        {74, 49},
        {75, 50},
        {76, 51},
        {77, 52},
        {78, 53},
        {81, 54},
        {82, 55},
        {83, 56},
        {84, 57},
        {85, 58},
        {86, 59}
    };
    std::map<int, int> const h1Ref{
        {6, 0},
        {7, 1},
        {8, 2},
        {11, 3},
        {12, 4},
        {13, 5},
        {16, 6},
        {17, 7},
        {18, 8},
        {21, 9},
        {22, 10},
        {23, 11},
        {26, 12},
        {27, 13},
        {28, 14}
    };
    std::vector<int> const hDref{
        0,  2,  4,  16, 18,
        20, 32, 34, 36, 48,
        50, 52, 64, 66, 68
    };
    std::vector<int> const hUref{
        0, 	0, 	0, 	2, 	2,
        2, 	0, 	0, 	0, 	2,
        2, 	2, 	0, 	0, 	0,
        2,	2,	2,	10,	10,
        10,	12,	12,	12,	10,
        10,	10,	12,	12,	12,
        10,	10,	10,	12,	12,
        12,	10,	10,	10,	12,
        12,	12,	20,	20,	20,
        22,	22,	22,	20,	20,
        20,	22,	22,	22,	20,
        20,	20,	22,	22,	22
    };
    // Generate points
    arma::Mat<float> X(60, 2);
    for(arma::uword i = 0 ; i < 10 ; ++i){
        float const xOffset = i==0 ? 0 : 0.5;
        for(arma::uword j = 0 ; j < 6 ; ++j){
            float const yOffset = j==0 ? 0 : 0.5;
            arma::uword const rowIdx = i*6+j;
            X.at(rowIdx, 0) = i+1 + xOffset;
            X.at(rowIdx, 1) = j+1 + yOffset;
        }
    }
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{1, 1};
    arma::Col<int> const wD{3};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{2};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestB(){
    // Reference values
    std::map<int, int> const h0Ref{
        {9, 0},
        {10, 1},
        {11, 2},
        {17, 3},
        {18, 4},
        {19, 5},
        {25, 6},
        {26, 7},
        {50, 8},
        {57, 9},
        {58, 10},
        {59, 11},
        {65, 12},
        {66, 13},
        {67, 14},
        {74, 15},
        {75, 16},
        {76, 17},
        {83, 18},
        {84, 19},
        {85, 20},
        {86, 21}
    };
    std::map<int, int> const h1Ref{
        {6, 0},
        {7, 1},
        {11, 2},
        {12, 3},
        {16, 4},
        {17, 5},
        {21, 6},
        {22, 7},
        {26, 8},
        {27, 9},
        {28, 10}
    };
    std::vector<int> const hDref{
        0,  2,  16, 18, 32,
        34, 48, 50, 64, 66,
        68
    };
    std::vector<int> const hUref{
        0,  0,  0,  0,
        0,  0,  0,  0,
        10, 10, 10, 10,
        20, 20, 20, 20,
        20, 22, 20, 22,
        22, 22
    };
    // Generate points
    arma::Mat<float> X{
        {1, 1},
        {1.1, 2.1},
        {1.1, 3.1},
        {2.1, 1.1},
        {2.1, 2.1},
        {2.1, 3.1},
        {3.1, 1.1},
        {3.1, 2.1},
        {6.1, 2.1},
        {7.1, 1.1},
        {7.1, 2.1},
        {7.1, 3.1},
        {8.1, 1.1},
        {8.1, 2.1},
        {8.1, 3.1},
        {9.1, 2.1},
        {9.1, 3.1},
        {9.1, 4.1},
        {10.1, 3.1},
        {10.1, 4.1},
        {10.1, 5.1},
        {10.1, 6.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{1, 1};
    arma::Col<int> const wD{3};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{2};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestC(){
    // Reference values
    std::map<int, int> const h0Ref{
        {22, 0},
        {23, 1},
        {32, 2},
        {75, 3},
        {76, 4},
        {77, 5},
        {85, 6},
        {86, 7},
        {87, 8},
        {116, 9},
        {117, 10},
        {125, 11},
        {126, 12},
        {127, 13}
    };
    std::map<int, int> const h1Ref{
        {16, 0},
        {17, 1},
        {24, 2},
        {25, 3},
        {31, 4},
        {32, 5},
        {38, 6},
        {39, 7}
    };
    std::vector<int> const hDref{
        10, 13, 43, 46, 73,
        76, 103, 106
    };
    std::vector<int> const hUref{
        0,  0,  0,  16, 16,
        18, 16, 16, 18, 30,
        32, 30, 30, 32
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {7.1, 5.1},
        {7.1, 6.1},
        {7.1, 7.1},
        {8.1, 5.1},
        {8.1, 6.1},
        {8.1, 7.1},
        {11.1, 6.1},
        {11.1, 7.1},
        {12.1, 5.1},
        {12.1, 6.1},
        {12.1, 7.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{2, 2};
    arma::Col<int> const wD{4};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestD(){

    // Reference values
    std::map<int, int> const h0Ref{
        {39, 0},
        {40, 1},
        {51, 2},
        {102, 3},
        {103, 4},
        {104, 5},
        {114, 6},
        {115, 7},
        {116, 8},
        {151, 9},
        {152, 10},
        {162, 11},
        {163, 12},
        {164, 13}
    };
    std::map<int, int> const h1Ref{
        {16, 0},
        {17, 1},
        {23, 2},
        {24, 3},
        {31, 4},
        {32, 5},
        {38, 6},
        {39, 7},
        {45, 8},
        {46, 9}
    };
    std::vector<int> const hDref{
        1,      4,      37,     40,     76,
        79,     112,    115,    148,    151
    };
    std::vector<int> const hUref{
        0,      2,      0,      16,     16,
        18,     30,     30,     32,     30,
        32,     44,     44,     46
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {7.1, 5.1},
        {7.1, 6.1},
        {7.1, 7.1},
        {8.1, 5.1},
        {8.1, 6.1},
        {8.1, 7.1},
        {11.1, 6.1},
        {11.1, 7.1},
        {12.1, 5.1},
        {12.1, 6.1},
        {12.1, 7.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{3, 2};
    arma::Col<int> const wD{4};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}


// ***  VALIDATION METHODS  *** //
// **************************** //
template <typename IndexType>
bool HierarchicalSparseGridTest::validateSubmanifoldMap(
    std::map<IndexType, IndexType> const &hRef,
    std::map<IndexType, IndexType> const &h
) const{
    // Check dimensionality (i.e., number of pairs)
    if(hRef.size() != h.size()){
        return false;
    }
    // Check mutual inclusion (i.e., set equality)
    typename std::map<IndexType, IndexType>::const_iterator it;
    for(it = hRef.cbegin() ; it != hRef.cend() ; ++it){
        IndexType const key = it->first;
        if(it->second != h.at(key)){
            return false;
        }
    }
    // If this point is reached, then the given map is valid
    return true;
}

template <typename IndexType>
bool HierarchicalSparseGridTest::validateDownsamplingMap(
    std::vector<IndexType> const &hDref,
    std::vector<IndexType> const &hD
) const{
    return validateUpsamplingMap(hDref, hD);
}

template <typename IndexType>
bool HierarchicalSparseGridTest::validateUpsamplingMap(
    std::vector<IndexType> const &hUref,
    std::vector<IndexType> const &hU
) const {
    // Check dimensionality (i.e., number of components)
    if(hUref.size() != hU.size()){
        return false;
    }
    // Check component-wise equalities
    for(size_t i = 0 ; i < hUref.size () ; ++i){
        if(hUref[i] != hU[i]){
            return false;
        }
    }
    // If this point is reached, then the given map is valid
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestE(){
    // Reference values
    std::map<int, int> const h0Ref{
        {22, 0},
        {23, 1},
        {32, 2},
        {75, 3},
        {76, 4},
        {77, 5},
        {85, 6},
        {86, 7},
        {87, 8},
        {116, 9},
        {117, 10},
        {125, 11},
        {126, 12},
        {127, 13}
    };
    std::map<int, int> const h1Ref{
        {30, 0},
        {31, 1},
        {40, 2},
        {41, 3},
        {49, 4},
        {50, 5},
        {58, 6},
        {59, 7}
    };
    std::vector<int> const hDref{
        10, 13, 43, 46, 73,
        76, 103, 106
    };
    std::vector<int> const hUref{
        0,  2,  0,  22, 22,
        22, 40, 40, 40, 40,
        40, 58, 58, 58
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {7.1, 5.1},
        {7.1, 6.1},
        {7.1, 7.1},
        {8.1, 5.1},
        {8.1, 6.1},
        {8.1, 7.1},
        {11.1, 6.1},
        {11.1, 7.1},
        {12.1, 5.1},
        {12.1, 6.1},
        {12.1, 7.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{2, 3};
    arma::Col<int> const wD{4};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestF(){
    // Reference values
    std::map<int, int> const h0Ref{
        {39, 0},
        {40, 1},
        {51, 2},
        {102, 3},
        {103, 4},
        {104, 5},
        {114, 6},
        {115, 7},
        {116, 8},
        {151, 9},
        {152, 10},
        {162, 11},
        {163, 12},
        {164, 13}
    };
    std::map<int, int> const h1Ref{
        {30, 0},
        {31, 1},
        {39, 2},
        {40, 3},
        {49, 4},
        {50, 5},
        {58, 6},
        {59, 7},
        {67, 8},
        {68, 9}
    };
    std::vector<int> const hDref{
        1,      4,      37,     40,     76,
        79,     112,    115,    148,    151
    };
    std::vector<int> const hUref{
        2,  2,  20, 40, 40,
        40, 40, 40, 40, 58,
        58, 58, 58, 58
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {7.1, 5.1},
        {7.1, 6.1},
        {7.1, 7.1},
        {8.1, 5.1},
        {8.1, 6.1},
        {8.1, 7.1},
        {11.1, 6.1},
        {11.1, 7.1},
        {12.1, 5.1},
        {12.1, 6.1},
        {12.1, 7.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{3, 3};
    arma::Col<int> const wD{4};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestG(){
    // Reference values
    std::map<int, int> const h0Ref{
        {60, 0},
        {61, 1},
        {74, 2},
        {133, 3},
        {134, 4},
        {135, 5},
        {147, 6},
        {148, 7},
        {149, 8},
        {190, 9},
        {191, 10},
        {203, 11},
        {204, 12},
        {205, 13}
    };
    std::map<int, int> const h1Ref{
        {65, 0},
        {78, 1},
        {79, 2},
        {90, 3},
        {91, 4},
        {102, 5},
        {103, 6}
    };
    std::vector<int> const hDref{
        45,     90,     93,     132,        135,
        174,    177
    };
    std::vector<int> const hUref{
        26,     26,     26,     52,     52,
        54,     76,     76,     78,     100,
        102,    100,    100,    102
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {7.1, 5.1},
        {7.1, 6.1},
        {7.1, 7.1},
        {8.1, 5.1},
        {8.1, 6.1},
        {8.1, 7.1},
        {11.1, 6.1},
        {11.1, 7.1},
        {12.1, 5.1},
        {12.1, 6.1},
        {12.1, 7.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{4, 4};
    arma::Col<int> const wD{4};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestH(){
    // Reference values
    std::map<int, int> const h0Ref{
        {30, 0},
        {31, 1},
        {44, 2},
        {94, 3},
        {95, 4},
        {100, 5},
        {101, 6},
        {102, 7},
        {108, 8},
        {109, 9},
        {149, 10},
        {150, 11},
        {151, 12},
        {163, 13},
        {164, 14},
        {165, 15},
        {200, 16},
        {202, 17},
        {214, 18},
        {215, 19},
        {216, 20},
        {221, 21},
        {228, 22},
        {229, 23},
        {230, 24},
        {234, 25},
        {235, 26}
    };
    std::map<int, int> const h1Ref{
        {22, 0},
        {23, 1},
        {32, 2},
        {33, 3},
        {46, 4},
        {47, 5},
        {52, 6},
        {53, 7},
        {54, 8},
        {56, 9},
        {57, 10},
        {66, 11},
        {67, 12},
        {76, 13},
        {77, 14},
        {83, 15},
        {84, 16},
        {85, 17},
        {93, 18},
        {94, 19},
        {95, 20},
        {96, 21},
        {97, 22},
        {103, 23},
        {104, 24},
        {105, 25},
        {106, 26},
        {107, 27}
    };
    std::map<int, int> const h2Ref{
        {18, 0},
        {19, 1},
        {26, 2},
        {27, 3},
        {28, 4},
        {29, 5},
        {34, 6},
        {35, 7},
        {36, 8},
        {37, 9},
        {43, 10},
        {44, 11},
        {45, 12},
        {51, 13},
        {52, 14},
        {53, 15},
        {59, 16},
        {60, 17},
        {61, 18}
    };
    std::vector<int> const hD0ref{
        0,      2,      28,     30,     64,
        66,     84,     86,     88,     92,
        94,     120,    122,    148,    150,
        170,    172,    174,    198,    200,
        202,    204,    206,    226,    228,
        230,    232,    234
    };
    std::vector<int> const hD1ref{
        0,      2,      20,     22,     24,
        26,     40,     42,     44,     46,
        62,     64,     66,     82,     84,
        86,     102,    104,    106
    };
    std::vector<int> const hU0ref{
        0,      0,      0,      24,     26,
        40,     40,     42,     44,     46,
        64,     64,     66,     64,     64,
        66,     82,     82,     82,     82,
        82,     86,     102,    102,    102,
        104,    106
    };
    std::vector<int> const hU1ref{
        0,      0,      0,      0,      18,
        20,     16,     16,     18,     18,
        20,     18,     20,     34,     36,
        32,     34,     34,     32,     34,
        34,     34,     36,     48,     50,
        50,     50,     52
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {2.1, 3.1},
        {3.1, 2.1},
        {6.1, 10.1},
        {6.1, 11.1},
        {7.1, 2.1},
        {7.1, 3.1},
        {7.1, 4.1},
        {7.1, 10.1},
        {7.1, 11.1},
        {10.1, 9.1},
        {10.1, 10.1},
        {10.1, 11.1},
        {11.1, 9.1},
        {11.1, 10.1},
        {11.1, 11.1},
        {14.1, 4.1},
        {14.1, 6.1},
        {15.1, 4.1},
        {15.1, 5.1},
        {15.1, 6.1},
        {15.1, 11.1},
        {16.1, 4.1},
        {16.1, 5.1},
        {16.1, 6.1},
        {16.1, 10.1},
        {16.1, 11.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{2, 2, 2};
    arma::Col<int> const wD{3, 3};
    arma::Col<int> const wU{3, 3};
    arma::Col<int> const sD{2, 2};
    arma::Col<int> const sU{2, 2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateSubmanifoldMap(h2Ref, hsg.getMap(2))) return false;
    if(!validateDownsamplingMap(hD0ref, hsg.getDownsamplingMap(0))){
        return false;
    }
    if(!validateDownsamplingMap(hD1ref, hsg.getDownsamplingMap(1))){
        return false;
    }
    if(!validateUpsamplingMap(hU0ref, hsg.getUpsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hU1ref, hsg.getUpsamplingMap(1))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestI(){
    // Reference values
    std::map<int, int> const h0Ref{
        {9, 0},
        {10, 1},
        {11, 2},
        {17, 3},
        {18, 4},
        {19, 5},
        {25, 6},
        {26, 7},
        {50, 8},
        {57, 9},
        {58, 10},
        {59, 11},
        {65, 12},
        {66, 13},
        {67, 14},
        {74, 15},
        {75, 16},
        {76, 17},
        {83, 18},
        {84, 19},
        {85, 20},
        {86, 21}
    };
    std::map<int, int> const h1Ref{
        {6, 0},
        {7, 1},
        {11, 2},
        {12, 3},
        {16, 4},
        {17, 5},
        {21, 6},
        {22, 7},
        {26, 8},
        {27, 9},
        {28, 10}
    };
    std::vector<int> const hDref{
        0,  2,  16, 18, 32,
        34, 48, 50, 64, 66,
        68
    };
    std::vector<int> const hUref{
        0,  0,  0,  0,
        0,  0,  0,  0,
        10, 10, 10, 10,
        20, 20, 20, 20,
        20, 22, 20, 22,
        22, 22
    };
    // Generate points
    arma::Mat<float> X{
        {1.1, 2.1},
        {1.1, 3.1},
        {6.1, 2.1},
        {7.1, 1.1},
        {7.1, 2.1},
        {7.1, 3.1},
        {8.1, 1.1},
        {8.1, 2.1},
        {8.1, 3.1},
        {9.1, 2.1},
        {9.1, 3.1},
        {9.1, 4.1},
        {1, 1},
        {10.1, 3.1},
        {10.1, 4.1},
        {10.1, 5.1},
        {10.1, 6.1},
        {2.1, 1.1},
        {2.1, 2.1},
        {2.1, 3.1},
        {3.1, 1.1},
        {3.1, 2.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{1, 1};
    arma::Col<int> const wD{3};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{2};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestJ(){
    // Reference values
    std::map<int, int> const h0Ref{
        {30, 0},
        {31, 1},
        {44, 2},
        {94, 3},
        {95, 4},
        {100, 5},
        {101, 6},
        {102, 7},
        {108, 8},
        {109, 9},
        {149, 10},
        {150, 11},
        {151, 12},
        {163, 13},
        {164, 14},
        {165, 15},
        {200, 16},
        {202, 17},
        {214, 18},
        {215, 19},
        {216, 20},
        {221, 21},
        {228, 22},
        {229, 23},
        {230, 24},
        {234, 25},
        {235, 26}
    };
    std::map<int, int> const h1Ref{
        {22, 0},
        {23, 1},
        {32, 2},
        {33, 3},
        {46, 4},
        {47, 5},
        {52, 6},
        {53, 7},
        {54, 8},
        {56, 9},
        {57, 10},
        {66, 11},
        {67, 12},
        {76, 13},
        {77, 14},
        {83, 15},
        {84, 16},
        {85, 17},
        {93, 18},
        {94, 19},
        {95, 20},
        {96, 21},
        {97, 22},
        {103, 23},
        {104, 24},
        {105, 25},
        {106, 26},
        {107, 27}
    };
    std::map<int, int> const h2Ref{
        {18, 0},
        {19, 1},
        {26, 2},
        {27, 3},
        {28, 4},
        {29, 5},
        {34, 6},
        {35, 7},
        {36, 8},
        {37, 9},
        {43, 10},
        {44, 11},
        {45, 12},
        {51, 13},
        {52, 14},
        {53, 15},
        {59, 16},
        {60, 17},
        {61, 18}
    };
    std::vector<int> const hD0ref{
        0,      2,      28,     30,     64,
        66,     84,     86,     88,     92,
        94,     120,    122,    148,    150,
        170,    172,    174,    198,    200,
        202,    204,    206,    226,    228,
        230,    232,    234
    };
    std::vector<int> const hD1ref{
        0,      2,      20,     22,     24,
        26,     40,     42,     44,     46,
        62,     64,     66,     82,     84,
        86,     102,    104,    106
    };
    std::vector<int> const hU0ref{
        0,      0,      0,      24,     26,
        40,     40,     42,     44,     46,
        64,     64,     66,     64,     64,
        66,     82,     82,     82,     82,
        82,     86,     102,    102,    102,
        104,    106
    };
    std::vector<int> const hU1ref{
        0,      0,      0,      0,      18,
        20,     16,     16,     18,     18,
        20,     18,     20,     34,     36,
        32,     34,     34,     32,     34,
        34,     34,     36,     48,     50,
        50,     50,     52
    };
    // Generate points
    arma::Mat<float> X{
        {2, 2},
        {6.1, 10.1},
        {6.1, 11.1},
        {10.1, 9.1},
        {10.1, 10.1},
        {10.1, 11.1},
        {14.1, 4.1},
        {14.1, 6.1},
        {15.1, 4.1},
        {15.1, 5.1},
        {15.1, 6.1},
        {15.1, 11.1},
        {16.1, 4.1},
        {16.1, 5.1},
        {16.1, 6.1},
        {16.1, 10.1},
        {16.1, 11.1},
        {2.1, 3.1},
        {3.1, 2.1},
        {11.1, 9.1},
        {11.1, 10.1},
        {11.1, 11.1},
        {7.1, 2.1},
        {7.1, 3.1},
        {7.1, 4.1},
        {7.1, 10.1},
        {7.1, 11.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{2, 2, 2};
    arma::Col<int> const wD{3, 3};
    arma::Col<int> const wU{3, 3};
    arma::Col<int> const sD{2, 2};
    arma::Col<int> const sU{2, 2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateSubmanifoldMap(h2Ref, hsg.getMap(2))) return false;
    if(!validateDownsamplingMap(hD0ref, hsg.getDownsamplingMap(0))){
        return false;
    }
    if(!validateDownsamplingMap(hD1ref, hsg.getDownsamplingMap(1))){
        return false;
    }
    if(!validateUpsamplingMap(hU0ref, hsg.getUpsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hU1ref, hsg.getUpsamplingMap(1))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple2DTestK(){
    // Reference values
    std::map<int, int> const h0Ref{
        {22, 0},
        {23, 1},
        {24, 2},
        {32, 3},
        {33, 4},
        {34, 5},
        {42, 6},
        {43, 7},
        {73, 8},
        {82, 9},
        {83, 10},
        {84, 11},
        {92, 12},
        {93, 13},
        {94, 14},
        {103, 15},
        {104, 16},
        {105, 17},
        {114, 18},
        {115, 19},
        {116, 20},
        {117, 21}
    };
    std::map<int, int> const h1Ref{
        {24, 0},
        {31, 1}
    };
    std::vector<int> const hDref{
        0,  30
    };
    std::vector<int> const hUref{
        0,      0,      2,      0,      0,
        2,      0,      0,      14,     14,
        14,     16,     14,     14,     16,
        28,     30,     30,     30,     30,
        30,     32
    };
    // Generate points
    arma::Mat<float> X{
        {1, 1},
        {1.1, 2.1},
        {1.1, 3.1},
        {2.1, 1.1},
        {2.1, 2.1},
        {2.1, 3.1},
        {3.1, 1.1},
        {3.1, 2.1},
        {6.1, 2.1},
        {7.1, 1.1},
        {7.1, 2.1},
        {7.1, 3.1},
        {8.1, 1.1},
        {8.1, 2.1},
        {8.1, 3.1},
        {9.1, 2.1},
        {9.1, 3.1},
        {9.1, 4.1},
        {10.1, 3.1},
        {10.1, 4.1},
        {10.1, 5.1},
        {10.1, 6.1}
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{1, 1};
    arma::Col<int> const wD{10};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{3};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}

bool HierarchicalSparseGridTest::simple3DTestA(){
    // Reference values
    std::map<int, int> const h0Ref{
        {73, 0},
        {74, 1},
        {82, 2},
        {83, 3},
        {136, 4},
        {137, 5},
        {346, 6},
        {556, 7},
        {610, 8},
        {619, 9}
    };
    std::map<int, int> const h1Ref{
        {37, 0},
        {38, 1},
        {43, 2},
        {44, 3},
        {67, 4},
        {68, 5},
        {104, 6},
        {105, 7},
        {142, 8},
        {166, 9},
        {172, 10}
    };
    std::vector<int> const hDref{
        0,      2,      18,     20,     126,
        128,    272,    274,    420,    528,
        546
    };
    std::vector<int> const hUref{
        0,      0,      0,      0,      0,
        0,      60,     134,    134,    134
    };
    // Generate points
    arma::Mat<float> X{
        {1, 1, 1},          // 73
        {1.1, 1.1, 2.1},    // 74
        {1.1, 2.1, 1.1},    // 82
        {1.1, 2.1, 2.1},    // 83
        {2.1, 1.1, 1.1},    // 136
        {2.1, 1.1, 2.1},    // 137
        {5.1, 3.1, 4.1},    // 346
        {8.1, 5.1, 7.1},    // 556
        {9.1, 4.1, 7.1},    // 610
        {9.1, 5.1, 7.1}     // 619
    };
    // Build hierarchical sparse grid
    float const size = 1.0;
    arma::Col<int> const w{1, 1};
    arma::Col<int> const wD{3};
    arma::Col<int> const wU{3};
    arma::Col<int> const sD{2};
    arma::Col<int> const sU{2};
    HierarchicalSparseGrid<float, int> hsg(size, w, wD, wU, sD, sU, 1);
    hsg.fit(X);
    // Validate results
    if(!validateSubmanifoldMap(h0Ref, hsg.getMap(0))) return false;
    if(!validateSubmanifoldMap(h1Ref, hsg.getMap(1))) return false;
    if(!validateDownsamplingMap(hDref, hsg.getDownsamplingMap(0))) return false;
    if(!validateUpsamplingMap(hUref, hsg.getUpsamplingMap(0))) return false;
    // On test passed
    return true;
}
