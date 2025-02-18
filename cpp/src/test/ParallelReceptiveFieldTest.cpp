#include <test/ParallelReceptiveFieldTest.hpp>
#include <util/VL3DPPException.hpp>
#include <rfield/DLFPSPostProcessor.hpp>

#include <armadillo>

#include <vector>


using namespace vl3dpp::test;
using namespace vl3dpp::rfield;
using vl3dpp::util::VL3DPPException;


// ***   R U N   *** //
// ***************** //
bool vl3dpp::test::ParallelReceptiveFieldTest::run(){
    if(!doPostProcessingTests()) return false;
    // All tests passed
    return true;
}

// ***  TEST METHODS  *** //
// ********************** //
bool ParallelReceptiveFieldTest::doPostProcessingTests(){
    // Prepare input data
    arma::uword bs = 32; // Input cases
    arma::uword bsClean = 28; // Input cases after cleaning
    arma::uword m = 128; // Num points
    arma::Col<arma::u16> K = arma::randi<arma::Col<arma::u16>>( // Neighs/case
        bsClean, arma::distr_param(8, 16)
    );
    arma::uword R = 4; // Points per receptive field
    arma::uword nc = 3; // Number of features (e.g., probabilities)
    arma::Cube<float> zBatch = arma::randn<arma::Cube<float>>(
        bs, R, nc, arma::distr_param(0.0, 1.0)
    );
    std::vector<arma::Mat<arma::u16>> MBatch(bsClean); // Encoding neighborhoods
    std::vector<arma::Col<arma::u16>> I(bsClean); // Decoding neighborhoods
    for(size_t k = 0 ; k < bsClean ; ++k){
        MBatch[k] = arma::randi<arma::Mat<arma::u16>>(
            m, K[k], arma::distr_param(0, R-1)
        );
        I[k] = arma::randi<arma::Col<arma::u16>>(
            K[k], arma::distr_param(0, m-1)
        );
    }
    // Run DLFPSPostProcessor : Mean
    DLFPSPostProcessor<float, uint16_t, uint16_t> dlfpspp(
        "mean_reduce", 1e-6, -1
    );
    arma::Mat<float> z = dlfpspp(m, MBatch, I, zBatch);
    DLFPSPostProcessor<float, uint16_t, uint16_t> dlfpspp_seq(
        "mean_reduce", 1e-6, 1
    );
    arma::Mat<float> z_seq = dlfpspp(m, MBatch, I, zBatch);
    if(arma::any(arma::vectorise(arma::abs(z-z_seq)) > this->eps)) return false;
    // Run DLFPSPostProcessor : Sum
    dlfpspp = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "sum_reduce", 1e-6, -1
    );
    z = dlfpspp(m, MBatch, I, zBatch);
    dlfpspp_seq = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "sum_reduce", 1e-6, 1
    );
    z_seq = dlfpspp_seq(m, MBatch, I, zBatch);
    if(arma::any(arma::vectorise(arma::abs(z-z_seq)) > this->eps)) return false;
    // Run DLFPSPostProcessor : Max
    dlfpspp = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "max_reduce", 1e-6, -1
    );
    z = dlfpspp(m, MBatch, I, zBatch);
    dlfpspp_seq = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "max_reduce", 1e-6, 1
    );
    z_seq = dlfpspp_seq(m, MBatch, I, zBatch);
    if(arma::any(arma::vectorise(arma::abs(z-z_seq)) > this->eps)) return false;
    // Run DLFPSPostProcessor : Entropic
    dlfpspp = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "entropic_reduce", 1e-6,-1
    );
    z = dlfpspp(m, MBatch, I, zBatch);
    dlfpspp_seq = DLFPSPostProcessor<float, uint16_t, uint16_t>(
        "entropic_reduce", 1e-6, 1
    );
    z_seq = dlfpspp_seq(m, MBatch, I, zBatch);
    if(arma::any(arma::vectorise(arma::abs(z-z_seq)) > this->eps)) return false;
    // All post-processing tests passed
    return true;
}
