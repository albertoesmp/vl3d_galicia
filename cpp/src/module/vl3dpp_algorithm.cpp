#include <pybind11/pybind11.h>
#include <module/AlgorithmModule.hpp>

void vl3dpp_algorithm_init(py::module &m){
    // ***  ALGORITHM COMPONENTS  *** //
    // ****************************** //
    // Support neighborhoods
    m.def(
        "alg_support_neighborhoods_fs32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du32u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u32, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fs16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u16, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fs16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::s8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_ds16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::s8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_fu16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<float, arma::u8, arma::u32>,
        "Compute support neighborhoods."
    );
    m.def(
        "alg_support_neighborhoods_du16u32",
        &vl3dpp::pymods::alg_support_neighborhoods<double, arma::u8, arma::u32>,
        "Compute support neighborhoods."
    );
    // Oversampler
    m.def(
        "alg_oversampler_du32",
        &vl3dpp::pymods::alg_oversampler<double, arma::u32>,
        "Compute oversampling."
    );
    m.def(
        "alg_oversampler_fu32",
        &vl3dpp::pymods::alg_oversampler<float, arma::u32>,
        "Compute oversampling."
    );
}
