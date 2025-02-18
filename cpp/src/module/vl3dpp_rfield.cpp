#include <pybind11/pybind11.h>
#include <module/ReceptiveFieldModule.hpp>

void vl3dpp_rfield_init(py::module &m){
    // ***  RECEPTIVE FIELD COMPONENTS  *** //
    // ************************************ //
    // Receptive field propagations
    m.def(
        "rf_propagate_mean_f8",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u8>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f16",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u16>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f32",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u32>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_f64",
        &vl3dpp::pymods::rf_propagate_mean<float, arma::u64>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d8",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u8>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d16",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u16>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d32",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u32>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_mean_d64",
        &vl3dpp::pymods::rf_propagate_mean<double, arma::u64>,
        "Propagate mean value."
    );
    m.def(
        "rf_propagate_closest_f8",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u8>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f16",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u16>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f32",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u32>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_f64",
        &vl3dpp::pymods::rf_propagate_closest<float, arma::u64>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d8",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u8>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d16",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u16>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d32",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u32>,
        "Propagate closest value."
    );
    m.def(
        "rf_propagate_closest_d64",
        &vl3dpp::pymods::rf_propagate_closest<double, arma::u64>,
        "Propagate closest value."
    );
    // Receptive field reductions
    m.def(
        "rf_reduce_mean_f8",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u8>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f16",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u16>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f32",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u32>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_f64",
        &vl3dpp::pymods::rf_reduce_mean<float, arma::u64>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d8",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u8>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d16",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u16>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d32",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u32>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_mean_d64",
        &vl3dpp::pymods::rf_reduce_mean<double, arma::u64>,
        "Reduce mean value."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u8u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u8, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s8u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s8, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u16u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u16, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s16u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s16, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u32u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u32, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s32u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s32, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u8u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u8, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u16u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u16, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u32u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u32, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u8",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u8>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u16",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u16>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u32",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u32>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_u64u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::u64, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_reduce_label_mode_s64u64u64",
        &vl3dpp::pymods::rf_reduce_label_mode<arma::s64, arma::u64, arma::u64>,
        "Reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xds32s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::s32, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xds16s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::s16, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xds8s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::s8, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xdu32s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::u32, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xdu16s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::u16, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xdu8s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            double, arma::u8, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfs32s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::s32, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfs16s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::s16, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfs8s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::s8, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfu32s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::u32, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfu16s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::u16, arma::s32
        >,
        "Sparse reduce label mode."
    );
    m.def(
        "rf_sparse_reduce_label_mode_Xfu8s32",
        &vl3dpp::pymods::rf_sparse_reduce_label_mode<
            float, arma::u8, arma::s32
        >,
        "Sparse reduce label mode."
    );
    // Deep learning's receptive field pre-processing
    m.def(
        "rf_dl_fps_preproc_Xdf_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            double, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xff_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            float, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xdd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            double, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_fps_preproc_Xfd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_fps_preproc<
            float, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xdf_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            double, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xff_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            float, float, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xdd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            double, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    m.def(
        "rf_dl_hfps_preproc_Xfd_Ff_Iu32u32_ys32",
        &vl3dpp::pymods::rf_dl_hfps_preproc<
            float, double, float, arma::u32, arma::u32, arma::s32
        >,
        "Deep learning FPS pre-processing."
    );
    // Deep learning's receptive field post-processing
    m.def(
        "rf_dl_fps_postproc_mean_fu8u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu8u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u8, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu16u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u16, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu32u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u32, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_fu64u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<float, arma::u64, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du8u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u8, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du16u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u16, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du32u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u32, arma::u64>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u8",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u8>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u16",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u16>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u32",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u32>,
        "Deep learning FPS post-processing."
    );
    m.def(
        "rf_dl_fps_postproc_mean_du64u64",
        &vl3dpp::pymods::rf_dl_fps_postproc_mean<double, arma::u64, arma::u64>,
        "Deep learning FPS post-processing."
    );
    // Deep learning's sparse receptive field fit
    m.def(
        "rf_dl_sg_fit_ds32",
        &vl3dpp::pymods::rf_dl_sg_fit<double, arma::s32>,
        "Deep learning SG fit."
    );
    m.def(
        "rf_dl_sg_fit_du32",
        &vl3dpp::pymods::rf_dl_sg_fit<double, arma::u32>,
        "Deep learning SG fit."
    );
    m.def(
        "rf_dl_sg_fit_fs32",
        &vl3dpp::pymods::rf_dl_sg_fit<float, arma::s32>,
        "Deep learning SG fit."
    );
    m.def(
        "rf_dl_sg_fit_fu32",
        &vl3dpp::pymods::rf_dl_sg_fit<float, arma::u32>,
        "Deep learning SG fit."
    );
    // Deep learning's hierarchical sparse receptive field pre-processing
    m.def(
        "rf_dl_hsg_preproc_Xd_Ff_Is32_ys32",
        &vl3dpp::pymods::rf_dl_hsg_preproc<double, float, arma::s32, arma::s32>,
        "Deep learning HSG pre-processing."
    );
    m.def(
        "rf_dl_hsg_preproc_Xf_Ff_Is32_ys32",
        &vl3dpp::pymods::rf_dl_hsg_preproc<double, float, arma::s32, arma::s32>,
        "Deep learning HSG pre-processing."
    );
    // Deep learning's sparse receptive field post-processing
    m.def(
        "rf_dl_sg_postproc_mean_Xd_Fd_Is32",
        &vl3dpp::pymods::rf_dl_sg_postproc_mean<double, double, arma::s32>,
        "Deep learning SG post-processing."
    );
    m.def(
        "rf_dl_sg_postproc_mean_Xd_Ff_Is32",
        &vl3dpp::pymods::rf_dl_sg_postproc_mean<double, float, arma::s32>,
        "Deep learning SG post-processing."
    );
    m.def(
        "rf_dl_sg_postproc_mean_Xf_Fd_Is32",
        &vl3dpp::pymods::rf_dl_sg_postproc_mean<float, double, arma::s32>,
        "Deep learning SG post-processing."
    );
    m.def(
        "rf_dl_sg_postproc_mean_Xf_Ff_Is32",
        &vl3dpp::pymods::rf_dl_sg_postproc_mean<float, float, arma::s32>,
        "Deep learning SG post-processing."
    );
}