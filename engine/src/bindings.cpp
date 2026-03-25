#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "models.h"

namespace py = pybind11;

PYBIND11_MODULE(underlens_engine, m) {
    m.doc() = "Underlens C++ fitting engine";

    py::class_<FitResult>(m, "FitResult")
        .def_readonly("model_name",     &FitResult::model_name)
        .def_readonly("params",         &FitResult::params)
        .def_readonly("residual_error", &FitResult::residual_error)
        .def_readonly("aic_score",      &FitResult::aic_score)
        .def("__repr__", [](const FitResult& r) {
            return "<FitResult model='" + r.model_name +
                   "' aic=" + std::to_string(r.aic_score) + ">";
        });

    m.def("fit_logistic",          &fit_logistic,          "Fit logistic growth");
    m.def("fit_exponential",       &fit_exponential,       "Fit exponential");
    m.def("fit_power_law",         &fit_power_law,         "Fit power law");
    m.def("fit_sinusoidal",        &fit_sinusoidal,        "Fit sinusoidal wave");
    m.def("fit_gaussian",          &fit_gaussian,          "Fit gaussian");
    m.def("fit_gompertz",          &fit_gompertz,          "Fit Gompertz growth");
    m.def("fit_damped_sinusoidal", &fit_damped_sinusoidal, "Fit damped sinusoidal");
    m.def("fit_linear",            &fit_linear,            "Fit linear");
    m.def("fit_quadratic",         &fit_quadratic,         "Fit quadratic");
    m.def("fit_cubic",             &fit_cubic,             "Fit cubic polynomial");
    m.def("fit_logarithmic",       &fit_logarithmic,       "Fit logarithmic growth");
    m.def("fit_double_exponential",&fit_double_exponential,"Fit double exponential decay");
    m.def("fit_michaelis_menten",  &fit_michaelis_menten,  "Fit Michaelis-Menten");
    m.def("fit_best_model",        &fit_best_model,        "Fit best model by AIC");
}