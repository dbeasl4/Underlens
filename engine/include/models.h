#pragma once
#include <vector>
#include <string>
#include <map>

struct FitResult {
    std::string model_name;
    std::map<std::string, double> params;
    double residual_error;
    double aic_score;
};

FitResult fit_logistic(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_exponential(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_power_law(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_sinusoidal(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_gaussian(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_gompertz(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_damped_sinusoidal(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_linear(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_quadratic(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_cubic(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_logarithmic(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_double_exponential(
    const std::vector<double>& t,
    const std::vector<double>& y
);

FitResult fit_michaelis_menten(
    const std::vector<double>& x,
    const std::vector<double>& y
);

FitResult fit_best_model(
    const std::vector<double>& x,
    const std::vector<double>& y
);