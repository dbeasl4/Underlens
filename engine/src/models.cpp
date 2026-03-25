#include "models.h"
#include <ceres/ceres.h>
#include <cmath>
#include <limits>
#include <algorithm>

// ── Cost functors ─────────────────────────────────────────────────────────────

struct LogisticResidual {
    LogisticResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const L,
                    const T* const k,
                    const T* const t0,
                    T* residual) const {
        T predicted = *L / (T(1.0) + exp(-(*k) * (T(t_) - *t0)));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct ExponentialResidual {
    ExponentialResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const a,
                    const T* const b,
                    T* residual) const {
        T predicted = *a * exp(*b * T(t_));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct PowerLawResidual {
    PowerLawResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const a,
                    const T* const b,
                    T* residual) const {
        T predicted = *a * pow(abs(T(x_)), *b);
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct SinusoidalResidual {
    SinusoidalResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const A,
                    const T* const freq,
                    const T* const phi,
                    const T* const C,
                    T* residual) const {
        T predicted = *A * sin(T(2.0 * M_PI) * *freq * T(t_) + *phi) + *C;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct GaussianResidual {
    GaussianResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const A,
                    const T* const mu,
                    const T* const sigma,
                    T* residual) const {
        T diff = T(x_) - *mu;
        T predicted = *A * exp(-(diff * diff) / (T(2.0) * *sigma * *sigma));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct GompertzResidual {
    GompertzResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const A,
                    const T* const b,
                    const T* const c,
                    T* residual) const {
        T predicted = *A * exp(-(*b) * exp(-(*c) * T(t_)));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct DampedSinusoidalResidual {
    DampedSinusoidalResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const A,
                    const T* const d,
                    const T* const freq,
                    const T* const phi,
                    const T* const C,
                    T* residual) const {
        T predicted = *A * exp(-(*d) * T(t_)) *
                      sin(T(2.0 * M_PI) * *freq * T(t_) + *phi) + *C;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct LinearResidual {
    LinearResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const m,
                    const T* const b,
                    T* residual) const {
        T predicted = *m * T(x_) + *b;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct QuadraticResidual {
    QuadraticResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const a,
                    const T* const b,
                    const T* const c,
                    T* residual) const {
        T xi = T(x_);
        T predicted = *a * xi * xi + *b * xi + *c;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct LogarithmicResidual {
    LogarithmicResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const a,
                    const T* const b,
                    T* residual) const {
        T predicted = *a * log(T(x_)) + *b;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct CubicResidual {
    CubicResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const a,
                    const T* const b,
                    const T* const c,
                    const T* const d,
                    T* residual) const {
        T xi = T(x_);
        T predicted = *a * xi*xi*xi + *b * xi*xi + *c * xi + *d;
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

struct DoubleExponentialResidual {
    DoubleExponentialResidual(double t, double y) : t_(t), y_(y) {}
    template <typename T>
    bool operator()(const T* const A,
                    const T* const b,
                    const T* const C,
                    const T* const d,
                    T* residual) const {
        T predicted = *A * exp(-(*b) * T(t_)) + *C * exp(-(*d) * T(t_));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double t_, y_;
};

struct MichaelisMentenResidual {
    MichaelisMentenResidual(double x, double y) : x_(x), y_(y) {}
    template <typename T>
    bool operator()(const T* const Vmax,
                    const T* const Km,
                    T* residual) const {
        T predicted = *Vmax * T(x_) / (*Km + T(x_));
        residual[0] = T(y_) - predicted;
        return true;
    }
    double x_, y_;
};

// ── AIC score ─────────────────────────────────────────────────────────────────

double compute_aic(double residual, int n, int k) {
    double mse = residual / n;
    double log_likelihood = -n * 0.5 * log(2 * M_PI * mse) - n / 2.0;
    return 2.0 * k - 2.0 * log_likelihood;
}

// ── Fit functions ─────────────────────────────────────────────────────────────

FitResult fit_logistic(const std::vector<double>& t,
                       const std::vector<double>& y) {
    double L  = *std::max_element(y.begin(), y.end());
    double k  = 1.0;
    double t0 = t[t.size() / 2];

    ceres::Problem problem;
    for (size_t i = 0; i < t.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LogisticResidual, 1, 1, 1, 1>(
                new LogisticResidual(t[i], y[i])),
            nullptr, &L, &k, &t0);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "logistic_growth";
    result.params["L"]  = L;
    result.params["k"]  = k;
    result.params["t0"] = t0;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, t.size(), 3);
    return result;
}

FitResult fit_exponential(const std::vector<double>& t,
                          const std::vector<double>& y) {
    double a = y[0];
    double b = 0.1;

    ceres::Problem problem;
    for (size_t i = 0; i < t.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
                new ExponentialResidual(t[i], y[i])),
            nullptr, &a, &b);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "exponential";
    result.params["a"] = a;
    result.params["b"] = b;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, t.size(), 2);
    return result;
}

FitResult fit_power_law(const std::vector<double>& x,
                        const std::vector<double>& y) {
    double a = 1.0;
    double b = 1.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] <= 0) continue;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<PowerLawResidual, 1, 1, 1>(
                new PowerLawResidual(x[i], y[i])),
            nullptr, &a, &b);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "power_law";
    result.params["a"] = a;
    result.params["b"] = b;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 2);
    return result;
}

FitResult fit_sinusoidal(const std::vector<double>& t,
                         const std::vector<double>& y) {
    double y_max = *std::max_element(y.begin(), y.end());
    double y_min = *std::min_element(y.begin(), y.end());
    double A = (y_max - y_min) / 2.0;
    double C = (y_max + y_min) / 2.0;

    std::vector<double> freq_guesses = {0.1, 0.5, 1.0, 2.0, 5.0};
    FitResult best_result;
    best_result.aic_score = std::numeric_limits<double>::max();

    for (double freq_init : freq_guesses) {
        double A_try    = A;
        double freq_try = freq_init;
        double phi_try  = 0.0;
        double C_try    = C;

        ceres::Problem problem;
        for (size_t i = 0; i < t.size(); i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<SinusoidalResidual, 1, 1, 1, 1, 1>(
                    new SinusoidalResidual(t[i], y[i])),
                nullptr, &A_try, &freq_try, &phi_try, &C_try);
        }
        problem.SetParameterLowerBound(&freq_try, 0, 0.001);
        problem.SetParameterLowerBound(&A_try,    0, 0.0);

        ceres::Solver::Options options;
        options.max_num_iterations = 500;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double aic = compute_aic(summary.final_cost, t.size(), 4);
        if (aic < best_result.aic_score) {
            best_result.model_name = "sinusoidal";
            best_result.params["A"]    = A_try;
            best_result.params["freq"] = freq_try;
            best_result.params["phi"]  = phi_try;
            best_result.params["C"]    = C_try;
            best_result.residual_error = summary.final_cost;
            best_result.aic_score      = aic;
        }
    }
    return best_result;
}

FitResult fit_gaussian(const std::vector<double>& x,
                       const std::vector<double>& y) {
    auto max_it  = std::max_element(y.begin(), y.end());
    int  max_idx = std::distance(y.begin(), max_it);

    double A     = *max_it;
    double mu    = x[max_idx];
    double sigma = (x.back() - x.front()) / 6.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GaussianResidual, 1, 1, 1, 1>(
                new GaussianResidual(x[i], y[i])),
            nullptr, &A, &mu, &sigma);
    }
    problem.SetParameterLowerBound(&sigma, 0, 0.001);

    ceres::Solver::Options options;
    options.max_num_iterations = 400;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "gaussian";
    result.params["A"]     = A;
    result.params["mu"]    = mu;
    result.params["sigma"] = sigma;
    result.residual_error  = summary.final_cost;
    result.aic_score       = compute_aic(summary.final_cost, x.size(), 3);
    return result;
}

FitResult fit_gompertz(const std::vector<double>& t,
                       const std::vector<double>& y) {
    double A = *std::max_element(y.begin(), y.end()) * 1.1;
    double b = 1.0;
    double c = 0.5;

    ceres::Problem problem;
    for (size_t i = 0; i < t.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<GompertzResidual, 1, 1, 1, 1>(
                new GompertzResidual(t[i], y[i])),
            nullptr, &A, &b, &c);
    }
    problem.SetParameterLowerBound(&A, 0, 0.001);
    problem.SetParameterLowerBound(&c, 0, 0.001);

    ceres::Solver::Options options;
    options.max_num_iterations = 400;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "gompertz";
    result.params["A"] = A;
    result.params["b"] = b;
    result.params["c"] = c;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, t.size(), 3);
    return result;
}

FitResult fit_damped_sinusoidal(const std::vector<double>& t,
                                const std::vector<double>& y) {
    double y_max = *std::max_element(y.begin(), y.end());
    double y_min = *std::min_element(y.begin(), y.end());

    FitResult best_result;
    best_result.aic_score = std::numeric_limits<double>::max();

    std::vector<double> freq_guesses = {0.1, 0.5, 1.0, 2.0, 5.0};
    for (double freq_init : freq_guesses) {
        double A    = (y_max - y_min) / 2.0;
        double d    = 0.1;
        double freq = freq_init;
        double phi  = 0.0;
        double C    = (y_max + y_min) / 2.0;

        ceres::Problem problem;
        for (size_t i = 0; i < t.size(); i++) {
            problem.AddResidualBlock(
                new ceres::AutoDiffCostFunction<DampedSinusoidalResidual, 1, 1, 1, 1, 1, 1>(
                    new DampedSinusoidalResidual(t[i], y[i])),
                nullptr, &A, &d, &freq, &phi, &C);
        }
        problem.SetParameterLowerBound(&d,    0, 0.0);
        problem.SetParameterLowerBound(&freq, 0, 0.001);
        problem.SetParameterLowerBound(&A,    0, 0.0);

        ceres::Solver::Options options;
        options.max_num_iterations = 500;
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = false;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        double aic = compute_aic(summary.final_cost, t.size(), 5);
        if (aic < best_result.aic_score) {
            best_result.model_name = "damped_sinusoidal";
            best_result.params["A"]    = A;
            best_result.params["d"]    = d;
            best_result.params["freq"] = freq;
            best_result.params["phi"]  = phi;
            best_result.params["C"]    = C;
            best_result.residual_error = summary.final_cost;
            best_result.aic_score      = aic;
        }
    }
    return best_result;
}

FitResult fit_linear(const std::vector<double>& x,
                     const std::vector<double>& y) {
    double m = 1.0;
    double b = 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LinearResidual, 1, 1, 1>(
                new LinearResidual(x[i], y[i])),
            nullptr, &m, &b);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "linear";
    result.params["m"] = m;
    result.params["b"] = b;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 2);
    return result;
}

FitResult fit_quadratic(const std::vector<double>& x,
                        const std::vector<double>& y) {
    double a = 1.0, b = 0.0, c = 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<QuadraticResidual, 1, 1, 1, 1>(
                new QuadraticResidual(x[i], y[i])),
            nullptr, &a, &b, &c);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "quadratic";
    result.params["a"] = a;
    result.params["b"] = b;
    result.params["c"] = c;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 3);
    return result;
}

FitResult fit_cubic(const std::vector<double>& x,
                    const std::vector<double>& y) {
    double a = 0.1, b = 0.0, c = 0.0, d = 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<CubicResidual, 1, 1, 1, 1, 1>(
                new CubicResidual(x[i], y[i])),
            nullptr, &a, &b, &c, &d);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "cubic";
    result.params["a"] = a;
    result.params["b"] = b;
    result.params["c"] = c;
    result.params["d"] = d;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 4);
    return result;
}

FitResult fit_logarithmic(const std::vector<double>& x,
                          const std::vector<double>& y) {
    double a = 1.0, b = 0.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] <= 0) continue;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<LogarithmicResidual, 1, 1, 1>(
                new LogarithmicResidual(x[i], y[i])),
            nullptr, &a, &b);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = 200;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "logarithmic";
    result.params["a"] = a;
    result.params["b"] = b;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 2);
    return result;
}

FitResult fit_double_exponential(const std::vector<double>& t,
                                 const std::vector<double>& y) {
    double A = *std::max_element(y.begin(), y.end()) * 0.6;
    double b = 0.5;
    double C = A * 0.4;
    double d = 0.1;

    ceres::Problem problem;
    for (size_t i = 0; i < t.size(); i++) {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<DoubleExponentialResidual, 1, 1, 1, 1, 1>(
                new DoubleExponentialResidual(t[i], y[i])),
            nullptr, &A, &b, &C, &d);
    }
    problem.SetParameterLowerBound(&b, 0, 0.0001);
    problem.SetParameterLowerBound(&d, 0, 0.0001);

    ceres::Solver::Options options;
    options.max_num_iterations = 400;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "double_exponential";
    result.params["A"] = A;
    result.params["b"] = b;
    result.params["C"] = C;
    result.params["d"] = d;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, t.size(), 4);
    return result;
}

FitResult fit_michaelis_menten(const std::vector<double>& x,
                               const std::vector<double>& y) {
    double Vmax = *std::max_element(y.begin(), y.end()) * 1.2;
    double Km   = *std::max_element(x.begin(), x.end()) / 2.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x.size(); i++) {
        if (x[i] <= 0) continue;
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<MichaelisMentenResidual, 1, 1, 1>(
                new MichaelisMentenResidual(x[i], y[i])),
            nullptr, &Vmax, &Km);
    }
    problem.SetParameterLowerBound(&Vmax, 0, 0.001);
    problem.SetParameterLowerBound(&Km,   0, 0.001);

    ceres::Solver::Options options;
    options.max_num_iterations = 400;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    FitResult result;
    result.model_name = "michaelis_menten";
    result.params["Vmax"] = Vmax;
    result.params["Km"]   = Km;
    result.residual_error = summary.final_cost;
    result.aic_score = compute_aic(summary.final_cost, x.size(), 2);
    return result;
}

// ── Best model selector ───────────────────────────────────────────────────────

FitResult fit_best_model(const std::vector<double>& x,
                         const std::vector<double>& y) {
    std::vector<FitResult> candidates;

    try { candidates.push_back(fit_logistic(x, y)); }           catch (...) {}
    try { candidates.push_back(fit_exponential(x, y)); }        catch (...) {}
    try { candidates.push_back(fit_power_law(x, y)); }          catch (...) {}
    try { candidates.push_back(fit_sinusoidal(x, y)); }         catch (...) {}
    try { candidates.push_back(fit_gaussian(x, y)); }           catch (...) {}
    try { candidates.push_back(fit_gompertz(x, y)); }           catch (...) {}
    try { candidates.push_back(fit_damped_sinusoidal(x, y)); }  catch (...) {}
    try { candidates.push_back(fit_linear(x, y)); }             catch (...) {}
    try { candidates.push_back(fit_quadratic(x, y)); }          catch (...) {}
    try { candidates.push_back(fit_cubic(x, y)); }              catch (...) {}
    try { candidates.push_back(fit_logarithmic(x, y)); }        catch (...) {}
    try { candidates.push_back(fit_double_exponential(x, y)); } catch (...) {}
    try { candidates.push_back(fit_michaelis_menten(x, y)); }   catch (...) {}

    if (candidates.empty()) {
        FitResult empty;
        empty.model_name = "none";
        return empty;
    }

    return *std::min_element(candidates.begin(), candidates.end(),
        [](const FitResult& a, const FitResult& b) {
            return a.aic_score < b.aic_score;
        });
}