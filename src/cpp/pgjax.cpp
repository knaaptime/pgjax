// pgjax: on-device Pólya-Gamma sampling as XLA FFI custom calls.
//
// Replaces the host `jax.pure_callback` PG draw (a device<->host round-trip
// every Gibbs sweep, which serializes under jax.pmap and taxes every sweep) with
// a native FFI custom call that samples PG on-device inside the JIT'd program.
//
// Sampler:
//   - small integer h (incl. Bernoulli/logit h = 1): the exact Devroye method
//     for PG(1, z) (Polson, Scott & Windle 2013), summed h times.
//   - real h (e.g. Negative-Binomial h = y + alpha): the exact infinite-sum
//     (Gamma) representation truncated at K terms with a moment-matched Gamma
//     tail correction whose mean/variance are closed-form, so the truncation
//     bias (which otherwise collapses alpha in NB models) is cancelled.
//
// RNG: a per-call splitmix64-seeded xoshiro256** — seeded from the JAX PRNG key,
// so draws are reproducible and each call owns its generator (safe under the
// concurrent per-device calls that jax.pmap issues).

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// ---------------------------------------------------------------------------
// RNG: xoshiro256** seeded via splitmix64. Small, fast, per-call (thread-safe).
// ---------------------------------------------------------------------------
namespace {

constexpr double PI = 3.141592653589793238462643383279502884;
constexpr double PI2 = PI * PI;
constexpr double PI4 = PI2 * PI2;
constexpr double TRUNC = 0.64;            // Devroye truncation point
constexpr double TRUNC_INV = 1.0 / 0.64;

struct Rng {
  uint64_t s[4];
  explicit Rng(uint64_t seed) {
    // splitmix64 to seed the state.
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL;
    for (int i = 0; i < 4; ++i) {
      z += 0x9E3779B97F4A7C15ULL;
      uint64_t x = z;
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      s[i] = x ^ (x >> 31);
    }
  }
  static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }
  inline uint64_t next() {
    const uint64_t result = rotl(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
  }
  // uniform in (0, 1): 53-bit mantissa, strictly > 0.
  inline double unif() {
    double u = (next() >> 11) * (1.0 / 9007199254740992.0);
    return u <= 0.0 ? 1e-300 : u;
  }
  inline double expon() { return -std::log(unif()); }           // Exp(1)
  inline double normal() {                                       // N(0,1) Box-Muller
    double u1 = unif(), u2 = unif();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * PI * u2);
  }
};

// Gamma(shape, 1) via Marsaglia-Tsang (shape >= 1) with the shape<1 boost.
inline double gamma_draw(double shape, Rng& rng) {
  if (shape < 1.0) {
    double u = rng.unif();
    return gamma_draw(shape + 1.0, rng) * std::pow(u, 1.0 / shape);
  }
  double d = shape - 1.0 / 3.0;
  double c = 1.0 / std::sqrt(9.0 * d);
  while (true) {
    double x, v;
    do {
      x = rng.normal();
      v = 1.0 + c * x;
    } while (v <= 0.0);
    v = v * v * v;
    double u = rng.unif();
    double x2 = x * x;
    if (u < 1.0 - 0.0331 * x2 * x2) return d * v;
    if (std::log(u) < 0.5 * x2 + d * (1.0 - v + std::log(v))) return d * v;
  }
}

// log Phi(x) — log standard-normal CDF, via erfc.
inline double pnorm_log(double x) {
  return std::log(0.5 * std::erfc(-x * 0.7071067811865476));
}

// n-th coefficient of the alternating series for the Jacobi density at x.
inline double a_coef(int n, double x) {
  double d = n + 0.5;
  if (x > TRUNC) {
    return PI * d * std::exp(-d * d * PI2 * x * 0.5);
  }
  return std::pow(2.0 / (PI * x), 1.5) * PI * d * std::exp(-2.0 * d * d / x);
}

// Probability mass assigned to the right (truncated-exponential) region.
inline double mass_texpon(double z) {
  double fz = PI2 * 0.125 + 0.5 * z * z;
  double b = std::sqrt(TRUNC_INV) * (TRUNC * z - 1.0);
  double a = -std::sqrt(TRUNC_INV) * (TRUNC * z + 1.0);
  double x0 = std::log(fz) + fz * TRUNC;
  double xb = x0 - z + pnorm_log(b);
  double xa = x0 + z + pnorm_log(a);
  double qdivp = 4.0 / PI * (std::exp(xb) + std::exp(xa));
  return 1.0 / (1.0 + qdivp);
}

// Sample from the inverse-Gaussian(mu=1/z, 1) truncated to (0, TRUNC).
inline double rtigauss(double z, Rng& rng) {
  z = std::fabs(z);
  double mu = 1.0 / z;
  double x = TRUNC + 1.0;
  if (mu > TRUNC) {
    // Rejection on the truncated region via the alternating series bound.
    double alpha = 0.0;
    while (rng.unif() > alpha) {
      double e1 = rng.expon(), e2 = rng.expon();
      while (e1 * e1 > 2.0 * e2 / TRUNC) {
        e1 = rng.expon();
        e2 = rng.expon();
      }
      x = TRUNC / ((1.0 + TRUNC * e1) * (1.0 + TRUNC * e1));
      alpha = std::exp(-0.5 * z * z * x);
    }
  } else {
    while (x > TRUNC) {
      double y = rng.normal();
      y = y * y;
      double half_mu = 0.5 * mu;
      double muY = mu * y;
      x = mu + half_mu * muY - half_mu * std::sqrt(4.0 * muY + muY * muY);
      if (rng.unif() > mu / (mu + x)) x = mu * mu / x;
    }
  }
  return x;
}

// One draw X ~ PG(1, z_in) via the Devroye method.
inline double sample_pg1(double z_in, Rng& rng) {
  double z = std::fabs(z_in) * 0.5;
  double fz = PI2 * 0.125 + 0.5 * z * z;
  double p_right = mass_texpon(z);
  while (true) {
    double x;
    if (rng.unif() < p_right) {
      x = TRUNC + rng.expon() / fz;  // right: truncated exponential
    } else {
      x = rtigauss(z, rng);          // left: truncated inverse-Gaussian
    }
    double s = a_coef(0, x);
    double y = rng.unif() * s;
    int n = 0;
    while (true) {
      ++n;
      if (n & 1) {
        s -= a_coef(n, x);
        if (y <= s) return 0.25 * x;  // PG(1,z) = X/4
      } else {
        s += a_coef(n, x);
        if (y > s) break;             // reject; redraw proposal
      }
    }
  }
}

// PG(h, z) for real h > 0 via the exact infinite-sum (Gamma) representation
//   omega = (1/2π²) Σ_{k≥1} G_k / ((k-½)² + a²),  G_k ~ Gamma(h,1),  a = |z|/2π,
// truncated at K terms with a moment-matched Gamma **tail correction** for the
// remainder Σ_{k>K}.  The tail's mean and variance have closed forms
//   Σ 1/d_k  = (π/2a) tanh(πa),   Σ 1/d_k² = π tanh(πa)/(4a³) − π² sech²(πa)/(4a²)
// so the correction removes the truncation bias exactly in the mean (it is that
// bias — truncation without the tail — that makes α collapse in NB models).
inline double sample_pg_gamma(double h, double z, Rng& rng) {
  const int K = 20;
  double a = std::fabs(z) / (2.0 * PI);
  double a2 = a * a;
  double acc = 0.0, sinvd = 0.0, sinvd2 = 0.0;
  for (int k = 1; k <= K; ++k) {
    double km = k - 0.5;
    double d = km * km + a2;
    acc += gamma_draw(h, rng) / d;
    sinvd += 1.0 / d;
    sinvd2 += 1.0 / (d * d);
  }
  // Closed-form full sums, so the tail = full − (first K).
  double Sinf, Tinf;
  if (a < 1e-4) {
    Sinf = PI2 / 2.0;          // Σ 1/(k-½)²  = π²/2
    Tinf = PI4 / 6.0;          // Σ 1/(k-½)⁴  = π⁴/6
  } else {
    double th = std::tanh(PI * a);
    double sech = 1.0 / std::cosh(PI * a);
    Sinf = (PI / (2.0 * a)) * th;
    Tinf = PI * th / (4.0 * a * a2) - PI2 * sech * sech / (4.0 * a2);
  }
  double tail_m = Sinf - sinvd;
  double tail_v = Tinf - sinvd2;
  if (tail_m < 0.0) tail_m = 0.0;
  if (tail_v < 0.0) tail_v = 0.0;

  double coef = 1.0 / (2.0 * PI2);
  double omega = coef * acc;
  double ER = h * coef * tail_m;               // E[remainder]
  double VarR = h * coef * coef * tail_v;      // Var[remainder]
  if (ER > 0.0 && VarR > 1e-300) {
    double shape = ER * ER / VarR;
    double scale = VarR / ER;
    omega += gamma_draw(shape, rng) * scale;
  } else {
    omega += ER;
  }
  return omega;
}

// ---------------------------------------------------------------------------
// Saddlepoint sampler for PG(h, z) (Windle, Polson & Scott 2014,
// arXiv:1405.0506). A rejection sampler whose proposal envelope is a
// two-piece bounding kernel and whose target is the saddlepoint density
// approximation. O(1) per accepted draw, preferred for large h.
//
// Algorithm and constants ported from zoj613/polyagamma (BSD-3-Clause,
// compatible with this package's license), adapted to pgjax's Rng and to
// double precision throughout.
// ---------------------------------------------------------------------------

// tanh(x)/x on [0, inf): Cody-Waite rational polynomial for x <= 4.95,
// 1/x for x > 4.95. Max relative error < 1e-4.
inline double tanh_x(double x) {
  if (x > 4.95) return 1.0 / x;
  static const double p0 = -0.16134119023996228053e+04,
                     p1 = -0.99225929672236083313e+02,
                     p2 = -0.96437492777225469787e+00,
                     q0 = 0.48402357071988688686e+04,
                     q1 = 0.22337720718962312926e+04,
                     q2 = 0.11274474380534949335e+03;
  double x2 = x * x;
  return 1.0 + x2 * ((p2 * x2 + p1) * x2 + p0) /
                     (((x2 + q2) * x2 + q1) * x2 + q0);
}

// Parameters carried through one saddlepoint draw.
struct SaddleParams {
  double right_tangent_intercept;
  double left_tangent_intercept;
  double right_tangent_slope;
  double left_tangent_slope;
  double right_kernel_coef;
  double left_kernel_coef;
  double sqrt_alpha;       // 1/sqrt(alpha_l)
  double log_cosh_z;       // log(cosh(z))
  double sqrt_h2pi;        // sqrt(h/(2*pi))
  double half_z2;         // 0.5 * z * z
  double logxc;           // log(xc)
  double xc;
  double h;
  double z;
  double x;               // current proposal
};

// f and fprime at a point.
struct FuncValue {
  double f, fprime;
};

// K(t), the cumulant generating function. t = u + half_z2.
inline double cumulant(double u, const SaddleParams* pr) {
  if (u < 0.0)
    return pr->log_cosh_z - std::log(std::cosh(std::sqrt(-2.0 * u)));
  if (u > 0.0)
    return pr->log_cosh_z - std::log(std::cos(std::sqrt(2.0 * u)));
  return pr->log_cosh_z;
}

// K'(t) and K''(t). f = K'(t), fprime = K''(t).
inline void cumulant_prime(double u, FuncValue* rv) {
  double s = 2.0 * u;
  if (s < 0.0) {
    rv->f = tanh_x(std::sqrt(-s));
  } else if (s > 0.0) {
    rv->f = std::tan(std::sqrt(s)) / std::sqrt(s);
  } else {
    rv->f = 1.0;
  }
  // K''(t) = K'(t)^2 + (1 - K'(t))/t. Limit as t->0 is 1/3.
  if (std::fabs(s) < 1e-12) {
    rv->fprime = 1.0 / 3.0;
  } else {
    rv->fprime = rv->f * rv->f + (1.0 - rv->f) / s;
  }
}

// Starting guess for Newton's method given x = K'(t). Upper bound of u is pi^2/8.
inline double select_starting_guess(double x) {
  if (x <= 0.25) return -9.0;
  if (x <= 0.5) return -1.78;
  if (x <= 1.0) return -0.147;
  if (x <= 1.5) return 0.345;
  if (x <= 2.5) return 0.72;
  if (x <= 4.0) return 0.95;
  return 1.15;
}

// Solve K'(t) = arg for t using Newton's method.
inline double newton_raphson(double arg, double x0, FuncValue* value) {
  static const double atol = 1e-05, rtol = 1e-05;
  unsigned int n = 0;
  double x = x0;
  do {
    x0 = x;
    cumulant_prime(x0, value);
    double fval = value->f - arg;
    if (std::fabs(fval) <= atol || value->fprime <= atol) return x0;
    x = x0 - fval / value->fprime;
    // Convergence: relative + absolute closeness of successive iterates.
  } while (std::fabs(x - x0) > atol + rtol * std::fabs(x) && ++n < 25);
  return x;
}

// log standard normal CDF.
inline double log_norm_cdf(double x) {
  return std::log1p(-0.5 * std::erfc(x * 0.7071067811865476));
}

// log CDF of Inverse-Gaussian(mu, lambda), via Giner & Smyth (2016).
inline double invgauss_logcdf(double x, double mu, double lambda) {
  double qm = x / mu;
  double tm = mu / lambda;
  double r = std::sqrt(x / lambda);
  double a = log_norm_cdf((qm - 1.0) / r);
  double b = 2.0 / tm + log_norm_cdf(-(qm + 1.0) / r);
  return a + std::log1p(std::exp(b - a));
}

// G(p, x) confluent hypergeometric ratio via continued fraction for x <= p.
inline double confluent_x_smaller(double p, double x) {
  float a = 1.0f, b = static_cast<float>(p);
  float r = static_cast<float>(-(p - 1.0) * x);
  float s = static_cast<float>(0.5 * x);
  float f = a / b, c = a / FLT_MIN, d = 1.0f / b;
  for (int n = 2; n < 100; ++n) {
    a = (n & 1) ? s * (n - 1) : r - s * n;
    b += 1.0f;
    c = b + a / c;
    if (c < FLT_MIN) c = FLT_MIN;
    d = a * d + b;
    if (d < FLT_MIN) d = FLT_MIN;
    d = 1.0f / d;
    float delta = c * d;
    f *= delta;
    if (std::fabs(delta - 1.0f) < FLT_EPSILON) break;
  }
  return f;
}

// G(p, x) for x > p.
inline double confluent_p_smaller(double p, double x) {
  float a = 1.0f, b = static_cast<float>(x - p + 1.0);
  float f = a / b, c = a / FLT_MIN, d = 1.0f / b;
  for (int n = 1; n < 100; ++n) {
    a = static_cast<float>(n * (p - n));
    b += 2.0f;
    c = b + a / c;
    if (c < FLT_MIN) c = FLT_MIN;
    d = a * d + b;
    if (d < FLT_MIN) d = FLT_MIN;
    d = 1.0f / d;
    float delta = c * d;
    f *= delta;
    if (std::fabs(delta - 1.0f) < FLT_EPSILON) break;
  }
  return f;
}

// (Normalized or not) upper incomplete gamma(p, x).
// Abergel & Moisan (2020) continued fractions + terminating series for
// small int/half-int p <= 30.
inline double upper_incomplete_gamma(double p, double x, bool normalized) {
  if (normalized) {
    int p_int = static_cast<int>(p);
    if (p == p_int && p < 30.0) {
      double sum, r;
      int k = 1;
      for (sum = r = 1.0; k < p_int; ++k) sum += (r *= x / k);
      return std::exp(-x) * sum;
    } else if (p == p_int + 0.5 && p < 30.0) {
      double sum, r;
      int k = 1;
      static const double one_sqrtpi = 0.5641895835477563;
      double sqrt_x = std::sqrt(x);
      for (r = 1.0, sum = 0.0; k < p_int + 1; ++k) sum += (r *= x / (k - 0.5));
      return std::erfc(sqrt_x) + std::exp(-x) * one_sqrtpi * sum / sqrt_x;
    }
  }
  bool x_smaller = p >= x;
  double f = x_smaller ? confluent_x_smaller(p, x)
                       : confluent_p_smaller(p, x);
  if (normalized) {
    double out = f * std::exp(-x + p * std::log(x) - std::lgamma(p));
    return x_smaller ? 1.0 - out : out;
  } else if (x_smaller) {
    double lgam = std::lgamma(p);
    double exp_lgam = std::exp(std::min(lgam, 88.7228));
    double arg = -x + p * std::log(x) - lgam;
    if (arg >= 88.7228) arg = 88.7228;
    if (arg <= -88.7228) arg = -88.7228;
    return (1.0 - f * std::exp(arg)) * exp_lgam;
  } else {
    double arg = -x + p * std::log(x);
    return f * (arg >= 88.7228 ? std::exp(88.7228) : std::exp(arg));
  }
}

// Sample X ~ Gamma(shape=a, rate=b) truncated to {x > t}.
inline double random_left_bounded_gamma(double a, double b, double t, Rng& rng) {
  if (a > 1.0) {
    double bt = t * b;
    double amin1 = a - 1.0;
    double bmina = b - a;
    double c0 = 0.5 * (bmina + std::sqrt(bmina * bmina + 4.0 * b)) / b;
    double one_minus_c0 = 1.0 - c0;
    double log_m = amin1 * (std::log(amin1 / one_minus_c0) - 1.0);
    double x, threshold;
    do {
      x = b + rng.expon() / c0;
      threshold = amin1 * std::log(x) - x * one_minus_c0 - log_m;
    } while (std::log1p(-rng.unif()) > threshold);
    return t * (x / b);
  } else if (a == 1.0) {
    return t + rng.expon() / b;
  } else {
    double amin1 = a - 1.0;
    double tb = t * b;
    double x;
    do {
      x = 1.0 + rng.expon() / tb;
    } while (std::log1p(-rng.unif()) > amin1 * std::log(x));
    return t * x;
  }
}

// Precompute envelope and kernel constants for one (h, z).
inline void set_sampling_parameters(SaddleParams* pr, double h, double z) {
  static const double log275 = 1.0116009116784799;
  static const double log3 = 1.0986122886681098;
  double xl, logxl;
  if (z > 0.0) {
    xl = tanh_x(z);
    logxl = std::log(xl);
    pr->half_z2 = 0.5 * z * z;
    pr->log_cosh_z = std::log(std::cosh(z));
  } else {
    xl = 1.0;
    logxl = 0.0;
    pr->half_z2 = 0.0;
    pr->log_cosh_z = 0.0;
  }
  pr->xc = 2.75 * xl;
  double xr = 3.0 * xl;
  pr->h = h;
  pr->z = z;
  double xc_inv = 1.0 / pr->xc;
  double xl_inv = 1.0 / xl;
  double ul = -pr->half_z2;
  FuncValue rv;
  double ur = newton_raphson(xr, select_starting_guess(xr), &rv);
  newton_raphson(pr->xc, select_starting_guess(pr->xc), &rv);
  double tr = ur + pr->half_z2;
  pr->left_tangent_slope = -0.5 * (xl_inv * xl_inv);
  pr->left_tangent_intercept = cumulant(ul, pr) - 0.5 * xc_inv + xl_inv;
  pr->logxc = log275 + logxl;
  pr->right_tangent_slope = -tr - 1.0 / xr;
  pr->right_tangent_intercept =
      cumulant(ur, pr) + 1.0 - log3 - logxl + pr->logxc;
  double alpha_r = rv.fprime * (xc_inv * xc_inv);  // K''(t(xc)) / xc^2
  double alpha_l = xc_inv * alpha_r;               // K''(t(xc)) / xc^3
  pr->sqrt_alpha = 1.0 / std::sqrt(alpha_l);
  pr->sqrt_h2pi = std::sqrt(h * (0.5 / PI));
  pr->left_kernel_coef = pr->sqrt_h2pi * pr->sqrt_alpha;
  pr->right_kernel_coef = pr->sqrt_h2pi / std::sqrt(alpha_r);
}

// Saddlepoint density estimate at pr->x.
inline double saddle_point(SaddleParams* pr) {
  FuncValue rv;
  double u = newton_raphson(pr->x, select_starting_guess(pr->x), &rv);
  double t = u + pr->half_z2;
  return std::exp(pr->h * (cumulant(u, pr) - t * pr->x)) *
         pr->sqrt_h2pi / std::sqrt(rv.fprime);
}

// Bounding kernel k(x|h,z) (Proposition 17 of Windle et al. 2014).
inline double bounding_kernel(const SaddleParams* pr) {
  double point;
  if (pr->x > pr->xc) {
    point = pr->right_tangent_slope * pr->x + pr->right_tangent_intercept;
    return std::exp(pr->h * (pr->logxc + point) + (pr->h - 1.0) * std::log(pr->x)) *
           pr->right_kernel_coef;
  }
  point = pr->left_tangent_slope * pr->x + pr->left_tangent_intercept;
  return std::exp(0.5 * pr->h * (1.0 / pr->xc - 1.0 / pr->x) +
                  pr->h * point - 1.5 * std::log(pr->x)) *
         pr->left_kernel_coef;
}

// One draw omega ~ PG(h, z) via the saddlepoint rejection sampler.
inline double sample_pg_saddlepoint(double h, double z, Rng& rng) {
  SaddleParams pr;
  // The tilting is absorbed by halving |z| internally (the envelope is built
  // in the un-tilted coordinate, and the final 0.25*h*x scale recovers PG).
  set_sampling_parameters(&pr, h, 0.5 * std::fabs(z));
  double sqrt_rho = std::sqrt(-2.0 * pr.left_tangent_slope);
  double sqrt_rho_inv = 1.0 / sqrt_rho;
  double p = std::exp(h * (0.5 / pr.xc + pr.left_tangent_intercept - sqrt_rho) +
                      invgauss_logcdf(pr.xc, sqrt_rho_inv, h)) *
             pr.sqrt_alpha;
  double hrho = -h * pr.right_tangent_slope;
  double q = upper_incomplete_gamma(h, hrho * pr.xc, false) *
             pr.right_kernel_coef *
             std::exp(h * (pr.right_tangent_intercept - std::log(hrho)));
  double proposal_probability = p / (p + q);
  double mu2 = sqrt_rho_inv * sqrt_rho_inv;
  while (true) {
    do {
      if (rng.unif() < proposal_probability) {
        // Left proposal: truncated Inverse-Gaussian via Michael-Schucany-Haas.
        do {
          double y = rng.normal();
          double w = sqrt_rho_inv + 0.5 * mu2 * y * y / h;
          pr.x = w - std::sqrt(std::fabs(w * w - mu2));
          if (rng.unif() * (1.0 + pr.x * sqrt_rho) > 1.0) pr.x = mu2 / pr.x;
        } while (pr.x >= pr.xc);
      } else {
        pr.x = random_left_bounded_gamma(h, hrho, pr.xc, rng);
      }
    } while (rng.unif() * bounding_kernel(&pr) > saddle_point(&pr));
    return std::max(0.25 * h * pr.x, 1e-12);
  }
}

// PG(h, z): hybrid dispatch. Saddlepoint for large h (O(1) per draw);
// exact sum-of-Devroye for small positive integer h (fast, exact — covers
// the Bernoulli/logit h=1); the tail-corrected Gamma sum otherwise
// (real h, e.g. the Negative-Binomial h = y + alpha).
inline double sample_pg(double h, double z, Rng& rng) {
  // Saddlepoint regime: large h, or mid h with small |z|. Checked first so
  // integer h >= 8 takes the O(1) path rather than summing h Devroye draws.
  if (h >= 8.0 || (h > 4.0 && std::fabs(z) <= 4.0))
    return sample_pg_saddlepoint(h, z, rng);
  double hr = std::floor(h + 0.5);
  if (hr >= 1.0 && hr <= 20.0 && std::fabs(h - hr) < 1e-9) {
    int m = static_cast<int>(hr);
    double acc = 0.0;
    for (int i = 0; i < m; ++i) acc += sample_pg1(z, rng);
    return acc;
  }
  return sample_pg_gamma(h, z, rng);
}

// Draw omega[i] ~ PG(h[i], z[i]) for i in [0, n).
void draw_range(const double* h, const double* z, int64_t n, uint64_t seed,
                double* out) {
  Rng rng(seed);
  for (int64_t i = 0; i < n; ++i) {
    out[i] = std::max(sample_pg(h[i], z[i], rng), 1e-12);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// FFI handlers
//   pg_sample_f64(h[n], z[n], seed[]) -> omega[n]
//   pg_sample_batched_f64(h[B,n], z[B,n], seed[B]) -> omega[B,n]
// seed is an int64 buffer (scalar for the unbatched call, (B,) for the batch),
// derived from the JAX PRNG key on the Python side.
// ---------------------------------------------------------------------------

static ffi::Error PgSampleF64Impl(ffi::Buffer<ffi::F64> h,
                                  ffi::Buffer<ffi::F64> z,
                                  ffi::Buffer<ffi::S64> seed,
                                  ffi::ResultBuffer<ffi::F64> out) {
  int64_t n = static_cast<int64_t>(h.element_count());
  if (static_cast<int64_t>(z.element_count()) != n)
    return ffi::Error::InvalidArgument("pgjax: h and z must have equal length");
  uint64_t s = seed.element_count() > 0
                   ? static_cast<uint64_t>(seed.typed_data()[0])
                   : 0ULL;
  draw_range(h.typed_data(), z.typed_data(), n, s, out->typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PgSampleF64, PgSampleF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()  // h
                                  .Arg<ffi::Buffer<ffi::F64>>()  // z
                                  .Arg<ffi::Buffer<ffi::S64>>()  // seed
                                  .Ret<ffi::Buffer<ffi::F64>>());

static ffi::Error PgSampleBatchedF64Impl(ffi::Buffer<ffi::F64> h,
                                         ffi::Buffer<ffi::F64> z,
                                         ffi::Buffer<ffi::S64> seed,
                                         ffi::ResultBuffer<ffi::F64> out) {
  auto d = h.dimensions();
  if (d.size() < 2)
    return ffi::Error::InvalidArgument("pgjax: batched h must be >= 2D");
  int64_t batch = d[0];
  int64_t n = static_cast<int64_t>(h.element_count()) / batch;
  const int64_t* seeds = seed.typed_data();
  int64_t nseed = static_cast<int64_t>(seed.element_count());
  const double* hd = h.typed_data();
  const double* zd = z.typed_data();
  double* od = out->typed_data();
  for (int64_t b = 0; b < batch; ++b) {
    uint64_t s = static_cast<uint64_t>(seeds[nseed == batch ? b : 0]);
    // Distinct stream per batch element even if a single seed was broadcast.
    s ^= 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(b + 1);
    draw_range(hd + b * n, zd + b * n, n, s, od + b * n);
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(PgSampleBatchedF64, PgSampleBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::F64>>()  // h [B,n]
                                  .Arg<ffi::Buffer<ffi::F64>>()  // z [B,n]
                                  .Arg<ffi::Buffer<ffi::S64>>()  // seed [B]
                                  .Ret<ffi::Buffer<ffi::F64>>());

NB_MODULE(pgjax_cpp, m) {
  m.doc() = "On-device Pólya-Gamma sampling as XLA FFI custom calls";
  m.def("pg_sample_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(PgSampleF64));
  });
  m.def("pg_sample_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(PgSampleBatchedF64));
  });
}
