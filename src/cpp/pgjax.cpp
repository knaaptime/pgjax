// pgjax: on-device Pólya-Gamma sampling as XLA FFI custom calls.
//
// Replaces the host `jax.pure_callback` PG draw (a device<->host round-trip
// every Gibbs sweep, which serializes under jax.pmap and taxes every sweep) with
// a native FFI custom call that samples PG on-device inside the JIT'd program.
//
// Sampler — the hybrid dispatch of zoj613/polyagamma (`method=None`), changed
// where that scheme is measurably biased (2M-draw checks against the exact PG
// mean and variance):
//   - h > 50: normal approximation with the exact PG mean and variance.
//   - 8 <= h <= 50: saddlepoint rejection sampler.  (polyagamma also uses it for
//     4 < h < 8 with |z| <= 4, where it runs 1-2% low in variance.)
//   - h = 1 (Bernoulli/logit), or integer h <= 4 with |z| <= 1: the exact
//     Devroye method for PG(1, z) (Polson, Scott & Windle 2013), summed h times.
//   - 1 < h < 8 otherwise (e.g. Negative-Binomial h = y + alpha): the alternate
//     method (Windle, Polson & Scott 2014), with polyagamma's right-kernel
//     constant corrected (see alt::bounding_kernel).
//   - h < 1: the Gamma-series representation, exact in mean and variance.  (The
//     alternate method assumes h >= 1; below it, it runs 0.4-0.8% low.)
//
// RNG: a per-call splitmix64-seeded xoshiro256** — seeded from the JAX PRNG key,
// so draws are reproducible and each call owns its generator (safe under
// concurrent calls from several threads) — with ziggurat normals and
// exponentials.

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

// Ziggurat tables (Marsaglia & Tsang 2000; Doornik 2005), built once at load: a
// 128-block normal and a 256-block exponential.  x[i] are the block edges
// (x[0] = V / f(R) is the base block) and r[i] = x[i+1] / x[i] the fast-accept
// ratios.  Several times cheaper per variate than Box-Muller or -log(u).
struct Ziggurat {
  static constexpr int kNormBlocks = 128;
  static constexpr double kNormR = 3.442619855899;
  static constexpr double kNormV = 9.91256303526217e-3;
  static constexpr int kExpBlocks = 256;
  static constexpr double kExpR = 7.69711747013104972;
  static constexpr double kExpV = 0.0039496598225815571993;
  double nx[kNormBlocks + 1], nr[kNormBlocks];
  double ex[kExpBlocks + 1], er[kExpBlocks];
  Ziggurat() {
    double f = std::exp(-0.5 * kNormR * kNormR);
    nx[0] = kNormV / f;
    nx[1] = kNormR;
    nx[kNormBlocks] = 0.0;
    for (int i = 2; i < kNormBlocks; ++i) {
      nx[i] = std::sqrt(-2.0 * std::log(kNormV / nx[i - 1] + f));
      f = std::exp(-0.5 * nx[i] * nx[i]);
    }
    for (int i = 0; i < kNormBlocks; ++i) nr[i] = nx[i + 1] / nx[i];
    f = std::exp(-kExpR);
    ex[0] = kExpV / f;
    ex[1] = kExpR;
    ex[kExpBlocks] = 0.0;
    for (int i = 2; i < kExpBlocks; ++i) {
      ex[i] = -std::log(kExpV / ex[i - 1] + f);
      f = std::exp(-ex[i]);
    }
    for (int i = 0; i < kExpBlocks; ++i) er[i] = ex[i + 1] / ex[i];
  }
};
const Ziggurat kZig;

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
  // Exp(1) by the ziggurat; the tail beyond R is R + Exp(1) (memoryless).
  // The block index takes the low 8 bits, the uniform the high 53.
  inline double expon() {
    double extra = 0.0;
    for (;;) {
      uint64_t r = next();
      unsigned i = static_cast<unsigned>(r & 0xFF);
      double u = (r >> 11) * (1.0 / 9007199254740992.0);
      if (u < kZig.er[i]) return extra + u * kZig.ex[i];
      if (i == 0) {
        extra += Ziggurat::kExpR;
        continue;
      }
      double x = u * kZig.ex[i];
      double f0 = std::exp(-(kZig.ex[i] - x));
      double f1 = std::exp(-(kZig.ex[i + 1] - x));
      if (f1 + unif() * (f0 - f1) < 1.0) return extra + x;
    }
  }
  // N(0, 1) by the ziggurat (Doornik's ZIGNOR); the tail beyond R by
  // Marsaglia's exponential rejection.  Block index: low 7 bits; uniform: high 53.
  inline double normal() {
    for (;;) {
      uint64_t r = next();
      unsigned i = static_cast<unsigned>(r & 0x7F);
      double u = 2.0 * ((r >> 11) * (1.0 / 9007199254740992.0)) - 1.0;
      if (std::fabs(u) < kZig.nr[i]) return u * kZig.nx[i];
      if (i == 0) {
        double x, y;
        do {
          x = std::log(unif()) / Ziggurat::kNormR;
          y = std::log(unif());
        } while (-2.0 * y < x * x);
        return u < 0.0 ? x - Ziggurat::kNormR : Ziggurat::kNormR - x;
      }
      double x = u * kZig.nx[i];
      double f0 = std::exp(-0.5 * (kZig.nx[i] * kZig.nx[i] - x * x));
      double f1 = std::exp(-0.5 * (kZig.nx[i + 1] * kZig.nx[i + 1] - x * x));
      if (f1 + unif() * (f0 - f1) < 1.0) return x;
    }
  }
};

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

// Gamma(shape, 1) via Marsaglia-Tsang (shape >= 1) with the shape < 1 boost.
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

// PG(h, z) for h < 1 from the infinite-sum representation
//   omega = (1/2π²) Σ_{k≥1} G_k / ((k-½)² + a²),  G_k ~ Gamma(h, 1),  a = |z|/2π.
// The first K terms are drawn; the remainder is a Gamma matched to its mean and
// variance, whose closed forms are
//   Σ 1/d_k = (π/2a) tanh(πa),   Σ 1/d_k² = π tanh(πa)/(4a³) − π² sech²(πa)/(4a²),
// so the draw is exact in mean and variance.  At K = 20 the remainder carries
// about 1% of the mean, so its shape hardly matters.
inline double sample_pg_gamma_series(double h, double z, Rng& rng) {
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
  double Sinf, Tinf;
  if (a < 1e-4) {
    Sinf = PI2 / 2.0;  // Σ 1/(k-½)² = π²/2
    Tinf = PI4 / 6.0;  // Σ 1/(k-½)⁴ = π⁴/6
  } else {
    double th = std::tanh(PI * a);
    double sech = 1.0 / std::cosh(PI * a);
    Sinf = (PI / (2.0 * a)) * th;
    Tinf = PI * th / (4.0 * a * a2) - PI2 * sech * sech / (4.0 * a2);
  }
  double tail_m = std::max(Sinf - sinvd, 0.0);
  double tail_v = std::max(Tinf - sinvd2, 0.0);
  double coef = 1.0 / (2.0 * PI2);
  double omega = coef * acc;
  double mean_r = h * coef * tail_m;
  double var_r = h * coef * coef * tail_v;
  if (mean_r > 0.0 && var_r > 1e-300) {
    omega += gamma_draw(mean_r * mean_r / var_r, rng) * (var_r / mean_r);
  } else {
    omega += mean_r;
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
    // Work with the standard Gamma(a, 1) variate truncated at t * b, and scale
    // back on return, as polyagamma does.  (Keeping the unscaled b here biased
    // every a > 1 draw: the saddlepoint and alternate right-hand proposals.)
    b = t * b;
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

// ---------------------------------------------------------------------------
// Alternate sampler for PG(h, z) (Windle, Polson & Scott 2014,
// arXiv:1405.0506, Section 4): exact, used for real h below the saddlepoint
// region (e.g. the Negative-Binomial h = y + alpha).  It draws J*(h, z/2) from a
// two-piece proposal — a left-truncated Gamma right of the truncation point t
// and a right-truncated inverse-Gaussian left of it — accepted through the
// alternating-series bound, and returns PG(h, z) = J*(h, z/2) / 4.  For h > 4
// the draw is a sum of J*(b_i, z/2) with sum(b_i) = h and every b_i <= 4.
//
// Algorithm, constants and truncation-point table ported from zoj613/polyagamma
// (src/pgm_alternate.c, BSD-3-Clause), with one constant corrected (see
// bounding_kernel).  The alternating series is evaluated in
// single precision as there, because its termination test relies on terms
// vanishing at float resolution.
// ---------------------------------------------------------------------------
namespace alt {

constexpr int kTableSize = 25;
constexpr double kMaxH = 4.0;
// Optimal truncation point t(h) on h = 1, 1.125, ..., 4.
constexpr double kTruncPoint[kTableSize] = {
    1.273239366, 1.901515423, 2.281992126, 2.607829551, 2.910421526,
    3.200449543, 3.482766779, 3.759955106, 4.033540671, 4.304486011,
    4.573437633, 4.840840644, 5.107017272, 5.372204821, 5.636581947,
    5.900288573, 6.163432428, 6.426094330, 6.688351603, 6.950254767,
    7.211854235, 7.473186206, 7.734284136, 7.995175158, 8.255882407};
// log(n!) for n < 16; lgamma beyond.
constexpr double kLogFactorial[16] = {
    0.0, 0.0, 0.69314718055994530943, 1.79175946922805500079,
    3.17805383034794561975, 4.78749174278204599415, 6.57925121201010099526,
    8.52516136106541430086, 10.60460290274525022719, 12.80182748008146961186,
    15.10441257307551529612, 17.50230784587388584150, 19.98721449566188614923,
    22.55216385312342288610, 25.19122118273868150135, 27.89927138384089156699};
constexpr double PI2_8 = 1.233700550136169827354311374984519;    // pi^2 / 8
constexpr double LOGPI_2 = 0.4515827052894548647261952298948821;  // log(pi / 2)
constexpr double LS2PI = 0.9189385332046727417803297364056177;    // log(sqrt(2 pi))
constexpr double LOG2 = 0.6931471805599453094172321214581766;

struct Params {
  float proposal_probability;  // q / (p + q)
  double log_lambda_z;
  double lambda_z;             // pi^2 / 8 + z^2 / 2
  double half_h2;
  double lgammah;              // lgamma(h)
  double hlog2;
  double t_inv;
  double logx;
  double h_z2;                 // (h / z)^2
  double h_z;
  double z2;
  double h;
  double z;
  double x;
  double t;
};

// Truncation point for h in [1, 4], linearly interpolated on the table (h <= 1
// takes the first entry).  polyagamma looks it up by binary search, whose
// neighbour access runs off the table for h in (1, 1.125); t only tunes the
// proposal's efficiency, so plain interpolation is equivalent and bounds-safe.
inline double truncation_point(double h) {
  if (h <= 1.0) return kTruncPoint[0];
  if (h >= kMaxH) return kTruncPoint[kTableSize - 1];
  double pos = (h - 1.0) * 8.0;
  int i = static_cast<int>(pos);
  double frac = pos - i;
  return kTruncPoint[i] + frac * (kTruncPoint[i + 1] - kTruncPoint[i]);
}

inline double log_factorial(unsigned int n) {
  return n < 16 ? kLogFactorial[n] : std::lgamma(n + 1.0);
}

// a^L_n(x | h), the n-th coefficient of the alternating sum S^L(x | h).
inline float piecewise_coef(unsigned int n, const Params& pr) {
  double a = 2.0 * n + pr.h;
  double b = n ? std::lgamma(n + pr.h) - pr.lgammah : 0.0;
  return static_cast<float>(
      std::exp(pr.hlog2 + b - log_factorial(n) - LS2PI - 1.5 * pr.logx -
               0.5 * a * a / pr.x) * a);
}

// Bounding kernel k(x | h).
inline float bounding_kernel(const Params& pr) {
  if (pr.x > pr.t) {
    // log(pi / 2).  polyagamma uses log(sqrt(pi / 2)), leaving this piece of the
    // envelope a factor (pi/2)^(h/2) below both the Gamma proposal it weights by
    // q, computed with (pi/2)^h, and the density's own tail,
    // (pi/2)^h x^(h-1) exp(-pi^2 x / 8) / Gamma(h) — Devroye's pi/2 at h = 1.
    // The shortfall over-accepts x > t: +0.4-0.9% in the mean and +1-5% in the
    // variance for 1 < h < 4 at small |z|.
    static const double a = LOGPI_2;
    return static_cast<float>(std::exp(pr.h * a + (pr.h - 1.0) * pr.logx -
                                       PI2_8 * pr.x - pr.lgammah));
  }
  if (pr.x > 0.0) {
    return static_cast<float>(
        std::exp(pr.hlog2 - pr.half_h2 / pr.x - 1.5 * pr.logx - LS2PI) * pr.h);
  }
  return 0.0f;
}

// CDF of the inverse-Gaussian proposal at t.
inline double invgauss_cdf(const Params& pr) {
  static const double sqrt2_inv = 0.7071067811865475;
  double st = std::sqrt(pr.t);
  double a = sqrt2_inv * pr.h / st;
  double b = pr.z * st * sqrt2_inv;
  double ez = std::exp(pr.h * pr.z);
  return 0.5 * (std::erfc(a - b) + ez * std::erfc(b + a) * ez);
}

// Set the h-dependent constants; with `update` the z-dependent ones from an
// earlier call are kept (the chunked draw for h > 4).
inline void set_sampling_parameters(Params& pr, double h, bool update) {
  double p;
  pr.h = h;
  pr.t = truncation_point(h);
  pr.t_inv = 1.0 / pr.t;
  pr.half_h2 = 0.5 * h * h;
  pr.lgammah = std::lgamma(h);
  pr.hlog2 = h * LOG2;
  if (pr.z > 0.0) {
    if (!update) {
      pr.z2 = pr.z * pr.z;
      pr.lambda_z = PI2_8 + 0.5 * pr.z2;
      pr.log_lambda_z = std::log(pr.lambda_z);
    }
    pr.h_z = h / pr.z;
    pr.h_z2 = pr.h_z * pr.h_z;
    p = std::exp(pr.hlog2 - h * pr.z) * invgauss_cdf(pr);
  } else {
    if (!update) {
      pr.lambda_z = PI2_8;
      pr.log_lambda_z = std::log(pr.lambda_z);
    }
    p = std::exp(pr.hlog2) * std::erfc(h / std::sqrt(2.0 * pr.t));
  }
  double q = std::exp(h * (LOGPI_2 - pr.log_lambda_z)) *
             upper_incomplete_gamma(h, pr.lambda_z * pr.t, true);
  pr.proposal_probability = static_cast<float>(q / (p + q));
}

// X ~ Inverse-Gaussian(h/z, h^2) truncated to {x < t} (Devroye 1986, p. 149
// when the mean is below t; a scaled inverse-chi-square proposal otherwise).
inline void right_bounded_invgauss(Rng& rng, Params& pr) {
  if (pr.t < pr.h_z) {
    do {
      pr.x = 1.0 / random_left_bounded_gamma(0.5, pr.half_h2, pr.t_inv, rng);
    } while (std::log1p(-rng.unif()) >= -0.5 * pr.z2 * pr.x);
    return;
  }
  do {
    double y = rng.normal();
    double w = pr.h_z + 0.5 * y * y / pr.z2;
    pr.x = w - std::sqrt(std::fabs(w * w - pr.h_z2));
    if (rng.unif() * (pr.h_z + pr.x) > pr.h_z) pr.x = pr.h_z2 / pr.x;
  } while (pr.x >= pr.t);
}

// One draw from J*(h, z) for h <= 4.
inline double jacobi_star(Rng& rng, Params& pr) {
  for (;;) {
    if (rng.unif() <= pr.proposal_probability) {
      pr.x = random_left_bounded_gamma(pr.h, pr.lambda_z, pr.t, rng);
    } else if (pr.z > 0.0) {
      right_bounded_invgauss(rng, pr);
    } else {
      pr.x = 1.0 / random_left_bounded_gamma(0.5, pr.half_h2, pr.t_inv, rng);
    }
    pr.logx = std::log(pr.x);
    float u = static_cast<float>(rng.unif()) * bounding_kernel(pr);
    float s = piecewise_coef(0, pr);
    for (unsigned int n = 1;; ++n) {
      float old_s = s;
      if (n & 1) {
        s -= piecewise_coef(n, pr);
        if (old_s >= s && u <= s) return pr.x;
      } else {
        s += piecewise_coef(n, pr);
        if (old_s >= s && u > s) break;
      }
    }
  }
}

}  // namespace alt

// One draw omega ~ PG(h, z) via the alternate method.
inline double sample_pg_alternate(double h, double z, Rng& rng) {
  alt::Params pr{};
  pr.z = 0.5 * std::fabs(z);
  if (h <= alt::kMaxH) {
    alt::set_sampling_parameters(pr, h, false);
    return 0.25 * alt::jacobi_star(rng, pr);
  }
  double chunk = h >= alt::kMaxH + 1.0 ? alt::kMaxH : alt::kMaxH - 1.0;
  alt::set_sampling_parameters(pr, chunk, false);
  double out = 0.0;
  while (h > alt::kMaxH) {
    out += 0.25 * alt::jacobi_star(rng, pr);
    h -= chunk;
  }
  alt::set_sampling_parameters(pr, h, true);
  return out + 0.25 * alt::jacobi_star(rng, pr);
}

// One draw omega ~ PG(h, z) from the normal with PG's mean and variance:
//   mean = h tanh(z/2) / (2z),  var = h (sinh z - z) sech^2(z/2) / (4 z^3).
// For small |z| the variance uses its series, h/24 - h z^2 / 240 (the closed
// form cancels catastrophically there).
inline double sample_pg_normal(double h, double z, Rng& rng) {
  z = std::fabs(z);
  double mean, var;
  if (z < 1e-3) {
    mean = h * (0.25 - z * z / 48.0);
    var = h * (1.0 / 24.0 - z * z / 240.0);
  } else {
    double th = std::tanh(0.5 * z);
    mean = 0.5 * h * th / z;
    var = 0.25 * h * (std::sinh(z) - z) * (1.0 - th * th) / (z * z * z);
  }
  return mean + rng.normal() * std::sqrt(var);
}

// PG(h, z): dispatch as described in the header.  PG is even in z, so the
// thresholds use |z|.
inline double sample_pg(double h, double z, Rng& rng) {
  double az = std::fabs(z);
  if (h > 50.0) return sample_pg_normal(h, z, rng);
  if (h >= 8.0) return sample_pg_saddlepoint(h, z, rng);
  if (h == 1.0 || (h == std::floor(h) && h <= 4.0 && az <= 1.0)) {
    int m = static_cast<int>(h);
    double acc = 0.0;
    for (int i = 0; i < m; ++i) acc += sample_pg1(z, rng);
    return acc;
  }
  if (h < 1.0) return sample_pg_gamma_series(h, z, rng);
  return sample_pg_alternate(h, z, rng);
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
