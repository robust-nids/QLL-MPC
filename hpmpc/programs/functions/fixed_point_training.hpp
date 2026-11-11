#pragma once
// Training building blocks on fixed-point shares, for the code programs/torch/torch_to_hpmpc.py generates into
// functions/generated/. Only what hpmpc lacks: arrays of products, ReLU with the 0/1 shares its backward needs, the
// ReLU backward, a truncation that does not fail, exp, log, reciprocal, division, logaddexp, and SGD. Local operations
// are loops in the generated code.
#include "../../datatypes/Additive_Share.hpp"
#include "../../datatypes/float_fixed_converter.hpp"
#include "GEMM.hpp"
#include "comparisons.hpp"
#include "prob_truncation.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

// Convert float to fixed
using FFConv = FloatFixedConverter<double, INT_TYPE, UINT_TYPE, FRACTIONAL>;

// Convert to fixed
static UINT_TYPE to_fx(double v){return FFConv::float_to_ufixed(v);}

// Convert to float
static double to_fl(UINT_TYPE v){return FFConv::ufixed_to_float(v);}

// x /= 2^bits with stochastic rounding
template <typename A>
static void trunc(A* x, int n, int bits = FRACTIONAL)
{
    const A down = A(UINT_TYPE(1) << (BITLENGTH - 2));  // trunc_2k_in_place adds 2^(l-1): net offset 2^(l-2)
    for (int i = 0; i < n; i++)
        x[i] = x[i] - down;
    trunc_2k_in_place(x, n, false, bits);
    const A up = A(UINT_TYPE(1) << (BITLENGTH - 2 - bits));  // it subtracts 2^(l-1-bits): net 2^(l-2-bits)
    for (int i = 0; i < n; i++)
        x[i] = x[i] + up;
}

// out[m][p] = a[m][f] b[p][f]^T with hpmpc's GEMM, truncated
template <typename A>
static void gemm(const A* a, const A* b, A* out, int m, int p, int f)
{
    for (int i = 0; i < m * p; i++)
        out[i] = A(UINT_TYPE(0));  // prepare_GEMM adds into out
    prepare_GEMM(const_cast<A*>(a), const_cast<A*>(b), out, m, p, f, false);
    A::communicate();
    complete_GEMM(out, m, p);
    trunc(out, m * p);
}

// out = x * y / 2^bits; the default is the fixed-point product
template <typename A>
static void mul(const A* x, const A* y, A* out, int n, int bits = FRACTIONAL)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = x[i].prepare_dot(y[i]);
        out[i].mask_and_send_dot_without_trunc();
    }
    A::communicate();
    for (int i = 0; i < n; i++)
        out[i].complete_mult_without_trunc();
    trunc(out, n, bits);
}

// out = c * x for a public real c; local when c is an integer, a truncation otherwise
template <typename A>
static void mul_public(const A* x, double c, A* out, int n)
{
    if (c == std::round(c) && std::abs(c) < (1 << 30))
    {
        const UINT_TYPE ci = UINT_TYPE(INT_TYPE(c));
        for (int i = 0; i < n; i++)
            out[i] = x[i].mult_public(ci);
        return;
    }
    const UINT_TYPE cf = to_fx(c);
    for (int i = 0; i < n; i++)
        out[i] = x[i].mult_public(cf);
    trunc(out, n);
}

// bit = [x >= 0] as 0/1 integer shares (hpmpc's LTZ): n comparisons in the rounds of one
template <typename Share>
static void geq_zero(const Additive_Share<DATATYPE, Share>* x, Additive_Share<DATATYPE, Share>* bit, int n)
{
    using A = Additive_Share<DATATYPE, Share>;
    pack_additive<0, BITLENGTH>(x, bit, n, LTZ<0, BITLENGTH, Share, DATATYPE>);
    for (int i = 0; i < n; i++)
        bit[i] = A(UINT_TYPE(1)) - bit[i];
}

// out = max(x, 0) and bit = [x >= 0] as 0/1 integer shares, kept for threshold_backward (a comparison, then a product)
template <typename A>
static void relu(const A* x, A* out, A* bit, int n)
{
    geq_zero(x, bit, n);
    for (int i = 0; i < n; i++)
    {
        out[i] = x[i].prepare_dot(bit[i]);
        out[i].mask_and_send_dot_without_trunc();
    }
    A::communicate();
    for (int i = 0; i < n; i++)
        out[i].complete_mult_without_trunc();
}

// out = g * bit for 0/1 integer shares bit (no truncation)
template <typename A>
static void relu_backward(const A* g, const A* bit, A* out, int n)
{
    for (int i = 0; i < n; i++)
    {
        out[i] = g[i].prepare_dot(bit[i]);
        out[i].mask_and_send_dot_without_trunc();
    }
    A::communicate();
    for (int i = 0; i < n; i++)
        out[i].complete_mult_without_trunc();
}

// out = c * bit for a public c, local
template <typename A>
static void relu_backward_public(double c, const A* bit, A* out, int n)
{
    const UINT_TYPE cf = to_fx(c);
    for (int i = 0; i < n; i++)
        out[i] = bit[i].mult_public(cf);
}

// out = c[0] + c[1] x + ... + c[N-1] x^(N-1) for |x| <= 1
template <typename A, size_t N>
static void poly(const A* x, const double (&c)[N], A* out, int n)
{
    std::vector<A> y(n);
    mul_public(x, c[N - 1], y.data(), n);
    for (int k = N - 2; k >= 0; k--)
    {
        const A coefficient(to_fx(c[k]));
        for (int i = 0; i < n; i++)
            y[i] = y[i] + coefficient;
        if (k > 0)
            mul(y.data(), x, y.data(), n);
    }
    std::copy(y.begin(), y.end(), out);
}

constexpr double LN2 = 0.6931471805599453;
// x >= 2^NORM_MIN and x < 2^NORM_MAX for x's normalized by normalize: (1 / x) 2^(NORM_MAX + FRACTIONAL)
constexpr int NORM_MIN = -FRACTIONAL;
constexpr int NORM_MAX = BITLENGTH - 3 - 2 * FRACTIONAL;
static_assert(NORM_MAX > 0, "normalize needs BITLENGTH > 2 * FRACTIONAL + 3");

constexpr double EXP_MIN = std::max(-16.0, -FRACTIONAL * LN2);
constexpr double EXP_MAX = std::min(16.0, (NORM_MAX - 1) * LN2);
// Chebyshev interpolants, lowest degree first: e^t for |t| <= 1, log(1.5 + s) for |s| <= 0.5
constexpr double EXP_POLY[] = {0.999999801887,
                                   0.999999978076,
                                   0.500006337442,
                                   0.166667368038,
                                   0.0416350384549,
                                   0.00832983182561,
                                   0.00143925671568,
                                   0.0002039935802};
constexpr double LOG_POLY[] = {0.405465108108,
                                   0.666666175543,
                                   -0.222221926388,
                                   0.0987914454491,
                                   -0.0493983836295,
                                   0.0259697921024,
                                   -0.0144105528948,
                                   0.0102381797537,
                                   -0.00600660504495};
// and 1 / sqrt(1.5 + s) for |s| <= 0.5 (error 6e-8)
constexpr double RSQRT_POLY[] = {0.816496580928,
                                     -0.272164841733,
                                     0.136082328604,
                                     -0.0756378043142,
                                     0.0441239119805,
                                     -0.0259489426435,
                                     0.0158457249217,
                                     -0.0126100713735,
                                     0.0079053081087};

// out = e^x: clamp x, then a polynomial for e^(x/16) and four squarings
template <typename A>
static void exp(const A* x, A* out, int n)
{
    std::vector<A> t(2 * n), r(2 * n), bit(2 * n);
    const A lo(to_fx(EXP_MIN)), hi(to_fx(EXP_MAX));
    for (int i = 0; i < n; i++)
    {
        t[i] = x[i] - lo;
        t[n + i] = x[i] - hi;
    }
    relu(t.data(), r.data(), bit.data(), 2 * n);
    for (int i = 0; i < n; i++)
        t[i] = lo + r[i] - r[n + i];  // lo + max(x - lo, 0) - max(x - hi, 0)
    trunc(t.data(), n, 4);
    poly(t.data(), EXP_POLY, out, n);
    for (int k = 0; k < 4; k++)
        mul(out, out, out, n);
}

// For x > 0 in [2^NORM_MIN, 2^NORM_MAX), with k = floor(log2 x): m = x / 2^k in [1, 2], and as integer shares k
// and v = 2^(NORM_MAX - k). Compares x with every power of two at once, all in the rounds of one comparison.
template <typename A>
static void normalize(const A* x, A* m, A* k, A* v, int n, std::vector<A>* bits = nullptr)
{
    const int powers = NORM_MAX - NORM_MIN;  // 2^(NORM_MIN + 1), ..., 2^NORM_MAX
    std::vector<A> t(n * powers), own;
    std::vector<A>& bit = bits ? *bits : own;
    bit.resize(n * powers);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < powers; j++)
            t[i * powers + j] = x[i] - A(UINT_TYPE(1) << (FRACTIONAL + NORM_MIN + 1 + j));
    geq_zero(t.data(), bit.data(), n * powers);
    for (int i = 0; i < n; i++)
    {
        k[i] = A(UINT_TYPE(INT_TYPE(NORM_MIN)));
        v[i] = A(UINT_TYPE(1) << powers);
        for (int j = 0; j < powers; j++)
        {
            k[i] = k[i] + bit[i * powers + j];
            v[i] = v[i] - bit[i * powers + j].mult_public(UINT_TYPE(1) << (powers - 1 - j));
        }
    }
    mul(x, v, m, n, NORM_MAX);
}

// out = log(m) for m in [1, 2]: no comparison
template <typename A>
static void log_mantissa(const A* m, A* out, int n)
{
    std::vector<A> s(n);
    const A mid(to_fx(1.5));
    for (int i = 0; i < n; i++)
        s[i] = m[i] - mid;
    poly(s.data(), LOG_POLY, out, n);
}

// out = log(x) = k log(2) + log(m) for x > 0
template <typename A>
static void log(const A* x, A* out, int n)
{
    std::vector<A> m(n), k(n), v(n);
    normalize(x, m.data(), k.data(), v.data(), n);
    log_mantissa(m.data(), out, n);
    const UINT_TYPE ln2 = to_fx(LN2);
    for (int i = 0; i < n; i++)
        out[i] = out[i] + k[i].mult_public(ln2);
}


// out = sqrt(x) for x = 0 or x in [2^NORM_MIN, 2^NORM_MAX): with normalize's x = m 2^k, sqrt(m) = m / sqrt(m) by a
// polynomial (0 at x = 0, where m = 0), times 2^(k/2), a public function of k and so a sum of public multiples of
// normalize's 0/1 comparisons (local)
template <typename A>
static void sqrt(const A* x, A* out, int n)
{
    const int powers = NORM_MAX - NORM_MIN, h = 1;  // 2^(k/2 + h) with h bits more, so its rounding stays below 1 ulp
    std::vector<A> m(n), k(n), v(n), bit, s(n), r(n), e(n);
    normalize(x, m.data(), k.data(), v.data(), n, &bit);
    const UINT_TYPE lowest = to_fx(std::pow(2.0, NORM_MIN / 2.0 + h));
    std::vector<UINT_TYPE> step(powers);  // differences of the rounded 2^(k/2 + h), so their sums are exact
    UINT_TYPE previous = lowest;
    for (int j = 0; j < powers; j++)
    {
        const UINT_TYPE next = to_fx(std::pow(2.0, (NORM_MIN + 1 + j) / 2.0 + h));
        step[j] = next - previous;
        previous = next;
    }
    const A first(lowest), mid(to_fx(1.5));
    for (int i = 0; i < n; i++)
    {
        e[i] = first;
        for (int j = 0; j < powers; j++)
            e[i] = e[i] + bit[i * powers + j].mult_public(step[j]);
        s[i] = m[i] - mid;
    }
    poly(s.data(), RSQRT_POLY, r.data(), n);
    mul(m.data(), r.data(), s.data(), n);
    mul(s.data(), e.data(), out, n, FRACTIONAL + h);
}

// y = 1 / m by three Newton steps from the best linear guess on [1, 2] (first error 1/17, then 1/17^8), and v, for x > 0
template <typename A>
static void inverse(const A* x, A* y, A* v, int n)
{
    std::vector<A> m(n), k(n), e(n);
    normalize(x, m.data(), k.data(), v, n);
    mul_public(m.data(), -8.0 / 17, y, n);
    const A first(to_fx(24.0 / 17)), two(to_fx(2.0));
    for (int i = 0; i < n; i++)
        y[i] = y[i] + first;
    for (int step = 0; step < 3; step++)
    {
        mul(m.data(), y, e.data(), n);
        for (int i = 0; i < n; i++)
            e[i] = two - e[i];
        mul(y, e.data(), y, n);
    }
}

// out = 1 / x for x > 0
template <typename A>
static void reciprocal(const A* x, A* out, int n)
{
    std::vector<A> y(n), v(n);
    inverse(x, y.data(), v.data(), n);
    mul(y.data(), v.data(), out, n, NORM_MAX);
}

// out = a / b for b > 0: a / m first, so a large b does not lose a's digits
template <typename A>
static void div(const A* a, const A* b, A* out, int n)
{
    std::vector<A> y(n), v(n);
    inverse(b, y.data(), v.data(), n);
    mul(a, y.data(), out, n);
    mul(out, v.data(), out, n, NORM_MAX);
}

// out = log(e^a + e^b) = max(a, b) + log(1 + e^-|a - b|), in range even where e^a overflows
template <typename A>
static void logaddexp(const A* a, const A* b, A* out, int n)
{
    std::vector<A> d(n), r(n), bit(n);
    for (int i = 0; i < n; i++)
        d[i] = a[i] - b[i];
    relu(d.data(), r.data(), bit.data(), n);
    for (int i = 0; i < n; i++)
        d[i] = d[i] - r[i].mult_public(UINT_TYPE(2));  // -|a - b|
    exp(d.data(), d.data(), n);
    const A one(to_fx(1.0));
    for (int i = 0; i < n; i++)
        d[i] = d[i] + one;
    log_mantissa(d.data(), out, n);
    for (int i = 0; i < n; i++)
        out[i] = out[i] + b[i] + r[i];
}

// w[k] -= lr * g[k] for every tensor k, one truncation for all of them
template <typename A>
static void sgd(std::vector<std::vector<A>>& w, const std::vector<std::vector<A>>& g, double lr)
{
    const UINT_TYPE cf = to_fx(lr);
    std::vector<A> step;
    for (size_t k = 0; k < w.size(); k++)
        for (size_t i = 0; i < w[k].size(); i++)
            step.push_back(g[k][i].mult_public(cf));
    trunc(step.data(), step.size());
    size_t j = 0;
    for (size_t k = 0; k < w.size(); k++)
        for (size_t i = 0; i < w[k].size(); i++)
            w[k][i] = w[k][i] - step[j++];
}

// Adam:
template <typename A>
static void adam(std::vector<std::vector<A>>& w, const std::vector<std::vector<A>>& g, std::vector<std::vector<A>>& m,
                 std::vector<std::vector<A>>& v, int t, double lr, double b1, double b2, double eps, int scale)
{
    if (m.empty())
        for (auto& x : w)
        {
            m.emplace_back(x.size(), A(UINT_TYPE(0)));
            v.emplace_back(x.size(), A(UINT_TYPE(0)));
        }
    std::vector<A> G, MV;  // all tensors in one, m^ then v^, so every step below runs in the rounds of one
    for (size_t k = 0; k < w.size(); k++)
    {
        G.insert(G.end(), g[k].begin(), g[k].end());
        MV.insert(MV.end(), m[k].begin(), m[k].end());
    }
    for (size_t k = 0; k < w.size(); k++)
        MV.insert(MV.end(), v[k].begin(), v[k].end());
    const int n = G.size();
    A *M = MV.data(), *V = MV.data() + n;
    if (scale >= 0)
        for (int i = 0; i < n; i++)
            G[i] = G[i].mult_public(UINT_TYPE(1) << scale);
    else
        trunc(G.data(), n, -scale);

    // m^ = a1 m^ + (1 - a1) g', v^ = a2 v^ + (1 - a2) g'^2, with 1 - a exact so v^ <= max g'^2
    const UINT_TYPE one = UINT_TYPE(1) << FRACTIONAL;
    const UINT_TYPE a1 = to_fx(b1 * (1 - std::pow(b1, t - 1)) / (1 - std::pow(b1, t)));
    const UINT_TYPE a2 = to_fx(b2 * (1 - std::pow(b2, t - 1)) / (1 - std::pow(b2, t)));
    std::vector<A> G2(n), D(n), U(n);
    mul(G.data(), G.data(), G2.data(), n);
    for (int i = 0; i < n; i++)
    {
        M[i] = M[i].mult_public(a1) + G[i].mult_public(one - a1);
        V[i] = V[i].mult_public(a2) + G2[i].mult_public(one - a2);
    }
    trunc(MV.data(), 2 * n);

    // u = m^ / (sqrt(v^) + eps 2^scale), a few at most in size
    sqrt(V, D.data(), n);
    const A e(to_fx(std::ldexp(eps, scale)));
    for (int i = 0; i < n; i++)
        D[i] = D[i] + e;
    div(M, D.data(), U.data(), n);

    // w -= lr u, with lr = c 2^p and c in [0.5, 1) so a small lr keeps all its digits
    int p;
    const UINT_TYPE c = to_fx(std::frexp(lr, &p));
    for (int i = 0; i < n; i++)
        U[i] = U[i].mult_public(c);
    trunc(U.data(), n, FRACTIONAL - p);
    int j = 0;
    for (size_t k = 0; k < w.size(); k++)
        for (size_t i = 0; i < w[k].size(); i++, j++)
        {
            w[k][i] = w[k][i] - U[j];
            m[k][i] = M[j];
            v[k][i] = V[j];
        }
}

// out = x, revealed to every party
template <typename A>
static void reveal(const A* x, double* out, int n)
{
    for (int i = 0; i < n; i++)
        x[i].prepare_reveal_to_all();
    A::communicate();
    for (int i = 0; i < n; i++)
    {
        DATATYPE dat = x[i].complete_reveal_to_all();
        UINT_TYPE value;
        unorthogonalize_arithmetic(&dat, &value, 1);
        out[i] = to_fl(value);
    }
}