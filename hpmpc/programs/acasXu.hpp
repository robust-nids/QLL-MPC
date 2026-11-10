// /*
// Fine-tunes the official AcasXu network with QLL in MPC
// uses HPMPC's Trip protocol:
//     P1: holds the weights and bias
//     P2: holds the input and label
//     P0: helper party for 3PC protocols
// Build and run: scripts/acasXu.sh mac
// */

#include "../datatypes/Additive_Share.hpp"
#include "../datatypes/float_fixed_converter.hpp"
#include "../programs/functions/fixed_point_training.hpp"
#include "../programs/functions/generated/acasXu_generated.hpp"
#include <openssl/rand.h>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define FUNCTION acasXu_training
#define RESULTTYPE DATATYPE

// Define batch size
#ifndef BATCH_SIZE
#define BATCH_SIZE 256
#endif

// Define training iterations
#ifndef ITERS
#define ITERS 100
#endif

#ifndef LEARNING_RATE
#define LEARNING_RATE 1e-4
#endif

#ifndef DATA_POINTS  // P2's points, a multiple of BATCH_SIZE (9,984 of ART's 10,000)
#define DATA_POINTS 9984
#endif

#ifndef PROPERTY_POINTS  // P1's points of property 1's region, BATCH_SIZE of them per step, in turn
#define PROPERTY_POINTS 43200
#endif

#ifndef REPORT_EVERY  // reveal the loss to every party every REPORT_EVERY steps; 0: never
#define REPORT_EVERY 10
#endif

#ifndef SHUFFLE_SEED  // public: every party orders the data batches the same way
#define SHUFFLE_SEED 0
#endif

#if DATTYPE != BITLENGTH
#error "acasXu trains one model: compile with DATTYPE equal to BITLENGTH"
#endif

static_assert(BATCH_SIZE == BATCH, "rerun programs/qll/torch_to_hpmpc.py --batch BATCH_SIZE");
static_assert(DATA_POINTS % BATCH_SIZE == 0, "DATA_POINTS must be a multiple of BATCH_SIZE");

// the count values of a file in ACASXU_DIR (default: the working directory)
static std::vector<double> acasxu_read(const std::string& name, size_t count)
{
    const char* dir = std::getenv("ACASXU_DIR");
    const std::string path = std::string(dir ? dir : ".") + "/" + name;
    std::ifstream file(path);
    std::vector<double> values;
    double v;
    while (file >> v)
        values.push_back(v);
    if (values.size() != count)
    {
        std::cerr << "P" << PARTY << ": expected " << count << " values in " << path << ", found " << values.size()
                  << std::endl;
        exit(1);
    }
    return values;
}

// shares of n values held by party id; values is empty on the other parties
template <int id, typename A>
static std::vector<A> acasxu_prepare_input(const std::vector<double>& values, size_t n)
{
    std::vector<A> x(n);
    for (size_t i = 0; i < n; i++)
        x[i].template prepare_receive_from<id>(PROMOTE(i < values.size() ? to_fx(values[i]) : UINT_TYPE(0)));
    return x;
}

template <int id, typename A>
static void acasxu_complete_input(std::vector<A>& x)
{
    for (auto& s : x)
        s.template complete_receive_from<id>();
}

// x revealed to P1 only: P1 shares a uniform mask r, every party sees x + r, P1 subtracts r
template <typename A>
static std::vector<double> acasxu_reveal_to_p1(const std::vector<A>& x)
{
    const size_t n = x.size();
    std::vector<UINT_TYPE> r(n, 0);
#if PARTY == 1
    RAND_bytes(reinterpret_cast<unsigned char*>(r.data()), n * sizeof(UINT_TYPE));
#endif
    std::vector<A> masked(n);
    for (size_t i = 0; i < n; i++)
        masked[i].template prepare_receive_from<P_1>(PROMOTE(r[i]));
    A::communicate();
    for (size_t i = 0; i < n; i++)
    {
        masked[i].template complete_receive_from<P_1>();
        masked[i] = masked[i] + x[i];
        masked[i].prepare_reveal_to_all();
    }
    A::communicate();
    std::vector<double> out(n);
    for (size_t i = 0; i < n; i++)
    {
        DATATYPE dat = masked[i].complete_reveal_to_all();
        UINT_TYPE m;
        unorthogonalize_arithmetic(&dat, &m, 1);
        out[i] = to_fl(m - r[i]);  // the weight on P1, noise on the others
    }
    return out;
}

template <typename Share>
void acasXu_training(DATATYPE* res)
{
    using A = Additive_Share<DATATYPE, Share>;
    const int B = BATCH_SIZE, R = ROWS, IN = INPUTS, OUT = OUTPUTS;

    // P1: weights.inp (the official network, nn.py's parameter order) and property.inp; P2: data.inp
    int params = 0;
    for (int k = 0; k < PARAMS; k++)
        params += PARAM_SIZE[k];
    std::vector<double> weights, property, data;
#if PARTY == 1
    weights = acasxu_read("weights.inp", params);
    property = acasxu_read("property.inp", PROPERTY_POINTS * IN);
#endif
#if PARTY == 2
    data = acasxu_read("data.inp", DATA_POINTS * IN);
#endif
    std::vector<std::vector<A>> w(PARAMS);
    int offset = 0;
    for (int k = 0; k < PARAMS; k++)
    {
        std::vector<double> part;
        if (!weights.empty())
            part.assign(weights.begin() + offset, weights.begin() + offset + PARAM_SIZE[k]);
        w[k] = acasxu_prepare_input<P_1, A>(part, PARAM_SIZE[k]);
        offset += PARAM_SIZE[k];
    }
    std::vector<A> points = acasxu_prepare_input<P_2, A>(data, DATA_POINTS * IN);
    std::vector<A> region = acasxu_prepare_input<P_1, A>(property, PROPERTY_POINTS * IN);
    Share::communicate();
    for (auto& t : w)
        acasxu_complete_input<P_1>(t);
    acasxu_complete_input<P_2>(points);
    acasxu_complete_input<P_1>(region);

    // the data term's targets: the official network's scores of every data point, ROWS points per forward pass
    Saved<Share> saved;
    std::vector<A> X(R * IN), Y, dY, T(DATA_POINTS * OUT), Tb(B * OUT);
    for (int start = 0; start < DATA_POINTS; start += R)
    {
        const int rows = std::min(R, DATA_POINTS - start);
        std::fill(X.begin(), X.end(), A(UINT_TYPE(0)));
        std::copy(points.begin() + start * IN, points.begin() + (start + rows) * IN, X.begin());
        forward<Share>(w, X, Y, saved);
        std::copy(Y.begin(), Y.begin() + rows * OUT, T.begin() + start * OUT);
    }

    // SGD: each step one data batch, in a public order shuffled every epoch, and the next BATCH_SIZE property points
    const int batches = DATA_POINTS / B;
    std::vector<int> order(batches);
    std::mt19937 rng(SHUFFLE_SEED);
    std::vector<std::vector<A>> grads(PARAMS);
    for (int it = 0; it < ITERS; it++)
    {
        if (it % batches == 0)
        {
            for (int i = 0; i < batches; i++)
                order[i] = i;
            for (int i = batches - 1; i > 0; i--)  // not std::shuffle: its order differs between standard libraries
                std::swap(order[i], order[rng() % (i + 1)]);
        }
        const int b = order[it % batches];
        std::copy(points.begin() + b * B * IN, points.begin() + (b + 1) * B * IN, X.begin());
        for (int r = 0; r < B; r++)
        {
            const int p = (it * B + r) % PROPERTY_POINTS;
            std::copy(region.begin() + p * IN, region.begin() + (p + 1) * IN, X.begin() + (B + r) * IN);
        }
        std::copy(T.begin() + b * B * OUT, T.begin() + (b + 1) * B * OUT, Tb.begin());

        A value;
        forward<Share>(w, X, Y, saved);
        loss<Share>(Y, X, Tb, value, dY);
        backward<Share>(w, X, saved, dY, grads);
        update<Share>(w, grads, LEARNING_RATE);

        if (REPORT_EVERY > 0 && (it + 1) % REPORT_EVERY == 0)
        {
            double v;
            reveal(&value, &v, 1);  // every party learns the loss
            if (current_phase == PHASE_LIVE)
                std::cout << "P" << PARTY << ": ACASXU step " << it + 1 << " batch " << b << " loss "
                          << std::setprecision(10) << v << std::endl;
        }
    }

    // the fine-tuned weights, to P1 only, in the order of weights.inp
    std::vector<A> all;
    for (auto& t : w)
        all.insert(all.end(), t.begin(), t.end());
    std::vector<double> finetuned = acasxu_reveal_to_p1(all);
#if PARTY == 1
    if (current_phase == PHASE_LIVE)
    {
        const char* dir = std::getenv("ACASXU_DIR");
        std::ofstream file(std::string(dir ? dir : ".") + "/finetuned.inp");
        for (double v : finetuned)
            file << std::setprecision(12) << v << "\n";
        std::cout << "P1: ACASXU wrote finetuned.inp after " << ITERS << " steps, " << sending_rounds << " rounds"
                  << std::endl;
    }
#endif
}