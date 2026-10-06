#include "prover.hpp"
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utils.hpp>

using namespace mcl::bn;
using std::unique_ptr;

namespace {

void require_prefix_vector_bound(const zkgpt::NativeVector<linear_poly>& values,
                                 std::size_t expected_size, const char* name) {
    if (values.size() != expected_size)
        throw std::logic_error(std::string("native GKR prefix vector has an unexpected size: ") + name);
    // Every phase releases its prior buffers before resizing from empty. A
    // larger capacity would invalidate the exact source-derived formal
    // admission bound, so fail closed if this STL allocates past the request.
    if (values.capacity() != expected_size)
        throw std::logic_error(std::string("native GKR prefix vector capacity differs from its requested size: ") + name);
}

void resize_prefix_vector_exact(zkgpt::NativeVector<linear_poly>& values,
                                std::size_t requested_size, const char* name) {
    if (!values.empty() || values.capacity() != 0)
        throw std::logic_error(std::string("native GKR prefix buffer was not released before resize: ") + name);
    values.resize(requested_size);
    require_prefix_vector_bound(values, requested_size, name);
}

} // namespace

linear_poly interpolate(const F &zero_v, const F &one_v) 
{
    return {one_v - zero_v, zero_v};
}

F prover::getCirValue(u8 layer_id, const zkgpt::NativeVector<u64> &ori, u64 u) {
    if (native_values != nullptr) {
        if (!layer_id && native_input_value_at) return native_input_value_at(ori.at(u));
        if (native_terminal_values != nullptr && static_cast<std::size_t>(layer_id) + 1 == native_values->size())
            return native_terminal_values->at(static_cast<std::size_t>(u));
        return !layer_id ? native_values->at(0).at(ori.at(u)) : native_values->at(layer_id).at(u);
    }
    return !layer_id ? val[0][ori[u]] : val[layer_id][u];
}

void prover::init()
{
    proof_size = 0;
    r_u.resize(activeCircuit().size + 1);
    r_v.resize(activeCircuit().size + 1);

}

std::size_t prover::sumcheck_storage_capacity_bytes() const {
    std::size_t bytes = 0;
    const auto add_capacity = [&bytes](const auto& values) {
        if (values.capacity() > std::numeric_limits<std::size_t>::max() / sizeof(linear_poly))
            throw std::overflow_error("native GKR sumcheck storage byte count overflows");
        const std::size_t current = values.capacity() * sizeof(linear_poly);
        if (current > std::numeric_limits<std::size_t>::max() - bytes) throw std::overflow_error("native GKR sumcheck storage byte count overflows");
        bytes += current;
    };
    for (unsigned lane = 0; lane < 2; ++lane) {
        add_capacity(V_mult[lane]);
        add_capacity(mult_array[lane]);
        add_capacity(tmp_V_mult[lane]);
        add_capacity(tmp_mult_array[lane]);
    }
    return bytes;
}

void prover::release_sumcheck_storage() {
    for (unsigned lane = 0; lane < 2; ++lane) {
        zkgpt::NativeVector<linear_poly>().swap(V_mult[lane]);
        zkgpt::NativeVector<linear_poly>().swap(mult_array[lane]);
        zkgpt::NativeVector<linear_poly>().swap(tmp_V_mult[lane]);
        zkgpt::NativeVector<linear_poly>().swap(tmp_mult_array[lane]);
    }
}

/**
 * This is to initialize all process.
 *
 * @param the random point to be evaluated at the output layer
 */
void prover::sumcheckInitAll(const vector<F>::const_iterator &r_0_from_v) 
{
    sumcheck_id = activeCircuit().size;
    i8 last_bl = activeCircuit().circuit[sumcheck_id - 1].bit_length;
    r_u[sumcheck_id].resize(last_bl);
    prove_timer.start();
    for (int i = 0; i < last_bl; ++i) 
        r_u[sumcheck_id][i] = r_0_from_v[i];
    prove_timer.stop();
}

/**
 * This is to initialize before the process of a single layer.
 *
 * @param the random combination coefficiants for multiple reduction points
 */
void prover::sumcheckInit(const F &alpha_0, const F &beta_0)
{
    if (sumcheck_id == 0) throw std::logic_error("native sumcheck layer index underflow");
    prove_timer.start();
    alpha = alpha_0;
    beta = beta_0;
    beta_point_0 = r_u.at(sumcheck_id);
    beta_point_1 = r_v.at(sumcheck_id);
    --sumcheck_id;
    prove_timer.stop();
}

void prover::sumcheckInit(const F &alpha_0, const F &beta_0,
                          const vector<F> &current_u_point,
                          const vector<F> &current_v_point)
{
    if (sumcheck_id == 0) throw std::logic_error("native sumcheck layer index underflow");
    prove_timer.start();
    alpha = alpha_0;
    beta = beta_0;
    beta_point_0 = current_u_point;
    beta_point_1 = current_v_point;
    --sumcheck_id;
    prove_timer.stop();
}
static ThreadSafeQueue<int> workerq,endq;

void sc_phase1_uni_worker(const layer& current, const zkgpt::NativeVector<uniGate> &beg, zkgpt::NativeVector<linear_poly> (&mult_array)[2],const zkgpt::native_gkr::CombinedEqualityWeightProvider& beta_g,u64*&L,u64*&R)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g.at(gate.g) * current.coefficient(gate);
        }
        endq.Push(idx);
    }
}

void sc_phase1_bin_worker(const layer& cur, const zkgpt::NativeVector<binGate> &beg, zkgpt::NativeVector<linear_poly> (&mult_array)[2],vector<vector<F> >& val,const zkgpt::native_gkr::CombinedEqualityWeightProvider& beta_g,u64*&L,u64*&R,u8 sumcheck_id)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv =  !gate.getLayerIdV(sumcheck_id) ? val[0][cur.ori_id_v[gate.v]] : val[gate.getLayerIdV(sumcheck_id)][gate.v];
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g.at(gate.g) * cur.coefficient(gate);  // Ahg for phase 1
        }
        endq.Push(idx);
    }
}

void prover::sumcheckInitPhase1(const F &relu_rou_0) 
{
    //fprintf(stderr, "sumcheck level %d, phase1 init start\n", sumcheck_id);
    auto &cur = activeCircuit().circuit[sumcheck_id];
    total[0] = zkgpt::native_gkr::checked_domain_size(cur.bit_length_u[0], "native GKR phase1 lane 0");
    total_size[0] = cur.size_u[0];
    total[1] = zkgpt::native_gkr::checked_domain_size(cur.bit_length_u[1], "native GKR phase1 lane 1");
    total_size[1] = cur.size_u[1];

    r_u[sumcheck_id].resize(cur.max_bl_u);
    for (unsigned lane = 0; lane < 2; ++lane) {
        if (total_size[lane] > total[lane]) throw std::logic_error("native GKR phase1 active prefix exceeds its domain");
    }
    for (unsigned lane = 0; lane < 2; ++lane) {
        const std::size_t active_extent = zkgpt::native_gkr::checked_extent(total_size[lane], "native GKR phase1 active prefix");
        const std::size_t fold_extent = zkgpt::native_gkr::checked_extent(std::max<u64>(1, total_size[lane] / 2 + total_size[lane] % 2), "native GKR phase1 fold prefix");
        resize_prefix_vector_exact(V_mult[lane], std::max<std::size_t>(1, active_extent), "phase1 values");
        resize_prefix_vector_exact(mult_array[lane], std::max<std::size_t>(1, active_extent), "phase1 multipliers");
        resize_prefix_vector_exact(tmp_V_mult[lane], fold_extent, "phase1 folded values");
        resize_prefix_vector_exact(tmp_mult_array[lane], fold_extent, "phase1 folded multipliers");
        if (active_extent == 0) { V_mult[lane][0].clear(); mult_array[lane][0].clear(); }
    }
    timer useless_t;
    useless_t.start();
    relu_rou = relu_rou_0;
    add_term.clear();
    
    for (int b = 0; b < 2; ++b)
        for (u64 u = 0; u < total_size[b]; ++u)
            mult_array[b][u].clear();
    
    
        for (int b = 0; b < 2; ++b) 
        {
            auto dep = !b ? 0 : sumcheck_id - 1;
            for (u64 u = 0; u < total_size[b]; ++u) 
            {
                V_mult[b][u] = getCirValue(dep, cur.ori_id_u, u);
            }
        }
    useless_t.stop();
    throw_time[sumcheck_id].push_back(useless_t.elapse_sec());
    prove_timer.start();
    
    const zkgpt::native_gkr::CombinedEqualityWeightProvider beta_g(beta_point_0, beta_point_1,
                                                       std::max<int>(0, cur.bit_length), alpha, beta);
    
    if(cur.uni_interval.size()>=2)
    {
        const int thd=32;
        std::vector<std::thread> workers;
        zkgpt::NativeVector<u64> L_values(cur.uni_interval.size()), R_values(cur.uni_interval.size());
        u64 *L=L_values.data(), *R=R_values.data();
        for (u64 j = 0; j <cur.uni_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.uni_interval[j].first;
                R[j]=cur.uni_interval[j].second;
        }
        for(int i=0;i<thd;i++)
        {
            workers.emplace_back(sc_phase1_uni_worker, std::cref(cur),std::ref(cur.uni_gates),std::ref(mult_array),std::cref(beta_g),std::ref(L),std::ref(R));
        }
        while(!workerq.Empty())
            this_thread::sleep_for (std::chrono::microseconds(1));
        while(endq.Size()!=cur.uni_interval.size())
            this_thread::sleep_for (std::chrono::microseconds(1));
        endq.Clear();
        for (auto& worker : workers) worker.join();
    }
    else for (auto &gate: cur.uni_gates) 
        {
            bool idx = gate.lu != 0;
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + beta_g.at(gate.g) * cur.coefficient(gate);
        }
    if(native_values == nullptr && cur.bin_interval.size()>=2)
    {
        const int thd=32;
        std::vector<std::thread> workers;
        zkgpt::NativeVector<u64> L_values(cur.bin_interval.size()), R_values(cur.bin_interval.size());
        u64 *L=L_values.data(), *R=R_values.data();
        for (u64 j = 0; j <cur.bin_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.bin_interval[j].first;
                R[j]=cur.bin_interval[j].second;
        }
        for(int i=0;i<thd;i++)
        {
            workers.emplace_back(sc_phase1_bin_worker, std::ref(cur),std::ref(cur.bin_gates),std::ref(mult_array),std::ref(val),std::cref(beta_g),std::ref(L),std::ref(R),sumcheck_id);
        }
        while(!workerq.Empty())
            this_thread::sleep_for (std::chrono::microseconds(1));
        while(endq.Size()!=cur.bin_interval.size())
            this_thread::sleep_for (std::chrono::microseconds(1));
        endq.Clear();
        for (auto& worker : workers) worker.join();
    }
    else  for (auto &gate: cur.bin_gates) 
        {
            bool idx = gate.getLayerIdU(sumcheck_id) != 0;
            auto val_lv = getCirValue(gate.getLayerIdV(sumcheck_id), cur.ori_id_v, gate.v);
            mult_array[idx][gate.u] = mult_array[idx][gate.u] + val_lv * beta_g.at(gate.g) * cur.coefficient(gate);  // Ahg for phase 1
        }
    round = 0;
    prove_timer.stop();
    //fprintf(stderr, "sumcheck level %d, phase1 init finished\n", sumcheck_id);
}


void sc_phase2_uni_worker(const layer& current, const zkgpt::NativeVector<uniGate> &beg, F& sum_value,F& V_u0,F&V_u1,const zkgpt::native_gkr::CombinedEqualityWeightProvider& beta_g,const zkgpt::native_gkr::EqualityWeightProvider& beta_u,u64*&L,u64*&R)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];
        F ss;
        ss.clear();
        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            auto V_u = !gate.lu ? V_u0 : V_u1;                  //V_u0 is claim 0, V_u1 is claim 1
            ss +=beta_g.at(gate.g) * beta_u.at(gate.u) * V_u * current.coefficient(gate);
        }
        sum_value+=ss;
        endq.Push(idx);
    }
}

void sc_phase2_bin_worker(const layer& current, const zkgpt::NativeVector<binGate> &beg, zkgpt::NativeVector<linear_poly> (&mult_array)[2],F& V_u0,F&V_u1,const zkgpt::native_gkr::CombinedEqualityWeightProvider& beta_g,const zkgpt::native_gkr::EqualityWeightProvider& beta_u,u64*&L,u64*&R,u8 sumcheck_id)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];

        for (size_t i = l; i <r ; ++i) 
        {
            auto &gate = beg[i];
            bool idx = gate.getLayerIdV(sumcheck_id);
            auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
            mult_array[idx][gate.v] =mult_array[idx][gate.v]+beta_g.at(gate.g) * beta_u.at(gate.u) * V_u * current.coefficient(gate);
        }
        endq.Push(idx);
    }
}
void prover::sumcheckInitPhase2() 
{
    //fprintf(stderr, "sumcheck level %d, phase2 init start\n", sumcheck_id);
    auto &cur = activeCircuit().circuit[sumcheck_id];
    total[0] = zkgpt::native_gkr::checked_domain_size(cur.bit_length_v[0], "native GKR phase2 lane 0");
    total_size[0] = cur.size_v[0];
    total[1] = zkgpt::native_gkr::checked_domain_size(cur.bit_length_v[1], "native GKR phase2 lane 1");
    total_size[1] = cur.size_v[1];
    i8 fft_bl = cur.fft_bit_length;
    i8 cnt_bl = cur.max_bl_v;

    timer useless_time;
    useless_time.start();
    r_v[sumcheck_id].resize(cur.max_bl_v);
    for (unsigned lane = 0; lane < 2; ++lane) {
        if (total_size[lane] > total[lane]) throw std::logic_error("native GKR phase2 active prefix exceeds its domain");
    }
    for (unsigned lane = 0; lane < 2; ++lane) {
        const std::size_t active_extent = zkgpt::native_gkr::checked_extent(total_size[lane], "native GKR phase2 active prefix");
        const std::size_t fold_extent = zkgpt::native_gkr::checked_extent(std::max<u64>(1, total_size[lane] / 2 + total_size[lane] % 2), "native GKR phase2 fold prefix");
        resize_prefix_vector_exact(V_mult[lane], std::max<std::size_t>(1, active_extent), "phase2 values");
        resize_prefix_vector_exact(mult_array[lane], std::max<std::size_t>(1, active_extent), "phase2 multipliers");
        resize_prefix_vector_exact(tmp_V_mult[lane], fold_extent, "phase2 folded values");
        resize_prefix_vector_exact(tmp_mult_array[lane], fold_extent, "phase2 folded multipliers");
        if (active_extent == 0) { V_mult[lane][0].clear(); mult_array[lane][0].clear(); }
    }


    add_term.clear();
    for (int b = 0; b < 2; ++b) 
    {
        for (u64 v = 0; v < total_size[b]; ++v)
            mult_array[b][v].clear();
    }
    useless_time.stop();
    throw_time[sumcheck_id].push_back(useless_time.elapse_sec());
    prove_timer.start();
    const zkgpt::native_gkr::CombinedEqualityWeightProvider beta_g(beta_point_0, beta_point_1,
                                                       std::max<int>(0, cur.bit_length), alpha, beta);
    const zkgpt::native_gkr::EqualityWeightProvider beta_u(
        r_u.at(sumcheck_id), cur.max_bl_u, F_ONE);
    for (int b = 0; b < 2; ++b) 
    {
        auto dep = !b ? 0 : sumcheck_id - 1;
        for (u64 v = 0; v < total_size[b]; ++v) 
        {
            V_mult[b][v] = getCirValue(dep, cur.ori_id_v, v);
        }
    }
    
    if(cur.uni_interval.size()>=2)
    {
        const int thd=32;
        std::vector<std::thread> workers;
        F sum[40];
        
        zkgpt::NativeVector<u64> L_values(cur.uni_interval.size()), R_values(cur.uni_interval.size());
        u64 *L=L_values.data(), *R=R_values.data();
        for (u64 j = 0; j <cur.uni_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.uni_interval[j].first;
                R[j]=cur.uni_interval[j].second;
        }
            for(int i=0;i<thd;i++)
            {
                sum[i].clear(); 
                workers.emplace_back(sc_phase2_uni_worker,std::cref(cur),std::ref(cur.uni_gates),std::ref(sum[i]),std::ref(V_u0),std::ref(V_u1),std::cref(beta_g),std::cref(beta_u),std::ref(L),std::ref(R));
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=cur.uni_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            endq.Clear();
        for (auto& worker : workers) worker.join();
            for(int i=0;i<thd;i++)
                add_term+=sum[i];
    }
    else for (auto &gate: cur.uni_gates) 
    {
        auto V_u = !gate.lu ? V_u0 : V_u1;                  //V_u0 is claim 0, V_u1 is claim 1
        add_term = add_term + beta_g.at(gate.g) * beta_u.at(gate.u) * V_u * cur.coefficient(gate);
    }
    if(native_values == nullptr && cur.bin_interval.size()>=2)
    {
        const int thd=32;
        std::vector<std::thread> workers;
        zkgpt::NativeVector<u64> L_values(cur.bin_interval.size()), R_values(cur.bin_interval.size());
        u64 *L=L_values.data(), *R=R_values.data();
        for (u64 j = 0; j <cur.bin_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur.bin_interval[j].first;
                R[j]=cur.bin_interval[j].second;
        }
            for(int i=0;i<thd;i++)
            {
                workers.emplace_back(sc_phase2_bin_worker,std::cref(cur),std::ref(cur.bin_gates),std::ref(mult_array),std::ref(V_u0),std::ref(V_u1),std::cref(beta_g),std::cref(beta_u),std::ref(L),std::ref(R),sumcheck_id);
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=cur.bin_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            endq.Clear();
        for (auto& worker : workers) worker.join();
    }
    else for (auto &gate: cur.bin_gates) 
    {
        bool idx = gate.getLayerIdV(sumcheck_id);
        auto V_u = !gate.getLayerIdU(sumcheck_id) ? V_u0 : V_u1;
        mult_array[idx][gate.v] = mult_array[idx][gate.v] + beta_g.at(gate.g) * beta_u.at(gate.u) * V_u * cur.coefficient(gate);
    }

    round = 0;
    prove_timer.stop();
}

void prover::sumcheckLassoInit(const vector<F> &s_u, const vector<F> &s_v,const vector<vector<F>>& r_uu, const vector<vector<F>>& r_vv) 
{
    
    sumcheck_id = 0;
    total[1] = (1ULL << activeCircuit().circuit[sumcheck_id].bit_length);
    total_size[1] = activeCircuit().circuit[sumcheck_id].size;
    lasso_mult_v.assign(total[1], F_ZERO);
    V_mult[1].resize(total[1]);
    mult_array[1].resize(total[1]);
    tmp_V_mult[1].resize(std::max<u64>(1, total[1] >> 1));
    tmp_mult_array[1].resize(std::max<u64>(1, total[1] >> 1));

    r_u[0].resize(activeCircuit().circuit[0].bit_length);
    timer ggg;
    ggg.start();

    for (u8 i = sumcheck_id + 1; i < activeCircuit().size; ++i) 
    {
        i8 bit_length_i = activeCircuit().circuit[i].bit_length_u[0];
        u64 size_i = activeCircuit().circuit[i].size_u[0];
        //timer a,b;
        if (~bit_length_i) 
        {
            r_u[i].resize(activeCircuit().circuit[i].max_bl_u);
            for(int j=0;j<activeCircuit().circuit[i].max_bl_u;j++)
                r_u[i][j]=r_uu[i][j];
            const zkgpt::native_gkr::EqualityWeightProvider beta(
                r_u[i], bit_length_i, s_u[i - 1]);
            for (u64 hu = 0; hu < size_i; ++hu) 
            {
                u64 u = activeCircuit().circuit[i].ori_id_u[hu];
                lasso_mult_v[u] += beta.at(hu);
            }
        }
        bit_length_i = activeCircuit().circuit[i].bit_length_v[0];
        size_i = activeCircuit().circuit[i].size_v[0];
        if (~bit_length_i) 
        {
            r_v[i].resize(activeCircuit().circuit[i].max_bl_v);
            for(int j=0;j<activeCircuit().circuit[i].max_bl_v;j++)
                r_v[i][j]=r_vv[i][j];
            const zkgpt::native_gkr::EqualityWeightProvider beta(
                r_v[i], bit_length_i, s_v[i - 1]);
            for (u64 hv = 0; hv < size_i; ++hv) 
            {
                u64 v = activeCircuit().circuit[i].ori_id_v[hv];
                lasso_mult_v[v] += beta.at(hv);
            }
        }
    }
    round = 0;
    ggg.stop();
}

quadratic_poly prover::sumcheckUpdate1(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_u[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate2(const F &previous_random) {
    return sumcheckUpdate(previous_random, r_v[sumcheck_id]);
}

quadratic_poly prover::sumcheckUpdate(const F &previous_random, vector<F> &r_arr) 
{
    prove_timer.start();

    if (round) r_arr.at(round - 1) = previous_random;
    ++round;
    quadratic_poly ret;

    add_term = add_term * (F_ONE - previous_random);
    for (int b = 0; b < 2; ++b)
        ret = ret + sumcheckUpdateEach(previous_random, b);
    ret = ret + quadratic_poly(F_ZERO, -add_term, add_term);

    prove_timer.stop();
    proof_size += F_BYTE_SIZE * 3;
    return ret;
}



void sumcheckUpdate_worker(quadratic_poly &sum,zkgpt::NativeVector<linear_poly> &tmp_v,zkgpt::NativeVector<linear_poly> &tmp_mult,zkgpt::NativeVector<linear_poly> &tmp_v_2,zkgpt::NativeVector<linear_poly> &tmp_mult_2,u64*&L,u64*&R,Fr previous_random,u64 active_extent)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];
        for(u64 i=l;i<r;i++)
        {
            tmp_v_2[i] = zkgpt::native_gkr::fold_prefix_linear(tmp_v, active_extent, i, previous_random);
            tmp_mult_2[i] = zkgpt::native_gkr::fold_prefix_linear(tmp_mult, active_extent, i, previous_random);
            sum = sum + tmp_mult_2[i] * tmp_v_2[i];
        }
        endq.Push(idx);
    }
}
quadratic_poly prover::sumcheckUpdateEach(const F &previous_random, bool idx) 
{
    auto &tmp_mult = mult_array[idx];
    auto &tmp_v = V_mult[idx];
    auto &tmp_mult_2 = tmp_mult_array[idx];
    auto &tmp_v_2 = tmp_V_mult[idx];

    if (total[idx] == 1) 
    {
        tmp_v[0] = tmp_v[0].eval(previous_random);
        tmp_mult[0] = tmp_mult[0].eval(previous_random);
        add_term = add_term + tmp_v[0].b * tmp_mult[0].b;
    }

    quadratic_poly ret;
    ret.clear();
    const u64 fold_count = total_size[idx] / 2 + total_size[idx] % 2;
    if(fold_count < (1ULL<<15))
    {
        for (u64 i = 0; i < fold_count; ++i) 
        {
            tmp_v[i] = zkgpt::native_gkr::fold_prefix_linear(tmp_v, total_size[idx], i, previous_random);
            tmp_mult[i] = zkgpt::native_gkr::fold_prefix_linear(tmp_mult, total_size[idx], i, previous_random);
            ret = ret + tmp_mult[i] * tmp_v[i];
        }
    }
    else
    {
        timer tt_f1,tt_f2;
        tt_f1.start();
        const int k=10;
        std::vector<std::thread> workers;
        const u64 task_count = u64(1) << k;
        u64 total_work=fold_count;
        zkgpt::NativeVector<u64> L_values(static_cast<std::size_t>(task_count)), R_values(static_cast<std::size_t>(task_count));
        u64 *L=L_values.data(), *R=R_values.data();
        const int thd=32;
        const u64 base = total_work / task_count;
        const u64 remainder = total_work % task_count;
        for (u64 j = 0; j < task_count; ++j) 
        {
            workerq.Push(j);
            L[j]=base*j+std::min(j, remainder);
            R[j]=base*(j+1)+std::min(j+1, remainder);
        }
        quadratic_poly qp[thd];
        for(int j=0;j<thd;j++)
        {
            workers.emplace_back(sumcheckUpdate_worker,std::ref(qp[j]),std::ref(tmp_v),std::ref(tmp_mult),std::ref(tmp_v_2),std::ref(tmp_mult_2),std::ref(L),std::ref(R),previous_random,total_size[idx]);
        }
        while(endq.Size()!=task_count)
            this_thread::sleep_for(std::chrono::microseconds(1));
        endq.Clear();
        for (auto& worker : workers) worker.join();
        for(int j=0;j<thd;j++)
            ret=ret+qp[j];
        tt_f1.stop();
        tt_f2.start();
        for(u64 i=0;i<total_work;i++)
        {
            tmp_mult[i]=tmp_mult_2[i];
            tmp_v[i]=tmp_v_2[i];
        }
        tt_f2.stop();
    }
    
    total[idx] >>= 1;
    total_size[idx] = fold_count;

    return ret;
}


/**
 * This is to evaluate a multi-linear extension at a random point.
 *
 * @param the value of the array & random point & the size of the array & the size of the random point
 * @return sum of `values`, or 0.0 if `values` is empty.
 */
F prover::Vres(const vector<F>::const_iterator &r, u64 output_size, u8 r_size,int layer_id,u64 start) 
{
    prove_timer.start();

    vector<F> output(output_size);
    for (u64 i = 0; i < output_size; ++i)
        output[i] = val[layer_id][i+start];
    u64 whole = 1ULL << r_size;
    for (u8 i = 0; i < r_size; ++i) {
        for (u64 j = 0; j < (whole >> 1); ++j) {
            if (j > 0)
                output[j].clear();
            if ((j << 1) < output_size)
                output[j] = output[j << 1] * (F_ONE - r[i]);
            if ((j << 1 | 1) < output_size)
                output[j] = output[j] + output[j << 1 | 1] * (r[i]);
        }
        whole >>= 1;
    }
    F res = output[0];

    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
    return res;
}

void prover::sumcheckFinalize1(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    if (round > 0)
        r_u[sumcheck_id].at(round - 1) = previous_random;
    V_u0 = claim_0 = total[0] ? V_mult[0][0].eval(previous_random) : (~activeCircuit().circuit[sumcheck_id].bit_length_u[0]) ? V_mult[0][0].b : F_ZERO;
    V_u1 = claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : (~activeCircuit().circuit[sumcheck_id].bit_length_u[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    release_sumcheck_storage();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheckFinalize2(const F &previous_random, F &claim_0, F &claim_1) {
    prove_timer.start();
    if (round > 0)
        r_v[sumcheck_id].at(round - 1) = previous_random;
    const bool has_round = round > 0;
    claim_0 = total[0] ? (has_round ? V_mult[0][0].eval(previous_random) : V_mult[0][0].b) : (~activeCircuit().circuit[sumcheck_id].bit_length_v[0]) ? V_mult[0][0].b : F_ZERO;
    claim_1 = total[1] ? (has_round ? V_mult[1][0].eval(previous_random) : V_mult[1][0].b) : (~activeCircuit().circuit[sumcheck_id].bit_length_v[1]) ? V_mult[1][0].b : F_ZERO;
    prove_timer.stop();

    release_sumcheck_storage();
    proof_size += F_BYTE_SIZE * 2;
}

void prover::sumcheck_lasso_Finalize(const F &previous_random, F &claim_1) 
{
    prove_timer.start();
    r_u[sumcheck_id].at(round - 1) = previous_random;
    claim_1 = total[1] ? V_mult[1][0].eval(previous_random) : V_mult[1][0].b;
    prove_timer.stop();
    proof_size += F_BYTE_SIZE;
}



void prover::commitInput(const vector<G1> &gens,int thr) 
{
    int len;
    if (activeCircuit().circuit[0].size != (1ULL << activeCircuit().circuit[0].bit_length)) 
    {
        len=val[0].size();
        val[0].resize(1ULL << activeCircuit().circuit[0].bit_length);
        for (int i = activeCircuit().circuit[0].size; i < val[0].size(); ++i)
            val[0][i].clear();
    }
    
    int l=ceil(log2(val[0].size()));
    ll* vi=new ll[1<<l];
    memset(vi,0,sizeof(ll)*(1<<l));
    ll mx=-1e9,mn=1e9;
    for(int i=0;i<val[0].size();i++)
    {
        vi[i]=convert(val[0][i]);
        mx=max(mx,vi[i]);
        mn=min(mn,vi[i]);
        
    }

    
    
    Fr* dat=new Fr[1<<l];
    memset(dat,0,sizeof(Fr)*(1<<l));
    memcpy(dat,val[0].data(),sizeof(Fr)*val[0].size());
    G1* ret=prover_commit(vi,(G1*)gens.data(),l,thr);
    cc.comm=ret;
    cc.G=gens.back();
    cc.g=(G1*)gens.data();
    cc.l=l;
    cc.w=dat;
    cc.ww=vi;
    val[0].resize(len);
}

__int128 convert(Fr x)	
{	
    int sign=0;	
    Fr abs;	
    if(x.isNegative())	
    {	
        sign=1;	
        abs=-x;	
    }	
    else	
        abs=x;	

    uint8_t bf[16]={0};	 //64 bit
    int size=abs.getLittleEndian(bf,16);	
    ll V=0;	
    for(int j=size-1;j>=0;j--)	
        V=V*256+bf[j];	
    if(sign)	
        V=-V;	
    return V;	
}
