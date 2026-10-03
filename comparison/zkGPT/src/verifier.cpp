//
// Created by 69029 on 3/9/2021.
//
#include "verifier.hpp"
#include "global_var.hpp"
#include "native_protocol.hpp"
#include <utils.hpp>

static bool masked_fconn_shape_matches(const layer &current, const NativeGkrMatmulShape &shape, std::string *reason);
#include <circuit.h>
#include <iostream>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <algorithm>
#include <limits>

using namespace std;
using namespace mcl::bn;
vector<F> beta_v;
static vector<F> beta_u, beta_gs;

namespace {

bool decode_transcript_field(const std::string &payload, F &value) {
    if (payload.empty() || payload.size() > 128) return false;
    try {
        value.setStr(payload, 16);
        return value.getStr(16) == payload;
    } catch (...) {
        return false;
    }
}

bool decode_transcript_fields(const std::string &payload,
                              std::vector<F> &values) {
    values.clear();
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t &value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned byte = 0; byte < 8; ++byte)
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(payload[offset++])) << (8 * byte);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count > (1ULL << 20)) return false;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        std::uint64_t size = 0;
        if (!read_u64(size) || size == 0 || size > 128 ||
            size > payload.size() - offset) return false;
        F value;
        if (!decode_transcript_field(
                payload.substr(offset, static_cast<std::size_t>(size)), value))
            return false;
        offset += static_cast<std::size_t>(size);
        values.push_back(value);
    }
    return offset == payload.size();
}

const NativeProofMessage *find_transcript_message(
    const NativeProofSession &session, NativeProofMessageKind kind,
    const std::string &label, std::string *reason) {
    const NativeProofMessage *found = nullptr;
    const std::vector<std::size_t> *positions = session.message_positions(label);
    if (positions == nullptr) {
        if (reason != nullptr) *reason = "missing GKR transcript message: " + label;
        return nullptr;
    }
    for (std::size_t position : *positions) {
        if (position >= session.messages().size()) {
            if (reason != nullptr)
                *reason = "GKR transcript label index is out of range: " + label;
            return nullptr;
        }
        const NativeProofMessage &message = session.messages()[position];
        if (message.kind != kind || message.label != label) continue;
        if (found != nullptr) {
            if (reason != nullptr) *reason = "duplicate GKR transcript message: " + label;
            return nullptr;
        }
        found = &message;
    }
    if (found == nullptr && reason != nullptr)
        *reason = "missing GKR transcript message: " + label;
    return found;
}

bool transcript_challenge(const NativeProofSession &session,
                          const std::string &label, F &value,
                          std::string *reason) {
    const NativeProofMessage *message = find_transcript_message(
        session, NativeProofMessageKind::Challenge, "challenge/" + label, reason);
    return message != nullptr && decode_transcript_field(message->payload, value);
}

bool transcript_claims(const NativeProofSession &session,
                       const std::string &label, std::size_t expected,
                       std::vector<F> &values, std::string *reason) {
    const NativeProofMessage *message = find_transcript_message(
        session, NativeProofMessageKind::FieldVector, label + "/claims", reason);
    if (message == nullptr || !decode_transcript_fields(message->payload, values) ||
        values.size() != expected) {
        if (reason != nullptr) *reason = "malformed GKR terminal claims: " + label;
        return false;
    }
    return true;
}

template<class Vector> F transcript_mle_eval(const Vector &values, const std::vector<F> &point) {
    if (point.empty() && values.size() != 1) throw std::invalid_argument("GKR scalar output domain is invalid");
    return zkgpt::native_mle_evaluate(values, point);
}

template<class Vector> void transcript_beta_phase1(const layer &current, std::size_t depth,
                            const std::vector<F> &r0,
                            const std::vector<F> &r1,
                            const std::vector<F> &ru,
                            const F &alpha, const F &beta,
                            const F &relu_rou,
                            Vector &beta_g,
                            Vector &beta_u) {
    beta_g.resize(std::size_t(1) << current.bit_length);
    initBetaTable(beta_g, current.bit_length, r0.begin(), r1.begin(),
                  alpha, beta);
    if (current.zero_start_id < current.size)
        for (u64 gate = current.zero_start_id;
             gate < (std::size_t(1) << current.bit_length); ++gate)
            beta_g[gate] *= relu_rou;
    beta_u.resize(std::size_t(1) << current.max_bl_u);
    initBetaTable(beta_u, current.max_bl_u, ru.begin(), F_ONE);
    (void)depth;
}

template<class Vector> F transcript_predicate_phase1(const layer &current,
                              const Vector &beta_g,
                              const Vector &beta_u,
                              std::vector<F> &uni,
                              std::vector<F> &bin) {
    uni.assign(2, F_ZERO);
    bin.assign(3, F_ZERO);
    for (const uniGate &gate : current.uni_gates)
        uni[gate.lu] += beta_g[gate.g] * beta_u[gate.u] * gate.sc;
    for (const binGate &gate : current.bin_gates)
        bin[gate.l] += beta_g[gate.g] * beta_u[gate.u] * gate.sc;
    return F_ZERO;
}

template<class Vector> void transcript_predicate_phase2(const layer &current,
                                 const Vector &beta_g,
                                 const Vector &beta_u,
                                 const Vector &beta_v,
                                 std::vector<F> &uni,
                                 std::vector<F> &bin) {
    if (!beta_v.empty()) {
        uni[0] *= beta_v[0];
        uni[1] *= beta_v[0];
    } else {
        uni[0].clear();
        uni[1].clear();
    }
    bin.assign(3, F_ZERO);
    for (const binGate &gate : current.bin_gates)
        bin[gate.l] += beta_g[gate.g] * beta_u[gate.u] * beta_v[gate.v] * gate.sc;
}

F transcript_final_value(const std::vector<F> &uni,
                         const std::vector<F> &bin,
                         const std::vector<F> &u,
                         const std::vector<F> &v) {
    return bin[0] * u[0] * v[0] + bin[1] * u[1] * v[1] +
           bin[2] * u[1] * v[0] + uni[0] * u[0] + uni[1] * u[1];
}

bool gkr_scope_message(const std::string &label) {
    return label.compare(0, sizeof("gkr/") - 1, "gkr/") == 0;
}

}

static bool masked_fconn_shape_matches(const layer &current,
                                       const NativeGkrMatmulShape &shape,
                                       std::string *reason) {
    if (shape.rows == 0 || shape.inner == 0 || shape.cols == 0 ||
        shape.rows > std::numeric_limits<std::size_t>::max() / shape.cols ||
        shape.rows * shape.cols != current.size) {
        if (reason != nullptr) *reason = "masked FCONN descriptor dimensions do not match layer";
        return false;
    }
    const bool layer_annotated = !current.native_relation_id.empty() ||
                                 !current.native_family_id.empty() ||
                                 current.native_rows != 0 ||
                                 current.native_inner != 0 ||
                                 current.native_cols != 0 ||
                                 current.native_lhs_offset != 0 ||
                                 current.native_rhs_offset != 0 ||
                                 current.native_output_offset != 0;
    if (!layer_annotated) return true;
    if (current.native_relation_id.empty() || current.native_family_id.empty() ||
        current.native_rows != shape.rows || current.native_inner != shape.inner ||
        current.native_cols != shape.cols ||
        current.native_relation_id != shape.relation_id ||
        current.native_family_id != shape.family_id ||
        current.native_lhs_offset != shape.lhs_offset ||
        current.native_rhs_offset != shape.rhs_offset ||
        current.native_output_offset != shape.output_offset) {
        if (reason != nullptr) *reason = "masked FCONN descriptor identity does not match layer";
        return false;
    }
    return true;
}

verifier::verifier(prover *pr, const layeredCircuit &cir):
    p(pr), C(cir) 
{
    comm=p->cc;
    final_claim_u0.resize(C.size + 2);
    final_claim_v0.resize(C.size + 2);
    prover_time=verifier_time=0;
    matrix_time=0;
    r_u.resize(C.size + 2);
    r_v.resize(C.size + 2);
    // make the prover ready
    p->init();
}

F verifier::getFinalValue(const F &claim_u0, const F &claim_u1, const F &claim_v0, const F &claim_v1) {
    auto test_value = bin_value[0] * (claim_u0 * claim_v0)
                      + bin_value[1] * (claim_u1 * claim_v1)
                      + bin_value[2] * (claim_u1 * claim_v0)
                      + uni_value[0] * claim_u0
                      + uni_value[1] * claim_u1;

    return test_value;
}

void verifier::betaInitPhase1(u8 depth, const F &alpha, const F &beta, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1, const F &relu_rou) {
    i8 bl = C.circuit[depth].bit_length;
    i8 fft_bl = C.circuit[depth].fft_bit_length;
    i8 fft_blh = C.circuit[depth].fft_bit_length - 1;
    i8 cnt_bl = bl - fft_bl, cnt_bl2 = C.circuit[depth].max_bl_u - fft_bl;


        
            beta_g.resize(1ULL << bl);
            initBetaTable(beta_g, C.circuit[depth].bit_length, r_0, r_1, alpha, beta );
            if (C.circuit[depth].zero_start_id < C.circuit[depth].size)
                for (u32 g = C.circuit[depth].zero_start_id; g < 1ULL << C.circuit[depth].bit_length; ++g)
                    beta_g[g] = beta_g[g] * relu_rou;
            beta_u.resize(1ULL << C.circuit[depth].max_bl_u);
            initBetaTable(beta_u, C.circuit[depth].max_bl_u, r_u[depth].begin(), F_ONE);
    
}

void verifier::betaInitPhase2(u8 depth) {
    beta_v.resize(1ULL << C.circuit[depth].max_bl_v);
    initBetaTable(beta_v, C.circuit[depth].max_bl_v, r_v[depth].begin(), F_ONE);
}

static ThreadSafeQueue<int> workerq,endq;
void pred1_worker(const zkgpt::NativeVector<uniGate> &beg, vector<F>& uni_value,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        for (auto gate=beg.begin()+l;gate!=beg.begin()+r;gate++) 
        {
            bool idx = gate->lu;
            uni_value[idx]+= beta_g[gate->g] * beta_u[gate->u] * gate->sc;
        }
        endq.Push(idx);
    }
}
void pred2_worker(const zkgpt::NativeVector<binGate> &beg, vector<F>& bin_value,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        for (auto gate=beg.begin()+l;gate!=beg.begin()+r;gate++) 
        {
            bin_value[gate->l] += beta_g[gate->g] * beta_u[gate->u] * beta_v[gate->v] * gate->sc;
        }
        endq.Push(idx);
    }
}

void verifier::predicatePhase1(u8 layer_id) 
{    
    auto &cur_layer = C.circuit[layer_id];
    uni_value[0].clear();
    uni_value[1].clear();
    vector<F>  univ[40];
    for(int i=0;i<32;i++)
    {
            univ[i].emplace_back(F(0));
            univ[i].emplace_back(F(0));
    }
    if(C.circuit[layer_id].uni_interval.size()>=2)
    {
        const int thd=32;
        int *L=new int [C.circuit[layer_id].uni_interval.size()],*R=new int [C.circuit[layer_id].uni_interval.size()];
        for (u64 j = 0; j <C.circuit[layer_id].uni_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur_layer.uni_interval[j].first;
                R[j]=cur_layer.uni_interval[j].second;
        }
            for(int i=0;i<thd;i++)
            { 
                thread t(pred1_worker, std::cref(cur_layer.uni_gates),std::ref(univ[i]),std::ref(beta_g),std::ref(beta_u),std::ref(L),std::ref(R)); 
                t.detach();
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=C.circuit[layer_id].uni_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            endq.Clear();
            for(int i=0;i<thd;i++)
            {
                uni_value[0]+=univ[i][0];
                uni_value[1]+=univ[i][1];
            }
    }
    else for (auto &gate: cur_layer.uni_gates) 
    {
        bool idx = gate.lu;
        uni_value[idx]+= beta_g[gate.g] * beta_u[gate.u] * gate.sc;
    }
    bin_value[0] = bin_value[1] = bin_value[2] = F_ZERO;
}

void verifier::predicatePhase2(u8 layer_id) 
{
    uni_value[0] = uni_value[0] * beta_v[0];
    uni_value[1] = uni_value[1] * beta_v[0];

    auto &cur_layer = C.circuit[layer_id];
    vector<F>  binv[40];
    for(int i=0;i<32;i++)
    {
            binv[i].emplace_back(F(0));
            binv[i].emplace_back(F(0));
            binv[i].emplace_back(F(0));
    }
    if(C.circuit[layer_id].bin_interval.size()>=2)
    {
        const int thd=32;
        int *L=new int [C.circuit[layer_id].bin_interval.size()],*R=new int [C.circuit[layer_id].bin_interval.size()];
        for (u64 j = 0; j <C.circuit[layer_id].bin_interval.size(); ++j) 
        {
                workerq.Push(j);
                L[j]=cur_layer.bin_interval[j].first;
                R[j]=cur_layer.bin_interval[j].second;
        }
            for(int i=0;i<thd;i++)
            { 
                thread t(pred2_worker, std::cref(cur_layer.bin_gates),std::ref(binv[i]),std::ref(beta_g),std::ref(beta_u),std::ref(L),std::ref(R)); 
                t.detach();
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=C.circuit[layer_id].bin_interval.size())
                this_thread::sleep_for (std::chrono::microseconds(1));
            endq.Clear();
            for(int i=0;i<thd;i++)
            {
                bin_value[0]+=binv[i][0];
                bin_value[1]+=binv[i][1];
                bin_value[2]+=binv[i][2];
            }
    }
    else for (auto &gate: cur_layer.bin_gates)
        bin_value[gate.l] += beta_g[gate.g] * beta_u[gate.u] * beta_v[gate.v] * gate.sc;
}

F verifier::challenge(const std::string &label)
{
    if (session_ != nullptr)
        return session_->challenge(label);
    F value;
    value.setByCSPRNG();
    return value;
}

void verifier::absorb_field(const std::string &label, const F &value)
{
    if (session_ != nullptr)
        session_->absorb_field(label, value);
}

void verifier::absorb_fields(const std::string &label, const std::vector<F> &values)
{
    if (session_ != nullptr)
        session_->absorb_fields(label, values);
}

bool verifier::prove(int commit_thread)
{
    (void)commit_thread;
    cout << "Circuit Initiation finished" << endl;
    cout << "Proving Service started" << endl;
    const bool gkr_ok = verifyGKR();
    const bool lasso_ok = gkr_ok && verifyLasso();
    const bool opening_ok = lasso_ok && openCommit();
    cout << "Our results:" << endl;
    cout << "Matrix multiplication Prover time: " << matrix_time << "s" << endl;
    cout << "Total Prover time: " << prover_time + range_prover_time << "s" << endl;
    cout << "Verifier time: " << verifier_time << "s" << endl;
    cout << "Proof size: " << (double)(p->proof_size) / 1024.0 << "KB" << endl;
    return gkr_ok && lasso_ok && opening_ok;
}

bool verifier::prove_with_session(NativeProofSession &session, int commit_thread)
{
    if (!session.valid())
        return false;
    const std::size_t checkpoint = session.checkpoint();
    NativeProofSession *previous = session_;
    session_ = &session;
    bool result = false;
    try
    {
        result = prove(commit_thread);
    }
    catch (...)
    {
        session_ = previous;
        session.rollback(checkpoint);
        throw;
    }
    session_ = previous;
    if (!result || !session.replay())
    {
        session.rollback(checkpoint);
        return false;
    }
    return true;
}

bool verifier::prove_gkr_with_session(NativeProofSession &session, int commit_thread)
{
    (void)commit_thread;
    if (!session.valid())
        return false;
    const std::size_t checkpoint = session.checkpoint();
    NativeProofSession *previous = session_;
    session_ = &session;
    bool result = false;
    try
    {
        result = verifyGKR();
    }
    catch (...)
    {
        session_ = previous;
        session.rollback(checkpoint);
        throw;
    }
    session_ = previous;
    if (!result || !session.replay())
    {
        session.rollback(checkpoint);
        return false;
    }
    return true;
}

Fr *init_book_keeping(int m, int n, vector<Fr> &vec, int offset, vector<Fr> &ra)
{
    assert(ra.size()==m+n);
    Fr* pa[30];
    for(int j=0;j<=n;j++)
    {
        pa[j]=new Fr[1<<(m+n-j)];
    }
    int t=min((1<<(m+n)),(int)vec.size()-offset);
    memset(pa[0]+t,0,sizeof(Fr)*((1<<(m+n))-t));
    
    for(int j=0;j<t;j++)
        pa[0][j]=vec[j+offset];
    for(int j=1;j<=n;j++)
    {
        for(int kp=0;kp<(1<<(m+n-j));kp++)
        {
            pa[j][kp]=pa[j-1][kp+(1<<(m+n-j))]*ra[m+n-j]+pa[j-1][kp]*(1-ra[m+n-j]);
        }
    }
    for(int j=0;j<n;j++)
    {
        delete [] pa[j];
    }
    return pa[n];
}
Fr fm[1<<8][65536];
Fr * init_book_keeping_fast(int m,int n,int* vec,vector<Fr> & ra)
{
    assert(ra.size()==m+n);
    timer tt;
    
    Fr* ret=new Fr[1<<m];
    Fr *eqq=new Fr[1<<n],*eq1=new Fr[1<<(n/2)],*eq2=new Fr[1<<(n/2)];
    
    for(int j=0;j<(1<<n);j++)
    {
        eqq[j]=1;
        for(int i=0;i<n;i++)
        {
            if(j&(1<<(n-1-i)))
                eqq[j]*=ra[m+n-1-i];
            else
                eqq[j]*=1-ra[m+n-1-i];
        }
    }
    for(int j=0;j<(1<<(n/2));j++)
    {
        eq1[j]=1;
        for(int i=0;i<n/2;i++)
        {
            if(j&(1<<(n/2-1-i)))
                eq1[j]*=ra[m+n-1-i];
            else
                eq1[j]*=1-ra[m+n-1-i];
        }
    }
    for(int j=0;j<(1<<(n/2));j++)
    {
        eq2[j]=1;
        for(int i=0;i<n/2;i++)
        {
            if(j&(1<<(n/2-1-i)))
                eq2[j]*=ra[m+n/2-1-i];
            else
                eq2[j]*=1-ra[m+n/2-1-i];
        }
    }
    std::vector<std::thread> threads1,threads2;
    int num_threads=32;
    int chunk_size = (1<<(n/2))/num_threads;
    auto worker1 = [&](int start_id, int end_id) 
    {
        for(int R=start_id;R<end_id;R++)
        {
            fm[R][0]=0;
            fm[R][1]=eq2[R];
            for(int i=2;i<(1<<16);i++)
                fm[R][i]=fm[R][i-1]+eq2[R];
        }
    };
    for (int t = 0; t < num_threads; t++) 
    {
        int start_col = t * chunk_size;
        int end_col = min(start_col + chunk_size, (1<<(n/2)));
        threads1.emplace_back(worker1, start_col, end_col);
    }
    for (auto &th : threads1) 
    {
        if (th.joinable())
            th.join();
    }
    auto worker2 = [&](int start_id, int end_id) 
    {
        for(int kp=start_id;kp<end_id;kp++)
        {
            ret[kp]=0;
            Fr sum=0;
            for(int L=0;L<(1<<(n/2));L++)
            {
                Fr sumL=0;
                for(int R=0;R<(1<<(n/2));R++)
                {
                    int j=((L<<(n/2))+R)<<m;
                    if(vec[kp+j]!=0)
                        sumL+=fm[R][vec[kp+j]];
                }
                sum+=eq1[L]*sumL;
            }
            ret[kp]=sum;
        }
    };
    chunk_size = (1<<m)/num_threads;
    for (int t = 0; t < num_threads; t++) 
    {
        int start_col = t * chunk_size;
        int end_col = min(start_col + chunk_size, (1<<m));
        threads2.emplace_back(worker2, start_col, end_col);
    }
    for (auto &th : threads2) 
    {
        if (th.joinable())
            th.join();
    }
    return ret;
}

void sc_last_worker(Fr& A, Fr&B,Fr&C,Fr*& read_1,Fr*& read_2,Fr*& write_1,Fr*& write_2, int*& L,int*& R,Fr ran)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        for(int kp=l;kp<r;kp++)
        {
            Fr & c=read_1[kp<<1|1], &d=read_1[kp<<1];
                if(c==d)
                    write_1[kp]=d;
                else
                    write_1[kp]=(c-d)*ran+d;
                Fr & a=read_2[kp<<1|1], &b=read_2[kp<<1];
                if(a==b)
                    write_2[kp]=b;
                else
                    write_2[kp]=(a-b)*ran+b;
                if(kp&1)
                {
                    bool az=write_2[kp-1].isZero(),bz=write_2[kp].isZero();
                    if(az && bz)
                        continue;
                    //if(!az)
                        A+=write_1[kp-1]*write_2[kp-1];
                    //if(!bz)
                        B+=write_1[kp]*write_2[kp];
                    C+=(write_1[kp]+write_1[kp]-write_1[kp-1])*(write_2[kp]+write_2[kp]-write_2[kp-1]);
                }
        }
        endq.Push(idx);
    }
}

void sc_last_worker_first(Fr& A, Fr&B,Fr&C,Fr*& read_1,Fr*& read_2, int*& L,int*& R)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        int l=L[idx],r=R[idx];
        for(int k=l;k<r;k++)
        {
            if(k&1)
            {
                bool a_z=read_2[k].isZero(),b_z=read_2[k-1].isZero();
                if(a_z && b_z)
                    continue;
                B+=read_1[k]*read_2[k];
                C+=(read_1[k]+read_1[k]-read_1[k-1])*(read_2[k]+read_2[k]-read_2[k-1]); 
            }
            else if(!read_2[k].isZero())
                A+=read_1[k]*read_2[k];  
        }
        endq.Push(idx);
    }
}



pair<Fr, Fr> sum_check_product(Fr *f, Fr *g, int m, Fr *r, Fr ans)
{
    Fr A=0,B=0,C=0;
    Fr *zf=new Fr[1<<m], *zg=new Fr[1<<m];
    for(int k=0;k<(1<<m);k++)
    {
        zf[k]=2*f[k*2+1]-f[k*2];
        zg[k]=2*g[k*2+1]-g[k*2];
        C+=zf[k]*zg[k];
    }
    for(int k=0;k<(1<<(m+1));k++)
    {
        if(k&1)
            B+=f[k]*g[k];
        else
        {
            A+=f[k]*g[k];
        }
    }
    Fr a=(C+A)/2-B,b=2*B-C/2-3*A/2,c=A;
    Fr g1=a+b+c,g0=c;
    if(!(ans-(g1+g0)).isZero())
    {
        cout<<"ans!=g(0)+g(1) "<<m<<endl;
        exit(0);
    }
    if(!(a+b+c-B).isZero())
    {
        cout<<"sum check fail! order 1 "<<m<<endl;
        exit(0);
    }
    if(!(4*a+2*b+c-C).isZero())
    {
        cout<<"sum check fail! order 2 "<<m<<endl;
        exit(0);
    }
    Fr send_r=r[0];
    Fr * new_f=new Fr[1<<m], *new_g=new Fr[1<<m];
    for(int k=0;k<(1<<m);k++)
    {
        new_f[k]=(1-send_r)*f[k*2]+(send_r)*f[k*2+1];
        new_g[k]=(1-send_r)*g[k*2]+(send_r)*g[k*2+1];
    }
    Fr new_ans=send_r*send_r*a+send_r*b+c; //g(r)
    if(m==0)
    {
        assert((new_ans-new_f[0]*new_g[0]).isZero());
        return make_pair(new_f[0],new_g[0]);
    }
    else
        return sum_check_product(new_f,new_g,m-1,r+1,new_ans);
}

pair<Fr,Fr> sum_check_product_session(Fr* f, Fr* g, int m,
                                       NativeProofSession &session,
                                       const std::string &label, Fr ans,
                                       int round = 0) {
    Fr A = 0, B = 0, C = 0;
    std::vector<Fr> zf(1 << m), zg(1 << m);
    for (int k = 0; k < (1 << m); ++k) {
        zf[k] = 2 * f[k * 2 + 1] - f[k * 2];
        zg[k] = 2 * g[k * 2 + 1] - g[k * 2];
        C += zf[k] * zg[k];
    }
    for (int k = 0; k < (1 << (m + 1)); ++k) {
        if (k & 1) B += f[k] * g[k];
        else A += f[k] * g[k];
    }
    Fr a = (C + A) / 2 - B;
    Fr b = 2 * B - C / 2 - 3 * A / 2;
    Fr c = A;
    if (ans != (a + b + c) + c)
        throw std::runtime_error("native matrix Sumcheck initial claim mismatch");
    session.absorb_fields(label + "/poly/" + std::to_string(round),
                          std::vector<F>{a, b, c});
    const Fr send_r = session.challenge(label + "/r/" + std::to_string(round));
    std::vector<Fr> new_f(1 << m), new_g(1 << m);
    for (int k = 0; k < (1 << m); ++k) {
        new_f[k] = (1 - send_r) * f[k * 2] + send_r * f[k * 2 + 1];
        new_g[k] = (1 - send_r) * g[k * 2] + send_r * g[k * 2 + 1];
    }
    const Fr new_ans = send_r * send_r * a + send_r * b + c;
    if (m == 0) {
        const bool terminal_ok = (new_ans - new_f[0] * new_g[0]).isZero();
        const pair<Fr, Fr> result(new_f[0], new_g[0]);
        if (!terminal_ok)
            throw std::runtime_error("native matrix Sumcheck terminal mismatch");
        return result;
    }
    const pair<Fr, Fr> result = sum_check_product_session(
        new_f.data(), new_g.data(), m - 1, session, label, new_ans, round + 1);
    return result;
}


bool verifier::verifyGKR() 
{
    
    F alpha = F_ONE, beta = F_ZERO, relu_rou;
    F final_claim_u1 = F_ZERO, final_claim_v1 = F_ZERO;
    r_u[C.size].resize(C.circuit[C.size - 1].bit_length);
    for (i8 i = 0; i < C.circuit[C.size - 1].bit_length; ++i)
        r_u[C.size][i] = challenge("gkr/output/r_u/" + std::to_string(static_cast<int>(i)));
    vector<F>::const_iterator r_0 = r_u[C.size].begin();
    vector<F>::const_iterator r_1;

    auto previousSum = p->Vres(r_0, C.circuit[C.size - 1].size, C.circuit[C.size - 1].bit_length,C.size-1);
    timer ptimer,vtimer;
    timer mat_timer;
    ptimer.start();
    p -> sumcheckInitAll(r_0);
    ptimer.stop();
    prover_time+=ptimer.elapse_sec();
    int fc_cnt=0,now_fc_id;
    Fr prev_u1=0,prev_v1=0;
    for (int i = C.size - 1; i; --i)
    {
        if(C.circuit[i].ty==layerType::FCONN)
            fc_cnt++;
    } 
    
    timer init,sc;
    double init_t=0,sc_t=0;
    double matrix_tt_time=0;
    for (int i = C.size - 1; i; --i) 
    {
        timer tmp;
        tmp.start();
        auto &cur = C.circuit[i];
        ptimer.start();
        p->sumcheckInit(alpha, beta);
        ptimer.stop();
        prover_time+=ptimer.elapse_sec();
        
        // phase 1
        if (cur.zero_start_id < cur.size)
        {
            relu_rou = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/relu");
        }
        else 
            relu_rou = F_ONE;
        F previousRandom = F_ZERO;
        if(C.circuit[i].ty==layerType::FCONN)   //substitute matrix with thaler13
        {
            now_fc_id=--fc_cnt;
            int n,m,k;
            n=ceilPow2BitLength(p->fc_input_row[now_fc_id]);
            k=ceilPow2BitLength(p->fc_col[now_fc_id]);
            m=ceilPow2BitLength(p->fc_row[now_fc_id]);
            r_u[i].resize(m+n);
            r_v[i].resize(m+k);
            for (int j = m; j < m+n; ++j)
                r_u[i][j]=r_u[i+1][j-m+k];
            for (int j = m; j < m+k; ++j)
                r_v[i][j]=r_u[i+1][j-m];
            prev_u1=final_claim_u1;
            previousRandom = F_ZERO;
            absorb_field("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/matmul/claim", prev_u1);
            for(int j=0;j<m;j++)
            {
                r_u[i][j] = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/matmul/r_u/" + std::to_string(j));
                r_v[i][j]=r_u[i][j];
            }
            Fr sum=0;
            ptimer.start();
            mat_timer.start();
            Fr* pa=init_book_keeping(m,n,p->val[0],p->fc_input_id[now_fc_id],r_u[i]);
            Fr* pb=init_book_keeping_fast(m,k,p->mat_val[now_fc_id],r_v[i]);
            pair<Fr,Fr> oracle;
            if (session_ != nullptr)
                oracle = sum_check_product_session(pa, pb, m - 1, *session_,
                                                   "gkr/layer/" + std::to_string(static_cast<int>(i)) + "/matmul",
                                                   prev_u1);
            else
                oracle = sum_check_product(pa,pb,m-1,r_u[i].data(),prev_u1);
            mat_timer.stop();
            ptimer.stop();
            matrix_time+=mat_timer.elapse_sec();
            prover_time+=ptimer.elapse_sec();    
            final_claim_u0[i]=oracle.first;
            final_claim_u1=0;
            final_claim_v0[i]=oracle.second;
            final_claim_v1=0;
            absorb_fields("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/matmul/claims",
                          std::vector<F>{final_claim_u0[i], final_claim_u1,
                                         final_claim_v0[i], final_claim_v1});
        }
        else
        {
            timer normal_timer;
            normal_timer.start();
            ptimer.start();
            p->sumcheckInitPhase1(relu_rou);
            ptimer.stop();
            prover_time+=ptimer.elapse_sec();            
            r_u[i].resize(cur.max_bl_u);
            for (int j = 0; j < cur.max_bl_u; ++j) {
                if (session_ == nullptr)
                    r_u[i][j].setByCSPRNG();
            }
            sc.start();
            for (i8 j = 0; j < cur.max_bl_u; ++j) 
            {
                F cur_claim, nxt_claim;
                ptimer.start();
                quadratic_poly poly = p->sumcheckUpdate1(previousRandom);
                ptimer.stop();
                prover_time+=ptimer.elapse_sec();
                absorb_fields("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase1/poly/" + std::to_string(j),
                              std::vector<F>{poly.a, poly.b, poly.c});

                vtimer.start();
                cur_claim = poly.eval(F_ZERO) + poly.eval(F_ONE);
                if (session_ != nullptr)
                    r_u[i][j] = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase1/r/" + std::to_string(j));
                nxt_claim = poly.eval(r_u[i][j]);

                if (cur_claim != previousSum)
                {
                    cerr << cur_claim << ' ' << previousSum << endl;
                    fprintf(stderr, "Verification fail, phase1, circuit %d, current bit %d\n", i, j);
                    return false;
                }
                vtimer.stop();
                verifier_time+=vtimer.elapse_sec();
                previousRandom = r_u[i][j];
                previousSum = nxt_claim;
            }
            sc.stop();
            sc_t+=sc.elapse_sec();
            prev_u1=final_claim_u1;
            ptimer.start();
            p->sumcheckFinalize1(previousRandom, final_claim_u0[i], final_claim_u1);
            absorb_fields("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase1/claims",
                          std::vector<F>{final_claim_u0[i], final_claim_u1});
            betaInitPhase1(i, alpha, beta, r_0, r_1, relu_rou);
            predicatePhase1(i);
            ptimer.stop();
            prover_time+=ptimer.elapse_sec();     
            
            if (cur.need_phase2) 
            {
                timer normal_timer2;
                normal_timer2.start();
                r_v[i].resize(cur.max_bl_v);
                for (int j = 0; j < cur.max_bl_v; ++j) {
                    if (session_ == nullptr)
                        r_v[i][j].setByCSPRNG();
                }
              
                ptimer.start();
                p->sumcheckInitPhase2();
                ptimer.stop();
                prover_time+=ptimer.elapse_sec();     
                previousRandom = F_ZERO;
                sc.start();
                for (u32 j = 0; j < cur.max_bl_v; ++j) 
                {
                        ptimer.start();
                        quadratic_poly poly = p->sumcheckUpdate2(previousRandom);
                        ptimer.stop();
                        prover_time+=ptimer.elapse_sec();
                        absorb_fields("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase2/poly/" + std::to_string(j),
                                      std::vector<F>{poly.a, poly.b, poly.c});
                        vtimer.start();
                        if (session_ != nullptr)
                            r_v[i][j] = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase2/r/" + std::to_string(j));
                        if (poly.eval(F_ZERO) + poly.eval(F_ONE) != previousSum)
                        {
                            fprintf(stderr, "Verification fail, phase2, circuit level %d, current bit %d, total is %d\n", i, j,
                                    cur.max_bl_v);
                            return false;
                        }
                        vtimer.stop();
                        verifier_time+=vtimer.elapse_sec();
                        previousRandom = r_v[i][j];
                        previousSum = poly.eval(previousRandom);
                }
                sc.stop();
                sc_t+=sc.elapse_sec();
                prev_v1=final_claim_v1;
                ptimer.start();
                p->sumcheckFinalize2(previousRandom, final_claim_v0[i], final_claim_v1);
                absorb_fields("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/phase2/claims",
                              std::vector<F>{final_claim_v0[i], final_claim_v1});
                betaInitPhase2(i);
                predicatePhase2(i);
                ptimer.stop();
                prover_time+=ptimer.elapse_sec();     
            }
            
        }
        
        
        vtimer.start();
        if (C.circuit[i].ty!=layerType::FCONN )  // for thaler13 no need to check this
        {
            F test_value = getFinalValue(final_claim_u0[i], final_claim_u1, final_claim_v0[i], final_claim_v1);
            if(previousSum != test_value)
            {
                std::cerr << test_value << ' ' << previousSum << std::endl;
                return false;
            }
        }
        
        

        if (~cur.bit_length_u[1])
            alpha = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/alpha");
        else
            alpha.clear();
        if ((~cur.bit_length_v[1]) || cur.ty == layerType::FFT)
            beta = challenge("gkr/layer/" + std::to_string(static_cast<int>(i)) + "/beta");
        else 
            beta.clear();
        previousSum = alpha * final_claim_u1 + beta * final_claim_v1;
        vtimer.stop();
        verifier_time+=vtimer.elapse_sec();

        r_0 = r_u[i].begin();
        r_1 = r_v[i].begin();
        beta_u.clear();
        beta_v.clear();
        tmp.stop();
    }

    
    return true;
}

bool verify_gkr_transcript(const NativeProofSession &session,
                           const layeredCircuit &circuit,
                           const zkgpt::NativeFieldVector &public_output,
                           const std::vector<NativeGkrMatmulShape> &matmuls,
                           std::string *reason) {
    try {
        if (!session.valid(reason) || !session.replay(reason)) return false;
        if (circuit.size != circuit.circuit.size() || circuit.size < 2) {
            if (reason != nullptr) *reason = "GKR circuit shape is invalid";
            return false;
        }
        const layer &last = circuit.circuit[circuit.size - 1];
        if (last.bit_length < 0 || public_output.size() != last.size) {
            if (reason != nullptr) *reason = "public GKR output shape is invalid";
            return false;
        }

        std::set<std::string> known;
        auto mark_field_challenge = [&](const std::string &label) {
            known.insert("challenge/" + label);
        };
        auto mark_quadratic_scope = [&](const std::string &label,
                                        std::size_t rounds) {
            for (std::size_t round = 0; round < rounds; ++round) {
                known.insert(label + "/poly/" + std::to_string(round));
                mark_field_challenge(label + "/r/" + std::to_string(round));
            }
            known.insert(label + "/claims");
        };
        auto mark_product_scope = [&](const std::string &label,
                                      std::size_t rounds) {
            known.insert(label + "/claim");
            mark_quadratic_scope(label, rounds);
        };

        std::vector<std::vector<F>> r_u(circuit.size + 1);
        std::vector<std::vector<F>> r_v(circuit.size + 1);
        r_u[circuit.size].resize(last.bit_length);
        for (int bit = 0; bit < last.bit_length; ++bit) {
            const std::string label = "gkr/output/r_u/" + std::to_string(bit);
            known.insert("challenge/" + label);
            if (!transcript_challenge(session, label, r_u[circuit.size][bit], reason))
                return false;
        }
        const F initial_claim = transcript_mle_eval(public_output, r_u[circuit.size]);
        F previous_sum = initial_claim;
        F alpha = F_ONE;
        F beta = F_ZERO;
        F final_claim_u1 = F_ZERO;
        F final_claim_v1 = F_ZERO;
        std::vector<F> final_claim_u0(circuit.size, F_ZERO);
        std::vector<F> final_claim_v0(circuit.size, F_ZERO);
        std::size_t next_matmul = matmuls.size();

        for (int layer_id = static_cast<int>(circuit.size) - 1;
             layer_id > 0; --layer_id) {
            const layer &current = circuit.circuit[layer_id];
            const std::string layer_prefix =
                "gkr/layer/" + std::to_string(layer_id);
            if (current.ty == layerType::FCONN) {
                if (next_matmul == 0) {
                    if (reason != nullptr) *reason = "missing public FCONN descriptor";
                    return false;
                }
                const NativeGkrMatmulShape shape = matmuls[--next_matmul];
                if (!masked_fconn_shape_matches(current, shape, reason)) return false;
                const int n = ceilPow2BitLength(static_cast<u64>(shape.rows));
                const int m = ceilPow2BitLength(static_cast<u64>(shape.inner));
                const int k = ceilPow2BitLength(static_cast<u64>(shape.cols));
                if (r_u[layer_id + 1].size() < static_cast<std::size_t>(m + k)) {
                    if (reason != nullptr) *reason = "FCONN predecessor point is too short";
                    return false;
                }
                r_u[layer_id].resize(static_cast<std::size_t>(m + n));
                r_v[layer_id].resize(static_cast<std::size_t>(m + k));
                for (int bit = m; bit < m + n; ++bit)
                    r_u[layer_id][bit] = r_u[layer_id + 1][bit - m + k];
                for (int bit = m; bit < m + k; ++bit)
                    r_v[layer_id][bit] = r_u[layer_id + 1][bit - m];
                for (int bit = 0; bit < m; ++bit) {
                    const std::string label = layer_prefix + "/matmul/r_u/" +
                                              std::to_string(bit);
                    known.insert("challenge/" + label);
                    if (!transcript_challenge(session, label, r_u[layer_id][bit], reason))
                        return false;
                    r_v[layer_id][bit] = r_u[layer_id][bit];
                }
                const std::string matmul = layer_prefix + "/matmul";
                known.insert(matmul + "/claim");
                mark_product_scope(matmul, static_cast<std::size_t>(m));
                const NativeProofMessage *claim_message = find_transcript_message(
                    session, NativeProofMessageKind::Field,
                    matmul + "/claim", reason);
                F matmul_claim;
                if (claim_message == nullptr ||
                    !decode_transcript_field(claim_message->payload, matmul_claim) ||
                    matmul_claim != final_claim_u1) {
                    if (reason != nullptr) *reason = "native FCONN claim mismatch";
                    return false;
                }
                F terminal_left = F_ZERO, terminal_right = F_ZERO;
                if (!verify_product_sumcheck_session(
                        session, matmul, matmul_claim, static_cast<std::size_t>(m),
                        &terminal_left, &terminal_right, reason))
                    return false;
                final_claim_u0[layer_id] = terminal_left;
                final_claim_u1 = F_ZERO;
                final_claim_v0[layer_id] = terminal_right;
                final_claim_v1 = F_ZERO;
                if (current.bit_length_u[1] >= 0) {
                    const std::string label = layer_prefix + "/alpha";
                    known.insert("challenge/" + label);
                    if (!transcript_challenge(session, label, alpha, reason)) return false;
                } else {
                    alpha.clear();
                }
                if (current.bit_length_v[1] >= 0 || current.ty == layerType::FFT) {
                    const std::string label = layer_prefix + "/beta";
                    known.insert("challenge/" + label);
                    if (!transcript_challenge(session, label, beta, reason)) return false;
                } else {
                    beta.clear();
                }
                previous_sum = alpha * final_claim_u1 + beta * final_claim_v1;
                continue;
            }

            F relu_rou = F_ONE;
            if (current.zero_start_id < current.size) {
                const std::string label = layer_prefix + "/relu";
                known.insert("challenge/" + label);
                if (!transcript_challenge(session, label, relu_rou, reason))
                    return false;
            }

            r_u[layer_id].resize(current.max_bl_u);
            for (int bit = 0; bit < current.max_bl_u; ++bit) {
                const std::string label = layer_prefix + "/phase1/r/" +
                                          std::to_string(bit);
                known.insert("challenge/" + label);
                if (!transcript_challenge(session, label, r_u[layer_id][bit], reason))
                    return false;
            }
            const std::string phase1 = layer_prefix + "/phase1";
            mark_quadratic_scope(phase1, current.max_bl_u);
            F phase1_sum = previous_sum;
            std::vector<F> phase1_claims;
            if (current.max_bl_u != 0) {
                if (!verify_quadratic_sumcheck_session(
                        session, phase1, previous_sum, current.max_bl_u,
                        &phase1_sum, &phase1_claims, reason))
                    return false;
            } else if (!transcript_claims(session, phase1, 2, phase1_claims, reason)) {
                return false;
            }
            if (phase1_claims.size() != 2) {
                if (reason != nullptr) *reason = "GKR phase1 claim arity mismatch";
                return false;
            }
            final_claim_u0[layer_id] = phase1_claims[0];
            final_claim_u1 = phase1_claims[1];

            std::vector<F> r0 = r_u[layer_id + 1];
            std::vector<F> r1 = r_v[layer_id + 1];
            if (r0.size() < static_cast<std::size_t>(current.bit_length))
                r0.resize(current.bit_length, F_ZERO);
            if (r1.size() < static_cast<std::size_t>(current.bit_length))
                r1.resize(current.bit_length, F_ZERO);
            zkgpt::NativeFieldVector beta_g, beta_u;
            std::vector<F> uni, bin;
            transcript_beta_phase1(current, layer_id, r0, r1,
                                   r_u[layer_id], alpha, beta, relu_rou,
                                   beta_g, beta_u);
            transcript_predicate_phase1(current, beta_g, beta_u, uni, bin);

            F current_sum = phase1_sum;
            if (current.need_phase2) {
                r_v[layer_id].resize(current.max_bl_v);
                for (int bit = 0; bit < current.max_bl_v; ++bit) {
                    const std::string label = layer_prefix + "/phase2/r/" +
                                              std::to_string(bit);
                    known.insert("challenge/" + label);
                    if (!transcript_challenge(session, label, r_v[layer_id][bit], reason))
                        return false;
                }
                const std::string phase2 = layer_prefix + "/phase2";
                mark_quadratic_scope(phase2, current.max_bl_v);
                std::vector<F> phase2_claims;
                if (current.max_bl_v == 0) {
                    if (!transcript_claims(session, phase2, 2, phase2_claims, reason))
                        return false;
                } else if (!verify_quadratic_sumcheck_session(
                               session, phase2, phase1_sum, current.max_bl_v,
                               &current_sum, &phase2_claims, reason)) {
                    return false;
                }
                if (phase2_claims.size() != 2) {
                    if (reason != nullptr) *reason = "GKR phase2 claim arity mismatch";
                    return false;
                }
                final_claim_v0[layer_id] = phase2_claims[0];
                final_claim_v1 = phase2_claims[1];
                zkgpt::NativeFieldVector beta_v;
                beta_v.resize(std::size_t(1) << current.max_bl_v);
                initBetaTable(beta_v, current.max_bl_v,
                              r_v[layer_id].begin(), F_ONE);
                transcript_predicate_phase2(current, beta_g, beta_u, beta_v,
                                            uni, bin);
            } else {
                final_claim_v0[layer_id] = F_ZERO;
                final_claim_v1 = F_ZERO;
            }

            const F predicate_value = transcript_final_value(
                uni, bin,
                std::vector<F>{final_claim_u0[layer_id], final_claim_u1},
                std::vector<F>{final_claim_v0[layer_id], final_claim_v1});
            if (current_sum != predicate_value) {
                if (reason != nullptr) *reason = "GKR predicate transition failed";
                return false;
            }

            if (current.bit_length_u[1] >= 0) {
                const std::string label = layer_prefix + "/alpha";
                known.insert("challenge/" + label);
                if (!transcript_challenge(session, label, alpha, reason)) return false;
            } else {
                alpha.clear();
            }
            if (current.bit_length_v[1] >= 0 || current.ty == layerType::FFT) {
                const std::string label = layer_prefix + "/beta";
                known.insert("challenge/" + label);
                if (!transcript_challenge(session, label, beta, reason)) return false;
            } else {
                beta.clear();
            }
            previous_sum = alpha * final_claim_u1 + beta * final_claim_v1;
        }

        for (const NativeProofMessage &message : session.messages()) {
            if (gkr_scope_message(message.label) &&
                known.find(message.label) == known.end()) {
                if (reason != nullptr)
                    *reason = "unexpected native GKR transcript message: " + message.label;
                return false;
            }
        }
        return true;
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool verifier::verifyLasso() 
{
    timer ptimer,vtimer;
    auto &cur = C.circuit[0];

    if (session_ != nullptr) {
        std::ostringstream context;
        context << "lasso-context-v1|" << C.size;
        for (int layer = 0; layer < C.size; ++layer) {
            const auto &shape = C.circuit[layer];
            context << '|' << static_cast<int>(shape.bit_length)
                    << ':' << static_cast<int>(shape.max_bl_u)
                    << ':' << static_cast<int>(shape.max_bl_v)
                    << ':' << shape.size_u[0] << ':' << shape.size_v[0];
        }
        session_->absorb("lasso/LC/context", context.str());
    }

    vector<F> sig_u(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i) {
        sig_u[i] = challenge("lasso/sig_u/" + std::to_string(i));
    }
    vector<F> sig_v(C.size - 1);
    for (int i = 0; i < C.size - 1; ++i) {
        sig_v[i] = challenge("lasso/sig_v/" + std::to_string(i));
    }
    r_u[0].resize(cur.bit_length);
    for (int i = 0; i < cur.bit_length; ++i) {
        if (session_ == nullptr) r_u[0][i].setByCSPRNG();
    }
    auto r_0 = r_u[0].begin();
    
    F previousSum = F_ZERO;
    for (int i = 1; i < C.size; ++i) 
    {
        if (~C.circuit[i].bit_length_u[0])
            previousSum = previousSum + sig_u[i - 1] * final_claim_u0[i];
        if (~C.circuit[i].bit_length_v[0])
            previousSum = previousSum + sig_v[i - 1] * final_claim_v0[i];
    }
    ptimer.start();
    p->sumcheckLassoInit(sig_u, sig_v,r_u,r_v);
    ptimer.stop();
    prover_time+=ptimer.elapse_sec();
    F previousRandom = F_ZERO;
    int n=C.circuit[0].bit_length;
    Fr*pb1=new Fr[1<<(n+1)],* pa1[30],*pc1=pb1;
    Fr*pb2=new Fr[1<<(n+1)],* pa2[30],*pc2=pb2;
    for(int j=0;j<=n;j++)
    {
        pa1[j]=pc1;
        pc1+=1<<(n-j);
        pa2[j]=pc2;
        pc2+=1<<(n-j);
    }

    const __int128 FILTER=65535,NEG_FILTER=-65535;
    int *L=new int[1<<10],*R=new int[1<<10];
    ptimer.start();
    for(int i=0;i<=n;i++)
    {
        Fr A=0,B=0,C=0;
        if(i==0)
        {
            Fr* read_1=p->lasso_mult_v.data(),*read_2=p->val[0].data();
            const int k=10;
            const int thread_n=32; 
            int total_work=1<<n;
            int task_cnt=0;
            for (u64 j = 0; j < (1<<k); ++j) 
            {
                L[j]=(total_work>>k)*j;
                if(L[j]>=p->total_size[1])
                    break;
                R[j]=(total_work>>k)*(1+j);
                task_cnt++;
                workerq.Push(j);
            }
            Fr Aa[40],Bb[40],Cc[40];
            memset(Aa,0,sizeof(Aa));
            memset(Bb,0,sizeof(Bb));
            memset(Cc,0,sizeof(Cc));
            for(int j=0;j<thread_n;j++)
            {
                thread t(sc_last_worker_first,std::ref(Aa[j]),std::ref(Bb[j]),std::ref(Cc[j]),std::ref(read_1),std::ref(read_2),std::ref(L),std::ref(R)); 
                t.detach();
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(1));
            while(endq.Size()!=task_cnt)
                this_thread::sleep_for (std::chrono::microseconds(1));
            endq.Clear();
            //Fr ax=0,bx=0,cx=0;
            for(int j=0;j<thread_n;j++)
            {
                A+=Aa[j];
                B+=Bb[j];
                C+=Cc[j];
            }
        }
        else if(i<=9)
        {
            timer tt;
            tt.start();
            Fr* read_1=p->lasso_mult_v.data(),*read_2=p->val[0].data(),*write_1=pa1[i],*write_2=pa2[i];
            if(i!=1)    
                read_1=pa1[i-1],read_2=pa2[i-1];
            const int k=10;
            const int thread_n=32; 
            int total_work=1<<(n-i);
            int taskcnt=0;
            for (u64 j = 0; j < (1<<k); ++j) 
            {
                L[j]=(total_work>>k)*j;
                R[j]=(total_work>>k)*(1+j);
                if((1<<i)*L[j]>=p->total_size[1])
                    break;
                ++taskcnt;
                workerq.Push(j);
            }
            Fr Aa[40],Bb[40],Cc[40];
            memset(Aa,0,sizeof(Aa));
            memset(Bb,0,sizeof(Bb));
            memset(Cc,0,sizeof(Cc));
            for(int j=0;j<thread_n;j++)
            {
                thread t(sc_last_worker,std::ref(Aa[j]),std::ref(Bb[j]),std::ref(Cc[j]),std::ref(read_1),std::ref(read_2),std::ref(write_1),std::ref(write_2),std::ref(L),std::ref(R),r_u[0][i-1]); 
                t.detach();
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(10));
            while(endq.Size()!=taskcnt)
                this_thread::sleep_for (std::chrono::microseconds(10));
            endq.Clear();
            tt.stop();
            for(int j=0;j<thread_n;j++)
            {
                A+=Aa[j];
                B+=Bb[j];
                C+=Cc[j];
            }
        }
        else
        {
            Fr* read_1=pa1[i-1],*read_2=pa2[i-1],*write_1=pa1[i],*write_2=pa2[i];
            for(int kp=0;kp<(1<<(n-i));kp++)
            {
                Fr & c=read_1[kp<<1|1], &d=read_1[kp<<1];
                if(c==d)
                    write_1[kp]=d;
                else
                    write_1[kp]=(c-d)*r_u[0][i-1]+d;
                Fr & a=read_2[kp<<1|1], &b=read_2[kp<<1];
                if(a==b)
                    write_2[kp]=b;
                else
                    write_2[kp]=(a-b)*r_u[0][i-1]+b;
                if(kp&1)
                {
                    B+=write_1[kp]*write_2[kp];
                    C+=(write_1[kp]+write_1[kp]-write_1[kp-1])*(write_2[kp]+write_2[kp]-write_2[kp-1]);
                }
                else
                    A+=write_1[kp]*write_2[kp];
            }
        }
        vtimer.start();
        Fr aa=(C+A)/2-B,bb=2*B-C/2-3*A/2,cc=A;
        Fr g1=aa+bb+cc,g0=cc;
        absorb_fields("lasso/poly/" + std::to_string(i),
                      std::vector<F>{aa, bb, cc});
        if(i!=n) {
            if (session_ != nullptr)
                r_u[0][i] = challenge("lasso/r/" + std::to_string(i));
            if (previousSum != g0 + g1) return false;
            previousSum=aa*r_u[0][i]*r_u[0][i]+bb*r_u[0][i]+cc;
        }
        vtimer.stop();
        verifier_time+=vtimer.elapse_sec();
    }
    
    ptimer.stop();
    prover_time+=ptimer.elapse_sec();
    F gr = F_ZERO;

    eval_in=pa2[n][0];
    absorb_field("lasso/eval_in", eval_in);
    
    beta_g.resize(1ULL << cur.bit_length);
    ptimer.start();
    
    initBetaTable(beta_g, cur.bit_length, r_0, F_ONE);
    for (int i = 1; i < C.size; ++i) 
    {
        if (~C.circuit[i].bit_length_u[0]) 
        {
            beta_u.resize(1ULL << C.circuit[i].bit_length_u[0]);
            initBetaTable(beta_u, C.circuit[i].bit_length_u[0], r_u[i].begin(), sig_u[i - 1]);
            for (u32 j = 0; j < C.circuit[i].size_u[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_u[j]] * beta_u[j];
        }

        if (~C.circuit[i].bit_length_v[0]) 
        {
            beta_v.resize(1ULL << C.circuit[i].bit_length_v[0]);
            initBetaTable(beta_v, C.circuit[i].bit_length_v[0], r_v[i].begin(), sig_v[i - 1]);
            for (u32 j = 0; j < C.circuit[i].size_v[0]; ++j)
                gr = gr + beta_g[C.circuit[i].ori_id_v[j]] * beta_v[j];
        }
    }
    ptimer.stop();
    
    prover_time+=ptimer.elapse_sec();
    
    beta_u.clear();
    beta_v.clear();

    vtimer.start();
    if (eval_in * gr != previousSum) return false;
    vtimer.stop();
    verifier_time+=vtimer.elapse_sec();

    fprintf(stderr, "All verification passed!!\n");

    beta_g.clear();
    beta_gs.clear();
    beta_u.clear();
    beta_v.clear();
    r_u.resize(1);
    r_v.clear();

    sig_u.clear();
    sig_v.clear();
    return true;
}
__int128 convertx(Fr x)	
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
    __int128 V=0;	
    for(int j=size-1;j>=0;j--)
    {	
        V=(V<<8)+bf[j];	
    }
    if(sign)	
        V=-V;	
    return V;	
}
bool verifier::openCommit()
{
    std::unique_ptr<Fr[]> Lp(new Fr[1 << (comm.l / 2)]);
    std::unique_ptr<Fr[]> Rp(new Fr[1 << (comm.l - comm.l / 2)]);
    timer verify_input;
    verify_input.start();
    brute_force_compute_LR(Lp.get(), Rp.get(), r_u[0].data(), comm.l);
    int rownum=(1<<comm.l/2),colnum=(1<<(comm.l-comm.l/2));
    
    pair<double,double> tim;
    if (session_ != nullptr) {
        std::vector<F> point(r_u[0].begin(), r_u[0].end());
        absorb_fields("hyrax/opening/r", point);
        absorb_field("hyrax/opening/eval", eval_in);
        tim = hyrax::open_with_session(comm.w, r_u[0].data(), eval_in,
                                       comm.G, comm.g, Lp.get(), Rp.get(), comm.comm,
                                       comm.l, *session_, "hyrax/opening");
    } else {
        tim = hyrax::open(comm.w, r_u[0].data(), eval_in, comm.G, comm.g,
                          Lp.get(), Rp.get(), comm.comm, comm.l);
    }
    prover_time+=tim.first;
    verifier_time+=tim.second;
    return true;
}

namespace {

void masked_set_reason(std::string *reason, const std::string &message) {
    if (reason != nullptr) *reason = message;
}

using zkgpt::native_pedersen::PedersenBasis;
using zkgpt::native_pedersen::ScalarCommitment;
using zkgpt::native_pedersen::ScalarOpening;

PedersenBasis masked_basis(const NativeProofSession &session) {
    return zkgpt::native_pedersen::derive_basis(
        "zkgpt/masked-sumcheck/v2/" + session.statement_digest());
}

bool masked_emit_opening(NativeProofSession &session, const PedersenBasis &basis,
                         const std::string &label, const ScalarOpening &opening,
                         std::string *reason) {
    return zkgpt::native_pedersen::prove_opening(session, basis, label, opening,
                                                  nullptr, reason);
}

bool masked_make_linear_claim(NativeProofSession &session, const PedersenBasis &basis,
                             const std::string &label,
                             const std::vector<ScalarOpening> &inputs,
                             const std::vector<F> &coefficients,
                             const F &value, ScalarOpening &output,
                             std::string *reason) {
    output = zkgpt::native_pedersen::commit_fresh(basis, value);
    if (!masked_emit_opening(session, basis, label + "/output", output, reason))
        return false;
    std::vector<ScalarOpening> operands{output};
    std::vector<F> relation_coefficients{F_ONE};
    operands.insert(operands.end(), inputs.begin(), inputs.end());
    for (const F &coefficient : coefficients)
        relation_coefficients.push_back(-coefficient);
    return zkgpt::native_pedersen::prove_hidden_linear(
        session, basis, label + "/relation", operands, relation_coefficients,
        F_ZERO, nullptr, reason);
}

struct MaskedGkrInputSource {
    enum class Kind {
        LayerU0,
        LayerV0,
        LayerCanonicalCopy,
        FconnLeft,
        FconnRight
    };
    Kind kind = Kind::LayerU0;
    std::size_t layer_id = 0;
    std::uint32_t layer_type = 0;
    std::size_t descriptor_index = 0;
    const zkgpt::NativeVector<u64> *input_coordinates = nullptr;
    zkgpt::NativeVector<u64> canonical_copy_coordinates;
    NativeGkrMatmulShape shape;
    std::vector<F> point;
    std::vector<F> outer_point;
    ScalarOpening masked_opening;
    ScalarOpening unmasked_opening;
    ScalarCommitment masked_commitment;
    ScalarCommitment unmasked_commitment;
    F masked_value = F_ZERO;
};

void masked_append_u64(zkgpt::NativeStreamingDigest& output, std::uint64_t value) { output.u64(value); }
void masked_append_string(zkgpt::NativeStreamingDigest& output, const std::string& value) { output.string(value); }

void masked_append_u64(std::string &output, std::uint64_t value) {
    for (unsigned byte = 0; byte < 8; ++byte)
        output.push_back(static_cast<char>((value >> (8 * byte)) & 0xff));
}

void masked_append_string(std::string &output, const std::string &value) {
    masked_append_u64(output, static_cast<std::uint64_t>(value.size()));
    output.append(value);
}

void masked_append_fields(std::string &output, const std::vector<F> &values) {
    masked_append_u64(output, static_cast<std::uint64_t>(values.size()));
    for (const F &value : values)
        masked_append_string(output, value.getStr(16));
}

bool masked_power_of_two(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

const char *masked_input_segment_name(std::size_t index) {
    static const char *const names[] = {"X_pub", "W", "Aux", "L_qry"};
    return index < 4 ? names[index] : "";
}

std::size_t masked_segment_index(const std::string &name) {
    for (std::size_t index = 0; index < 4; ++index)
        if (name == masked_input_segment_name(index)) return index;
    return 4;
}

std::string masked_input_coordinate_map_digest(
    const zkgpt::NativeVector<u64> &coordinates, std::size_t input_size,
    std::size_t segment_domain,
    const std::vector<std::size_t> &segment_logical_lengths);

const zkgpt::NativeVector<u64> &masked_source_input_coordinates(
    const MaskedGkrInputSource &source) {
    if (source.kind == MaskedGkrInputSource::Kind::LayerCanonicalCopy)
        return source.canonical_copy_coordinates;
    if (source.input_coordinates == nullptr)
        throw std::invalid_argument("masked GKR input source map is missing");
    return *source.input_coordinates;
}

bool masked_canonical_input_copy_coordinates(
    const layer &current, std::size_t input_size,
    std::size_t segment_domain,
    const std::vector<std::size_t> &segment_logical_lengths,
    zkgpt::NativeVector<u64> &coordinates, std::string *reason) {
    coordinates.clear();
    if (current.ty != layerType::ADD_BIAS || current.size == 0 ||
        current.need_phase2 || !current.bin_gates.empty() ||
        current.uni_gates.size() != current.size ||
        current.size_u[0] != current.ori_id_u.size() ||
        segment_domain == 0 || segment_logical_lengths.size() != 4 ||
        segment_domain > std::numeric_limits<std::size_t>::max() / 4 ||
        segment_domain * 4 != input_size) {
        masked_set_reason(reason,
                          "masked GKR branch is not a canonical input-only copy layer");
        return false;
    }
    const std::size_t aux_base = 2 * segment_domain;
    if (segment_logical_lengths[2] > segment_domain) {
        masked_set_reason(reason,
                          "masked GKR canonical Aux branch segment exceeds its domain");
        return false;
    }
    coordinates.resize(current.size);
    zkgpt::NativeVector<unsigned char> output_seen(current.size, 0);
    for (const uniGate &gate : current.uni_gates) {
        if (gate.g >= current.size || output_seen[gate.g] || gate.lu != 0 ||
            gate.sc != 1 || gate.u >= current.ori_id_u.size()) {
            masked_set_reason(reason,
                              "masked GKR input-only copy gate map is not one-to-one by output");
            return false;
        }
        const std::size_t coordinate = current.ori_id_u[gate.u];
        if (coordinate < aux_base || coordinate - aux_base >=
                                         segment_logical_lengths[2] ||
            coordinate >= input_size) {
            masked_set_reason(reason,
                              "masked GKR input-only copy source is not canonical Aux");
            return false;
        }
        coordinates[gate.g] = coordinate;
        output_seen[gate.g] = true;
    }
    if (std::find(output_seen.begin(), output_seen.end(), false) !=
        output_seen.end()) {
        masked_set_reason(reason,
                          "masked GKR input-only copy relation omits an output coordinate");
        return false;
    }
    return true;
}

std::string masked_input_copy_branch_identity(
    std::size_t layer_id, const layer &current,
    const zkgpt::NativeVector<u64> &coordinates, std::size_t input_size,
    std::size_t segment_domain,
    const std::vector<std::size_t> &segment_logical_lengths) {
    zkgpt::NativeStreamingDigest encoded("zkgpt-gkr-input-copy-branch-anchor-v2");
    masked_append_u64(encoded, 2);
    masked_append_u64(encoded, static_cast<std::uint64_t>(layer_id));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.ty));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.size));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.zero_start_id));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.bit_length));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.bit_length_u[0]));
    masked_append_u64(encoded, static_cast<std::uint64_t>(current.size_u[0]));
    masked_append_u64(encoded, current.need_phase2 ? 1 : 0);
    masked_append_string(encoded, "input-only-copy-permutation-to-Aux-v1");
    masked_append_u64(encoded, static_cast<std::uint64_t>(input_size));
    masked_append_u64(encoded, static_cast<std::uint64_t>(segment_domain));
    masked_append_u64(encoded, static_cast<std::uint64_t>(segment_logical_lengths.size()));
    for (const std::size_t length : segment_logical_lengths)
        masked_append_u64(encoded, static_cast<std::uint64_t>(length));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(current.ori_id_u.size()));
    for (const u64 coordinate : current.ori_id_u)
        masked_append_u64(encoded, coordinate);
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(current.uni_gates.size()));
    for (const uniGate &gate : current.uni_gates) {
        masked_append_u64(encoded, gate.g);
        masked_append_u64(encoded, gate.u);
        masked_append_u64(encoded, gate.lu);
        masked_append_u64(encoded, static_cast<std::uint64_t>(gate.sc));
    }
    masked_append_string(
        encoded, masked_input_coordinate_map_digest(
                     coordinates, input_size, segment_domain,
                     segment_logical_lengths));
    masked_append_string(encoded, "output-MLE-point-pulls-back-to-ordered-Aux-source-map-v1");
    return encoded.finish();
}

bool masked_input_segment_domain(
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::size_t input_size, std::size_t &segment_domain,
    std::vector<std::size_t> &segment_logical_lengths,
    std::string *reason) {
    segment_domain = 0;
    segment_logical_lengths.clear();
    bool saw_synthetic = false;
    bool saw_canonical = false;
    for (const NativeGkrMatmulShape &shape : matmuls) {
        if (shape.input_segment_domain == 0) {
            saw_synthetic = true;
            if (saw_canonical) {
                masked_set_reason(reason,
                                  "masked GKR mixes canonical and synthetic MatMul mappings");
                return false;
            }
            if (!shape.lhs_segment.empty() || !shape.rhs_segment.empty() ||
                shape.lhs_segment_offset != 0 || shape.rhs_segment_offset != 0 ||
                !shape.input_segment_logical_lengths.empty() ||
                !shape.output_segment.empty() ||
                shape.output_segment_offset != 0) {
                masked_set_reason(reason,
                                  "masked GKR partial canonical segment mapping");
                return false;
            }
            continue;
        }
        saw_canonical = true;
        if (saw_synthetic) {
            masked_set_reason(reason,
                              "masked GKR mixes synthetic and canonical MatMul mappings");
            return false;
        }
        if (segment_domain != 0 &&
            segment_domain != shape.input_segment_domain) {
            masked_set_reason(reason,
                              "masked GKR MatMul segment domains disagree");
            return false;
        }
        if (shape.input_segment_logical_lengths.size() != 4) {
            masked_set_reason(reason,
                              "masked GKR canonical segment lengths are incomplete");
            return false;
        }
        if (segment_logical_lengths.empty()) {
            segment_logical_lengths = shape.input_segment_logical_lengths;
        } else if (segment_logical_lengths !=
                   shape.input_segment_logical_lengths) {
            masked_set_reason(reason,
                              "masked GKR MatMul segment lengths disagree");
            return false;
        }
        segment_domain = shape.input_segment_domain;
    }
    if (segment_domain == 0) return true;
    if (segment_domain > std::numeric_limits<std::size_t>::max() / 4 ||
        segment_domain * 4 != input_size) {
        masked_set_reason(reason,
                          "masked GKR canonical four-segment input domain mismatch");
        return false;
    }
    for (const std::size_t length : segment_logical_lengths) {
        if (length > segment_domain) {
            masked_set_reason(reason,
                              "masked GKR canonical segment exceeds physical basis");
            return false;
        }
    }
    return true;
}

bool masked_validate_matmul_segment(const NativeGkrMatmulShape &shape,
                                    std::size_t input_size,
                                    std::string *reason) {
    if (shape.input_segment_domain == 0) {
        if (!shape.lhs_segment.empty() || !shape.rhs_segment.empty() ||
            shape.lhs_segment_offset != 0 || shape.rhs_segment_offset != 0 ||
            !shape.input_segment_logical_lengths.empty() ||
            !shape.output_segment.empty() ||
            shape.output_segment_offset != 0) {
            masked_set_reason(reason,
                              "masked GKR MatMul segment metadata is incomplete");
            return false;
        }
        return true;
    }
    if (shape.rows == 0 || shape.inner == 0 || shape.cols == 0 ||
        shape.rows > std::numeric_limits<std::size_t>::max() / shape.inner ||
        shape.inner > std::numeric_limits<std::size_t>::max() / shape.cols ||
        shape.rows > std::numeric_limits<std::size_t>::max() / shape.cols) {
        masked_set_reason(reason, "masked GKR MatMul dimensions overflow");
        return false;
    }
    const std::size_t domain = shape.input_segment_domain;
    if (domain > std::numeric_limits<std::size_t>::max() / 4 ||
        domain * 4 != input_size ||
        shape.input_segment_logical_lengths.size() != 4) {
        masked_set_reason(reason,
                          "masked GKR MatMul segment domain does not cover input");
        return false;
    }
    for (const std::size_t length : shape.input_segment_logical_lengths) {
        if (length > domain) {
            masked_set_reason(reason,
                              "masked GKR MatMul logical segment exceeds basis");
            return false;
        }
    }
    const auto validate_side = [&](const std::string &segment,
                                   std::size_t segment_offset,
                                   std::size_t absolute_offset,
                                   std::size_t span,
                                   const char *side) {
        const std::size_t index = masked_segment_index(segment);
        if (index >= 4 || segment_offset > domain ||
            span > domain - segment_offset ||
            span > shape.input_segment_logical_lengths[index] ||
            segment_offset >
                shape.input_segment_logical_lengths[index] - span ||
            index > std::numeric_limits<std::size_t>::max() / domain ||
            absolute_offset != index * domain + segment_offset) {
            masked_set_reason(reason,
                              std::string("masked GKR MatMul ") + side +
                                  " segment/offset/span mismatch");
            return false;
        }
        return true;
    };
    if (shape.rows != 0 &&
        shape.inner > std::numeric_limits<std::size_t>::max() / shape.rows) {
        masked_set_reason(reason, "masked GKR MatMul LHS span overflows");
        return false;
    }
    if (shape.inner != 0 &&
        shape.cols > std::numeric_limits<std::size_t>::max() / shape.inner) {
        masked_set_reason(reason, "masked GKR MatMul RHS span overflows");
        return false;
    }
    const std::size_t output_index = masked_segment_index(shape.output_segment);
    const std::size_t output_span = shape.rows * shape.cols;
    const bool output_valid = output_index < 4 &&
        shape.output_segment_offset <= domain &&
        output_span <= domain - shape.output_segment_offset &&
        output_span <= shape.input_segment_logical_lengths[output_index] &&
        shape.output_segment_offset <=
            shape.input_segment_logical_lengths[output_index] - output_span &&
        output_index <= std::numeric_limits<std::size_t>::max() / domain &&
        shape.output_offset ==
            output_index * domain + shape.output_segment_offset;
    if (!output_valid) {
        masked_set_reason(reason,
                          "masked GKR MatMul output segment/offset/span mismatch");
        return false;
    }
    return validate_side(shape.lhs_segment, shape.lhs_segment_offset,
                         shape.lhs_offset, shape.rows * shape.inner, "LHS") &&
           validate_side(shape.rhs_segment, shape.rhs_segment_offset,
                         shape.rhs_offset, shape.inner * shape.cols, "RHS");
}

std::string masked_source_kind_name(MaskedGkrInputSource::Kind kind) {
    switch (kind) {
    case MaskedGkrInputSource::Kind::LayerU0: return "layer-u0";
    case MaskedGkrInputSource::Kind::LayerV0: return "layer-v0";
    case MaskedGkrInputSource::Kind::LayerCanonicalCopy:
        return "layer-canonical-input-copy";
    case MaskedGkrInputSource::Kind::FconnLeft: return "fconn-left";
    case MaskedGkrInputSource::Kind::FconnRight: return "fconn-right";
    }
    return "invalid";
}

std::string masked_input_coordinate_map_digest(
    const zkgpt::NativeVector<u64> &coordinates, std::size_t input_size,
    std::size_t segment_domain,
    const std::vector<std::size_t> &segment_logical_lengths) {
    zkgpt::NativeStreamingDigest encoded("zkgpt-masked-gkr-input-map-v2");
    masked_append_u64(encoded, 2);
    masked_append_u64(encoded, static_cast<std::uint64_t>(input_size));
    masked_append_u64(encoded, static_cast<std::uint64_t>(segment_domain));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(segment_logical_lengths.size()));
    for (const std::size_t length : segment_logical_lengths)
        masked_append_u64(encoded, static_cast<std::uint64_t>(length));
    masked_append_u64(encoded, static_cast<std::uint64_t>(coordinates.size()));
    for (u64 coordinate : coordinates) {
        masked_append_u64(encoded, coordinate);
        if (segment_domain != 0 && coordinate < input_size) {
            const std::size_t segment = coordinate / segment_domain;
            masked_append_string(encoded, masked_input_segment_name(segment));
            const std::size_t local = coordinate % segment_domain;
            masked_append_u64(encoded, local);
            const bool logical = segment < segment_logical_lengths.size() &&
                                 local < segment_logical_lengths[segment];
            masked_append_u64(encoded, logical ? 1 : 0);
        } else {
            masked_append_string(encoded, "input");
            masked_append_u64(encoded, coordinate);
            masked_append_u64(encoded, 1);
        }
    }
    return encoded.finish();
}

std::string masked_input_source_ledger(
    const std::vector<MaskedGkrInputSource> &sources,
    std::size_t input_size, std::size_t segment_domain,
    const std::vector<std::size_t> &segment_logical_lengths) {
    std::string encoded = "zkgpt-masked-gkr-source-ledger-v1";
    masked_append_u64(encoded, static_cast<std::uint64_t>(input_size));
    masked_append_u64(encoded, static_cast<std::uint64_t>(segment_domain));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(segment_logical_lengths.size()));
    for (const std::size_t length : segment_logical_lengths)
        masked_append_u64(encoded, static_cast<std::uint64_t>(length));
    masked_append_u64(encoded, static_cast<std::uint64_t>(sources.size()));
    for (const MaskedGkrInputSource &source : sources) {
        masked_append_string(encoded, masked_source_kind_name(source.kind));
        masked_append_u64(encoded, static_cast<std::uint64_t>(source.layer_id));
        masked_append_u64(encoded, source.layer_type);
        masked_append_u64(encoded,
                          static_cast<std::uint64_t>(source.descriptor_index));
        masked_append_fields(encoded, source.point);
        masked_append_fields(encoded, source.outer_point);
        if (source.kind == MaskedGkrInputSource::Kind::LayerU0 ||
            source.kind == MaskedGkrInputSource::Kind::LayerV0 ||
            source.kind == MaskedGkrInputSource::Kind::LayerCanonicalCopy) {
            const zkgpt::NativeVector<u64> &coordinates =
                masked_source_input_coordinates(source);
            masked_append_string(
                encoded,
                source.kind == MaskedGkrInputSource::Kind::LayerCanonicalCopy
                    ? "input-only-copy-permutation-to-Aux-v1"
                    : "native-initSubset-input-coordinate-map-v1");
            masked_append_u64(
                encoded, static_cast<std::uint64_t>(coordinates.size()));
            masked_append_string(
                encoded,
                masked_input_coordinate_map_digest(coordinates, input_size,
                                                   segment_domain,
                                                   segment_logical_lengths));
        } else {
            const NativeGkrMatmulShape &shape = source.shape;
            masked_append_string(encoded, shape.relation_id);
            masked_append_string(encoded, shape.family_id);
            masked_append_u64(encoded, static_cast<std::uint64_t>(shape.rows));
            masked_append_u64(encoded, static_cast<std::uint64_t>(shape.inner));
            masked_append_u64(encoded, static_cast<std::uint64_t>(shape.cols));
            masked_append_u64(encoded,
                              static_cast<std::uint64_t>(shape.lhs_offset));
            masked_append_u64(encoded,
                              static_cast<std::uint64_t>(shape.rhs_offset));
            masked_append_u64(encoded,
                              static_cast<std::uint64_t>(shape.input_segment_domain));
            masked_append_string(encoded, shape.lhs_segment);
            masked_append_string(encoded, shape.rhs_segment);
            masked_append_u64(
                encoded, static_cast<std::uint64_t>(shape.lhs_segment_offset));
            masked_append_u64(
                encoded, static_cast<std::uint64_t>(shape.rhs_segment_offset));
            masked_append_string(encoded, shape.output_segment);
            masked_append_u64(
                encoded, static_cast<std::uint64_t>(shape.output_segment_offset));
            masked_append_u64(
                encoded, static_cast<std::uint64_t>(shape.input_segment_logical_lengths.size()));
            for (const std::size_t length : shape.input_segment_logical_lengths)
                masked_append_u64(encoded, static_cast<std::uint64_t>(length));
            masked_append_string(encoded,
                                 "lhs-row-major-padded-inner-low-v1");
            masked_append_string(encoded,
                                 "rhs-column-major-padded-inner-low-v1");
            masked_append_string(encoded,
                                 "output-row-major-aux-segment-v1");
        }
        masked_append_string(encoded, source.unmasked_commitment.domain);
        masked_append_string(encoded,
                             source.unmasked_commitment.point.serializeToHexStr());
        masked_append_string(encoded, source.masked_commitment.domain);
        masked_append_string(encoded,
                             source.masked_commitment.point.serializeToHexStr());
        masked_append_string(encoded, source.masked_value.getStr(16));
    }
    F digest;
    digest.setHashOf(encoded.data(), encoded.size());
    return digest.getStr(16);
}

zkgpt::NativeFieldVector masked_equality_weights(const std::vector<F> &point) {
    zkgpt::NativeFieldVector weights(1, F_ONE);
    for (const F &challenge : point) {
        const std::size_t old_size = weights.size();
        weights.resize(old_size * 2);
        for (std::size_t index = 0; index < old_size; ++index) {
            const F value = weights[index];
            weights[index] = value * (F_ONE - challenge);
            weights[index + old_size] = value * challenge;
        }
    }
    return weights;
}

std::array<std::uint64_t, 4> masked_input_source_support_extents(
    const std::vector<MaskedGkrInputSource> &sources,
    std::uint64_t input_size, std::uint64_t segment_domain,
    const std::vector<std::size_t> &logical_lengths) {
    if (logical_lengths.size() != 4 || segment_domain == 0 ||
        (segment_domain & (segment_domain - 1)) != 0 ||
        segment_domain > std::numeric_limits<std::uint64_t>::max() / 4 ||
        4 * segment_domain != input_size)
        throw std::invalid_argument(
            "masked GKR source support layout is invalid");

    std::array<std::uint64_t, 4> extents{};
    for (const MaskedGkrInputSource &source : sources) {
        if (source.kind == MaskedGkrInputSource::Kind::LayerU0 ||
            source.kind == MaskedGkrInputSource::Kind::LayerV0 ||
            source.kind == MaskedGkrInputSource::Kind::LayerCanonicalCopy) {
            const zkgpt::NativeVector<u64> *coordinates = nullptr;
            try {
                coordinates = &masked_source_input_coordinates(source);
            } catch (const std::exception &) {
                throw std::invalid_argument(
                    "masked GKR direct source support map is missing");
            }
            for (const std::uint64_t coordinate : *coordinates) {
                if (coordinate >= input_size)
                    throw std::out_of_range(
                        "masked GKR direct source support exceeds input domain");
                const std::uint64_t segment = coordinate / segment_domain;
                const std::uint64_t local = coordinate % segment_domain;
                if (segment >= extents.size())
                    throw std::out_of_range(
                        "masked GKR direct source support segment is invalid");
                if (local < logical_lengths[static_cast<std::size_t>(segment)])
                    extents[static_cast<std::size_t>(segment)] = std::max(
                        extents[static_cast<std::size_t>(segment)], local + 1);
            }
            continue;
        }

        std::string validation_reason;
        if (!masked_validate_matmul_segment(source.shape, input_size,
                                            &validation_reason))
            throw std::invalid_argument(validation_reason.empty()
                                            ? "masked GKR FCONN source support is invalid"
                                            : validation_reason);
        const NativeGkrMatmulShape &shape = source.shape;
        const bool left = source.kind == MaskedGkrInputSource::Kind::FconnLeft;
        const std::string &segment_name =
            left ? shape.lhs_segment : shape.rhs_segment;
        const std::uint64_t local_begin = left
            ? static_cast<std::uint64_t>(shape.lhs_segment_offset)
            : static_cast<std::uint64_t>(shape.rhs_segment_offset);
        const std::uint64_t span = left
            ? static_cast<std::uint64_t>(shape.rows) * shape.inner
            : static_cast<std::uint64_t>(shape.inner) * shape.cols;
        const std::size_t segment = masked_segment_index(segment_name);
        if (segment >= extents.size() || local_begin > logical_lengths[segment] ||
            span > logical_lengths[segment] - local_begin)
            throw std::out_of_range(
                "masked GKR FCONN source support exceeds logical segment");
        extents[segment] = std::max(extents[segment], local_begin + span);
    }
    return extents;
}

struct MaskedInputFoldBlock {
    std::uint64_t begin = 0;
    zkgpt::NativeFieldVector values;
    std::uint64_t end() const { return begin + static_cast<std::uint64_t>(values.size()); }
};

using MaskedInputFoldBlocks = std::vector<MaskedInputFoldBlock>;

std::vector<NativeGkrInputSupportInterval> masked_pair_support(
    const std::vector<NativeGkrInputSupportInterval> &support) {
    std::vector<NativeGkrInputSupportInterval> pairs;
    pairs.reserve(support.size());
    for (const NativeGkrInputSupportInterval &interval : support) {
        if (interval.begin >= interval.end) continue;
        pairs.push_back({interval.begin / 2, interval.end / 2 + interval.end % 2});
    }
    std::vector<NativeGkrInputSupportInterval> merged;
    merged.reserve(pairs.size());
    for (const NativeGkrInputSupportInterval &interval : pairs) {
        if (!merged.empty() && interval.begin <= merged.back().end)
            merged.back().end = std::max(merged.back().end, interval.end);
        else
            merged.push_back(interval);
    }
    return merged;
}

const MaskedInputFoldBlock *masked_find_fold_block(
    const MaskedInputFoldBlocks &blocks, std::uint64_t index) {
    const auto found = std::upper_bound(
        blocks.begin(), blocks.end(), index,
        [](std::uint64_t coordinate, const MaskedInputFoldBlock &block) {
            return coordinate < block.begin;
        });
    if (found == blocks.begin()) return nullptr;
    const MaskedInputFoldBlock &candidate = *std::prev(found);
    return index < candidate.end() ? &candidate : nullptr;
}

F masked_fold_block_value(const MaskedInputFoldBlocks &blocks,
                          std::uint64_t index) {
    const MaskedInputFoldBlock *block = masked_find_fold_block(blocks, index);
    return block == nullptr ? F_ZERO
                            : block->values[static_cast<std::size_t>(index - block->begin)];
}

bool masked_support_contains(
    const std::vector<NativeGkrInputSupportInterval> &support,
    std::uint64_t index) {
    const auto found = std::upper_bound(
        support.begin(), support.end(), index,
        [](std::uint64_t coordinate, const NativeGkrInputSupportInterval &interval) {
            return coordinate < interval.begin;
        });
    if (found == support.begin()) return false;
    const NativeGkrInputSupportInterval &candidate = *std::prev(found);
    return index >= candidate.begin && index < candidate.end;
}

template <class ValueAt>
MaskedInputFoldBlocks masked_fold_blocks(
    const MaskedInputFoldBlocks &current,
    const std::vector<NativeGkrInputSupportInterval> &support,
    std::uint64_t old_domain, const F &challenge, bool materialized,
    ValueAt value_at) {
    const std::vector<NativeGkrInputSupportInterval> ranges =
        masked_pair_support(materialized ? [&]() {
            std::vector<NativeGkrInputSupportInterval> result;
            result.reserve(current.size());
            for (const MaskedInputFoldBlock &block : current)
                result.push_back({block.begin, block.end()});
            return result;
        }() : support);
    MaskedInputFoldBlocks next;
    next.reserve(ranges.size());
    for (const NativeGkrInputSupportInterval &range : ranges) {
        if (range.begin >= range.end || range.end > old_domain / 2 + old_domain % 2)
            throw std::logic_error("masked GKR folded support exceeds its active domain");
        if (range.end - range.begin > std::numeric_limits<std::size_t>::max())
            throw std::length_error("masked GKR folded support exceeds host storage");
        MaskedInputFoldBlock block;
        block.begin = range.begin;
        block.values.resize(static_cast<std::size_t>(range.end - range.begin));
        for (std::uint64_t index = range.begin; index < range.end; ++index) {
            const std::uint64_t even_index = index * 2;
            const F even = value_at(current, support, even_index, materialized);
            const F odd = even_index + 1 < old_domain
                              ? value_at(current, support, even_index + 1, materialized)
                              : F_ZERO;
            block.values[static_cast<std::size_t>(index - range.begin)] =
                even * (F_ONE - challenge) + odd * challenge;
        }
        next.push_back(std::move(block));
    }
    return next;
}

bool masked_fold_blocks_in_place(MaskedInputFoldBlocks& blocks,
                                 std::uint64_t old_domain,
                                 const F& challenge) {
    std::vector<NativeGkrInputSupportInterval> support;
    support.reserve(blocks.size());
    for (const MaskedInputFoldBlock& block : blocks)
        support.push_back({block.begin, block.end()});
    const std::vector<NativeGkrInputSupportInterval> ranges =
        masked_pair_support(support);
    if (ranges.size() != blocks.size()) return false;
    for (std::size_t block_index = 0; block_index < blocks.size(); ++block_index) {
        MaskedInputFoldBlock& block = blocks[block_index];
        const std::uint64_t old_begin = block.begin;
        const std::uint64_t old_end = block.end();
        const std::uint64_t next_begin = old_begin / 2;
        const std::uint64_t next_end = old_end / 2 + old_end % 2;
        if (ranges[block_index].begin != next_begin ||
            ranges[block_index].end != next_end || old_end > old_domain ||
            next_end - next_begin > std::numeric_limits<std::size_t>::max())
            return false;
    }
    for (MaskedInputFoldBlock& block : blocks) {
        const std::uint64_t old_begin = block.begin;
        const std::uint64_t old_end = block.end();
        const std::uint64_t next_begin = old_begin / 2;
        const std::uint64_t next_end = old_end / 2 + old_end % 2;
        const std::uint64_t next_size = next_end - next_begin;
        for (std::uint64_t index = 0; index < next_size; ++index) {
            const std::uint64_t pair = next_begin + index;
            const std::uint64_t even_coordinate = pair * 2;
            const std::uint64_t odd_coordinate = even_coordinate + 1;
            const F even = even_coordinate >= old_begin && even_coordinate < old_end
                ? block.values[static_cast<std::size_t>(even_coordinate - old_begin)] : F_ZERO;
            const F odd = odd_coordinate < old_domain && odd_coordinate >= old_begin && odd_coordinate < old_end
                ? block.values[static_cast<std::size_t>(odd_coordinate - old_begin)] : F_ZERO;
            block.values[static_cast<std::size_t>(index)] =
                even * (F_ONE - challenge) + odd * challenge;
        }
        block.begin = next_begin;
        block.values.resize(static_cast<std::size_t>(next_size));
    }
    return true;
}

void masked_validate_input_support(
    const NativeGkrInputSupport &support,
    const std::array<std::uint64_t, 4> &logical_lengths,
    std::uint64_t segment_domain) {
    for (std::size_t segment = 0; segment < support.size(); ++segment) {
        std::uint64_t previous_end = 0;
        for (const NativeGkrInputSupportInterval &interval : support[segment]) {
            if (interval.begin >= interval.end || interval.end > logical_lengths[segment] ||
                interval.end > segment_domain || interval.begin < previous_end)
                throw std::invalid_argument("masked GKR trusted support intervals are invalid");
            previous_end = interval.end;
        }
    }
}

NativeGkrInputSupport masked_input_default_support(
    const std::array<std::uint64_t, 4> &logical_lengths) {
    NativeGkrInputSupport support;
    for (std::size_t segment = 0; segment < support.size(); ++segment)
        if (logical_lengths[segment] != 0)
            support[segment].push_back({0, logical_lengths[segment]});
    return support;
}

class MaskedInputWitnessFold {
  public:
    explicit MaskedInputWitnessFold(const NativeGkrInputProvider &provider)
        : segment_domain_(provider.segment_domain),
          logical_lengths_(provider.logical_lengths), value_at_(provider.value_at),
          support_(provider.support_complete ? provider.segment_support :
                   masked_input_default_support(provider.logical_lengths)) {
        if (provider.full_size == 0 || segment_domain_ == 0 ||
            (segment_domain_ & (segment_domain_ - 1)) != 0 ||
            segment_domain_ > std::numeric_limits<std::uint64_t>::max() / 4 ||
            provider.full_size != 4 * segment_domain_ || !value_at_)
            throw std::invalid_argument("masked GKR witness provider domain is invalid");
        for (std::uint64_t length : logical_lengths_)
            if (length > segment_domain_)
                throw std::invalid_argument("masked GKR witness segment extent is invalid");
        masked_validate_input_support(support_, logical_lengths_, segment_domain_);
    }

    F segment_value(std::size_t segment, std::uint64_t index) const {
        if (segment >= segments_.size()) throw std::out_of_range("masked GKR witness segment is invalid");
        if (index >= (segment_domain_ >> local_rounds_)) return F_ZERO;
        if (materialized_) return masked_fold_block_value(segments_[segment], index);
        if (!masked_support_contains(support_[segment], index)) return F_ZERO;
        return value_at_(static_cast<std::uint64_t>(segment) * segment_domain_ + index);
    }
    const zkgpt::NativeFieldVector &selectors() const { return selectors_; }

    void begin_selector_fold() {
        selectors_.resize(4);
        for (std::size_t segment = 0; segment < segments_.size(); ++segment)
            selectors_[segment] = segment_value(segment, 0);
    }

    void fold_round(const F &challenge, std::uint64_t round,
                    std::uint64_t local_bits) {
        if (round < local_bits) {
            const std::uint64_t old_domain = segment_domain_ >> round;
            for (std::size_t segment = 0; segment < segments_.size(); ++segment) {
                if (!materialized_ || !masked_fold_blocks_in_place(segments_[segment], old_domain, challenge))
                    segments_[segment] = masked_fold_blocks(
                        segments_[segment], support_[segment], old_domain,
                        challenge, materialized_,
                        [this, segment](const MaskedInputFoldBlocks &current,
                                        const std::vector<NativeGkrInputSupportInterval> &,
                                        std::uint64_t index, bool materialized) {
                            if (materialized) return masked_fold_block_value(current, index);
                            if (!masked_support_contains(support_[segment], index)) return F_ZERO;
                            return value_at_(static_cast<std::uint64_t>(segment) * segment_domain_ + index);
                        });
            }
            materialized_ = true;
            local_rounds_ = round + 1;
            return;
        }
        if (selectors_.empty() || (selectors_.size() & 1U) != 0)
            throw std::logic_error("masked GKR selector witness fold is inconsistent");
        const std::size_t next_size = selectors_.size() / 2;
        for (std::size_t index = 0; index < next_size; ++index)
            selectors_[index] = selectors_[index * 2] * (F_ONE - challenge) +
                                selectors_[index * 2 + 1] * challenge;
        selectors_.resize(next_size);
    }

    F terminal_value() const {
        if (selectors_.size() != 1)
            throw std::logic_error("masked GKR witness fold is incomplete");
        return selectors_.front();
    }

  private:
    std::uint64_t segment_domain_ = 0;
    std::uint64_t local_rounds_ = 0;
    std::array<std::uint64_t, 4> logical_lengths_{};
    std::function<F(std::uint64_t)> value_at_;
    NativeGkrInputSupport support_;
    std::array<MaskedInputFoldBlocks, 4> segments_;
    zkgpt::NativeFieldVector selectors_;
    bool materialized_ = false;
};

class MaskedInputCoefficientOracle {
  public:
    MaskedInputCoefficientOracle(
        std::uint64_t input_size, std::uint64_t segment_domain,
        const std::vector<std::size_t> &logical_lengths,
        const NativeGkrInputSupport &support,
        const std::vector<MaskedGkrInputSource> &sources)
        : input_size_(input_size), segment_domain_(segment_domain),
          logical_lengths_(logical_lengths.begin(), logical_lengths.end()),
          support_(support) {
        if (logical_lengths.size() != coefficients_.size() || segment_domain_ == 0 ||
            (segment_domain_ & (segment_domain_ - 1)) != 0 ||
            segment_domain_ > std::numeric_limits<std::uint64_t>::max() / 4 ||
            4 * segment_domain_ != input_size_)
            throw std::invalid_argument("masked GKR input coefficient layout is invalid");
        local_bits_ = 0;
        for (std::uint64_t size = segment_domain_; size > 1; size >>= 1)
            ++local_bits_;
        masked_validate_input_support(support_, support_logical_lengths(logical_lengths), segment_domain_);
        const std::array<std::uint64_t, 4> source_extents =
            masked_input_source_support_extents(sources, input_size_, segment_domain_, logical_lengths);
        for (std::size_t segment = 0; segment < coefficients_.size(); ++segment) {
            if (source_extents[segment] > logical_lengths_[segment])
                throw std::invalid_argument("masked GKR input coefficient source extent exceeds its segment");
            if (segment != 3) {
                if (source_extents[segment] != 0)
                    coefficients_[segment].push_back(make_zero_block(0, source_extents[segment]));
                continue;
            }
            for (const NativeGkrInputSupportInterval &interval : support_[segment]) {
                if (interval.begin >= source_extents[segment]) continue;
                if (!power_of_two_interval(interval.begin, interval.end))
                    throw std::invalid_argument("masked GKR L_qry coefficient support is not a canonical padded query block");
                coefficients_[segment].push_back(make_zero_block(interval.begin, interval.end));
            }
        }
        selector_coefficients_.assign(4, F_ZERO);
    }

    std::uint64_t size() const { return input_size_; }
    std::uint64_t segment_domain() const { return segment_domain_; }
    const std::vector<std::size_t> &logical_lengths() const { return logical_lengths_; }

    bool add(std::uint64_t coordinate, const F &value, std::string *reason) {
        if (coordinate >= input_size_) {
            masked_set_reason(reason, "masked GKR source coefficient exceeds the canonical input domain");
            return false;
        }
        const std::uint64_t segment = coordinate / segment_domain_;
        const std::uint64_t local = coordinate % segment_domain_;
        if (segment >= coefficients_.size()) {
            masked_set_reason(reason, "masked GKR source coefficient segment is invalid");
            return false;
        }
        if (local >= logical_lengths_[static_cast<std::size_t>(segment)]) {
            masked_set_reason(reason, "masked GKR source coefficient lands in canonical zero padding");
            return false;
        }
        if (!masked_support_contains(support_[static_cast<std::size_t>(segment)], local)) {
            masked_set_reason(reason, "masked GKR source coefficient falls outside its trusted live support interval");
            return false;
        }
        MaskedInputFoldBlock *block = find_block(coefficients_[static_cast<std::size_t>(segment)], local);
        if (block == nullptr) {
            masked_set_reason(reason, "masked GKR source coefficient has no allocated trusted support block");
            return false;
        }
        block->values[static_cast<std::size_t>(local - block->begin)] += value;
        return true;
    }

    F value_at(std::uint64_t coordinate) const {
        if (coordinate >= input_size_)
            throw std::out_of_range("masked GKR coefficient query exceeds the canonical input");
        const std::uint64_t segment = coordinate / segment_domain_;
        const std::uint64_t local = coordinate % segment_domain_;
        if (segment >= coefficients_.size()) return F_ZERO;
        return masked_fold_block_value(coefficients_[static_cast<std::size_t>(segment)], local);
    }

    F evaluate_mle(const std::vector<F> &point) const {
        if (point.size() != static_cast<std::size_t>(local_bits_ + 2))
            throw std::invalid_argument("masked GKR coefficient point has the wrong dimension");
        if (!folded_challenges_.empty()) {
            if (folded_challenges_ != point || folded_challenges_.size() != input_rounds())
                throw std::logic_error("masked GKR folded coefficient MLE requested at another point");
            return terminal_value();
        }
        std::array<F, 4> segment_values{};
        const std::vector<F> local_point(point.begin(), point.begin() + static_cast<std::ptrdiff_t>(local_bits_));
        for (std::size_t segment = 0; segment < coefficients_.size(); ++segment)
            segment_values[segment] = evaluate_blocks_mle(coefficients_[segment], local_point);
        const F s2 = point[local_bits_];
        const F s1 = point[local_bits_ + 1];
        const F low = segment_values[0] * (F_ONE - s2) + segment_values[1] * s2;
        const F high = segment_values[2] * (F_ONE - s2) + segment_values[3] * s2;
        return low * (F_ONE - s1) + high * s1;
    }

    void round_sums(const MaskedInputWitnessFold &witness,
                    std::uint64_t active, std::uint64_t round, F &even_sum,
                    F &odd_sum, F &at_two_sum) const {
        if (active == 0 || (active & (active - 1)) != 0 || round >= input_rounds())
            throw std::invalid_argument("masked GKR coefficient round domain is invalid");
        even_sum = F_ZERO;
        odd_sum = F_ZERO;
        at_two_sum = F_ZERO;
        if (round < local_bits_) {
            const std::uint64_t local_domain = segment_domain_ >> round;
            if (local_domain < 2 || active != 4 * local_domain)
                throw std::logic_error("masked GKR local coefficient fold is inconsistent");
            for (std::size_t segment = 0; segment < coefficients_.size(); ++segment) {
                std::vector<NativeGkrInputSupportInterval> ranges;
                ranges.reserve(coefficients_[segment].size());
                for (const MaskedInputFoldBlock &block : coefficients_[segment])
                    ranges.push_back({block.begin, block.end()});
                for (const NativeGkrInputSupportInterval &pairs : masked_pair_support(ranges))
                    for (std::uint64_t pair = pairs.begin; pair < pairs.end; ++pair) {
                        const std::uint64_t even_index = pair * 2;
                        const std::uint64_t odd_index = even_index + 1;
                        const F public_even = value_at(segment, even_index);
                        const F public_odd = value_at(segment, odd_index);
                        const F witness_even = witness.segment_value(segment, even_index);
                        const F witness_odd = witness.segment_value(segment, odd_index);
                        even_sum += public_even * witness_even;
                        odd_sum += public_odd * witness_odd;
                        at_two_sum += (F(2) * public_odd - public_even) * (F(2) * witness_odd - witness_even);
                    }
            }
            return;
        }
        if (active != selector_coefficients_.size() || active != (std::uint64_t(4) >> (round - local_bits_)))
            throw std::logic_error("masked GKR selector coefficient fold is inconsistent");
        const auto &selector_values = witness.selectors();
        if (selector_values.size() != active)
            throw std::logic_error("masked GKR selector witness fold has the wrong dimension");
        for (std::uint64_t index = 0; index < active; ++index) {
            const F product = selector_coefficients_[static_cast<std::size_t>(index)] * selector_values[static_cast<std::size_t>(index)];
            if (index & 1U) odd_sum += product; else even_sum += product;
        }
        for (std::uint64_t index = 0; index < active; index += 2) {
            const F public_slope = F(2) * selector_coefficients_[static_cast<std::size_t>(index + 1)] - selector_coefficients_[static_cast<std::size_t>(index)];
            const F witness_slope = F(2) * selector_values[static_cast<std::size_t>(index + 1)] - witness.selectors()[static_cast<std::size_t>(index)];
            at_two_sum += public_slope * witness_slope;
        }
    }

    void fold_round(const F &challenge, std::uint64_t active,
                    std::uint64_t round, MaskedInputWitnessFold &witness) {
        if (active < 2 || (active & 1U) != 0 || round >= input_rounds())
            throw std::invalid_argument("masked GKR coefficient fold domain is invalid");
        folded_challenges_.push_back(challenge);
        if (round < local_bits_) {
            const std::uint64_t old_domain = segment_domain_ >> round;
            for (std::size_t segment = 0; segment < coefficients_.size(); ++segment)
                if (!masked_fold_blocks_in_place(coefficients_[segment], old_domain, challenge))
                    coefficients_[segment] = masked_fold_blocks(
                        coefficients_[segment], support_[segment], old_domain,
                        challenge, true,
                        [](const MaskedInputFoldBlocks &current,
                           const std::vector<NativeGkrInputSupportInterval> &,
                           std::uint64_t index, bool) {
                            return masked_fold_block_value(current, index);
                        });
            witness.fold_round(challenge, round, local_bits_);
        } else {
            if (active != selector_coefficients_.size())
                throw std::logic_error("masked GKR selector fold size mismatch");
            const std::size_t selector_half = selector_coefficients_.size() / 2;
            for (std::size_t index = 0; index < selector_half; ++index)
                selector_coefficients_[index] = selector_coefficients_[index * 2] * (F_ONE - challenge) + selector_coefficients_[index * 2 + 1] * challenge;
            selector_coefficients_.resize(selector_half);
            witness.fold_round(challenge, round, local_bits_);
        }
    }

    void begin_selector_fold(std::uint64_t active, MaskedInputWitnessFold &witness) {
        if (active != 4 || selector_coefficients_.size() != 4 || local_bits_ >= input_rounds())
            throw std::logic_error("masked GKR selector fold cannot begin");
        for (std::size_t segment = 0; segment < coefficients_.size(); ++segment)
            selector_coefficients_[segment] = value_at(segment, 0);
        witness.begin_selector_fold();
    }

    std::uint64_t input_rounds() const { return local_bits_ + 2; }
    F terminal_value() const {
        if (selector_coefficients_.size() != 1)
            throw std::logic_error("masked GKR public coefficient fold is incomplete");
        return selector_coefficients_.front();
    }

  private:
    static std::array<std::uint64_t, 4> support_logical_lengths(const std::vector<std::size_t> &lengths) {
        if (lengths.size() != 4) throw std::invalid_argument("masked GKR logical segment count is invalid");
        return {{static_cast<std::uint64_t>(lengths[0]), static_cast<std::uint64_t>(lengths[1]),
                 static_cast<std::uint64_t>(lengths[2]), static_cast<std::uint64_t>(lengths[3])}};
    }

    static MaskedInputFoldBlock *find_block(MaskedInputFoldBlocks &blocks, std::uint64_t index) {
        return const_cast<MaskedInputFoldBlock *>(masked_find_fold_block(blocks, index));
    }

    static bool power_of_two_interval(std::uint64_t begin, std::uint64_t end) {
        const std::uint64_t length = end - begin;
        return length != 0 && (length & (length - 1)) == 0 && begin % length == 0;
    }

    static MaskedInputFoldBlock make_zero_block(std::uint64_t begin, std::uint64_t end) {
        if (begin >= end || end - begin > std::numeric_limits<std::size_t>::max())
            throw std::length_error("masked GKR coefficient support exceeds host storage");
        MaskedInputFoldBlock block;
        block.begin = begin;
        block.values.assign(static_cast<std::size_t>(end - begin), F_ZERO);
        return block;
    }

    static F evaluate_blocks_mle(const MaskedInputFoldBlocks &blocks,
                                 const std::vector<F> &point) {
        F total = F_ZERO;
        for (const MaskedInputFoldBlock &block : blocks) {
            if (block.begin == 0) {
                total += zkgpt::native_mle_evaluate(block.values, point);
                continue;
            }
            const std::uint64_t length = static_cast<std::uint64_t>(block.values.size());
            if (length == 0 || (length & (length - 1)) != 0 || block.begin % length != 0)
                throw std::logic_error("masked GKR nonzero support block is not aligned for MLE evaluation");
            std::size_t low_bits = 0;
            for (std::uint64_t size = length; size > 1; size >>= 1) ++low_bits;
            if (low_bits > point.size()) throw std::logic_error("masked GKR support block exceeds MLE point");
            const std::vector<F> low_point(point.begin(), point.begin() + static_cast<std::ptrdiff_t>(low_bits));
            F high = F_ONE;
            for (std::size_t bit = low_bits; bit < point.size(); ++bit)
                high *= ((block.begin >> bit) & 1U) ? point[bit] : F_ONE - point[bit];
            total += high * zkgpt::native_mle_evaluate(block.values, low_point);
        }
        return total;
    }

    std::uint64_t input_size_ = 0;
    std::uint64_t segment_domain_ = 0;
    std::uint64_t local_bits_ = 0;
    std::vector<std::size_t> logical_lengths_;
    NativeGkrInputSupport support_;
    std::array<MaskedInputFoldBlocks, 4> coefficients_;
    zkgpt::NativeFieldVector selector_coefficients_;
    std::vector<F> folded_challenges_;
};

std::shared_ptr<const NativeMaskedPublicProvider>
masked_input_public_provider(
    const std::shared_ptr<MaskedInputCoefficientOracle> &oracle) {
    if (oracle == nullptr)
        throw std::invalid_argument("masked GKR public coefficient provider is missing");
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = oracle->size();
    provider->value_at = [oracle](std::uint64_t coordinate) {
        return oracle->value_at(coordinate);
    };
    provider->evaluate_mle = [oracle](const std::vector<F> &point) {
        return oracle->evaluate_mle(point);
    };
    return provider;
}

bool masked_add_input_source_weights(
    const MaskedGkrInputSource &source, const F &rho,
    MaskedInputCoefficientOracle &coefficients,
    std::string *reason) {
    if (source.kind == MaskedGkrInputSource::Kind::LayerU0 ||
        source.kind == MaskedGkrInputSource::Kind::LayerV0 ||
        source.kind == MaskedGkrInputSource::Kind::LayerCanonicalCopy) {
        const zkgpt::NativeVector<u64> *coordinates = nullptr;
        try {
            coordinates = &masked_source_input_coordinates(source);
        } catch (const std::exception &) {
            masked_set_reason(reason,
                              "masked GKR direct input source map is missing");
            return false;
        }
        if (source.point.size() >= std::numeric_limits<std::size_t>::digits || coordinates->size() > (std::size_t(1) << source.point.size())) {
            masked_set_reason(reason, "masked GKR direct input source exceeds its lane domain");
            return false;
        }
        bool valid = true;
        zkgpt::native_for_each_equality_weight(source.point, coordinates->size(), [&](std::size_t index, const F& weight) {
            const std::uint64_t coordinate = coordinates->at(index);
            if (coordinate >= coefficients.size()) { valid = false; return; }
            if (!coefficients.add(coordinate, rho * weight, reason)) valid = false;
        });
        if (!valid) {
            if (reason == nullptr || reason->empty()) masked_set_reason(reason, "masked GKR direct input source coordinate/segment is invalid");
            return false;
        }
        return true;
    }

    const NativeGkrMatmulShape &shape = source.shape;
    if (!masked_validate_matmul_segment(shape, static_cast<std::size_t>(coefficients.size()), reason))
        return false;
    const zkgpt::NativeFieldVector inner_weights = masked_equality_weights(source.point);
    const zkgpt::NativeFieldVector outer_weights =
        masked_equality_weights(source.outer_point);
    if (inner_weights.size() < shape.inner) {
        masked_set_reason(reason, "masked FCONN inner point has the wrong dimension");
        return false;
    }
    if (source.kind == MaskedGkrInputSource::Kind::FconnLeft) {
        if (outer_weights.size() < shape.rows) {
            masked_set_reason(reason, "masked FCONN row point has the wrong dimension");
            return false;
        }
        for (std::size_t row = 0; row < shape.rows; ++row)
            for (std::size_t term = 0; term < shape.inner; ++term) {
                const std::uint64_t coordinate =
                    static_cast<std::uint64_t>(shape.lhs_offset) +
                    static_cast<std::uint64_t>(row) * shape.inner + term;
                if (coordinate >= coefficients.size()) {
                    masked_set_reason(reason,
                                      "masked FCONN LHS coordinate is out of range");
                    return false;
                }
                if (!coefficients.add(coordinate,
                                      rho * outer_weights[row] * inner_weights[term], reason))
                    return false;
            }
        return true;
    }
    if (outer_weights.size() < shape.cols) {
        masked_set_reason(reason, "masked FCONN column point has the wrong dimension");
        return false;
    }
    for (std::size_t term = 0; term < shape.inner; ++term)
        for (std::size_t col = 0; col < shape.cols; ++col) {
            const std::uint64_t coordinate =
                static_cast<std::uint64_t>(shape.rhs_offset) +
                static_cast<std::uint64_t>(term) * shape.cols + col;
            if (coordinate >= coefficients.size()) {
                masked_set_reason(reason,
                                  "masked FCONN RHS coordinate is out of range");
                return false;
            }
            if (!coefficients.add(coordinate,
                                  rho * inner_weights[term] * outer_weights[col], reason))
                return false;
        }
    return true;
}

bool masked_fconn_output_binding_shape(
    const layeredCircuit &circuit, std::size_t binding_layer_id,
    const NativeGkrMatmulShape &shape, std::string *reason) {
    if (binding_layer_id == 0 || binding_layer_id >= circuit.circuit.size()) {
        masked_set_reason(reason,
                          "masked GKR FCONN output-binding layer id is invalid");
        return false;
    }
    const layer &binding = circuit.circuit[binding_layer_id];
    const layer &fconn = circuit.circuit[binding_layer_id - 1];
    if (fconn.ty != layerType::FCONN || binding.ty != layerType::ADD_BIAS ||
        !masked_fconn_shape_matches(fconn, shape, reason) ||
        shape.output_segment != "Aux" || binding.size != fconn.size ||
        binding.zero_start_id != binding.size || binding.need_phase2 ||
        !binding.bin_gates.empty() ||
        binding.size > std::numeric_limits<std::size_t>::max() / 2 ||
        binding.uni_gates.size() != 2 * binding.size ||
        shape.rows > std::numeric_limits<std::size_t>::max() / shape.cols ||
        shape.rows * shape.cols != binding.size ||
        binding.ori_id_u.size() != binding.size) {
        if (reason != nullptr && reason->empty())
            *reason = "masked GKR FCONN output-binding statement is not canonical";
        return false;
    }
    for (std::size_t index = 0; index < binding.size; ++index) {
        const uniGate &previous = binding.uni_gates[index * 2];
        const uniGate &input = binding.uni_gates[index * 2 + 1];
        if (previous.g != index || previous.u != index || previous.lu != 1 ||
            previous.sc != 1 || input.g != index || input.lu != 0 ||
            input.sc != -1 || input.u >= binding.ori_id_u.size() ||
            binding.ori_id_u[input.u] != shape.output_offset + index) {
            masked_set_reason(reason,
                              "masked GKR FCONN output-binding gate map mismatch");
            return false;
        }
    }
    return true;
}

std::string masked_fconn_output_binding_identity(
    std::size_t binding_layer_id, const layer &binding,
    const NativeGkrMatmulShape &shape) {
    zkgpt::NativeStreamingDigest encoded("zkgpt-gkr-fconn-output-binding-anchor-v2");
    masked_append_u64(encoded, 2);
    masked_append_u64(encoded, static_cast<std::uint64_t>(binding_layer_id));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(binding_layer_id - 1));
    masked_append_u64(encoded, static_cast<std::uint64_t>(binding.ty));
    masked_append_string(encoded, shape.relation_id);
    masked_append_string(encoded, shape.family_id);
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.rows));
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.inner));
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.cols));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(shape.input_segment_domain));
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.lhs_offset));
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.rhs_offset));
    masked_append_string(encoded, shape.lhs_segment);
    masked_append_u64(
        encoded, static_cast<std::uint64_t>(shape.lhs_segment_offset));
    masked_append_string(encoded, shape.rhs_segment);
    masked_append_u64(
        encoded, static_cast<std::uint64_t>(shape.rhs_segment_offset));
    masked_append_u64(encoded, static_cast<std::uint64_t>(shape.output_offset));
    masked_append_string(encoded, shape.output_segment);
    masked_append_u64(
        encoded, static_cast<std::uint64_t>(shape.output_segment_offset));
    masked_append_u64(
        encoded, static_cast<std::uint64_t>(
                     shape.input_segment_logical_lengths.size()));
    for (const std::size_t length : shape.input_segment_logical_lengths)
        masked_append_u64(encoded, static_cast<std::uint64_t>(length));
    masked_append_string(encoded, "FCONN_output-canonical_Aux=0");
    masked_append_string(encoded, "row-major-exact-output-span-v1");
    masked_append_u64(encoded, static_cast<std::uint64_t>(binding.size));
    masked_append_u64(encoded,
                      static_cast<std::uint64_t>(binding.zero_start_id));
    masked_append_u64(encoded, static_cast<std::uint64_t>(binding.bit_length));
    for (std::size_t index = 0; index < binding.size; ++index) {
        const uniGate &previous = binding.uni_gates[index * 2];
        const uniGate &input = binding.uni_gates[index * 2 + 1];
        masked_append_u64(encoded, static_cast<std::uint64_t>(previous.g));
        masked_append_u64(encoded, static_cast<std::uint64_t>(previous.u));
        masked_append_u64(encoded, static_cast<std::uint64_t>(previous.lu));
        masked_append_u64(encoded, static_cast<std::uint64_t>(previous.sc));
        masked_append_u64(encoded, static_cast<std::uint64_t>(input.g));
        masked_append_u64(encoded, static_cast<std::uint64_t>(input.u));
        masked_append_u64(encoded, static_cast<std::uint64_t>(input.lu));
        masked_append_u64(encoded, static_cast<std::uint64_t>(input.sc));
        masked_append_u64(encoded, static_cast<std::uint64_t>(
            binding.ori_id_u[input.u]));
    }
    return encoded.finish();
}

bool masked_public_product_prove(
    NativeProofSession &session, const std::string &label,
    const PedersenBasis &basis,
    const std::shared_ptr<const zkgpt::native_pedersen::ProverFastPath>
        &fast_path,
    const std::shared_ptr<MaskedInputCoefficientOracle> &public_oracle,
    const std::shared_ptr<const NativeGkrInputProvider> &input_provider,
    const F &masked_initial, const F &unmasked_initial,
    const ScalarOpening &unmasked_initial_opening,
    const ScalarOpening &masked_initial_opening,
    NativeMaskedSumcheckResult &result, std::string *reason) {
    if (public_oracle == nullptr || input_provider == nullptr ||
        public_oracle->size() != input_provider->full_size ||
        !masked_power_of_two(input_provider->full_size)) {
        masked_set_reason(reason,
                          "masked GKR input aggregation domain is invalid");
        return false;
    }
    MaskedInputWitnessFold witness(*input_provider);
    std::uint64_t expected_rounds = 0;
    for (std::uint64_t active = public_oracle->size(); active > 1; active >>= 1)
        ++expected_rounds;
    if (public_oracle->input_rounds() != expected_rounds) {
        masked_set_reason(reason,
                          "masked GKR input coefficient round count is invalid");
        return false;
    }
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::Product2Public;
    spec.public_provider = masked_input_public_provider(public_oracle);
    spec.has_external_binding = true;
    spec.public_initial_claim = false;
    spec.initial_opening = unmasked_initial_opening;
    spec.initial_commitment = unmasked_initial_opening.commitment;
    spec.has_masked_initial_binding = true;
    spec.masked_initial_opening = masked_initial_opening;
    spec.masked_initial_commitment = masked_initial_opening.commitment;
    NativeMaskedSumcheckProver sumcheck(
        session, label, 2, masked_initial, spec, &unmasked_initial, &basis,
        fast_path);
    std::uint64_t active = public_oracle->size();
    for (std::uint64_t round = 0; active > 1; ++round) {
        if (round + 2 == public_oracle->input_rounds())
            public_oracle->begin_selector_fold(active, witness);
        F even_sum = F_ZERO, odd_sum = F_ZERO, at_two_sum = F_ZERO;
        public_oracle->round_sums(witness, active, round, even_sum, odd_sum,
                                  at_two_sum);
        const F quadratic = (at_two_sum + even_sum) / F(2) - odd_sum;
        const F linear = F(2) * odd_sum - at_two_sum / F(2) -
                         F(3) * even_sum / F(2);
        F challenge;
        if (!sumcheck.append_round(
                {even_sum, even_sum + linear + quadratic,
                 F(4) * quadratic + F(2) * linear + even_sum},
                &challenge, reason))
            return false;
        public_oracle->fold_round(challenge, active, round, witness);
        active >>= 1;
    }
    spec.public_evaluation = public_oracle->terminal_value();
    spec.has_public_evaluation = true;
    const F terminal_value = witness.terminal_value();
    spec.unmasked_terminal_openings = {
        zkgpt::native_pedersen::commit_fresh(basis, terminal_value, fast_path)};
    if (!sumcheck.set_terminal_spec(spec, reason) ||
        !sumcheck.finish({terminal_value}, &result, reason))
        return false;
    return true;
}

std::string masked_terminal_aggregate_rho_label(std::size_t index,
                                                std::size_t source_count) {
    if (source_count == 0 || index >= source_count)
        throw std::invalid_argument(
            "masked GKR terminal aggregate rho index is out of range");
    const std::size_t width = std::to_string(source_count - 1).size();
    std::string encoded_index = std::to_string(index);
    if (encoded_index.size() < width)
        encoded_index.insert(0, width - encoded_index.size(), '0');
    return "gkr/terminal-aggregate/rho/" + encoded_index;
}

bool masked_expect_message(const NativeProofSession &session, std::size_t &cursor,
                           NativeProofMessageKind kind, const std::string &label,
                           const NativeProofMessage **message,
                           std::string *reason) {
    if (cursor >= session.messages().size()) {
        masked_set_reason(reason, "masked GKR cursor reached transcript end at " + label);
        return false;
    }
    const NativeProofMessage &candidate = session.messages()[cursor];
    if (candidate.kind != kind || candidate.label != label) {
        masked_set_reason(reason, "masked GKR cursor expected " + label);
        return false;
    }
    if (message != nullptr) *message = &candidate;
    ++cursor;
    return true;
}

bool masked_expect_challenge(const NativeProofSession &session, std::size_t &cursor,
                             const std::string &label, F &value,
                             std::string *reason) {
    const NativeProofMessage *message = nullptr;
    if (!masked_expect_message(session, cursor, NativeProofMessageKind::Challenge,
                               "challenge/" + label, &message, reason) ||
        !decode_transcript_field(message->payload, value)) {
        masked_set_reason(reason, "malformed masked GKR challenge: " + label);
        return false;
    }
    return true;
}

bool masked_lookup_challenge(const NativeProofSession &session,
                             const std::string &label, F &value,
                             std::string *reason) {
    const std::string challenge_label = "challenge/" + label;
    const std::vector<std::size_t> *positions =
        session.message_positions(challenge_label);
    if (positions == nullptr || positions->empty()) {
        masked_set_reason(reason, "masked GKR challenge is missing or malformed");
        return false;
    }
    if (positions->size() != 1) {
        masked_set_reason(reason, "masked GKR challenge label is duplicated");
        return false;
    }
    const std::size_t found = positions->front();
    if (found >= session.messages().size() ||
        session.messages()[found].kind != NativeProofMessageKind::Challenge ||
        !decode_transcript_field(session.messages()[found].payload, value)) {
        masked_set_reason(reason, "masked GKR challenge is missing or malformed");
        return false;
    }
    return true;
}

bool masked_lookup_opening_commitment(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, ScalarCommitment &commitment,
    std::string *reason) {
    const std::string meta_label = label + "/meta";
    const std::vector<std::size_t> *positions =
        session.message_positions(meta_label);
    if (positions == nullptr || positions->empty()) {
        masked_set_reason(reason,
                          "masked GKR branch initial opening is missing or malformed");
        return false;
    }
    if (positions->size() != 1) {
        masked_set_reason(reason,
                          "masked GKR branch opening label is duplicated");
        return false;
    }
    const std::size_t found = positions->front();
    if (found >= session.messages().size() ||
        session.messages()[found].kind != NativeProofMessageKind::Data ||
        !native_masked_opening_commitment_at(session, basis, found, label,
                                             commitment, reason)) {
        masked_set_reason(reason,
                          "masked GKR branch initial opening is missing or malformed");
        return false;
    }
    return true;
}

bool masked_consume_opening(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, std::size_t &cursor,
    const ScalarCommitment *expected, ScalarCommitment *actual,
    std::string *reason) {
    const auto &messages = session.messages();
    if (cursor + 4 > messages.size() ||
        messages[cursor].label != label + "/meta" ||
        messages[cursor].kind != NativeProofMessageKind::Data) {
        masked_set_reason(reason, "masked GKR cursor expected opening " + label);
        return false;
    }
    ScalarCommitment commitment;
    if (!native_masked_opening_commitment_at(session, basis, cursor, label,
                                             commitment, reason))
        return false;
    if (expected != nullptr &&
        (commitment.domain != expected->domain ||
         commitment.point != expected->point)) {
        masked_set_reason(reason, "masked GKR opening commitment mismatch");
        return false;
    }
    if (!zkgpt::native_pedersen::verify_opening(session, basis, label,
                                                commitment, reason))
        return false;
    if (actual != nullptr) *actual = commitment;
    cursor += 4;
    return true;
}

bool masked_consume_linear(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, std::size_t &cursor,
    const std::vector<ScalarCommitment> &operands,
    const std::vector<F> &coefficients, const F &public_value,
    std::string *reason) {
    const auto &messages = session.messages();
    if (cursor + 6 > messages.size() ||
        messages[cursor].label != label + "/meta" ||
        messages[cursor].kind != NativeProofMessageKind::Data) {
        masked_set_reason(reason, "masked GKR cursor expected linear relation " + label);
        return false;
    }
    if (!zkgpt::native_pedersen::verify_hidden_linear(
            session, basis, label, operands, coefficients, public_value, reason))
        return false;
    cursor += 6;
    return true;
}

bool masked_consume_linear_claim(
    const NativeProofSession &session, const PedersenBasis &basis,
    const std::string &label, std::size_t &cursor,
    const std::vector<ScalarCommitment> &inputs,
    const std::vector<F> &coefficients, const F &public_value,
    ScalarCommitment &output, std::string *reason) {
    if (!masked_consume_opening(session, basis, label + "/output", cursor,
                                nullptr, &output, reason))
        return false;
    std::vector<ScalarCommitment> operands{output};
    std::vector<F> relation_coefficients{F_ONE};
    operands.insert(operands.end(), inputs.begin(), inputs.end());
    for (const F &coefficient : coefficients)
        relation_coefficients.push_back(-coefficient);
    return masked_consume_linear(session, basis, label + "/relation", cursor,
                                 operands, relation_coefficients, public_value,
                                 reason);
}

bool masked_consume_scope(
    const NativeProofSession &session, const std::string &label,
    std::size_t degree, std::size_t rounds, const F &masked_initial,
    const ScalarCommitment &unmasked_initial,
    const ScalarCommitment &masked_initial_commitment,
    const NativeMaskedTerminalSpec &spec, std::size_t &cursor,
    NativeMaskedSumcheckResult &result, std::string *reason) {
    if (cursor >= session.messages().size() ||
        session.messages()[cursor].label != label + "/initial/mode") {
        masked_set_reason(reason, "masked GKR cursor expected Sumcheck " + label);
        return false;
    }
    NativeMaskedTerminalSpec effective = spec;
    effective.has_external_binding = true;
    effective.public_initial_claim = spec.public_initial_claim;
    effective.initial_commitment = unmasked_initial;
    effective.has_masked_initial_binding = !spec.public_initial_claim;
    if (!spec.public_initial_claim)
        effective.masked_initial_commitment = masked_initial_commitment;
    if (!native_masked_sumcheck_verify(session, label, degree, rounds,
                                       masked_initial, effective, &result,
                                       reason))
        return false;
    if (result.transcript_end <= cursor ||
        result.transcript_end > session.messages().size()) {
        masked_set_reason(reason, "masked GKR Sumcheck cursor did not advance canonically");
        return false;
    }
    cursor = result.transcript_end;
    return true;
}

bool masked_advance_scope(const NativeProofSession &session, const std::string &label,
                          std::size_t rounds, std::size_t &cursor,
                          std::string *reason) {
    for (std::size_t round = 0; round < rounds; ++round) {
        const std::string base = label + "/mask/" + std::to_string(round);
        if (!masked_expect_message(session, cursor, NativeProofMessageKind::Data,
                                   base + "/commitment", nullptr, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::Challenge,
                                   "challenge/" + base + "/knowledge", nullptr, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                   base + "/response", nullptr, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                   label + "/poly/" + std::to_string(round), nullptr,
                                   reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::Challenge,
                                   "challenge/" + label + "/r/" +
                                       std::to_string(round), nullptr, reason))
            return false;
    }
    return masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                 label + "/terminal", nullptr, reason);
}

bool masked_scope_challenges(const NativeProofSession &session,
                             const std::string &label, std::size_t rounds,
                             std::size_t start, std::vector<F> &challenges,
                             std::string *reason) {
    challenges.clear();
    std::size_t cursor = start;
    for (std::size_t round = 0; round < rounds; ++round) {
        const NativeProofMessage *ignored = nullptr;
        const std::string base = label + "/mask/" + std::to_string(round);
        if (!masked_expect_message(session, cursor, NativeProofMessageKind::Data,
                                   base + "/commitment", &ignored, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::Challenge,
                                   "challenge/" + base + "/knowledge", &ignored, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                   base + "/response", &ignored, reason) ||
            !masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                   label + "/poly/" + std::to_string(round), &ignored,
                                   reason))
            return false;
        F challenge;
        if (!masked_expect_challenge(session, cursor,
                                     label + "/r/" + std::to_string(round),
                                     challenge, reason))
            return false;
        challenges.push_back(challenge);
    }
    return true;
}

bool masked_decode_public_output(const NativeProofMessage *message, const zkgpt::NativeFieldVector &expected, std::string *reason) {
    if (message == nullptr || message->kind != NativeProofMessageKind::Data || message->payload != zkgpt::native_field_vector_digest(expected, "zkgpt-gkr-public-output-v2")) { masked_set_reason(reason, "masked GKR public output binding failed"); return false; }
    return true;
}

bool masked_power_domain(std::size_t bits, std::size_t &domain,
                         std::string *reason) {
    if (bits >= sizeof(std::size_t) * 8) {
        masked_set_reason(reason, "masked GKR MLE dimension is too large");
        return false;
    }
    domain = std::size_t(1) << bits;
    return true;
}

bool masked_fold_matmul_operands(const NativeGkrMatmulWitness &witness,
                                 const std::vector<F> &left_point,
                                 const std::vector<F> &right_point,
                                 std::size_t &inner_domain,
                                 zkgpt::NativeFieldVector &left,
                                 zkgpt::NativeFieldVector &right,
                                 std::string *reason) {
    const auto &shape = witness.shape;
    if (shape.rows == 0 || shape.inner == 0 || shape.cols == 0) {
        masked_set_reason(reason, "masked FCONN descriptor dimensions are invalid");
        return false;
    }
    const std::size_t row_bits = ceilPow2BitLength(
        static_cast<u64>(shape.rows));
    const std::size_t inner_bits = ceilPow2BitLength(
        static_cast<u64>(shape.inner));
    const std::size_t col_bits = ceilPow2BitLength(
        static_cast<u64>(shape.cols));
    std::size_t rows_domain = 0, cols_domain = 0;
    if (!masked_power_domain(row_bits, rows_domain, reason) ||
        !masked_power_domain(inner_bits, inner_domain, reason) ||
        !masked_power_domain(col_bits, cols_domain, reason))
        return false;
    if (shape.rows > std::numeric_limits<std::size_t>::max() / shape.inner ||
        inner_domain > std::numeric_limits<std::size_t>::max() / cols_domain) {
        masked_set_reason(reason, "masked FCONN witness dimensions overflow");
        return false;
    }
    const std::size_t canonical_left_size = shape.rows * shape.inner;
    const std::size_t padded_right_size = inner_domain * cols_domain;
    if (left_point.size() != inner_bits + row_bits ||
        right_point.size() != inner_bits + col_bits) {
        masked_set_reason(reason, "masked FCONN fold point dimensions are invalid");
        return false;
    }
    if ((!witness.left_value_at && witness.left.size() != canonical_left_size) ||
        (!witness.right_value_at && witness.right.size() != padded_right_size) ||
        (witness.left_value_at && !witness.left.empty()) ||
        (witness.right_value_at && !witness.right.empty())) {
        masked_set_reason(reason, "masked FCONN witness matrix dimensions are invalid");
        return false;
    }
    for (std::size_t col = 0; col < cols_domain; ++col)
        for (std::size_t term = 0; term < inner_domain; ++term) {
            if (col < shape.cols && term < shape.inner) continue;
            const std::size_t index = col * inner_domain + term;
            const F value = witness.right_value_at
                ? witness.right_value_at(index)
                : witness.right[index];
            if (!value.isZero()) {
                masked_set_reason(reason,
                                  "masked FCONN RHS padding is not zero");
                return false;
            }
        }
    left.assign(inner_domain, F_ZERO);
    right.assign(inner_domain, F_ZERO);
    for (std::size_t row = 0; row < shape.rows; ++row) {
        F row_weight = F_ONE;
        for (std::size_t bit = 0; bit < row_bits; ++bit) {
            const F &challenge = left_point[inner_bits + bit];
            row_weight *= ((row >> bit) & 1U) ? challenge : F_ONE - challenge;
        }
        for (std::size_t term = 0; term < shape.inner; ++term) {
            const F value = witness.left_value_at
                ? witness.left_value_at(row * shape.inner + term)
                : witness.left[row * shape.inner + term];
            left[term] += row_weight * value;
        }
    }
    for (std::size_t col = 0; col < shape.cols; ++col) {
        F col_weight = F_ONE;
        for (std::size_t bit = 0; bit < col_bits; ++bit) {
            const F &challenge = right_point[inner_bits + bit];
            col_weight *= ((col >> bit) & 1U) ? challenge : F_ONE - challenge;
        }
        for (std::size_t term = 0; term < shape.inner; ++term) {
            const std::size_t index = col * inner_domain + term;
            const F value = witness.right_value_at
                ? witness.right_value_at(index)
                : witness.right[index];
            right[term] += col_weight * value;
        }
    }
    return true;
}

bool masked_product_prove(NativeProofSession &session, const std::string &label,
                          const PedersenBasis &basis,
                          const std::shared_ptr<const
                              zkgpt::native_pedersen::ProverFastPath> &fast_path,
                          zkgpt::NativeFieldVector left,
                          zkgpt::NativeFieldVector right,
                          std::size_t inner_bits,
                          const F &masked_initial, const F &unmasked_initial,
                          const ScalarOpening &initial_opening,
                          const ScalarOpening &masked_initial_opening,
                          NativeMaskedSumcheckResult &result,
                          std::string *reason) {
    if (left.size() != (std::size_t(1) << inner_bits) ||
        right.size() != (std::size_t(1) << inner_bits)) {
        masked_set_reason(reason, "masked FCONN product domain is invalid");
        return false;
    }
    NativeMaskedTerminalSpec spec;
    spec.kind = NativeMaskedTerminalKind::Product2;
    spec.has_external_binding = true;
    spec.public_initial_claim = false;
    spec.initial_opening = initial_opening;
    spec.initial_commitment = initial_opening.commitment;
    spec.has_masked_initial_binding = true;
    spec.masked_initial_opening = masked_initial_opening;
    spec.masked_initial_commitment = masked_initial_opening.commitment;
    NativeMaskedSumcheckProver sumcheck(
        session, label, 2, masked_initial, spec, &unmasked_initial, &basis,
        fast_path);
    zkgpt::NativeFieldVector f = std::move(left);
    zkgpt::NativeFieldVector g = std::move(right);
    F previous_random = F_ZERO;
    std::size_t active = f.size();
    for (std::size_t round = 0; round < inner_bits; ++round) {
        const std::size_t half = active >> 1;
        F A = F_ZERO, B = F_ZERO, C = F_ZERO;
        for (std::size_t index = 0; index < half; ++index) {
            const F zf = F(2) * f[index * 2 + 1] - f[index * 2];
            const F zg = F(2) * g[index * 2 + 1] - g[index * 2];
            C += zf * zg;
        }
        for (std::size_t index = 0; index < active; ++index) {
            if (index & 1) B += f[index] * g[index];
            else A += f[index] * g[index];
        }
        const F a = (C + A) / F(2) - B;
        const F b = F(2) * B - C / F(2) - F(3) * A / F(2);
        const F c = A;
        F challenge;
        if (!sumcheck.append_round({c, a + b + c,
                                    F(4) * a + F(2) * b + c},
                                   &challenge, reason))
            return false;
        zkgpt::NativeFieldVector next_f(half), next_g(half);
        for (std::size_t index = 0; index < half; ++index) {
            next_f[index] = f[index * 2] * (F_ONE - challenge) +
                            f[index * 2 + 1] * challenge;
            next_g[index] = g[index * 2] * (F_ONE - challenge) +
                            g[index * 2 + 1] * challenge;
        }
        f.swap(next_f);
        g.swap(next_g);
        active = half;
        previous_random = challenge;
    }
    (void)previous_random;
    spec.unmasked_terminal_openings = {
        zkgpt::native_pedersen::commit_fresh(sumcheck.result().basis, f[0],
                                             fast_path),
        zkgpt::native_pedersen::commit_fresh(sumcheck.result().basis, g[0],
                                             fast_path)};
    if (!sumcheck.set_terminal_spec(spec, reason)) return false;
    return sumcheck.finish({f[0], g[0]}, &result, reason);
}

F masked_eq_at(const std::vector<F> &point, u64 index) {
    if (point.size() >= 64 || (point.size() < 64 && (index >> point.size()) != 0))
        return F_ZERO;
    F value = F_ONE;
    for (std::size_t bit = 0; bit < point.size(); ++bit)
        value *= ((index >> bit) & 1) != 0 ? point[bit] : F_ONE - point[bit];
    return value;
}

std::vector<F> masked_gkr_coefficients(const layer &current,
                                        const std::vector<F> &r0,
                                        const std::vector<F> &r1,
                                        const std::vector<F> &ru,
                                        const F &alpha, const F &beta,
                                        const F &relu_rou,
                                        const std::vector<F> *rv = nullptr) {
    F uni[2] = {F_ZERO, F_ZERO};
    F bin[3] = {F_ZERO, F_ZERO, F_ZERO};
    const auto gate_weight = [&](u64 gate, u64 input, u64 right) {
        F weight = (alpha * masked_eq_at(r0, gate) +
                    beta * masked_eq_at(r1, gate)) *
                   masked_eq_at(ru, input);
        if (current.zero_start_id < current.size && gate >= current.zero_start_id)
            weight *= relu_rou;
        if (rv != nullptr) weight *= masked_eq_at(*rv, right);
        return weight;
    };
    for (const uniGate &gate : current.uni_gates) {
        F weight = gate_weight(gate.g, gate.u, 0);
        uni[gate.lu] += weight * gate.sc;
    }
    for (const binGate &gate : current.bin_gates)
        bin[gate.l] += gate_weight(gate.g, gate.u, gate.v) * gate.sc;
    return {bin[0], bin[1], bin[2], uni[0], uni[1]};
}

F masked_fixed_u_predicate(const std::vector<F> &coefficients,
                           const std::vector<F> &u,
                           const std::vector<F> &v) {
    return coefficients[0] * u[0] * v[0] +
           coefficients[1] * u[1] * v[1] +
           coefficients[2] * u[1] * v[0] +
           coefficients[3] * u[0] + coefficients[4] * u[1];
}

F masked_fixed_v_predicate(const std::vector<F> &coefficients,
                           const std::vector<F> &u,
                           const std::vector<F> &v) {
    return masked_fixed_u_predicate(coefficients, u, v);
}

bool masked_validate_circuit(const layeredCircuit &circuit,
                             const zkgpt::NativeFieldVector &public_output,
                             std::string *reason) {
    if (circuit.size != circuit.circuit.size() || circuit.size < 2) {
        masked_set_reason(reason, "masked GKR circuit shape is invalid");
        return false;
    }
    const layer &last = circuit.circuit.back();
    if (last.size == 0 || last.bit_length < 0 || public_output.size() != last.size) {
        masked_set_reason(reason, "masked GKR public output shape is invalid");
        return false;
    }
    for (const layer &current : circuit.circuit) {
        if (current.size == 0 || current.bit_length < 0 || current.bit_length >= 63 ||
            (current.ty != layerType::FCONN && current.max_bl_u < 0) ||
            (current.need_phase2 && current.ty != layerType::FCONN &&
             current.max_bl_v < 0)) {
            masked_set_reason(reason, "masked GKR layer dimensions are invalid");
            return false;
        }
        if (current.ty == layerType::FCONN) {
            const bool annotated = !current.native_relation_id.empty() ||
                                    !current.native_family_id.empty() ||
                                    current.native_rows != 0 ||
                                    current.native_inner != 0 ||
                                    current.native_cols != 0 ||
                                    current.native_lhs_offset != 0 ||
                                    current.native_rhs_offset != 0 ||
                                    current.native_output_offset != 0;
            if (!current.uni_gates.empty() || !current.bin_gates.empty() ||
                !current.ori_id_u.empty() || !current.ori_id_v.empty() ||
                (annotated && (current.native_relation_id.empty() ||
                               current.native_family_id.empty() ||
                               current.native_rows == 0 ||
                               current.native_inner == 0 ||
                               current.native_cols == 0))) {
                masked_set_reason(reason, "masked FCONN layer is not a canonical native relation");
                return false;
            }
        }
    }
    return true;
}

void masked_gkr_progress(const std::string &message) {
    std::cerr << "[zkGPT][gkr] " << message << std::endl;
}

}

bool prove_masked_gkr_transcript(
    NativeProofSession &session, const layeredCircuit &circuit,
    NativeLayeredGkrWitness &witness,
    const zkgpt::NativeFieldVector &public_output, std::string *reason,
    NativeGkrInputBinding *input_binding,
    NativeGkrProverResult *prover_result) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (prover_result != nullptr)
        {
            prover_result->has_unmasked_input_claim_opening = false;
            prover_result->has_unmasked_input_value_opening = false;
        }
        if (!session.valid(reason) || !masked_validate_circuit(circuit, public_output, reason))
            return false;
        if (witness.values().size() != circuit.circuit.size()) {
            masked_set_reason(reason, "masked GKR layer witness count mismatch");
            return false;
        }
        if (witness.input_provider != nullptr &&
            (!witness.input_provider->value_at ||
             witness.input_provider->full_size != circuit.circuit.front().size)) {
            masked_set_reason(reason, "masked GKR canonical input provider is malformed");
            return false;
        }
        if (witness.input_provider != nullptr && !witness.values().front().empty()) {
            masked_set_reason(reason, "masked GKR input layer must use its segmented provider");
            return false;
        }
        if (witness.input_provider == nullptr &&
            witness.values().front().size() != circuit.circuit.front().size) {
            masked_set_reason(reason, "masked GKR legacy input layer shape mismatch");
            return false;
        }
        for (std::size_t index = 1; index < circuit.circuit.size(); ++index)
            if (witness.values()[index].size() != circuit.circuit[index].size) {
                masked_set_reason(reason, "masked GKR layer witness shape mismatch");
                return false;
            }
        if (witness.values().back().size() != public_output.size() || !std::equal(public_output.begin(), public_output.end(), witness.values().back().begin())) {
            masked_set_reason(reason, "masked GKR public output does not match layer witness");
            return false;
        }
        std::size_t fconn_count = 0;
        for (std::size_t index = 1; index < circuit.circuit.size(); ++index)
            if (circuit.circuit[index].ty == layerType::FCONN) ++fconn_count;
        if (fconn_count != witness.matmuls.size()) {
            masked_set_reason(reason, "masked GKR FCONN descriptor count mismatch");
            return false;
        }
        masked_gkr_progress("begin layers=" + std::to_string(circuit.size) +
                            " fconn=" + std::to_string(fconn_count));

        session.absorb("gkr/public-output-digest-v2", zkgpt::native_field_vector_digest(public_output, "zkgpt-gkr-public-output-v2"));
        std::vector<std::vector<F>> r_u(circuit.size + 1);
        std::vector<std::vector<F>> r_v(circuit.size + 1);
        const layer &last = circuit.circuit.back();
        r_u[circuit.size].resize(last.bit_length);
        for (int bit = 0; bit < last.bit_length; ++bit)
            r_u[circuit.size][bit] = session.challenge(
                "gkr/output/r/" + std::to_string(bit));
        F previous_unmasked = transcript_mle_eval(public_output,
                                                   r_u[circuit.size]);
        F previous_masked = previous_unmasked;
        const PedersenBasis basis = masked_basis(session);
        const auto fast_path =
            zkgpt::native_pedersen::make_prover_fast_path(session, basis);
        const std::uint64_t input_size_u64 =
            witness.input_provider != nullptr
                ? witness.input_provider->full_size
                : witness.values().front().size();
        if (input_size_u64 >
            static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
            throw std::runtime_error("masked GKR input domain exceeds host indexing");
        const std::size_t input_size = static_cast<std::size_t>(input_size_u64);
        if (!masked_power_of_two(input_size))
            throw std::runtime_error(
                "masked GKR canonical input domain is not a power of two");
        std::vector<NativeGkrMatmulShape> descriptor_shapes;
        descriptor_shapes.reserve(witness.matmuls.size());
        for (const NativeGkrMatmulWitness &matmul : witness.matmuls) {
            descriptor_shapes.push_back(matmul.shape);
            if (!masked_validate_matmul_segment(matmul.shape, input_size, reason))
                throw std::runtime_error(*reason);
        }
        std::size_t input_segment_domain = 0;
        std::vector<std::size_t> input_segment_logical_lengths;
        if (!masked_input_segment_domain(descriptor_shapes, input_size,
                                         input_segment_domain,
                                         input_segment_logical_lengths, reason))
            throw std::runtime_error(*reason);
        std::shared_ptr<const NativeGkrInputProvider> input_provider =
            witness.input_provider;
        if (input_provider == nullptr) {
            auto legacy_provider = std::make_shared<NativeGkrInputProvider>();
            legacy_provider->full_size = input_size;
            legacy_provider->segment_domain = input_segment_domain;
            if (input_segment_logical_lengths.size() != 4)
                throw std::runtime_error("masked GKR legacy segment metadata is incomplete");
            std::copy(input_segment_logical_lengths.begin(),
                      input_segment_logical_lengths.end(),
                      legacy_provider->logical_lengths.begin());
            const zkgpt::NativeFieldVector *legacy_input =
                &witness.values().front();
            legacy_provider->value_at =
                [legacy_input, input_size](std::uint64_t coordinate) {
                    if (coordinate >= input_size)
                        throw std::out_of_range("masked GKR legacy input coordinate is invalid");
                    return legacy_input->at(static_cast<std::size_t>(coordinate));
                };
            input_provider = legacy_provider;
        }
        if (input_segment_logical_lengths.size() != 4 ||
            input_provider->segment_domain != input_segment_domain ||
            !std::equal(input_segment_logical_lengths.begin(),
                        input_segment_logical_lengths.end(),
                        input_provider->logical_lengths.begin()))
            throw std::runtime_error(
                "masked GKR input provider extents do not match canonical FCONN segments");
        const ScalarOpening public_base{
            zkgpt::native_pedersen::commit(basis, F_ZERO, F_ZERO),
            F_ZERO, F_ZERO};
        const ScalarOpening public_claim{
            zkgpt::native_pedersen::commit(basis, previous_unmasked, F_ZERO),
            previous_unmasked, F_ZERO};
        if (!masked_emit_opening(session, basis, "gkr/output/public-base",
                                 public_base, reason) ||
            !masked_emit_opening(session, basis, "gkr/output/public-claim",
                                 public_claim, reason) ||
            !zkgpt::native_pedersen::prove_public_affine(
                session, basis, "gkr/output/public-affine", public_base,
                public_claim, F_ZERO, previous_unmasked, nullptr, reason))
            throw std::runtime_error(reason != nullptr && !reason->empty()
                                         ? *reason : "public output binding failed");
        ScalarOpening previous_unmasked_opening = public_claim;
        ScalarOpening previous_masked_opening = public_claim;
        F previous_u1_unmasked = F_ZERO, previous_u1_masked = F_ZERO;
        ScalarOpening previous_u1_unmasked_opening;
        ScalarOpening previous_u1_masked_opening;
        bool have_previous_u1 = false;
        bool previous_lane_active = true;
        F alpha = F_ONE, beta = F_ZERO;

        prover native_prover;
        native_prover.native_circuit = &circuit;
        native_prover.native_values = &witness.values();
        if (input_provider != nullptr)
            native_prover.native_input_value_at = input_provider->value_at;
        native_prover.init();
        native_prover.sumcheckInitAll(r_u[circuit.size].begin());

        std::size_t next_matmul = witness.matmuls.size();
        std::vector<MaskedGkrInputSource> input_sources;
        for (int layer_id = static_cast<int>(circuit.size) - 1;
             layer_id > 0; --layer_id) {
            const layer &current = circuit.circuit[layer_id];
            F relu_rou = F_ONE;
            const std::string prefix = "gkr/layer/" + std::to_string(layer_id);
            masked_gkr_progress(
                "layer=" + std::to_string(layer_id) + " begin ty=" +
                std::to_string(static_cast<int>(current.ty)) +
                " size=" + std::to_string(current.size) +
                " bits=" + std::to_string(current.bit_length) +
                " u=" + std::to_string(current.max_bl_u) +
                " v=" + std::to_string(current.max_bl_v) +
                " phase2=" + (current.need_phase2 ? "1" : "0"));
            session.absorb(prefix + "/begin", "masked-layer-v1");
            if (current.zero_start_id < current.size)
                relu_rou = session.challenge("gkr/layer/" +
                                             std::to_string(layer_id) + "/relu");

            bool public_zero_branch_anchor = false;
            if (!previous_lane_active) {
                const std::string anchor_label = prefix + "/branch-anchor";
                if (layer_id > 1 && next_matmul > 0 &&
                    circuit.circuit[layer_id - 1].ty == layerType::FCONN) {
                    const NativeGkrMatmulShape &branch_shape =
                        witness.matmuls[next_matmul - 1].shape;
                    if (!masked_fconn_output_binding_shape(
                            circuit, static_cast<std::size_t>(layer_id),
                            branch_shape, reason))
                        throw std::runtime_error(*reason);
                    session.absorb(anchor_label + "/identity",
                                   masked_fconn_output_binding_identity(
                                       static_cast<std::size_t>(layer_id),
                                       current, branch_shape));
                    std::vector<F> branch_point(current.bit_length);
                    for (int bit = 0; bit < current.bit_length; ++bit)
                        branch_point[bit] = session.challenge(
                            anchor_label + "/r/" + std::to_string(bit));
                    if (!transcript_mle_eval(witness.values()[layer_id],
                                             branch_point).isZero())
                        throw std::runtime_error(
                            "masked GKR FCONN residual branch is not zero at its bound point");
                    r_u[layer_id + 1] = branch_point;
                    r_v[layer_id + 1].assign(branch_point.size(), F_ZERO);
                    previous_unmasked = F_ZERO;
                    previous_masked = F_ZERO;
                    previous_unmasked_opening = public_base;
                    previous_masked_opening = public_base;
                    public_zero_branch_anchor = true;
                } else {
                    zkgpt::NativeVector<u64> canonical_copy_coordinates;
                    if (!masked_canonical_input_copy_coordinates(
                            current, input_size, input_segment_domain,
                            input_segment_logical_lengths,
                            canonical_copy_coordinates, reason))
                        throw std::runtime_error(
                            "masked GKR inactive lane has no canonical source anchor: " +
                            (reason != nullptr ? *reason : std::string()));
                    session.absorb(
                        anchor_label + "/identity",
                        masked_input_copy_branch_identity(
                            static_cast<std::size_t>(layer_id), current,
                            canonical_copy_coordinates, input_size,
                            input_segment_domain,
                            input_segment_logical_lengths));
                    std::vector<F> branch_point(current.bit_length);
                    for (int bit = 0; bit < current.bit_length; ++bit)
                        branch_point[bit] = session.challenge(
                            anchor_label + "/r/" + std::to_string(bit));
                    const F unmasked_branch_claim = transcript_mle_eval(
                        witness.values()[layer_id], branch_point);
                    F branch_mask;
                    branch_mask.setByCSPRNG();
                    const F masked_branch_claim =
                        unmasked_branch_claim + branch_mask;
                    previous_unmasked_opening =
                        zkgpt::native_pedersen::commit_fresh(
                            basis, unmasked_branch_claim, fast_path);
                    previous_masked_opening =
                        zkgpt::native_pedersen::commit_fresh(
                            basis, masked_branch_claim, fast_path);
                    session.absorb_field(anchor_label + "/masked-claim",
                                         masked_branch_claim);
                    MaskedGkrInputSource source;
                    source.kind =
                        MaskedGkrInputSource::Kind::LayerCanonicalCopy;
                    source.layer_id = static_cast<std::size_t>(layer_id);
                    source.layer_type =
                        static_cast<std::uint32_t>(current.ty);
                    source.canonical_copy_coordinates =
                        std::move(canonical_copy_coordinates);
                    source.point = branch_point;
                    source.masked_opening = previous_masked_opening;
                    source.unmasked_opening = previous_unmasked_opening;
                    source.masked_commitment =
                        previous_masked_opening.commitment;
                    source.unmasked_commitment =
                        previous_unmasked_opening.commitment;
                    source.masked_value = masked_branch_claim;
                    input_sources.push_back(std::move(source));
                    r_u[layer_id + 1] = branch_point;
                    r_v[layer_id + 1].assign(branch_point.size(), F_ZERO);
                    previous_unmasked = unmasked_branch_claim;
                    previous_masked = masked_branch_claim;
                }
                previous_lane_active = true;
                alpha = F_ONE;
                beta = F_ZERO;
            }
            native_prover.sumcheckInit(alpha, beta);

            if (current.ty == layerType::FCONN) {
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " fconn begin");
                if (next_matmul == 0) {
                    masked_set_reason(reason, "masked GKR FCONN descriptor underflow");
                    throw std::runtime_error(*reason);
                }
                const std::size_t descriptor_index = --next_matmul;
                const NativeGkrMatmulWitness &matmul =
                    witness.matmuls[descriptor_index];
                if (!masked_fconn_shape_matches(current, matmul.shape, reason))
                    throw std::runtime_error(*reason);
                if (!masked_validate_matmul_segment(matmul.shape, input_size,
                                                    reason))
                    throw std::runtime_error(*reason);
                const std::size_t n = ceilPow2BitLength(
                    static_cast<u64>(matmul.shape.rows));
                const std::size_t m = ceilPow2BitLength(
                    static_cast<u64>(matmul.shape.inner));
                const std::size_t k = ceilPow2BitLength(
                    static_cast<u64>(matmul.shape.cols));
                if (r_u[layer_id + 1].size() < n + k)
                    throw std::runtime_error("masked GKR FCONN predecessor point is too short");
                r_u[layer_id].assign(m + n, F_ZERO);
                r_v[layer_id].assign(m + k, F_ZERO);
                for (std::size_t bit = m; bit < m + n; ++bit)
                    r_u[layer_id][bit] = r_u[layer_id + 1][bit - m + k];
                for (std::size_t bit = m; bit < m + k; ++bit)
                    r_v[layer_id][bit] = r_u[layer_id + 1][bit - m];
                for (std::size_t bit = 0; bit < m; ++bit) {
                    const std::string label = prefix + "/matmul_point/r_u/" +
                                              std::to_string(bit);
                    r_u[layer_id][bit] = session.challenge(label);
                    r_v[layer_id][bit] = r_u[layer_id][bit];
                }
                std::size_t inner_domain = 0;
                zkgpt::NativeFieldVector left_mle, right_mle;
                if (!masked_fold_matmul_operands(
                        matmul, r_u[layer_id], r_v[layer_id], inner_domain,
                        left_mle, right_mle, reason))
                    throw std::runtime_error(*reason);
                if (inner_domain != (std::size_t(1) << m))
                    throw std::runtime_error(
                        "masked GKR FCONN folded inner dimension mismatch");
                const F initial_unmasked = have_previous_u1
                    ? previous_u1_unmasked : previous_unmasked;
                const F initial_masked = have_previous_u1
                    ? previous_u1_masked : previous_masked;
                NativeMaskedSumcheckResult product_result;
                if (!masked_product_prove(session, prefix + "/matmul", basis,
                                           fast_path,
                                           std::move(left_mle), std::move(right_mle), m, initial_masked,
                                           initial_unmasked,
                                           have_previous_u1
                                               ? previous_u1_unmasked_opening
                                               : previous_unmasked_opening,
                                           have_previous_u1
                                               ? previous_u1_masked_opening
                                               : previous_masked_opening,
                                           product_result, reason))
                    throw std::runtime_error(*reason);
                if (product_result.terminal_values.size() != 2)
                    throw std::runtime_error("masked GKR FCONN terminal arity mismatch");
                if (product_result.terminal_openings.size() != 2 ||
                    product_result.unmasked_terminal_openings.size() != 2 ||
                    product_result.terminal_commitments.size() != 2 ||
                    product_result.unmasked_terminal_commitments.size() != 2 ||
                    product_result.challenges.size() != m)
                    throw std::runtime_error(
                        "masked GKR FCONN terminal source binding is incomplete");
                MaskedGkrInputSource left_source;
                left_source.kind = MaskedGkrInputSource::Kind::FconnLeft;
                left_source.layer_id = static_cast<std::size_t>(layer_id);
                left_source.layer_type = static_cast<std::uint32_t>(current.ty);
                left_source.descriptor_index = descriptor_index;
                left_source.shape = matmul.shape;
                left_source.point = product_result.challenges;
                left_source.outer_point.assign(
                    r_u[layer_id].begin() + static_cast<std::ptrdiff_t>(m),
                    r_u[layer_id].end());
                left_source.masked_opening = product_result.terminal_openings[0];
                left_source.unmasked_opening =
                    product_result.unmasked_terminal_openings[0];
                left_source.masked_commitment =
                    product_result.terminal_commitments[0];
                left_source.unmasked_commitment =
                    product_result.unmasked_terminal_commitments[0];
                left_source.masked_value = product_result.terminal_values[0];
                input_sources.push_back(std::move(left_source));

                MaskedGkrInputSource right_source;
                right_source.kind = MaskedGkrInputSource::Kind::FconnRight;
                right_source.layer_id = static_cast<std::size_t>(layer_id);
                right_source.layer_type = static_cast<std::uint32_t>(current.ty);
                right_source.descriptor_index = descriptor_index;
                right_source.shape = matmul.shape;
                right_source.point = product_result.challenges;
                right_source.outer_point.assign(
                    r_v[layer_id].begin() + static_cast<std::ptrdiff_t>(m),
                    r_v[layer_id].end());
                right_source.masked_opening = product_result.terminal_openings[1];
                right_source.unmasked_opening =
                    product_result.unmasked_terminal_openings[1];
                right_source.masked_commitment =
                    product_result.terminal_commitments[1];
                right_source.unmasked_commitment =
                    product_result.unmasked_terminal_commitments[1];
                right_source.masked_value = product_result.terminal_values[1];
                input_sources.push_back(std::move(right_source));
                previous_u1_unmasked = F_ZERO;
                previous_u1_masked = F_ZERO;
                have_previous_u1 = false;
                previous_unmasked = F_ZERO;
                previous_masked = F_ZERO;
                previous_unmasked_opening = ScalarOpening{};
                previous_masked_opening = ScalarOpening{};
                if (current.bit_length_u[1] >= 0)
                    alpha = session.challenge(prefix + "/alpha");
                else alpha.clear();
                if (current.bit_length_v[1] >= 0 || current.ty == layerType::FFT)
                    beta = session.challenge(prefix + "/beta");
                else beta.clear();
                previous_lane_active = false;
                previous_u1_unmasked_opening = ScalarOpening{};
                previous_u1_masked_opening = ScalarOpening{};
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " fconn end rounds=" + std::to_string(m));
                continue;
            }

            NativeMaskedTerminalSpec phase1_placeholder;
            phase1_placeholder.kind = NativeMaskedTerminalKind::Claims;
            phase1_placeholder.has_external_binding = true;
            phase1_placeholder.public_initial_claim = public_zero_branch_anchor;
            phase1_placeholder.initial_opening = public_zero_branch_anchor
                ? public_base : previous_unmasked_opening;
            phase1_placeholder.initial_commitment =
                phase1_placeholder.initial_opening.commitment;
            phase1_placeholder.has_masked_initial_binding =
                !public_zero_branch_anchor;
            if (!public_zero_branch_anchor) {
                phase1_placeholder.masked_initial_opening = previous_masked_opening;
                phase1_placeholder.masked_initial_commitment =
                    previous_masked_opening.commitment;
            }
            NativeMaskedSumcheckProver phase1(
                session, prefix + "/phase1", 2, previous_masked,
                phase1_placeholder,
                public_zero_branch_anchor ? nullptr : &previous_unmasked,
                &basis, fast_path);
            masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                " phase1 begin rounds=" +
                                std::to_string(current.max_bl_u));
            native_prover.sumcheckInitPhase1(relu_rou);
            F previous_random = F_ZERO;
            F phase1_unmasked = previous_unmasked;
            for (int bit = 0; bit < current.max_bl_u; ++bit) {
                const quadratic_poly poly = native_prover.sumcheckUpdate1(previous_random);
                if (phase1_unmasked != poly.eval(F_ZERO) + poly.eval(F_ONE))
                    throw std::runtime_error("masked GKR local phase1 claim mismatch");
                F challenge;
                if (!phase1.append_round({poly.eval(F_ZERO), poly.eval(F_ONE),
                                          poly.eval(F(2))}, &challenge, reason))
                    throw std::runtime_error("masked GKR phase1 append: " +
                                             (reason != nullptr ? *reason : std::string()));
                previous_random = challenge;
                phase1_unmasked = poly.eval(challenge);
            }
            F claim_u0 = F_ZERO, claim_u1 = F_ZERO;
            native_prover.sumcheckFinalize1(previous_random, claim_u0, claim_u1);
            masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                " phase1 rounds done");
            r_u[layer_id] = phase1.result().challenges;
            std::vector<F> r0 = r_u[layer_id + 1];
            std::vector<F> r1 = r_v[layer_id + 1];
            if (r0.size() < static_cast<std::size_t>(current.bit_length))
                r0.resize(current.bit_length, F_ZERO);
            if (r1.size() < static_cast<std::size_t>(current.bit_length))
                r1.resize(current.bit_length, F_ZERO);
            std::vector<F> coefficients = masked_gkr_coefficients(
                current, r0, r1, r_u[layer_id], alpha, beta, relu_rou);
            NativeMaskedSumcheckResult phase1_result;
            std::vector<F> masked_u;
            std::vector<F> masked_v{F_ZERO, F_ZERO};
            F phase2_unmasked = phase1_unmasked;
            NativeMaskedSumcheckResult phase2_result;
            if (current.need_phase2) {
                NativeMaskedTerminalSpec phase1_spec;
                phase1_spec.kind = NativeMaskedTerminalKind::Claims;
                phase1_spec.terminal_linear_coefficients = {F_ONE, F_ONE};
                phase1_spec.terminal_linear_offset = F_ZERO;
                phase1_spec.defer_terminal_relation = true;
                phase1_spec.unmasked_terminal_openings = {
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u0, fast_path),
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u1, fast_path)};
                if (!phase1.set_terminal_spec(phase1_spec, reason) ||
                    !phase1.finish({claim_u0, claim_u1}, &phase1_result, reason))
                    throw std::runtime_error(*reason);
                masked_u = phase1_result.terminal_values;
                if (phase1_result.terminal_values.size() != 2 ||
                    phase1_result.terminal_commitments.size() != 2 ||
                    phase1_result.terminal_openings.size() != 2) {
                    throw std::runtime_error(
                        "masked GKR phase1 terminal binding has invalid arity");
                }
                NativeMaskedTerminalSpec phase2_placeholder;
                phase2_placeholder.kind = NativeMaskedTerminalKind::Claims;
                phase2_placeholder.has_external_binding = true;
                phase2_placeholder.public_initial_claim = false;
                phase2_placeholder.initial_opening = phase1_result.unmasked_claim_opening;
                phase2_placeholder.initial_commitment =
                    phase1_result.unmasked_claim_commitment;
                phase2_placeholder.has_masked_initial_binding = true;
                phase2_placeholder.masked_initial_opening = phase1_result.claim_opening;
                phase2_placeholder.masked_initial_commitment =
                    phase1_result.claim_commitment;
                NativeMaskedSumcheckProver phase2(
                    session, prefix + "/phase2", 2, phase1_result.final_claim,
                    phase2_placeholder, &phase1_unmasked, &basis, fast_path);
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " phase2 begin rounds=" +
                                    std::to_string(current.max_bl_v));
                native_prover.sumcheckInitPhase2();
                previous_random = F_ZERO;
                for (int bit = 0; bit < current.max_bl_v; ++bit) {
                    const quadratic_poly poly = native_prover.sumcheckUpdate2(previous_random);
                    F challenge;
                    if (!phase2.append_round({poly.eval(F_ZERO), poly.eval(F_ONE),
                                              poly.eval(F(2))}, &challenge, reason))
                        throw std::runtime_error(*reason);
                    previous_random = challenge;
                    phase2_unmasked = poly.eval(challenge);
                }
                F claim_v0 = F_ZERO, claim_v1 = F_ZERO;
                native_prover.sumcheckFinalize2(previous_random, claim_v0, claim_v1);
                r_v[layer_id] = phase2.result().challenges;
                const std::vector<F> phase2_coefficients = masked_gkr_coefficients(
                    current, r0, r1, r_u[layer_id], alpha, beta, relu_rou,
                    &r_v[layer_id]);
                NativeMaskedTerminalSpec phase2_spec;
                phase2_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedU;
                phase2_spec.gkr_coefficients = phase2_coefficients;
                phase2_spec.fixed_values = phase1_result.terminal_values;
                phase2_spec.defer_terminal_relation =
                    phase2_spec.gkr_coefficients.size() == 5 &&
                    (phase2_spec.gkr_coefficients[0] *
                         phase2_spec.fixed_values[0] +
                     phase2_spec.gkr_coefficients[2] *
                         phase2_spec.fixed_values[1])
                            .isZero() &&
                    (phase2_spec.gkr_coefficients[1] *
                         phase2_spec.fixed_values[1])
                        .isZero();
                phase2_spec.unmasked_fixed_values = {claim_u0, claim_u1};
                phase2_spec.fixed_openings = phase1_result.terminal_openings;
                phase2_spec.fixed_commitments = phase1_result.terminal_commitments;
                phase2_spec.unmasked_fixed_openings =
                    phase1_result.unmasked_terminal_openings;
                phase2_spec.unmasked_terminal_openings = {
                    zkgpt::native_pedersen::commit_fresh(
                        phase2.result().basis, claim_v0, fast_path),
                    zkgpt::native_pedersen::commit_fresh(
                        phase2.result().basis, claim_v1, fast_path)};
                if (!phase2.set_terminal_spec(phase2_spec, reason) ||
                    !phase2.finish({claim_v0, claim_v1}, &phase2_result, reason))
                    throw std::runtime_error(*reason);
                masked_v = phase2_result.terminal_values;
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " phase2 end");
            } else {
                NativeMaskedTerminalSpec phase1_spec;
                phase1_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedV;
                phase1_spec.gkr_coefficients = coefficients;
                phase1_spec.fixed_values = {F_ZERO, F_ZERO};
                phase1_spec.defer_terminal_relation =
                    coefficients.size() == 5 && coefficients[3].isZero() &&
                    coefficients[4].isZero();
                phase1_spec.unmasked_fixed_values = {F_ZERO, F_ZERO};
                const ScalarOpening zero_opening{
                    zkgpt::native_pedersen::commit(
                        basis, F_ZERO, F_ZERO),
                    F_ZERO, F_ZERO, nullptr};
                phase1_spec.fixed_openings = {
                    zero_opening, zero_opening};
                phase1_spec.unmasked_fixed_openings = {zero_opening, zero_opening};
                phase1_spec.unmasked_terminal_openings = {
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u0, fast_path),
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u1, fast_path)};
                if (!phase1.set_terminal_spec(phase1_spec, reason) ||
                    !phase1.finish({claim_u0, claim_u1}, &phase1_result, reason))
                    throw std::runtime_error(*reason);
                masked_u = phase1_result.terminal_values;
                phase1_result.final_claim = phase1_result.final_claim;
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " phase1 terminal end");
            }
            if (masked_u.size() != 2 || masked_v.size() != 2)
                throw std::runtime_error("masked GKR terminal claim arity mismatch");
            if (current.bit_length_u[0] >= 0) {
                if (current.size_u[0] == 0 ||
                    current.ori_id_u.size() != current.size_u[0] ||
                    phase1_result.terminal_openings.size() != 2 ||
                    phase1_result.unmasked_terminal_openings.size() != 2 ||
                    phase1_result.terminal_commitments.size() != 2 ||
                    phase1_result.unmasked_terminal_commitments.size() != 2 ||
                    phase1_result.challenges.size() <
                        static_cast<std::size_t>(current.bit_length_u[0]))
                    throw std::runtime_error(
                        "masked GKR phase1 direct-input source binding is incomplete");
                MaskedGkrInputSource source;
                source.kind = MaskedGkrInputSource::Kind::LayerU0;
                source.layer_id = static_cast<std::size_t>(layer_id);
                source.layer_type = static_cast<std::uint32_t>(current.ty);
                source.input_coordinates = &current.ori_id_u;
                source.point.assign(
                    phase1_result.challenges.begin(),
                    phase1_result.challenges.begin() +
                        static_cast<std::ptrdiff_t>(current.bit_length_u[0]));
                source.masked_opening = phase1_result.terminal_openings[0];
                source.unmasked_opening =
                    phase1_result.unmasked_terminal_openings[0];
                source.masked_commitment = phase1_result.terminal_commitments[0];
                source.unmasked_commitment =
                    phase1_result.unmasked_terminal_commitments[0];
                source.masked_value = phase1_result.terminal_values[0];
                input_sources.push_back(std::move(source));
            }
            if (current.bit_length_v[0] >= 0) {
                if (!current.need_phase2 || current.size_v[0] == 0 ||
                    current.ori_id_v.size() != current.size_v[0] ||
                    phase2_result.terminal_openings.size() != 2 ||
                    phase2_result.unmasked_terminal_openings.size() != 2 ||
                    phase2_result.terminal_commitments.size() != 2 ||
                    phase2_result.unmasked_terminal_commitments.size() != 2 ||
                    phase2_result.challenges.size() <
                        static_cast<std::size_t>(current.bit_length_v[0]))
                    throw std::runtime_error(
                        "masked GKR phase2 direct-input source binding is incomplete");
                MaskedGkrInputSource source;
                source.kind = MaskedGkrInputSource::Kind::LayerV0;
                source.layer_id = static_cast<std::size_t>(layer_id);
                source.layer_type = static_cast<std::uint32_t>(current.ty);
                source.input_coordinates = &current.ori_id_v;
                source.point.assign(
                    phase2_result.challenges.begin(),
                    phase2_result.challenges.begin() +
                        static_cast<std::ptrdiff_t>(current.bit_length_v[0]));
                source.masked_opening = phase2_result.terminal_openings[0];
                source.unmasked_opening =
                    phase2_result.unmasked_terminal_openings[0];
                source.masked_commitment = phase2_result.terminal_commitments[0];
                source.unmasked_commitment =
                    phase2_result.unmasked_terminal_commitments[0];
                source.masked_value = phase2_result.terminal_values[0];
                input_sources.push_back(std::move(source));
            }
            previous_u1_unmasked = claim_u1;
            previous_u1_masked = masked_u[1];
            previous_u1_unmasked_opening =
                phase1_result.unmasked_terminal_openings[1];
            previous_u1_masked_opening = phase1_result.terminal_openings[1];
            have_previous_u1 = true;
            if (current.bit_length_u[1] >= 0)
                alpha = session.challenge(prefix + "/alpha");
            else alpha.clear();
            if (current.bit_length_v[1] >= 0 || current.ty == layerType::FFT)
                beta = session.challenge(prefix + "/beta");
            else beta.clear();
            previous_unmasked = alpha * claim_u1 + beta * (current.need_phase2 ?
                                                            phase2_unmasked : F_ZERO);
            previous_masked = alpha * masked_u[1] + beta *
                              (current.need_phase2 ? masked_v[1] : F_ZERO);
            const ScalarOpening phase2_unmasked_opening = current.need_phase2
                ? phase2_result.unmasked_terminal_openings[1]
                : phase1_result.unmasked_fixed_openings[1];
            const ScalarOpening phase2_masked_opening = current.need_phase2
                ? phase2_result.terminal_openings[1]
                : phase1_result.fixed_openings[1];
            ScalarOpening next_unmasked_opening;
            ScalarOpening next_masked_opening;
            if (!masked_make_linear_claim(
                    session, basis, prefix + "/next-claim/unmasked",
                    {previous_u1_unmasked_opening, phase2_unmasked_opening},
                    {alpha, beta}, previous_unmasked, next_unmasked_opening,
                    reason) ||
                !masked_make_linear_claim(
                    session, basis, prefix + "/next-claim/masked",
                    {previous_u1_masked_opening, phase2_masked_opening},
                    {alpha, beta}, previous_masked, next_masked_opening,
                    reason))
                throw std::runtime_error(*reason);
            previous_unmasked_opening = next_unmasked_opening;
            previous_masked_opening = next_masked_opening;
            masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                " end");
        }
        native_prover.release_sumcheck_storage();
        if (next_matmul != 0)
            throw std::runtime_error("masked GKR FCONN descriptor was not consumed");
        if (input_sources.empty())
            throw std::runtime_error(
                "masked GKR has no canonical input terminal claims to reduce");
        for (const MaskedGkrInputSource &source : input_sources) {
            if (source.masked_opening.value != source.masked_value ||
                source.masked_opening.commitment.point !=
                    source.masked_commitment.point ||
                source.unmasked_opening.commitment.point !=
                    source.unmasked_commitment.point)
                throw std::runtime_error(
                    "masked GKR input source opening/commitment mismatch");
        }
        session.absorb("gkr/input/source-ledger",
                       masked_input_source_ledger(input_sources, input_size,
                                                  input_segment_domain,
                                                  input_segment_logical_lengths));
        const NativeGkrInputSupport input_support = input_provider->support_complete
            ? input_provider->segment_support
            : masked_input_default_support(input_provider->logical_lengths);
        auto input_coefficients = std::make_shared<MaskedInputCoefficientOracle>(
            input_size, input_segment_domain, input_segment_logical_lengths,
            input_support, input_sources);
        std::vector<F> source_rhos;
        std::vector<ScalarOpening> source_masked_openings;
        std::vector<ScalarOpening> source_unmasked_openings;
        source_rhos.reserve(input_sources.size());
        source_masked_openings.reserve(input_sources.size());
        source_unmasked_openings.reserve(input_sources.size());
        F masked_aggregate_initial = F_ZERO;
        F unmasked_aggregate_initial = F_ZERO;
        for (std::size_t index = 0; index < input_sources.size(); ++index) {
            const F rho = session.challenge(masked_terminal_aggregate_rho_label(
                index, input_sources.size()));
            const MaskedGkrInputSource &source = input_sources[index];
            if (!masked_add_input_source_weights(source, rho,
                                                 *input_coefficients, reason))
                throw std::runtime_error(*reason);
            source_rhos.push_back(rho);
            source_masked_openings.push_back(source.masked_opening);
            source_unmasked_openings.push_back(source.unmasked_opening);
            masked_aggregate_initial += rho * source.masked_value;
            unmasked_aggregate_initial += rho * source.unmasked_opening.value;
        }
        ScalarOpening unmasked_aggregate_opening;
        ScalarOpening masked_aggregate_opening;
        if (!masked_make_linear_claim(
                session, basis, "gkr/input/aggregate/unmasked",
                source_unmasked_openings, source_rhos,
                unmasked_aggregate_initial, unmasked_aggregate_opening,
                reason) ||
            !masked_make_linear_claim(
                session, basis, "gkr/input/aggregate/masked",
                source_masked_openings, source_rhos,
                masked_aggregate_initial, masked_aggregate_opening, reason))
            throw std::runtime_error(*reason);

        NativeMaskedSumcheckResult input_result;
        if (!masked_public_product_prove(
                session, "gkr/input/aggregate", basis, fast_path,
                input_coefficients, input_provider,
                masked_aggregate_initial, unmasked_aggregate_initial,
                unmasked_aggregate_opening, masked_aggregate_opening,
                input_result, reason))
            throw std::runtime_error(*reason);
        if (input_result.terminal_openings.size() != 1 ||
            input_result.unmasked_terminal_openings.size() != 1 ||
            input_result.terminal_commitments.size() != 1 ||
            input_result.unmasked_terminal_commitments.size() != 1 ||
            !input_result.has_unmasked_claim_opening ||
            !input_result.has_claim_opening)
            throw std::runtime_error(
                "masked GKR canonical input aggregate terminal is incomplete");
        const F input_weight_evaluation = input_coefficients->terminal_value();
        session.absorb_fields("gkr/input/point", input_result.challenges);
        const ScalarOpening input_mask = zkgpt::native_pedersen::commit_fresh(
            basis, input_result.claim_opening.value -
                       input_result.unmasked_claim_opening.value);
        if (!masked_emit_opening(session, basis, "gkr/input/claim",
                                 input_result.claim_opening, reason) ||
            !masked_emit_opening(session, basis, "gkr/input/mask", input_mask,
                                 reason) ||
            !zkgpt::native_pedersen::prove_hidden_linear(
                session, basis, "gkr/input/bind",
                {input_result.claim_opening,
                 input_result.unmasked_claim_opening, input_mask},
                {F_ONE, -F_ONE, -F_ONE}, F_ZERO, nullptr, reason))
            throw std::runtime_error(reason != nullptr && !reason->empty()
                                         ? *reason
                                         : "masked GKR input binding failed");
        NativeGkrInputBinding final_binding;
        final_binding.point = input_result.challenges;
        final_binding.claim_commitment = input_result.claim_opening.commitment;
        final_binding.unmasked_claim_commitment =
            input_result.unmasked_claim_opening.commitment;
        final_binding.mask_commitment = input_mask.commitment;
        final_binding.unmasked_input_value_commitment =
            input_result.unmasked_terminal_openings[0].commitment;
        final_binding.masked_input_value_commitment =
            input_result.terminal_openings[0].commitment;
        final_binding.input_weight_evaluation = input_weight_evaluation;
        if (input_binding != nullptr) *input_binding = final_binding;
        if (prover_result != nullptr) {
            prover_result->input_binding = final_binding;
            prover_result->unmasked_input_claim_opening =
                input_result.unmasked_claim_opening;
            prover_result->has_unmasked_input_claim_opening = true;
            prover_result->unmasked_input_value_opening =
                input_result.unmasked_terminal_openings[0];
            prover_result->has_unmasked_input_value_opening = true;
        }
        if (!session.valid(reason) || !session.replay(reason))
            throw std::runtime_error(reason != nullptr ? *reason :
                                     "masked GKR transcript replay failed");
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        masked_set_reason(reason, error.what());
        return false;
    }
}

bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const zkgpt::NativeFieldVector &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::size_t start, std::size_t *end, std::string *reason,
    NativeGkrInputBinding *input_binding,
    const NativeGkrInputSupport *input_support) {
    try {
        if (!session.valid(reason) || !session.replay(reason) ||
            !masked_validate_circuit(circuit, public_output, reason))
            return false;
        if (start >= session.messages().size()) {
            masked_set_reason(reason, "masked GKR scope start is outside the session");
            return false;
        }
        std::size_t cursor = start;
        const NativeProofMessage *output_message = nullptr;
        if (!masked_expect_message(session, cursor, NativeProofMessageKind::Data,
                                   "gkr/public-output-digest-v2", &output_message, reason) ||
            !masked_decode_public_output(output_message, public_output, reason))
            return false;
        const std::size_t input_size = circuit.circuit.front().size;
        if (!masked_power_of_two(input_size)) {
            masked_set_reason(reason,
                              "masked GKR canonical input domain is not a power of two");
            return false;
        }
        std::size_t fconn_count = 0;
        for (std::size_t index = 1; index < circuit.circuit.size(); ++index)
            if (circuit.circuit[index].ty == layerType::FCONN) ++fconn_count;
        if (fconn_count != matmuls.size()) {
            masked_set_reason(reason,
                              "masked GKR FCONN descriptor count mismatch");
            return false;
        }
        for (const NativeGkrMatmulShape &shape : matmuls)
            if (!masked_validate_matmul_segment(shape, input_size, reason))
                return false;
        std::size_t input_segment_domain = 0;
        std::vector<std::size_t> input_segment_logical_lengths;
        if (!masked_input_segment_domain(matmuls, input_size,
                                         input_segment_domain,
                                         input_segment_logical_lengths,
                                         reason))
            return false;
        std::vector<std::vector<F>> r_u(circuit.size + 1);
        std::vector<std::vector<F>> r_v(circuit.size + 1);
        const layer &last = circuit.circuit.back();
        r_u[circuit.size].resize(last.bit_length);
        for (int bit = 0; bit < last.bit_length; ++bit)
            if (!masked_expect_challenge(session, cursor,
                                         "gkr/output/r/" + std::to_string(bit),
                                         r_u[circuit.size][bit], reason))
                return false;
        F previous_masked = transcript_mle_eval(public_output, r_u[circuit.size]);
        const PedersenBasis basis = masked_basis(session);
        const ScalarCommitment public_base{
            basis.domain, zkgpt::native_pedersen::commit(basis, F_ZERO, F_ZERO).point};
        const ScalarCommitment public_claim{
            basis.domain,
            zkgpt::native_pedersen::commit(basis, previous_masked, F_ZERO).point};
        if (!masked_consume_opening(session, basis, "gkr/output/public-base",
                                    cursor, &public_base, nullptr, reason) ||
            !masked_consume_opening(session, basis, "gkr/output/public-claim",
                                    cursor, &public_claim, nullptr, reason) ||
            !masked_consume_linear(
                session, basis, "gkr/output/public-affine", cursor,
                {public_claim, public_base}, {F_ONE, F_ZERO},
                previous_masked, reason))
            return false;
    ScalarCommitment previous_unmasked_commitment = public_claim;
    ScalarCommitment previous_masked_commitment = public_claim;
        ScalarCommitment previous_u1_unmasked_commitment;
        ScalarCommitment previous_u1_masked_commitment;
        F previous_u1_masked = F_ZERO;
        bool have_previous_u1 = false;
        bool previous_lane_active = true;
        F alpha = F_ONE, beta = F_ZERO;
        std::size_t next_matmul = matmuls.size();
        std::vector<MaskedGkrInputSource> input_sources;
        for (int layer_id = static_cast<int>(circuit.size) - 1;
             layer_id > 0; --layer_id) {
            const layer &current = circuit.circuit[layer_id];
            const std::string prefix = "gkr/layer/" + std::to_string(layer_id);
            const NativeProofMessage *layer_begin = nullptr;
            if (!masked_expect_message(session, cursor, NativeProofMessageKind::Data,
                                       prefix + "/begin", &layer_begin, reason) ||
                layer_begin->payload != "masked-layer-v1") {
                masked_set_reason(reason, "masked GKR layer boundary is malformed");
                return false;
            }
            F relu_rou = F_ONE;
            if (current.zero_start_id < current.size &&
                !masked_expect_challenge(session, cursor, prefix + "/relu",
                                          relu_rou, reason))
                return false;
            bool public_zero_branch_anchor = false;
            if (!previous_lane_active) {
                const std::string anchor_label = prefix + "/branch-anchor";
                if (layer_id > 1 && next_matmul > 0 &&
                    circuit.circuit[layer_id - 1].ty == layerType::FCONN) {
                    const NativeGkrMatmulShape &branch_shape =
                        matmuls[next_matmul - 1];
                    if (!masked_fconn_output_binding_shape(
                            circuit, static_cast<std::size_t>(layer_id),
                            branch_shape, reason))
                        return false;
                    const NativeProofMessage *identity_message = nullptr;
                    if (!masked_expect_message(
                            session, cursor, NativeProofMessageKind::Data,
                            anchor_label + "/identity", &identity_message,
                            reason) ||
                        identity_message->payload !=
                            masked_fconn_output_binding_identity(
                                static_cast<std::size_t>(layer_id), current,
                                branch_shape)) {
                        masked_set_reason(
                            reason,
                            "masked GKR FCONN residual branch identity mismatch");
                        return false;
                    }
                    std::vector<F> branch_point(current.bit_length);
                    for (int bit = 0; bit < current.bit_length; ++bit)
                        if (!masked_expect_challenge(
                                session, cursor,
                                anchor_label + "/r/" + std::to_string(bit),
                                branch_point[bit], reason))
                            return false;
                    r_u[layer_id + 1] = branch_point;
                    r_v[layer_id + 1].assign(branch_point.size(), F_ZERO);
                    previous_masked = F_ZERO;
                    previous_unmasked_commitment = public_base;
                    previous_masked_commitment = public_base;
                    public_zero_branch_anchor = true;
                } else {
                    zkgpt::NativeVector<u64> canonical_copy_coordinates;
                    if (!masked_canonical_input_copy_coordinates(
                            current, input_size, input_segment_domain,
                            input_segment_logical_lengths,
                            canonical_copy_coordinates, reason)) {
                        masked_set_reason(
                            reason,
                            "masked GKR inactive lane has no canonical source anchor");
                        return false;
                    }
                    const NativeProofMessage *identity_message = nullptr;
                    if (!masked_expect_message(
                            session, cursor, NativeProofMessageKind::Data,
                            anchor_label + "/identity", &identity_message,
                            reason) ||
                        identity_message->payload !=
                            masked_input_copy_branch_identity(
                                static_cast<std::size_t>(layer_id), current,
                                canonical_copy_coordinates, input_size,
                                input_segment_domain,
                                input_segment_logical_lengths)) {
                        masked_set_reason(
                            reason,
                            "masked GKR canonical input-copy branch identity mismatch");
                        return false;
                    }
                    std::vector<F> branch_point(current.bit_length);
                    for (int bit = 0; bit < current.bit_length; ++bit)
                        if (!masked_expect_challenge(
                                session, cursor,
                                anchor_label + "/r/" + std::to_string(bit),
                                branch_point[bit], reason))
                            return false;
                    const NativeProofMessage *claim_message = nullptr;
                    if (!masked_expect_message(
                            session, cursor, NativeProofMessageKind::Field,
                            anchor_label + "/masked-claim", &claim_message,
                            reason) ||
                        !decode_transcript_field(claim_message->payload,
                                                 previous_masked)) {
                        masked_set_reason(
                            reason,
                            "masked GKR canonical input-copy branch claim is malformed");
                        return false;
                    }
                    const std::string phase1_label = prefix + "/phase1";
                    if (!masked_lookup_opening_commitment(
                            session, basis,
                            phase1_label + "/initial/opening",
                            previous_unmasked_commitment, reason) ||
                        !masked_lookup_opening_commitment(
                            session, basis,
                            phase1_label + "/initial/masked",
                            previous_masked_commitment, reason))
                        return false;
                    MaskedGkrInputSource source;
                    source.kind =
                        MaskedGkrInputSource::Kind::LayerCanonicalCopy;
                    source.layer_id = static_cast<std::size_t>(layer_id);
                    source.layer_type =
                        static_cast<std::uint32_t>(current.ty);
                    source.canonical_copy_coordinates =
                        std::move(canonical_copy_coordinates);
                    source.point = branch_point;
                    source.masked_commitment = previous_masked_commitment;
                    source.unmasked_commitment =
                        previous_unmasked_commitment;
                    source.masked_value = previous_masked;
                    input_sources.push_back(std::move(source));
                    r_u[layer_id + 1] = branch_point;
                    r_v[layer_id + 1].assign(branch_point.size(), F_ZERO);
                }
                previous_lane_active = true;
                alpha = F_ONE;
                beta = F_ZERO;
            }
            if (current.ty == layerType::FCONN) {
                if (!previous_lane_active) {
                    masked_set_reason(
                        reason,
                        "masked GKR FCONN consumed an inactive previous lane");
                    return false;
                }
                if (next_matmul == 0) {
                    masked_set_reason(reason, "masked GKR FCONN descriptor underflow");
                    return false;
                }
                const NativeGkrMatmulShape &shape = matmuls[--next_matmul];
                if (!masked_fconn_shape_matches(current, shape, reason)) return false;
                const std::size_t n = ceilPow2BitLength(
                    static_cast<u64>(shape.rows));
                const std::size_t m = ceilPow2BitLength(
                    static_cast<u64>(shape.inner));
                const std::size_t k = ceilPow2BitLength(
                    static_cast<u64>(shape.cols));
                if (r_u[layer_id + 1].size() < n + k) {
                    masked_set_reason(
                        reason,
                        "masked GKR FCONN predecessor point has the wrong dimension");
                    return false;
                }
                r_u[layer_id].assign(m + n, F_ZERO);
                r_v[layer_id].assign(m + k, F_ZERO);
                for (std::size_t bit = m; bit < m + n; ++bit)
                    r_u[layer_id][bit] = r_u[layer_id + 1][bit - m + k];
                for (std::size_t bit = m; bit < m + k; ++bit)
                    r_v[layer_id][bit] = r_u[layer_id + 1][bit - m];
                for (std::size_t bit = 0; bit < m; ++bit) {
                    if (!masked_expect_challenge(session, cursor,
                                                 prefix + "/matmul_point/r_u/" +
                                                     std::to_string(bit),
                                                 r_u[layer_id][bit], reason))
                        return false;
                    r_v[layer_id][bit] = r_u[layer_id][bit];
                }
                const F initial = have_previous_u1 ? previous_u1_masked : previous_masked;
                const std::string matmul_label = prefix + "/matmul";
                NativeMaskedSumcheckResult product_result;
                NativeMaskedTerminalSpec product_spec;
                product_spec.kind = NativeMaskedTerminalKind::Product2;
                const ScalarCommitment &product_initial =
                    have_previous_u1 ? previous_u1_unmasked_commitment
                                     : previous_unmasked_commitment;
                if (!masked_consume_scope(session, matmul_label, 2, m, initial,
                                          product_initial,
                                          have_previous_u1 ? previous_u1_masked_commitment
                                                           : previous_masked_commitment,
                                          product_spec, cursor,
                                          product_result, reason) ||
                    product_result.terminal_values.size() != 2) {
                    masked_set_reason(reason, "masked FCONN Sumcheck failed at " +
                                      matmul_label + ": " +
                                      (reason != nullptr ? *reason : std::string()));
                    return false;
                }
                if (product_result.terminal_commitments.size() != 2 ||
                    product_result.unmasked_terminal_commitments.size() != 2) {
                    masked_set_reason(
                        reason,
                        "masked GKR FCONN terminal source commitments are incomplete");
                    return false;
                }
                MaskedGkrInputSource left_source;
                left_source.kind = MaskedGkrInputSource::Kind::FconnLeft;
                left_source.layer_id = static_cast<std::size_t>(layer_id);
                left_source.layer_type = static_cast<std::uint32_t>(current.ty);
                left_source.descriptor_index = next_matmul;
                left_source.shape = shape;
                left_source.point = product_result.challenges;
                left_source.outer_point.assign(
                    r_u[layer_id].begin() + static_cast<std::ptrdiff_t>(m),
                    r_u[layer_id].end());
                left_source.masked_commitment =
                    product_result.terminal_commitments[0];
                left_source.unmasked_commitment =
                    product_result.unmasked_terminal_commitments[0];
                left_source.masked_value = product_result.terminal_values[0];
                input_sources.push_back(std::move(left_source));

                MaskedGkrInputSource right_source;
                right_source.kind = MaskedGkrInputSource::Kind::FconnRight;
                right_source.layer_id = static_cast<std::size_t>(layer_id);
                right_source.layer_type = static_cast<std::uint32_t>(current.ty);
                right_source.descriptor_index = next_matmul;
                right_source.shape = shape;
                right_source.point = product_result.challenges;
                right_source.outer_point.assign(
                    r_v[layer_id].begin() + static_cast<std::ptrdiff_t>(m),
                    r_v[layer_id].end());
                right_source.masked_commitment =
                    product_result.terminal_commitments[1];
                right_source.unmasked_commitment =
                    product_result.unmasked_terminal_commitments[1];
                right_source.masked_value = product_result.terminal_values[1];
                input_sources.push_back(std::move(right_source));
                previous_u1_masked = F_ZERO;
                have_previous_u1 = false;
                if (current.bit_length_u[1] >= 0 &&
                    !masked_expect_challenge(session, cursor, prefix + "/alpha",
                                              alpha, reason))
                    return false;
                if (current.bit_length_u[1] < 0) alpha.clear();
                if ((current.bit_length_v[1] >= 0 || current.ty == layerType::FFT) &&
                    !masked_expect_challenge(session, cursor, prefix + "/beta",
                                              beta, reason))
                    return false;
                if (current.bit_length_v[1] < 0 && current.ty != layerType::FFT)
                    beta.clear();
                previous_masked = F_ZERO;
                previous_unmasked_commitment = ScalarCommitment{};
                previous_masked_commitment = ScalarCommitment{};
                previous_u1_unmasked_commitment = ScalarCommitment{};
                previous_u1_masked_commitment = ScalarCommitment{};
                previous_u1_masked = F_ZERO;
                have_previous_u1 = false;
                previous_lane_active = false;
                continue;
            }
            const std::string phase1_label = prefix + "/phase1";
            NativeMaskedSumcheckResult phase1_result;
            NativeMaskedTerminalSpec phase1_spec;
            phase1_spec.public_initial_claim = public_zero_branch_anchor;
            if (current.need_phase2) {
                phase1_spec.kind = NativeMaskedTerminalKind::Claims;
                phase1_spec.terminal_linear_coefficients = {F_ONE, F_ONE};
                phase1_spec.defer_terminal_relation = true;
            } else {
                phase1_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedV;
                phase1_spec.fixed_values = {F_ZERO, F_ZERO};
                r_u[layer_id].resize(current.max_bl_u);
                for (int bit = 0; bit < current.max_bl_u; ++bit)
                    if (!masked_lookup_challenge(
                            session, phase1_label + "/r/" + std::to_string(bit),
                            r_u[layer_id][bit], reason))
                        return false;
                std::vector<F> r0_for_spec = r_u[layer_id + 1];
                std::vector<F> r1_for_spec = r_v[layer_id + 1];
                if (r0_for_spec.size() < static_cast<std::size_t>(current.bit_length))
                    r0_for_spec.resize(current.bit_length, F_ZERO);
                if (r1_for_spec.size() < static_cast<std::size_t>(current.bit_length))
                    r1_for_spec.resize(current.bit_length, F_ZERO);
                phase1_spec.gkr_coefficients = masked_gkr_coefficients(
                    current, r0_for_spec, r1_for_spec, r_u[layer_id], alpha,
                    beta, relu_rou);
                phase1_spec.defer_terminal_relation =
                    phase1_spec.gkr_coefficients.size() == 5 &&
                    phase1_spec.gkr_coefficients[3].isZero() &&
                    phase1_spec.gkr_coefficients[4].isZero();
                const ScalarCommitment zero_commitment{
                    basis.domain,
                    zkgpt::native_pedersen::commit(basis, F_ZERO, F_ZERO).point};
                phase1_spec.fixed_commitments = {zero_commitment, zero_commitment};
                phase1_spec.expected_unmasked_fixed_commitments = {
                    zero_commitment, zero_commitment};
            }
            if (!masked_consume_scope(
                    session, phase1_label, 2, current.max_bl_u, previous_masked,
                    previous_unmasked_commitment, previous_masked_commitment,
                    phase1_spec, cursor,
                    phase1_result, reason) ||
                phase1_result.terminal_values.size() != 2) {
                masked_set_reason(reason, "masked phase1 Sumcheck failed at " +
                                  phase1_label + ": " +
                                  (reason != nullptr ? *reason : std::string()));
                return false;
            }
            r_u[layer_id] = phase1_result.challenges;
            std::vector<F> r0 = r_u[layer_id + 1];
            std::vector<F> r1 = r_v[layer_id + 1];
            if (r0.size() < static_cast<std::size_t>(current.bit_length))
                r0.resize(current.bit_length, F_ZERO);
            if (r1.size() < static_cast<std::size_t>(current.bit_length))
                r1.resize(current.bit_length, F_ZERO);
            std::vector<F> coefficients = masked_gkr_coefficients(
                current, r0, r1, r_u[layer_id], alpha, beta, relu_rou);
            std::vector<F> masked_v{F_ZERO, F_ZERO};
            NativeMaskedSumcheckResult phase2_result;
            if (current.need_phase2) {
                r_v[layer_id].resize(current.max_bl_v);
                const std::string phase2_label = prefix + "/phase2";
                for (int bit = 0; bit < current.max_bl_v; ++bit)
                    if (!masked_lookup_challenge(
                            session, phase2_label + "/r/" + std::to_string(bit),
                            r_v[layer_id][bit], reason))
                        return false;
                const std::vector<F> phase2_coefficients = masked_gkr_coefficients(
                    current, r0, r1, r_u[layer_id], alpha, beta, relu_rou,
                    &r_v[layer_id]);
                NativeMaskedTerminalSpec fixed_u_spec;
                fixed_u_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedU;
                fixed_u_spec.gkr_coefficients = phase2_coefficients;
                fixed_u_spec.fixed_values = phase1_result.terminal_values;
                if (phase1_result.terminal_values.size() != 2 ||
                    phase1_result.terminal_commitments.size() != 2) {
                    masked_set_reason(reason,
                                      "masked phase1 terminal binding has invalid arity");
                    return false;
                }
                fixed_u_spec.defer_terminal_relation =
                    fixed_u_spec.gkr_coefficients.size() == 5 &&
                    (fixed_u_spec.gkr_coefficients[0] *
                         fixed_u_spec.fixed_values[0] +
                     fixed_u_spec.gkr_coefficients[2] *
                         fixed_u_spec.fixed_values[1])
                            .isZero() &&
                    (fixed_u_spec.gkr_coefficients[1] *
                         fixed_u_spec.fixed_values[1])
                        .isZero();
                fixed_u_spec.fixed_commitments = phase1_result.terminal_commitments;
                fixed_u_spec.expected_unmasked_fixed_commitments =
                    phase1_result.unmasked_terminal_commitments;
                if (!masked_consume_scope(
                        session, phase2_label, 2, current.max_bl_v,
                        phase1_result.final_claim,
                        phase1_result.unmasked_claim_commitment,
                        phase1_result.claim_commitment,
                        fixed_u_spec, cursor, phase2_result, reason) ||
                    phase2_result.terminal_values.size() != 2)
                {
                    masked_set_reason(reason, "masked phase2 Sumcheck failed at " +
                                      phase2_label + ": " +
                                      (reason != nullptr ? *reason : std::string()));
                    return false;
                }
                masked_v = phase2_result.terminal_values;
            } else {
            }
            if (current.bit_length_u[0] >= 0) {
                if (current.size_u[0] == 0 ||
                    current.ori_id_u.size() != current.size_u[0] ||
                    phase1_result.terminal_values.size() != 2 ||
                    phase1_result.terminal_commitments.size() != 2 ||
                    phase1_result.unmasked_terminal_commitments.size() != 2 ||
                    phase1_result.challenges.size() <
                        static_cast<std::size_t>(current.bit_length_u[0])) {
                    masked_set_reason(
                        reason,
                        "masked GKR phase1 direct-input source binding is incomplete");
                    return false;
                }
                MaskedGkrInputSource source;
                source.kind = MaskedGkrInputSource::Kind::LayerU0;
                source.layer_id = static_cast<std::size_t>(layer_id);
                source.layer_type = static_cast<std::uint32_t>(current.ty);
                source.input_coordinates = &current.ori_id_u;
                source.point.assign(
                    phase1_result.challenges.begin(),
                    phase1_result.challenges.begin() +
                        static_cast<std::ptrdiff_t>(current.bit_length_u[0]));
                source.masked_commitment = phase1_result.terminal_commitments[0];
                source.unmasked_commitment =
                    phase1_result.unmasked_terminal_commitments[0];
                source.masked_value = phase1_result.terminal_values[0];
                input_sources.push_back(std::move(source));
            }
            if (current.bit_length_v[0] >= 0) {
                if (!current.need_phase2 || current.size_v[0] == 0 ||
                    current.ori_id_v.size() != current.size_v[0] ||
                    phase2_result.terminal_values.size() != 2 ||
                    phase2_result.terminal_commitments.size() != 2 ||
                    phase2_result.unmasked_terminal_commitments.size() != 2 ||
                    phase2_result.challenges.size() <
                        static_cast<std::size_t>(current.bit_length_v[0])) {
                    masked_set_reason(
                        reason,
                        "masked GKR phase2 direct-input source binding is incomplete");
                    return false;
                }
                MaskedGkrInputSource source;
                source.kind = MaskedGkrInputSource::Kind::LayerV0;
                source.layer_id = static_cast<std::size_t>(layer_id);
                source.layer_type = static_cast<std::uint32_t>(current.ty);
                source.input_coordinates = &current.ori_id_v;
                source.point.assign(
                    phase2_result.challenges.begin(),
                    phase2_result.challenges.begin() +
                        static_cast<std::ptrdiff_t>(current.bit_length_v[0]));
                source.masked_commitment = phase2_result.terminal_commitments[0];
                source.unmasked_commitment =
                    phase2_result.unmasked_terminal_commitments[0];
                source.masked_value = phase2_result.terminal_values[0];
                input_sources.push_back(std::move(source));
            }
            previous_u1_masked = phase1_result.terminal_values[1];
            previous_u1_unmasked_commitment =
                phase1_result.unmasked_terminal_commitments[1];
            previous_u1_masked_commitment = phase1_result.terminal_commitments[1];
            have_previous_u1 = true;
            if (current.bit_length_u[1] >= 0 &&
                !masked_expect_challenge(session, cursor, prefix + "/alpha",
                                          alpha, reason))
                return false;
            if (current.bit_length_u[1] < 0) alpha.clear();
            if ((current.bit_length_v[1] >= 0 || current.ty == layerType::FFT) &&
                !masked_expect_challenge(session, cursor, prefix + "/beta",
                                          beta, reason))
                return false;
            if (current.bit_length_v[1] < 0 && current.ty != layerType::FFT)
                beta.clear();
            previous_masked = alpha * previous_u1_masked +
                              beta * (current.need_phase2 ? masked_v[1] : F_ZERO);
            const ScalarCommitment phase2_unmasked_commitment = current.need_phase2
                ? phase2_result.unmasked_terminal_commitments[1]
                : phase1_result.unmasked_fixed_commitments[1];
            const ScalarCommitment phase2_masked_commitment = current.need_phase2
                ? phase2_result.terminal_commitments[1]
                : phase1_result.fixed_commitments[1];
            ScalarCommitment next_unmasked_commitment;
            ScalarCommitment next_masked_commitment;
            if (!masked_consume_linear_claim(
                    session, basis, prefix + "/next-claim/unmasked", cursor,
                    {previous_u1_unmasked_commitment,
                     phase2_unmasked_commitment},
                    {alpha, beta}, F_ZERO, next_unmasked_commitment, reason) ||
                !masked_consume_linear_claim(
                        session, basis, prefix + "/next-claim/masked", cursor,
                        {previous_u1_masked_commitment,
                         phase2_masked_commitment},
                        {alpha, beta}, F_ZERO, next_masked_commitment, reason))
                return false;
            previous_unmasked_commitment = next_unmasked_commitment;
            previous_masked_commitment = next_masked_commitment;
        }
        if (next_matmul != 0) {
            masked_set_reason(reason, "masked GKR transcript has unconsumed FCONN descriptors");
            return false;
        }
        if (input_sources.empty()) {
            masked_set_reason(
                reason,
                "masked GKR has no canonical input terminal claims to reduce");
            return false;
        }
        const NativeProofMessage *source_ledger_message = nullptr;
        if (!masked_expect_message(
                session, cursor, NativeProofMessageKind::Data,
                "gkr/input/source-ledger", &source_ledger_message, reason) ||
            source_ledger_message->payload !=
                masked_input_source_ledger(input_sources, input_size,
                                           input_segment_domain,
                                           input_segment_logical_lengths)) {
            masked_set_reason(reason,
                              "masked GKR canonical input source ledger mismatch");
            return false;
        }
        const NativeGkrInputSupport trusted_input_support = input_support == nullptr
            ? masked_input_default_support({{
                  static_cast<std::uint64_t>(input_segment_logical_lengths[0]),
                  static_cast<std::uint64_t>(input_segment_logical_lengths[1]),
                  static_cast<std::uint64_t>(input_segment_logical_lengths[2]),
                  static_cast<std::uint64_t>(input_segment_logical_lengths[3])}})
            : *input_support;
        auto input_coefficients = std::make_shared<MaskedInputCoefficientOracle>(
            input_size, input_segment_domain, input_segment_logical_lengths,
            trusted_input_support, input_sources);
        std::vector<F> source_rhos;
        std::vector<ScalarCommitment> source_masked_commitments;
        std::vector<ScalarCommitment> source_unmasked_commitments;
        source_rhos.reserve(input_sources.size());
        source_masked_commitments.reserve(input_sources.size());
        source_unmasked_commitments.reserve(input_sources.size());
        F masked_aggregate_initial = F_ZERO;
        for (std::size_t index = 0; index < input_sources.size(); ++index) {
            F rho;
            if (!masked_expect_challenge(
                    session, cursor, masked_terminal_aggregate_rho_label(
                        index, input_sources.size()),
                    rho, reason))
                return false;
            const MaskedGkrInputSource &source = input_sources[index];
            if (!masked_add_input_source_weights(
                    source, rho, *input_coefficients, reason))
                return false;
            source_rhos.push_back(rho);
            source_masked_commitments.push_back(source.masked_commitment);
            source_unmasked_commitments.push_back(source.unmasked_commitment);
            masked_aggregate_initial += rho * source.masked_value;
        }
        ScalarCommitment unmasked_aggregate_commitment;
        ScalarCommitment masked_aggregate_commitment;
        if (!masked_consume_linear_claim(
                session, basis, "gkr/input/aggregate/unmasked", cursor,
                source_unmasked_commitments, source_rhos, F_ZERO,
                unmasked_aggregate_commitment, reason) ||
            !masked_consume_linear_claim(
                session, basis, "gkr/input/aggregate/masked", cursor,
                source_masked_commitments, source_rhos, F_ZERO,
                masked_aggregate_commitment, reason))
            return false;

        std::size_t input_rounds = 0;
        for (std::size_t domain = input_size; domain > 1; domain >>= 1)
            ++input_rounds;
        NativeMaskedTerminalSpec input_spec;
        input_spec.kind = NativeMaskedTerminalKind::Product2Public;
        const std::shared_ptr<const NativeMaskedPublicProvider> input_public_provider =
            masked_input_public_provider(input_coefficients);
        input_spec.public_provider = input_public_provider;
        NativeMaskedSumcheckResult input_result;
        if (!masked_consume_scope(
                session, "gkr/input/aggregate", 2, input_rounds,
                masked_aggregate_initial, unmasked_aggregate_commitment,
                masked_aggregate_commitment, input_spec, cursor,
                input_result, reason) ||
            input_result.terminal_values.size() != 1 ||
            input_result.terminal_commitments.size() != 1 ||
            input_result.unmasked_terminal_commitments.size() != 1) {
            masked_set_reason(reason,
                              "masked GKR input Product2Public aggregation failed");
            return false;
        }
        const F input_weight_evaluation = input_public_provider->evaluate_mle(
            input_result.challenges);
        std::vector<F> encoded_input_point;
        const NativeProofMessage *input_point_message = nullptr;
        if (!masked_expect_message(session, cursor,
                                   NativeProofMessageKind::FieldVector,
                                   "gkr/input/point", &input_point_message,
                                   reason) ||
            !decode_transcript_fields(input_point_message->payload,
                                      encoded_input_point) ||
            encoded_input_point != input_result.challenges) {
            masked_set_reason(reason, "masked GKR final input point binding failed");
            return false;
        }
        ScalarCommitment input_claim_commitment;
        ScalarCommitment input_mask_commitment;
        if (!masked_consume_opening(session, basis, "gkr/input/claim", cursor,
                                    &input_result.claim_commitment,
                                    &input_claim_commitment, reason) ||
            !masked_consume_opening(session, basis, "gkr/input/mask", cursor,
                                    nullptr, &input_mask_commitment, reason) ||
            !masked_consume_linear(
                session, basis, "gkr/input/bind", cursor,
                {input_claim_commitment, input_result.unmasked_claim_commitment,
                 input_mask_commitment},
                {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
        if (input_binding != nullptr) {
            input_binding->point = input_result.challenges;
            input_binding->claim_commitment = input_claim_commitment;
            input_binding->unmasked_claim_commitment =
                input_result.unmasked_claim_commitment;
            input_binding->mask_commitment = input_mask_commitment;
            input_binding->unmasked_input_value_commitment =
                input_result.unmasked_terminal_commitments[0];
            input_binding->masked_input_value_commitment =
                input_result.terminal_commitments[0];
            input_binding->input_weight_evaluation = input_weight_evaluation;
        }
        for (std::size_t index = cursor; index < session.messages().size(); ++index) {
            const std::string &label = session.messages()[index].label;
            if (label.compare(0, sizeof("gkr/") - 1, "gkr/") == 0 ||
                label.compare(0, sizeof("challenge/gkr/") - 1,
                              "challenge/gkr/") == 0) {
                masked_set_reason(reason, "masked GKR transcript has trailing GKR messages");
                return false;
            }
        }
        if (end != nullptr) *end = cursor;
        return true;
    } catch (const std::exception &error) {
        masked_set_reason(reason, error.what());
        return false;
    }
}

bool verify_masked_gkr_transcript(
    const NativeProofSession &session, const layeredCircuit &circuit,
    const zkgpt::NativeFieldVector &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls, std::string *reason,
    NativeGkrInputBinding *input_binding) {
    std::size_t end = 0;
    if (!verify_masked_gkr_transcript(session, circuit, public_output, matmuls,
                                      0, &end, reason, input_binding))
        return false;
    if (end != session.messages().size()) {
        masked_set_reason(reason, "masked GKR transcript has an unexpected suffix");
        return false;
    }
    return true;
}
