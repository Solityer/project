//
// Created by 69029 on 3/9/2021.
//
#include "verifier.hpp"
#include "global_var.hpp"
#include "native_protocol.hpp"
#include <utils.hpp>

// Formal FCONN layers carry an adapter-local descriptor identity.  The
// helper is defined with the other masked-GKR utilities below and is also
// used by the legacy transcript replay path above it.
static bool masked_fconn_shape_matches(const layer &current,
                                       const NativeGkrMatmulShape &shape,
                                       std::string *reason);
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
    for (const NativeProofMessage &message : session.messages()) {
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

F transcript_mle_eval(const std::vector<F> &values, const std::vector<F> &point) {
    if (point.empty()) {
        if (values.size() != 1) throw std::invalid_argument("GKR scalar output domain is invalid");
        return values[0];
    }
    if (point.size() >= sizeof(std::size_t) * 8)
        throw std::invalid_argument("GKR output point dimension is invalid");
    const std::size_t domain = std::size_t(1) << point.size();
    if (values.size() > domain) throw std::invalid_argument("GKR output exceeds MLE domain");
    std::vector<F> folded(domain, F_ZERO);
    std::copy(values.begin(), values.end(), folded.begin());
    std::size_t active = domain;
    for (const F &coordinate : point) {
        const std::size_t half = active >> 1;
        for (std::size_t index = 0; index < half; ++index)
            folded[index] = folded[index * 2] * (F_ONE - coordinate) +
                            folded[index * 2 + 1] * coordinate;
        active = half;
    }
    return folded[0];
}

void transcript_beta_phase1(const layer &current, std::size_t depth,
                            const std::vector<F> &r0,
                            const std::vector<F> &r1,
                            const std::vector<F> &ru,
                            const F &alpha, const F &beta,
                            const F &relu_rou,
                            std::vector<F> &beta_g,
                            std::vector<F> &beta_u) {
    beta_g.resize(std::size_t(1) << current.bit_length);
    initBetaTable(beta_g, current.bit_length, r0.begin(), r1.begin(),
                  alpha, beta);
    if (current.zero_start_id < current.size)
        for (u32 gate = current.zero_start_id;
             gate < (std::size_t(1) << current.bit_length); ++gate)
            beta_g[gate] *= relu_rou;
    beta_u.resize(std::size_t(1) << current.max_bl_u);
    initBetaTable(beta_u, current.max_bl_u, ru.begin(), F_ONE);
    (void)depth;
}

F transcript_predicate_phase1(const layer &current,
                              const std::vector<F> &beta_g,
                              const std::vector<F> &beta_u,
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

void transcript_predicate_phase2(const layer &current,
                                 const std::vector<F> &beta_g,
                                 const std::vector<F> &beta_u,
                                 const std::vector<F> &beta_v,
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

} // namespace



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
void pred1_worker(const vector<uniGate> &beg, vector<F>& uni_value,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
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
void pred2_worker(const vector<binGate> &beg, vector<F>& bin_value,vector<F>& beta_g, vector<F>& beta_u,int*&L,int*&R) //F*& uni_value, layer &cur_layer,
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

F verifier::challenge(const std::string &label) {
    if (session_ != nullptr) return session_->challenge(label);
    F value;
    value.setByCSPRNG();
    return value;
}

void verifier::absorb_field(const std::string &label, const F &value) {
    if (session_ != nullptr) session_->absorb_field(label, value);
}

void verifier::absorb_fields(const std::string &label, const std::vector<F> &values) {
    if (session_ != nullptr) session_->absorb_fields(label, values);
}

bool verifier::prove(int commit_thread)
{
    (void)commit_thread;
    cout<<"Circuit Initiation finished"<<endl;
    cout<<"Proving Service started"<<endl;
    const bool gkr_ok = verifyGKR();
    const bool lasso_ok = gkr_ok && verifyLasso();
    const bool opening_ok = lasso_ok && openCommit();
    cout<<"Our results:"<<endl;
    cout<<"Matrix multiplication Prover time: "<<matrix_time<<"s"<<endl;
    cout<<"Total Prover time: "<<prover_time + range_prover_time<<"s"<<endl;
    cout<<"Verifier time: "<<verifier_time<<"s"<<endl;
    cout<<"Proof size: "<<(double)(p->proof_size)/1024.0<<"KB"<<endl;
    return gkr_ok && lasso_ok && opening_ok;
}

bool verifier::prove_with_session(NativeProofSession &session, int commit_thread) {
    if (!session.valid()) return false;
    const std::size_t checkpoint = session.checkpoint();
    NativeProofSession *previous = session_;
    session_ = &session;
    bool result = false;
    try {
        result = prove(commit_thread);
    } catch (...) {
        session_ = previous;
        session.rollback(checkpoint);
        throw;
    }
    session_ = previous;
    if (!result || !session.replay()) {
        session.rollback(checkpoint);
        return false;
    }
    return true;
}

bool verifier::prove_gkr_with_session(NativeProofSession &session, int commit_thread) {
    (void)commit_thread;
    if (!session.valid()) return false;
    const std::size_t checkpoint = session.checkpoint();
    NativeProofSession *previous = session_;
    session_ = &session;
    bool result = false;
    try {
        // verifyGKR is the native zkGPT GKR/MatMul prover-verifier loop.  It
        // consumes the prover's layered values only while producing the
        // transcript; the returned session is the public proof boundary.
        result = verifyGKR();
    } catch (...) {
        session_ = previous;
        session.rollback(checkpoint);
        throw;
    }
    session_ = previous;
    if (!result || !session.replay()) {
        session.rollback(checkpoint);
        return false;
    }
    return true;
}

Fr * init_book_keeping(int m,int n,vector<Fr> &vec,int offset, vector<Fr> & ra)
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



pair<Fr,Fr> sum_check_product(Fr* f,Fr* g,int m,Fr* r,Fr ans)
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

// Session-bound variant of the native dedicated matrix-product Sumcheck.  It
// keeps the original polynomial/folding arithmetic, but records each
// prover polynomial before deriving the next challenge instead of consuming
// verifier-side CSPRNG state.  The independent verifier can replay these
// degree-two messages without accessing the product witness.
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
                           const std::vector<F> &public_output,
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
                const int n = ceilPow2BitLength(static_cast<u32>(shape.rows));
                const int m = ceilPow2BitLength(static_cast<u32>(shape.inner));
                const int k = ceilPow2BitLength(static_cast<u32>(shape.cols));
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
            std::vector<F> beta_g, beta_u, uni, bin;
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
                std::vector<F> beta_v;
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

    // Bind the public lookup/constraint shape before any Lasso challenge is
    // derived.  This is deliberately only a structural context binding;
    // the native LC_f/M_f/input-consistency argument remains unsupported and
    // therefore still keeps the formal masked_full adapter fail-closed.
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
    std::size_t found = 0;
    bool seen = false;
    for (std::size_t index = 0; index < session.messages().size(); ++index) {
        const NativeProofMessage &message = session.messages()[index];
        if (message.label != "challenge/" + label) continue;
        if (seen) {
            masked_set_reason(reason, "masked GKR challenge label is duplicated");
            return false;
        }
        seen = true;
        found = index;
    }
    if (!seen || found >= session.messages().size() ||
        session.messages()[found].kind != NativeProofMessageKind::Challenge ||
        !decode_transcript_field(session.messages()[found].payload, value)) {
        masked_set_reason(reason, "masked GKR challenge is missing or malformed");
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
    effective.public_initial_claim = false;
    effective.initial_commitment = unmasked_initial;
    effective.has_masked_initial_binding = true;
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

bool masked_decode_public_output(const NativeProofMessage *message,
                                 const std::vector<F> &expected,
                                 std::string *reason) {
    std::vector<F> decoded;
    if (message == nullptr ||
        message->kind != NativeProofMessageKind::FieldVector ||
        !decode_transcript_fields(message->payload, decoded) || decoded != expected) {
        masked_set_reason(reason, "masked GKR public output binding failed");
        return false;
    }
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

std::vector<F> masked_fold_outer(const std::vector<F> &values,
                                 std::size_t inner_bits, std::size_t outer_bits,
                                 const std::vector<F> &point) {
    const std::size_t total_bits = inner_bits + outer_bits;
    const std::size_t full_domain = std::size_t(1) << total_bits;
    if (values.size() != full_domain || point.size() != total_bits)
        throw std::invalid_argument("masked FCONN witness MLE dimensions are invalid");
    std::vector<F> current = values;
    std::size_t active = full_domain;
    for (std::size_t round = 0; round < outer_bits; ++round) {
        const F &challenge = point[total_bits - 1 - round];
        const std::size_t half = active >> 1;
        std::vector<F> next(half);
        for (std::size_t index = 0; index < half; ++index)
            next[index] = current[index] * (F_ONE - challenge) +
                          current[index + half] * challenge;
        current.swap(next);
        active = half;
    }
    return current;
}

bool masked_matmul_witness_vectors(const NativeGkrMatmulWitness &witness,
                                   std::size_t &rows_domain,
                                   std::size_t &inner_domain,
                                   std::size_t &cols_domain,
                                   std::vector<F> &left,
                                   std::vector<F> &right,
                                   std::string *reason) {
    const auto &shape = witness.shape;
    if (shape.rows == 0 || shape.inner == 0 || shape.cols == 0 ||
        shape.rows > std::numeric_limits<std::uint32_t>::max() ||
        shape.inner > std::numeric_limits<std::uint32_t>::max() ||
        shape.cols > std::numeric_limits<std::uint32_t>::max()) {
        masked_set_reason(reason, "masked FCONN descriptor dimensions are invalid");
        return false;
    }
    const std::size_t row_bits = ceilPow2BitLength(
        static_cast<std::uint32_t>(shape.rows));
    const std::size_t inner_bits = ceilPow2BitLength(
        static_cast<std::uint32_t>(shape.inner));
    const std::size_t col_bits = ceilPow2BitLength(
        static_cast<std::uint32_t>(shape.cols));
    if (!masked_power_domain(row_bits, rows_domain, reason) ||
        !masked_power_domain(inner_bits, inner_domain, reason) ||
        !masked_power_domain(col_bits, cols_domain, reason))
        return false;
    const std::size_t canonical_left_size = shape.rows * shape.inner;
    const std::size_t padded_right_size = inner_domain * cols_domain;
    if (witness.left.size() != canonical_left_size ||
        witness.right.size() != padded_right_size) {
        masked_set_reason(reason, "masked FCONN witness matrix dimensions are invalid");
        return false;
    }
    left.assign(rows_domain * inner_domain, F_ZERO);
    right.assign(inner_domain * cols_domain, F_ZERO);
    // Keep the inner coordinate as the low MLE bits.  The LHS is canonical
    // row-major, so each row receives its full padded-inner stride.  The RHS
    // already uses native column-major blocks, and is copied block-for-block;
    // this preserves the gap between a non-power-of-two inner dimension and
    // the next column.
    for (std::size_t row = 0; row < shape.rows; ++row)
        for (std::size_t term = 0; term < shape.inner; ++term)
            left[row * inner_domain + term] =
                witness.left[row * shape.inner + term];
    for (std::size_t col = 0; col < shape.cols; ++col)
        for (std::size_t term = 0; term < shape.inner; ++term)
            right[col * inner_domain + term] =
                witness.right[col * inner_domain + term];
    for (std::size_t col = 0; col < cols_domain; ++col)
        for (std::size_t term = 0; term < inner_domain; ++term) {
            const bool padding = col >= shape.cols || term >= shape.inner;
            if (padding && !witness.right[col * inner_domain + term].isZero()) {
                masked_set_reason(reason,
                                  "masked FCONN RHS padding is not zero");
                return false;
            }
        }
    return true;
}

bool masked_product_prove(NativeProofSession &session, const std::string &label,
                          const PedersenBasis &basis,
                          const std::shared_ptr<const
                              zkgpt::native_pedersen::ProverFastPath> &fast_path,
                          const std::vector<F> &left,
                          const std::vector<F> &right,
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
    std::vector<F> f = left;
    std::vector<F> g = right;
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
        std::vector<F> next_f(half), next_g(half);
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

std::vector<F> masked_gkr_coefficients(const layer &current,
                                        const std::vector<F> &r0,
                                        const std::vector<F> &r1,
                                        const std::vector<F> &ru,
                                        const F &alpha, const F &beta,
                                        const F &relu_rou) {
    std::vector<F> beta_g, beta_u, uni, bin;
    transcript_beta_phase1(current, 0, r0, r1, ru, alpha, beta, relu_rou,
                           beta_g, beta_u);
    transcript_predicate_phase1(current, beta_g, beta_u, uni, bin);
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
                             const std::vector<F> &public_output,
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
    // Diagnostics only: do not absorb progress labels into the proof
    // transcript or alter the independent verifier's wire.
    std::cerr << "[zkGPT][gkr] " << message << std::endl;
}

} // namespace

bool prove_masked_gkr_transcript(
    NativeProofSession &session, const layeredCircuit &circuit,
    const NativeLayeredGkrWitness &witness,
    const std::vector<F> &public_output, std::string *reason,
    NativeGkrInputBinding *input_binding,
    NativeGkrProverResult *prover_result) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (prover_result != nullptr)
            prover_result->has_unmasked_input_claim_opening = false;
        if (!session.valid(reason) || !masked_validate_circuit(circuit, public_output, reason))
            return false;
        if (witness.layer_values.size() != circuit.circuit.size()) {
            masked_set_reason(reason, "masked GKR layer witness count mismatch");
            return false;
        }
        for (std::size_t index = 0; index < circuit.circuit.size(); ++index)
            if (witness.layer_values[index].size() != circuit.circuit[index].size) {
                masked_set_reason(reason, "masked GKR layer witness shape mismatch");
                return false;
            }
        if (witness.layer_values.back() != public_output) {
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

        session.absorb_fields("gkr/public-output", public_output);
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
        F alpha = F_ONE, beta = F_ZERO;

        prover native_prover;
        native_prover.C = circuit;
        native_prover.val = witness.layer_values;
        native_prover.init();
        native_prover.sumcheckInitAll(r_u[circuit.size].begin());

        std::size_t next_matmul = witness.matmuls.size();
        for (int layer_id = static_cast<int>(circuit.size) - 1;
             layer_id > 0; --layer_id) {
            const layer &current = circuit.circuit[layer_id];
            native_prover.sumcheckInit(alpha, beta);
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

            if (current.ty == layerType::FCONN) {
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " fconn begin");
                if (next_matmul == 0) {
                    masked_set_reason(reason, "masked GKR FCONN descriptor underflow");
                    throw std::runtime_error(*reason);
                }
                const NativeGkrMatmulWitness &matmul = witness.matmuls[--next_matmul];
                if (!masked_fconn_shape_matches(current, matmul.shape, reason))
                    throw std::runtime_error(*reason);
                std::size_t rows_domain = 0, inner_domain = 0, cols_domain = 0;
                std::vector<F> left, right;
                if (!masked_matmul_witness_vectors(matmul, rows_domain, inner_domain,
                                                   cols_domain, left, right, reason))
                    throw std::runtime_error(*reason);
                const std::size_t n = ceilPow2BitLength(
                    static_cast<std::uint32_t>(matmul.shape.rows));
                const std::size_t m = ceilPow2BitLength(
                    static_cast<std::uint32_t>(matmul.shape.inner));
                const std::size_t k = ceilPow2BitLength(
                    static_cast<std::uint32_t>(matmul.shape.cols));
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
                const std::vector<F> left_mle = masked_fold_outer(
                    left, m, n, r_u[layer_id]);
                const std::vector<F> right_mle = masked_fold_outer(
                    right, m, k, r_v[layer_id]);
                const F initial_unmasked = have_previous_u1
                    ? previous_u1_unmasked : previous_unmasked;
                const F initial_masked = have_previous_u1
                    ? previous_u1_masked : previous_masked;
                NativeMaskedSumcheckResult product_result;
                if (!masked_product_prove(session, prefix + "/matmul", basis,
                                           fast_path,
                                           left_mle, right_mle, m, initial_masked,
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
                previous_u1_unmasked = F_ZERO;
                previous_u1_masked = F_ZERO;
                have_previous_u1 = true;
                if (current.bit_length_u[1] >= 0)
                    alpha = session.challenge(prefix + "/alpha");
                else alpha.clear();
                if (current.bit_length_v[1] >= 0 || current.ty == layerType::FFT)
                    beta = session.challenge(prefix + "/beta");
                else beta.clear();
                previous_unmasked = F_ZERO;
                previous_masked = F_ZERO;
                ScalarOpening zero_unmasked_opening;
                ScalarOpening zero_masked_opening;
                if (!masked_make_linear_claim(
                        session, basis, prefix + "/next-claim/zero-unmasked",
                        {product_result.unmasked_claim_opening}, {F_ZERO},
                        F_ZERO, zero_unmasked_opening, reason) ||
                    !masked_make_linear_claim(
                        session, basis, prefix + "/next-claim/zero-masked",
                        {product_result.claim_opening}, {F_ZERO}, F_ZERO,
                        zero_masked_opening, reason))
                    throw std::runtime_error(*reason);
                previous_unmasked_opening = zero_unmasked_opening;
                previous_masked_opening = zero_masked_opening;
                previous_u1_unmasked_opening = ScalarOpening{};
                previous_u1_masked_opening = ScalarOpening{};
                have_previous_u1 = false;
                masked_gkr_progress("layer=" + std::to_string(layer_id) +
                                    " fconn end rounds=" + std::to_string(m));
                continue;
            }

            NativeMaskedTerminalSpec phase1_placeholder;
            phase1_placeholder.kind = NativeMaskedTerminalKind::Claims;
            phase1_placeholder.has_external_binding = true;
            phase1_placeholder.public_initial_claim = false;
            phase1_placeholder.initial_opening = previous_unmasked_opening;
            phase1_placeholder.initial_commitment = previous_unmasked_opening.commitment;
            phase1_placeholder.has_masked_initial_binding = true;
            phase1_placeholder.masked_initial_opening = previous_masked_opening;
            phase1_placeholder.masked_initial_commitment = previous_masked_opening.commitment;
            NativeMaskedSumcheckProver phase1(
                session, prefix + "/phase1", 2, previous_masked,
                phase1_placeholder, &previous_unmasked, &basis, fast_path);
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
                std::vector<F> beta_v(std::size_t(1) << current.max_bl_v);
                initBetaTable(beta_v, current.max_bl_v, r_v[layer_id].begin(), F_ONE);
                std::vector<F> beta_g, beta_u, uni, bin;
                transcript_beta_phase1(current, layer_id, r0, r1,
                                       r_u[layer_id], alpha, beta, relu_rou,
                                       beta_g, beta_u);
                transcript_predicate_phase1(current, beta_g, beta_u, uni, bin);
                transcript_predicate_phase2(current, beta_g, beta_u, beta_v,
                                            uni, bin);
                NativeMaskedTerminalSpec phase2_spec;
                phase2_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedU;
                phase2_spec.gkr_coefficients = {bin[0], bin[1], bin[2], uni[0], uni[1]};
                // Phase2's fixed U side is exactly phase1's public masked
                // terminal.  Carry both representations and reject any
                // later substitution instead of trusting a recomputed vector.
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
                // With fixed V=(0,0), only uni[0] and uni[1] remain in the
                // predicate.  If both vanish, the unmasked relation is the
                // canonical constant zero; the masked Sumcheck may carry a
                // hidden terminal mask and must defer only that masked
                // predicate check while retaining all opening bindings.
                phase1_spec.defer_terminal_relation =
                    coefficients.size() == 5 && coefficients[3].isZero() &&
                    coefficients[4].isZero();
                phase1_spec.unmasked_fixed_values = {F_ZERO, F_ZERO};
                const ScalarOpening zero_opening{
                    {basis.domain,
                     zkgpt::native_pedersen::commit(basis, F_ZERO, F_ZERO).point},
                    F_ZERO, F_ZERO};
                phase1_spec.fixed_openings = {
                    zero_opening, zero_opening};
                phase1_spec.unmasked_fixed_openings = {zero_opening, zero_opening};
                phase1_spec.unmasked_terminal_openings = {
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u0),
                    zkgpt::native_pedersen::commit_fresh(
                        phase1.result().basis, claim_u1)};
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
        if (next_matmul != 0)
            throw std::runtime_error("masked GKR FCONN descriptor was not consumed");
        // The descending GKR walk ends at the canonical input MLE point.  It
        // is part of the same masked session so the adapter can later bind
        // this claim to the four input segments and the Lasso query claim.
        // A one-element input MLE has a legitimate zero-bit evaluation
        // point.  Keep the power-of-two/domain check authoritative rather
        // than rejecting its empty challenge vector.
        if (r_u[1].size() >= sizeof(std::size_t) * 8 ||
            (std::size_t(1) << r_u[1].size()) != witness.layer_values.front().size())
            throw std::runtime_error("masked GKR final input point is invalid");
        const F input_unmasked = transcript_mle_eval(
            witness.layer_values.front(), r_u[1]);
        // The native layered walk may end with a linear combination of
        // several input-routed layer claims (not a single literal MLE of the
        // flattened input when the circuit contains independent FCONN
        // layers).  Keep the authenticated masked terminal emitted by that
        // walk; the adapter's selector/combine relation binds it to the
        // canonical four-segment input MLE below.
        (void)input_unmasked;
        session.absorb_fields("gkr/input/point", r_u[1]);
        const ScalarOpening input_mask = zkgpt::native_pedersen::commit_fresh(
            basis, previous_masked - previous_unmasked_opening.value);
        if (!masked_emit_opening(session, basis, "gkr/input/claim",
                                 previous_masked_opening, reason) ||
            !masked_emit_opening(session, basis, "gkr/input/mask", input_mask,
                                 reason) ||
            !zkgpt::native_pedersen::prove_hidden_linear(
                session, basis, "gkr/input/bind",
                {previous_masked_opening, previous_unmasked_opening, input_mask},
                {F_ONE, -F_ONE, -F_ONE}, F_ZERO, nullptr, reason))
            throw std::runtime_error(reason != nullptr && !reason->empty()
                                         ? *reason
                                         : "masked GKR input binding failed");
        NativeGkrInputBinding final_binding;
        final_binding.point = r_u[1];
        final_binding.claim_commitment = previous_masked_opening.commitment;
        final_binding.unmasked_claim_commitment =
            previous_unmasked_opening.commitment;
        final_binding.mask_commitment = input_mask.commitment;
        if (input_binding != nullptr) *input_binding = final_binding;
        if (prover_result != nullptr) {
            prover_result->input_binding = final_binding;
            // This is the exact opening that was threaded through every
            // authenticated next-claim transition; do not re-commit here.
            prover_result->unmasked_input_claim_opening =
                previous_unmasked_opening;
            prover_result->has_unmasked_input_claim_opening = true;
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
    const std::vector<F> &public_output,
    const std::vector<NativeGkrMatmulShape> &matmuls,
    std::size_t start, std::size_t *end, std::string *reason,
    NativeGkrInputBinding *input_binding) {
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
        if (!masked_expect_message(session, cursor, NativeProofMessageKind::FieldVector,
                                   "gkr/public-output", &output_message, reason) ||
            !masked_decode_public_output(output_message, public_output, reason))
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
        F alpha = F_ONE, beta = F_ZERO;
        std::size_t next_matmul = matmuls.size();
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
            if (current.ty == layerType::FCONN) {
                if (next_matmul == 0) {
                    masked_set_reason(reason, "masked GKR FCONN descriptor underflow");
                    return false;
                }
                const NativeGkrMatmulShape &shape = matmuls[--next_matmul];
                if (!masked_fconn_shape_matches(current, shape, reason)) return false;
                const std::size_t n = ceilPow2BitLength(
                    static_cast<std::uint32_t>(shape.rows));
                const std::size_t m = ceilPow2BitLength(
                    static_cast<std::uint32_t>(shape.inner));
                const std::size_t k = ceilPow2BitLength(
                    static_cast<std::uint32_t>(shape.cols));
                if (r_u[layer_id + 1].size() < n + k)
                    return false;
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
                previous_u1_masked = F_ZERO;
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
                previous_masked = F_ZERO;
                ScalarCommitment zero_unmasked_commitment;
                ScalarCommitment zero_masked_commitment;
                if (!masked_consume_linear_claim(
                        session, basis, prefix + "/next-claim/zero-unmasked",
                        cursor, {product_result.unmasked_claim_commitment},
                        {F_ZERO}, F_ZERO, zero_unmasked_commitment, reason) ||
                    !masked_consume_linear_claim(
                        session, basis, prefix + "/next-claim/zero-masked",
                        cursor, {product_result.claim_commitment}, {F_ZERO},
                        F_ZERO, zero_masked_commitment, reason))
                    return false;
                previous_unmasked_commitment = zero_unmasked_commitment;
                previous_masked_commitment = zero_masked_commitment;
                previous_u1_unmasked_commitment = ScalarCommitment{};
                have_previous_u1 = false;
                continue;
            }
            if (current.max_bl_u == 0) return false;
            const std::string phase1_label = prefix + "/phase1";
            NativeMaskedSumcheckResult phase1_result;
            NativeMaskedTerminalSpec phase1_spec;
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
                std::vector<F> beta_v(std::size_t(1) << current.max_bl_v);
                initBetaTable(beta_v, current.max_bl_v,
                              r_v[layer_id].begin(), F_ONE);
                std::vector<F> beta_g, beta_u, uni, bin;
                transcript_beta_phase1(current, layer_id, r0, r1,
                                       r_u[layer_id], alpha, beta, relu_rou,
                                       beta_g, beta_u);
                transcript_predicate_phase1(current, beta_g, beta_u, uni, bin);
                transcript_predicate_phase2(current, beta_g, beta_u, beta_v,
                                            uni, bin);
                NativeMaskedTerminalSpec fixed_u_spec;
                fixed_u_spec.kind = NativeMaskedTerminalKind::GkrPredicateFixedU;
                fixed_u_spec.gkr_coefficients = {bin[0], bin[1], bin[2], uni[0], uni[1]};
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
                // The phase2 fixed side must consume the exact authenticated
                // phase1 terminal commitments.  Its public values and
                // commitments are supplied together to the native verifier.
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
                // The fixed-V relation is checked inside the authenticated
                // terminal scope; no raw terminal claim is trusted here.
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
        std::vector<F> encoded_input_point;
        const NativeProofMessage *input_point_message = nullptr;
        if (!masked_expect_message(session, cursor,
                                   NativeProofMessageKind::FieldVector,
                                   "gkr/input/point", &input_point_message,
                                   reason) ||
            !decode_transcript_fields(input_point_message->payload,
                                      encoded_input_point) ||
            encoded_input_point != r_u[1]) {
            masked_set_reason(reason, "masked GKR final input point binding failed");
            return false;
        }
        ScalarCommitment input_claim_commitment;
        ScalarCommitment input_mask_commitment;
        // The final input claim is the authenticated output of the last
        // next-claim transition.  Do not accept a replacement commitment at
        // the boundary merely because its later mask relation is valid.
        if (!masked_consume_opening(session, basis, "gkr/input/claim", cursor,
                                    &previous_masked_commitment,
                                    &input_claim_commitment, reason) ||
            !masked_consume_opening(session, basis, "gkr/input/mask", cursor,
                                    nullptr, &input_mask_commitment, reason) ||
            !masked_consume_linear(
                session, basis, "gkr/input/bind", cursor,
                {input_claim_commitment, previous_unmasked_commitment,
                 input_mask_commitment},
                {F_ONE, -F_ONE, -F_ONE}, F_ZERO, reason))
            return false;
        if (input_binding != nullptr) {
            input_binding->point = r_u[1];
            input_binding->claim_commitment = input_claim_commitment;
            input_binding->unmasked_claim_commitment = previous_unmasked_commitment;
            input_binding->mask_commitment = input_mask_commitment;
        }
        // The adapter consumes the mask commitment together with the three
        // terminal opening mask commitments in input-combine.  Any remaining
        // GKR-labelled message here would be an inserted or trailing proof
        // object; unrelated combine/opening suffixes are allowed.
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
    const std::vector<F> &public_output,
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
