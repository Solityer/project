#include "hyrax.hpp"
#include "proof_session.hpp"
#include "native_masked_sumcheck.hpp"
#include "timer.hpp"
#include <cmath>
#include <thread>
#include <vector>
#include <mutex>
#include <iostream>
#include <chrono>
#include <sstream>
#include <stdexcept>
#include <memory>
#include <limits>
using namespace std;
using namespace mcl::bn;

namespace
{

std::string point_encoding(const G1 &point)
{
    return point.serializeToHexStr();
}

bool valid_point(const G1 &point, bool require_nonzero)
{
    return (!require_nonzero || !point.isZero()) && point.isValid() && point.isValidOrder();
}

std::string basis_encoding(const G1 &G, const std::vector<G1> &generators, const std::vector<Fr> &basis)
{
    std::ostringstream payload;
    payload << point_encoding(G) << '|' << generators.size();
    for (std::size_t i = 0; i < generators.size(); ++i)
        payload << '|' << point_encoding(generators[i]) << '|' << basis[i].getStr(16);
    return payload.str();
}

void absorb_fold(NativeProofSession &session, const std::string &label, const G1 &gamma_minus1, const G1 &gamma_1, const Fr &cross_left, const Fr &cross_right)
{
    std::ostringstream payload;
    payload << point_encoding(gamma_minus1) << '|' << point_encoding(gamma_1);
    (void)cross_left;
    (void)cross_right;
    session.absorb(label, payload.str());
}

} // namespace

const int MAX_MSM_LEN=1e4;
const int COMM_OPT_MAX=65536; //don't optimize if larger than this
const int logmax=16;  /// max number=2^18-1
const int block_num=5;  //2^80


G1 perdersen_commit(G1* g,ll* f,int n,G1* W)
{
    G1 ret;
    ret.clear();
    //timer t(true);
    //t.start();
    
    bool *used=new bool[COMM_OPT_MAX*block_num];
    memset(used,0,sizeof(bool)*COMM_OPT_MAX*block_num);
    ll bar[10];
    ll bar_t=1;
    for(int i=0;i<8;i++)
    {
        bar[i]=bar_t;
        bar_t<<=logmax;
    }
    for(int i=0;i<n;i++)
    {
            if(f[i]==0)
                continue;
            
            if(f[i]<0)
            {
                ll tmp=-f[i];
                for(int j=0;j<block_num;j++)
                {
                    if(tmp<bar[j])
                        break;
                    ll fnow=(tmp>>(logmax*j))&65535;
                    W[fnow+(j<<logmax)]-=g[i];
                    used[fnow+(j<<logmax)]=1;
                }
            }
            else
            {
                ll tmp=f[i];
                for(int j=0;j<block_num;j++)
                {
                    if(tmp<bar[j])
                        break;
                    ll fnow=(tmp>>(logmax*j))&65535;
                    W[fnow+(j<<logmax)]+=g[i];
                    used[fnow+(j<<logmax)]=1;
                }
            }
    }
    G1 gg[logmax*block_num];
    for(int j=0;j<logmax*block_num;j++)
        gg[j].clear();
    for(int j=0;j<COMM_OPT_MAX*block_num;j++)
    {
        if(used[j])
        {
            int jj=j%COMM_OPT_MAX;
            int blk=j/COMM_OPT_MAX;
            for(int k=0;k<logmax;k++)
            {
                if(jj&(1<<k))
                    gg[k+logmax*blk]+=W[j];
            }
            W[j].clear();
            used[j]=0;            
        }
    }
    for(int j=0;j<logmax*block_num;j++)
    {
        if(j>60)
        {
            G1 gd=gg[j]*(1ll<<48);
            ret+=gd*(1ll<<(j-48)); //split
        }
        else
            ret+=gg[j]*(1ll<<j);
    }
    delete []used;
    return ret;
}


G1 perdersen_commit(G1* g,int* f,int n,G1* W)
{
    G1 ret;
    ret.clear();
    bool *used=new bool[COMM_OPT_MAX];
    memset(used,0,sizeof(bool)*COMM_OPT_MAX);
    for(int i=0;i<n;i++)
    {
            if(f[i]==0)
                continue;
            
            if(f[i]<0)
            {
                W[-f[i]]-=g[i];
                used[-f[i]]=1;
                assert(-f[i]<COMM_OPT_MAX);
            }
            else
            {
                W[f[i]]+=g[i];
                used[f[i]]=1;
                assert(f[i]<COMM_OPT_MAX);
            }
    }
    //t.stop("add ",false);
    const int logn=log2(COMM_OPT_MAX)+1;
    G1 gg[40];
    for(int j=0;j<logn;j++)
        gg[j].clear();
    for(int j=1;j<COMM_OPT_MAX;j++)
    {
        if(used[j])
        {
            for(int k=0;k<logn;k++)
            {
                if(j&(1<<k))
                    gg[k]+=W[j];
            }
            W[j].clear();
            used[j]=0;            
        }
    }
    for(int j=0;j<logn;j++)
        ret+=gg[j]*(1<<j);

    //t.stop("accu",false);
    //t.stop("ALL: ",true);

    delete []used;
    return ret;
}

G1 perdersen_commit(G1* g,Fr* f,int n)
{
    G1 ret;
    G1::mulVec(ret,g,f,n);
    return ret;
}

Fr lagrange(Fr *r,int l,int k)
{
    assert(k>=0 && k<(1<<l));
    Fr ret=1;
    for(int i=0;i<l;i++)
    {
        if(k&(1<<i))
            ret*=r[i];
        else
            ret*=1-r[i];
    }
    return ret;
}
void brute_force_compute_LR(Fr* L,Fr* R,Fr* r,int l)
{
    int halfl=l/2,c=l-halfl;
    for(int k=0;k<(1<<c);k++)
        L[k]=lagrange(r,c,k);
    for(int k=0;k<(1<<halfl);k++)
        R[k]=lagrange(r+c,halfl,k);
}

Fr brute_force_compute_eval(Fr* w,Fr* r,int l)
{
    Fr ret=0;
    for(int k=0;k<(1<<l);k++)
        ret+=lagrange(r,l,k)*w[k];
    return ret;
}


G1 compute_Tprime(int l,Fr* R,G1* Tk) 
{
    //w has 2^l length
    //assert(l%2==0);
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    G1 ret=perdersen_commit(Tk,R,rownum);
    return ret;
}

G1 compute_RT_singlethread(Fr*w ,Fr*R,int l,G1*g,Fr*& ret) // L is row number length
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    Fr* res=new Fr[colnum];
    for(int i=0;i<colnum;i++)
        res[i]=0;
    timer TT;
    TT.start();
    for(int j=0;j<colnum;j++)
    for(int i=0;i<rownum;i++)
    {
        if(!w[j+i*colnum].isZero())
            res[j]+=R[i]*w[j+i*colnum];  // mat mult  (1,row)*(row,col)=(1,col)
    }
    TT.stop();
    G1 comm=perdersen_commit(g,res,colnum);
    ret=res;
    return comm;
}
G1 compute_RT(Fr *w, Fr *R, int l, G1 *g, Fr *&ret) // L is row number length
{
    int halfl = l / 2;
    int rownum = (1 << halfl), colnum = (1 << (l - halfl));
    Fr *res = new Fr[colnum];
    for (int i = 0; i < colnum; i++)
        res[i] = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Define the number of threads
    const int num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    std::mutex res_mutex;

    auto worker = [&](int start_col, int end_col) {
        Fr *local_res = new Fr[colnum]; // Thread-local result storage
        for (int i = start_col; i < end_col; i++)
            local_res[i] = 0;

        for (int j = start_col; j < end_col; j++) {
            for (int i = 0; i < rownum; i++) {
                if (!w[j + i * colnum].isZero()) {
                    local_res[j] += R[i] * w[j + i * colnum];
                }
            }
        }

        // Merge local results into global result
        std::lock_guard<std::mutex> guard(res_mutex);
        for (int j = start_col; j < end_col; j++) {
            res[j] += local_res[j];
        }

        delete[] local_res; // Free thread-local storage
    };

    // Split columns into chunks for each thread
    int chunk_size = (colnum + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; t++) {
        int start_col = t * chunk_size;
        int end_col = std::min(start_col + chunk_size, colnum);
        if (start_col < colnum) { // Avoid starting threads if no work is left
            threads.emplace_back(worker, start_col, end_col);
        }
    }

    // Join threads
    for (auto &th : threads) {
        if (th.joinable())
            th.join();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    // Commit the result
    G1 comm = perdersen_commit(g, res, colnum);
    ret = res;
    return comm;
}

G1 gen_gi(G1* g,int n)
{
    G1 base;
    for(int i=0;i<n;i++)
    {
        Fr tmp;
        tmp.setByCSPRNG();
        g[i]=base*tmp;
    }
    return base;
}

double blt_vtime=0;
Pack bullet_reduce(G1 gamma, Fr*a,G1*g,int n,G1& G,Fr* x,Fr y,bool need_free) // length n
{
    timer vtimer;
    if(n==1)
    {
        Pack p(gamma,a[0],g[0],x[0],y);
        return p;
    }
    
    //step2  prover fold
    G1 gamma_minus1,gamma_1;
    Fr x1a2=0,x2a1=0;
    for(int i=0;i<n/2;i++)
    {
        x1a2+=x[i]*a[n/2+i];
        x2a1+=x[n/2+i]*a[i];
    }
    gamma_minus1=G*x1a2+perdersen_commit(g+n/2,x,n/2);
    gamma_1=G*x2a1+perdersen_commit(g,x+n/2,n/2);
    Fr c,invc;
    do {
        c.setByCSPRNG();
    } while (c.isZero());
    //prover verifier both comp
    vtimer.start();
    Fr::inv(invc,c);
    G1 gamma_prime=gamma_minus1*c*c+gamma_1*invc*invc+gamma;
    Fr* aprime=new Fr[n/2];       
    for(int i=0;i<n/2;i++)
        aprime[i]=a[i]*invc+a[i+n/2]*c;
    G1* gprime=new G1[n/2];        
    if(n<2048)
    {   
        for(int i=0;i<n/2;i++)
            gprime[i]=g[i]*invc+g[i+n/2]*c;
    }
    else  
    {
        int num_threads=16;
        int chunk_size = n/2/num_threads;
        std::vector<std::thread> threads1;
        auto worker1 = [&](int start_id, int end_id) 
        {
            for(int i=start_id;i<end_id;i++)
                gprime[i]=g[i]*invc+g[i+n/2]*c;
        };
        for (int t = 0; t < num_threads; t++) 
        {
            int start_col = t * chunk_size;
            int end_col = min(start_col + chunk_size, n/2);
            threads1.emplace_back(worker1, start_col, end_col);
        }
        for (auto &th : threads1) 
        {
            if (th.joinable())
                th.join();
        }
    }
    vtimer.stop();

    blt_vtime+=vtimer.elapse_sec();

    //prover single compute
    Fr* xprime=new Fr[n/2];         
    Fr yprime;
    for(int i=0;i<n/2;i++)
        xprime[i]=c*x[i]+invc*x[i+n/2];
    yprime=c*c*x1a2+invc*invc*x2a1+y;

    if(need_free)
    {
        delete []a;
        delete []g;
        delete []x;
    }
    
    return bullet_reduce(gamma_prime,aprime,gprime,n/2,G,xprime,yprime,true);
}   

void prove_dot_product(G1 comm_x, G1 comm_y, Fr *a, G1 *g, G1 &G, Fr *x, Fr y, int n)
{
    G1 gamma=comm_x+comm_y;
    Pack p=bullet_reduce(gamma,a,g,n,G,x,y);
    assert(p.y==p.x*p.a);
    assert(p.gamma==p.g*p.x+G*p.y);
}

namespace {

Pack bullet_reduce_with_session(G1 gamma, Fr *a, G1 *g, int n,
                                G1 &G, Fr *x, Fr y,
                                NativeProofSession &session,
                                const std::string &label, int round) {
    if (n <= 0 || (n & (n - 1)) != 0)
        throw std::invalid_argument("Hyrax opening length is not a power of two");
    if (n == 1) {
        Pack result(gamma, a[0], g[0], x[0], y);
        session.absorb(label + "/terminal",
                       point_encoding(gamma) + "|" + point_encoding(g[0]) +
                       "|" + a[0].getStr(16) + "|" + x[0].getStr(16) +
                       "|" + y.getStr(16));
        if (result.y != result.x * result.a ||
            result.gamma != result.g * result.x + G * result.y)
            throw std::runtime_error("Hyrax terminal opening relation failed");
        return result;
    }
    G1 gamma_minus1, gamma_1;
    Fr x1a2 = 0, x2a1 = 0;
    for (int i = 0; i < n / 2; ++i) {
        x1a2 += x[i] * a[n / 2 + i];
        x2a1 += x[n / 2 + i] * a[i];
    }
    gamma_minus1 = G * x1a2 + perdersen_commit(g + n / 2, x, n / 2);
    gamma_1 = G * x2a1 + perdersen_commit(g, x + n / 2, n / 2);
    absorb_fold(session, label + "/fold/" + std::to_string(round),
                gamma_minus1, gamma_1, x1a2, x2a1);
    const Fr c = session.challenge(label + "/c/" + std::to_string(round));
    if (c.isZero())
        throw std::runtime_error("Hyrax Fiat-Shamir challenge is zero");
    Fr invc;
    Fr::inv(invc, c);
    G1 gamma_prime = gamma_minus1 * c * c + gamma_1 * invc * invc + gamma;
    std::vector<Fr> aprime(n / 2);
    std::vector<G1> gprime(n / 2);
    std::vector<Fr> xprime(n / 2);
    for (int i = 0; i < n / 2; ++i) {
        aprime[i] = a[i] * invc + a[i + n / 2] * c;
        gprime[i] = g[i] * invc + g[i + n / 2] * c;
        xprime[i] = c * x[i] + invc * x[i + n / 2];
    }
    const Fr yprime = c * c * x1a2 + invc * invc * x2a1 + y;
    const Pack result = bullet_reduce_with_session(
        gamma_prime, aprime.data(), gprime.data(), n / 2, G, xprime.data(), yprime, session,
        label, round + 1);
    return result;
}

}

void prove_dot_product_with_session(G1 comm_x, G1 comm_y, Fr *a, G1 *g,
                                    G1 &G, Fr *x, Fr y, int n,
                                    NativeProofSession &session,
                                    const std::string &label,
                                    bool enforce_statement_basis) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || a == nullptr || g == nullptr || x == nullptr ||
            n <= 0 || (n & (n - 1)) != 0 ||
            !valid_point(G, true))
            throw std::invalid_argument("invalid Hyrax opening basis");
        std::vector<G1> generators(g, g + n);
        std::vector<Fr> basis(a, a + n);
        for (const G1 &generator : generators)
            if (!valid_point(generator, true))
                throw std::invalid_argument("invalid Hyrax generator");
        if (enforce_statement_basis && session.statement().basis_digest !=
            NativeProofStatement::basis_digest_for(G, generators, basis, n))
            throw std::invalid_argument("Hyrax basis digest does not match statement");
        session.absorb(label + "/basis", basis_encoding(G, generators, basis));
        session.absorb(label + "/evaluation",
                       point_encoding(comm_x) + "|" + y.getStr(16));
        const G1 gamma = comm_x + comm_y;
        const Pack result = bullet_reduce_with_session(gamma, a, g, n, G, x, y,
                                                       session, label, 0);
        if (result.y != result.x * result.a ||
            result.gamma != result.g * result.x + G * result.y)
            throw std::runtime_error("Hyrax transcript opening relation failed");
    } catch (...) {
        session.rollback(checkpoint);
        throw;
    }
}
static ThreadSafeQueue<int> workerq,endq;


void ll_commit_worker(G1*& Tk,G1*& g, ll*& row,int colnum,G1*& W)
{
    int idx;
    while (true)
    {
            bool ret=workerq.TryPop(idx);
            if(ret==false)
                return;
            Tk[idx]=perdersen_commit(g,row+idx*colnum,colnum,W);
            endq.Push(idx);
    }
}
G1* prover_commit(ll* w, G1* g, int l,int thread_n) //compute Tk, int version with pippenger
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    G1 *Tk=new G1[rownum];
    ll* row=new ll[1<<l];
    G1** W=new G1*[thread_n];
    for(int i=0;i<thread_n;i++)
        W[i]=new G1[COMM_OPT_MAX*block_num];
    for(int i=0;i<thread_n;i++)
        memset(W[i],0,sizeof(G1)*COMM_OPT_MAX*block_num);
    for (u64 i = 0; i < rownum; ++i)  //work for rownum 
        workerq.Push(i);
    for(int i=0;i<thread_n;i++)
    {
        thread t(ll_commit_worker,std::ref(Tk),std::ref(g),std::ref(w),colnum,std::ref(W[i])); 
        t.detach();
    }
    while(!workerq.Empty())
        this_thread::sleep_for (std::chrono::microseconds(10));
    while(endq.Size()!=rownum)
        this_thread::sleep_for (std::chrono::microseconds(10));
    endq.Clear();
    assert(endq.Size()==0);
    for(int i=0;i<thread_n;i++)
        delete [] W[i];
    delete []W;
    return Tk;
}



Fr prover_evaluate(Fr*w ,Fr*r,G1& G,G1* g, Fr*L,Fr*R,int l)  // nlogn brute force 
{
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    timer t(true);
    t.start();
    brute_force_compute_LR(L,R,r,l);
    Fr eval=brute_force_compute_eval(w,r,l);
    t.stop("eval total ",true,false);
    return eval;
}
namespace hyrax
{
pair<double, double> open(Fr *w, Fr *r, Fr eval, G1 &G, G1 *g, Fr *L, Fr *R, G1 *tk, int l)
{
    double prover_time=0,verifier_time=0;
    int halfl=l/2;
    int rownum=(1<<halfl),colnum=(1<<(l-halfl));
    timer verf;
    verf.start();
    Fr *rt_raw = nullptr;
    compute_RT(w, R, l, g, rt_raw);
    std::unique_ptr<Fr[]> RT(rt_raw);
    verf.stop();
    prover_time+=verf.elapse_sec();
    verf.start();
    G1 tprime=compute_Tprime(l,R,tk);
    verf.stop();
    prover_time+=verf.elapse_sec();
    verifier_time+=verf.elapse_sec();
    verf.start();
    prove_dot_product(tprime, G*eval, L, g , G,RT.get(),eval,colnum);
    verf.stop();
    prover_time+=verf.elapse_sec();
    verifier_time+=blt_vtime;
    return make_pair(prover_time,verifier_time);
}

pair<double,double> open_with_session(Fr *w, Fr *r, Fr eval, G1 &G, G1 *g,
                                      Fr *L, Fr *R, G1 *tk, int l,
                                      NativeProofSession &session,
                                      const std::string &label) {
    const std::size_t checkpoint = session.checkpoint();
    try {
    double prover_time = 0, verifier_time = 0;
    const int halfl = l / 2;
    const int colnum = (1 << (l - halfl));
    timer verf;
    verf.start();
    Fr *rt_raw = nullptr;
    compute_RT(w, R, l, g, rt_raw);
    std::unique_ptr<Fr[]> RT(rt_raw);
    verf.stop();
    prover_time += verf.elapse_sec();
    verf.start();
    G1 tprime = compute_Tprime(l, R, tk);
    verf.stop();
    prover_time += verf.elapse_sec();
    verifier_time += verf.elapse_sec();
    verf.start();
    prove_dot_product_with_session(tprime, G * eval, L, g, G, RT.get(), eval,
                                   colnum, session, label + "/bullet");
    verf.stop();
    prover_time += verf.elapse_sec();
    verifier_time += blt_vtime;
    return make_pair(prover_time, verifier_time);
    } catch (...) {
        session.rollback(checkpoint);
        throw;
    }
}

namespace {

void set_open_reason(std::string *reason, const std::string &message) {
    if (reason != nullptr) *reason = message;
}

bool split_open_payload(const std::string &payload, std::size_t expected,
                        std::vector<std::string> &parts) {
    parts.clear();
    std::size_t begin = 0;
    while (true) {
        const std::size_t end = payload.find('|', begin);
        parts.push_back(payload.substr(begin, end == std::string::npos ? end : end - begin));
        if (parts.size() > expected) return false;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return parts.size() == expected;
}

bool decode_open_field(const std::string &encoded, Fr &value) {
    if (encoded.empty() || encoded.size() > 128) return false;
    try {
        value.setStr(encoded, 16);
        return value.getStr(16) == encoded;
    } catch (...) {
        return false;
    }
}

bool decode_open_point(const std::string &encoded, G1 &value) {
    if (encoded.empty() || encoded.size() > 4096) return false;
    try {
        value.deserializeHexStr(encoded);
        return value.serializeToHexStr() == encoded && valid_point(value, false);
    } catch (...) {
        return false;
    }
}

const NativeProofMessage *find_open_message(
    const NativeProofSession &session, const std::string &label,
    std::string *reason) {
    const std::vector<std::size_t> *positions = session.message_positions(label);
    if (positions == nullptr || positions->empty()) {
        set_open_reason(reason, "missing Hyrax opening message: " + label);
        return nullptr;
    }
    if (positions->size() != 1) {
        set_open_reason(reason, "duplicate Hyrax opening message: " + label);
        return nullptr;
    }
    const std::size_t position = positions->front();
    if (position >= session.messages().size()) {
        set_open_reason(reason, "Hyrax opening message index is out of range: " + label);
        return nullptr;
    }
    return &session.messages()[position];
}

bool has_prefix(const std::string &value, const std::string &prefix) {
    return value.compare(0, prefix.size(), prefix) == 0;
}

bool power_of_two_size(std::size_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t log2_size(std::size_t value) {
    std::size_t result = 0;
    while (value > 1) {
        value >>= 1;
        ++result;
    }
    return result;
}

std::vector<Fr> mle_basis_from_point(const std::vector<Fr> &point) {
    if (point.size() >= sizeof(std::size_t) * 8)
        throw std::invalid_argument("masked Hyrax evaluation point is invalid");
    const std::size_t size = std::size_t(1) << point.size();
    if (point.empty()) return std::vector<Fr>{Fr(1)};
    std::vector<Fr> basis(size);
    basis[0] = Fr(1) - point[0];
    basis[1] = point[0];
    for (std::size_t bit = 1; bit < point.size(); ++bit) {
        const std::size_t old_size = std::size_t(1) << bit;
        for (std::size_t index = old_size; index-- > 0;) {
            const Fr previous = basis[index];
            basis[index] = previous * (Fr(1) - point[bit]);
            basis[index + old_size] = previous * point[bit];
        }
    }
    return basis;
}

std::string masked_claim_payload(
    const G1 &mask_commitment, const Fr &claim,
    const zkgpt::native_pedersen::ScalarOpening *raw_opening = nullptr,
    const zkgpt::native_pedersen::ScalarOpening *masked_opening = nullptr) {
    std::string payload = point_encoding(mask_commitment) + "|" + claim.getStr(16);
    if (raw_opening != nullptr || masked_opening != nullptr) {
        if (raw_opening == nullptr || masked_opening == nullptr)
            throw std::invalid_argument("incomplete masked scalar binding");
        payload += "|" + raw_opening->commitment.domain + "|" +
                   point_encoding(raw_opening->commitment.point) + "|" +
                   point_encoding(masked_opening->commitment.point);
    }
    return payload;
}

std::string masked_announce_payload(const G1 &announce_commitment,
                                   const G1 &mask_announce,
                                   const Fr &announce_eval,
                                   const G1 *raw_announce = nullptr,
                                   const G1 *masked_announce = nullptr) {
    std::string payload = point_encoding(announce_commitment) + "|" +
                          point_encoding(mask_announce) + "|" +
                          announce_eval.getStr(16);
    if (raw_announce != nullptr || masked_announce != nullptr) {
        if (raw_announce == nullptr || masked_announce == nullptr)
            throw std::invalid_argument("incomplete masked scalar announcement");
        payload += "|" + point_encoding(*raw_announce) + "|" +
                   point_encoding(*masked_announce);
    }
    return payload;
}

bool decode_field_vector_payload(const std::string &payload,
                                std::vector<Fr> &values) {
    values.clear();
    if (payload.size() < sizeof(std::uint64_t)) return false;
    std::size_t offset = 0;
    auto read_u64 = [&](std::uint64_t &value) {
        if (payload.size() - offset < sizeof(std::uint64_t)) return false;
        value = 0;
        for (unsigned index = 0; index < 8; ++index)
            value |= static_cast<std::uint64_t>(
                         static_cast<unsigned char>(payload[offset++]))
                     << (8 * index);
        return true;
    };
    std::uint64_t count = 0;
    if (!read_u64(count) || count == 0 || count > (1ULL << 20)) return false;
    values.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        std::uint64_t length = 0;
        if (!read_u64(length) || length == 0 || length > 128 ||
            length > payload.size() - offset)
            return false;
        Fr value;
        if (!decode_open_field(payload.substr(offset,
                                             static_cast<std::size_t>(length)),
                               value))
            return false;
        offset += static_cast<std::size_t>(length);
        values.push_back(value);
    }
    return offset == payload.size();
}

const NativeProofMessage *consume_masked_message(
    const NativeProofSession &session, std::size_t &position,
    NativeProofMessageKind kind, const std::string &label,
    std::string *reason) {
    if (position >= session.messages().size() ||
        session.messages()[position].kind != kind ||
        session.messages()[position].label != label) {
        set_open_reason(reason, "masked Hyrax grammar mismatch at " + label);
        return nullptr;
    }
    return &session.messages()[position++];
}

bool reject_masked_trailing(const NativeProofSession &session,
                            std::size_t position, const std::string &label,
                            bool allow_trailing_unrelated,
                            std::string *reason) {
    if (!allow_trailing_unrelated && position != session.messages().size()) {
        set_open_reason(reason, "masked Hyrax opening has a post-proof message");
        return false;
    }
    if (allow_trailing_unrelated) {
        const std::string prefix = label + "/";
        const std::string challenge_prefix = "challenge/" + prefix;
        for (std::size_t index = position; index < session.messages().size(); ++index) {
            const std::string &message_label = session.messages()[index].label;
            if (has_prefix(message_label, prefix) ||
                has_prefix(message_label, challenge_prefix)) {
                set_open_reason(reason,
                                "masked Hyrax opening has a trailing scoped message");
                return false;
            }
        }
    }
    return true;
}

}

bool verify_opening_with_session(
    const G1 &comm_x, const Fr &eval, const G1 &G,
    const std::vector<G1> &generators, const std::vector<Fr> &basis,
    const NativeProofSession &session, const std::string &label,
    std::string *reason, bool enforce_statement_basis,
    bool allow_trailing_unrelated) {
    try {
        if (label.empty() || generators.empty() || generators.size() != basis.size() ||
            (generators.size() & (generators.size() - 1)) != 0) {
            set_open_reason(reason, "Hyrax opening basis is not a non-empty power-of-two vector");
            return false;
        }
        if (!valid_point(G, true)) {
            set_open_reason(reason, "Hyrax blinding generator is invalid");
            return false;
        }
        for (const G1 &generator : generators) {
            if (!valid_point(generator, true)) {
                set_open_reason(reason, "Hyrax generator is invalid");
                return false;
            }
        }
        if (!session.replay(reason)) return false;
        if (enforce_statement_basis && session.statement().basis_digest !=
            NativeProofStatement::basis_digest_for(G, generators, basis,
                                                   session.statement().l_dimension)) {
            set_open_reason(reason, "Hyrax basis digest does not match statement");
            return false;
        }

        const std::string opening_prefix = label + "/";
        const std::string challenge_prefix = "challenge/" + label + "/";
        const std::vector<std::size_t> *basis_positions =
            session.message_positions(label + "/basis");
        if (basis_positions == nullptr || basis_positions->empty()) {
            set_open_reason(reason, "missing Hyrax basis binding");
            return false;
        }
        if (basis_positions->size() != 1) {
            set_open_reason(reason, "duplicate Hyrax basis binding");
            return false;
        }
        const std::size_t basis_index = basis_positions->front();
        if (basis_index >= session.messages().size()) {
            set_open_reason(reason, "Hyrax basis binding index is out of range");
            return false;
        }
        for (std::size_t index = 0; index < basis_index; ++index) {
            const std::string &message_label = session.messages()[index].label;
            if (has_prefix(message_label, opening_prefix) ||
                has_prefix(message_label, challenge_prefix)) {
                set_open_reason(reason, "Hyrax opening has a non-adjacent prelude");
                return false;
            }
        }
        std::size_t position = basis_index;
        const auto consume = [&](NativeProofMessageKind kind,
                                 const std::string &expected) -> const NativeProofMessage * {
            if (position >= session.messages().size() ||
                session.messages()[position].kind != kind ||
                session.messages()[position].label != expected) {
                set_open_reason(reason, "Hyrax opening grammar mismatch at " + expected);
                return nullptr;
            }
            return &session.messages()[position++];
        };

        const NativeProofMessage *basis_message = consume(
            NativeProofMessageKind::Data, label + "/basis");
        if (basis_message == nullptr) return false;
        std::vector<std::string> basis_parts;
        if (!split_open_payload(basis_message->payload, 2 + 2 * generators.size(),
                                basis_parts)) {
            set_open_reason(reason, "malformed Hyrax basis binding");
            return false;
        }
        G1 encoded_G;
        if (!decode_open_point(basis_parts[0], encoded_G) || encoded_G != G) {
            set_open_reason(reason, "Hyrax blinding generator binding mismatch");
            return false;
        }
        std::size_t encoded_size = 0;
        try {
            std::size_t consumed = 0;
            encoded_size = std::stoull(basis_parts[1], &consumed, 10);
            if (consumed != basis_parts[1].size()) throw std::invalid_argument("size");
        } catch (...) {
            set_open_reason(reason, "malformed Hyrax basis size");
            return false;
        }
        if (encoded_size != generators.size()) {
            set_open_reason(reason, "Hyrax basis size binding mismatch");
            return false;
        }
        for (std::size_t i = 0; i < generators.size(); ++i) {
            G1 encoded_generator;
            Fr encoded_basis;
            if (!decode_open_point(basis_parts[2 + 2 * i], encoded_generator) ||
                !decode_open_field(basis_parts[3 + 2 * i], encoded_basis) ||
                encoded_generator != generators[i] || encoded_basis != basis[i]) {
                set_open_reason(reason, "Hyrax basis/generator binding mismatch");
                return false;
            }
        }

        const NativeProofMessage *evaluation = consume(
            NativeProofMessageKind::Data, label + "/evaluation");
        if (evaluation == nullptr) return false;
        std::vector<std::string> evaluation_parts;
        if (!split_open_payload(evaluation->payload, 2, evaluation_parts)) {
            set_open_reason(reason, "malformed Hyrax evaluation message");
            return false;
        }
        G1 encoded_comm;
        Fr encoded_eval;
        if (!decode_open_point(evaluation_parts[0], encoded_comm) ||
            !decode_open_field(evaluation_parts[1], encoded_eval) ||
            encoded_comm != comm_x || encoded_eval != eval) {
            set_open_reason(reason, "Hyrax evaluation message does not match the public claim");
            return false;
        }

        std::vector<G1> folded_generators = generators;
        std::vector<Fr> folded_basis = basis;
        G1 folded_gamma = comm_x + G * eval;
        std::size_t round_count = 0;
        for (std::size_t size = generators.size(); size > 1; size >>= 1) ++round_count;
        for (std::size_t round = 0; round < round_count; ++round) {
            const std::string fold_label = label + "/fold/" + std::to_string(round);
            const NativeProofMessage *fold = consume(
                NativeProofMessageKind::Data, fold_label);
            if (fold == nullptr) return false;
            const NativeProofMessage *challenge = consume(
                NativeProofMessageKind::Challenge,
                "challenge/" + label + "/c/" + std::to_string(round));
            if (challenge == nullptr) return false;
            std::vector<std::string> fold_parts;
            if (!split_open_payload(fold->payload, 2, fold_parts)) {
                set_open_reason(reason, "malformed Hyrax fold message");
                return false;
            }
            G1 gamma_minus1, gamma_1;
            Fr c;
            if (!decode_open_point(fold_parts[0], gamma_minus1) ||
                !decode_open_point(fold_parts[1], gamma_1) ||
                !decode_open_field(challenge->payload, c) || c.isZero()) {
                set_open_reason(reason, "malformed Hyrax fold field or point");
                return false;
            }
            Fr inv_c;
            Fr::inv(inv_c, c);
            folded_gamma = gamma_minus1 * c * c + gamma_1 * inv_c * inv_c + folded_gamma;

            const std::size_t half = folded_generators.size() / 2;
            std::vector<G1> next_generators(half);
            std::vector<Fr> next_basis(half);
            for (std::size_t i = 0; i < half; ++i) {
                next_generators[i] = folded_generators[i] * inv_c +
                                     folded_generators[i + half] * c;
                next_basis[i] = folded_basis[i] * inv_c +
                                folded_basis[i + half] * c;
            }
            folded_generators.swap(next_generators);
            folded_basis.swap(next_basis);
        }

        const NativeProofMessage *terminal = consume(
            NativeProofMessageKind::Data, label + "/terminal");
        if (terminal == nullptr) return false;
        std::vector<std::string> terminal_parts;
        if (!split_open_payload(terminal->payload, 5, terminal_parts)) {
            set_open_reason(reason, "malformed Hyrax terminal opening");
            return false;
        }
        G1 terminal_gamma, terminal_generator;
        Fr terminal_basis, terminal_x, terminal_eval;
        if (!decode_open_point(terminal_parts[0], terminal_gamma) ||
            !decode_open_point(terminal_parts[1], terminal_generator) ||
            !decode_open_field(terminal_parts[2], terminal_basis) ||
            !decode_open_field(terminal_parts[3], terminal_x) ||
            !decode_open_field(terminal_parts[4], terminal_eval) ||
            folded_generators.size() != 1 || folded_basis.size() != 1 ||
            terminal_gamma != folded_gamma || terminal_generator != folded_generators[0] ||
            terminal_basis != folded_basis[0] ||
            terminal_eval != terminal_x * terminal_basis ||
            terminal_gamma != terminal_generator * terminal_x + G * terminal_eval) {
            set_open_reason(reason, "Hyrax terminal opening relation failed");
            return false;
        }

        if (!allow_trailing_unrelated && position != session.messages().size()) {
            set_open_reason(reason, "Hyrax opening has a post-terminal message");
            return false;
        }
        if (allow_trailing_unrelated) {
            const std::string opening_prefix = label + "/";
            const std::string challenge_prefix = "challenge/" + label + "/";
            for (std::size_t index = position; index < session.messages().size(); ++index) {
                const std::string &message_label = session.messages()[index].label;
                if (has_prefix(message_label, opening_prefix) ||
                    has_prefix(message_label, challenge_prefix)) {
                    set_open_reason(reason, "Hyrax opening has a trailing protocol message");
                    return false;
                }
            }
        }
        return true;
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}

G1 derive_pedersen_blinding_generator(const std::string &basis_id) {
    if (basis_id.empty() || basis_id.size() > 128)
        throw std::invalid_argument("Pedersen MLE basis identity is invalid");
    G1 result;
    hashAndMapToG1(result, "zkGPT/hyrax/pedersen/H/v1/" + basis_id);
    if (!valid_point(result, true))
        throw std::runtime_error("Pedersen MLE blinding generator is invalid");
    return result;
}

G1 derive_pedersen_evaluation_generator(const std::string &basis_id,
                                        const std::string &label) {
    if (basis_id.empty() || basis_id.size() > 128 || label.empty() ||
        label.size() > 256)
        throw std::invalid_argument("Pedersen MLE evaluation identity is invalid");
    G1 result;
    hashAndMapToG1(result, "zkGPT/hyrax/pedersen/J/v1/" + basis_id + "/" +
                             label);
    if (!valid_point(result, true))
        throw std::runtime_error("Pedersen MLE evaluation generator is invalid");
    return result;
}

G1 derive_shared_pedersen_evaluation_generator(const std::string &identity) {
    if (identity.empty() || identity.size() > 512)
        throw std::invalid_argument("shared Pedersen evaluation identity is invalid");
    G1 result;
    hashAndMapToG1(result, "zkGPT/hyrax/pedersen/shared-J/v1/" + identity);
    if (!valid_point(result, true))
        throw std::runtime_error("shared Pedersen evaluation generator is invalid");
    return result;
}

G1 derive_masked_response_ipa_base(const std::string &basis_id) {
    if (basis_id.empty() || basis_id.size() > 128)
        throw std::invalid_argument("masked response IPA basis identity is invalid");
    G1 result;
    hashAndMapToG1(result, "zkGPT/hyrax/masked-response-ipa/evaluation-base/v2/" + basis_id);
    if (!valid_point(result, true))
        throw std::runtime_error("masked response IPA base is invalid");
    return result;
}

namespace {

bool safe_open_text(const std::string &value, std::size_t maximum) {
    return !value.empty() && value.size() <= maximum &&
           value.find('|') == std::string::npos &&
           value.find('\n') == std::string::npos &&
           value.find('\r') == std::string::npos;
}

std::uint64_t next_power_of_two_u64(std::uint64_t value) {
    if (value == 0 || value > (std::uint64_t(1) << 62))
        throw std::invalid_argument("masked Hyrax logical extent is invalid");
    std::uint64_t result = 1;
    while (result < value) result <<= 1;
    return result;
}

void validate_masked_view(const HyraxMleBasisView &view) {
    if (view.point.size() >= 63 ||
        view.full_domain != (std::uint64_t(1) << view.point.size()) ||
        view.logical_length == 0 || view.logical_length > view.full_domain ||
        view.active_domain != next_power_of_two_u64(view.logical_length) ||
        view.active_domain > view.full_domain ||
        view.active_domain > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max() / sizeof(G1)) ||
        view.active_domain > static_cast<std::uint64_t>(
            std::numeric_limits<std::size_t>::max() / sizeof(Fr)) ||
        view.generators.size != view.full_domain || !view.generators.at ||
        (view.generators.scheme != HyraxGeneratorProviderScheme::OpaqueProvider &&
         view.generators.scheme != HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1) ||
        !safe_open_text(view.role, 128) || !safe_open_text(view.basis_id, 128) ||
        !safe_open_text(view.generators.canonical_id, 128))
        throw std::invalid_argument("masked Hyrax canonical basis view is invalid");
}

Fr mle_coefficient_at(const std::vector<Fr> &point, std::uint64_t index) {
    Fr coefficient(1);
    for (std::size_t bit = 0; bit < point.size(); ++bit)
        coefficient *= ((index >> bit) & 1) ? point[bit] : Fr(1) - point[bit];
    return coefficient;
}

struct PreparedMaskedView {
    zkgpt::NativeVector<G1> generators;
    zkgpt::NativeFieldVector coefficients;
    std::string generator_set_digest;
};

template<class ScalarAt>
G1 masked_provider_commit(const zkgpt::NativeVector<G1> &generators, std::uint64_t count, ScalarAt scalar_at, std::uint64_t generator_offset = 0) {
    if (count == 0 || generator_offset > generators.size() || count > generators.size() - generator_offset)
        throw std::invalid_argument("masked provider commitment length is invalid");
    constexpr std::uint64_t chunk_limit = 1ULL << 16;
    const std::size_t chunk_size = static_cast<std::size_t>(
        std::min(count, chunk_limit));
    zkgpt::NativeFieldVector scalars(chunk_size);
    G1 result;
    result.clear();
    for (std::uint64_t offset = 0; offset < count; offset += chunk_limit) {
        const std::uint64_t actual = std::min(chunk_limit, count - offset);
        for (std::uint64_t index = 0; index < actual; ++index)
            scalars[static_cast<std::size_t>(index)] = scalar_at(offset + index);
        result += perdersen_commit(const_cast<G1 *>(generators.data() + static_cast<std::size_t>(generator_offset + offset)), scalars.data(), static_cast<int>(actual));
    }
    return result;
}

PreparedMaskedView prepare_masked_view(const HyraxMleBasisView &view) {
    validate_masked_view(view);
    const std::size_t active = static_cast<std::size_t>(view.active_domain);
    PreparedMaskedView prepared{zkgpt::NativeVector<G1>(active),
                                zkgpt::NativeFieldVector(active), {}};
    zkgpt::native_storage::native_for_each_equality_weight(view.point, active, [&prepared](std::size_t index, const Fr &coefficient) { prepared.coefficients[index] = coefficient; });
    zkgpt::NativeStreamingDigest digest("zkgpt-hyrax-masked-canonical-generator-set-v2");
    digest.u64(view.full_domain);
    digest.u64(view.logical_length);
    digest.u64(view.active_domain);
    digest.u64(static_cast<std::uint64_t>(view.generators.scheme));
    digest.string(view.role);
    digest.string(view.basis_id);
    digest.string(view.generators.canonical_id);
    for (std::uint64_t index = 0; index < view.active_domain; ++index) {
        G1 generator = view.generators.at(index);
        if (!valid_point(generator, true))
            throw std::invalid_argument("masked Hyrax canonical generator is invalid");
        prepared.generators[static_cast<std::size_t>(index)] = generator;
        if (view.generators.scheme == HyraxGeneratorProviderScheme::OpaqueProvider) {
            digest.u64(index);
            digest.string(generator.serializeToHexStr());
        }
    }
    prepared.generator_set_digest = digest.finish();
    return prepared;
}

std::string masked_view_payload(const char *marker, const HyraxMleBasisView &view,
                                const std::string &generator_set_digest,
                                const G1 *commitment, const G1 *H, const G1 *J,
                                const G1 *Q) {
    std::ostringstream payload;
    payload << marker;
    payload << '|' << static_cast<std::uint64_t>(view.generators.scheme);
    if (commitment != nullptr && H != nullptr && J != nullptr)
        payload << '|' << point_encoding(*commitment) << '|' << point_encoding(*H)
                << '|' << point_encoding(*J);
    if (Q != nullptr) payload << '|' << point_encoding(*Q);
    payload << '|' << view.basis_id << '|' << view.role << '|'
            << view.generators.canonical_id << '|' << view.full_domain << '|'
            << view.logical_length << '|' << view.active_domain << '|'
            << view.point.size();
    for (const Fr &coordinate : view.point)
        payload << '|' << coordinate.getStr(16);
    payload << '|' << generator_set_digest;
    return payload.str();
}

void bullet_reduce_masked_response(
    G1 gamma, zkgpt::NativeVector<G1> &generators,
    zkgpt::NativeFieldVector &basis, zkgpt::NativeFieldVector &response,
    const G1 &evaluation_base, Fr evaluation, NativeProofSession &session,
    const std::string &label) {
    std::size_t size = response.size();
    std::size_t round = 0;
    while (size > 1) {
        const std::size_t half = size / 2;
        Fr cross_left(0), cross_right(0);
        for (std::size_t index = 0; index < half; ++index) {
            cross_left += response[index] * basis[index + half];
            cross_right += response[index + half] * basis[index];
        }
        const G1 gamma_minus1 = evaluation_base * cross_left + masked_provider_commit(generators, half, [&response](std::uint64_t index) { return response[static_cast<std::size_t>(index)]; }, half);
        const G1 gamma_1 = evaluation_base * cross_right +
            masked_provider_commit(generators, half,
                [&response, half](std::uint64_t index) {
                    return response[static_cast<std::size_t>(half + index)];
                });
        absorb_fold(session, label + "/fold/" + std::to_string(round),
                    gamma_minus1, gamma_1, cross_left, cross_right);
        const Fr challenge = session.challenge(label + "/c/" + std::to_string(round));
        if (challenge.isZero())
            throw std::runtime_error("masked response IPA challenge is zero");
        Fr inverse;
        Fr::inv(inverse, challenge);
        gamma = gamma_minus1 * challenge * challenge +
                gamma_1 * inverse * inverse + gamma;
        evaluation = challenge * challenge * cross_left +
                     inverse * inverse * cross_right + evaluation;
        for (std::size_t index = 0; index < half; ++index) {
            basis[index] = basis[index] * inverse + basis[index + half] * challenge;
            generators[index] = generators[index] * inverse +
                                generators[index + half] * challenge;
            response[index] = challenge * response[index] +
                              inverse * response[index + half];
        }
        size = half;
        ++round;
    }
    if (size != 1 || evaluation != response[0] * basis[0] ||
        gamma != generators[0] * response[0] + evaluation_base * evaluation)
        throw std::runtime_error("masked response IPA terminal relation failed");
    std::ostringstream terminal;
    terminal << point_encoding(gamma) << '|' << point_encoding(generators[0])
             << '|' << basis[0].getStr(16) << '|' << response[0].getStr(16)
             << '|' << evaluation.getStr(16);
    session.absorb(label + "/terminal", terminal.str());
}

bool verify_masked_response_ipa(
    G1 initial_gamma, zkgpt::NativeVector<G1> &generators,
    zkgpt::NativeFieldVector &basis,
    const G1 &evaluation_base, const NativeProofSession &session,
    std::size_t &cursor, const std::string &label, std::string *reason) {
    std::size_t size = generators.size();
    std::size_t round_count = 0;
    for (std::size_t count = size; count > 1; count >>= 1) ++round_count;
    G1 gamma = initial_gamma;
    for (std::size_t round = 0; round < round_count; ++round) {
        const NativeProofMessage *fold = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data,
            label + "/fold/" + std::to_string(round), reason);
        const NativeProofMessage *challenge_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Challenge,
            "challenge/" + label + "/c/" + std::to_string(round), reason);
        if (fold == nullptr || challenge_message == nullptr) return false;
        std::vector<std::string> parts;
        G1 gamma_minus1, gamma_1;
        Fr challenge;
        if (!split_open_payload(fold->payload, 2, parts) ||
            !decode_open_point(parts[0], gamma_minus1) ||
            !decode_open_point(parts[1], gamma_1) ||
            !decode_open_field(challenge_message->payload, challenge) ||
            challenge.isZero()) {
            set_open_reason(reason, "malformed masked response IPA fold");
            return false;
        }
        Fr inverse;
        Fr::inv(inverse, challenge);
        gamma = gamma_minus1 * challenge * challenge +
                gamma_1 * inverse * inverse + gamma;
        const std::size_t half = size / 2;
        for (std::size_t index = 0; index < half; ++index) {
            basis[index] = basis[index] * inverse + basis[index + half] * challenge;
            generators[index] = generators[index] * inverse +
                                generators[index + half] * challenge;
        }
        size = half;
    }
    const NativeProofMessage *terminal = consume_masked_message(
        session, cursor, NativeProofMessageKind::Data, label + "/terminal", reason);
    if (terminal == nullptr) return false;
    std::vector<std::string> terminal_parts;
    G1 terminal_gamma, terminal_generator;
    Fr terminal_basis, terminal_response, terminal_evaluation;
    if (!split_open_payload(terminal->payload, 5, terminal_parts) ||
        !decode_open_point(terminal_parts[0], terminal_gamma) ||
        !decode_open_point(terminal_parts[1], terminal_generator) ||
        !decode_open_field(terminal_parts[2], terminal_basis) ||
        !decode_open_field(terminal_parts[3], terminal_response) ||
        !decode_open_field(terminal_parts[4], terminal_evaluation) ||
        size != 1 || terminal_gamma != gamma ||
        terminal_generator != generators[0] || terminal_basis != basis[0] ||
        terminal_evaluation != terminal_response * terminal_basis ||
        terminal_gamma != terminal_generator * terminal_response +
                              evaluation_base * terminal_evaluation) {
        set_open_reason(reason, "masked response IPA terminal relation failed");
        return false;
    }
    return true;
}

} // namespace

Fr hyrax_mle_basis_at(const HyraxMleBasisView &view,
                      std::uint64_t canonical_index) {
    validate_masked_view(view);
    if (canonical_index >= view.active_domain)
        throw std::out_of_range("masked Hyrax active coordinate is out of range");
    return mle_coefficient_at(view.point, canonical_index);
}

std::string hyrax_generator_set_digest(const HyraxMleBasisView &view) {
    validate_masked_view(view);
    zkgpt::NativeStreamingDigest digest("zkgpt-hyrax-masked-canonical-generator-set-v2");
    digest.u64(view.full_domain);
    digest.u64(view.logical_length);
    digest.u64(view.active_domain);
    digest.u64(static_cast<std::uint64_t>(view.generators.scheme));
    digest.string(view.role);
    digest.string(view.basis_id);
    digest.string(view.generators.canonical_id);
    if (view.generators.scheme == HyraxGeneratorProviderScheme::OpaqueProvider) {
        for (std::uint64_t index = 0; index < view.active_domain; ++index) {
            const G1 generator = view.generators.at(index);
            if (!valid_point(generator, true))
                throw std::invalid_argument("masked Hyrax opaque generator is invalid");
            digest.u64(index);
            digest.string(generator.serializeToHexStr());
        }
    }
    return digest.finish();
}

template<class Values> bool make_mle_evaluation_opening_impl(
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    const std::vector<Fr> &point, const Values &values,
    zkgpt::native_pedersen::ScalarOpening &opening, std::string *reason) {
    try {
        if (!scalar_basis.valid(reason) || values.empty() ||
            !power_of_two_size(values.size()) ||
            point.size() >= sizeof(std::size_t) * 8 ||
            values.size() != (std::size_t(1) << point.size()))
            throw std::invalid_argument("MLE scalar evaluation dimensions are invalid");
        const std::vector<Fr> basis = mle_basis_from_point(point);
        Fr evaluation = Fr(0);
        for (std::size_t index = 0; index < values.size(); ++index)
            evaluation += basis[index] * values[index];
        opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, evaluation);
        return true;
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool make_mle_evaluation_opening(const zkgpt::native_pedersen::PedersenBasis& basis, const std::vector<Fr>& point, const std::vector<Fr>& values, zkgpt::native_pedersen::ScalarOpening& opening, std::string* reason) { return make_mle_evaluation_opening_impl(basis, point, values, opening, reason); }
bool make_mle_evaluation_opening(const zkgpt::native_pedersen::PedersenBasis& basis, const std::vector<Fr>& point, const zkgpt::NativeFieldVector& values, zkgpt::native_pedersen::ScalarOpening& opening, std::string* reason) { return make_mle_evaluation_opening_impl(basis, point, values, opening, reason); }

template<class Values> PedersenMleCommitment pedersen_commit_mle_impl(
    const std::vector<G1> &generators, const Values &values,
    const std::string &basis_id) {
    if (generators.empty() || generators.size() != values.size() ||
        !power_of_two_size(values.size()) || values.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("Pedersen MLE commitment dimensions are invalid");
    const G1 H = derive_pedersen_blinding_generator(basis_id);
    for (const G1 &generator : generators)
        if (!valid_point(generator, true))
            throw std::invalid_argument("Pedersen MLE commitment basis is invalid");
    Fr rho;
    rho.setByCSPRNG();
    PedersenMleCommitment result;
    result.commitment = perdersen_commit(
        const_cast<G1 *>(generators.data()), const_cast<Fr *>(values.data()),
        static_cast<int>(values.size()));
    result.commitment += H * rho;
    result.blinding_generator = H;
    result.blinding = rho;
    result.basis_id = basis_id;
    return result;
}

PedersenMleCommitment pedersen_commit_mle(const std::vector<G1>& generators, const std::vector<Fr>& values, const std::string& basis_id) { return pedersen_commit_mle_impl(generators, values, basis_id); }
PedersenMleCommitment pedersen_commit_mle(const std::vector<G1>& generators, const zkgpt::NativeFieldVector& values, const std::string& basis_id) { return pedersen_commit_mle_impl(generators, values, basis_id); }

PedersenMleCommitment pedersen_commit_mle(const HyraxMleBasisView &view,
                                         const HyraxScalarProvider &values) {
    PreparedMaskedView prepared = prepare_masked_view(view);
    if (!values.at || values.size != view.full_domain)
        throw std::invalid_argument("provider Pedersen MLE values are invalid");
    const G1 H = derive_pedersen_blinding_generator(view.basis_id);
    Fr rho;
    rho.setByCSPRNG();
    PedersenMleCommitment result;
    result.commitment = masked_provider_commit(
        prepared.generators, view.active_domain,
        [&values, &view](std::uint64_t index) {
            const Fr value = values.at(index);
            if (index >= view.logical_length && !value.isZero())
                throw std::invalid_argument("provider Pedersen MLE public zero extension is nonzero");
            return value;
        }) + H * rho;
    result.blinding_generator = H;
    result.blinding = rho;
    result.basis_id = view.basis_id;
    return result;
}

static bool open_masked_mle_provider_impl(
    const PedersenMleCommitment &commitment, const HyraxMleBasisView &view,
    const HyraxScalarProvider &values, const Fr *forced_masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        validate_masked_view(view);
        if (label.empty() || commitment.basis_id != view.basis_id ||
            !values.at || values.size != view.full_domain ||
            !valid_point(commitment.blinding_generator, true) ||
            commitment.blinding_generator != derive_pedersen_blinding_generator(view.basis_id))
            throw std::invalid_argument("masked Hyrax provider opening dimensions are invalid");
        if (enforce_statement_basis &&
            view.generators.canonical_id != session.statement().basis_digest)
            throw std::invalid_argument("masked Hyrax provider identity does not match the statement");

        PreparedMaskedView prepared = prepare_masked_view(view);
        const G1 H = commitment.blinding_generator;
        const G1 J = mask_identity.empty()
            ? derive_pedersen_evaluation_generator(view.basis_id, label)
            : derive_shared_pedersen_evaluation_generator(mask_identity);
        const G1 Q = derive_masked_response_ipa_base(view.basis_id);
        if (J == H || Q == H || Q == J)
            throw std::invalid_argument("masked Hyrax auxiliary generators are not independent");
        if (scalar_basis != nullptr && (!scalar_basis->valid() || Q == scalar_basis->G || Q == scalar_basis->H))
            throw std::invalid_argument("masked response IPA base overlaps the scalar binding basis");
        for (const G1 &generator : prepared.generators)
            if (generator == H || generator == J || generator == Q)
                throw std::invalid_argument("masked Hyrax generator domain separation failed");

        const std::size_t active = static_cast<std::size_t>(view.active_domain);
        const G1 expected_commitment = masked_provider_commit(
            prepared.generators, view.active_domain,
            [&values, &view](std::uint64_t index) {
                const Fr value = values.at(index);
                if (index >= view.logical_length && !value.isZero())
                    throw std::invalid_argument("masked Hyrax public zero extension is nonzero");
                return value;
            }) + H * commitment.blinding;
        if (expected_commitment != commitment.commitment)
            throw std::invalid_argument("masked Hyrax commitment does not bind the active witness");

        Fr raw_evaluation(0);
        for (std::size_t index = 0; index < active; ++index)
            raw_evaluation += prepared.coefficients[index] * values.at(index);
        Fr mu;
        if (forced_masked_evaluation != nullptr)
            mu = *forced_masked_evaluation - raw_evaluation;
        else
            mu.setByCSPRNG();
        opening = MaskedMleOpening{};
        opening.evaluation_mask_commitment = J * mu;
        opening.masked_evaluation = raw_evaluation + mu;
        if (scalar_basis == nullptr &&
            (forced_masked_opening != nullptr || forced_raw_opening != nullptr))
            throw std::invalid_argument("masked scalar opening requires a scalar basis");
        if (scalar_basis != nullptr) {
            if (!scalar_basis->valid())
                throw std::invalid_argument("masked scalar binding basis is invalid");
            opening.has_scalar_binding = true;
            opening.scalar_basis = *scalar_basis;
            if (forced_raw_opening != nullptr) {
                if (!forced_raw_opening->valid(*scalar_basis) ||
                    forced_raw_opening->value != raw_evaluation)
                    throw std::invalid_argument("forced raw scalar opening does not match the MLE evaluation");
                opening.raw_evaluation_opening = *forced_raw_opening;
            } else {
                opening.raw_evaluation_opening =
                    zkgpt::native_pedersen::commit_fresh(*scalar_basis, raw_evaluation);
            }
            if (forced_masked_opening != nullptr) {
                if (!forced_masked_opening->valid(*scalar_basis) ||
                    forced_masked_opening->value != opening.masked_evaluation)
                    throw std::invalid_argument("forced masked scalar opening does not match the claim");
                opening.masked_evaluation_opening = *forced_masked_opening;
            } else {
                opening.masked_evaluation_opening =
                    zkgpt::native_pedersen::commit_fresh(*scalar_basis, opening.masked_evaluation);
            }
        }

        zkgpt::NativeFieldVector random_response(active);
        for (Fr &value : random_response) value.setByCSPRNG();
        Fr random_rho, random_mu;
        random_rho.setByCSPRNG();
        random_mu.setByCSPRNG();
        const G1 announce_commitment = masked_provider_commit(
            prepared.generators, view.active_domain,
            [&random_response](std::uint64_t index) {
                return random_response[static_cast<std::size_t>(index)];
            }) + H * random_rho;
        const G1 announce_mask = J * random_mu;
        Fr announce_eval = random_mu;
        for (std::size_t index = 0; index < active; ++index)
            announce_eval += prepared.coefficients[index] * random_response[index];
        G1 raw_announce, masked_announce;
        Fr random_raw_blinding, random_masked_blinding;
        if (opening.has_scalar_binding) {
            const auto &sb = opening.scalar_basis;
            random_raw_blinding.setByCSPRNG();
            random_masked_blinding.setByCSPRNG();
            raw_announce = sb.G * (announce_eval - random_mu) +
                           sb.H * random_raw_blinding;
            masked_announce = sb.G * announce_eval + sb.H * random_masked_blinding;
        }

        session.absorb(label + "/basis",
                       masked_view_payload("zkgpt-hyrax-masked-response-ipa-v2",
                                           view, prepared.generator_set_digest,
                                           &commitment.commitment, &H, &J, nullptr));
        session.absorb(label + "/claim",
                       masked_claim_payload(opening.evaluation_mask_commitment,
                                            opening.masked_evaluation,
                                            opening.has_scalar_binding
                                                ? &opening.raw_evaluation_opening : nullptr,
                                            opening.has_scalar_binding
                                                ? &opening.masked_evaluation_opening : nullptr));
        session.absorb(label + "/announce",
                       masked_announce_payload(announce_commitment, announce_mask,
                                               announce_eval,
                                               opening.has_scalar_binding ? &raw_announce : nullptr,
                                               opening.has_scalar_binding ? &masked_announce : nullptr));
        const Fr challenge = session.challenge(label + "/c");
        if (challenge.isZero())
            throw std::runtime_error("masked Hyrax outer challenge is zero");
        const Fr response_rho = random_rho + challenge * commitment.blinding;
        const Fr response_mu = random_mu + challenge * mu;
        for (std::size_t index = 0; index < active; ++index)
            random_response[index] += challenge * values.at(index);
        std::vector<Fr> scalar_response;
        scalar_response.reserve(opening.has_scalar_binding ? 4 : 2);
        scalar_response.push_back(response_rho);
        scalar_response.push_back(response_mu);
        if (opening.has_scalar_binding) {
            scalar_response.push_back(random_raw_blinding + challenge *
                                      opening.raw_evaluation_opening.blinding);
            scalar_response.push_back(random_masked_blinding + challenge *
                                      opening.masked_evaluation_opening.blinding);
        }
        session.absorb_fields(label + "/response", scalar_response);

        const G1 response_commitment = announce_commitment + commitment.commitment * challenge;
        const G1 D = response_commitment - H * response_rho;
        const Fr response_evaluation = announce_eval +
            opening.masked_evaluation * challenge - response_mu;
        session.absorb(label + "/response-ipa/basis_ref",
                       masked_view_payload("zkgpt-hyrax-known-basis-ipa-v2",
                                           view, prepared.generator_set_digest,
                                           nullptr, nullptr, nullptr, &Q));
        session.absorb(label + "/response-ipa/evaluation",
                       point_encoding(D) + "|" + response_evaluation.getStr(16));
        const Fr randomizer = session.challenge(
            label + "/response-ipa/evaluation-base");
        if (randomizer.isZero())
            throw std::runtime_error("masked response IPA evaluation-base challenge is zero");
        const G1 U = Q * randomizer;
        if (!valid_point(U, true) || U == H || U == J)
            throw std::runtime_error("masked response IPA randomized base is invalid");
        if (scalar_basis != nullptr && (U == scalar_basis->G || U == scalar_basis->H))
            throw std::runtime_error("masked response IPA randomized base overlaps the scalar binding basis");
        for (const G1 &generator : prepared.generators)
            if (generator == U)
                throw std::runtime_error("masked response IPA base collides with a canonical generator");
        bullet_reduce_masked_response(
            D + U * response_evaluation, prepared.generators,
            prepared.coefficients, random_response, U, response_evaluation,
            session, label + "/response-ipa");
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool open_masked_mle_with_session(
    const PedersenMleCommitment &commitment, const HyraxMleBasisView &view,
    const HyraxScalarProvider &values, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return open_masked_mle_provider_impl(commitment, view, values, nullptr,
                                         session, label, opening, reason,
                                         enforce_statement_basis, mask_identity,
                                         scalar_basis, forced_masked_opening,
                                         forced_raw_opening);
}

bool open_masked_mle_with_claim(
    const PedersenMleCommitment &commitment, const HyraxMleBasisView &view,
    const HyraxScalarProvider &values, const Fr &masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return open_masked_mle_provider_impl(commitment, view, values,
                                         &masked_evaluation, session, label,
                                         opening, reason, enforce_statement_basis,
                                         mask_identity, scalar_basis,
                                         forced_masked_opening, forced_raw_opening);
}

template<class Values>
HyraxMleBasisView legacy_masked_view(const std::vector<G1> &generators,
                                    const std::vector<Fr> &point,
                                    const Values &values,
                                    const PedersenMleCommitment &commitment,
                                    NativeProofSession &session) {
    HyraxMleBasisView view;
    view.full_domain = generators.size();
    view.logical_length = values.size();
    view.active_domain = view.full_domain;
    view.role = "legacy-masked-mle";
    view.basis_id = commitment.basis_id;
    view.point = point;
    view.generators.size = generators.size();
    view.generators.canonical_id = session.statement().basis_digest;
    view.generators.at = [&generators](std::uint64_t index) {
        return generators.at(static_cast<std::size_t>(index));
    };
    return view;
}

template<class Values>
HyraxScalarProvider legacy_masked_values(const Values &values) {
    HyraxScalarProvider provider;
    provider.size = values.size();
    provider.at = [&values](std::uint64_t index) {
        return values.at(static_cast<std::size_t>(index));
    };
    return provider;
}

template<class Values>
bool legacy_open_masked_mle(
    const PedersenMleCommitment &commitment, const std::vector<G1> &generators,
    const std::vector<Fr> &point, const Values &values,
    const Fr *forced_masked_evaluation, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    try {
        if (generators.empty() || values.size() != generators.size() ||
            !power_of_two_size(values.size()) ||
            point.size() >= sizeof(std::size_t) * 8 ||
            (std::size_t(1) << point.size()) != values.size())
            throw std::invalid_argument("masked Hyrax MLE dimensions are invalid");
        const std::vector<Fr> basis = mle_basis_from_point(point);
        if (enforce_statement_basis &&
            session.statement().basis_digest != NativeProofStatement::basis_digest_for(
                commitment.blinding_generator, generators, basis, values.size()))
            throw std::invalid_argument("masked Hyrax statement basis mismatch");
        return open_masked_mle_provider_impl(
            commitment, legacy_masked_view(generators, point, values, commitment, session),
            legacy_masked_values(values), forced_masked_evaluation, session,
            label, opening, reason, enforce_statement_basis, mask_identity,
            scalar_basis, forced_masked_opening, forced_raw_opening);
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool open_masked_mle_with_session(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return legacy_open_masked_mle(commitment, generators, point, values,
                                  nullptr, session, label, opening, reason,
                                  enforce_statement_basis, mask_identity,
                                  scalar_basis, forced_masked_opening,
                                  forced_raw_opening);
}

bool open_masked_mle_with_session(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const zkgpt::NativeFieldVector &values, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return legacy_open_masked_mle(commitment, generators, point, values,
                                  nullptr, session, label, opening, reason,
                                  enforce_statement_basis, mask_identity,
                                  scalar_basis, forced_masked_opening,
                                  forced_raw_opening);
}

bool open_masked_mle_with_claim(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, const Fr &masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return legacy_open_masked_mle(commitment, generators, point, values,
                                  &masked_evaluation, session, label, opening,
                                  reason, enforce_statement_basis, mask_identity,
                                  scalar_basis, forced_masked_opening,
                                  forced_raw_opening);
}

bool open_masked_mle_with_claim(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const zkgpt::NativeFieldVector &values, const Fr &masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening,
    const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return legacy_open_masked_mle(commitment, generators, point, values,
                                  &masked_evaluation, session, label, opening,
                                  reason, enforce_statement_basis, mask_identity,
                                  scalar_basis, forced_masked_opening,
                                  forced_raw_opening);
}

bool verify_masked_mle_opening_with_session(
    const G1 &commitment, const G1 &evaluation_mask_commitment,
    const Fr &masked_evaluation, const HyraxMleBasisView &view,
    const NativeProofSession &session, const std::string &label,
    std::string *reason, bool enforce_statement_basis,
    bool allow_trailing_unrelated, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment,
    const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment) {
    try {
        const bool scalar_binding = scalar_basis != nullptr;
        if (scalar_binding != (expected_raw_evaluation_commitment != nullptr &&
                              expected_masked_evaluation_commitment != nullptr))
            throw std::invalid_argument("masked scalar binding arguments are incomplete");
        if (scalar_binding &&
            (!scalar_basis->valid(reason) ||
             !expected_raw_evaluation_commitment->valid(*scalar_basis, reason) ||
             !expected_masked_evaluation_commitment->valid(*scalar_basis, reason)))
            throw std::invalid_argument("masked scalar binding commitment is invalid");
        validate_masked_view(view);
        if (label.empty() || !valid_point(commitment, false) ||
            !valid_point(evaluation_mask_commitment, false) ||
            (enforce_statement_basis &&
             view.generators.canonical_id != session.statement().basis_digest))
            throw std::invalid_argument("masked Hyrax verifier context is invalid");

        PreparedMaskedView prepared = prepare_masked_view(view);
        const G1 H = derive_pedersen_blinding_generator(view.basis_id);
        const G1 J = mask_identity.empty()
            ? derive_pedersen_evaluation_generator(view.basis_id, label)
            : derive_shared_pedersen_evaluation_generator(mask_identity);
        const G1 Q = derive_masked_response_ipa_base(view.basis_id);
        if (!valid_point(H, true) || !valid_point(J, true) ||
            !valid_point(Q, true) || H == J || H == Q || J == Q)
            throw std::invalid_argument("masked Hyrax auxiliary basis is invalid");
        if (scalar_binding && (Q == scalar_basis->G || Q == scalar_basis->H))
            throw std::invalid_argument("masked response IPA base overlaps the scalar binding basis");
        for (const G1 &generator : prepared.generators)
            if (generator == H || generator == J || generator == Q)
                throw std::invalid_argument("masked Hyrax basis is not domain separated");
        if (!session.replay(reason)) return false;

        const std::vector<std::size_t> *basis_positions =
            session.message_positions(label + "/basis");
        if (basis_positions == nullptr || basis_positions->empty()) {
            set_open_reason(reason, "missing masked Hyrax basis message");
            return false;
        }
        if (basis_positions->size() != 1) {
            set_open_reason(reason, "duplicate masked Hyrax basis message");
            return false;
        }
        const std::size_t position = basis_positions->front();
        if (position >= session.messages().size()) {
            set_open_reason(reason, "masked Hyrax basis message index is out of range");
            return false;
        }
        const std::string opening_prefix = label + "/";
        const std::string challenge_prefix = "challenge/" + label + "/";
        for (std::size_t index = 0; index < position; ++index) {
            const std::string &message_label = session.messages()[index].label;
            if (has_prefix(message_label, opening_prefix) ||
                has_prefix(message_label, challenge_prefix)) {
                set_open_reason(reason, "masked Hyrax opening has a non-adjacent prelude");
                return false;
            }
        }

        std::size_t cursor = position;
        const NativeProofMessage *basis_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/basis", reason);
        if (basis_message == nullptr) return false;
        if (basis_message->payload != masked_view_payload(
                "zkgpt-hyrax-masked-response-ipa-v2", view,
                prepared.generator_set_digest, &commitment, &H, &J, nullptr)) {
            set_open_reason(reason, "masked Hyrax canonical basis header mismatch");
            return false;
        }

        const NativeProofMessage *claim_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/claim", reason);
        if (claim_message == nullptr) return false;
        std::vector<std::string> claim_parts;
        if (!split_open_payload(claim_message->payload,
                                scalar_binding ? 5 : 2, claim_parts)) {
            set_open_reason(reason, "malformed masked Hyrax claim message");
            return false;
        }
        G1 encoded_mask_commitment;
        Fr encoded_masked_evaluation;
        if (!decode_open_point(claim_parts[0], encoded_mask_commitment) ||
            !decode_open_field(claim_parts[1], encoded_masked_evaluation) ||
            encoded_mask_commitment != evaluation_mask_commitment ||
            encoded_masked_evaluation != masked_evaluation) {
            set_open_reason(reason, "masked Hyrax public claim mismatch");
            return false;
        }
        G1 encoded_raw_commitment, encoded_masked_commitment;
        if (scalar_binding &&
            (claim_parts[2] != scalar_basis->domain ||
             !decode_open_point(claim_parts[3], encoded_raw_commitment) ||
             !decode_open_point(claim_parts[4], encoded_masked_commitment) ||
             encoded_raw_commitment != expected_raw_evaluation_commitment->point ||
             encoded_masked_commitment != expected_masked_evaluation_commitment->point)) {
            set_open_reason(reason, "masked Hyrax scalar claim binding mismatch");
            return false;
        }

        const NativeProofMessage *announce_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/announce", reason);
        if (announce_message == nullptr) return false;
        std::vector<std::string> announce_parts;
        if (!split_open_payload(announce_message->payload,
                                scalar_binding ? 5 : 3, announce_parts)) {
            set_open_reason(reason, "malformed masked Hyrax announcement");
            return false;
        }
        G1 announce_commitment, announce_mask;
        Fr announce_eval;
        if (!decode_open_point(announce_parts[0], announce_commitment) ||
            !decode_open_point(announce_parts[1], announce_mask) ||
            !decode_open_field(announce_parts[2], announce_eval)) {
            set_open_reason(reason, "malformed masked Hyrax announcement fields");
            return false;
        }
        G1 raw_announce, masked_announce;
        if (scalar_binding &&
            (!decode_open_point(announce_parts[3], raw_announce) ||
             !decode_open_point(announce_parts[4], masked_announce))) {
            set_open_reason(reason, "malformed masked Hyrax scalar announcement");
            return false;
        }

        const NativeProofMessage *challenge_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Challenge,
            "challenge/" + label + "/c", reason);
        if (challenge_message == nullptr) return false;
        Fr challenge;
        if (!decode_open_field(challenge_message->payload, challenge) ||
            challenge.isZero()) {
            set_open_reason(reason, "masked Hyrax outer challenge is invalid");
            return false;
        }
        const NativeProofMessage *response_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::FieldVector,
            label + "/response", reason);
        if (response_message == nullptr) return false;
        std::vector<Fr> response;
        if (!decode_field_vector_payload(response_message->payload, response) ||
            response.size() != (scalar_binding ? 4 : 2)) {
            set_open_reason(reason, "masked Hyrax scalar response is invalid");
            return false;
        }
        const Fr &response_rho = response[0];
        const Fr &response_mu = response[1];
        if (J * response_mu !=
            announce_mask + evaluation_mask_commitment * challenge) {
            set_open_reason(reason, "masked Hyrax evaluation-mask relation failed");
            return false;
        }

        const G1 D = announce_commitment + commitment * challenge - H * response_rho;
        const Fr response_evaluation =
            announce_eval + masked_evaluation * challenge - response_mu;
        const NativeProofMessage *basis_ref_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data,
            label + "/response-ipa/basis_ref", reason);
        if (basis_ref_message == nullptr) return false;
        if (basis_ref_message->payload != masked_view_payload(
                "zkgpt-hyrax-known-basis-ipa-v2", view,
                prepared.generator_set_digest, nullptr, nullptr, nullptr, &Q)) {
            set_open_reason(reason, "masked response IPA basis reference mismatch");
            return false;
        }
        const NativeProofMessage *evaluation_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data,
            label + "/response-ipa/evaluation", reason);
        if (evaluation_message == nullptr) return false;
        if (evaluation_message->payload !=
            point_encoding(D) + "|" + response_evaluation.getStr(16)) {
            set_open_reason(reason, "masked response IPA fixed statement mismatch");
            return false;
        }
        const NativeProofMessage *base_challenge_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Challenge,
            "challenge/" + label + "/response-ipa/evaluation-base", reason);
        if (base_challenge_message == nullptr) return false;
        Fr randomizer;
        if (!decode_open_field(base_challenge_message->payload, randomizer) ||
            randomizer.isZero()) {
            set_open_reason(reason, "masked response IPA base challenge is invalid");
            return false;
        }
        const G1 U = Q * randomizer;
        if (!valid_point(U, true) || U == H || U == J) {
            set_open_reason(reason, "masked response IPA randomized base is invalid");
            return false;
        }
        if (scalar_binding && (U == scalar_basis->G || U == scalar_basis->H)) {
            set_open_reason(reason, "masked response IPA randomized base overlaps the scalar binding basis");
            return false;
        }
        for (const G1 &generator : prepared.generators)
            if (generator == U) {
                set_open_reason(reason, "masked response IPA base collides with a canonical generator");
                return false;
            }

        if (!verify_masked_response_ipa(D + U * response_evaluation,
                                        prepared.generators,
                                        prepared.coefficients, U, session,
                                        cursor, label + "/response-ipa", reason))
            return false;

        if (scalar_binding) {
            const auto &sb = *scalar_basis;
            const Fr &response_raw_blinding = response[2];
            const Fr &response_masked_blinding = response[3];
            if (sb.G * response_evaluation + sb.H * response_raw_blinding !=
                    raw_announce +
                        expected_raw_evaluation_commitment->point * challenge) {
                set_open_reason(reason, "masked Hyrax raw evaluation relation failed");
                return false;
            }
            if (sb.G * (response_evaluation + response_mu) +
                    sb.H * response_masked_blinding !=
                    masked_announce +
                        expected_masked_evaluation_commitment->point * challenge) {
                set_open_reason(reason, "masked Hyrax masked evaluation relation failed");
                return false;
            }
            if (sb.G * response_mu +
                    sb.H * (response_masked_blinding - response_raw_blinding) !=
                    (masked_announce - raw_announce) +
                        (expected_masked_evaluation_commitment->point -
                         expected_raw_evaluation_commitment->point) * challenge) {
                set_open_reason(reason, "masked Hyrax scalar mask relation failed");
                return false;
            }
        }
        return reject_masked_trailing(session, cursor, label,
                                      allow_trailing_unrelated, reason);
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}

bool verify_masked_mle_opening_with_session(
    const G1 &commitment, const G1 &evaluation_mask_commitment,
    const Fr &masked_evaluation, const std::string &basis_id,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const NativeProofSession &session, const std::string &label,
    std::string *reason, bool enforce_statement_basis,
    bool allow_trailing_unrelated, const std::string &mask_identity,
    const zkgpt::native_pedersen::PedersenBasis *scalar_basis,
    const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment,
    const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment) {
    try {
        if (generators.empty() || !power_of_two_size(generators.size()) ||
            point.size() >= sizeof(std::size_t) * 8 ||
            (std::size_t(1) << point.size()) != generators.size())
            throw std::invalid_argument("masked Hyrax verifier dimensions are invalid");
        const G1 H = derive_pedersen_blinding_generator(basis_id);
        const std::vector<Fr> basis = mle_basis_from_point(point);
        if (enforce_statement_basis &&
            session.statement().basis_digest != NativeProofStatement::basis_digest_for(
                H, generators, basis, generators.size()))
            throw std::invalid_argument("masked Hyrax verifier statement basis mismatch");
        HyraxMleBasisView view;
        view.full_domain = generators.size();
        view.logical_length = generators.size();
        view.active_domain = generators.size();
        view.role = "legacy-masked-mle";
        view.basis_id = basis_id;
        view.point = point;
        view.generators.size = generators.size();
        view.generators.canonical_id = session.statement().basis_digest;
        view.generators.scheme = HyraxGeneratorProviderScheme::OpaqueProvider;
        view.generators.at = [&generators](std::uint64_t index) {
            return generators.at(static_cast<std::size_t>(index));
        };
        return verify_masked_mle_opening_with_session(
            commitment, evaluation_mask_commitment, masked_evaluation, view,
            session, label, reason, enforce_statement_basis,
            allow_trailing_unrelated, mask_identity, scalar_basis,
            expected_raw_evaluation_commitment,
            expected_masked_evaluation_commitment);
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}

namespace {

struct HyraxTensorShape { std::size_t full_bits; std::size_t active_bits; std::size_t row_bits; std::size_t column_bits; };

std::size_t tensor_log2(std::uint64_t value) {
    if (value == 0 || (value & (value - 1)) != 0) throw std::invalid_argument("Hyrax tensor domain is not a power of two");
    std::size_t bits = 0;
    while (value > 1) { value >>= 1; ++bits; }
    return bits;
}

HyraxTensorShape validate_tensor_view(const HyraxTensorMleView &view, const std::vector<Fr> *point = nullptr) {
    if (view.schema != kHyraxTensorMleSchema || view.method != kHyraxTensorMleMethod || view.version != kHyraxTensorMleVersion || view.full_domain == 0 || view.full_domain > (std::uint64_t(1) << 62) || view.logical_length == 0 || view.logical_length > view.full_domain || !safe_open_text(view.role, 110) || !safe_open_text(view.basis_id, 128) || view.column_generators.scheme != HyraxGeneratorProviderScheme::CanonicalAbsoluteHashToCurveV1 || !view.column_generators.at || !safe_open_text(view.column_generators.canonical_id, 128))
        throw std::invalid_argument("Hyrax tensor MLE view is invalid");
    const std::uint64_t active = next_power_of_two_u64(view.logical_length);
    const std::size_t full_bits = tensor_log2(view.full_domain), active_bits = tensor_log2(active), column_bits = active_bits / 2;
    const std::uint64_t columns = std::uint64_t(1) << column_bits, rows = active / columns;
    if (active > view.full_domain || view.active_domain != active || view.rows != rows || view.columns != columns || view.column_generators.size != columns || columns > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(G1)) || rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(G1)) || rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / sizeof(Fr)))
        throw std::invalid_argument("Hyrax tensor MLE shape is invalid");
    if (point != nullptr && point->size() != full_bits) throw std::invalid_argument("Hyrax tensor full point arity is invalid");
    return HyraxTensorShape{full_bits, active_bits, active_bits - column_bits, column_bits};
}

zkgpt::NativeVector<G1> tensor_column_generators(const HyraxTensorMleView &view) {
    const G1 H = derive_pedersen_blinding_generator(view.basis_id);
    zkgpt::NativeVector<G1> generators(static_cast<std::size_t>(view.columns));
    for (std::uint64_t column = 0; column < view.columns; ++column) {
        const G1 generator = view.column_generators.at(column);
        if (!valid_point(generator, true) || generator == H) throw std::invalid_argument("Hyrax tensor canonical column generator is invalid");
        generators[static_cast<std::size_t>(column)] = generator;
    }
    return generators;
}

std::string tensor_root_digest(const HyraxTensorMleCommitment &commitment) {
    zkgpt::NativeStreamingDigest digest("zkgpt-hyrax-balanced-tensor-root-v2");
    digest.u64(commitment.version);
    digest.string(commitment.schema);
    digest.string(commitment.method);
    digest.string(commitment.role);
    digest.u64(commitment.full_domain);
    digest.u64(commitment.logical_length);
    digest.u64(commitment.active_domain);
    digest.u64(commitment.rows);
    digest.u64(commitment.columns);
    digest.string(commitment.basis_id);
    digest.u64(static_cast<std::uint64_t>(commitment.column_generator_scheme));
    digest.string(commitment.column_generator_id);
    digest.u64(commitment.row_commitments.size());
    for (std::size_t row = 0; row < commitment.row_commitments.size(); ++row) {
        digest.u64(row);
        digest.string(commitment.row_commitments[row].serializeToHexStr());
    }
    return digest.finish();
}

void validate_tensor_commitment(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment) {
    (void)validate_tensor_view(view);
    if (commitment.schema != view.schema || commitment.method != view.method || commitment.version != view.version ||
        commitment.role != view.role || commitment.basis_id != view.basis_id ||
        commitment.column_generator_scheme != view.column_generators.scheme || commitment.column_generator_id != view.column_generators.canonical_id ||
        commitment.full_domain != view.full_domain || commitment.logical_length != view.logical_length || commitment.active_domain != view.active_domain ||
        commitment.rows != view.rows || commitment.columns != view.columns || commitment.row_commitments.size() != view.rows ||
        commitment.root_digest.size() != 64)
        throw std::invalid_argument("Hyrax tensor commitment descriptor does not match trusted context");
    for (const G1 &row_commitment : commitment.row_commitments)
        if (!valid_point(row_commitment, false)) throw std::invalid_argument("Hyrax tensor row commitment is invalid");
    if (tensor_root_digest(commitment) != commitment.root_digest) throw std::invalid_argument("Hyrax tensor row commitment root mismatch");
}

std::string tensor_public_payload(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<Fr> &point) {
    std::ostringstream payload;
    payload << "zkgpt-hyrax-tensor-full-mle-opening-v2" << '|' << view.method << '|' << view.schema << '|' << view.version << '|' << view.role << '|' << view.basis_id;
    payload << '|' << view.full_domain << '|' << view.logical_length << '|' << view.active_domain << '|' << view.rows << '|' << view.columns;
    payload << '|' << static_cast<std::uint64_t>(view.column_generators.scheme) << '|' << view.column_generators.canonical_id << '|' << commitment.root_digest << '|' << point.size();
    for (const Fr &coordinate : point) payload << '|' << coordinate.getStr(16);
    return payload.str();
}

HyraxMleBasisView tensor_column_view(const HyraxTensorMleView &view, const std::vector<Fr> &full_point, const HyraxTensorShape &shape) {
    HyraxMleBasisView result;
    result.full_domain = view.columns;
    result.logical_length = view.columns;
    result.active_domain = view.columns;
    result.role = view.role + "/tensor-column";
    result.basis_id = view.basis_id;
    result.point.assign(full_point.begin(), full_point.begin() + static_cast<std::ptrdiff_t>(shape.column_bits));
    result.generators = view.column_generators;
    return result;
}

zkgpt::NativeFieldVector tensor_row_weights(const std::vector<Fr> &full_point, const HyraxTensorShape &shape, std::uint64_t rows) {
    std::vector<Fr> row_point(full_point.begin() + static_cast<std::ptrdiff_t>(shape.column_bits), full_point.begin() + static_cast<std::ptrdiff_t>(shape.active_bits));
    Fr high_zero_factor(1);
    for (std::size_t bit = shape.active_bits; bit < full_point.size(); ++bit) high_zero_factor *= Fr(1) - full_point[bit];
    zkgpt::NativeFieldVector weights(static_cast<std::size_t>(rows));
    zkgpt::native_storage::native_for_each_equality_weight(row_point, static_cast<std::size_t>(rows), [&weights, &high_zero_factor](std::size_t row, const Fr &weight) {
        weights[row] = high_zero_factor * weight;
    });
    return weights;
}

struct HyraxTensorFoldedColumn { G1 commitment; Fr blinding; zkgpt::NativeFieldVector values; };

HyraxTensorFoldedColumn tensor_fold_prover(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const HyraxTensorShape &shape, const zkgpt::NativeVector<G1> &generators, const HyraxScalarProvider &values, const std::vector<Fr> &full_point) {
    if (state.row_blindings.size() != view.rows || !values.at || values.size != view.logical_length)
        throw std::invalid_argument("Hyrax tensor prover state or value provider is invalid");
    zkgpt::NativeFieldVector weights = tensor_row_weights(full_point, shape, view.rows);
    HyraxTensorFoldedColumn folded;
    folded.commitment.clear();
    folded.blinding = Fr(0);
    folded.values.resize(static_cast<std::size_t>(view.columns));
    for (Fr &value : folded.values) value = Fr(0);
    const G1 H = derive_pedersen_blinding_generator(view.basis_id);
    for (std::uint64_t row = 0; row < view.rows; ++row) {
        const Fr rho = state.row_blindings[static_cast<std::size_t>(row)];
        const G1 expected_row = masked_provider_commit(generators, view.columns, [&values, &view, row](std::uint64_t column) {
            const std::uint64_t index = row * view.columns + column;
            return index < view.logical_length ? values.at(index) : Fr(0);
        }) + H * rho;
        if (expected_row != commitment.row_commitments[static_cast<std::size_t>(row)]) throw std::invalid_argument("Hyrax tensor row commitment does not match the prover witness");
        const Fr &weight = weights[static_cast<std::size_t>(row)];
        folded.commitment += commitment.row_commitments[static_cast<std::size_t>(row)] * weight;
        folded.blinding += rho * weight;
        for (std::uint64_t column = 0; column < view.columns; ++column) {
            const std::uint64_t index = row * view.columns + column;
            if (index < view.logical_length) folded.values[static_cast<std::size_t>(column)] += weight * values.at(index);
        }
    }
    return folded;
}

G1 tensor_fold_verifier(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorShape &shape, const std::vector<Fr> &full_point) {
    zkgpt::NativeFieldVector weights = tensor_row_weights(full_point, shape, view.rows);
    G1 folded;
    folded.clear();
    for (std::uint64_t row = 0; row < view.rows; ++row) folded += commitment.row_commitments[static_cast<std::size_t>(row)] * weights[static_cast<std::size_t>(row)];
    return folded;
}

bool verify_tensor_public_frame(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<Fr> &point, const NativeProofSession &session, const std::string &label, std::string *reason) {
    const auto *public_positions = session.message_positions(label + "/tensor/public");
    const auto *column_positions = session.message_positions(label + "/column/basis");
    if (public_positions == nullptr || public_positions->size() != 1 || column_positions == nullptr || column_positions->size() != 1) {
        set_open_reason(reason, "missing or duplicate Hyrax tensor public frame or column opening");
        return false;
    }
    const std::size_t public_position = public_positions->front(), column_position = column_positions->front();
    if (public_position >= session.messages().size() || public_position + 1 != column_position) {
        set_open_reason(reason, "Hyrax tensor public frame is not immediately bound to its column opening");
        return false;
    }
    const NativeProofMessage &message = session.messages()[public_position];
    if (message.kind != NativeProofMessageKind::Data || message.payload != tensor_public_payload(view, commitment, point)) {
        set_open_reason(reason, "Hyrax tensor public frame does not match trusted opening context");
        return false;
    }
    return true;
}

bool open_hyrax_tensor_mle_impl(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const std::vector<Fr> &full_point, const HyraxScalarProvider &values, const Fr *forced_masked_evaluation, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening, std::string *reason, const std::string &mask_identity, const zkgpt::native_pedersen::PedersenBasis *scalar_basis, const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening, const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        const HyraxTensorShape shape = validate_tensor_view(view, &full_point);
        validate_tensor_commitment(view, commitment);
        if (!safe_open_text(label, 200) || !values.at || values.size != view.logical_length) throw std::invalid_argument("Hyrax tensor opening arguments are invalid");
        const zkgpt::NativeVector<G1> generators = tensor_column_generators(view);
        HyraxTensorFoldedColumn folded = tensor_fold_prover(view, commitment, state, shape, generators, values, full_point);
        PedersenMleCommitment column_commitment{folded.commitment, derive_pedersen_blinding_generator(view.basis_id), folded.blinding, view.basis_id};
        const HyraxMleBasisView column_view = tensor_column_view(view, full_point, shape);
        const HyraxScalarProvider column_values{view.columns, [&folded](std::uint64_t index) { return folded.values.at(static_cast<std::size_t>(index)); }};
        session.absorb(label + "/tensor/public", tensor_public_payload(view, commitment, full_point));
        const bool opened = forced_masked_evaluation == nullptr
            ? open_masked_mle_with_session(column_commitment, column_view, column_values, session, label + "/column", opening, reason, false, mask_identity, scalar_basis, forced_masked_opening, forced_raw_opening)
            : open_masked_mle_with_claim(column_commitment, column_view, column_values, *forced_masked_evaluation, session, label + "/column", opening, reason, false, mask_identity, scalar_basis, forced_masked_opening, forced_raw_opening);
        if (!opened) { session.rollback(checkpoint); return false; }
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

} // namespace

bool commit_hyrax_tensor_mle(const HyraxTensorMleView &view, const HyraxScalarProvider &values, HyraxTensorMleCommitment &commitment, HyraxTensorMleProverState &state, std::string *reason) {
    try {
        (void)validate_tensor_view(view);
        if (!values.at || values.size != view.logical_length) throw std::invalid_argument("Hyrax tensor commit value provider is invalid");
        const zkgpt::NativeVector<G1> generators = tensor_column_generators(view);
        const G1 H = derive_pedersen_blinding_generator(view.basis_id);
        HyraxTensorMleCommitment next_commitment;
        next_commitment.role = view.role;
        next_commitment.basis_id = view.basis_id;
        next_commitment.column_generator_scheme = view.column_generators.scheme;
        next_commitment.column_generator_id = view.column_generators.canonical_id;
        next_commitment.full_domain = view.full_domain;
        next_commitment.logical_length = view.logical_length;
        next_commitment.active_domain = view.active_domain;
        next_commitment.rows = view.rows;
        next_commitment.columns = view.columns;
        next_commitment.row_commitments.reserve(static_cast<std::size_t>(view.rows));
        HyraxTensorMleProverState next_state;
        next_state.row_blindings.reserve(static_cast<std::size_t>(view.rows));
        for (std::uint64_t row = 0; row < view.rows; ++row) {
            Fr rho;
            rho.setByCSPRNG();
            const G1 row_commitment = masked_provider_commit(generators, view.columns, [&values, &view, row](std::uint64_t column) {
                const std::uint64_t index = row * view.columns + column;
                return index < view.logical_length ? values.at(index) : Fr(0);
            }) + H * rho;
            next_commitment.row_commitments.push_back(row_commitment);
            next_state.row_blindings.push_back(rho);
        }
        next_commitment.root_digest = tensor_root_digest(next_commitment);
        commitment = std::move(next_commitment);
        state = std::move(next_state);
        return true;
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool open_hyrax_tensor_mle_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const std::vector<Fr> &full_point, const HyraxScalarProvider &values, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening, std::string *reason, const std::string &mask_identity, const zkgpt::native_pedersen::PedersenBasis *scalar_basis, const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening, const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return open_hyrax_tensor_mle_impl(view, commitment, state, full_point, values, nullptr, session, label, opening, reason, mask_identity, scalar_basis, forced_masked_opening, forced_raw_opening);
}

bool open_hyrax_tensor_mle_with_claim(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const std::vector<Fr> &full_point, const HyraxScalarProvider &values, const Fr &masked_evaluation, NativeProofSession &session, const std::string &label, MaskedMleOpening &opening, std::string *reason, const std::string &mask_identity, const zkgpt::native_pedersen::PedersenBasis *scalar_basis, const zkgpt::native_pedersen::ScalarOpening *forced_masked_opening, const zkgpt::native_pedersen::ScalarOpening *forced_raw_opening) {
    return open_hyrax_tensor_mle_impl(view, commitment, state, full_point, values, &masked_evaluation, session, label, opening, reason, mask_identity, scalar_basis, forced_masked_opening, forced_raw_opening);
}

bool verify_hyrax_tensor_mle_opening_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<Fr> &full_point, const G1 &evaluation_mask_commitment, const Fr &masked_evaluation, const NativeProofSession &session, const std::string &label, std::string *reason, bool allow_trailing_unrelated, const std::string &mask_identity, const zkgpt::native_pedersen::PedersenBasis *scalar_basis, const zkgpt::native_pedersen::ScalarCommitment *expected_raw_evaluation_commitment, const zkgpt::native_pedersen::ScalarCommitment *expected_masked_evaluation_commitment) {
    try {
        const HyraxTensorShape shape = validate_tensor_view(view, &full_point);
        validate_tensor_commitment(view, commitment);
        if (!safe_open_text(label, 200) || !session.replay(reason)) return false;
        if (!verify_tensor_public_frame(view, commitment, full_point, session, label, reason)) return false;
        const G1 folded_commitment = tensor_fold_verifier(view, commitment, shape, full_point);
        const HyraxMleBasisView column_view = tensor_column_view(view, full_point, shape);
        return verify_masked_mle_opening_with_session(folded_commitment, evaluation_mask_commitment, masked_evaluation, column_view, session, label + "/column", reason, false, allow_trailing_unrelated, mask_identity, scalar_basis, expected_raw_evaluation_commitment, expected_masked_evaluation_commitment);
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}

namespace {

std::string tensor_tail_id(const HyraxTensorMleView &view) {
    if (view.role == "canonical/Aux") return "aux";
    if (view.role == "canonical/L_qry") return "L_qry";
    throw std::invalid_argument("Hyrax tensor zero-tail role is not trusted");
}

std::string tensor_tail_label(const HyraxTensorMleView &view) { return "tensor/" + tensor_tail_id(view) + "/tail"; }

std::string tensor_tail_public_payload(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment) {
    std::ostringstream payload;
    payload << "zkgpt-hyrax-tensor-zero-tail-product2public-v1" << '|' << view.method << '|' << view.schema << '|' << view.version << '|' << tensor_tail_id(view);
    payload << '|' << view.role << '|' << view.basis_id << '|' << view.full_domain << '|' << view.logical_length << '|' << view.active_domain;
    payload << '|' << view.rows << '|' << view.columns << '|' << static_cast<std::uint64_t>(view.column_generators.scheme) << '|' << view.column_generators.canonical_id;
    payload << '|' << commitment.root_digest << "|Product2Public|2|0|" << (view.logical_length == view.active_domain ? 1 : 0);
    return payload.str();
}

Fr tensor_weighted_tail_eval(const std::vector<Fr> &beta, const std::vector<Fr> &point, std::uint64_t logical_length) {
    if (beta.size() != point.size() || beta.size() >= 63) throw std::invalid_argument("Hyrax tensor tail evaluation point is invalid");
    const std::uint64_t domain = std::uint64_t(1) << beta.size();
    if (logical_length > domain) throw std::invalid_argument("Hyrax tensor tail logical length is invalid");
    if (logical_length == domain) return Fr(0);
    std::vector<Fr> zero(beta.size()), one(beta.size()), lower_total(beta.size() + 1, Fr(1));
    for (std::size_t bit = 0; bit < beta.size(); ++bit) {
        zero[bit] = (Fr(1) - beta[bit]) * (Fr(1) - point[bit]);
        one[bit] = beta[bit] * point[bit];
        lower_total[bit + 1] = lower_total[bit] * (zero[bit] + one[bit]);
    }
    Fr prefix(1), below(0);
    for (std::size_t cursor = beta.size(); cursor-- > 0;) {
        if (((logical_length >> cursor) & 1U) != 0) {
            below += prefix * zero[cursor] * lower_total[cursor];
            prefix *= one[cursor];
        } else {
            prefix *= zero[cursor];
        }
    }
    return lower_total.back() - below;
}

Fr tensor_tail_weight_at(const std::vector<Fr> &beta, std::uint64_t logical_length, std::uint64_t index) {
    if (beta.size() >= 63 || index >= (std::uint64_t(1) << beta.size())) throw std::out_of_range("Hyrax tensor tail coefficient index is invalid");
    if (index < logical_length) return Fr(0);
    Fr weight(1);
    for (std::size_t bit = 0; bit < beta.size(); ++bit) weight *= ((index >> bit) & 1U) ? beta[bit] : Fr(1) - beta[bit];
    return weight;
}

std::shared_ptr<const NativeMaskedPublicProvider> tensor_tail_public_provider(const std::vector<Fr> &beta, std::uint64_t logical_length) {
    if (beta.size() >= 63) throw std::invalid_argument("Hyrax tensor tail challenge dimension is invalid");
    const std::vector<Fr> fixed_beta = beta;
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = std::uint64_t(1) << beta.size();
    provider->value_at = [fixed_beta, logical_length](std::uint64_t index) { return tensor_tail_weight_at(fixed_beta, logical_length, index); };
    provider->evaluate_mle = [fixed_beta, logical_length](const std::vector<F> &point) { return tensor_weighted_tail_eval(fixed_beta, point, logical_length); };
    return provider;
}

Fr tensor_tail_block_mle(const HyraxScalarProvider &values, std::uint64_t start, std::uint64_t length, const std::vector<Fr> &point) {
    if (!values.at || length != (std::uint64_t(1) << point.size()) || start > values.size || length > values.size - start) throw std::invalid_argument("Hyrax tensor tail boundary chunk is invalid");
    Fr evaluation(0);
    zkgpt::native_storage::native_for_each_equality_weight(point, static_cast<std::size_t>(length), [&](std::size_t index, const Fr &weight) { evaluation += values.at(start + index) * weight; });
    return evaluation;
}

Fr tensor_tail_full_mle(const HyraxScalarProvider &values, std::uint64_t logical_length, std::uint64_t active_domain, const std::vector<Fr> &point) {
    if (!values.at || values.size != logical_length || active_domain != (std::uint64_t(1) << point.size())) throw std::invalid_argument("Hyrax tensor tail terminal dimensions are invalid");
    Fr evaluation(0);
    zkgpt::native_storage::native_for_each_equality_weight(point, static_cast<std::size_t>(active_domain), [&](std::size_t index, const Fr &weight) { if (static_cast<std::uint64_t>(index) < logical_length) evaluation += values.at(index) * weight; });
    return evaluation;
}

Fr tensor_tail_beta_suffix_weight(const std::vector<Fr> &beta, std::size_t first_bit, std::uint64_t suffix_index) {
    Fr weight(1);
    for (std::size_t bit = first_bit; bit < beta.size(); ++bit) weight *= ((suffix_index >> (bit - first_bit)) & 1U) ? beta[bit] : Fr(1) - beta[bit];
    return weight;
}

std::string tensor_zero_interval_id(const HyraxTensorMleView &view) {
    if (view.role != "canonical/L_qry") throw std::invalid_argument("Hyrax tensor zero-interval role is not trusted");
    return "L_qry";
}

std::string tensor_zero_interval_label(const HyraxTensorMleView &view) { return "tensor/" + tensor_zero_interval_id(view) + "/zero-gap"; }

void validate_tensor_zero_intervals(const HyraxTensorMleView &view, const std::vector<HyraxTensorZeroInterval> &intervals) {
    (void)tensor_zero_interval_id(view);
    std::uint64_t previous_end = 0;
    for (const HyraxTensorZeroInterval &interval : intervals) {
        if (interval.begin >= interval.end || interval.end > view.logical_length || interval.begin < previous_end) throw std::invalid_argument("Hyrax tensor zero interval list is invalid");
        previous_end = interval.end;
    }
}

std::string tensor_zero_interval_payload(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<HyraxTensorZeroInterval> &intervals) {
    std::ostringstream payload;
    payload << "zkgpt-hyrax-tensor-zero-interval-product2public-v1" << '|' << view.method << '|' << view.schema << '|' << view.version << '|' << tensor_zero_interval_id(view);
    payload << '|' << view.role << '|' << view.basis_id << '|' << view.full_domain << '|' << view.logical_length << '|' << view.active_domain;
    payload << '|' << view.rows << '|' << view.columns << '|' << static_cast<std::uint64_t>(view.column_generators.scheme) << '|' << view.column_generators.canonical_id;
    payload << '|' << commitment.root_digest << "|Product2Public|2|0|" << intervals.size();
    for (const HyraxTensorZeroInterval &interval : intervals) payload << '|' << interval.begin << '|' << interval.end;
    return payload.str();
}

Fr tensor_weighted_interval_prefix_eval(const std::vector<Fr> &beta, const std::vector<Fr> &point, std::size_t bits, std::uint64_t bound) {
    if (beta.size() < bits || point.size() < bits || bits >= 63 || bound > (std::uint64_t(1) << bits)) throw std::invalid_argument("Hyrax tensor interval prefix dimensions are invalid");
    if (bound == 0) return Fr(0);
    std::vector<Fr> zero(bits), one(bits), lower_total(bits + 1, Fr(1));
    for (std::size_t bit = 0; bit < bits; ++bit) {
        zero[bit] = (Fr(1) - beta[bit]) * (Fr(1) - point[bit]);
        one[bit] = beta[bit] * point[bit];
        lower_total[bit + 1] = lower_total[bit] * (zero[bit] + one[bit]);
    }
    const std::uint64_t domain = std::uint64_t(1) << bits;
    if (bound == domain) return lower_total.back();
    Fr prefix(1), below(0);
    for (std::size_t cursor = bits; cursor-- > 0;) {
        if (((bound >> cursor) & 1U) != 0) {
            below += prefix * zero[cursor] * lower_total[cursor];
            prefix *= one[cursor];
        } else {
            prefix *= zero[cursor];
        }
    }
    return below;
}

Fr tensor_weighted_interval_eval(const std::vector<Fr> &beta, const std::vector<Fr> &point, std::size_t bits, std::uint64_t begin, std::uint64_t end) {
    if (bits >= 63 || begin > end || end > (std::uint64_t(1) << bits)) throw std::invalid_argument("Hyrax tensor weighted interval is invalid");
    return tensor_weighted_interval_prefix_eval(beta, point, bits, end) - tensor_weighted_interval_prefix_eval(beta, point, bits, begin);
}

std::shared_ptr<const NativeMaskedPublicProvider> tensor_zero_interval_public_provider(const std::vector<Fr> &beta, const HyraxTensorZeroInterval &interval) {
    if (beta.size() >= 63 || interval.begin >= interval.end || interval.end > (std::uint64_t(1) << beta.size())) throw std::invalid_argument("Hyrax tensor zero-interval public provider is invalid");
    const std::vector<Fr> fixed_beta = beta;
    auto provider = std::make_shared<NativeMaskedPublicProvider>();
    provider->size = std::uint64_t(1) << beta.size();
    provider->value_at = [fixed_beta, interval](std::uint64_t index) {
        if (index >= (std::uint64_t(1) << fixed_beta.size())) throw std::out_of_range("Hyrax tensor zero-interval coefficient index is invalid");
        if (index < interval.begin || index >= interval.end) return Fr(0);
        Fr weight(1);
        for (std::size_t bit = 0; bit < fixed_beta.size(); ++bit) weight *= ((index >> bit) & 1U) ? fixed_beta[bit] : Fr(1) - fixed_beta[bit];
        return weight;
    };
    provider->evaluate_mle = [fixed_beta, interval](const std::vector<F> &point) {
        if (point.size() != fixed_beta.size()) throw std::invalid_argument("Hyrax tensor zero-interval evaluation point is invalid");
        return tensor_weighted_interval_eval(fixed_beta, point, fixed_beta.size(), interval.begin, interval.end);
    };
    return provider;
}

Fr tensor_zero_interval_block_mle(const HyraxScalarProvider &values, std::uint64_t logical_length, std::uint64_t start, std::uint64_t length, const std::vector<Fr> &point) {
    if (!values.at || values.size != logical_length || length != (std::uint64_t(1) << point.size())) throw std::invalid_argument("Hyrax tensor zero-interval boundary block is invalid");
    Fr evaluation(0);
    zkgpt::native_storage::native_for_each_equality_weight(point, static_cast<std::size_t>(length), [&](std::size_t offset, const Fr &weight) { const std::uint64_t index = start + offset; if (index < logical_length) evaluation += values.at(index) * weight; });
    return evaluation;
}

Fr tensor_zero_interval_folded_weight(const std::vector<Fr> &beta, const std::vector<Fr> &round_point, std::size_t round, const HyraxTensorZeroInterval &interval, std::uint64_t suffix_index) {
    const std::uint64_t block_size = std::uint64_t(1) << round, block_start = suffix_index * block_size;
    if (block_start >= interval.end || block_start + block_size <= interval.begin) return Fr(0);
    const std::uint64_t low_begin = interval.begin > block_start ? interval.begin - block_start : 0;
    const std::uint64_t low_end = interval.end < block_start + block_size ? interval.end - block_start : block_size;
    if (low_begin >= low_end) return Fr(0);
    Fr high_weight(1);
    for (std::size_t bit = round; bit < beta.size(); ++bit) high_weight *= ((suffix_index >> (bit - round)) & 1U) ? beta[bit] : Fr(1) - beta[bit];
    return high_weight * tensor_weighted_interval_eval(beta, round_point, round, low_begin, low_end);
}

void tensor_zero_interval_boundary_round(const HyraxScalarProvider &values, std::uint64_t logical_length, const HyraxTensorZeroInterval &interval, const std::vector<Fr> &beta, const std::vector<Fr> &round_point, std::size_t round, std::vector<F> &polynomial) {
    polynomial.assign(3, Fr(0));
    const std::uint64_t block_size = std::uint64_t(1) << round, pair_width = block_size * 2;
    std::uint64_t pairs[2];
    std::size_t pair_count = 0;
    if (interval.begin % pair_width != 0) pairs[pair_count++] = interval.begin / pair_width;
    if (interval.end % pair_width != 0) {
        const std::uint64_t pair = interval.end / pair_width;
        if (pair_count == 0 || pairs[0] != pair) pairs[pair_count++] = pair;
    }
    for (std::size_t candidate = 0; candidate < pair_count; ++candidate) {
        const std::uint64_t left = pairs[candidate] * 2, right = left + 1;
        const Fr private_left = tensor_zero_interval_block_mle(values, logical_length, left * block_size, block_size, round_point);
        const Fr private_right = tensor_zero_interval_block_mle(values, logical_length, right * block_size, block_size, round_point);
        const Fr public_left = tensor_zero_interval_folded_weight(beta, round_point, round, interval, left);
        const Fr public_right = tensor_zero_interval_folded_weight(beta, round_point, round, interval, right);
        polynomial[0] += private_left * public_left;
        polynomial[1] += private_right * public_right;
        polynomial[2] += (Fr(2) * private_right - private_left) * (Fr(2) * public_right - public_left);
    }
}

void tensor_tail_boundary_round(const Fr &private_left, const Fr &private_right, const Fr &public_left, const Fr &public_right, std::vector<F> &polynomial) {
    polynomial = {private_left * public_left, private_right * public_right, (Fr(2) * private_right - private_left) * (Fr(2) * public_right - public_left)};
}

bool tensor_tail_scalar_domain_matches(const NativeProofSession &session, const zkgpt::native_pedersen::PedersenBasis &basis, std::string *reason) {
    const std::string expected = "zkgpt/masked-sumcheck/v2/" + session.statement_digest();
    if (!basis.valid(reason) || basis.domain != expected) {
        set_open_reason(reason, "Hyrax tensor tail scalar basis is not bound to the proof statement");
        return false;
    }
    return true;
}

bool tensor_tail_parse_opening_claim(const NativeProofSession &session, std::size_t position, const std::string &label, const zkgpt::native_pedersen::PedersenBasis &basis, G1 &mask_commitment, Fr &masked_evaluation, zkgpt::native_pedersen::ScalarCommitment &raw_commitment, zkgpt::native_pedersen::ScalarCommitment &masked_commitment, std::string *reason) {
    const auto *positions = session.message_positions(label);
    if (positions == nullptr || positions->size() != 1 || positions->front() != position || position >= session.messages().size()) {
        set_open_reason(reason, "Hyrax tensor tail terminal claim is missing, duplicated, or out of order");
        return false;
    }
    const NativeProofMessage &message = session.messages()[position];
    std::vector<std::string> fields;
    if (message.kind != NativeProofMessageKind::Data || !split_open_payload(message.payload, 5, fields) || fields[2] != basis.domain || !decode_open_point(fields[0], mask_commitment) || !decode_open_field(fields[1], masked_evaluation) || !decode_open_point(fields[4], masked_commitment.point)) {
        set_open_reason(reason, "malformed Hyrax tensor tail terminal claim");
        return false;
    }
    if (!decode_open_point(fields[3], raw_commitment.point)) { set_open_reason(reason, "malformed Hyrax tensor tail raw scalar commitment"); return false; }
    raw_commitment.domain = basis.domain;
    masked_commitment.domain = basis.domain;
    return raw_commitment.valid(basis, reason) && masked_commitment.valid(basis, reason);
}

bool tensor_tail_consume_challenges(const NativeProofSession &session, std::size_t &position, const std::string &label, std::size_t count, std::vector<Fr> &beta, std::string *reason) {
    beta.clear();
    beta.reserve(count);
    for (std::size_t bit = 0; bit < count; ++bit) {
        const std::string challenge_label = "challenge/" + label + "/weight/" + std::to_string(bit);
        const auto *positions = session.message_positions(challenge_label);
        Fr challenge;
        if (positions == nullptr || positions->size() != 1 || positions->front() != position || position >= session.messages().size() || session.messages()[position].kind != NativeProofMessageKind::Challenge || session.messages()[position].label != challenge_label || !decode_open_field(session.messages()[position].payload, challenge)) {
            set_open_reason(reason, "Hyrax tensor tail challenge schedule is invalid");
            return false;
        }
        beta.push_back(challenge);
        ++position;
    }
    return true;
}

} // namespace

bool prove_hyrax_tensor_zero_tail_certificate_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const HyraxScalarProvider &values, NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<Fr> &beta, std::vector<Fr> &full_terminal_point, ::NativeMaskedSumcheckResult &sumcheck_result, MaskedMleOpening &terminal_opening, std::string *reason, std::vector<std::vector<Fr>> *unmasked_round_trace) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        const HyraxTensorShape shape = validate_tensor_view(view);
        validate_tensor_commitment(view, commitment);
        const std::string tail_label = tensor_tail_label(view), sum_label = tail_label + "/sum";
        if (exact_position != checkpoint || !values.at || values.size != view.logical_length || !tensor_tail_scalar_domain_matches(session, scalar_basis, reason)) throw std::invalid_argument("Hyrax tensor tail prover position, values, or scalar basis is invalid");
        session.absorb(tail_label + "/public", tensor_tail_public_payload(view, commitment));
        beta.clear();
        if (unmasked_round_trace != nullptr) unmasked_round_trace->clear();
        for (std::size_t bit = 0; bit < shape.active_bits; ++bit) beta.push_back(session.challenge(tail_label + "/weight/" + std::to_string(bit)));
        const auto public_provider = tensor_tail_public_provider(beta, view.logical_length);
        const auto initial_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, Fr(0));
        NativeMaskedTerminalSpec terminal_spec;
        terminal_spec.kind = NativeMaskedTerminalKind::Product2Public;
        terminal_spec.public_provider = public_provider;
        terminal_spec.has_external_binding = true;
        terminal_spec.public_initial_claim = true;
        terminal_spec.initial_opening = initial_opening;
        terminal_spec.initial_commitment = initial_opening.commitment;
        NativeMaskedSumcheckProver prover(session, sum_label, 2, Fr(0), terminal_spec, nullptr, &scalar_basis);
        Fr private_boundary(0), public_total(1), public_tail(1);
        std::vector<Fr> round_point;
        round_point.reserve(shape.active_bits);
        std::vector<F> polynomial;
        for (std::size_t round = 0; round < shape.active_bits; ++round) {
            Fr private_left(0), private_right(0), public_left(0), public_right(0);
            if (view.logical_length < view.active_domain) {
                const std::uint64_t block_size = std::uint64_t(1) << round;
                const std::uint64_t pair_index = view.logical_length >> (round + 1);
                const std::uint64_t left_index = pair_index * 2, right_index = left_index + 1;
                const bool logical_bit = ((view.logical_length >> round) & 1U) != 0;
                if (logical_bit) {
                    private_left = tensor_tail_block_mle(values, pair_index * (block_size * 2), block_size, round_point);
                    private_right = private_boundary;
                    public_right = tensor_tail_beta_suffix_weight(beta, round, right_index) * public_tail;
                } else {
                    private_left = private_boundary;
                    public_left = tensor_tail_beta_suffix_weight(beta, round, left_index) * public_tail;
                    public_right = tensor_tail_beta_suffix_weight(beta, round, right_index) * public_total;
                }
            }
            tensor_tail_boundary_round(private_left, private_right, public_left, public_right, polynomial);
            if (unmasked_round_trace != nullptr) unmasked_round_trace->push_back(polynomial);
            Fr challenge;
            if (!prover.append_round(polynomial, &challenge, reason)) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor tail initial round failed");
            if (view.logical_length < view.active_domain) {
                const bool logical_bit = ((view.logical_length >> round) & 1U) != 0;
                if (logical_bit) private_boundary = private_left * (Fr(1) - challenge) + private_right * challenge;
                else private_boundary = private_left * (Fr(1) - challenge);
                const Fr low_zero = (Fr(1) - beta[round]) * (Fr(1) - challenge), low_one = beta[round] * challenge;
                if (logical_bit) public_tail *= low_one;
                else public_tail = low_zero * public_tail + low_one * public_total;
                public_total *= low_zero + low_one;
            }
            round_point.push_back(challenge);
        }
        const Fr terminal_value = tensor_tail_full_mle(values, view.logical_length, view.active_domain, round_point);
        terminal_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(scalar_basis, terminal_value)};
        if (!prover.set_terminal_spec(terminal_spec, reason) || !prover.finish({terminal_value}, &sumcheck_result, reason)) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor tail terminal relation failed");
        if (sumcheck_result.challenges != round_point || sumcheck_result.challenges.size() != shape.active_bits || sumcheck_result.unmasked_terminal_openings.size() != 1)
            throw std::logic_error("Hyrax tensor tail sumcheck result is incomplete");
        full_terminal_point = sumcheck_result.challenges;
        full_terminal_point.resize(shape.full_bits, Fr(0));
        if (!open_hyrax_tensor_mle_with_session(view, commitment, state, full_terminal_point, values, session, tail_label + "/opening", terminal_opening, reason, "", &scalar_basis, nullptr, &sumcheck_result.unmasked_terminal_openings[0])) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor tail terminal opening failed");
        exact_position = session.checkpoint();
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        exact_position = checkpoint;
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool verify_hyrax_tensor_zero_tail_certificate_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<Fr> &beta, std::vector<Fr> &full_terminal_point, ::NativeMaskedSumcheckResult &sumcheck_result, std::string *reason) {
    try {
        const HyraxTensorShape shape = validate_tensor_view(view);
        validate_tensor_commitment(view, commitment);
        const std::string tail_label = tensor_tail_label(view), sum_label = tail_label + "/sum";
        if (!session.replay(reason) || !tensor_tail_scalar_domain_matches(session, scalar_basis, reason) || exact_position >= session.messages().size()) return false;
        const auto *header_positions = session.message_positions(tail_label + "/public");
        const NativeProofMessage &header = session.messages()[exact_position];
        if (header_positions == nullptr || header_positions->size() != 1 || header_positions->front() != exact_position || header.kind != NativeProofMessageKind::Data || header.label != tail_label + "/public" || header.payload != tensor_tail_public_payload(view, commitment)) {
            set_open_reason(reason, "Hyrax tensor tail public context is missing or mismatched");
            return false;
        }
        ++exact_position;
        if (!tensor_tail_consume_challenges(session, exact_position, tail_label, shape.active_bits, beta, reason)) return false;
        NativeMaskedTerminalSpec terminal_spec;
        terminal_spec.kind = NativeMaskedTerminalKind::Product2Public;
        terminal_spec.public_provider = tensor_tail_public_provider(beta, view.logical_length);
        terminal_spec.has_external_binding = true;
        terminal_spec.public_initial_claim = true;
        if (exact_position >= session.messages().size() || session.messages()[exact_position].kind != NativeProofMessageKind::Data || session.messages()[exact_position].label != sum_label + "/initial/mode" || session.messages()[exact_position].payload != "zero-mask-v2") {
            set_open_reason(reason, "Hyrax tensor tail sumcheck does not immediately follow its public challenges");
            return false;
        }
        if (!native_masked_opening_commitment_at(session, scalar_basis, exact_position + 1, sum_label + "/initial/opening", terminal_spec.initial_commitment, reason)) return false;
        const auto *mode_positions = session.message_positions(sum_label + "/initial/mode");
        if (mode_positions == nullptr || mode_positions->size() != 1 || mode_positions->front() != exact_position ||
            !native_masked_sumcheck_verify(session, sum_label, 2, shape.active_bits, Fr(0), terminal_spec, &sumcheck_result, reason)) return false;
        if (sumcheck_result.challenges.size() != shape.active_bits || sumcheck_result.unmasked_terminal_commitments.size() != 1 || sumcheck_result.transcript_end <= exact_position) {
            set_open_reason(reason, "Hyrax tensor tail sumcheck result is incomplete");
            return false;
        }
        exact_position = sumcheck_result.transcript_end;
        full_terminal_point = sumcheck_result.challenges;
        full_terminal_point.resize(shape.full_bits, Fr(0));
        const std::string opening_label = tail_label + "/opening";
        const auto *tensor_public_positions = session.message_positions(opening_label + "/tensor/public");
        const auto *claim_positions = session.message_positions(opening_label + "/column/claim");
        if (tensor_public_positions == nullptr || tensor_public_positions->size() != 1 || tensor_public_positions->front() != exact_position || claim_positions == nullptr || claim_positions->size() != 1 || claim_positions->front() != exact_position + 2) {
            set_open_reason(reason, "Hyrax tensor tail opening does not immediately follow its sumcheck");
            return false;
        }
        G1 mask_commitment;
        Fr masked_evaluation;
        zkgpt::native_pedersen::ScalarCommitment raw_commitment, masked_commitment;
        if (!tensor_tail_parse_opening_claim(session, claim_positions->front(), opening_label + "/column/claim", scalar_basis, mask_commitment, masked_evaluation, raw_commitment, masked_commitment, reason)) return false;
        const auto &expected_raw = sumcheck_result.unmasked_terminal_commitments[0];
        if (raw_commitment.domain != expected_raw.domain || raw_commitment.point != expected_raw.point) {
            set_open_reason(reason, "Hyrax tensor tail opening raw claim differs from Product2Public terminal");
            return false;
        }
        if (!verify_hyrax_tensor_mle_opening_with_session(view, commitment, full_terminal_point, mask_commitment, masked_evaluation, session, opening_label, reason, true, "", &scalar_basis, &expected_raw, &masked_commitment)) return false;
        const std::string terminal_label = opening_label + "/column/response-ipa/terminal";
        const auto *terminal_positions = session.message_positions(terminal_label);
        if (terminal_positions == nullptr || terminal_positions->size() != 1 || terminal_positions->front() < exact_position) {
            set_open_reason(reason, "Hyrax tensor tail response-IPA terminal is missing");
            return false;
        }
        exact_position = terminal_positions->front() + 1;
        return true;
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}

bool prove_hyrax_tensor_zero_interval_certificates_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const HyraxTensorMleProverState &state, const HyraxScalarProvider &values, const std::vector<HyraxTensorZeroInterval> &trusted_intervals, NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<HyraxTensorZeroIntervalCertificate> &certificates, std::string *reason, std::vector<std::vector<std::vector<Fr>>> *unmasked_round_traces) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        const HyraxTensorShape shape = validate_tensor_view(view);
        validate_tensor_commitment(view, commitment);
        validate_tensor_zero_intervals(view, trusted_intervals);
        if (exact_position != checkpoint || !values.at || values.size != view.logical_length || !tensor_tail_scalar_domain_matches(session, scalar_basis, reason)) throw std::invalid_argument("Hyrax tensor zero-interval prover position, values, or scalar basis is invalid");
        const std::string base_label = tensor_zero_interval_label(view);
        session.absorb(base_label + "/public", tensor_zero_interval_payload(view, commitment, trusted_intervals));
        std::vector<HyraxTensorZeroIntervalCertificate> pending_certificates;
        pending_certificates.reserve(trusted_intervals.size());
        if (unmasked_round_traces != nullptr) unmasked_round_traces->assign(trusted_intervals.size(), {});
        for (std::size_t interval_index = 0; interval_index < trusted_intervals.size(); ++interval_index) {
            const HyraxTensorZeroInterval &interval = trusted_intervals[interval_index];
            const std::string interval_label = base_label + "/" + std::to_string(interval_index), sum_label = interval_label + "/sum";
            HyraxTensorZeroIntervalCertificate certificate;
            certificate.begin = interval.begin;
            certificate.end = interval.end;
            certificate.beta.reserve(shape.active_bits);
            for (std::size_t bit = 0; bit < shape.active_bits; ++bit) certificate.beta.push_back(session.challenge(interval_label + "/weight/" + std::to_string(bit)));
            NativeMaskedTerminalSpec terminal_spec;
            terminal_spec.kind = NativeMaskedTerminalKind::Product2Public;
            terminal_spec.public_provider = tensor_zero_interval_public_provider(certificate.beta, interval);
            terminal_spec.has_external_binding = true;
            terminal_spec.public_initial_claim = true;
            terminal_spec.initial_opening = zkgpt::native_pedersen::commit_fresh(scalar_basis, Fr(0));
            terminal_spec.initial_commitment = terminal_spec.initial_opening.commitment;
            NativeMaskedSumcheckProver prover(session, sum_label, 2, Fr(0), terminal_spec, nullptr, &scalar_basis);
            std::vector<Fr> round_point;
            round_point.reserve(shape.active_bits);
            std::vector<F> polynomial;
            for (std::size_t round = 0; round < shape.active_bits; ++round) {
                tensor_zero_interval_boundary_round(values, view.logical_length, interval, certificate.beta, round_point, round, polynomial);
                if (unmasked_round_traces != nullptr) (*unmasked_round_traces)[interval_index].push_back(polynomial);
                Fr challenge;
                if (!prover.append_round(polynomial, &challenge, reason)) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor zero-interval sumcheck round failed");
                round_point.push_back(challenge);
            }
            const Fr terminal_value = tensor_tail_full_mle(values, view.logical_length, view.active_domain, round_point);
            terminal_spec.unmasked_terminal_openings = {zkgpt::native_pedersen::commit_fresh(scalar_basis, terminal_value)};
            NativeMaskedSumcheckResult sumcheck_result;
            if (!prover.set_terminal_spec(terminal_spec, reason) || !prover.finish({terminal_value}, &sumcheck_result, reason)) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor zero-interval terminal relation failed");
            if (sumcheck_result.challenges != round_point || sumcheck_result.challenges.size() != shape.active_bits || sumcheck_result.unmasked_terminal_openings.size() != 1) throw std::logic_error("Hyrax tensor zero-interval sumcheck result is incomplete");
            certificate.sumcheck_challenges = sumcheck_result.challenges;
            certificate.full_terminal_point = sumcheck_result.challenges;
            certificate.full_terminal_point.resize(shape.full_bits, Fr(0));
            MaskedMleOpening terminal_opening;
            if (!open_hyrax_tensor_mle_with_session(view, commitment, state, certificate.full_terminal_point, values, session, interval_label + "/opening", terminal_opening, reason, "", &scalar_basis, nullptr, &sumcheck_result.unmasked_terminal_openings[0])) throw std::runtime_error(reason != nullptr && !reason->empty() ? *reason : "Hyrax tensor zero-interval terminal opening failed");
            if (terminal_opening.raw_evaluation_opening.value != terminal_value) throw std::logic_error("Hyrax tensor zero-interval raw opening differs from its sumcheck terminal");
            pending_certificates.push_back(std::move(certificate));
        }
        exact_position = session.checkpoint();
        certificates = std::move(pending_certificates);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        exact_position = checkpoint;
        certificates.clear();
        if (unmasked_round_traces != nullptr) unmasked_round_traces->clear();
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool verify_hyrax_tensor_zero_interval_certificates_with_session(const HyraxTensorMleView &view, const HyraxTensorMleCommitment &commitment, const std::vector<HyraxTensorZeroInterval> &trusted_intervals, const NativeProofSession &session, std::size_t &exact_position, const zkgpt::native_pedersen::PedersenBasis &scalar_basis, std::vector<HyraxTensorZeroIntervalCertificate> &certificates, std::string *reason) {
    certificates.clear();
    try {
        const HyraxTensorShape shape = validate_tensor_view(view);
        validate_tensor_commitment(view, commitment);
        validate_tensor_zero_intervals(view, trusted_intervals);
        if (!session.replay(reason) || !tensor_tail_scalar_domain_matches(session, scalar_basis, reason) || exact_position >= session.messages().size()) return false;
        const std::string base_label = tensor_zero_interval_label(view), public_label = base_label + "/public";
        const auto *public_positions = session.message_positions(public_label);
        const NativeProofMessage &header = session.messages()[exact_position];
        if (public_positions == nullptr || public_positions->size() != 1 || public_positions->front() != exact_position || header.kind != NativeProofMessageKind::Data || header.label != public_label || header.payload != tensor_zero_interval_payload(view, commitment, trusted_intervals)) {
            set_open_reason(reason, "Hyrax tensor zero-interval public context is missing or mismatched");
            return false;
        }
        ++exact_position;
        std::vector<HyraxTensorZeroIntervalCertificate> pending_certificates;
        pending_certificates.reserve(trusted_intervals.size());
        for (std::size_t interval_index = 0; interval_index < trusted_intervals.size(); ++interval_index) {
            const HyraxTensorZeroInterval &interval = trusted_intervals[interval_index];
            const std::string interval_label = base_label + "/" + std::to_string(interval_index), sum_label = interval_label + "/sum";
            HyraxTensorZeroIntervalCertificate certificate;
            certificate.begin = interval.begin;
            certificate.end = interval.end;
            if (!tensor_tail_consume_challenges(session, exact_position, interval_label, shape.active_bits, certificate.beta, reason)) return false;
            NativeMaskedTerminalSpec terminal_spec;
            terminal_spec.kind = NativeMaskedTerminalKind::Product2Public;
            terminal_spec.public_provider = tensor_zero_interval_public_provider(certificate.beta, interval);
            terminal_spec.has_external_binding = true;
            terminal_spec.public_initial_claim = true;
            if (exact_position >= session.messages().size() || session.messages()[exact_position].kind != NativeProofMessageKind::Data || session.messages()[exact_position].label != sum_label + "/initial/mode" || session.messages()[exact_position].payload != "zero-mask-v2") {
                set_open_reason(reason, "Hyrax tensor zero-interval sumcheck does not immediately follow its public challenges");
                return false;
            }
            if (!native_masked_opening_commitment_at(session, scalar_basis, exact_position + 1, sum_label + "/initial/opening", terminal_spec.initial_commitment, reason)) return false;
            const auto *mode_positions = session.message_positions(sum_label + "/initial/mode");
            NativeMaskedSumcheckResult sumcheck_result;
            if (mode_positions == nullptr || mode_positions->size() != 1 || mode_positions->front() != exact_position || !native_masked_sumcheck_verify(session, sum_label, 2, shape.active_bits, Fr(0), terminal_spec, &sumcheck_result, reason)) return false;
            if (sumcheck_result.challenges.size() != shape.active_bits || sumcheck_result.unmasked_terminal_commitments.size() != 1 || sumcheck_result.transcript_end <= exact_position) {
                set_open_reason(reason, "Hyrax tensor zero-interval sumcheck result is incomplete");
                return false;
            }
            exact_position = sumcheck_result.transcript_end;
            certificate.sumcheck_challenges = sumcheck_result.challenges;
            certificate.full_terminal_point = sumcheck_result.challenges;
            certificate.full_terminal_point.resize(shape.full_bits, Fr(0));
            const std::string opening_label = interval_label + "/opening";
            const auto *tensor_public_positions = session.message_positions(opening_label + "/tensor/public");
            const auto *claim_positions = session.message_positions(opening_label + "/column/claim");
            if (tensor_public_positions == nullptr || tensor_public_positions->size() != 1 || tensor_public_positions->front() != exact_position || claim_positions == nullptr || claim_positions->size() != 1 || claim_positions->front() != exact_position + 2) {
                set_open_reason(reason, "Hyrax tensor zero-interval opening does not immediately follow its sumcheck");
                return false;
            }
            G1 mask_commitment;
            Fr masked_evaluation;
            zkgpt::native_pedersen::ScalarCommitment raw_commitment, masked_commitment;
            if (!tensor_tail_parse_opening_claim(session, claim_positions->front(), opening_label + "/column/claim", scalar_basis, mask_commitment, masked_evaluation, raw_commitment, masked_commitment, reason)) return false;
            const auto &expected_raw = sumcheck_result.unmasked_terminal_commitments[0];
            if (raw_commitment.domain != expected_raw.domain || raw_commitment.point != expected_raw.point) {
                set_open_reason(reason, "Hyrax tensor zero-interval opening raw claim differs from Product2Public terminal");
                return false;
            }
            if (!verify_hyrax_tensor_mle_opening_with_session(view, commitment, certificate.full_terminal_point, mask_commitment, masked_evaluation, session, opening_label, reason, true, "", &scalar_basis, &expected_raw, &masked_commitment)) return false;
            const std::string terminal_label = opening_label + "/column/response-ipa/terminal";
            const auto *terminal_positions = session.message_positions(terminal_label);
            if (terminal_positions == nullptr || terminal_positions->size() != 1 || terminal_positions->front() < exact_position) {
                set_open_reason(reason, "Hyrax tensor zero-interval response-IPA terminal is missing");
                return false;
            }
            exact_position = terminal_positions->front() + 1;
            pending_certificates.push_back(std::move(certificate));
        }
        certificates = std::move(pending_certificates);
        return true;
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

namespace {

std::string chunked_digest(const std::string &payload) {
    Fr digest;
    digest.setHashOf(payload.data(), payload.size());
    std::string encoded = digest.getStr(16);
    if (encoded.empty()) encoded = "0";
    if (encoded.size() > 64)
        throw std::runtime_error("chunked Hyrax digest overflow");
    return std::string(64 - encoded.size(), '0') + encoded;
}

bool chunked_power_of_two(std::uint64_t value) {
    return value != 0 && (value & (value - 1)) == 0;
}

std::size_t chunked_log2(std::uint64_t value) {
    if (!chunked_power_of_two(value))
        throw std::invalid_argument("chunked Hyrax domain is not a power of two");
    std::size_t bits = 0;
    while (value > 1) {
        value >>= 1;
        ++bits;
    }
    return bits;
}

std::vector<G1> chunked_generators(const std::string &basis_id,
                                   std::uint64_t offset,
                                   std::uint64_t count) {
    if (count == 0 || count > (1ULL << 16))
        throw std::invalid_argument("chunked Hyrax generator bound is invalid");
    std::vector<G1> generators(static_cast<std::size_t>(count));
    for (std::uint64_t index = 0; index < count; ++index) {
        hashAndMapToG1(generators[static_cast<std::size_t>(index)],
                       "zkGPT/hyrax/chunked/v1/" + basis_id + "/" +
                           std::to_string(offset) + "/" +
                           std::to_string(index));
        if (!valid_point(generators[static_cast<std::size_t>(index)], true))
            throw std::runtime_error("chunked Hyrax generator is invalid");
    }
    return generators;
}

std::string chunked_basis_id(const std::string &basis_id,
                             std::uint64_t offset) {
    const std::string result = chunked_digest(
        "zkgpt-hyrax/chunked-basis/v1/" + basis_id + "/" +
        std::to_string(offset));
    return result;
}

std::string chunked_point_digest(const std::vector<Fr> &point) {
    std::ostringstream payload;
    payload << "zkgpt-hyrax/chunked-point/v1/" << point.size();
    for (const Fr &value : point) payload << '/' << value.getStr(16);
    return chunked_digest(payload.str());
}

std::string chunked_root_digest(const ChunkedMleWire &wire) {
    std::ostringstream payload;
    payload << wire.schema << '|' << wire.version << '|' << wire.label << '|'
            << wire.basis_id << '|' << wire.scalar_domain << '|'
            << wire.logical_length << '|' << wire.padded_length << '|'
            << wire.chunk_size << '|' << wire.point_digest;
    for (const ChunkedMleChunkWire &chunk : wire.chunks) {
        payload << '|' << chunk.offset << '|' << chunk.actual_length << '|'
                << chunk.padded_length << '|' << chunk.label << '|'
                << chunk.basis_id << '|' << chunk.commitment_hex << '|'
                << chunk.evaluation_mask_commitment_hex << '|'
                << chunk.masked_evaluation << '|'
                << chunk.raw_evaluation_commitment_hex << '|'
                << chunk.masked_evaluation_commitment_hex;
    }
    return chunked_digest(payload.str());
}

bool chunked_shape(std::uint64_t logical_length,
                   std::uint64_t padded_length,
                   std::uint64_t chunk_size,
                   const std::vector<Fr> &point,
                   std::string *reason) {
    if (logical_length == 0 || padded_length == 0 ||
        logical_length > padded_length || !chunked_power_of_two(padded_length) ||
        !chunked_power_of_two(chunk_size) || chunk_size < 2 ||
        chunk_size > (1ULL << 16) || padded_length % chunk_size != 0 ||
        point.size() != chunked_log2(padded_length) ||
        point.size() >= sizeof(std::size_t) * 8) {
        if (reason != nullptr) *reason = "chunked Hyrax shape is invalid";
        return false;
    }
    return true;
}

bool chunked_scalar_commitment(
    const std::string &encoded, const zkgpt::native_pedersen::PedersenBasis &basis,
    zkgpt::native_pedersen::ScalarCommitment &result, std::string *reason) {
    G1 point;
    if (!decode_open_point(encoded, point)) {
        if (reason != nullptr) *reason = "chunked scalar commitment is malformed";
        return false;
    }
    result.domain = basis.domain;
    result.point = point;
    return result.valid(basis, reason);
}

}

bool prove_chunked_mle_opening(
    NativeProofSession &session, const std::string &label,
    std::uint64_t logical_length, std::uint64_t padded_length,
    std::uint64_t chunk_size, const std::vector<Fr> &point,
    const ChunkedMleValueProvider &provider,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    ChunkedMleProverResult &result, std::string *reason) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || !provider || !scalar_basis.valid(reason) ||
            !chunked_shape(logical_length, padded_length, chunk_size, point,
                           reason))
            throw std::invalid_argument(reason != nullptr && !reason->empty()
                                            ? *reason
                                            : "invalid chunked Hyrax opening");
        const std::size_t local_bits = chunked_log2(chunk_size);
        const std::vector<Fr> local_point(point.begin(),
                                          point.begin() + local_bits);
        result = ChunkedMleProverResult{};
        result.wire.label = label;
        result.wire.basis_id = chunked_digest(
            "zkgpt-hyrax/chunked-root-basis/v1/" + label + "/" +
            scalar_basis.domain);
        result.wire.scalar_domain = scalar_basis.domain;
        result.wire.logical_length = logical_length;
        result.wire.padded_length = padded_length;
        result.wire.chunk_size = chunk_size;
        result.wire.point_digest = chunked_point_digest(point);
        session.absorb(label + "/schedule",
                       result.wire.basis_id + "|" + scalar_basis.domain + "|" +
                           std::to_string(logical_length) + "|" +
                           std::to_string(padded_length) + "|" +
                           std::to_string(chunk_size) + "|" +
                           result.wire.point_digest);

        const std::uint64_t chunk_count = padded_length / chunk_size;
        result.wire.chunks.reserve(static_cast<std::size_t>(chunk_count));
        std::vector<zkgpt::native_pedersen::ScalarOpening> raw_openings;
        std::vector<Fr> aggregate_coefficients;
        raw_openings.reserve(static_cast<std::size_t>(chunk_count));
        aggregate_coefficients.reserve(static_cast<std::size_t>(chunk_count) + 1);
        Fr aggregate = F_ZERO;
        for (std::uint64_t chunk_index = 0; chunk_index < chunk_count;
             ++chunk_index) {
            const std::uint64_t offset = chunk_index * chunk_size;
            const std::uint64_t actual =
                offset < logical_length
                    ? std::min<std::uint64_t>(chunk_size,
                                               logical_length - offset)
                    : 0;
            if (actual == 0 && offset >= logical_length && chunk_index + 1 != chunk_count)
                throw std::invalid_argument("chunked Hyrax has an interior empty chunk");
            std::vector<Fr> values(static_cast<std::size_t>(chunk_size), F_ZERO);
            for (std::uint64_t index = 0; index < actual; ++index)
                values[static_cast<std::size_t>(index)] = provider(offset + index);
            const std::vector<G1> generators =
                chunked_generators(result.wire.basis_id, offset, chunk_size);
            const std::string basis_id = chunked_basis_id(result.wire.basis_id, offset);
            const hyrax::PedersenMleCommitment commitment = pedersen_commit_mle(
                generators, values, basis_id);
            const Fr local_eval = [&]() {
                const std::vector<Fr> basis = mle_basis_from_point(local_point);
                Fr value = F_ZERO;
                for (std::size_t index = 0; index < values.size(); ++index)
                    value += basis[index] * values[index];
                return value;
            }();
            const std::size_t high_bits = point.size() - local_bits;
            Fr high_weight = F_ONE;
            for (std::size_t bit = 0; bit < high_bits; ++bit) {
                const bool set = ((chunk_index >> bit) & 1U) != 0;
                high_weight *= set ? point[local_bits + bit]
                                   : (F_ONE - point[local_bits + bit]);
            }
            aggregate += high_weight * local_eval;
            const std::string chunk_label =
                label + "/chunk/" + std::to_string(chunk_index);
            MaskedMleOpening opening;
            if (!open_masked_mle_with_claim(
                    commitment, generators, local_point, values, local_eval,
                    session, chunk_label, opening, reason, false, "",
                    &scalar_basis))
                throw std::runtime_error(reason != nullptr ? *reason
                                                             : "chunked Hyrax chunk opening failed");
            ChunkedMleChunkWire wire;
            wire.offset = offset;
            wire.actual_length = actual;
            wire.padded_length = chunk_size;
            wire.label = chunk_label;
            wire.basis_id = basis_id;
            wire.commitment_hex = point_encoding(commitment.commitment);
            wire.evaluation_mask_commitment_hex =
                point_encoding(opening.evaluation_mask_commitment);
            wire.masked_evaluation = opening.masked_evaluation.getStr(16);
            wire.raw_evaluation_commitment_hex =
                point_encoding(opening.raw_evaluation_opening.commitment.point);
            wire.masked_evaluation_commitment_hex =
                point_encoding(opening.masked_evaluation_opening.commitment.point);
            result.wire.chunks.push_back(std::move(wire));
            raw_openings.push_back(opening.raw_evaluation_opening);
            aggregate_coefficients.push_back(high_weight);
        }
        result.wire.root_digest = chunked_root_digest(result.wire);
        session.absorb(label + "/root", result.wire.root_digest);
        result.aggregate_opening =
            zkgpt::native_pedersen::commit_fresh(scalar_basis, aggregate);
        result.wire.aggregate_commitment_hex =
            point_encoding(result.aggregate_opening.commitment.point);
        raw_openings.push_back(result.aggregate_opening);
        aggregate_coefficients.push_back(-F_ONE);
        if (!zkgpt::native_pedersen::prove_hidden_linear(
                session, scalar_basis, label + "/aggregate-link", raw_openings,
                aggregate_coefficients, F_ZERO, nullptr, reason))
            throw std::runtime_error(reason != nullptr ? *reason
                                                         : "chunked Hyrax aggregate link failed");
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool verify_chunked_mle_opening(
    const NativeProofSession &session, const ChunkedMleWire &wire,
    const std::string &label, std::uint64_t logical_length,
    std::uint64_t padded_length, std::uint64_t chunk_size,
    const std::vector<Fr> &point,
    const zkgpt::native_pedersen::PedersenBasis &scalar_basis,
    std::string *reason) {
    try {
        if (wire.schema != kChunkedMleSchema ||
            wire.version != kChunkedMleVersion || wire.label != label ||
            wire.basis_id.empty() || wire.scalar_domain != scalar_basis.domain ||
            wire.logical_length != logical_length ||
            wire.padded_length != padded_length || wire.chunk_size != chunk_size ||
            !scalar_basis.valid(reason) ||
            !chunked_shape(logical_length, padded_length, chunk_size, point,
                           reason))
            throw std::invalid_argument(reason != nullptr && !reason->empty()
                                            ? *reason
                                            : "chunked Hyrax wire shape is invalid");
        if (wire.point_digest != chunked_point_digest(point) ||
            wire.root_digest != chunked_root_digest(wire))
            throw std::invalid_argument("chunked Hyrax canonical root mismatch");
        const std::size_t local_bits = chunked_log2(chunk_size);
        const std::vector<Fr> local_point(point.begin(),
                                          point.begin() + local_bits);
        const std::uint64_t chunk_count = padded_length / chunk_size;
        if (wire.chunks.size() != static_cast<std::size_t>(chunk_count))
            throw std::invalid_argument("chunked Hyrax chunk count mismatch");
        std::vector<zkgpt::native_pedersen::ScalarCommitment> raw_commitments;
        std::vector<Fr> aggregate_coefficients;
        raw_commitments.reserve(wire.chunks.size() + 1);
        aggregate_coefficients.reserve(wire.chunks.size() + 1);
        for (std::size_t index = 0; index < wire.chunks.size(); ++index) {
            const ChunkedMleChunkWire &chunk = wire.chunks[index];
            const std::uint64_t expected_offset =
                static_cast<std::uint64_t>(index) * chunk_size;
            const std::uint64_t expected_actual =
                expected_offset < logical_length
                    ? std::min<std::uint64_t>(chunk_size,
                                               logical_length - expected_offset)
                    : 0;
            if (chunk.offset != expected_offset ||
                chunk.actual_length != expected_actual ||
                chunk.padded_length != chunk_size ||
                chunk.label != label + "/chunk/" + std::to_string(index) ||
                chunk.basis_id != chunked_basis_id(wire.basis_id, chunk.offset))
                throw std::invalid_argument("chunked Hyrax descriptor order mismatch");
            G1 commitment_point, mask_point;
            if (!decode_open_point(chunk.commitment_hex, commitment_point) ||
                !decode_open_point(chunk.evaluation_mask_commitment_hex, mask_point))
                throw std::invalid_argument("chunked Hyrax group commitment is malformed");
            Fr masked_evaluation;
            if (!decode_open_field(chunk.masked_evaluation, masked_evaluation))
                throw std::invalid_argument("chunked Hyrax masked evaluation is malformed");
            zkgpt::native_pedersen::ScalarCommitment raw_commitment;
            zkgpt::native_pedersen::ScalarCommitment masked_commitment;
            if (!chunked_scalar_commitment(chunk.raw_evaluation_commitment_hex,
                                           scalar_basis, raw_commitment, reason) ||
                !chunked_scalar_commitment(chunk.masked_evaluation_commitment_hex,
                                           scalar_basis, masked_commitment, reason))
                return false;
            const std::vector<G1> generators = chunked_generators(
                wire.basis_id, chunk.offset, chunk_size);
            if (!verify_masked_mle_opening_with_session(
                    commitment_point, mask_point, masked_evaluation,
                    chunk.basis_id, generators, local_point, session,
                    chunk.label, reason, false, true, "", &scalar_basis,
                    &raw_commitment, &masked_commitment))
                return false;
            raw_commitments.push_back(raw_commitment);
            Fr high_weight = F_ONE;
            const std::size_t high_bits = point.size() - local_bits;
            for (std::size_t bit = 0; bit < high_bits; ++bit) {
                const bool set = ((static_cast<std::uint64_t>(index) >> bit) & 1U) != 0;
                high_weight *= set ? point[local_bits + bit]
                                   : (F_ONE - point[local_bits + bit]);
            }
            aggregate_coefficients.push_back(high_weight);
        }
        zkgpt::native_pedersen::ScalarCommitment aggregate_commitment;
        if (!chunked_scalar_commitment(wire.aggregate_commitment_hex,
                                       scalar_basis, aggregate_commitment, reason))
            return false;
        raw_commitments.push_back(aggregate_commitment);
        aggregate_coefficients.push_back(-F_ONE);
        return zkgpt::native_pedersen::verify_hidden_linear(
            session, scalar_basis, label + "/aggregate-link", raw_commitments,
            aggregate_coefficients, F_ZERO, reason);
    } catch (const std::exception &error) {
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}
}
