#include "hyrax.hpp"
#include "proof_session.hpp"
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

namespace {

std::string point_encoding(const G1 &point) {
    return point.serializeToHexStr();
}

bool valid_point(const G1 &point, bool require_nonzero) {
    return (!require_nonzero || !point.isZero()) && point.isValid() &&
           point.isValidOrder();
}

std::string basis_encoding(const G1 &G, const std::vector<G1> &generators,
                           const std::vector<Fr> &basis) {
    std::ostringstream payload;
    payload << point_encoding(G) << '|' << generators.size();
    for (std::size_t i = 0; i < generators.size(); ++i)
        payload << '|' << point_encoding(generators[i]) << '|'
                << basis[i].getStr(16);
    return payload.str();
}

void absorb_fold(NativeProofSession &session, const std::string &label,
                 const G1 &gamma_minus1, const G1 &gamma_1,
                 const Fr &cross_left, const Fr &cross_right) {
    std::ostringstream payload;
    payload << point_encoding(gamma_minus1) << '|'
            << point_encoding(gamma_1);
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
        c.setByCSPRNG();  // step3 V choose random c
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

void prove_dot_product(G1 comm_x, G1 comm_y, Fr* a, G1*g ,G1& G,Fr* x,Fr y,int n)  // y= <a,x> ,
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

} // namespace

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
pair<double,double> open(Fr*w,Fr*r,Fr eval,G1&G,G1*g,Fr*L,Fr*R,G1*tk,int l)
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
    const NativeProofMessage *found = nullptr;
    for (const NativeProofMessage &message : session.messages()) {
        if (message.label != label) continue;
        if (found != nullptr) {
            set_open_reason(reason, "duplicate Hyrax opening message: " + label);
            return nullptr;
        }
        found = &message;
    }
    if (found == nullptr) set_open_reason(reason, "missing Hyrax opening message: " + label);
    return found;
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
    if (point.empty() || point.size() >= sizeof(std::size_t) * 8)
        throw std::invalid_argument("masked Hyrax evaluation point is invalid");
    const std::size_t size = std::size_t(1) << point.size();
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

std::string masked_basis_payload(const G1 &commitment, const G1 &H,
                                 const G1 &J, const std::string &basis_id,
                                 const std::vector<G1> &generators,
                                 const std::vector<Fr> &basis) {
    std::ostringstream payload;
    payload << point_encoding(commitment) << '|' << point_encoding(H) << '|'
            << point_encoding(J) << '|' << basis_id << '|' << generators.size();
    for (std::size_t index = 0; index < generators.size(); ++index)
        payload << '|' << point_encoding(generators[index]) << '|'
                << basis[index].getStr(16);
    return payload.str();
}

std::string masked_claim_payload(const G1 &mask_commitment, const Fr &claim) {
    return point_encoding(mask_commitment) + "|" + claim.getStr(16);
}

std::string masked_announce_payload(const G1 &announce_commitment,
                                   const G1 &mask_announce,
                                   const Fr &announce_eval) {
    return point_encoding(announce_commitment) + "|" +
           point_encoding(mask_announce) + "|" + announce_eval.getStr(16);
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

} // namespace

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
        std::size_t basis_index = session.messages().size();
        for (std::size_t index = 0; index < session.messages().size(); ++index) {
            if (session.messages()[index].label == label + "/basis") {
                if (basis_index != session.messages().size()) {
                    set_open_reason(reason, "duplicate Hyrax basis binding");
                    return false;
                }
                basis_index = index;
            }
        }
        if (basis_index == session.messages().size()) {
            set_open_reason(reason, "missing Hyrax basis binding");
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

PedersenMleCommitment pedersen_commit_mle(
    const std::vector<G1> &generators, const std::vector<Fr> &values,
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

static bool open_masked_mle_impl(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, const Fr *forced_masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis) {
    const std::size_t checkpoint = session.checkpoint();
    try {
        if (label.empty() || values.empty() || values.size() != generators.size() ||
            !power_of_two_size(values.size()) || values.size() < 2 ||
            point.empty() || point.size() >= sizeof(std::size_t) * 8 ||
            (std::size_t(1) << point.size()) != values.size())
            throw std::invalid_argument("masked Hyrax MLE dimensions are invalid");
        if (commitment.basis_id.empty() || commitment.basis_id.size() > 128 ||
            !valid_point(commitment.blinding_generator, true) ||
            commitment.blinding_generator !=
                derive_pedersen_blinding_generator(commitment.basis_id))
            throw std::invalid_argument("masked Hyrax commitment blinding identity is invalid");
        for (const G1 &generator : generators)
            if (!valid_point(generator, true))
                throw std::invalid_argument("masked Hyrax commitment basis is invalid");
        const std::vector<Fr> basis = mle_basis_from_point(point);
        if (basis.size() != values.size())
            throw std::invalid_argument("masked Hyrax MLE basis size mismatch");
        if (enforce_statement_basis &&
            session.statement().basis_digest != NativeProofStatement::basis_digest_for(
                commitment.blinding_generator, generators, basis, values.size()))
            throw std::invalid_argument("masked Hyrax statement basis mismatch");

        const G1 expected_commitment =
            perdersen_commit(const_cast<G1 *>(generators.data()),
                             const_cast<Fr *>(values.data()),
                             static_cast<int>(values.size())) +
            commitment.blinding_generator * commitment.blinding;
        if (expected_commitment != commitment.commitment)
            throw std::invalid_argument("masked Hyrax commitment does not bind the witness");

        const G1 J = derive_pedersen_evaluation_generator(commitment.basis_id, label);
        if (J == commitment.blinding_generator)
            throw std::invalid_argument("masked Hyrax evaluation generator is not independent");
        for (const G1 &generator : generators)
            if (generator == commitment.blinding_generator || generator == J)
                throw std::invalid_argument("masked Hyrax generators are not domain separated");
        Fr raw_evaluation(0);
        for (std::size_t index = 0; index < values.size(); ++index)
            raw_evaluation += basis[index] * values[index];
        Fr mu;
        if (forced_masked_evaluation != nullptr) {
            // The forced claim is still hidden: only its difference from the
            // private raw evaluation is retained as the evaluation mask.
            mu = *forced_masked_evaluation - raw_evaluation;
        } else {
            mu.setByCSPRNG();
        }
        opening.evaluation_mask_commitment = J * mu;
        opening.masked_evaluation = raw_evaluation + mu;

        std::vector<Fr> random_vector(values.size());
        for (Fr &value : random_vector) value.setByCSPRNG();
        Fr random_rho;
        Fr random_mu;
        random_rho.setByCSPRNG();
        random_mu.setByCSPRNG();
        const G1 announce_commitment =
            perdersen_commit(const_cast<G1 *>(generators.data()),
                             random_vector.data(),
                             static_cast<int>(random_vector.size())) +
            commitment.blinding_generator * random_rho;
        const G1 announce_mask = J * random_mu;
        Fr announce_eval = random_mu;
        for (std::size_t index = 0; index < values.size(); ++index)
            announce_eval += basis[index] * random_vector[index];

        session.absorb(label + "/basis",
                       masked_basis_payload(commitment.commitment,
                                            commitment.blinding_generator, J,
                                            commitment.basis_id, generators, basis));
        session.absorb(label + "/claim",
                       masked_claim_payload(opening.evaluation_mask_commitment,
                                            opening.masked_evaluation));
        session.absorb(label + "/announce",
                       masked_announce_payload(announce_commitment, announce_mask,
                                               announce_eval));
        const Fr challenge = session.challenge(label + "/c");
        std::vector<Fr> response(values.size() + 2);
        for (std::size_t index = 0; index < values.size(); ++index)
            response[index] = random_vector[index] + challenge * values[index];
        response[values.size()] = random_rho + challenge * commitment.blinding;
        response[values.size() + 1] = random_mu + challenge * mu;
        session.absorb_fields(label + "/response", response);
        return true;
    } catch (const std::exception &error) {
        session.rollback(checkpoint);
        if (reason != nullptr) *reason = error.what();
        return false;
    }
}

bool open_masked_mle_with_session(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, NativeProofSession &session,
    const std::string &label, MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis) {
    return open_masked_mle_impl(commitment, generators, point, values, nullptr,
                                session, label, opening, reason,
                                enforce_statement_basis);
}

bool open_masked_mle_with_claim(
    const PedersenMleCommitment &commitment,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const std::vector<Fr> &values, const Fr &masked_evaluation,
    NativeProofSession &session, const std::string &label,
    MaskedMleOpening &opening, std::string *reason,
    bool enforce_statement_basis) {
    return open_masked_mle_impl(commitment, generators, point, values,
                                &masked_evaluation, session, label, opening,
                                reason, enforce_statement_basis);
}

bool verify_masked_mle_opening_with_session(
    const G1 &commitment, const G1 &evaluation_mask_commitment,
    const Fr &masked_evaluation, const std::string &basis_id,
    const std::vector<G1> &generators, const std::vector<Fr> &point,
    const NativeProofSession &session, const std::string &label,
    std::string *reason, bool enforce_statement_basis,
    bool allow_trailing_unrelated) {
    try {
        if (label.empty() || basis_id.empty() || point.empty() ||
            point.size() >= sizeof(std::size_t) * 8 || generators.empty() ||
            generators.size() != (std::size_t(1) << point.size()) ||
            !power_of_two_size(generators.size()) || generators.size() < 2)
            throw std::invalid_argument("masked Hyrax verifier dimensions are invalid");
        const std::vector<Fr> basis = mle_basis_from_point(point);
        if (basis.size() != generators.size())
            throw std::invalid_argument("masked Hyrax verifier basis size mismatch");
        const G1 H = derive_pedersen_blinding_generator(basis_id);
        const G1 J = derive_pedersen_evaluation_generator(basis_id, label);
        if (!valid_point(commitment, false) || !valid_point(evaluation_mask_commitment, false) ||
            !valid_point(H, true) || !valid_point(J, true) || H == J)
            throw std::invalid_argument("masked Hyrax verifier point is invalid");
        for (const G1 &generator : generators) {
            if (!valid_point(generator, true) || generator == H || generator == J)
                throw std::invalid_argument("masked Hyrax verifier basis is not domain separated");
        }
        if (enforce_statement_basis &&
            session.statement().basis_digest != NativeProofStatement::basis_digest_for(
                H, generators, basis, generators.size()))
            throw std::invalid_argument("masked Hyrax verifier statement basis mismatch");
        if (!session.replay(reason)) return false;

        std::size_t position = session.messages().size();
        for (std::size_t index = 0; index < session.messages().size(); ++index) {
            if (session.messages()[index].label != label + "/basis") continue;
            if (position != session.messages().size()) {
                set_open_reason(reason, "duplicate masked Hyrax basis message");
                return false;
            }
            position = index;
        }
        if (position == session.messages().size()) {
            set_open_reason(reason, "missing masked Hyrax basis message");
            return false;
        }
        std::size_t cursor = position;
        const NativeProofMessage *basis_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/basis", reason);
        if (basis_message == nullptr) return false;
        std::vector<std::string> basis_parts;
        if (!split_open_payload(basis_message->payload, 5 + 2 * generators.size(),
                                basis_parts)) {
            set_open_reason(reason, "malformed masked Hyrax basis message");
            return false;
        }
        G1 encoded_commitment, encoded_H, encoded_J;
        if (!decode_open_point(basis_parts[0], encoded_commitment) ||
            !decode_open_point(basis_parts[1], encoded_H) ||
            !decode_open_point(basis_parts[2], encoded_J) ||
            encoded_commitment != commitment || encoded_H != H || encoded_J != J ||
            basis_parts[3] != basis_id || basis_parts[4] != std::to_string(generators.size())) {
            set_open_reason(reason, "masked Hyrax basis identity mismatch");
            return false;
        }
        for (std::size_t index = 0; index < generators.size(); ++index) {
            G1 encoded_generator;
            Fr encoded_basis;
            if (!decode_open_point(basis_parts[5 + 2 * index], encoded_generator) ||
                !decode_open_field(basis_parts[6 + 2 * index], encoded_basis) ||
                encoded_generator != generators[index] || encoded_basis != basis[index]) {
                set_open_reason(reason, "masked Hyrax basis coordinate mismatch");
                return false;
            }
        }

        const NativeProofMessage *claim_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/claim", reason);
        if (claim_message == nullptr) return false;
        std::vector<std::string> claim_parts;
        if (!split_open_payload(claim_message->payload, 2, claim_parts)) {
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

        const NativeProofMessage *announce_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Data, label + "/announce", reason);
        if (announce_message == nullptr) return false;
        std::vector<std::string> announce_parts;
        if (!split_open_payload(announce_message->payload, 3, announce_parts)) {
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
        const NativeProofMessage *challenge_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::Challenge,
            "challenge/" + label + "/c", reason);
        if (challenge_message == nullptr) return false;
        Fr challenge;
        if (!decode_open_field(challenge_message->payload, challenge) || challenge.isZero()) {
            set_open_reason(reason, "masked Hyrax challenge is invalid");
            return false;
        }
        const NativeProofMessage *response_message = consume_masked_message(
            session, cursor, NativeProofMessageKind::FieldVector, label + "/response", reason);
        if (response_message == nullptr) return false;
        std::vector<Fr> response;
        if (!decode_field_vector_payload(response_message->payload, response) ||
            response.size() != generators.size() + 2) {
            set_open_reason(reason, "masked Hyrax response vector is invalid");
            return false;
        }
        const Fr &response_rho = response[generators.size()];
        const Fr &response_mu = response[generators.size() + 1];
        G1 response_commitment = perdersen_commit(
            const_cast<G1 *>(generators.data()), response.data(),
            static_cast<int>(generators.size()));
        response_commitment += H * response_rho;
        if (response_commitment != announce_commitment + commitment * challenge) {
            set_open_reason(reason, "masked Hyrax Pedersen relation failed");
            return false;
        }
        if (J * response_mu != announce_mask + evaluation_mask_commitment * challenge) {
            set_open_reason(reason, "masked Hyrax evaluation-mask relation failed");
            return false;
        }
        Fr response_eval = response_mu;
        for (std::size_t index = 0; index < generators.size(); ++index)
            response_eval += basis[index] * response[index];
        if (response_eval != announce_eval + masked_evaluation * challenge) {
            set_open_reason(reason, "masked Hyrax MLE evaluation relation failed");
            return false;
        }
        return reject_masked_trailing(session, cursor, label,
                                      allow_trailing_unrelated, reason);
    } catch (const std::exception &error) {
        set_open_reason(reason, error.what());
        return false;
    }
}
}
