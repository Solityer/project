//
// Created by 69029 on 3/9/2021.
//

#include <cmath>
#include <iostream>

#include "utils.hpp"

using std::cerr;
using std::endl;
using std::string;
using std::cin;

int ceilPow2BitLengthSigned(double n) {
    return (i8) ceil(log2(n));
}

int floorPow2BitLengthSigned(double n) {
    return (i8) floor(log2(n));
}

i8 ceilPow2BitLength(u64 n) 
{
    if (!n) return -1;
    i8 bits = 0;
    for (--n; n; n >>= 1) ++bits;
    return bits;
}

i8 floorPow2BitLength(u64 n) 
{
    if (!n) return -1;
    i8 bits = 0;
    for (; n >>= 1;) ++bits;
    return bits;
}

void initHalfTable(vector<F> &beta_f, vector<F> &beta_s, const vector<F>::const_iterator &r, const F &init, u64 first_half, u64 second_half) {
    beta_f.at(0) = init;
    beta_s.at(0) = F_ONE;
    for (u64 i = 0; i < first_half; ++i) 
    {
        for (u64 j = 0; j < (1ULL << i); ++j) 
        {
            auto tmp = beta_f.at(j) * r[i];
            beta_f.at(j | (1ULL << i)) = tmp;
            beta_f.at(j) = beta_f[j] - tmp;
        }
    }
    for (u64 i = 0; i < second_half; ++i) {
        for (u64 j = 0; j < (1ULL << i); ++j) {
            auto tmp = beta_s[j] * r[(i + first_half)];
            beta_s[j | (1ULL << i)] = tmp;
            beta_s[j] = beta_s[j] - tmp;
        }
    }
}




static ThreadSafeQueue<int> workerq,endq;
template<class Vector> void initBetaTable_worker(Vector &beta_g,vector<F> &beta_f,vector<F> &beta_s,u64*&L,u64*&R,int first_half,u64 mask_fhalf)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];
        for(u64 kp=l;kp<r;kp++)
        {
            beta_g[kp] = beta_f[kp & mask_fhalf] * beta_s[kp >> first_half];
        }
        endq.Push(idx);
    }
}
template<class Vector> void initBetaTable_worker2(Vector &beta_g,vector<F> &beta_f,vector<F> &beta_s,u64*&L,u64*&R,int first_half,u64 mask_fhalf)
{
    int idx;
    while (true)
    {
        bool ret=workerq.TryPop(idx);
        if(ret==false)
            return;
        u64 l=L[idx],r=R[idx];
        for(u64 kp=l;kp<r;kp++)
        {
            beta_g[kp] += beta_f[kp & mask_fhalf] * beta_s[kp >> first_half];
        }
        endq.Push(idx);
    }
}
template<class Vector> void initBetaTable_impl(Vector &beta_g, u8 gLength, const vector<F>::const_iterator &r_0, const vector<F>::const_iterator &r_1,
              const F &alpha, const F &beta,int thd) 
{
    u8 first_half = gLength >> 1, second_half = gLength - first_half;
    u64 mask_fhalf = (1ULL << first_half) - 1;

    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);
    if (!beta.isZero()) {
        initHalfTable(beta_f, beta_s, r_1, beta, first_half, second_half);
        if(thd==1 || gLength<15)
        {
            for (u64 i = 0; i < (1ULL << gLength); ++i)
                beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
        }
        else
        {
            const int k=10;
            std::vector<std::thread> workers;
            u64 total_work=(1ULL << gLength);
            u64 *L=new u64 [(1<<k)],*R=new u64 [(1<<k)];
            for (u64 j = 0; j < (1<<k); ++j) 
            {
                workerq.Push(j);
                L[j]=(total_work>>k)*j;
                R[j]=(total_work>>k)*(1+j);
            }
            for(int j=0;j<thd;j++)
            {
                workers.emplace_back(initBetaTable_worker<Vector>,std::ref(beta_g),std::ref(beta_f),std::ref(beta_s),std::ref(L),std::ref(R),first_half,mask_fhalf);
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(10));
            while(endq.Size()!=(1<<k))
                this_thread::sleep_for (std::chrono::microseconds(10));
            endq.Clear();
        for (auto& worker : workers) worker.join();
        delete[] L;
        delete[] R;
        }
    } else for (u64 i = 0; i < (1ULL << gLength); ++i)
        beta_g[i].clear();

    if (alpha.isZero()) return;
    initHalfTable(beta_f, beta_s, r_0, alpha, first_half, second_half);
    if(thd==1 || gLength<15)
    {
        for (u64 i = 0; i < (1ULL << gLength); ++i)
            beta_g[i] += beta_f[i & mask_fhalf] * beta_s[i >> first_half];
    }
    else
        {
            const int k=10;
            std::vector<std::thread> workers;
            u64 total_work=(1ULL << gLength);
            u64 *L=new u64 [(1<<k)],*R=new u64 [(1<<k)];
            for (u64 j = 0; j < (1<<k); ++j) 
            {
                workerq.Push(j);
                L[j]=(total_work>>k)*j;
                R[j]=(total_work>>k)*(1+j);
            }
            for(int j=0;j<thd;j++)
            {
                workers.emplace_back(initBetaTable_worker2<Vector>,std::ref(beta_g),std::ref(beta_f),std::ref(beta_s),std::ref(L),std::ref(R),first_half,mask_fhalf);
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(10));
            while(endq.Size()!=(1<<k))
                this_thread::sleep_for (std::chrono::microseconds(10));
            endq.Clear();
        for (auto& worker : workers) worker.join();
        delete[] L;
        delete[] R;
        }
}


template<class Vector> void initBetaTable_impl(Vector &beta_g, u8 gLength, const vector<F>::const_iterator &r, const F &init,int thd) 
{
    if (gLength == -1) 
        return;
    int first_half = gLength >> 1, second_half = gLength - first_half;
    u64 mask_fhalf = (1ULL << first_half) - 1;
    vector<F> beta_f(1ULL << first_half), beta_s(1ULL << second_half);
    if (!init.isZero()) 
    {
        initHalfTable(beta_f, beta_s, r, init, first_half, second_half);
        //cout<<"compute beta "<<(int)gLength<<endl;
        if(thd==1 || gLength<15)
        {
            for (u64 i = 0; i < (1ULL << gLength); ++i)
                beta_g[i] = beta_f[i & mask_fhalf] * beta_s[i >> first_half];
        }
        else
        {
            const int k=10;
            std::vector<std::thread> workers;
            u64 total_work=(1ULL << gLength);
            u64 *L=new u64 [(1<<k)],*R=new u64 [(1<<k)];
            for (u64 j = 0; j < (1<<k); ++j) 
            {
                workerq.Push(j);
                L[j]=(total_work>>k)*j;
                R[j]=(total_work>>k)*(1+j);
            }
            for(int j=0;j<thd;j++)
            {
                workers.emplace_back(initBetaTable_worker<Vector>,std::ref(beta_g),std::ref(beta_f),std::ref(beta_s),std::ref(L),std::ref(R),first_half,mask_fhalf);
            }
            while(!workerq.Empty())
                this_thread::sleep_for (std::chrono::microseconds(10));
            while(endq.Size()!=(1<<k))
                this_thread::sleep_for (std::chrono::microseconds(10));
            endq.Clear();
        for (auto& worker : workers) worker.join();
        delete[] L;
        delete[] R;
        }
    } 
    else for (u64 i = 0; i < (1ULL << gLength); ++i)
        beta_g[i].clear();
}

void initBetaTable(vector<F> &values, u8 bits, const vector<F>::const_iterator &r0, const vector<F>::const_iterator &r1, const F &alpha, const F &beta, int threads) { initBetaTable_impl(values, bits, r0, r1, alpha, beta, threads); }
void initBetaTable(vector<F> &values, u8 bits, const vector<F>::const_iterator &point, const F &initial, int threads) { initBetaTable_impl(values, bits, point, initial, threads); }
void initBetaTable(zkgpt::NativeFieldVector &values, u8 bits, const vector<F>::const_iterator &r0, const vector<F>::const_iterator &r1, const F &alpha, const F &beta, int threads) { initBetaTable_impl(values, bits, r0, r1, alpha, beta, threads); }
void initBetaTable(zkgpt::NativeFieldVector &values, u8 bits, const vector<F>::const_iterator &point, const F &initial, int threads) { initBetaTable_impl(values, bits, point, initial, threads); }

bool check(long x, long y, long nx, long ny) 
{
    return 0 <= x && x < nx && 0 <= y && y < ny;
}

void initLayer(layer &circuit, long size, layerType ty) 
{
    circuit.size = circuit.zero_start_id = size;
    circuit.bit_length = ceilPow2BitLength(size);
    circuit.ty = ty;
    //circuit.uni_gates.clear();
    //circuit.bin_gates.clear();
}

long sqr(long x) {
    return x * x;
}

double byte2KB(size_t x) { return x / 1024.0; }

double byte2MB(size_t x) { return x / 1024.0 / 1024.0; }

double byte2GB(size_t x) { return x / 1024.0 / 1024.0 / 1024.0; }

long matIdx(long x, long y, long n) { // row number , n: row len, y : column num
    assert(y < n);
    return x * n + y;
}

void field(const char* f,Fr x)
{
    if(x.isNegative())
        cout<<f<<" -"<<-x<<endl;
    else
        cout<<f<<" "<<x<<endl;
    
}
long cubIdx(long x, long y, long z, long n, long m) {
    assert(y < n && z < m);
    return matIdx(matIdx(x, y, n), z, m);
}

long tesIdx(long w, long x, long y, long z, long n, long m, long l) {
    assert(x < n && y < m && z < l);
    return matIdx(cubIdx(w, x, y, n, m), z, l);
}

F getRootOfUnit(int n) {
    F res = -F_ONE;
    if (!n) return F_ONE;
    while (--n) {
        bool b = F::squareRoot(res, res);
        assert(b);
    }
    return res;
}


