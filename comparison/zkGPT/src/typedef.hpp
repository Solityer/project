#ifndef HYRAX_P224_TYPEDEF_HPP
#define HYRAX_P224_TYPEDEF_HPP

//typedef unsigned __int128 u128;
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

//typedef __int128 i128;
typedef long long i64;
typedef int i32;
// Circuit bit lengths use -1 as the official "absent subset" sentinel.
// Plain char is unsigned on the aarch64 build host, which turns that sentinel
// into 255 and makes every tiny circuit appear to have a 255-bit domain.
// Keep the one-byte ABI while preserving the upstream signed-sentinel
// semantics on every target.
typedef signed char i8;
typedef __int128 ll;
#endif //HYRAX_P224_TYPEDEF_HPP
