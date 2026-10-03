#include "circuit.h"
#include "utils.hpp"
#include <stdexcept>
#include <algorithm>

void layeredCircuit::initSubset() 
{
    for (int i = 1; i < size; ++i) 
    {
        auto &cur = circuit[i], &lst = circuit[i - 1];
        zkgpt::NativeVector<std::pair<u64, u64>> subset_uidx, subset_vidx;
        bool has_pre_layer_u = cur.ty == layerType::FFT || cur.ty == layerType::IFFT;
        bool has_pre_layer_v = false, has_v_operand = false;
        auto collect = [&](auto& map, u64 coordinate) {
            if (coordinate >= circuit[0].size) throw std::out_of_range("native GKR input index exceeds domain");
            map.emplace_back(coordinate, map.size());
        };
        for (const auto& gate : cur.uni_gates) {
            if (!gate.lu) collect(subset_uidx, gate.u);
            has_pre_layer_u |= gate.lu != 0;
        }
        for (const auto& gate : cur.bin_gates) {
            has_v_operand = true;
            if (!gate.getLayerIdU(i)) collect(subset_uidx, gate.u);
            if (!gate.getLayerIdV(i)) collect(subset_vidx, gate.v);
            has_pre_layer_u |= gate.getLayerIdU(i) != 0;
            has_pre_layer_v |= gate.getLayerIdV(i) != 0;
        }
        auto finish_map = [](auto& map, auto& original, u64& count) {
            std::sort(map.begin(), map.end());
            map.erase(std::unique(map.begin(), map.end(), [](const auto& left, const auto& right) { return left.first == right.first; }), map.end());
            std::sort(map.begin(), map.end(), [](const auto& left, const auto& right) { return left.second < right.second; });
            original.reserve(map.size());
            count = map.size();
            for (u64 index = 0; index < count; ++index) { original.push_back(map[index].first); map[index].second = index; }
            std::sort(map.begin(), map.end());
        };
        finish_map(subset_uidx, cur.ori_id_u, cur.size_u[0]);
        finish_map(subset_vidx, cur.ori_id_v, cur.size_v[0]);
        auto subset_index = [](const auto& map, u64 coordinate) {
            const auto found = std::lower_bound(map.begin(), map.end(), coordinate, [](const auto& entry, u64 target) { return entry.first < target; });
            if (found == map.end() || found->first != coordinate) throw std::logic_error("native GKR input subset map is incomplete");
            return found->second;
        };
        for (auto& gate : cur.uni_gates) if (!gate.lu) gate.u = subset_index(subset_uidx, gate.u);
        for (auto& gate : cur.bin_gates) {
            if (!gate.getLayerIdU(i)) gate.u = subset_index(subset_uidx, gate.u);
            if (!gate.getLayerIdV(i)) gate.v = subset_index(subset_vidx, gate.v);
        }
        cur.bit_length_u[0] = ceilPow2BitLength(cur.size_u[0]);
        cur.bit_length_v[0] = ceilPow2BitLength(cur.size_v[0]);

        if (has_pre_layer_u) 
            switch (cur.ty) {
                case layerType::FFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length - 1;
                    cur.bit_length_u[1] = cur.fft_bit_length - 1;
                    break;
                case layerType::IFFT:
                    cur.size_u[1] = 1ULL << cur.fft_bit_length;
                    cur.bit_length_u[1] = cur.fft_bit_length;
                    break;
                default:
                    cur.size_u[1] = lst.size ;
                    cur.bit_length_u[1] = lst.bit_length;
                    break;
            } else {
            cur.size_u[1] = 0;
            cur.bit_length_u[1] = -1;
        }
        if (has_pre_layer_v) 
        {
            if (cur.ty == layerType::DOT_PROD) {
                cur.size_v[1] = lst.size >> cur.fft_bit_length;
                cur.bit_length_v[1] = lst.bit_length - cur.fft_bit_length;
            } else {
                cur.size_v[1] = lst.size;
                cur.bit_length_v[1] = lst.bit_length;
            }
        } 
        else 
        {
            cur.size_v[1] = 0;
            cur.bit_length_v[1] = -1;
        }
        if (cur.ty != layerType::FCONN)
            cur.need_phase2 = has_v_operand;
        cur.updateSize();
    }
}

void layeredCircuit::init(u8 q_bit_size, u8 _layer_sz) 
{
    size = _layer_sz;
    circuit.resize(size);
}
