#include "gat_ezkl/formal_hash.hpp"

#include <array>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace gat_ezkl {
namespace {

constexpr std::array<std::uint32_t, 64> K = {
    0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
    0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
    0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
    0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
    0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
    0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
    0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
    0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

std::uint32_t rotr(std::uint32_t x, unsigned n) { return (x >> n) | (x << (32 - n)); }

class Sha256 {
public:
    void update(const std::uint8_t* data, std::size_t size) {
        total_ += size;
        while (size != 0) {
            const auto take = std::min(size, block_.size() - used_);
            std::memcpy(block_.data() + used_, data, take);
            used_ += take; data += take; size -= take;
            if (used_ == block_.size()) { compress(block_.data()); used_ = 0; }
        }
    }
    std::array<std::uint8_t, 32> finish() {
        const std::uint64_t bit_length = static_cast<std::uint64_t>(total_) * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) { while (used_ < 64) block_[used_++] = 0; compress(block_.data()); used_ = 0; }
        while (used_ < 56) block_[used_++] = 0;
        for (int i = 7; i >= 0; --i) block_[used_++] = static_cast<std::uint8_t>(bit_length >> (8 * i));
        compress(block_.data());
        std::array<std::uint8_t, 32> out{};
        for (std::size_t i = 0; i < state_.size(); ++i)
            for (int j = 0; j < 4; ++j) out[i * 4 + j] = static_cast<std::uint8_t>(state_[i] >> (24 - 8 * j));
        return out;
    }
private:
    void compress(const std::uint8_t* input) {
        std::uint32_t w[64]{};
        for (std::size_t i = 0; i < 16; ++i)
            w[i] = (std::uint32_t(input[i*4]) << 24) | (std::uint32_t(input[i*4+1]) << 16) |
                   (std::uint32_t(input[i*4+2]) << 8) | input[i*4+3];
        for (std::size_t i = 16; i < 64; ++i) {
            const auto s0 = rotr(w[i-15],7) ^ rotr(w[i-15],18) ^ (w[i-15] >> 3);
            const auto s1 = rotr(w[i-2],17) ^ rotr(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        auto a=state_[0],b=state_[1],c=state_[2],d=state_[3],e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (std::size_t i=0;i<64;++i) {
            const auto s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g);
            const auto t1=h+s1+ch+K[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22);
            const auto maj=(a&b)^(a&c)^(b&c), t2=s0+maj;
            h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        state_[0]+=a;state_[1]+=b;state_[2]+=c;state_[3]+=d;
        state_[4]+=e;state_[5]+=f;state_[6]+=g;state_[7]+=h;
    }
    std::array<std::uint32_t,8> state_{{0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19}};
    std::array<std::uint8_t,64> block_{};
    std::size_t used_=0,total_=0;
};

void append_u64(std::vector<std::uint8_t>* out, std::uint64_t value) {
    for (unsigned i=0;i<8;++i) out->push_back(static_cast<std::uint8_t>(value >> (8*i)));
}
std::vector<std::uint8_t> text_bytes(const std::string& text) { return {text.begin(), text.end()}; }
std::vector<std::uint8_t> file_bytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to hash formal file: " + path.string());
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
std::vector<std::uint8_t> u64_vector(const std::vector<std::size_t>& values) {
    std::vector<std::uint8_t> out; append_u64(&out, values.size());
    for (const auto value: values) append_u64(&out, value);
    return out;
}
std::string hex_digest(const std::array<std::uint8_t,32>& digest) {
    std::ostringstream out; out << "sha256:" << std::hex << std::setfill('0');
    for (const auto byte: digest) out << std::setw(2) << unsigned(byte);
    return out.str();
}

}  // namespace

const char* canonical_model_schema() {
    return "ezkl_formal_model_parameters_v1";
}

const char* canonical_model_order() {
    return "hidden:(layer_index,local_head_index,global_head_index):seq_kernel,attn_src_kernel,attn_dst_kernel;output:(head_index):seq_kernel,attn_src_kernel,attn_dst_kernel,output_bias";
}

std::string canonical_model_order_digest() {
    return formal_sha256_text(
        "canonical_model_order", "descriptor",
        std::string(canonical_model_schema()) + "\n" + canonical_model_order());
}

std::string formal_sha256_bytes(
    const std::string& domain,
    const std::vector<std::pair<std::string, std::vector<std::uint8_t>>>& fields) {
    std::vector<std::uint8_t> canonical;
    const std::string prefix = "GAT-ZKML/ezkl/canonical/v1";
    append_u64(&canonical, prefix.size()); canonical.insert(canonical.end(), prefix.begin(), prefix.end());
    append_u64(&canonical, domain.size()); canonical.insert(canonical.end(), domain.begin(), domain.end());
    append_u64(&canonical, fields.size());
    for (const auto& [role, bytes] : fields) {
        append_u64(&canonical, role.size()); canonical.insert(canonical.end(), role.begin(), role.end());
        append_u64(&canonical, bytes.size()); canonical.insert(canonical.end(), bytes.begin(), bytes.end());
    }
    Sha256 sha; sha.update(canonical.data(), canonical.size()); return hex_digest(sha.finish());
}
std::string formal_sha256_text(const std::string& domain, const std::string& role, const std::string& text) {
    return formal_sha256_bytes(domain, {{role, text_bytes(text)}});
}
std::string formal_sha256_file(const std::string& domain, const std::string& role,
                               const std::filesystem::path& path) {
    return formal_sha256_bytes(domain, {{role, file_bytes(path)}});
}
std::string formal_hash_checkpoint(const std::filesystem::path& manifest,
                                   const std::filesystem::path& tensors) {
    return formal_sha256_bytes("checkpoint_bundle", {{"manifest",file_bytes(manifest)}, {"tensors",file_bytes(tensors)}});
}
std::string formal_hash_graph(const NormalizedGraph& graph) {
    std::vector<std::uint8_t> dimensions;
    for (const auto value : {graph.n_total,graph.node_count,graph.edge_count,graph.graph_count,graph.feature_dim,graph.class_count})
        append_u64(&dimensions, value);
    std::vector<std::int64_t> quantized_features;
    quantized_features.reserve(graph.node_count * graph.feature_dim);
    for (const auto& row: graph.features_fp) for (const auto value: row)
        quantized_features.push_back(static_cast<std::int64_t>(value >= 0 ? value + 0.5 : value - 0.5));
    std::vector<std::uint8_t> features; append_u64(&features, quantized_features.size());
    for (const auto value: quantized_features) append_u64(&features, static_cast<std::uint64_t>(value));
    return formal_sha256_bytes("public_graph", {{"dimensions",dimensions},{"absolute_ids",u64_vector(graph.absolute_ids)},
        {"node_ptr",u64_vector(graph.node_ptr)},{"edge_ptr",u64_vector(graph.edge_ptr)},
        {"src",u64_vector(graph.src)},{"dst",u64_vector(graph.dst)},{"features_i64",features}});
}
std::string formal_hash_shape(const NormalizedGraph& graph, const CheckpointBundle& bundle) {
    std::vector<std::uint8_t> payload;
    for (const auto value : {graph.node_count,graph.edge_count,graph.graph_count,graph.feature_dim,
                              graph.class_count,bundle.layer_count,bundle.K_out}) append_u64(&payload,value);
    append_u64(&payload,bundle.hidden_profile.size());
    for (const auto& layer: bundle.hidden_profile) { append_u64(&payload,layer.head_count); append_u64(&payload,layer.head_dim); }
    return formal_sha256_bytes("fixed_circuit_shape", {{"shape",payload}});
}
std::string formal_hash_metadata(const std::string& metadata_json) {
    return formal_sha256_text("adapter_metadata", "canonical_json", metadata_json);
}

}  // namespace gat_ezkl
