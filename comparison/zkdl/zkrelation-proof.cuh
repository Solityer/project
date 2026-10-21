#ifndef ZKDL_RELATION_PROOF_CUH
#define ZKDL_RELATION_PROOF_CUH

#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif

/*
 * Verifier-facing wire and host implementation for the GAT relation layer.
 *
 * This is intentionally a thin extension of the proof stack already used by
 * zkMatMul/zkProd: field values are FrWire, Fiat--Shamir is proof.cuh's
 * Transcript, the relation proof is proof.cuh's SumcheckProof and every
 * private tensor opening is cpu-backend.cuh's Hyrax opening.  There is no
 * second commitment, field, or transcript implementation here.
 *
 * The relation descriptor is supplied by the authenticated relation plan on
 * both sides.  A serialized proof repeats it only as an auditable wire and
 * the verifier compares the repeated descriptor byte-for-byte with the
 * expected descriptor before using any claim.  Consequently a prover cannot
 * change (for example) a product relation into a copy relation by editing
 * proof metadata.
 */

#include "cpu-backend.cuh"
#include "zkmatmul.cuh"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <iostream>
#include <openssl/hmac.h>
#include <iterator>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <type_traits>
#include <utility>
#include <unistd.h>
#include <vector>

namespace zkdl_gat {
namespace relation {

using byte = zkdl::byte;
using FrWire = zkdl::FrWire;
#if defined(ZKDL_USE_BLST_G1_CODEC)
using ZkSumcheckFoldedScalar = zkdl_blst::blst_fr;
inline FrWire zksc_folded_scalar_to_wire(const ZkSumcheckFoldedScalar& value) {
    return zkdl::cpu_ipa_montgomery_to_frwire(value);
}
#else
using ZkSumcheckFoldedScalar = FrWire;
inline FrWire zksc_folded_scalar_to_wire(const ZkSumcheckFoldedScalar& value) {
    return value;
}
#endif
using ZkSumcheckFoldedState = std::vector<std::vector<ZkSumcheckFoldedScalar>>;
using RouteWorkProgressCallback =
    std::function<void(const char*, std::uint64_t, std::uint64_t)>;
using SegmentCommitmentProgressCallback =
    std::function<void(std::uint32_t, std::uint32_t)>;

struct NativeOperationProgressReporter {
    using Clock = std::chrono::steady_clock;
    std::string phase;
    std::string operation;
    Clock::time_point started = Clock::now();
    Clock::time_point last_report = started;

    NativeOperationProgressReporter(std::string phase_name, std::string operation_id)
        : phase(std::move(phase_name)), operation(std::move(operation_id)) {}

    void update(const char* step, std::uint64_t completed, std::uint64_t total,
                std::uint32_t item_index = std::numeric_limits<std::uint32_t>::max(),
                std::uint32_t item_total = 0U,
                std::uint32_t round = 0U, std::uint32_t total_rounds = 0U) {
        const auto now = Clock::now();
        if (now - last_report < std::chrono::seconds(15)) return;
        std::clog << "zkdl_native_progress phase=" << phase << " status=PROGRESS"
                  << " operation_id=" << operation << " step=" << step;
        if (total_rounds != 0U) std::clog << " round=" << round << '/' << total_rounds;
        if (item_index != std::numeric_limits<std::uint32_t>::max())
            std::clog << " item_index=" << item_index + 1U << '/' << item_total;
        std::clog << " completed=" << completed << '/' << total
                  << " elapsed_wall_seconds="
                  << std::chrono::duration<double>(now - started).count()
                  << '\n' << std::flush;
        last_report = now;
    }
};

// Relation-family wire caps are deliberately finite.  They are large enough
// for the configured lookup/range domains, but never become an unbounded
// parser escape hatch for malformed proof bytes.
// Lookup compatibility tables are authenticated logical ranges padded to a
// native power-of-two MLE domain.  The multiplication tree therefore needs
// a constraint budget larger than the old 4096-entry parser cap for the
// shipped range_bits=12/14 configurations.
constexpr std::size_t kMaxRelationConstraints = 1U << 20;
constexpr std::size_t kMaxRelationTerms = 1U << 20;
constexpr std::size_t kMaxRelationOpenings = 4096U;
// Native tensor domains are admitted at most up to 2^30.  Point vectors are
// consequently bounded to the corresponding number of MLE coordinates; this
// is deliberately separate from the much larger opening-count cap above.
constexpr std::size_t kMaxRelationPointBits = 30U;
constexpr std::uint32_t kMaxRelationSegments = 1U << 12U;

// Public selector masks are often copied from authenticated plan descriptors
// into several operation views.  Keep the exact byte contents while allowing
// those views to share one immutable backing until a mutable access occurs.
class CowByteVector {
public:
    using value_type = std::uint8_t;
    using container_type = std::vector<value_type>;
    using iterator = container_type::iterator;
    using const_iterator = container_type::const_iterator;

    class reference {
    public:
        reference(CowByteVector* owner, std::size_t index)
            : owner_(owner), index_(index) {}
        reference& operator=(value_type value) {
            owner_->set_value(index_, value);
            return *this;
        }
        reference& operator=(const reference& other) {
            return *this = static_cast<value_type>(other);
        }
        operator value_type() const noexcept {
            return static_cast<const CowByteVector&>(*owner_)[index_];
        }

    private:
        CowByteVector* owner_ = nullptr;
        std::size_t index_ = 0U;
    };

    CowByteVector() = default;
    CowByteVector(std::initializer_list<value_type> values) { assign(values); }
    CowByteVector(container_type&& values) { assign(std::move(values)); }

    CowByteVector& operator=(std::initializer_list<value_type> values) {
        assign(values);
        return *this;
    }
    CowByteVector& operator=(container_type&& values) {
        assign(std::move(values));
        return *this;
    }

    void assign(std::size_t count, value_type value) {
        data_ = std::make_shared<container_type>(count, value);
    }
    void assign(std::initializer_list<value_type> values) {
        data_ = std::make_shared<container_type>(values);
    }
    void assign(container_type&& values) {
        data_ = std::make_shared<container_type>(std::move(values));
    }
    void resize(std::size_t count) {
        make_unique();
        data_->resize(count);
    }
    void clear() noexcept { data_.reset(); }
    void push_back(value_type value) {
        make_unique();
        data_->push_back(value);
    }
    void pop_back() {
        make_unique();
        data_->pop_back();
    }

    std::size_t size() const noexcept { return data_ ? data_->size() : 0U; }
    std::size_t capacity() const noexcept { return data_ ? data_->capacity() : 0U; }
    bool empty() const noexcept { return size() == 0U; }
    const void* backing_identity() const noexcept { return data_.get(); }
    std::uint64_t unique_capacity_bytes() const noexcept {
        return data_ && data_.unique()
            ? static_cast<std::uint64_t>(data_->capacity()) * sizeof(value_type)
            : 0U;
    }

    const value_type& operator[](std::size_t index) const noexcept {
        return (*data_)[index];
    }
    reference operator[](std::size_t index) noexcept { return reference(this, index); }
    const_iterator begin() const noexcept {
        static const container_type empty_values;
        return data_ ? data_->cbegin() : empty_values.cbegin();
    }
    const_iterator end() const noexcept {
        static const container_type empty_values;
        return data_ ? data_->cend() : empty_values.cend();
    }
    iterator begin() {
        make_unique();
        return data_->begin();
    }
    iterator end() {
        make_unique();
        return data_->end();
    }

private:
    void set_value(std::size_t index, value_type value) {
        make_unique();
        (*data_)[index] = value;
    }
    void make_unique() {
        if (!data_) data_ = std::make_shared<container_type>();
        else if (!data_.unique()) data_ = std::make_shared<container_type>(*data_);
    }

    std::shared_ptr<container_type> data_;
};

// A public index map is often the identity on a padded domain.  Keep that
// representation implicit in memory while retaining the historical wire
// encoding (a count followed by every u32 value).  Explicit maps remain
// byte-for-byte unchanged.
class IndexMapU32 {
public:
    using value_type = std::uint32_t;

    class reference {
    public:
        reference(IndexMapU32* owner, std::size_t index)
            : owner_(owner), index_(index) {}
        reference& operator=(std::uint32_t value) {
            owner_->set_value(index_, value);
            return *this;
        }
        reference& operator=(const reference& other) {
            return *this = static_cast<std::uint32_t>(other);
        }
        operator std::uint32_t() const noexcept {
            return static_cast<const IndexMapU32&>(*owner_)[index_];
        }

    private:
        IndexMapU32* owner_ = nullptr;
        std::size_t index_ = 0U;
    };

    class const_iterator {
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::uint32_t;
        using difference_type = std::ptrdiff_t;
        using pointer = void;
        using reference = std::uint32_t;

        const_iterator() = default;
        const_iterator(const IndexMapU32* owner, std::size_t index)
            : owner_(owner), index_(index) {}
        value_type operator*() const { return (*owner_)[index_]; }
        const_iterator& operator++() { ++index_; return *this; }
        const_iterator operator++(int) {
            const_iterator previous = *this;
            ++index_;
            return previous;
        }
        bool operator==(const const_iterator& other) const {
            return owner_ == other.owner_ && index_ == other.index_;
        }
        bool operator!=(const const_iterator& other) const {
            return !(*this == other);
        }

    private:
        const IndexMapU32* owner_ = nullptr;
        std::size_t index_ = 0U;
    };

    IndexMapU32() = default;
    IndexMapU32(std::initializer_list<std::uint32_t> values) { assign(values); }
    IndexMapU32(std::vector<std::uint32_t>&& values) { assign(std::move(values)); }

    static IndexMapU32 identity(std::size_t count) {
        IndexMapU32 result;
        result.set_identity(count);
        return result;
    }

    IndexMapU32& operator=(std::initializer_list<std::uint32_t> values) {
        assign(values);
        return *this;
    }
    IndexMapU32& operator=(std::vector<std::uint32_t>&& values) {
        assign(std::move(values));
        return *this;
    }

    void set_identity(std::size_t count) {
        explicit_.reset();
        identity_count_ = count;
    }
    void set_explicit_size(std::size_t count) {
        identity_count_ = 0U;
        explicit_ = std::make_shared<std::vector<std::uint32_t>>(count);
    }
    void assign(std::size_t count, std::uint32_t value) {
        identity_count_ = 0U;
        explicit_ = std::make_shared<std::vector<std::uint32_t>>(count, value);
        if (count == 1U && value == 0U) set_identity(1U);
    }
    void assign(std::initializer_list<std::uint32_t> values) {
        std::vector<std::uint32_t> copy(values);
        assign(std::move(copy));
    }
    void assign(std::vector<std::uint32_t>&& values) {
        bool is_identity = true;
        for (std::size_t index = 0U; index < values.size(); ++index) {
            if (values[index] != index) { is_identity = false; break; }
        }
        if (is_identity) {
            set_identity(values.size());
        } else {
            identity_count_ = 0U;
            explicit_ = std::make_shared<std::vector<std::uint32_t>>(std::move(values));
        }
    }
    void resize(std::size_t count) {
        materialize_identity();
        if (!explicit_) explicit_ = std::make_shared<std::vector<std::uint32_t>>();
        make_unique();
        explicit_->resize(count);
    }
    void clear() {
        identity_count_ = 0U;
        explicit_.reset();
    }

    std::size_t size() const noexcept {
        return identity_count_ != 0U ? identity_count_ : (explicit_ ? explicit_->size() : 0U);
    }
    bool empty() const noexcept { return size() == 0U; }
    std::size_t capacity() const noexcept { return explicit_ ? explicit_->capacity() : 0U; }
    bool is_identity() const noexcept { return identity_count_ != 0U; }
    const void* backing_identity() const noexcept { return explicit_.get(); }
    std::uint64_t resident_bytes() const noexcept {
        return static_cast<std::uint64_t>(capacity()) * sizeof(std::uint32_t);
    }
    std::uint64_t unique_capacity_bytes() const noexcept {
        return explicit_ && explicit_.unique() ? resident_bytes() : 0U;
    }

    std::uint32_t operator[](std::size_t index) const noexcept {
        return identity_count_ != 0U ? static_cast<std::uint32_t>(index)
                                     : (*explicit_)[index];
    }
    // Mutable reads must not expand an implicit identity map.  This proxy
    // materializes only when an assignment changes the indexed value.
    reference operator[](std::size_t index) noexcept { return reference(this, index); }

    const_iterator begin() const noexcept { return const_iterator(this, 0U); }
    const_iterator end() const noexcept { return const_iterator(this, size()); }

private:
    void set_value(std::size_t index, std::uint32_t value) {
        if (identity_count_ != 0U && index < identity_count_ && value == index)
            return;
        materialize_identity();
        if (!explicit_) explicit_ = std::make_shared<std::vector<std::uint32_t>>();
        make_unique();
        (*explicit_)[index] = value;
    }

    void make_unique() {
        if (explicit_ && !explicit_.unique())
            explicit_ = std::make_shared<std::vector<std::uint32_t>>(*explicit_);
    }
    void materialize_identity() {
        if (identity_count_ == 0U) return;
        explicit_ = std::make_shared<std::vector<std::uint32_t>>(identity_count_);
        for (std::size_t index = 0U; index < identity_count_; ++index)
            (*explicit_)[index] = static_cast<std::uint32_t>(index);
        identity_count_ = 0U;
    }

    std::size_t identity_count_ = 0U;
    std::shared_ptr<std::vector<std::uint32_t>> explicit_;
};

template <typename T>
inline bool checked_relation_u32(T value, std::uint32_t& output,
                                 const char* label,
                                 std::string* error = nullptr) {
    static_assert(std::is_integral<T>::value && std::is_unsigned<T>::value,
                  "checked_relation_u32 requires an unsigned integer");
    static_assert(sizeof(T) <= sizeof(std::uint64_t),
                  "checked_relation_u32 only supports integers up to uint64");
    const std::uint64_t wide = static_cast<std::uint64_t>(value);
    if (wide > std::numeric_limits<std::uint32_t>::max()) {
        if (error) *error = std::string(label) + " exceeds uint32";
        return false;
    }
    output = static_cast<std::uint32_t>(wide);
    return true;
}

inline bool parse_relation_u32_suffix(const std::string& text,
                                      std::uint32_t& output) {
    if (text.empty()) return false;
    std::uint64_t value = 0U;
    for (const char digit : text) {
        if (digit < '0' || digit > '9') return false;
        const std::uint64_t numeric = static_cast<std::uint64_t>(digit - '0');
        if (value > (std::numeric_limits<std::uint32_t>::max() - numeric) / 10U)
            return false;
        value = value * 10U + numeric;
    }
    output = static_cast<std::uint32_t>(value);
    return true;
}

struct ZkRelationSegmentLayout {
    std::uint32_t logical_domain = 0U;
    std::uint32_t basis_domain = 0U;
    std::uint32_t segment_count = 0U;
    std::uint32_t segment_size = 0U;
};

inline bool zktr_segment_layout_for_size(std::uint64_t logical_domain,
                                         std::size_t basis_size,
                                         std::uint32_t shared_segment_size,
                                         ZkRelationSegmentLayout& layout,
                                         std::string* error = nullptr) {
    if (logical_domain == 0U ||
        logical_domain > (std::uint64_t(1U) << kMaxRelationPointBits) ||
        logical_domain > std::numeric_limits<std::uint32_t>::max() ||
        (logical_domain & (logical_domain - 1U)) != 0U ||
        basis_size == 0U || basis_size > (1U << 18U) ||
        (basis_size & (basis_size - 1U)) != 0U ||
        shared_segment_size == 0U ||
        (shared_segment_size & (shared_segment_size - 1U)) != 0U ||
        shared_segment_size > basis_size ||
        shared_segment_size > logical_domain ||
        logical_domain % shared_segment_size != 0U ||
        logical_domain / shared_segment_size > kMaxRelationSegments) {
        if (error) *error = "invalid authenticated shared segment layout";
        return false;
    }
    layout.logical_domain = static_cast<std::uint32_t>(logical_domain);
    layout.basis_domain = shared_segment_size;
    layout.segment_size = shared_segment_size;
    layout.segment_count = static_cast<std::uint32_t>(logical_domain / shared_segment_size);
    return true;
}

inline bool zktr_segment_layout(std::uint64_t logical_domain,
                                std::size_t basis_size,
                                ZkRelationSegmentLayout& layout,
                                std::string* error = nullptr) {
    const std::uint32_t natural_size =
        logical_domain > std::numeric_limits<std::uint32_t>::max() ||
                basis_size > std::numeric_limits<std::uint32_t>::max()
            ? 0U
            : static_cast<std::uint32_t>(std::min<std::uint64_t>(
                  logical_domain, static_cast<std::uint64_t>(basis_size)));
    return zktr_segment_layout_for_size(logical_domain, basis_size, natural_size,
                                        layout, error);
}

/*
 * A relation witness is logically a power-of-two MLE, but the authenticated
 * IPA basis is intentionally bounded.  This provider is the seam between
 * those two domains.  Readers return only the requested logical range; the
 * provider pads the final range with canonical zeroes for the bounded IPA
 * opening.  The common vector adapter below captures an existing vector by
 * reference and therefore does not create a second logical-domain copy.
 */
class SegmentedTensorProvider {
public:
    using RangeReader = std::function<bool(
        std::uint64_t, std::uint32_t, std::vector<FrWire>&, std::string*)>;

    SegmentedTensorProvider() = default;

    SegmentedTensorProvider(std::uint32_t logical_domain,
                            std::uint32_t segment_size,
                            RangeReader reader)
        : logical_domain_(logical_domain), segment_size_(segment_size),
          segment_count_(segment_size == 0U ? 0U : logical_domain / segment_size),
          reader_(std::move(reader)) {}

    static bool shape(std::uint64_t logical_domain, std::size_t basis_size,
                      std::uint32_t& segment_size, std::uint32_t& segment_count,
                      std::string* error = nullptr) {
        if (logical_domain == 0U ||
            logical_domain > (std::uint64_t(1U) << kMaxRelationPointBits) ||
            logical_domain > std::numeric_limits<std::uint32_t>::max() ||
            (logical_domain & (logical_domain - 1U)) != 0U || basis_size == 0U ||
            basis_size > (1U << 18U) || (basis_size & (basis_size - 1U)) != 0U) {
            if (error) *error = "invalid segmented tensor provider shape";
            return false;
        }
        const std::uint64_t bounded = std::min<std::uint64_t>(logical_domain, basis_size);
        if (logical_domain % bounded != 0U ||
            bounded > std::numeric_limits<std::uint32_t>::max() ||
            logical_domain / bounded > kMaxRelationSegments) {
            if (error) *error = "segmented tensor provider coverage is not canonical";
            return false;
        }
        if (!checked_relation_u32(bounded, segment_size,
                                  "segmented tensor segment size", error) ||
            !checked_relation_u32(logical_domain / bounded, segment_count,
                                  "segmented tensor segment count", error))
            return false;
        return true;
    }

    static bool from_values(const std::vector<FrWire>& values, std::size_t basis_size,
                            SegmentedTensorProvider& provider,
                            std::string* error = nullptr) {
        std::uint32_t segment_size = 0U, segment_count = 0U;
        if (!shape(values.size(), basis_size, segment_size, segment_count, error)) return false;
        std::uint32_t logical_domain = 0U;
        if (!checked_relation_u32(values.size(), logical_domain,
                                  "segmented tensor value count", error))
            return false;
        const auto* value_source = &values;
        provider = SegmentedTensorProvider(
            logical_domain, segment_size,
            [value_source](std::uint64_t offset, std::uint32_t count,
                      std::vector<FrWire>& output, std::string* read_error) {
                if (offset > value_source->size() || count > value_source->size() - offset) {
                    if (read_error) *read_error = "segmented tensor vector range is out of bounds";
                    return false;
                }
                output.assign(value_source->begin() + static_cast<std::ptrdiff_t>(offset),
                              value_source->begin() + static_cast<std::ptrdiff_t>(offset + count));
                return true;
            });
        (void)segment_count;
        return true;
    }

    bool valid(std::string* error = nullptr) const {
        if (logical_domain_ == 0U || segment_size_ == 0U || segment_count_ == 0U ||
            static_cast<std::uint64_t>(logical_domain_) !=
                static_cast<std::uint64_t>(segment_size_) * segment_count_ ||
            !reader_) {
            if (error) *error = "invalid segmented tensor provider";
            return false;
        }
        return true;
    }

    std::uint32_t logical_domain() const noexcept { return logical_domain_; }
    std::uint32_t segment_size() const noexcept { return segment_size_; }
    std::uint32_t segment_count() const noexcept { return segment_count_; }

    bool evaluate_mle(const std::vector<FrWire>& point, FrWire& output, std::string* error = nullptr,
                      NativeOperationProgressReporter* progress = nullptr) const {
        if (!valid(error) || point.size() >= 31U || (std::uint64_t(1U) << point.size()) != logical_domain_) { if (error) *error = "segmented MLE point/domain mismatch"; return false; }
        std::uint32_t low_bits = 0U;
        for (auto size = segment_size_; size > 1U; size >>= 1U) ++low_bits;
        const std::vector<FrWire> high(point.begin(), point.end() - low_bits), low(point.end() - low_bits, point.end());
        const auto high_weights = zkdl::cpu_ipa_mle_weights(high), low_weights = zkdl::cpu_ipa_mle_weights(low);
        output = FrWire{};
        for (std::uint32_t segment = 0U; segment < segment_count_; ++segment) { std::vector<FrWire> values; if (!read_segment(segment, values, error)) return false; output = zkdl::fr_add(output, zkdl::fr_mul(high_weights[segment], zkdl::cpu_ipa_inner_product(values, low_weights))); if (progress != nullptr) progress->update("mle_segments", segment + 1U, segment_count_); }
        return true;
    }

    bool read_segment(std::uint32_t segment, std::vector<FrWire>& output,
                      std::string* error = nullptr) const {
        if (!valid(error) || segment >= segment_count_) {
            if (error && error->empty()) *error = "segmented tensor segment index is invalid";
            return false;
        }
        const std::uint64_t offset = static_cast<std::uint64_t>(segment) * segment_size_;
        std::uint32_t actual = 0U;
        if (!checked_relation_u32(
                std::min<std::uint64_t>(segment_size_, logical_domain_ - offset),
                actual, "segmented tensor actual segment length", error))
            return false;
        std::vector<FrWire> raw;
        if (!reader_(offset, actual, raw, error) || raw.size() != actual) {
            if (error && error->empty()) *error = "segmented tensor reader returned the wrong length";
            return false;
        }
        for (const auto& value : raw) {
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "segmented tensor reader returned a non-canonical field value";
                return false;
            }
        }
        if (raw.size() != segment_size_) {
            if (error) *error = "segmented tensor provider returned a partial authenticated segment";
            return false;
        }
        output = std::move(raw);
        return true;
    }

private:
    std::uint32_t logical_domain_ = 0U;
    std::uint32_t segment_size_ = 0U;
    std::uint32_t segment_count_ = 0U;
    RangeReader reader_;
};

constexpr std::uint64_t kZkSumcheckFoldedStateBudgetBytes =
    std::uint64_t{32U} << 30U;

inline bool zksc_montgomery_row_scratch_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t row_count,
    std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (tensor_count == 0U || tensor_count >
            (std::numeric_limits<std::uint64_t>::max() - 9U) / 3U)
        return false;
    row_count = std::max<std::uint64_t>(1U, row_count);
    const std::uint64_t worker_chunks =
        static_cast<std::uint64_t>(zkdl::cpu_parallel_worker_count()) + 1U;
    const std::uint64_t chunks = row_count < 4096U
        ? 1U : std::min(row_count, worker_chunks);
    const std::uint64_t fields_per_chunk = tensor_count * 3U + 9U;
    if (fields_per_chunk > std::numeric_limits<std::uint64_t>::max() /
                               sizeof(zkdl_blst::blst_fr))
        return false;
    const std::uint64_t fields_bytes =
        fields_per_chunk * sizeof(zkdl_blst::blst_fr);
    const std::uint64_t vector_headers_per_chunk =
        2U * sizeof(std::vector<zkdl_blst::blst_fr>) + sizeof(unsigned char);
    if (chunks > std::numeric_limits<std::uint64_t>::max() / fields_bytes ||
        chunks > std::numeric_limits<std::uint64_t>::max() /
                     vector_headers_per_chunk)
        return false;
    const std::uint64_t payload_bytes = chunks * fields_bytes;
    const std::uint64_t metadata_bytes = chunks * vector_headers_per_chunk;
    if (payload_bytes > std::numeric_limits<std::uint64_t>::max() - metadata_bytes)
        return false;
    peak_bytes = payload_bytes + metadata_bytes;
#else
    (void)tensor_count;
    (void)row_count;
#endif
    return true;
}

inline bool zksc_segmented_streaming_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t segment_size,
    std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (tensor_count == 0U || segment_size == 0U)
        return false;
    const auto multiply = [](std::uint64_t left, std::uint64_t right,
                             std::uint64_t& product) noexcept {
        if (left != 0U && right > std::numeric_limits<std::uint64_t>::max() / left)
            return false;
        product = left * right;
        return true;
    };
    const auto add = [](std::uint64_t left, std::uint64_t right,
                        std::uint64_t& sum) noexcept {
        if (right > std::numeric_limits<std::uint64_t>::max() - left)
            return false;
        sum = left + right;
        return true;
    };
    std::uint64_t segment_values = 0U, scratch_values = 0U;
    std::uint64_t payload_bytes = 0U;
    std::uint64_t vector_headers = 0U, terminal_values = 0U;
    std::uint64_t metadata_bytes = 0U;
    if (!multiply(tensor_count, segment_size, segment_values) ||
        !multiply(segment_values, 4U, scratch_values) ||
        !multiply(scratch_values, sizeof(FrWire), payload_bytes) ||
        !multiply(tensor_count, sizeof(std::vector<FrWire>) +
                  sizeof(std::vector<ZkSumcheckFoldedScalar>), vector_headers) ||
        !multiply(tensor_count, sizeof(FrWire), terminal_values) ||
        !add(vector_headers, terminal_values, metadata_bytes) ||
        !add(payload_bytes, metadata_bytes, peak_bytes)) {
        peak_bytes = 0U;
        return false;
    }
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::uint64_t montgomery_scratch_bytes = 0U;
    if (!zksc_montgomery_row_scratch_peak_bytes(
            tensor_count, segment_size / 2U, montgomery_scratch_bytes) ||
        montgomery_scratch_bytes >
            std::numeric_limits<std::uint64_t>::max() - peak_bytes) {
        peak_bytes = 0U;
        return false;
    }
    peak_bytes += montgomery_scratch_bytes;
#endif
    return true;
}

inline bool zksc_folded_state_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t segment_size, std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (tensor_count == 0U || logical_domain < 2U) return false;
    std::uint64_t streaming_bytes = 0U, cached_values = 0U, cached_bytes = 0U;
    std::uint64_t cached_round_scratch_bytes = 0U;
    if (!zksc_segmented_streaming_peak_bytes(
            tensor_count, segment_size, streaming_bytes) ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (logical_domain / 2U))
        return false;
    cached_values = tensor_count * (logical_domain / 2U);
#if defined(ZKDL_USE_BLST_G1_CODEC)
    constexpr std::uint64_t kCachedScalarBytes = sizeof(zkdl_blst::blst_fr);
#else
    constexpr std::uint64_t kCachedScalarBytes = sizeof(FrWire);
#endif
    if (cached_values > std::numeric_limits<std::uint64_t>::max() / kCachedScalarBytes)
        return false;
    cached_bytes = cached_values * kCachedScalarBytes;
    if (tensor_count > (std::numeric_limits<std::uint64_t>::max() -
                        sizeof(ZkSumcheckFoldedState)) /
                           sizeof(std::vector<ZkSumcheckFoldedScalar>))
        return false;
    const std::uint64_t folded_headers = sizeof(ZkSumcheckFoldedState) +
        tensor_count * sizeof(std::vector<ZkSumcheckFoldedScalar>);
    if (folded_headers > std::numeric_limits<std::uint64_t>::max() - cached_bytes)
        return false;
    cached_bytes += folded_headers;
    if (logical_domain > 2U &&
        !zksc_montgomery_row_scratch_peak_bytes(
            tensor_count, logical_domain / 4U, cached_round_scratch_bytes))
        return false;
    const std::uint64_t phase_scratch_bytes =
        std::max(streaming_bytes, cached_round_scratch_bytes);
    if (phase_scratch_bytes >
        std::numeric_limits<std::uint64_t>::max() - cached_bytes)
        return false;
    peak_bytes = cached_bytes + phase_scratch_bytes;
    return true;
}

inline bool zksc_spilled_state_ram_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t segment_size, std::uint64_t& peak_bytes) noexcept;

inline bool zksc_selected_sumcheck_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t segment_size, std::uint64_t montgomery_plan_peak_bytes,
    std::uint64_t& peak_bytes,
    std::uint64_t extra_live_bytes = 0U) noexcept {
    if (!zksc_folded_state_peak_bytes(
            tensor_count, logical_domain, segment_size, peak_bytes))
        return false;
    if (montgomery_plan_peak_bytes >
        std::numeric_limits<std::uint64_t>::max() - peak_bytes ||
        extra_live_bytes > std::numeric_limits<std::uint64_t>::max() -
                               peak_bytes - montgomery_plan_peak_bytes) return false;
    if (peak_bytes + montgomery_plan_peak_bytes + extra_live_bytes <=
        kZkSumcheckFoldedStateBudgetBytes) {
        peak_bytes += montgomery_plan_peak_bytes + extra_live_bytes;
        return true;
    }
    std::uint64_t provider_peak = 0U, spill_peak = 0U;
    if (!zksc_segmented_streaming_peak_bytes(
            tensor_count, segment_size, provider_peak) ||
        !zksc_spilled_state_ram_peak_bytes(
            tensor_count, logical_domain, segment_size, spill_peak)) return false;
    peak_bytes = std::max(provider_peak, spill_peak);
    if (montgomery_plan_peak_bytes >
            std::numeric_limits<std::uint64_t>::max() - peak_bytes ||
        extra_live_bytes > std::numeric_limits<std::uint64_t>::max() -
                               peak_bytes - montgomery_plan_peak_bytes) return false;
    peak_bytes += montgomery_plan_peak_bytes + extra_live_bytes;
    return true;
}

inline bool zksc_folded_state_within_budget(
    const std::vector<SegmentedTensorProvider>& providers,
    std::uint64_t montgomery_plan_peak_bytes = 0U,
    std::uint64_t extra_live_bytes = 0U) noexcept {
    if (providers.empty()) return false;
    std::uint64_t peak_bytes = 0U;
    return zksc_folded_state_peak_bytes(
               providers.size(), providers.front().logical_domain(),
               providers.front().segment_size(), peak_bytes) &&
           montgomery_plan_peak_bytes <=
               std::numeric_limits<std::uint64_t>::max() - peak_bytes &&
           extra_live_bytes <= std::numeric_limits<std::uint64_t>::max() -
                                   peak_bytes - montgomery_plan_peak_bytes &&
           peak_bytes + montgomery_plan_peak_bytes + extra_live_bytes <=
               kZkSumcheckFoldedStateBudgetBytes;
}

constexpr std::uint64_t kZkSumcheckSpillIoAlignment = 4096U;
constexpr std::uint64_t kZkSumcheckSpillIoRows = 4096U;
constexpr std::uint64_t kZkSumcheckSpillRamBudgetBytes = std::uint64_t{1U} << 30U;
// The largest current range provider decodes one 107-byte sorted scratch row
// into a 32-byte FrWire; include both live buffers and an extra fixed reserve.
constexpr std::uint64_t kZkSumcheckSpillMaxProviderRecordBytes = 107U;
constexpr std::uint64_t kZkSumcheckSpillProviderScratchReserveBytes =
    std::uint64_t{16U} << 20U;
static_assert(sizeof(FrWire) == 32U,
              "Sumcheck spill canonical FrWire encoding requires 32-byte field wires");

inline bool zksc_spill_power_of_two(std::uint64_t value) noexcept {
    return value != 0U && (value & (value - 1U)) == 0U;
}

inline bool zksc_spill_round_up(std::uint64_t value, std::uint64_t alignment,
                                std::uint64_t& output) noexcept {
    if (alignment == 0U || (alignment & (alignment - 1U)) != 0U ||
        value > std::numeric_limits<std::uint64_t>::max() - (alignment - 1U))
        return false;
    output = (value + alignment - 1U) & ~(alignment - 1U);
    return true;
}

inline bool zksc_spilled_state_disk_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (tensor_count == 0U || logical_domain < 4U ||
        !zksc_spill_power_of_two(logical_domain)) return false;
    std::uint64_t first_values = 0U, next_values = 0U;
    std::uint64_t first_bytes = 0U, next_bytes = 0U;
    if (tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (logical_domain / 2U) ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (logical_domain / 4U)) return false;
    first_values = tensor_count * (logical_domain / 2U);
    next_values = tensor_count * (logical_domain / 4U);
    if (first_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        next_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire))
        return false;
    first_bytes = first_values * sizeof(FrWire);
    next_bytes = next_values * sizeof(FrWire);
    if (!zksc_spill_round_up(first_bytes, kZkSumcheckSpillIoAlignment,
                             first_bytes) ||
        !zksc_spill_round_up(next_bytes, kZkSumcheckSpillIoAlignment,
                             next_bytes) ||
        first_bytes > std::numeric_limits<std::uint64_t>::max() - next_bytes)
        return false;
    peak_bytes = first_bytes + next_bytes;
    return true;
}

inline bool zksc_spilled_state_io_bytes_upper_bound(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t& io_bytes) noexcept {
    io_bytes = 0U;
    if (tensor_count == 0U || logical_domain < 4U ||
        !zksc_spill_power_of_two(logical_domain)) return false;
    std::uint64_t values = 0U, payload_bytes = 0U;
    if (tensor_count > std::numeric_limits<std::uint64_t>::max() / logical_domain)
        return false;
    values = tensor_count * logical_domain;
    if (values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire))
        return false;
    payload_bytes = values * sizeof(FrWire);
    if (payload_bytes > std::numeric_limits<std::uint64_t>::max() / 3U ||
        logical_domain > std::numeric_limits<std::uint64_t>::max() / 4U)
        return false;
    std::uint64_t bits = 0U;
    for (std::uint64_t value = logical_domain; value > 1U; value >>= 1U) ++bits;
    if (logical_domain > (std::numeric_limits<std::uint64_t>::max() -
                          (bits + 2U) * 8192U) / 4U ||
        bits > (std::numeric_limits<std::uint64_t>::max() -
                logical_domain * 4U) / 12288U) return false;
    const std::uint64_t alignment_read_overhead = logical_domain * 4U +
                                                   (bits + 2U) * 8192U;
    const std::uint64_t file_tail_overhead = bits * 4096U;
    const std::uint64_t spill_payload_io = payload_bytes * 3U;
    if (alignment_read_overhead > std::numeric_limits<std::uint64_t>::max() -
                                      file_tail_overhead ||
        spill_payload_io > std::numeric_limits<std::uint64_t>::max() -
                               alignment_read_overhead - file_tail_overhead) return false;
    io_bytes = spill_payload_io + alignment_read_overhead + file_tail_overhead;
    return true;
}

inline bool zksc_spilled_state_ram_peak_bytes(
    std::uint64_t tensor_count, std::uint64_t logical_domain,
    std::uint64_t segment_size, std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (tensor_count == 0U || logical_domain < 4U ||
        !zksc_spill_power_of_two(logical_domain) || segment_size < 2U ||
        !zksc_spill_power_of_two(segment_size) || segment_size > logical_domain ||
        logical_domain % segment_size != 0U) return false;

    std::uint64_t init_tile_values = 0U, init_tile_bytes = 0U;
    std::uint64_t provider_decode_bytes = 0U;
    if (tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (segment_size / 2U)) return false;
    init_tile_values = tensor_count * (segment_size / 2U);
    if (init_tile_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        segment_size > (std::numeric_limits<std::uint64_t>::max() -
                        kZkSumcheckSpillProviderScratchReserveBytes) /
                           (kZkSumcheckSpillMaxProviderRecordBytes + sizeof(FrWire)))
        return false;
    init_tile_bytes = init_tile_values * sizeof(FrWire);
    provider_decode_bytes = segment_size *
        (kZkSumcheckSpillMaxProviderRecordBytes + sizeof(FrWire)) +
        kZkSumcheckSpillProviderScratchReserveBytes;
    if (!zksc_spill_round_up(init_tile_bytes, kZkSumcheckSpillIoAlignment,
                             init_tile_bytes) ||
        provider_decode_bytes > std::numeric_limits<std::uint64_t>::max() -
                                    kZkSumcheckSpillIoAlignment ||
        init_tile_bytes > std::numeric_limits<std::uint64_t>::max() -
                              provider_decode_bytes - kZkSumcheckSpillIoAlignment)
        return false;
    const std::uint64_t initialize_peak = init_tile_bytes + provider_decode_bytes +
        kZkSumcheckSpillIoAlignment;

    const std::uint64_t max_io_rows = std::min<std::uint64_t>(
        logical_domain / 4U, kZkSumcheckSpillIoRows);
    std::uint64_t input_values = 0U, output_values = 0U;
    if (max_io_rows == 0U ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (2U * max_io_rows) ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           max_io_rows) return false;
    input_values = tensor_count * (2U * max_io_rows);
    output_values = tensor_count * max_io_rows;
    if (input_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        output_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire))
        return false;
    std::uint64_t io_peak = input_values * sizeof(FrWire);
    std::uint64_t output_bytes = output_values * sizeof(FrWire);
    if (!zksc_spill_round_up(io_peak, kZkSumcheckSpillIoAlignment, io_peak) ||
        !zksc_spill_round_up(output_bytes, kZkSumcheckSpillIoAlignment, output_bytes) ||
        output_bytes > std::numeric_limits<std::uint64_t>::max() -
                           2U * kZkSumcheckSpillIoAlignment ||
        io_peak > std::numeric_limits<std::uint64_t>::max() - output_bytes -
                      2U * kZkSumcheckSpillIoAlignment)
        return false;
    io_peak += output_bytes + 2U * kZkSumcheckSpillIoAlignment;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::uint64_t montgomery_scratch_bytes = 0U;
    if (!zksc_montgomery_row_scratch_peak_bytes(
            tensor_count, max_io_rows, montgomery_scratch_bytes) ||
        montgomery_scratch_bytes >
            std::numeric_limits<std::uint64_t>::max() - io_peak)
        return false;
    io_peak += montgomery_scratch_bytes;
#endif
    peak_bytes = std::max(initialize_peak, io_peak);
    return peak_bytes <= kZkSumcheckSpillRamBudgetBytes;
}

class ZkSumcheckAlignedBuffer {
public:
    ZkSumcheckAlignedBuffer() = default;
    ZkSumcheckAlignedBuffer(const ZkSumcheckAlignedBuffer&) = delete;
    ZkSumcheckAlignedBuffer& operator=(const ZkSumcheckAlignedBuffer&) = delete;
    ZkSumcheckAlignedBuffer(ZkSumcheckAlignedBuffer&& other) noexcept
        : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0U;
    }
    ZkSumcheckAlignedBuffer& operator=(ZkSumcheckAlignedBuffer&& other) noexcept {
        if (this != &other) {
            std::free(data_);
            data_ = other.data_;
            size_ = other.size_;
            other.data_ = nullptr;
            other.size_ = 0U;
        }
        return *this;
    }
    ~ZkSumcheckAlignedBuffer() { std::free(data_); }

    bool allocate(std::uint64_t bytes, std::string* error = nullptr) {
        std::uint64_t allocation = 0U;
        if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max() ||
            !zksc_spill_round_up(bytes, kZkSumcheckSpillIoAlignment, allocation) ||
            allocation > std::numeric_limits<std::size_t>::max()) {
            if (error) *error = "Sumcheck spill aligned-buffer size is invalid";
            return false;
        }
        std::free(data_);
        data_ = nullptr;
        size_ = 0U;
        void* aligned = nullptr;
        const int status = ::posix_memalign(
            &aligned, static_cast<std::size_t>(kZkSumcheckSpillIoAlignment),
            static_cast<std::size_t>(allocation));
        if (status != 0) {
            if (error) *error = "Sumcheck spill aligned-buffer allocation failed: " +
                                std::to_string(status);
            return false;
        }
        data_ = aligned;
        size_ = static_cast<std::size_t>(allocation);
        std::memset(data_, 0, size_);
        return true;
    }
    void* data() noexcept { return data_; }
    const void* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    FrWire* wires() noexcept { return static_cast<FrWire*>(data_); }
    const FrWire* wires() const noexcept { return static_cast<const FrWire*>(data_); }

private:
    void* data_ = nullptr;
    std::size_t size_ = 0U;
};

class ZkSumcheckFoldedSpillFile {
public:
    ZkSumcheckFoldedSpillFile() = default;
    ZkSumcheckFoldedSpillFile(const ZkSumcheckFoldedSpillFile&) = delete;
    ZkSumcheckFoldedSpillFile& operator=(const ZkSumcheckFoldedSpillFile&) = delete;
    ZkSumcheckFoldedSpillFile(ZkSumcheckFoldedSpillFile&& other) noexcept
        : fd_(other.fd_), directory_(std::move(other.directory_)),
          operation_identity_(std::move(other.operation_identity_)),
          logical_domain_(other.logical_domain_), tensor_count_(other.tensor_count_),
          logical_rows_(other.logical_rows_), round_index_(other.round_index_),
          logical_bytes_(other.logical_bytes_), allocated_bytes_(other.allocated_bytes_),
          logical_written_(other.logical_written_), physical_written_(other.physical_written_),
          staging_(std::move(other.staging_)), staging_used_(other.staging_used_),
          complete_(other.complete_) {
        other.fd_ = -1;
        other.logical_domain_ = 0U;
        other.tensor_count_ = 0U;
        other.logical_rows_ = 0U;
        other.logical_bytes_ = 0U;
        other.allocated_bytes_ = 0U;
        other.logical_written_ = 0U;
        other.physical_written_ = 0U;
        other.staging_used_ = 0U;
        other.complete_ = false;
    }
    ZkSumcheckFoldedSpillFile& operator=(ZkSumcheckFoldedSpillFile&& other) noexcept {
        if (this != &other) {
            close_fd();
            fd_ = other.fd_;
            directory_ = std::move(other.directory_);
            operation_identity_ = std::move(other.operation_identity_);
            logical_domain_ = other.logical_domain_;
            tensor_count_ = other.tensor_count_;
            logical_rows_ = other.logical_rows_;
            round_index_ = other.round_index_;
            logical_bytes_ = other.logical_bytes_;
            allocated_bytes_ = other.allocated_bytes_;
            logical_written_ = other.logical_written_;
            physical_written_ = other.physical_written_;
            staging_ = std::move(other.staging_);
            staging_used_ = other.staging_used_;
            complete_ = other.complete_;
            other.fd_ = -1;
            other.logical_domain_ = 0U;
            other.tensor_count_ = 0U;
            other.logical_rows_ = 0U;
            other.logical_bytes_ = 0U;
            other.allocated_bytes_ = 0U;
            other.logical_written_ = 0U;
            other.physical_written_ = 0U;
            other.staging_used_ = 0U;
            other.complete_ = false;
        }
        return *this;
    }
    ~ZkSumcheckFoldedSpillFile() { close_fd(); }

    static bool create(const std::string& directory,
                       const std::string& operation_identity,
                       std::uint64_t logical_domain,
                       std::uint64_t tensor_count,
                       std::uint64_t logical_rows,
                       std::uint32_t round_index,
                       ZkSumcheckFoldedSpillFile& output,
                       std::string* error = nullptr) {
        output = ZkSumcheckFoldedSpillFile{};
#if !defined(__BYTE_ORDER__) || !defined(__ORDER_LITTLE_ENDIAN__) || \
    __BYTE_ORDER__ != __ORDER_LITTLE_ENDIAN__
        if (error) *error = "Sumcheck spill requires canonical little-endian FrWire storage";
        return false;
#endif
        if (directory.empty() || operation_identity.empty() ||
            logical_domain < 2U || !zksc_spill_power_of_two(logical_domain) ||
            tensor_count == 0U || logical_rows == 0U ||
            logical_rows > logical_domain ||
            tensor_count > std::numeric_limits<std::uint64_t>::max() / logical_rows) {
            if (error) *error = "Sumcheck spill file identity or shape is invalid";
            return false;
        }
        const std::uint64_t elements = tensor_count * logical_rows;
        if (elements > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire)) {
            if (error) *error = "Sumcheck spill file element byte count overflows";
            return false;
        }
        const std::uint64_t logical_bytes = elements * sizeof(FrWire);
        std::uint64_t allocated_bytes = 0U;
        if (!zksc_spill_round_up(logical_bytes, kZkSumcheckSpillIoAlignment,
                                 allocated_bytes) ||
            allocated_bytes > static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
            if (error) *error = "Sumcheck spill file size exceeds the host file-offset range";
            return false;
        }
        const int directory_fd = ::open(
            directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW);
        if (directory_fd < 0) {
            if (error) *error = "Sumcheck spill directory open failed: errno=" +
                                std::to_string(errno);
            return false;
        }
        struct stat directory_stat{};
        const bool directory_valid = ::fstat(directory_fd, &directory_stat) == 0 &&
            S_ISDIR(directory_stat.st_mode) && directory_stat.st_uid == ::geteuid() &&
            (directory_stat.st_mode & 0777U) == 0700U;
        if (!directory_valid) {
            const int saved_errno = errno;
            ::close(directory_fd);
            if (error) *error = "Sumcheck spill directory owner/mode/type is not canonical" +
                                (saved_errno != 0 ? ": errno=" + std::to_string(saved_errno) : "");
            return false;
        }
        const int fd = ::openat(directory_fd, ".",
            O_TMPFILE | O_RDWR | O_CLOEXEC | O_DIRECT, 0600);
        const int open_errno = errno;
        struct stat file_stat{};
        const bool file_valid = fd >= 0 && ::fstat(fd, &file_stat) == 0 &&
            S_ISREG(file_stat.st_mode) && file_stat.st_uid == ::geteuid() &&
            (file_stat.st_mode & 0777U) == 0600U &&
            file_stat.st_nlink == 0 &&
            file_stat.st_dev == directory_stat.st_dev;
        ::close(directory_fd);
        if (!file_valid) {
            if (fd >= 0) ::close(fd);
            if (error) *error = "Sumcheck anonymous direct-I/O file creation failed closed: errno=" +
                                std::to_string(open_errno);
            return false;
        }
        const int allocation_status = ::posix_fallocate(
            fd, 0, static_cast<off_t>(allocated_bytes));
        struct stat allocated_stat{};
        if (allocation_status != 0 || ::fstat(fd, &allocated_stat) != 0 ||
            allocated_stat.st_size != static_cast<off_t>(allocated_bytes)) {
            const int saved_errno = allocation_status != 0 ? allocation_status : errno;
            ::close(fd);
            if (error) *error = "Sumcheck spill file exact preallocation failed closed: errno=" +
                                std::to_string(saved_errno);
            return false;
        }
        output.fd_ = fd;
        output.directory_ = directory;
        output.operation_identity_ = operation_identity;
        output.logical_domain_ = logical_domain;
        output.tensor_count_ = tensor_count;
        output.logical_rows_ = logical_rows;
        output.round_index_ = round_index;
        output.logical_bytes_ = logical_bytes;
        output.allocated_bytes_ = allocated_bytes;
        if (!output.staging_.allocate(kZkSumcheckSpillIoAlignment, error)) {
            output.close_fd();
            return false;
        }
        return true;
    }

    bool append_values(const FrWire* values, std::uint64_t count,
                       std::string* error = nullptr) {
        if (fd_ < 0 || complete_ || values == nullptr || count == 0U ||
            count > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire)) {
            if (error) *error = "Sumcheck spill append state is invalid";
            return false;
        }
        const std::uint64_t bytes = count * sizeof(FrWire);
        if (logical_written_ > logical_bytes_ || bytes > logical_bytes_ - logical_written_) {
            if (error) *error = "Sumcheck spill append exceeds its canonical prefix length";
            return false;
        }
        const auto* source = reinterpret_cast<const unsigned char*>(values);
        std::uint64_t remaining = bytes;
        while (remaining != 0U) {
            if (staging_used_ == 0U && remaining >= kZkSumcheckSpillIoAlignment &&
                (reinterpret_cast<std::uintptr_t>(source) % kZkSumcheckSpillIoAlignment) == 0U) {
                const std::uint64_t direct_bytes =
                    remaining & ~(kZkSumcheckSpillIoAlignment - 1U);
                if (!write_direct(source, direct_bytes, error)) return false;
                source += direct_bytes;
                remaining -= direct_bytes;
                logical_written_ += direct_bytes;
                continue;
            }
            const std::size_t available = kZkSumcheckSpillIoAlignment - staging_used_;
            const std::size_t copied = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, available));
            std::memcpy(static_cast<unsigned char*>(staging_.data()) + staging_used_,
                        source, copied);
            staging_used_ += copied;
            source += copied;
            remaining -= copied;
            logical_written_ += copied;
            if (staging_used_ == kZkSumcheckSpillIoAlignment) {
                if (!write_direct(staging_.data(), staging_used_, error)) return false;
                staging_used_ = 0U;
            }
        }
        return true;
    }

    bool finish(std::string* error = nullptr) {
        if (fd_ < 0 || complete_ || logical_written_ != logical_bytes_) {
            if (error) *error = "Sumcheck spill file did not cover its complete logical prefix";
            return false;
        }
        if (staging_used_ != 0U) {
            std::memset(static_cast<unsigned char*>(staging_.data()) + staging_used_,
                        0, kZkSumcheckSpillIoAlignment - staging_used_);
            if (!write_direct(staging_.data(), kZkSumcheckSpillIoAlignment, error)) return false;
            staging_used_ = 0U;
        }
        struct stat status{};
        if (physical_written_ != allocated_bytes_ || ::fstat(fd_, &status) != 0 ||
            status.st_size != static_cast<off_t>(allocated_bytes_)) {
            if (error) *error = "Sumcheck spill file exact length/complete marker validation failed";
            return false;
        }
        complete_ = true;
        return true;
    }

    bool read_values(std::uint64_t offset, std::uint64_t count,
                     FrWire* output, std::uint64_t output_capacity,
                     const std::string& expected_operation,
                     std::uint64_t expected_domain,
                     std::uint64_t expected_tensor_count,
                     std::uint64_t expected_rows,
                     std::uint32_t expected_round,
                     std::string* error = nullptr) const {
        if (!validate_complete(expected_operation, expected_domain,
                               expected_tensor_count, expected_rows,
                               expected_round, error) ||
            output == nullptr || count == 0U || count > output_capacity ||
            offset > tensor_count_ * logical_rows_ ||
            count > tensor_count_ * logical_rows_ - offset ||
            count > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire)) {
            if (error && error->empty()) *error = "Sumcheck spill read range is invalid";
            return false;
        }
        const std::uint64_t byte_offset = offset * sizeof(FrWire);
        const std::uint64_t byte_count = count * sizeof(FrWire);
        std::uint64_t read_end = 0U;
        if (byte_offset > std::numeric_limits<std::uint64_t>::max() - byte_count ||
            !zksc_spill_round_up(byte_offset + byte_count,
                                 kZkSumcheckSpillIoAlignment, read_end)) {
            if (error) *error = "Sumcheck spill read byte range overflows";
            return false;
        }
        const std::uint64_t read_begin = byte_offset &
            ~(kZkSumcheckSpillIoAlignment - 1U);
        if (read_end > allocated_bytes_ || read_end < read_begin) {
            if (error) *error = "Sumcheck spill padded read exceeds its preallocated extent";
            return false;
        }
        const bool direct = read_begin == byte_offset &&
            read_end - read_begin == byte_count &&
            (reinterpret_cast<std::uintptr_t>(output) % kZkSumcheckSpillIoAlignment) == 0U;
        ZkSumcheckAlignedBuffer temporary;
        void* destination = output;
        if (!direct) {
            if (!temporary.allocate(read_end - read_begin, error)) return false;
            destination = temporary.data();
        }
        if (!read_direct(destination, read_end - read_begin, read_begin, error)) return false;
        if (!direct) {
            const std::uint64_t prefix = byte_offset - read_begin;
            const std::uint64_t payload_end = byte_offset + byte_count;
            if (read_end > logical_bytes_) {
                const std::uint64_t padding_begin = std::max(logical_bytes_, byte_offset);
                const std::uint64_t padding_offset = padding_begin - read_begin;
                const std::uint64_t padding_end = read_end - read_begin;
                const auto* raw = static_cast<const unsigned char*>(temporary.data());
                for (std::uint64_t index = padding_offset; index < padding_end; ++index)
                    if (raw[index] != 0U) {
                        if (error) *error = "Sumcheck spill final-page padding is not canonical zero";
                        return false;
                    }
            }
            std::memcpy(output, static_cast<const unsigned char*>(temporary.data()) + prefix,
                        static_cast<std::size_t>(byte_count));
        }
        for (std::uint64_t index = 0U; index < count; ++index)
            if (!zkdl::fr_canonical(output[index])) {
                if (error) *error = "Sumcheck spill read encountered a non-canonical FrWire";
                return false;
            }
        return true;
    }

    bool validate_complete(const std::string& expected_operation,
                           std::uint64_t expected_domain,
                           std::uint64_t expected_tensor_count,
                           std::uint64_t expected_rows,
                           std::uint32_t expected_round,
                           std::string* error = nullptr) const {
        struct stat status{};
        if (fd_ < 0 || !complete_ || operation_identity_ != expected_operation ||
            logical_domain_ != expected_domain || tensor_count_ != expected_tensor_count ||
            logical_rows_ != expected_rows || round_index_ != expected_round ||
            logical_written_ != logical_bytes_ || physical_written_ != allocated_bytes_ ||
            ::fstat(fd_, &status) != 0 || !S_ISREG(status.st_mode) ||
            status.st_uid != ::geteuid() || (status.st_mode & 0777U) != 0600U ||
            status.st_nlink != 0 ||
            status.st_size != static_cast<off_t>(allocated_bytes_)) {
            if (error) *error = "Sumcheck spill state identity, prefix, or exact length validation failed";
            return false;
        }
        return true;
    }

    const std::string& directory() const noexcept { return directory_; }
    const std::string& operation_identity() const noexcept { return operation_identity_; }
    std::uint64_t logical_domain() const noexcept { return logical_domain_; }
    std::uint64_t tensor_count() const noexcept { return tensor_count_; }
    std::uint64_t logical_rows() const noexcept { return logical_rows_; }
    std::uint32_t round_index() const noexcept { return round_index_; }

private:
    void close_fd() noexcept {
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
    }

    bool write_direct(const void* source, std::uint64_t bytes,
                      std::string* error) {
        if (fd_ < 0 || source == nullptr || bytes == 0U ||
            (bytes % kZkSumcheckSpillIoAlignment) != 0U ||
            (reinterpret_cast<std::uintptr_t>(source) % kZkSumcheckSpillIoAlignment) != 0U ||
            physical_written_ > allocated_bytes_ ||
            bytes > allocated_bytes_ - physical_written_) {
            if (error) *error = "Sumcheck spill direct write alignment/prefix is invalid";
            return false;
        }
        const auto* cursor = static_cast<const unsigned char*>(source);
        std::uint64_t remaining = bytes;
        while (remaining != 0U) {
            const ssize_t written = ::pwrite(
                fd_, cursor, static_cast<std::size_t>(remaining),
                static_cast<off_t>(physical_written_));
            if (written < 0 && errno == EINTR) continue;
            if (written <= 0 ||
                (static_cast<std::uint64_t>(written) % kZkSumcheckSpillIoAlignment) != 0U) {
                if (error) *error = "Sumcheck spill direct write was short or failed: errno=" +
                                    std::to_string(errno);
                return false;
            }
            physical_written_ += static_cast<std::uint64_t>(written);
            cursor += written;
            remaining -= static_cast<std::uint64_t>(written);
        }
        return true;
    }

    bool read_direct(void* destination, std::uint64_t bytes,
                     std::uint64_t offset, std::string* error) const {
        if (fd_ < 0 || destination == nullptr || bytes == 0U ||
            (bytes % kZkSumcheckSpillIoAlignment) != 0U ||
            (offset % kZkSumcheckSpillIoAlignment) != 0U ||
            (reinterpret_cast<std::uintptr_t>(destination) % kZkSumcheckSpillIoAlignment) != 0U ||
            offset > allocated_bytes_ || bytes > allocated_bytes_ - offset) {
            if (error) *error = "Sumcheck spill direct read alignment/range is invalid";
            return false;
        }
        auto* cursor = static_cast<unsigned char*>(destination);
        std::uint64_t remaining = bytes;
        std::uint64_t current_offset = offset;
        while (remaining != 0U) {
            const ssize_t read = ::pread(
                fd_, cursor, static_cast<std::size_t>(remaining),
                static_cast<off_t>(current_offset));
            if (read < 0 && errno == EINTR) continue;
            if (read <= 0 ||
                (static_cast<std::uint64_t>(read) % kZkSumcheckSpillIoAlignment) != 0U) {
                if (error) *error = "Sumcheck spill direct read was short or failed: errno=" +
                                    std::to_string(errno);
                return false;
            }
            cursor += read;
            current_offset += static_cast<std::uint64_t>(read);
            remaining -= static_cast<std::uint64_t>(read);
        }
        return true;
    }

    int fd_ = -1;
    std::string directory_;
    std::string operation_identity_;
    std::uint64_t logical_domain_ = 0U;
    std::uint64_t tensor_count_ = 0U;
    std::uint64_t logical_rows_ = 0U;
    std::uint32_t round_index_ = 0U;
    std::uint64_t logical_bytes_ = 0U;
    std::uint64_t allocated_bytes_ = 0U;
    std::uint64_t logical_written_ = 0U;
    std::uint64_t physical_written_ = 0U;
    ZkSumcheckAlignedBuffer staging_;
    std::size_t staging_used_ = 0U;
    bool complete_ = false;
};


/*
 * Range-backed helpers for the authenticated sorted-lookup primitive.  None
 * of these adapters materializes a logical tensor: one call reads at most one
 * authenticated basis segment and all derived providers keep only bounded
 * segment buffers.
 */
inline bool sorted_lookup_read_range(const SegmentedTensorProvider& provider,
                                     std::uint64_t offset, std::uint32_t count,
                                     std::vector<FrWire>& output,
                                     std::string* error = nullptr) {
    if (!provider.valid(error) || count > (1U << 18U) || offset > provider.logical_domain() ||
        count > provider.logical_domain() - offset) {
        if (error && error->empty()) *error = "sorted-lookup range is outside provider domain";
        return false;
    }
    output.clear();
    output.reserve(count);
    std::uint64_t cursor = offset;
    std::uint32_t remaining = count;
    while (remaining != 0U) {
        std::uint32_t segment = 0U, inside = 0U;
        if (!checked_relation_u32(cursor / provider.segment_size(), segment,
                                  "sorted-lookup segment index", error) ||
            !checked_relation_u32(cursor % provider.segment_size(), inside,
                                  "sorted-lookup segment offset", error))
            return false;
        std::vector<FrWire> block;
        if (!provider.read_segment(segment, block, error) || inside >= block.size()) return false;
        std::uint32_t available = 0U;
        if (!checked_relation_u32(block.size() - inside, available,
                                  "sorted-lookup segment remainder", error))
            return false;
        const std::uint32_t take = std::min(remaining, available);
        output.insert(output.end(), block.begin() + inside, block.begin() + inside + take);
        cursor += take;
        remaining -= take;
    }
    return output.size() == count;
}

inline bool sorted_lookup_read_scalar(const SegmentedTensorProvider& provider,
                                      std::uint64_t index, FrWire& value,
                                      std::string* error = nullptr) {
    std::vector<FrWire> one;
    if (!sorted_lookup_read_range(provider, index, 1U, one, error) || one.size() != 1U)
        return false;
    value = one.front();
    return true;
}

inline bool sorted_lookup_u128_from_field(const FrWire& value, std::uint32_t bits,
                                          unsigned __int128& result,
                                          std::string* error = nullptr) {
    if (!zkdl::fr_canonical(value) || bits == 0U || bits > 127U) {
        if (error) *error = "sorted-lookup packed key has an invalid field/bound width";
        return false;
    }
    for (std::size_t limb = 4U; limb < value.size(); ++limb)
        if (value[limb] != 0U) {
            if (error) *error = "sorted-lookup packed key exceeds u128 range";
            return false;
        }
    result = static_cast<unsigned __int128>(value[0]) |
             (static_cast<unsigned __int128>(value[1]) << 32U) |
             (static_cast<unsigned __int128>(value[2]) << 64U) |
             (static_cast<unsigned __int128>(value[3]) << 96U);
    if (bits < 128U && (result >> bits) != 0U) {
        if (error) *error = "sorted-lookup packed key exceeds declared bit width";
        return false;
    }
    return true;
}

inline FrWire sorted_lookup_field_from_u128(unsigned __int128 value) {
    FrWire result{};
    result[0] = static_cast<std::uint32_t>(value);
    result[1] = static_cast<std::uint32_t>(value >> 32U);
    result[2] = static_cast<std::uint32_t>(value >> 64U);
    result[3] = static_cast<std::uint32_t>(value >> 96U);
    return result;
}

inline bool sorted_lookup_boolean(const FrWire& value) {
    return zkdl::fr_canonical(value) &&
           (value == zkdl::fr_from_u32(0U) || value == zkdl::fr_from_u32(1U));
}

inline bool sorted_lookup_provider_shape(const SegmentedTensorProvider& provider,
                                         std::uint32_t logical_domain,
                                         std::uint32_t segment_size,
                                         std::uint32_t segment_count,
                                         std::string* error = nullptr) {
    if (!provider.valid(error) || provider.logical_domain() != logical_domain ||
        provider.segment_size() != segment_size || provider.segment_count() != segment_count) {
        if (error && error->empty()) *error = "sorted-lookup provider segment layout mismatch";
        return false;
    }
    return true;
}

inline bool make_authenticated_sorted_lookup_packed_key_provider(
    const SegmentedTensorProvider& x_provider,
    const SegmentedTensorProvider& y_provider,
    std::uint32_t x_bits, std::uint32_t y_bits,
    SegmentedTensorProvider& packed_provider, std::string* error = nullptr) {
    if (!x_provider.valid(error) || !y_provider.valid(error) ||
        x_provider.logical_domain() != y_provider.logical_domain() ||
        x_provider.segment_size() != y_provider.segment_size() ||
        x_provider.segment_count() != y_provider.segment_count() ||
        x_bits == 0U || y_bits == 0U || x_bits + y_bits > 127U) {
        if (error && error->empty()) *error = "invalid sorted-lookup packed key inputs";
        return false;
    }
    const std::uint32_t domain = x_provider.logical_domain();
    const std::uint32_t segment_size = x_provider.segment_size();
    packed_provider = SegmentedTensorProvider(
        domain, segment_size,
        [x_provider, y_provider, x_bits, y_bits](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> xs, ys;
            if (!sorted_lookup_read_range(x_provider, offset, count, xs, read_error) ||
                !sorted_lookup_read_range(y_provider, offset, count, ys, read_error) ||
                xs.size() != ys.size()) return false;
            output.resize(count);
            const unsigned __int128 base = static_cast<unsigned __int128>(1U) << y_bits;
            for (std::size_t index = 0U; index < xs.size(); ++index) {
                unsigned __int128 x = 0U, y = 0U;
                if (!sorted_lookup_u128_from_field(xs[index], x_bits, x, read_error) ||
                    !sorted_lookup_u128_from_field(ys[index], y_bits, y, read_error)) return false;
                const unsigned __int128 packed = x * base + y;
                // x_bits+y_bits <= 127 makes this strictly below 2^127,
                // hence below (p-1)/2; no signed modular reduction occurs.
                output[index] = sorted_lookup_field_from_u128(packed);
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_bit_provider(
    const SegmentedTensorProvider& packed_provider, std::uint32_t key_bits,
    std::uint32_t bit_index, SegmentedTensorProvider& bit_provider,
    std::string* error = nullptr) {
    if (!packed_provider.valid(error) || key_bits == 0U || key_bits > 127U ||
        bit_index >= key_bits) {
        if (error && error->empty()) *error = "invalid sorted-lookup bit-provider request";
        return false;
    }
    bit_provider = SegmentedTensorProvider(
        packed_provider.logical_domain(), packed_provider.segment_size(),
        [packed_provider, key_bits, bit_index](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> keys;
            if (!sorted_lookup_read_range(packed_provider, offset, count, keys, read_error))
                return false;
            output.resize(count);
            for (std::size_t index = 0U; index < keys.size(); ++index) {
                unsigned __int128 packed = 0U;
                if (!sorted_lookup_u128_from_field(keys[index], key_bits, packed, read_error))
                    return false;
                output[index] = zkdl::fr_from_u32(
                    static_cast<std::uint32_t>((packed >> bit_index) & 1U));
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_delta_provider(
    const SegmentedTensorProvider& sorted_keys,
    const SegmentedTensorProvider& active_mask, std::uint32_t key_bits,
    SegmentedTensorProvider& delta_provider, std::string* error = nullptr) {
    if (!sorted_lookup_provider_shape(sorted_keys, sorted_keys.logical_domain(),
                                      sorted_keys.segment_size(), sorted_keys.segment_count(), error) ||
        !sorted_lookup_provider_shape(active_mask, sorted_keys.logical_domain(),
                                      sorted_keys.segment_size(), sorted_keys.segment_count(), error) ||
        key_bits == 0U || key_bits > 127U) return false;
    delta_provider = SegmentedTensorProvider(
        sorted_keys.logical_domain(), sorted_keys.segment_size(),
        [sorted_keys, active_mask, key_bits](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> keys, active;
            if (!sorted_lookup_read_range(sorted_keys, offset, count, keys, read_error) ||
                !sorted_lookup_read_range(active_mask, offset, count, active, read_error))
                return false;
            FrWire previous{};
            FrWire previous_active{};
            if (offset != 0U &&
                (!sorted_lookup_read_scalar(sorted_keys, offset - 1U, previous, read_error) ||
                 !sorted_lookup_read_scalar(active_mask, offset - 1U, previous_active, read_error)))
                return false;
            unsigned __int128 previous_key = 0U;
            if (offset != 0U && previous_active == zkdl::fr_from_u32(1U) &&
                !sorted_lookup_u128_from_field(previous, key_bits, previous_key, read_error))
                return false;
            output.resize(count);
            for (std::size_t index = 0U; index < keys.size(); ++index) {
                if (!sorted_lookup_boolean(active[index])) {
                    if (read_error) *read_error = "sorted-lookup active mask is not boolean";
                    return false;
                }
                const std::uint64_t global = offset + index;
                unsigned __int128 current = 0U;
                if (!sorted_lookup_u128_from_field(keys[index], key_bits, current, read_error))
                    return false;
                if (active[index] == zkdl::fr_from_u32(0U)) {
                    if (current != 0U) {
                        if (read_error) *read_error = "sorted-lookup inactive key is nonzero";
                        return false;
                    }
                    output[index] = FrWire{};
                    continue;
                }
                if (global != 0U && previous_active != zkdl::fr_from_u32(1U)) {
                    if (read_error) *read_error = "sorted-lookup active mask is not contiguous";
                    return false;
                }
                if (global != 0U && current < previous_key) {
                    if (read_error) *read_error = "sorted-lookup sorted key decreased";
                    return false;
                }
                const unsigned __int128 difference =
                    global == 0U ? current : current - previous_key;
                output[index] = sorted_lookup_field_from_u128(difference);
                previous_key = current;
                previous_active = active[index];
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_marker_provider(
    const SegmentedTensorProvider& sorted_keys,
    const SegmentedTensorProvider& active_mask, std::uint32_t key_bits,
    SegmentedTensorProvider& marker_provider, std::string* error = nullptr) {
    if (!sorted_keys.valid(error) || !active_mask.valid(error) ||
        sorted_keys.logical_domain() != active_mask.logical_domain() ||
        sorted_keys.segment_size() != active_mask.segment_size() ||
        sorted_keys.segment_count() != active_mask.segment_count() ||
        key_bits == 0U || key_bits > 127U) {
        if (error && error->empty()) *error = "invalid sorted-lookup marker inputs";
        return false;
    }
    marker_provider = SegmentedTensorProvider(
        sorted_keys.logical_domain(), sorted_keys.segment_size(),
        [sorted_keys, active_mask, key_bits](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> keys, active;
            if (!sorted_lookup_read_range(sorted_keys, offset, count, keys, read_error) ||
                !sorted_lookup_read_range(active_mask, offset, count, active, read_error))
                return false;
            FrWire previous{};
            FrWire previous_active{};
            if (offset != 0U &&
                (!sorted_lookup_read_scalar(sorted_keys, offset - 1U, previous, read_error) ||
                 !sorted_lookup_read_scalar(active_mask, offset - 1U, previous_active, read_error)))
                return false;
            unsigned __int128 previous_key = 0U;
            if (offset != 0U && previous_active == zkdl::fr_from_u32(1U) &&
                !sorted_lookup_u128_from_field(previous, key_bits, previous_key, read_error))
                return false;
            output.resize(count);
            for (std::size_t index = 0U; index < keys.size(); ++index) {
                if (!sorted_lookup_boolean(active[index])) {
                    if (read_error) *read_error = "sorted-lookup active mask is not boolean";
                    return false;
                }
                const std::uint64_t global = offset + index;
                if (active[index] == zkdl::fr_from_u32(0U)) {
                    output[index] = FrWire{};
                    continue;
                }
                unsigned __int128 current = 0U;
                if (!sorted_lookup_u128_from_field(keys[index], key_bits, current, read_error))
                    return false;
                if (global != 0U && previous_active != zkdl::fr_from_u32(1U)) {
                    if (read_error) *read_error = "sorted-lookup marker has a gap before an active row";
                    return false;
                }
                output[index] = zkdl::fr_from_u32(
                    global == 0U || previous_active == zkdl::fr_from_u32(0U) ||
                    current != previous_key ? 1U : 0U);
                previous_key = current;
                previous_active = active[index];
            }
            if (offset == 0U && !output.empty() &&
                active_mask.logical_domain() != 0U &&
                active_mask.segment_count() != 0U &&
                active_mask.segment_size() != 0U &&
                active_mask.segment_size() > 0U &&
                output.front() != zkdl::fr_from_u32(1U)) {
                if (read_error) *read_error = "sorted-lookup first active marker is not one";
                return false;
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_inverse_provider(
    const SegmentedTensorProvider& delta_provider,
    SegmentedTensorProvider& inverse_provider, std::string* error = nullptr) {
    if (!delta_provider.valid(error)) return false;
    inverse_provider = SegmentedTensorProvider(
        delta_provider.logical_domain(), delta_provider.segment_size(),
        [delta_provider](std::uint64_t offset, std::uint32_t count,
                         std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> delta;
            if (!sorted_lookup_read_range(delta_provider, offset, count, delta, read_error))
                return false;
            output.resize(count);
            const FrWire exponent = zkdl::fr_modulus_minus_two();
            for (std::size_t index = 0U; index < delta.size(); ++index)
                output[index] = delta[index] == FrWire{}
                                    ? FrWire{}
                                    : zkdl::fr_pow(delta[index], 256U, exponent);
            return true;
        });
    return true;
}

struct AuthenticatedSortedLookupPrefixCache {
    std::uint32_t logical_domain = 0U;
    std::uint32_t segment_size = 0U;
    std::uint32_t segment_count = 0U;
    std::uint32_t key_bits = 0U;
    FrWire beta{};
    FrWire gamma{};
    std::array<byte, 32U> descriptor_digest{};
    std::mutex mutex;
    bool initialized = false;
    bool failed = false;
    std::string failure;
    std::vector<FrWire> segment_prefix;
};

inline bool make_authenticated_sorted_lookup_prefix_product_provider(
    const SegmentedTensorProvider& keys, const SegmentedTensorProvider& active_mask,
    const FrWire& beta, const FrWire& gamma,
    SegmentedTensorProvider& prefix_provider, std::string* error = nullptr,
    NativeOperationProgressReporter* progress = nullptr,
    const std::array<byte, 32U>& descriptor_digest = {},
    std::uint32_t key_bits = 127U) {
    if (!keys.valid(error) || !active_mask.valid(error) ||
        keys.logical_domain() != active_mask.logical_domain() ||
        keys.segment_size() != active_mask.segment_size() ||
        keys.segment_count() != active_mask.segment_count() ||
        !zkdl::fr_canonical(beta) || !zkdl::fr_canonical(gamma) ||
        key_bits == 0U || key_bits > 127U ||
        beta == FrWire{} || gamma == FrWire{}) {
        if (error && error->empty()) *error = "invalid sorted-lookup prefix-product inputs";
        return false;
    }
    const auto cache = std::make_shared<AuthenticatedSortedLookupPrefixCache>();
    cache->logical_domain = keys.logical_domain();
    cache->segment_size = keys.segment_size();
    cache->segment_count = keys.segment_count();
    cache->key_bits = key_bits;
    cache->beta = beta;
    cache->gamma = gamma;
    cache->descriptor_digest = descriptor_digest;
    prefix_provider = SegmentedTensorProvider(
        keys.logical_domain(), keys.segment_size(),
        [keys, active_mask, beta, gamma, cache, progress, key_bits, descriptor_digest](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            if (offset > keys.logical_domain() || count > keys.logical_domain() - offset) {
                if (read_error) *read_error = "sorted-lookup prefix range is out of bounds";
                return false;
            }
            std::lock_guard<std::mutex> guard(cache->mutex);
            if (cache->failed) {
                if (read_error) *read_error = cache->failure;
                return false;
            }
            const auto fail_cache = [&](const std::string& message) {
                cache->failed = true;
                cache->failure = message;
                if (read_error) *read_error = message;
                return false;
            };
            if (cache->logical_domain != keys.logical_domain() ||
                cache->segment_size != keys.segment_size() ||
                cache->segment_count != keys.segment_count() ||
                cache->key_bits != key_bits ||
                cache->beta != beta || cache->gamma != gamma ||
                cache->descriptor_digest != descriptor_digest) {
                return fail_cache("sorted-lookup prefix cache identity mismatch");
            }
            if (!cache->initialized) {
                std::vector<FrWire> segment_prefix;
                segment_prefix.reserve(keys.segment_count());
                FrWire product = zkdl::fr_from_u32(1U);
                bool padding_seen = false;
                for (std::uint32_t segment = 0U; segment < keys.segment_count(); ++segment) {
                    std::vector<FrWire> key_block, active_block;
                    if (!keys.read_segment(segment, key_block, read_error) ||
                        !active_mask.read_segment(segment, active_block, read_error) ||
                        key_block.size() != keys.segment_size() ||
                        active_block.size() != keys.segment_size()) {
                        return fail_cache(read_error && !read_error->empty()
                            ? *read_error : "sorted-lookup prefix cache source segment shape mismatch");
                    }
                    segment_prefix.push_back(product);
                    for (std::uint32_t inside = 0U; inside < keys.segment_size(); ++inside) {
                        if (!sorted_lookup_boolean(active_block[inside])) {
                            return fail_cache("sorted-lookup prefix active mask is not boolean");
                        }
                        unsigned __int128 key = 0U;
                        if (!sorted_lookup_u128_from_field(key_block[inside], key_bits, key, read_error))
                            return fail_cache(read_error && !read_error->empty()
                                ? *read_error : "sorted-lookup prefix key is not canonical");
                        if (active_block[inside] == zkdl::fr_from_u32(0U)) {
                            padding_seen = true;
                            if (key != 0U) return fail_cache("sorted-lookup prefix inactive key is nonzero");
                        } else {
                            if (padding_seen) return fail_cache("sorted-lookup prefix active mask is not contiguous");
                            product = zkdl::fr_mul(
                                product, zkdl::fr_add(
                                    zkdl::fr_mul(beta, key_block[inside]), gamma));
                        }
                    }
                    if (progress != nullptr)
                        progress->update("prefix_seed_segments", segment + 1U,
                                         keys.segment_count());
                }
                cache->segment_prefix = std::move(segment_prefix);
                cache->initialized = true;
            }
            output.assign(count, FrWire{});
            std::uint64_t cursor = offset;
            std::uint32_t remaining = count;
            std::uint32_t completed = 0U;
            const std::uint32_t requested_segments = static_cast<std::uint32_t>(
                (offset % keys.segment_size() + count + keys.segment_size() - 1U) /
                keys.segment_size());
            while (remaining != 0U) {
                std::uint32_t segment = 0U, inside = 0U;
                if (!checked_relation_u32(cursor / keys.segment_size(), segment,
                                          "sorted-lookup prefix segment index", read_error) ||
                    !checked_relation_u32(cursor % keys.segment_size(), inside,
                                          "sorted-lookup prefix segment offset", read_error) ||
                    segment >= cache->segment_prefix.size()) return false;
                std::vector<FrWire> key_block, active_block;
                if (!keys.read_segment(segment, key_block, read_error) ||
                    !active_mask.read_segment(segment, active_block, read_error) ||
                    key_block.size() != keys.segment_size() ||
                    active_block.size() != keys.segment_size())
                    return fail_cache(read_error && !read_error->empty()
                        ? *read_error : "sorted-lookup prefix requested segment shape mismatch");
                const std::uint32_t take = std::min(
                    remaining, keys.segment_size() - inside);
                FrWire product = cache->segment_prefix[segment];
                const std::uint32_t stop = inside + take;
                for (std::uint32_t local = 0U; local < stop; ++local) {
                    if (!sorted_lookup_boolean(active_block[local])) {
                        return fail_cache("sorted-lookup prefix active mask is not boolean");
                    }
                    unsigned __int128 key = 0U;
                    if (!sorted_lookup_u128_from_field(key_block[local], key_bits, key, read_error))
                        return fail_cache(read_error && !read_error->empty()
                            ? *read_error : "sorted-lookup prefix key is not canonical");
                    const std::uint64_t global =
                        static_cast<std::uint64_t>(segment) * keys.segment_size() + local;
                    if (active_block[local] == zkdl::fr_from_u32(0U) && key != 0U)
                        return fail_cache("sorted-lookup prefix inactive key is nonzero");
                    if (local >= inside && global < offset + count &&
                        active_block[local] == zkdl::fr_from_u32(1U))
                        output[static_cast<std::size_t>(global - offset)] = product;
                    if (active_block[local] == zkdl::fr_from_u32(1U))
                        product = zkdl::fr_mul(
                            product, zkdl::fr_add(
                                zkdl::fr_mul(beta, key_block[local]), gamma));
                }
                cursor += take;
                remaining -= take;
                ++completed;
                if (progress != nullptr)
                    progress->update("prefix_output_segments", completed,
                                     requested_segments);
            }
            return true;
        });
    return true;
}

inline bool authenticated_sorted_lookup_prefix_product_endpoint(
    const SegmentedTensorProvider& keys, const SegmentedTensorProvider& active_mask,
    std::uint32_t active_length, const FrWire& beta, const FrWire& gamma,
    FrWire& endpoint, std::string* error = nullptr,
    NativeOperationProgressReporter* progress = nullptr,
    const std::array<byte, 32U>& descriptor_digest = {},
    const SegmentedTensorProvider* existing_prefix = nullptr) {
    SegmentedTensorProvider prefix;
    if (existing_prefix != nullptr) prefix = *existing_prefix;
    else if (!make_authenticated_sorted_lookup_prefix_product_provider(
                 keys, active_mask, beta, gamma, prefix, error, progress,
                 descriptor_digest)) return false;
    if (active_length > keys.logical_domain()) {
        if (error) *error = "sorted-lookup prefix endpoint length is out of bounds";
        return false;
    }
    if (active_length == 0U) {
        endpoint = zkdl::fr_from_u32(1U);
        return true;
    }
    FrWire previous{}, last_key{}, last_active{};
    if (!sorted_lookup_read_scalar(prefix, active_length - 1U, previous, error) ||
        !sorted_lookup_read_scalar(keys, active_length - 1U, last_key, error) ||
        !sorted_lookup_read_scalar(active_mask, active_length - 1U, last_active, error) ||
        last_active != zkdl::fr_from_u32(1U)) {
        if (error && error->empty()) *error = "sorted-lookup endpoint active row is not canonical";
        return false;
    }
    endpoint = zkdl::fr_mul(previous,
                            zkdl::fr_add(zkdl::fr_mul(beta, last_key), gamma));
    return true;
}

inline bool make_authenticated_sorted_lookup_shifted_key_provider(
    const SegmentedTensorProvider& keys, const SegmentedTensorProvider& active_mask,
    SegmentedTensorProvider& previous_provider,
    std::string* error = nullptr) {
    if (!keys.valid(error) || !active_mask.valid(error) ||
        keys.logical_domain() != active_mask.logical_domain() ||
        keys.segment_size() != active_mask.segment_size() ||
        keys.segment_count() != active_mask.segment_count()) {
        if (error && error->empty())
            *error = "invalid sorted-lookup shifted-key provider inputs";
        return false;
    }
    previous_provider = SegmentedTensorProvider(
        keys.logical_domain(), keys.segment_size(),
        [keys, active_mask](std::uint64_t offset, std::uint32_t count,
               std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> source;
            std::vector<FrWire> active;
            const std::uint64_t source_offset = offset == 0U ? 0U : offset - 1U;
            if (!sorted_lookup_read_range(keys, source_offset, count, source,
                                          read_error) ||
                !sorted_lookup_read_range(active_mask, offset, count, active,
                                          read_error)) return false;
            output.assign(count, FrWire{});
            for (std::uint32_t index = 0U; index < count; ++index) {
                const std::uint64_t global = offset + index;
                if (!sorted_lookup_boolean(active[index])) {
                    if (read_error)
                        *read_error = "sorted-lookup shifted-key active mask is not boolean";
                    return false;
                }
                if (global != 0U && active[index] == zkdl::fr_from_u32(1U)) {
                    const std::size_t source_index =
                        offset == 0U ? index - 1U : index;
                    if (source_index >= source.size()) {
                        if (read_error) *read_error = "sorted-lookup shifted-key range mismatch";
                        return false;
                    }
                    output[index] = source[source_index];
                }
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_nonzero_provider(
    const SegmentedTensorProvider& values, SegmentedTensorProvider& result,
    std::string* error = nullptr) {
    if (!values.valid(error)) return false;
    result = SegmentedTensorProvider(
        values.logical_domain(), values.segment_size(),
        [values](std::uint64_t offset, std::uint32_t count,
                 std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> source;
            if (!sorted_lookup_read_range(values, offset, count, source, read_error)) return false;
            output.resize(count);
            for (std::size_t index = 0U; index < source.size(); ++index)
                output[index] = source[index] == FrWire{} ? FrWire{} : zkdl::fr_from_u32(1U);
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_selector_provider(
    std::uint32_t domain, std::uint32_t segment_size, std::uint32_t selected_index,
    SegmentedTensorProvider& result, std::string* error = nullptr) {
    if (domain == 0U || segment_size == 0U || domain % segment_size != 0U ||
        (domain & (domain - 1U)) != 0U || (segment_size & (segment_size - 1U)) != 0U ||
        selected_index >= domain) {
        if (error) *error = "invalid sorted-lookup selector provider shape";
        return false;
    }
    result = SegmentedTensorProvider(
        domain, segment_size,
        [selected_index](std::uint64_t offset, std::uint32_t count,
                         std::vector<FrWire>& output, std::string* /*read_error*/) {
            output.assign(count, FrWire{});
            if (selected_index >= offset && selected_index - offset < count)
                output[static_cast<std::size_t>(selected_index - offset)] = zkdl::fr_from_u32(1U);
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_prefix_next_provider(
    const SegmentedTensorProvider& prefix, const SegmentedTensorProvider& keys,
    const SegmentedTensorProvider& active_mask, std::uint32_t active_length,
    const FrWire& beta, const FrWire& gamma, SegmentedTensorProvider& result,
    std::string* error = nullptr) {
    if (!prefix.valid(error) || !keys.valid(error) || !active_mask.valid(error) ||
        prefix.logical_domain() != keys.logical_domain() ||
        prefix.logical_domain() != active_mask.logical_domain() ||
        prefix.segment_size() != keys.segment_size() ||
        prefix.segment_size() != active_mask.segment_size() ||
        active_length == 0U || active_length > keys.logical_domain() ||
        !zkdl::fr_canonical(beta) || !zkdl::fr_canonical(gamma)) {
        if (error && error->empty()) *error = "invalid sorted-lookup prefix-next provider inputs";
        return false;
    }
    result = SegmentedTensorProvider(
        keys.logical_domain(), keys.segment_size(),
        [prefix, keys, active_mask, active_length, beta, gamma](
            std::uint64_t offset, std::uint32_t count,
            std::vector<FrWire>& output, std::string* read_error) {
            std::vector<FrWire> current, next, key, active;
            if (!sorted_lookup_read_range(prefix, offset, count, current, read_error) ||
                !sorted_lookup_read_range(keys, offset, count, key, read_error) ||
                !sorted_lookup_read_range(active_mask, offset, count, active, read_error))
                return false;
            const std::uint64_t domain = prefix.logical_domain();
            const std::uint64_t shifted_available =
                offset < domain ? domain - offset - 1U : 0U;
            const std::uint32_t next_count = static_cast<std::uint32_t>(
                std::min<std::uint64_t>(count, shifted_available));
            if (next_count != 0U) {
                if (!sorted_lookup_read_range(
                        prefix, offset + 1U, next_count, next, read_error))
                    return false;
            }
            output.assign(count, FrWire{});
            for (std::uint32_t index = 0U; index < count; ++index) {
                const std::uint64_t global = offset + index;
                if (global >= active_length) continue;
                if (global + 1U < active_length) {
                    if (index >= next.size()) {
                        if (read_error)
                            *read_error = "sorted-lookup prefix-next shifted range is truncated";
                        return false;
                    }
                    output[index] = next[index];
                } else {
                    output[index] = zkdl::fr_mul(
                        current[index],
                        zkdl::fr_add(zkdl::fr_mul(beta, key[index]), gamma));
                }
                if (!sorted_lookup_boolean(active[index])) {
                    if (read_error) *read_error = "sorted-lookup prefix active mask is not boolean";
                    return false;
                }
            }
            return true;
        });
    return true;
}

inline bool make_authenticated_sorted_lookup_constant_provider(
    std::uint32_t domain, std::uint32_t segment_size, const FrWire& value,
    SegmentedTensorProvider& result, std::string* error = nullptr) {
    if (domain == 0U || segment_size == 0U || domain % segment_size != 0U ||
        (domain & (domain - 1U)) != 0U || (segment_size & (segment_size - 1U)) != 0U ||
        !zkdl::fr_canonical(value)) {
        if (error) *error = "invalid sorted-lookup constant provider shape";
        return false;
    }
    result = SegmentedTensorProvider(
        domain, segment_size,
        [value](std::uint64_t /*offset*/, std::uint32_t count,
                std::vector<FrWire>& output, std::string* /*read_error*/) {
            output.assign(count, value);
            return true;
        });
    return true;
}

inline bool authenticated_sorted_lookup_public_table_product(
    const SegmentedTensorProvider& table_keys, const SegmentedTensorProvider& table_active,
    std::uint32_t active_length, const FrWire& beta, const FrWire& gamma,
    FrWire& product, std::string* error = nullptr,
    NativeOperationProgressReporter* progress = nullptr) {
    if (!table_keys.valid(error) || !table_active.valid(error) ||
        table_keys.logical_domain() != table_active.logical_domain() ||
        table_keys.segment_size() != table_active.segment_size() ||
        active_length == 0U || active_length > table_keys.logical_domain() ||
        !zkdl::fr_canonical(beta) || !zkdl::fr_canonical(gamma)) {
        if (error && error->empty()) *error = "invalid public sorted-lookup table product inputs";
        return false;
    }
    product = zkdl::fr_from_u32(1U);
    for (std::uint32_t segment = 0U; segment < table_keys.segment_count(); ++segment) {
        std::vector<FrWire> keys, active;
        if (!table_keys.read_segment(segment, keys, error) ||
            !table_active.read_segment(segment, active, error)) return false;
        const std::uint64_t base = static_cast<std::uint64_t>(segment) * table_keys.segment_size();
        for (std::uint32_t index = 0U; index < table_keys.segment_size(); ++index) {
            const std::uint64_t global = base + index;
            if (global >= active_length) break;
            if (active[index] != zkdl::fr_from_u32(1U)) {
                if (error) *error = "public sorted-lookup table active prefix is not canonical";
                return false;
            }
            product = zkdl::fr_mul(product,
                                   zkdl::fr_add(zkdl::fr_mul(beta, keys[index]), gamma));
        }
        if (progress != nullptr)
            progress->update("public_table_product_segments", segment + 1U,
                             table_keys.segment_count());
    }
    return true;
}

struct AuthenticatedSortedLookupProviderSet {
    SegmentedTensorProvider table_keys;
    SegmentedTensorProvider query_keys;
    SegmentedTensorProvider query_x;
    SegmentedTensorProvider query_y;
    SegmentedTensorProvider query_prefix;
    SegmentedTensorProvider query_prefix_next;
    SegmentedTensorProvider merge_keys;
    SegmentedTensorProvider table_active;
    SegmentedTensorProvider query_active;
    SegmentedTensorProvider merge_active;
    SegmentedTensorProvider merge_delta;
    SegmentedTensorProvider merge_marker;
    SegmentedTensorProvider merge_inverse;
    SegmentedTensorProvider merge_prefix;
    SegmentedTensorProvider merge_prefix_next;
    SegmentedTensorProvider merge_previous_key;
    SegmentedTensorProvider merge_nonzero;
    SegmentedTensorProvider merge_first;
    SegmentedTensorProvider merge_last;
    SegmentedTensorProvider query_first;
    SegmentedTensorProvider query_last;
    std::vector<SegmentedTensorProvider> merge_bits;
    std::vector<SegmentedTensorProvider> merge_delta_bits;
    std::vector<SegmentedTensorProvider> query_bits;

    bool valid(const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
               std::string* error = nullptr) const {
        return valid_impl(descriptor, error, true);
    }

    // Use only after valid() has checked the unchanged base providers and this
    // prover has appended derived providers.  Recheck all shapes and derived
    // merge-bit contents without rescanning the unchanged base tensors.
    bool valid_after_derived_providers(
        const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
        std::string* error = nullptr) const {
        return valid_impl(descriptor, error, false);
    }

private:
    bool valid_impl(const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
                    std::string* error, bool validate_base_contents) const {
        if (!descriptor.valid(error) ||
            !sorted_lookup_provider_shape(table_keys, descriptor.table_domain,
                                          descriptor.segment_size, descriptor.table_segment_count, error) ||
            !sorted_lookup_provider_shape(query_keys, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(query_x, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(query_y, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(query_prefix, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(query_prefix_next, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(merge_keys, descriptor.merge_domain,
                                          descriptor.segment_size, descriptor.merge_segment_count, error) ||
            !sorted_lookup_provider_shape(table_active, descriptor.table_domain,
                                          descriptor.segment_size, descriptor.table_segment_count, error) ||
            !sorted_lookup_provider_shape(query_active, descriptor.query_domain,
                                          descriptor.segment_size, descriptor.query_segment_count, error) ||
            !sorted_lookup_provider_shape(merge_active, descriptor.merge_domain,
                                          descriptor.segment_size, descriptor.merge_segment_count, error) ||
            !sorted_lookup_provider_shape(merge_delta, descriptor.merge_domain,
                                          descriptor.segment_size, descriptor.merge_segment_count, error) ||
            !sorted_lookup_provider_shape(merge_marker, descriptor.merge_domain,
                                          descriptor.segment_size, descriptor.merge_segment_count, error) ||
            !sorted_lookup_provider_shape(merge_inverse, descriptor.merge_domain,
                                          descriptor.segment_size, descriptor.merge_segment_count, error))
            return false;
        for (const auto* provider : {&merge_prefix, &merge_prefix_next, &merge_previous_key,
                                     &merge_nonzero, &merge_first, &merge_last})
            if (!sorted_lookup_provider_shape(*provider, descriptor.merge_domain,
                                              descriptor.segment_size,
                                              descriptor.merge_segment_count, error)) return false;
        for (const auto* provider : {&query_first, &query_last})
            if (!sorted_lookup_provider_shape(*provider, descriptor.query_domain,
                                              descriptor.segment_size,
                                              descriptor.query_segment_count, error)) return false;
        const std::size_t bit_count =
            static_cast<std::size_t>(descriptor.x_bits + descriptor.y_bits);
        if (merge_bits.size() != bit_count) {
            if (error) *error = "sorted-lookup merge bit-provider count mismatch";
            return false;
        }
        for (const auto& bit : merge_bits)
            if (!sorted_lookup_provider_shape(bit, descriptor.merge_domain,
                                              descriptor.segment_size,
                                              descriptor.merge_segment_count, error)) return false;
        if (merge_delta_bits.size() != bit_count) {
            if (error) *error = "sorted-lookup merge delta bit-provider count mismatch";
            return false;
        }
        for (const auto& bit : merge_delta_bits)
            if (!sorted_lookup_provider_shape(bit, descriptor.merge_domain,
                                              descriptor.segment_size,
                                              descriptor.merge_segment_count, error)) return false;
        for (const auto& bit : query_bits)
            if (!sorted_lookup_provider_shape(bit, descriptor.query_domain,
                                              descriptor.segment_size,
                                              descriptor.query_segment_count, error)) return false;

        const auto validate_mask = [&](const SegmentedTensorProvider& keys,
                                       const SegmentedTensorProvider& mask,
                                       const char* kind,
                                       std::uint32_t active_length,
                                       bool require_nondecreasing,
                                       bool strict_unique) {
            bool previous_set = false;
            unsigned __int128 previous = 0U;
            std::uint64_t previous_index = 0U;
            std::uint32_t observed = 0U;
            for (std::uint32_t segment = 0U; segment < keys.segment_count(); ++segment) {
                std::vector<FrWire> key_block, mask_block;
                if (!keys.read_segment(segment, key_block, error) ||
                    !mask.read_segment(segment, mask_block, error)) return false;
                const std::uint64_t base = static_cast<std::uint64_t>(segment) * keys.segment_size();
                for (std::uint32_t inside = 0U; inside < keys.segment_size(); ++inside) {
                    const std::uint64_t index = base + inside;
                    if (index >= keys.logical_domain()) break;
                    if (!sorted_lookup_boolean(mask_block[inside])) {
                        if (error) *error = std::string("sorted-lookup ") + kind +
                            " active mask is not boolean at index=" +
                            std::to_string(index) + " active_length=" +
                            std::to_string(active_length) + " domain=" +
                            std::to_string(keys.logical_domain());
                        return false;
                    }
                    const bool active = mask_block[inside] == zkdl::fr_from_u32(1U);
                    if (active != (index < active_length)) {
                        if (error) *error = std::string("sorted-lookup ") + kind +
                            " active mask does not match declared length at index=" +
                            std::to_string(index) + " active_length=" +
                            std::to_string(active_length) + " domain=" +
                            std::to_string(keys.logical_domain());
                        return false;
                    }
                    unsigned __int128 current = 0U;
                    if (!sorted_lookup_u128_from_field(key_block[inside],
                                                       descriptor.x_bits + descriptor.y_bits,
                                                       current, error)) {
                        if (error) *error = std::string("sorted-lookup ") + kind +
                            " key is not canonical at index=" + std::to_string(index) +
                            " active_length=" + std::to_string(active_length) +
                            " domain=" + std::to_string(keys.logical_domain());
                        return false;
                    }
                    if (zkdl::fr_compare(key_block[inside], descriptor.key_max) > 0) {
                        if (error) *error = std::string("sorted-lookup ") + kind +
                            " key exceeds authenticated key_max at index=" +
                            std::to_string(index) + " active_length=" +
                            std::to_string(active_length) + " domain=" +
                            std::to_string(keys.logical_domain());
                        return false;
                    }
                    if (!active && current != 0U) {
                        if (error) *error = std::string("sorted-lookup ") + kind +
                            " padded key is nonzero at index=" + std::to_string(index) +
                            " active_length=" + std::to_string(active_length) +
                            " domain=" + std::to_string(keys.logical_domain());
                        return false;
                    }
                    if (active) {
                        if (previous_set &&
                            ((require_nondecreasing && current < previous) ||
                             (strict_unique && current == previous))) {
                            if (error) *error = std::string("sorted-lookup ") + kind +
                                " key sequence is not canonical sorted order at index=" +
                                std::to_string(index) + " previous_index=" +
                                std::to_string(previous_index) + " active_length=" +
                                std::to_string(active_length) + " domain=" +
                                std::to_string(keys.logical_domain());
                            return false;
                        }
                        previous = current;
                        previous_index = index;
                        previous_set = true;
                        ++observed;
                    }
                }
            }
            return observed == active_length;
        };

        if (validate_base_contents &&
            (!validate_mask(table_keys, table_active, "table",
                            descriptor.table_active_length, true, true) ||
             !validate_mask(query_keys, query_active, "query",
                            descriptor.query_active_length, false, false) ||
             !validate_mask(merge_keys, merge_active, "merge",
                            descriptor.merge_active_length, true, false)))
            return false;

        if (validate_base_contents) {
            for (std::uint32_t segment = 0U; segment < merge_keys.segment_count(); ++segment) {
                std::vector<FrWire> keys, active, delta, marker, inverse;
                if (!merge_keys.read_segment(segment, keys, error) ||
                    !merge_active.read_segment(segment, active, error) ||
                    !merge_delta.read_segment(segment, delta, error) ||
                    !merge_marker.read_segment(segment, marker, error) ||
                    !merge_inverse.read_segment(segment, inverse, error)) return false;
                const std::uint64_t base = static_cast<std::uint64_t>(segment) * merge_keys.segment_size();
                for (std::uint32_t inside = 0U; inside < merge_keys.segment_size(); ++inside) {
                    const std::uint64_t index = base + inside;
                    if (index >= merge_keys.logical_domain()) break;
                    if (!sorted_lookup_boolean(marker[inside])) {
                        if (error) *error = "sorted-lookup group marker is not boolean";
                        return false;
                    }
                    if (!sorted_lookup_boolean(active[inside])) return false;
                    unsigned __int128 key = 0U, d = 0U;
                    if (!sorted_lookup_u128_from_field(keys[inside], descriptor.x_bits + descriptor.y_bits,
                                                       key, error) ||
                        !sorted_lookup_u128_from_field(delta[inside], descriptor.x_bits + descriptor.y_bits,
                                                       d, error)) return false;
                    if (index < descriptor.merge_active_length &&
                        (marker[inside] == zkdl::fr_from_u32(1U)) !=
                        (index == 0U || (inside != 0U ? keys[inside] != keys[inside - 1U]
                                                       : true))) {
                        // Cross-segment equality is checked by the derived
                        // marker provider; the wire validator only enforces the
                        // local boolean/range shape here.
                    }
                    if (index >= descriptor.merge_active_length &&
                        (delta[inside] != FrWire{} || marker[inside] != FrWire{} ||
                         inverse[inside] != FrWire{})) {
                        if (error) *error = "sorted-lookup padded auxiliary row is nonzero";
                        return false;
                    }
                    if (d == 0U) {
                        if (inverse[inside] != FrWire{}) {
                            if (error) *error = "sorted-lookup zero delta has a nonzero inverse";
                            return false;
                        }
                    } else if (zkdl::fr_mul(delta[inside], inverse[inside]) != zkdl::fr_from_u32(1U)) {
                        if (error) *error = "sorted-lookup delta/inverse relation is false";
                        return false;
                    }
                }
            }
        }
        for (const auto& bit : merge_bits)
            for (std::uint32_t segment = 0U; segment < bit.segment_count(); ++segment) {
                std::vector<FrWire> block;
                if (!bit.read_segment(segment, block, error)) return false;
                for (const auto& value : block)
                    if (!sorted_lookup_boolean(value)) {
                        if (error) *error = "sorted-lookup merge bit decomposition is non-boolean";
                        return false;
                    }
            }
        return true;
    }
};

inline bool make_segmented_tensor_providers(
    const std::vector<std::vector<FrWire>>& values, std::size_t basis_size,
    std::vector<SegmentedTensorProvider>& providers, std::string* error = nullptr) {
    providers.clear();
    providers.reserve(values.size());
    for (const auto& value : values) {
        SegmentedTensorProvider provider;
        if (!SegmentedTensorProvider::from_values(value, basis_size, provider, error)) return false;
        providers.push_back(std::move(provider));
    }
    return !providers.empty();
}

// The native CPU backend keeps transcript/opening claims as Fr_t while the
// relation wire format uses FrWire.  Compare through the canonical wire
// representation so mixed verifier checks do not depend on an implicit
// conversion that the field implementation intentionally does not provide.
inline int fr_compare_mixed(const Fr_t& a, const Fr_t& b) noexcept {
    return zkdl::fr_compare(zkdl::fr_wire(a), zkdl::fr_wire(b));
}

inline int fr_compare_mixed(const Fr_t& a, const FrWire& b) noexcept {
    return zkdl::fr_compare(zkdl::fr_wire(a), b);
}

inline int fr_compare_mixed(const FrWire& a, const Fr_t& b) noexcept {
    return zkdl::fr_compare(a, zkdl::fr_wire(b));
}

enum class RelationKind : std::uint32_t {
    Projection = 1U,
    AttentionInnerProduct = 2U,
    Product = 3U,
    SparseRoute = 4U,
    Reindex = 5U,
    Add = 6U,
    Bias = 7U,
    GroupSum = 8U,
    TableLookup = 9U,
    MaxSelector = 10U,
    RangeBits = 11U,
    Exp = 12U,
    LeakyRelu = 13U,
    Elu = 14U,
    Inverse = 15U,
    Alpha = 16U,
    FieldIntegerConsistency = 17U,
    SignedRoundDiv = 18U,
    Concat = 19U,
    HeadSum = 20U,
    FinalAverage = 21U,
    PublicOutput = 22U,
    Boolean = 23U,
    Reconstruction = 24U
};

inline const char* relation_kind_name(RelationKind kind) noexcept {
    switch (kind) {
        case RelationKind::Projection: return "projection";
        case RelationKind::AttentionInnerProduct: return "attention_inner_product";
        case RelationKind::Product: return "product";
        case RelationKind::SparseRoute: return "sparse_route";
        case RelationKind::Reindex: return "reindex";
        case RelationKind::Add: return "add";
        case RelationKind::Bias: return "bias";
        case RelationKind::GroupSum: return "group_sum";
        case RelationKind::TableLookup: return "table_lookup";
        case RelationKind::MaxSelector: return "max_selector";
        case RelationKind::RangeBits: return "range_bits";
        case RelationKind::Exp: return "exp";
        case RelationKind::LeakyRelu: return "leaky_relu";
        case RelationKind::Elu: return "elu";
        case RelationKind::Inverse: return "inverse";
        case RelationKind::Alpha: return "alpha";
        case RelationKind::FieldIntegerConsistency: return "field_integer_consistency";
        case RelationKind::SignedRoundDiv: return "signed_round_div";
        case RelationKind::Concat: return "concat";
        case RelationKind::HeadSum: return "head_sum";
        case RelationKind::FinalAverage: return "final_average";
        case RelationKind::PublicOutput: return "public_output";
        case RelationKind::Boolean: return "boolean";
        case RelationKind::Reconstruction: return "reconstruction";
    }
    return "unknown";
}

inline bool relation_kind_valid(std::uint32_t value) noexcept {
    return value >= static_cast<std::uint32_t>(RelationKind::Projection) &&
           value <= static_cast<std::uint32_t>(RelationKind::Reconstruction);
}

/* A monomial is coefficient * product(tensor[ref][x]).  The refs are local
 * indices into OperationDescriptor::tensor_refs, not global commitment
 * indices.  Repeated refs are allowed and intentionally represent powers. */
struct Monomial {
    FrWire coefficient{};
    std::vector<std::uint32_t> refs;
};

struct ConstraintDescriptor {
    std::string id;
    std::uint32_t degree = 0U;
    FrWire constant{};
    std::vector<Monomial> terms;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;

    bool valid(std::string* error = nullptr) const {
        if (id.empty() || degree == 0U || degree > 8U) {
            if (error) *error = "relation constraint has invalid id/degree";
            return false;
        }
        if (logical_shape.size() != padded_shape.size() || logical_shape.size() > 16U) {
            if (error) *error = "relation constraint active shape rank mismatch";
            return false;
        }
        for (std::size_t axis = 0U; axis < logical_shape.size(); ++axis) {
            if (logical_shape[axis] == 0U || padded_shape[axis] < logical_shape[axis] || padded_shape[axis] > (1U << 30U) || (padded_shape[axis] & (padded_shape[axis] - 1U)) != 0U) {
                if (error) *error = "relation constraint active shape is invalid";
                return false;
            }
        }
        if (!zkdl::fr_canonical(constant)) {
            if (error) *error = "relation constraint constant is not canonical";
            return false;
        }
        for (const auto& term : terms) {
            if (term.refs.empty() || term.refs.size() > degree ||
                !zkdl::fr_canonical(term.coefficient)) {
                if (error) *error = "relation monomial is incompatible with degree";
                return false;
            }
        }
        return true;
    }
};

/* A primitive descriptor is part of the public relation plan.  Kind 2 is the
 * formal hidden matrix relation below.  Kind 1 is retained only as an
 * explicit legacy wire value so old proof bytes can be rejected rather than
 * accidentally routed through the value-revealing upstream demo. */
struct PrimitiveDescriptor {
    std::uint32_t kind = 0U;  // 2 = committed-coefficient hidden MatMul
    zkdl::MatMulShape matmul_shape;
    std::vector<std::string> tensor_refs;  // A, B, C in native layout order

    bool valid(const std::vector<std::string>& /*operation_refs*/,
               std::string* error = nullptr) const {
        if (kind != 2U || !matmul_shape.valid(error) || tensor_refs.size() != 3U) {
            if (error && error->empty()) *error = "relation primitive descriptor is not canonical";
            return false;
        }
        std::set<std::string> refs;
        for (const auto& ref : tensor_refs) {
            if (ref.empty() || !refs.insert(ref).second) {
                if (error) *error = "relation primitive tensor reference is not canonical";
                return false;
            }
        }
        std::array<std::uint64_t, 3U> sizes{};
        if (!matmul_shape.tensor_sizes(sizes) || sizes[0U] == 0U ||
            sizes[1U] == 0U || sizes[2U] == 0U ||
            sizes[0U] > (std::uint64_t(1U) << kMaxRelationPointBits) ||
            sizes[1U] > (std::uint64_t(1U) << kMaxRelationPointBits) ||
            sizes[2U] > (std::uint64_t(1U) << kMaxRelationPointBits)) {
            if (error) *error = "relation primitive matrix size is not canonical";
            return false;
        }
        return true;
    }
};

struct FacBindingDescriptor {
    // operation-view reference -> canonical plan reference.  Both ids are
    // public RelationPlan material; the prover cannot choose a selector or
    // reorder the claims after descriptor authentication.
    std::string view_ref;
    std::string canonical_ref;
    // Public selector-MLE layout.  The view is indexed by
    // (view_outer, view_feature); every view slot names the canonical source
    // outer/feature coordinate.  These arrays are reconstructed from the
    // authenticated plan/graph, never selected by the prover.
    std::uint32_t view_outer_domain = 0U;
    std::uint32_t canonical_outer_domain = 0U;
    std::uint32_t view_feature_domain = 0U;
    std::uint32_t canonical_feature_domain = 0U;
    IndexMapU32 source_outer_index;
    IndexMapU32 source_feature_index;
    CowByteVector selector_active;

    bool valid(const std::vector<std::string>& operation_refs,
               std::string* error = nullptr) const {
        const auto present = [&](const std::string& value) {
            return !value.empty() &&
                   std::find(operation_refs.begin(), operation_refs.end(), value) !=
                       operation_refs.end();
        };
        if (!present(view_ref) || !present(canonical_ref) || view_ref == canonical_ref) {
            if (error) *error = "FAC4DNN view binding references are not canonical";
            return false;
        }
        const auto power_of_two = [](std::uint32_t value) {
            return value != 0U && (value & (value - 1U)) == 0U;
        };
        if (!power_of_two(view_outer_domain) || !power_of_two(canonical_outer_domain) ||
            !power_of_two(view_feature_domain) || !power_of_two(canonical_feature_domain) ||
            source_outer_index.size() != view_outer_domain ||
            source_feature_index.size() != view_feature_domain ||
            selector_active.size() != view_outer_domain) {
            if (error) *error = "FAC4DNN selector layout has incomplete power-of-two domains";
            return false;
        }
        for (std::uint32_t slot = 0U; slot < view_outer_domain; ++slot) {
            if (selector_active[slot] > 1U ||
                (!selector_active[slot] && source_outer_index[slot] != 0U) ||
                (selector_active[slot] && source_outer_index[slot] >= canonical_outer_domain)) {
                if (error) *error = "FAC4DNN selector outer index is out of bounds";
                return false;
            }
        }
        for (const auto source : source_feature_index)
            if (source >= canonical_feature_domain) {
                if (error) *error = "FAC4DNN selector feature index is out of bounds";
                return false;
            }
        return true;
    }
};

struct OperationDescriptor {
    std::string id;                 // stage-qualified, never a basename
    RelationKind kind = RelationKind::Add;
    std::uint32_t degree = 0U;
    // source_refs are the logical plan inputs/outputs.  tensor_refs are the
    // canonical, same-domain MLE views consumed by the relation polynomial;
    // reindex/route operations may require explicit derived views.
    std::vector<std::string> source_refs;
    std::vector<std::string> tensor_refs;
    std::vector<std::string> selector_refs;
    std::string mapping;
    std::string padding;
    std::string terminal_claim;
    std::string terminal_point;
    bool public_route = false;
    bool requires_reindex = false;
    std::vector<FrWire> constants;
    std::vector<ConstraintDescriptor> constraints;
    std::vector<PrimitiveDescriptor> primitives;
    // Public selector/equality-MLE descriptor.  For sparse route this is
    // reconstructed from Graph::src/dst by the adapter on both sides.
    std::uint32_t selector_edge_domain = 0U;
    std::uint32_t selector_node_domain = 0U;
    std::uint32_t selector_feature_domain = 0U;
    std::uint32_t selector_source_feature_domain = 0U;
    IndexMapU32 selector_src;
    IndexMapU32 selector_dst;
    // Public graph slots.  Inactive slots are fixed zero padding and must
    // carry src=dst=0; they contribute no selector mass.
    CowByteVector selector_active;
    // Output feature coordinate -> source feature coordinate.  Standard
    // graph routing uses identity; scalar broadcast repeats source 0.
    IndexMapU32 selector_feature_map;
    std::string selector_source_ref;
    std::string selector_output_ref;
    // A normal sparse route selects source[src[e]].  A broadcast route (for
    // example inverse denominator -> edge) selects source[dst[e]].
    bool route_by_destination = false;
    // Public FAC4DNN Line8 view bindings.  A single native proof compresses
    // this fixed-order list with one FS rho and opens the resulting claim
    // against the canonical commitments.
    std::vector<FacBindingDescriptor> fac_bindings;
    // A family proof combines this operation's fixed low-degree constraints
    // with FS coefficients into one Sumcheck.  Empty means ordinary
    // constraint proofs are used.
    std::string batch_family;
    // Authenticated static-table tensor id for indexed lookup families.
    std::string public_table_ref;
    // Canonical public Q_table selector for the logical (unpadded) table
    // rows.  It is descriptor material, not a prover-supplied witness: the
    // adapter derives it from the authenticated table shape and the native
    // relation binds every query's chi leaf to this selector.
    std::vector<FrWire> public_table_selector;
    // A broadcast divisor that is a public scale/constant rather than a
    // witness tensor (output-scale alignment and final head averaging).
    bool has_public_divisor = false;
    FrWire public_divisor{};

    bool authenticated_sorted_lookup_body_valid() const {
        if (kind != RelationKind::TableLookup ||
            batch_family != "authenticated_sorted_lookup" || degree != 2U ||
            tensor_refs.size() != 3U || source_refs.size() != 8U ||
            !constraints.empty() || !primitives.empty() ||
            fac_bindings.size() != 3U || !requires_reindex || public_route ||
            padding != "fixed-zero-padding;authenticated-sorted-lookup-v1" ||
            public_table_ref.empty() ||
            source_refs[1U] != public_table_ref ||
            std::find(source_refs.begin(), source_refs.end(), public_table_ref) == source_refs.end() ||
            std::find(tensor_refs.begin(), tensor_refs.end(), public_table_ref) != tensor_refs.end()) {
            return false;
        }
        const std::array<std::string, 3U> expected_views = {
            id + "/view/query", id + "/view/index", id + "/view/output0"};
        const std::array<std::string, 3U> expected_canonical = {
            source_refs[0U], source_refs[2U], source_refs[4U]};
        const std::set<std::string> tensor_set(tensor_refs.begin(), tensor_refs.end());
        if (tensor_set.size() != 3U || tensor_set != std::set<std::string>(expected_views.begin(), expected_views.end()))
            return false;

        std::set<std::string> bound_views;
        std::uint32_t common_outer = 0U;
        std::uint32_t common_feature = 0U;
        for (const auto& binding : fac_bindings) {
            const auto view = std::find(expected_views.begin(), expected_views.end(), binding.view_ref);
            if (view == expected_views.end() ||
                !bound_views.insert(binding.view_ref).second ||
                binding.canonical_ref.empty() || binding.canonical_ref == binding.view_ref ||
                binding.canonical_ref == public_table_ref ||
                binding.canonical_ref != expected_canonical[static_cast<std::size_t>(view - expected_views.begin())] ||
                std::find(source_refs.begin(), source_refs.end(), binding.canonical_ref) == source_refs.end() ||
                binding.view_outer_domain == 0U ||
                (binding.view_outer_domain & (binding.view_outer_domain - 1U)) != 0U ||
                binding.view_outer_domain != binding.canonical_outer_domain ||
                binding.view_feature_domain == 0U ||
                (binding.view_feature_domain & (binding.view_feature_domain - 1U)) != 0U ||
                binding.view_feature_domain != binding.canonical_feature_domain ||
                binding.source_outer_index.size() != binding.view_outer_domain ||
                binding.selector_active.size() != binding.view_outer_domain ||
                binding.source_feature_index.size() != binding.view_feature_domain) {
                return false;
            }
            if (common_outer == 0U) {
                common_outer = binding.view_outer_domain;
                common_feature = binding.view_feature_domain;
            } else if (binding.view_outer_domain != common_outer ||
                       binding.view_feature_domain != common_feature) {
                return false;
            }
            for (std::uint32_t index = 0U; index < binding.view_outer_domain; ++index)
                if (binding.source_outer_index[index] != index || binding.selector_active[index] != 1U)
                    return false;
            for (std::uint32_t index = 0U; index < binding.view_feature_domain; ++index)
                if (binding.source_feature_index[index] != index) return false;
        }
        return bound_views == tensor_set;
    }

    bool valid(std::string* error = nullptr) const {
        const bool canonical_root = id == "head_sum" || id == "final_average" ||
                                    id == "terminal_opening";
        const bool stage_qualified = id.find('/') != std::string::npos;
        const bool root_kind = (id == "head_sum" && kind == RelationKind::HeadSum) ||
                               (id == "final_average" && kind == RelationKind::FinalAverage) ||
                               (id == "terminal_opening" && kind == RelationKind::PublicOutput);
        const auto power_of_two = [](std::uint32_t value) {
            return value != 0U && (value & (value - 1U)) == 0U;
        };
        if (id.empty() || (!stage_qualified && !canonical_root) ||
            (canonical_root && !root_kind) ||
            !relation_kind_valid(static_cast<std::uint32_t>(kind)) || degree == 0U || degree > 8U) {
            if (error) *error = "relation operation descriptor is not canonical";
            return false;
        }
        if (mapping.empty() || padding.empty() || terminal_claim.empty() || terminal_point.empty()) {
            if (error) *error = "relation operation descriptor is missing mapping/padding/terminal identity";
            return false;
        }
        const bool authenticated_sorted_lookup_body = authenticated_sorted_lookup_body_valid();
        if (batch_family == "authenticated_sorted_lookup" &&
            !authenticated_sorted_lookup_body) {
            if (error) *error = "authenticated sorted-lookup descriptor lacks its complete canonical FAC/table body";
            return false;
        }
        if (source_refs.empty() || tensor_refs.empty() ||
            (constraints.empty() && primitives.empty() &&
             kind != RelationKind::SparseRoute && kind != RelationKind::PublicOutput &&
             !authenticated_sorted_lookup_body)) {
            if (error) *error = "relation operation descriptor has no tensor/constraint body";
            return false;
        }
        std::set<std::string> source_ids;
        for (const auto& ref : source_refs)
            if (ref.empty() || !source_ids.insert(ref).second) {
                if (error) *error = "relation operation has duplicate/non-qualified source ref";
                return false;
            }
        std::set<std::string> refs;
        for (const auto& ref : tensor_refs) {
            if (ref.empty() || !refs.insert(ref).second) {
                if (error) *error = "relation operation has duplicate/non-qualified tensor ref";
                return false;
            }
        }
        std::set<std::string> constraint_ids;
        for (const auto& constraint : constraints) {
            if (!constraint.valid(error) || constraint.degree > degree ||
                !constraint_ids.insert(constraint.id).second) return false;
            for (const auto& term : constraint.terms)
                for (const std::uint32_t ref : term.refs)
                    if (ref >= tensor_refs.size()) {
                        if (error) *error = "relation monomial references tensor outside operation";
                        return false;
                    }
        }
        for (const auto& constant : constants)
            if (!zkdl::fr_canonical(constant)) {
                if (error) *error = "relation operation constant is not canonical";
                return false;
            }
        std::vector<std::string> primitive_refs = tensor_refs;
        primitive_refs.insert(primitive_refs.end(), source_refs.begin(), source_refs.end());
        for (const auto& primitive : primitives)
            if (!primitive.valid(primitive_refs, error)) return false;
        std::set<std::string> fac_views;
        std::vector<std::string> binding_refs = tensor_refs;
        binding_refs.insert(binding_refs.end(), source_refs.begin(), source_refs.end());
        std::string previous_view;
        std::string previous_canonical;
        for (const auto& binding : fac_bindings) {
            if (!binding.valid(binding_refs, error) ||
                !fac_views.insert(binding.view_ref).second ||
                (!previous_view.empty() &&
                 std::pair<std::string, std::string>(previous_view, previous_canonical) >=
                     std::pair<std::string, std::string>(binding.view_ref,
                                                          binding.canonical_ref))) {
                if (error && error->empty()) *error = "duplicate FAC4DNN view binding";
                return false;
            }
            previous_view = binding.view_ref;
            previous_canonical = binding.canonical_ref;
        }
        if (!batch_family.empty() && batch_family != "indexed_public_table" &&
            batch_family != "signed_round_div" &&
            batch_family != "authenticated_sorted_lookup") {
            if (error) *error = "relation operation has an unknown batched family";
            return false;
        }
        if ((batch_family == "indexed_public_table" ||
             batch_family == "authenticated_sorted_lookup") && public_table_ref.empty()) {
            if (error) *error = "indexed table relation has no authenticated table reference";
            return false;
        }
        if ((batch_family == "indexed_public_table" ||
             batch_family == "authenticated_sorted_lookup") &&
            std::find(source_refs.begin(), source_refs.end(), public_table_ref) == source_refs.end()) {
            if (error) *error = "indexed table relation does not expose its table in the public source set";
            return false;
        }
        if (batch_family == "indexed_public_table") {
            if (public_table_selector.empty() ||
                (public_table_selector.size() & (public_table_selector.size() - 1U)) != 0U) {
                if (error) *error = "indexed table relation has no power-of-two Q_table selector";
                return false;
            }
            bool has_legal_row = false;
            for (const auto& selector : public_table_selector) {
                if (!zkdl::fr_canonical(selector) ||
                    (selector != zkdl::fr_from_u32(0U) && selector != zkdl::fr_from_u32(1U))) {
                    if (error) *error = "indexed table Q_table selector is not canonical boolean material";
                    return false;
                }
                has_legal_row = has_legal_row || selector == zkdl::fr_from_u32(1U);
            }
            if (!has_legal_row) {
                if (error) *error = "indexed table Q_table selector has no legal row";
                return false;
            }
        } else if (!public_table_selector.empty()) {
            if (error) *error = "non-table relation carries a Q_table selector";
            return false;
        }
        if (batch_family != "indexed_public_table" &&
            batch_family != "authenticated_sorted_lookup" && !public_table_ref.empty()) {
            if (error) *error = "non-table relation carries an unexpected public table reference";
            return false;
        }
        if (has_public_divisor && !zkdl::fr_canonical(public_divisor)) {
            if (error) *error = "relation public divisor is not canonical";
            return false;
        }
        if (has_public_divisor && batch_family != "signed_round_div") {
            if (error) *error = "non-round-div relation carries a public divisor";
            return false;
        }
        if (constraints.size() > kMaxRelationConstraints) {
            if (error) *error = "relation operation has too many low-degree constraints";
            return false;
        }
        if (!batch_family.empty() && constraints.empty() &&
            batch_family != "authenticated_sorted_lookup") {
            if (error) *error = "batched relation has no fixed public constraint family";
            return false;
        }
        const bool route_relation = kind == RelationKind::SparseRoute ||
                                    !selector_source_ref.empty() ||
                                    !selector_output_ref.empty();
        if (route_relation) {
            if (selector_edge_domain == 0U || selector_node_domain == 0U ||
                selector_feature_domain == 0U || selector_source_feature_domain == 0U ||
                !power_of_two(selector_edge_domain) || !power_of_two(selector_node_domain) ||
                !power_of_two(selector_feature_domain) ||
                !power_of_two(selector_source_feature_domain) ||
                selector_src.size() != selector_edge_domain ||
                selector_dst.size() != selector_edge_domain ||
                selector_active.size() != selector_edge_domain ||
                selector_feature_map.size() != selector_feature_domain ||
                selector_source_ref.empty() || selector_output_ref.empty()) {
                if (error) *error = "sparse route selector descriptor is incomplete";
                return false;
            }
            if (std::find(tensor_refs.begin(), tensor_refs.end(), selector_output_ref) == tensor_refs.end()) {
                if (error) *error = "sparse route output is not a canonical relation tensor";
                return false;
            }
            for (std::uint32_t edge = 0U; edge < selector_edge_domain; ++edge) {
                if (selector_active[edge] > 1U ||
                    (!selector_active[edge] &&
                     (selector_src[edge] != 0U || selector_dst[edge] != 0U)) ||
                    (selector_active[edge] &&
                     (selector_src[edge] >= selector_node_domain ||
                      selector_dst[edge] >= selector_node_domain))) {
                    if (error) *error = "sparse route selector index is out of bounds";
                    return false;
                }
            }
            for (const auto source : selector_feature_map)
                if (source >= selector_source_feature_domain) {
                    if (error) *error = "sparse route feature selector index is out of bounds";
                    return false;
                }
        } else if (selector_edge_domain != 0U || selector_node_domain != 0U ||
                   selector_feature_domain != 0U || selector_source_feature_domain != 0U ||
                   !selector_src.empty() ||
                   !selector_dst.empty() || !selector_active.empty() ||
                   !selector_feature_map.empty() ||
                   !selector_source_ref.empty() ||
                   !selector_output_ref.empty() || route_by_destination) {
            if (error) *error = "non-route operation carries an unexpected selector descriptor";
            return false;
        }
        return true;
    }
};

struct TensorWitness {
    std::string id;
    std::uint32_t visibility = 0U;  // 1 public, 2 registered model, 3 private witness
    std::uint32_t domain_size = 0U;
    std::vector<FrWire> values;
    SegmentedTensorProvider stored_provider;

    bool segmented(std::size_t basis_size, SegmentedTensorProvider& output, std::string* error = nullptr) const {
        if (values.empty()) { std::uint32_t segment_size = 0U, segment_count = 0U; if (!stored_provider.valid(error) || stored_provider.logical_domain() != domain_size || !SegmentedTensorProvider::shape(domain_size, basis_size, segment_size, segment_count, error) || stored_provider.segment_size() != segment_size || stored_provider.segment_count() != segment_count) return false; output = stored_provider; return true; }
        return SegmentedTensorProvider::from_values(values, basis_size, output, error);
    }

    std::vector<FrWire> materialize() const {
        if (!values.empty()) return values;
        if (!stored_provider.valid() || stored_provider.logical_domain() != domain_size || domain_size > (1U << 27U)) throw std::runtime_error("dense native primitive tensor is outside the bounded materialization domain");
        std::vector<FrWire> output;
        output.reserve(domain_size);
        std::string error;
        for (std::uint32_t segment = 0U; segment < stored_provider.segment_count(); ++segment) { std::vector<FrWire> block; if (!stored_provider.read_segment(segment, block, &error)) throw std::runtime_error(error); output.insert(output.end(), block.begin(), block.end()); }
        return output;
    }

    bool metadata_valid(std::string* error = nullptr) const {
        if (id.empty() ||
            domain_size == 0U || (domain_size & (domain_size - 1U)) != 0U ||
            (!values.empty() && values.size() != domain_size) || visibility == 0U || visibility > 3U) {
            if (error) *error = "relation tensor witness has invalid canonical shape";
            return false;
        }
        if (values.empty()) {
            if (!stored_provider.valid(error) || stored_provider.logical_domain() != domain_size) return false;
            return true;
        }
        return true;
    }

    bool valid(std::string* error = nullptr) const {
        if (!metadata_valid(error)) return false;
        if (values.empty()) {
            for (std::uint32_t segment = 0U; segment < stored_provider.segment_count(); ++segment) { std::vector<FrWire> block; if (!stored_provider.read_segment(segment, block, error)) return false; }
            return true;
        }
        for (const auto& value : values)
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "relation tensor witness contains non-canonical field residue";
                return false;
            }
        return true;
    }

    bool materialize_validated(std::vector<FrWire>& output,
                               std::string* error = nullptr,
                               const RouteWorkProgressCallback* progress = nullptr) const {
        output.clear();
        if (!metadata_valid(error)) return false;
        if (!values.empty()) {
            constexpr std::size_t kProgressValues = 1U << 18U;
            for (std::size_t begin = 0U; begin < values.size();) {
                const std::size_t end = progress == nullptr
                    ? values.size() : std::min(values.size(), begin + kProgressValues);
                for (std::size_t index = begin; index < end; ++index)
                    if (!zkdl::fr_canonical(values[index])) {
                        if (error) *error = "relation tensor witness contains non-canonical field residue";
                        return false;
                    }
                begin = end;
                if (progress != nullptr)
                    (*progress)("source_canonical_values", end, values.size());
            }
            output = values;
            return true;
        }
        if (domain_size > (1U << 27U)) {
            if (error) *error = "dense native primitive tensor is outside the bounded materialization domain";
            return false;
        }
        output.reserve(domain_size);
        for (std::uint32_t segment = 0U; segment < stored_provider.segment_count(); ++segment) {
            std::vector<FrWire> block;
            if (!stored_provider.read_segment(segment, block, error) ||
                block.size() != stored_provider.segment_size()) return false;
            output.insert(output.end(), block.begin(), block.end());
            if (progress != nullptr)
                (*progress)("source_materialize_segments", segment + 1U,
                            stored_provider.segment_count());
        }
        if (output.size() != domain_size) {
            if (error) *error = "materialized relation tensor has the wrong canonical length";
            output.clear();
            return false;
        }
        return true;
    }
};

/* Public shape/padding contract used by the independent verifier.  Values are
 * intentionally absent: only the authenticated public graph/table material
 * may be supplied separately through VerifierContext::public_tensor_values. */
struct TensorDescriptor {
    std::string id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    std::vector<std::uint64_t> logical_shape;
    std::vector<std::uint64_t> padded_shape;
    std::string mle_order;
    std::string stack_axis;
    std::string mapping;
    std::string padding_rule;
    std::string registry_id;

    bool valid(std::string* error = nullptr) const {
        if (id.empty() || visibility == 0U || visibility > 3U || domain_size == 0U ||
            logical_shape.empty() || logical_shape.size() != padded_shape.size() ||
            mle_order.empty() || stack_axis.empty() || mapping.empty() ||
            padding_rule.empty()) {
            if (error) *error = "relation tensor descriptor is incomplete";
            return false;
        }
        std::uint64_t product = 1U;
        for (std::size_t i = 0U; i < logical_shape.size(); ++i) {
            if (logical_shape[i] == 0U || padded_shape[i] < logical_shape[i] ||
                padded_shape[i] == 0U || (padded_shape[i] & (padded_shape[i] - 1U)) != 0U ||
                product > std::numeric_limits<std::uint64_t>::max() / padded_shape[i]) {
                if (error) *error = "relation tensor descriptor has invalid logical/padded shape";
                return false;
            }
            product *= padded_shape[i];
        }
        if (product != domain_size ||
            domain_size > (std::uint64_t(1U) << kMaxRelationPointBits) ||
            (domain_size & (domain_size - 1U)) != 0U) {
            if (error) *error = "relation tensor descriptor domain does not match padded shape";
            return false;
        }
        if (visibility == 2U && registry_id.empty()) {
            if (error) *error = "model tensor descriptor has no authenticated registry id";
            return false;
        }
        return true;
    }
};

struct RelationStatement {
    // Versioned segment binding is part of the public relation statement.
    // The digest is filled after the ordered CommitmentWire vector is built;
    // a verifier recomputes it from the parsed wire before replaying FS.
    static constexpr std::uint32_t kSegmentBindingVersion = 1U;
    std::uint32_t segment_binding_version = kSegmentBindingVersion;
    std::array<byte, 32> segment_commitments_digest{};
    std::array<byte, 32> statement_digest{};
    std::array<byte, 32> model_digest{};
    std::array<byte, 32> table_digest{};
    std::array<byte, 32> ledger_digest{};
    std::array<byte, 32> srs_digest{};
    std::array<byte, 32> registry_digest{};
    std::array<byte, 32> public_output_digest{};
    std::vector<byte> public_output;
};

struct SegmentCommitmentWire {
    std::uint32_t segment_index = 0U;
    std::uint64_t logical_offset = 0U;
    std::uint32_t actual_length = 0U;
    std::uint32_t padded_length = 0U;
    std::uint32_t padding_length = 0U;
    std::array<byte, 32U> frame_digest{};
    G1Jacobian_t commitment{};
};

struct CommitmentWire {
    std::string tensor_id;
    std::uint32_t visibility = 0U;
    std::uint32_t domain_size = 0U;
    G1Jacobian_t commitment{};
    // The single G1 above remains the legacy compatibility handle and is the
    // unweighted sum of this ordered vector.  Terminal relation soundness is
    // provided by the authenticated vector/digest, never by that handle.
    std::uint32_t segment_binding_version = RelationStatement::kSegmentBindingVersion;
    std::uint32_t segment_size = 0U;
    std::uint32_t segment_count = 0U;
    std::array<byte, 32U> srs_manifest_digest{};
    std::array<byte, 32U> basis_digest{};
    std::array<byte, 32U> segment_commitment_digest{};
    std::vector<SegmentCommitmentWire> segment_commitments;
};

struct ConstraintProof {
    ConstraintDescriptor descriptor;
    // The terminal relation wire carries one hidden evaluation opening for
    // every canonical operation tensor.  Keep the explicit index maps on the
    // outer wire so the independent verifier can reject a reordered or
    // shortened terminal claim vector before it enters the transcript.
    std::vector<std::uint32_t> opening_tensor_indices;
    std::vector<std::uint32_t> opening_commitment_indices;
    // Formal private constraints use the native zero-knowledge Sumcheck and
    // the Hyrax v3/Pedersen terminal relation together.  There is deliberately
    // no ordinary Sumcheck or value-revealing IPA fallback in this wire.
    zkdl::ZkSumcheckProof zk_sumcheck;
    zkdl::ZkTerminalRelationProof terminal_relation;
};

/* Formal hidden matrix primitive.  The upstream zkMatMul wire exposes row
 * claims and value-revealing openings, so it is not used by the GAT relation.
 * This wire keeps only the authenticated shape/index statement, a hidden
 * committed-coefficient Sumcheck over the inner dimension, three canonical
 * Hyrax v3 openings at verifier challenges, and Sigma links for the terminal
 * product/equality.  No matrix evaluation or blinder is serialized. */
struct HiddenMatMulOpening {
    G1Jacobian_t tau{};
    std::vector<zkdl::HyraxOpeningProofV3> segments;
};

struct HiddenMatMulProof {
    static const std::uint32_t kVersion = 2U;
    static const std::uint64_t kMaxProofBytes = 1U << 28U;
    std::uint32_t version = kVersion;
    zkdl::MatMulShape shape;
    std::uint32_t a_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t b_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t c_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::array<byte, 32U> primitive_digest{};
    std::vector<Fr_t> row_point;
    std::vector<Fr_t> column_point;
    std::vector<Fr_t> batch_point;
    zkdl::ZkSumcheckProof sumcheck;
    HiddenMatMulOpening opening_a;
    HiddenMatMulOpening opening_b;
    HiddenMatMulOpening opening_c;
    zkdl::PedersenProductProof terminal_product;
    zkdl::PedersenScalarEqualityProof terminal_binding;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            HiddenMatMulProof& proof,
                            std::string* error = nullptr);
};

struct OperationProof {
    OperationDescriptor descriptor;
    std::vector<ConstraintProof> constraints;
    // One serialized proof per descriptor primitive, in descriptor order.
    std::vector<std::vector<byte>> primitive_proofs;
    // One binding record per primitive.  Each canonical opening is made
    // against the outer tensor commitment at the exact point used by the
    // official zkMatMul row commitment/opening.
    struct PrimitiveBinding {
        std::uint32_t a_commitment_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t b_commitment_index = std::numeric_limits<std::uint32_t>::max();
        std::uint32_t c_commitment_index = std::numeric_limits<std::uint32_t>::max();
        zkdl::EvaluationOpeningProof opening_a;
        zkdl::EvaluationOpeningProof opening_b;
        zkdl::EvaluationOpeningProof opening_c;
    };
    std::vector<PrimitiveBinding> primitive_bindings;
    std::vector<byte> route_proof;
    // FAC4DNN Line8 relation for all non-identity same-domain views in this
    // operation.  The bytes are parsed strictly before any claim is used.
    std::vector<byte> fac_proof;
    // One serialized ConstraintProof for a fixed-family FS linear
    // combination.  It replaces per-bit/per-row proof repetition.
    std::vector<byte> batch_proof;
};

/* Proof of the public sparse gather relation

       R(e,f) = S(src[e],f).

 * The selector is a public equality-MLE derived from selector_src.  The
 * Sumcheck runs over the source-node coordinates after the edge/feature
 * verifier challenges are fixed, and the two Hyrax openings close the source
 * and routed-output claims against their canonical commitments. */
struct SparseRouteProof {
    // v4 binds one weighted virtual-frame opening to the ordered outer
    // segment vector; v3 unweighted full-domain openings are rejected.
    static const std::uint32_t kVersion = 4U;
    std::uint32_t version = kVersion;
    std::uint32_t source_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t output_commitment_index = std::numeric_limits<std::uint32_t>::max();
    std::uint32_t edge_domain = 0U;
    std::uint32_t node_domain = 0U;
    // Output/view feature domain.  The source feature domain may differ for
    // a public scalar broadcast or another authenticated feature map.
    std::uint32_t feature_domain = 0U;
    std::uint32_t source_feature_domain = 0U;
    std::vector<Fr_t> edge_point;
    std::vector<Fr_t> feature_point;
    std::vector<Fr_t> source_feature_point;
    // FS challenges for the source-node Sumcheck variables.  They are part
    // of the wire so the verifier never has to replay or trust prover state.
    std::vector<Fr_t> node_point;
    // The route is a zero-initial-claim Sumcheck.  Its terminal linear
    // combination is linked to hidden evaluations on the two canonical
    // tensor domains; no private evaluation scalar is present on the wire.
    zkdl::ZkSumcheckProof sumcheck;
    zkdl::HyraxOpeningProofV3 source_opening;
    zkdl::HyraxOpeningProofV3 output_opening;
    zkdl::PedersenScalarEqualityProof terminal_binding;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            SparseRouteProof& proof,
                            std::string* error = nullptr);
};

/* Prover-local output of the hidden sparse-route terminal relation.  These
 * scalars are needed only to aggregate several Line8 terminals after the FS
 * rho challenges have been sampled.  They are never serialized and never
 * enter VerifierContext. */
struct SparseRouteProverSecrets {
    FrWire terminal_value{};
    FrWire residual_blinder{};
    FrWire sumcheck_blinder{};
};

/* FAC4DNN Line8 proof for a fixed, public view-binding list.  Each binding is
 * reduced by a public selector-MLE Sumcheck to one opening of its canonical
 * source commitment.  The hidden terminal commitments are compressed in the
 * authenticated order with FS rho challenges and connected by one Pedersen
 * equality proof.  No terminal evaluation, residual, or compressed claim is
 * carried on the wire. */
struct FacLine8Proof {
    static const std::uint32_t kVersion = 3U;
    std::uint32_t version = kVersion;
    std::vector<SparseRouteProof> selector_proofs;
    zkdl::PedersenScalarEqualityProof aggregate_binding;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            FacLine8Proof& proof,
                            std::string* error = nullptr);
};

/* One same-mask Qpad random-linear-combination proof group.  The selector is
 * reconstructed from the authenticated TensorDescriptors and is never
 * committed or serialized.  Member order is canonical descriptor order. */
struct PaddingProofRecord {
    static const std::uint32_t kVersion = 2U;
    static constexpr std::uint32_t kMaxMembers = 128U;
    std::uint32_t version = kVersion;
    std::uint32_t group_index = std::numeric_limits<std::uint32_t>::max();
    std::array<byte, 32U> group_key{};
    std::uint32_t domain_size = 0U;
    std::vector<std::string> member_tensor_ids;
    std::vector<std::uint32_t> member_commitment_indices;
    ConstraintProof relation;

    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes,
                            PaddingProofRecord& record,
                            std::string* error = nullptr,
                            std::uint64_t max_record_body_bytes =
                                zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes,
                            std::uint64_t max_terminal_body_bytes =
                                zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes);
};

struct Proof {
    static const std::uint32_t kVersion = 8U;
    std::uint32_t version = kVersion;
    RelationStatement statement;
    std::vector<CommitmentWire> commitments;
    std::vector<PaddingProofRecord> padding_records;
    // Legacy hiding records are kept as an explicit zero-length wire field for
    // ABI compatibility and are rejected when non-empty.  Formal GATR binds
    // model/private tensors through the hidden IPA/Pedersen opening proofs.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    std::vector<OperationProof> operations;
    // One authenticated sorted-lookup proof per RelationPlan descriptor, in
    // descriptor order.  The v4 wire has no legacy chi/table-operation form.
    std::vector<zkdl::AuthenticatedSortedLookupProof> sorted_lookup_proofs;
    std::uint32_t public_output_commitment = std::numeric_limits<std::uint32_t>::max();
    HiddenMatMulOpening public_output_opening;
    zkdl::PedersenScalarValueProof public_output_binding;
    std::array<byte, 32> transcript_digest{};

    std::uint64_t serialized_size() const;
    std::vector<byte> serialize() const;
    static bool deserialize(const std::vector<byte>& bytes, Proof& proof,
                            std::string* error = nullptr);
    // A formal verifier supplies the authenticated plan-derived Qpad groups and
    // their native typed per-record byte ceiling.  The generic wire-only
    // entry point above retains its smaller untrusted-item limit.
    static bool deserialize(const std::vector<byte>& bytes, Proof& proof,
                            const std::vector<std::vector<std::string>>& expected_qpad_groups,
                            std::uint64_t max_qpad_record_bytes,
                            std::uint64_t max_terminal_body_bytes,
                            std::uint64_t max_sorted_lookup_relation_body_bytes,
                            std::string* error = nullptr);
};

struct ProverContext {
    RelationStatement statement;
    std::vector<TensorWitness> tensors;
    std::vector<TensorDescriptor> tensor_descriptors;
    std::vector<OperationDescriptor> operations;
    std::vector<zkdl::CpuG1Point> generators;
    // IPA basis material is independent from the scalar-generator vector and
    // must be supplied by the authenticated resource loader.  Empty H/U is a
    // hard admission failure; no generator-derived fallback is permitted.
    zkdl::CpuIpaBasisContext ipa_generators;
    zkdl::CpuG1Point ipa_blinding_generator;
    zkdl::CpuG1Point ipa_u_generator;
    std::vector<CommitmentWire> registered_model_commitments;
    // Legacy hiding-record ABI material is retained only for strict rejection;
    // formal GATR uses the per-tensor Pedersen blinder carried by the native
    // hidden openings below and requires this vector to remain empty.
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    // One blinder is sampled per private/model tensor and reused for every
    // opening of that canonical commitment in this proof.
    std::map<std::string, FrWire> blinding_scalars;
    // Native formal proving uses anonymous O_DIRECT folded-state spill files
    // under its existing private witness directory when a Sumcheck state
    // exceeds the in-memory cache budget. Generic callers leave this empty.
    std::string sumcheck_spill_directory;
    // Native core supplies one bounded provider set and one authenticated
    // outer commitment map per sorted-lookup descriptor.
    std::vector<zkdl::AuthenticatedSortedLookupDescriptor> sorted_lookup_descriptors;
    std::vector<AuthenticatedSortedLookupProviderSet> sorted_lookup_providers;
    std::vector<std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>> sorted_lookup_commitment_refs;
    std::vector<std::vector<zkdl::CpuG1Point>> sorted_lookup_outer_commitments;
    std::vector<std::vector<FrWire>> sorted_lookup_outer_blinders;
    // The native outer stage owns these ordered shares/Xi points.  Relation
    // proving replays them verbatim when constructing v5 lookup wires; it
    // must not sample a second decomposition of the same aggregate handle.
    std::vector<std::vector<std::vector<FrWire>>> sorted_lookup_outer_segment_blinders;
    std::vector<std::vector<std::vector<zkdl::CpuG1Point>>> sorted_lookup_outer_segment_commitments;
    std::vector<std::string> sorted_lookup_commitment_ids;
    // Public graph-derived material may be attached to a non-public view
    // commitment (for example group-boundary bits).  The verifier accepts
    // only ids explicitly reconstructed from Graph/CSR, never arbitrary
    // caller-supplied private values.
    std::set<std::string> graph_public_tensor_ids;
    std::map<std::string, std::vector<FrWire>> public_tensor_values;
    std::map<std::string, SegmentedTensorProvider> public_tensor_providers;
    // Optional native-core timing sink for the actual per-inference tensor
    // commitment operations. It is instrumentation only and never affects
    // proof equations.
    double* commitment_open_time_seconds = nullptr;
    // Portion of commitment_open_time_seconds spent in relation::prove;
    // native-core subtracts this interval from Prove Time exactly once.
    double* relation_commitment_time_seconds = nullptr;
    // Explicit file authentication/read/decode time from witness providers.
    // Mapped payload access and page faults remain in the measured operation.
    double* provider_file_io_decode_time_seconds = nullptr;
    // Explicit provider file I/O/decode consumed by whole commitment calls;
    // this portion is already removed with relation_commitment_time_seconds.
    double* relation_commitment_file_io_decode_time_seconds = nullptr;
    // Native core sets this for a requested CUDA proof.  The relation layer
    // currently fails closed until the complete GAT relation (not only a
    // primitive) is implemented on native CUDA cryptographic kernels.
    bool gpu_backend = false;
    // Borrowed proof-lifetime session for the explicitly selected Hyrax
    // generator-fold CUDA stage.  All relation, scalar and transcript work
    // remains on the CPU; null preserves the ordinary CPU path.
    zkdl::CpuHyraxGpuFoldSession* hyrax_gpu_fold_session = nullptr;
};

struct VerifierContext {
    RelationStatement statement;
    std::vector<OperationDescriptor> operations;
    // The independent verifier may carry shape-only ids for every committed
    // public/model/witness tensor.  When present this set is authoritative;
    // it prevents an envelope from smuggling an unplanned commitment into the
    // transcript.  Pure relation callers may leave it empty, in which case
    // the operation references are the conservative expected set.
    std::vector<std::string> tensor_ids;
    std::map<std::string, std::uint32_t> tensor_visibility;
    std::vector<TensorDescriptor> tensor_descriptors;
    std::vector<CommitmentWire> registered_model_commitments;
    std::vector<zkdl::CommitmentRecord> hiding_commitment_records;
    std::set<std::string> graph_public_tensor_ids;
    std::map<std::string, std::vector<FrWire>> public_tensor_values;
    std::map<std::string, SegmentedTensorProvider> public_tensor_providers;
    std::vector<zkdl::CpuG1Point> generators;
    zkdl::CpuIpaBasisContext ipa_generators;
    zkdl::CpuG1Point ipa_blinding_generator;
    zkdl::CpuG1Point ipa_u_generator;
    std::vector<zkdl::AuthenticatedSortedLookupDescriptor> sorted_lookup_descriptors;
    std::vector<SegmentedTensorProvider> sorted_lookup_table_keys;
    std::vector<SegmentedTensorProvider> sorted_lookup_table_active;
    std::vector<std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>> sorted_lookup_commitment_refs;
    std::vector<std::vector<zkdl::CpuG1Point>> sorted_lookup_outer_commitments;
    std::vector<std::string> sorted_lookup_commitment_ids;
    // Native core sets this for a requested CUDA verification.  The relation
    // layer currently fails closed until every proof family has a native CUDA
    // verifier path.
    bool gpu_backend = false;
};

inline bool same_wire(const FrWire& a, const FrWire& b) noexcept {
    return zkdl::fr_compare(a, b) == 0;
}

inline bool same_bytes(const std::array<byte, 32>& a, const std::array<byte, 32>& b) noexcept {
    return a == b;
}

inline bool same_g1_basis(const std::vector<zkdl::CpuG1Point>& left,
                          const std::vector<zkdl::CpuG1Point>& right) {
    if (left.size() != right.size()) return false;
    for (std::size_t i = 0U; i < left.size(); ++i)
        if (!zkdl::cpu_g1_equal(left[i], right[i])) return false;
    return true;
}

inline std::uint32_t log2_exact(std::uint32_t value) {
    if (value == 0U || (value & (value - 1U)) != 0U) return std::numeric_limits<std::uint32_t>::max();
    std::uint32_t bits = 0U;
    while (value > 1U) { value >>= 1U; ++bits; }
    return bits;
}

inline FrWire fr_from_signed_i64(std::int64_t value) {
    if (value >= 0) { const auto positive = static_cast<std::uint64_t>(value); FrWire out{}; out[0] = static_cast<std::uint32_t>(positive); out[1] = static_cast<std::uint32_t>(positive >> 32U); return out; }
    const std::uint64_t magnitude = static_cast<std::uint64_t>(-(value + 1)) + 1U;
    FrWire out{};
    out[0] = static_cast<std::uint32_t>(magnitude);
    out[1] = static_cast<std::uint32_t>(magnitude >> 32U);
    return zkdl::fr_neg(out);
}

inline FrWire eval_monomial(const Monomial& term,
                            const std::vector<FrWire>& values) {
    FrWire result = term.coefficient;
    for (const std::uint32_t ref : term.refs) result = zkdl::fr_mul(result, values.at(ref));
    return result;
}

inline FrWire eval_constraint(const ConstraintDescriptor& descriptor,
                              const std::vector<FrWire>& values) {
    FrWire result = descriptor.constant;
    for (const auto& term : descriptor.terms)
        result = zkdl::fr_add(result, eval_monomial(term, values));
    return result;
}

inline std::vector<ConstraintDescriptor> field_integer_consistency_constraints(const std::string& id, std::uint32_t denominator, std::uint32_t aggregate_field, std::uint32_t aggregate_integer, std::uint32_t residual) {
    ConstraintDescriptor consistency;
    consistency.id = id + "/field-integer";
    consistency.degree = 2U;
    consistency.terms = {Monomial{zkdl::fr_from_u32(1U), {denominator, aggregate_field}}, Monomial{zkdl::fr_neg(zkdl::fr_from_u32(1U)), {aggregate_integer}}};
    ConstraintDescriptor residual_zero;
    residual_zero.id = id + "/field-integer-residual-zero";
    residual_zero.degree = 1U;
    residual_zero.terms = {Monomial{zkdl::fr_from_u32(1U), {residual}}};
    return {consistency, residual_zero};
}

template <typename Output>
inline void append_fr_wire(Output& out, const FrWire& value) {
    for (const std::uint32_t limb : value) zkdl::append_u32(out, limb);
}

inline bool read_fr_wire(const std::vector<byte>& in, size_t& offset, FrWire& value) {
    for (std::uint32_t& limb : value) if (!zkdl::read_u32(in, offset, limb)) return false;
    return zkdl::fr_canonical(value);
}

template <typename Output>
inline void append_digest(Output& out, const std::array<byte, 32>& value) {
    out.insert(out.end(), value.begin(), value.end());
}

inline bool read_digest(const std::vector<byte>& in, size_t& offset,
                        std::array<byte, 32>& value) {
    if (offset > in.size() || in.size() - offset < value.size()) return false;
    std::copy(in.begin() + static_cast<std::ptrdiff_t>(offset),
              in.begin() + static_cast<std::ptrdiff_t>(offset + value.size()), value.begin());
    offset += value.size();
    return true;
}

template <typename Output>
inline void append_string_vector(Output& out,
                                 const std::vector<std::string>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) zkdl::append_string(out, value);
}

inline bool read_string_vector(const std::vector<byte>& in, size_t& offset,
                               std::vector<std::string>& values) {
    std::uint64_t count = 0;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    values.clear(); values.reserve(static_cast<size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        std::string value;
        if (!zkdl::read_string(in, offset, value)) return false;
        values.push_back(std::move(value));
    }
    return true;
}

template <typename Output>
inline void append_fr_vector(Output& out, const std::vector<FrWire>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) append_fr_wire(out, value);
}

inline void append_fr_vector(zkdl::WireSizeCounter& out,
                             const std::vector<FrWire>& values) {
    const std::uint64_t count = static_cast<std::uint64_t>(values.size());
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire))
        throw std::overflow_error("field vector wire size overflows uint64");
    zkdl::append_u64(out, count);
    out.add(count * sizeof(FrWire));
}

template <typename Output>
inline void append_u32_vector(Output& out,
                              const CowByteVector& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) zkdl::append_u32(out, value);
}

inline void append_u32_vector(zkdl::WireSizeCounter& out,
                              const CowByteVector& values) {
    const std::uint64_t count = static_cast<std::uint64_t>(values.size());
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t))
        throw std::overflow_error("u32 vector wire size overflows uint64");
    zkdl::append_u64(out, count);
    out.add(count * sizeof(std::uint32_t));
}

inline bool read_fr_vector(const std::vector<byte>& in, size_t& offset,
                           std::vector<FrWire>& values) {
    std::uint64_t count = 0;
    if (!zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    values.resize(static_cast<size_t>(count));
    for (auto& value : values) if (!read_fr_wire(in, offset, value)) return false;
    return true;
}

inline std::vector<byte> encode_public_field_vector(const std::vector<FrWire>& values) {
    std::vector<byte> encoded;
    append_fr_vector(encoded, values);
    return encoded;
}

inline bool decode_public_field_vector(const std::vector<byte>& encoded,
                                      std::vector<FrWire>& values) {
    size_t offset = 0U;
    if (!read_fr_vector(encoded, offset, values) || offset != encoded.size() || values.empty()) return false;
    return true;
}

inline void fold_values(std::vector<FrWire>& values, const FrWire& challenge);

inline FrWire mle_value(std::vector<FrWire> values, const std::vector<FrWire>& point) {
    if (values.empty() || values.size() != (size_t(1U) << point.size()))
        throw std::invalid_argument("public MLE shape mismatch");
    for (const auto& challenge : point) fold_values(values, challenge);
    return values.front();
}

template <typename Output>
inline void append_monomial(Output& out, const Monomial& term) {
    append_fr_wire(out, term.coefficient);
    zkdl::append_u64(out, static_cast<std::uint64_t>(term.refs.size()));
    for (const std::uint32_t ref : term.refs) zkdl::append_u32(out, ref);
}

inline bool read_monomial(const std::vector<byte>& in, size_t& offset, Monomial& term) {
    std::uint64_t count = 0;
    if (!read_fr_wire(in, offset, term.coefficient) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    term.refs.resize(static_cast<size_t>(count));
    for (std::uint32_t& ref : term.refs) if (!zkdl::read_u32(in, offset, ref)) return false;
    return true;
}

template <typename Output>
inline void append_constraint(Output& out, const ConstraintDescriptor& descriptor) {
    zkdl::append_string(out, descriptor.id);
    zkdl::append_u32(out, descriptor.degree);
    append_fr_wire(out, descriptor.constant);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.terms.size()));
    for (const auto& term : descriptor.terms) append_monomial(out, term);
    zkdl::append_u64(out, descriptor.logical_shape.size());
    for (const auto axis : descriptor.logical_shape) zkdl::append_u64(out, axis);
    for (const auto axis : descriptor.padded_shape) zkdl::append_u64(out, axis);
}

inline bool read_constraint(const std::vector<byte>& in, size_t& offset,
                            ConstraintDescriptor& descriptor) {
    std::uint64_t count = 0;
    if (!zkdl::read_string(in, offset, descriptor.id) ||
        !zkdl::read_u32(in, offset, descriptor.degree) ||
        !read_fr_wire(in, offset, descriptor.constant) ||
        !zkdl::read_u64(in, offset, count) || count > kMaxRelationTerms) return false;
    descriptor.terms.resize(static_cast<size_t>(count));
    for (auto& term : descriptor.terms) if (!read_monomial(in, offset, term)) return false;
    if (!zkdl::read_u64(in, offset, count) || count > 16U) return false;
    descriptor.logical_shape.resize(static_cast<std::size_t>(count));
    descriptor.padded_shape.resize(static_cast<std::size_t>(count));
    for (auto& axis : descriptor.logical_shape) if (!zkdl::read_u64(in, offset, axis)) return false;
    for (auto& axis : descriptor.padded_shape) if (!zkdl::read_u64(in, offset, axis)) return false;
    return descriptor.valid();
}

template <typename Output>
inline void append_hiding_record(Output& out,
                                 const zkdl::CommitmentRecord& record) {
    zkdl::append_string(out, record.label);
    zkdl::append_u32(out, record.visibility);
    zkdl::append_u64(out, static_cast<std::uint64_t>(record.points.size()));
    for (const auto& point : record.points) zkdl::append_g1(out, point);
    zkdl::append_g1(out, record.blinding_generator);
    zkdl::append_g1(out, record.commitment);
}

inline bool read_hiding_record(const std::vector<byte>& in, size_t& offset,
                               zkdl::CommitmentRecord& record) {
    std::uint64_t count = 0U;
    if (!zkdl::read_string(in, offset, record.label) ||
        !zkdl::read_u32(in, offset, record.visibility) ||
        !zkdl::read_u64(in, offset, count) || count > (1U << 20)) return false;
    record.points.resize(static_cast<std::size_t>(count));
    for (auto& point : record.points)
        if (!zkdl::read_g1(in, offset, point)) return false;
    return zkdl::read_g1(in, offset, record.blinding_generator) &&
           zkdl::read_g1(in, offset, record.commitment);
}

template <typename Output>
inline void append_native_fr_vector(Output& out,
                                    const std::vector<Fr_t>& values);
inline bool read_native_fr_vector(const std::vector<byte>& in, size_t& offset,
                                  std::vector<Fr_t>& values);
struct ConstraintProof;
inline std::vector<byte> serialize_constraint_payload(const ConstraintProof& proof);
inline bool deserialize_constraint_payload(const std::vector<byte>& bytes,
                                           ConstraintProof& proof,
                                           std::string* error = nullptr,
                                           std::uint64_t max_terminal_body_bytes =
                                               zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes);

template <typename Output>
inline void append_primitive(Output& out,
                             const PrimitiveDescriptor& primitive) {
    zkdl::append_u32(out, primitive.kind);
    zkdl::append_u32(out, primitive.matmul_shape.num);
    zkdl::append_u32(out, primitive.matmul_shape.m);
    zkdl::append_u32(out, primitive.matmul_shape.n);
    zkdl::append_u32(out, primitive.matmul_shape.k);
    append_string_vector(out, primitive.tensor_refs);
}

inline bool read_primitive(const std::vector<byte>& in, size_t& offset,
                           PrimitiveDescriptor& primitive) {
    return zkdl::read_u32(in, offset, primitive.kind) &&
           zkdl::matmul_read_shape(in, offset, primitive.matmul_shape) &&
           read_string_vector(in, offset, primitive.tensor_refs);
}

inline bool relation_descriptor_domain_allowed(std::uint32_t domain,
                                                bool allow_zero) noexcept {
    if (domain == 0U) return allow_zero;
    return domain <= (std::uint64_t{1} << kMaxRelationPointBits) &&
           (domain & (domain - 1U)) == 0U;
}

inline bool read_index_map_exact(const std::vector<byte>& in, size_t& offset,
                                 std::uint32_t expected_count,
                                 IndexMapU32& output) {
    std::uint64_t count = 0U;
    if (!zkdl::read_u64(in, offset, count) || count != expected_count ||
        count > (std::uint64_t{1} << kMaxRelationPointBits) || offset > in.size() ||
        count > (in.size() - offset) / sizeof(std::uint32_t)) return false;

    const size_t values_offset = offset;
    size_t scan_offset = values_offset;
    bool identity = true;
    for (std::uint64_t index = 0U; index < count; ++index) {
        std::uint32_t value = 0U;
        if (!zkdl::read_u32(in, scan_offset, value)) return false;
        if (value != index) identity = false;
    }
    if (identity) {
        output.set_identity(static_cast<std::size_t>(count));
        offset = scan_offset;
        return true;
    }

    output.set_explicit_size(static_cast<std::size_t>(count));
    scan_offset = values_offset;
    for (std::size_t index = 0U; index < static_cast<std::size_t>(count); ++index) {
        std::uint32_t value = 0U;
        if (!zkdl::read_u32(in, scan_offset, value)) return false;
        output[index] = value;
    }
    offset = scan_offset;
    return true;
}

inline bool read_selector_active_exact(const std::vector<byte>& in,
                                       size_t& offset,
                                       std::uint32_t expected_count,
                                       CowByteVector& output) {
    std::uint64_t count = 0U;
    if (!zkdl::read_u64(in, offset, count) || count != expected_count ||
        count > (std::uint64_t{1} << kMaxRelationPointBits) || offset > in.size() ||
        count > (in.size() - offset) / sizeof(std::uint32_t)) return false;
    output.resize(static_cast<std::size_t>(count));
    for (auto& active : output) {
        std::uint32_t value = 0U;
        if (!zkdl::read_u32(in, offset, value) || value > 1U) return false;
        active = static_cast<std::uint8_t>(value);
    }
    return true;
}

template <typename Output>
inline void append_index_map(Output& out, const IndexMapU32& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto value : values) zkdl::append_u32(out, value);
}

inline void append_index_map(zkdl::WireSizeCounter& out,
                             const IndexMapU32& values) {
    const std::uint64_t count = static_cast<std::uint64_t>(values.size());
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(std::uint32_t))
        throw std::overflow_error("identity index map wire size overflows uint64");
    zkdl::append_u64(out, count);
    out.add(count * sizeof(std::uint32_t));
}

template <typename Output>
inline void append_operation(Output& out, const OperationDescriptor& descriptor) {
    zkdl::append_string(out, descriptor.id);
    zkdl::append_u32(out, static_cast<std::uint32_t>(descriptor.kind));
    zkdl::append_u32(out, descriptor.degree);
    append_string_vector(out, descriptor.source_refs);
    append_string_vector(out, descriptor.tensor_refs);
    append_string_vector(out, descriptor.selector_refs);
    zkdl::append_string(out, descriptor.mapping);
    zkdl::append_string(out, descriptor.padding);
    zkdl::append_string(out, descriptor.terminal_claim);
    zkdl::append_string(out, descriptor.terminal_point);
    zkdl::append_u32(out, descriptor.public_route ? 1U : 0U);
    zkdl::append_u32(out, descriptor.requires_reindex ? 1U : 0U);
    append_fr_vector(out, descriptor.constants);
    zkdl::append_u32(out, descriptor.selector_edge_domain);
    zkdl::append_u32(out, descriptor.selector_node_domain);
    zkdl::append_u32(out, descriptor.selector_feature_domain);
    zkdl::append_u32(out, descriptor.selector_source_feature_domain);
    append_index_map(out, descriptor.selector_src);
    append_index_map(out, descriptor.selector_dst);
    append_u32_vector(out, descriptor.selector_active);
    append_index_map(out, descriptor.selector_feature_map);
    zkdl::append_string(out, descriptor.selector_source_ref);
    zkdl::append_string(out, descriptor.selector_output_ref);
    zkdl::append_u32(out, descriptor.route_by_destination ? 1U : 0U);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.fac_bindings.size()));
    for (const auto& binding : descriptor.fac_bindings) {
        zkdl::append_string(out, binding.view_ref);
        zkdl::append_string(out, binding.canonical_ref);
        zkdl::append_u32(out, binding.view_outer_domain);
        zkdl::append_u32(out, binding.canonical_outer_domain);
        zkdl::append_u32(out, binding.view_feature_domain);
        zkdl::append_u32(out, binding.canonical_feature_domain);
        append_index_map(out, binding.source_outer_index);
        append_index_map(out, binding.source_feature_index);
        append_u32_vector(out, binding.selector_active);
    }
    zkdl::append_string(out, descriptor.batch_family);
    zkdl::append_string(out, descriptor.public_table_ref);
    append_fr_vector(out, descriptor.public_table_selector);
    zkdl::append_u32(out, descriptor.has_public_divisor ? 1U : 0U);
    append_fr_wire(out, descriptor.public_divisor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.primitives.size()));
    for (const auto& primitive : descriptor.primitives) append_primitive(out, primitive);
    zkdl::append_u64(out, static_cast<std::uint64_t>(descriptor.constraints.size()));
    for (const auto& constraint : descriptor.constraints) append_constraint(out, constraint);
}

inline zkdl::CpuIpaOpeningMetadata relation_opening_metadata(
    const OperationDescriptor& descriptor, std::uint32_t commitment_index,
    const std::vector<FrWire>& outer_point, const RelationStatement& statement) {
    if (statement.registry_digest == std::array<byte, 32U>{} ||
        statement.srs_digest == std::array<byte, 32U>{})
        throw std::invalid_argument("relation opening requires authenticated registry/SRS digests");
    std::vector<byte> encoded;
    append_operation(encoded, descriptor);
    zkdl::append_u32(encoded, commitment_index);
    for (const auto& coordinate : outer_point) zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.outer_point = outer_point;
    metadata.descriptor_digest = zkdl::sha256(encoded);
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

/* Constraint openings use the same authenticated IPA header as operation
 * openings, but their public descriptor is the canonical constraint wire.
 * Keep this as a distinct overload so a verifier cannot silently hash a
 * lossy OperationDescriptor projection (and so prover/verifier derive the
 * exact same commitment-opening identity). */
inline zkdl::CpuIpaOpeningMetadata relation_opening_metadata(
    const ConstraintDescriptor& descriptor, std::uint32_t commitment_index,
    const std::vector<FrWire>& outer_point, const RelationStatement& statement) {
    if (statement.registry_digest == std::array<byte, 32U>{} ||
        statement.srs_digest == std::array<byte, 32U>{})
        throw std::invalid_argument("relation opening requires authenticated registry/SRS digests");
    std::vector<byte> encoded;
    append_constraint(encoded, descriptor);
    zkdl::append_u32(encoded, commitment_index);
    for (const auto& coordinate : outer_point)
        zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.outer_point = outer_point;
    metadata.descriptor_digest = zkdl::sha256(encoded);
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

inline OperationDescriptor public_output_opening_descriptor() {
    OperationDescriptor descriptor;
    descriptor.id = "terminal_opening";
    descriptor.kind = RelationKind::PublicOutput;
    descriptor.degree = 1U;
    descriptor.source_refs = {"public_Y"};
    descriptor.tensor_refs = {"public_Y"};
    descriptor.mapping = "canonical-public-output";
    descriptor.padding = "authenticated-public-output";
    descriptor.terminal_claim = "public_Y";
    descriptor.terminal_point = "public_Y";
    return descriptor;
}

inline FrWire relation_blinding_scalar(
    const TensorWitness& tensor, const std::map<std::string, FrWire>& blinders) {
    if (tensor.visibility == 1U) return FrWire{};
    const auto found = blinders.find(tensor.id);
    if (found == blinders.end() || !zkdl::fr_canonical(found->second) ||
        zkdl::fr_compare(found->second, FrWire{}) == 0)
        throw std::invalid_argument("BLOCKED: missing nonzero per-tensor IPA blinder: " + tensor.id);
    return found->second;
}

/* Every non-public canonical tensor gets one independently sampled scalar.
 * A map prevents duplicate ids, but it does not prevent two tensor ids from
 * accidentally sharing the same scalar (for example after importing a
 * malformed model-blinder sidecar).  That would couple the hidden opening
 * witnesses and is outside the formal relation contract. */
inline bool relation_blinding_scalars_unique(
    const std::vector<TensorWitness>& tensors,
    const std::map<std::string, FrWire>& blinders,
    std::string* error = nullptr) {
    std::set<std::string> expected_ids;
    std::vector<FrWire> seen;
    for (const auto& tensor : tensors) {
        if (tensor.visibility == 1U) continue;
        if (!expected_ids.insert(tensor.id).second) {
            if (error) *error = "duplicate tensor id in blinding-scalar contract: " + tensor.id;
            return false;
        }
        const auto found = blinders.find(tensor.id);
        if (found == blinders.end() || !zkdl::fr_canonical(found->second) ||
            zkdl::fr_compare(found->second, FrWire{}) == 0) {
            if (error) *error = "missing/nonzero blinding scalar for tensor: " + tensor.id;
            return false;
        }
        for (const auto& prior : seen) {
            if (zkdl::fr_compare(prior, found->second) == 0) {
                if (error) *error = "blinding scalar is reused across private/model tensors";
                return false;
            }
        }
        seen.push_back(found->second);
    }
    if (blinders.size() != expected_ids.size()) {
        if (error) *error = "blinding-scalar map contains an unexpected tensor id";
        return false;
    }
    for (const auto& entry : blinders)
        if (!expected_ids.count(entry.first)) {
            if (error) *error = "blinding-scalar map contains an unexpected tensor id: " + entry.first;
            return false;
        }
    return true;
}

/* The operation reader keeps its two boolean fields separate so their
 * canonical representation is unambiguous across languages. */
inline bool read_operation_strict(const std::vector<byte>& in, size_t& offset,
                                  OperationDescriptor& descriptor) {
    std::uint32_t kind = 0U, public_route = 0U, reindex = 0U,
                  route_by_destination = 0U;
    std::uint64_t count = 0U;
    if (!zkdl::read_string(in, offset, descriptor.id) ||
        !zkdl::read_u32(in, offset, kind) || !relation_kind_valid(kind) ||
        !zkdl::read_u32(in, offset, descriptor.degree) ||
        !read_string_vector(in, offset, descriptor.source_refs) ||
        !read_string_vector(in, offset, descriptor.tensor_refs) ||
        !read_string_vector(in, offset, descriptor.selector_refs) ||
        !zkdl::read_string(in, offset, descriptor.mapping) ||
        !zkdl::read_string(in, offset, descriptor.padding) ||
        !zkdl::read_string(in, offset, descriptor.terminal_claim) ||
        !zkdl::read_string(in, offset, descriptor.terminal_point) ||
        !zkdl::read_u32(in, offset, public_route) || public_route > 1U ||
        !zkdl::read_u32(in, offset, reindex) || reindex > 1U ||
        !read_fr_vector(in, offset, descriptor.constants) ||
        !zkdl::read_u32(in, offset, descriptor.selector_edge_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_node_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_feature_domain) ||
        !zkdl::read_u32(in, offset, descriptor.selector_source_feature_domain) ||
        !relation_descriptor_domain_allowed(descriptor.selector_edge_domain, true) ||
        !relation_descriptor_domain_allowed(descriptor.selector_node_domain, true) ||
        !relation_descriptor_domain_allowed(descriptor.selector_feature_domain, true) ||
        !relation_descriptor_domain_allowed(descriptor.selector_source_feature_domain, true) ||
        !read_index_map_exact(in, offset, descriptor.selector_edge_domain,
                              descriptor.selector_src) ||
        !read_index_map_exact(in, offset, descriptor.selector_edge_domain,
                              descriptor.selector_dst) ||
        !read_selector_active_exact(in, offset, descriptor.selector_edge_domain,
                                    descriptor.selector_active) ||
        !read_index_map_exact(in, offset, descriptor.selector_feature_domain,
                              descriptor.selector_feature_map)) return false;
    if (!zkdl::read_string(in, offset, descriptor.selector_source_ref) ||
        !zkdl::read_string(in, offset, descriptor.selector_output_ref) ||
        !zkdl::read_u32(in, offset, route_by_destination) || route_by_destination > 1U ||
        !zkdl::read_u64(in, offset, count) || count > kMaxRelationOpenings) return false;
    descriptor.fac_bindings.resize(static_cast<std::size_t>(count));
    for (auto& binding : descriptor.fac_bindings) {
        if (!zkdl::read_string(in, offset, binding.view_ref) ||
            !zkdl::read_string(in, offset, binding.canonical_ref) ||
            !zkdl::read_u32(in, offset, binding.view_outer_domain) ||
            !zkdl::read_u32(in, offset, binding.canonical_outer_domain) ||
            !zkdl::read_u32(in, offset, binding.view_feature_domain) ||
            !zkdl::read_u32(in, offset, binding.canonical_feature_domain) ||
            !relation_descriptor_domain_allowed(binding.view_outer_domain, false) ||
            !relation_descriptor_domain_allowed(binding.canonical_outer_domain, false) ||
            !relation_descriptor_domain_allowed(binding.view_feature_domain, false) ||
            !relation_descriptor_domain_allowed(binding.canonical_feature_domain, false) ||
            !read_index_map_exact(in, offset, binding.view_outer_domain,
                                  binding.source_outer_index) ||
            !read_index_map_exact(in, offset, binding.view_feature_domain,
                                  binding.source_feature_index) ||
            !read_selector_active_exact(in, offset, binding.view_outer_domain,
                                        binding.selector_active)) return false;
    }
    std::uint32_t has_public_divisor = 0U;
    if (!zkdl::read_string(in, offset, descriptor.batch_family) ||
        !zkdl::read_string(in, offset, descriptor.public_table_ref) ||
        !read_fr_vector(in, offset, descriptor.public_table_selector) ||
        !zkdl::read_u32(in, offset, has_public_divisor) || has_public_divisor > 1U ||
        !read_fr_wire(in, offset, descriptor.public_divisor) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    descriptor.has_public_divisor = has_public_divisor != 0U;
    descriptor.primitives.resize(static_cast<size_t>(count));
    for (auto& primitive : descriptor.primitives)
        if (!read_primitive(in, offset, primitive)) return false;
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationConstraints ||
        (count == 0U && descriptor.primitives.empty() &&
         kind != static_cast<std::uint32_t>(RelationKind::PublicOutput))) return false;
    descriptor.kind = static_cast<RelationKind>(kind);
    descriptor.public_route = public_route != 0U;
    descriptor.requires_reindex = reindex != 0U;
    descriptor.route_by_destination = route_by_destination != 0U;
    descriptor.constraints.resize(static_cast<size_t>(count));
    for (auto& constraint : descriptor.constraints)
        if (!read_constraint(in, offset, constraint)) return false;
    return descriptor.valid();
}

template <typename Output>
inline void append_commitment(Output& out, const CommitmentWire& commitment) {
    zkdl::append_string(out, commitment.tensor_id);
    zkdl::append_u32(out, commitment.visibility);
    zkdl::append_u32(out, commitment.domain_size);
    zkdl::append_g1(out, commitment.commitment);
    zkdl::append_u32(out, commitment.segment_binding_version);
    zkdl::append_u32(out, commitment.segment_size);
    zkdl::append_u32(out, commitment.segment_count);
    append_digest(out, commitment.srs_manifest_digest);
    append_digest(out, commitment.basis_digest);
    append_digest(out, commitment.segment_commitment_digest);
    zkdl::append_u64(out, static_cast<std::uint64_t>(commitment.segment_commitments.size()));
    for (const auto& segment : commitment.segment_commitments) {
        zkdl::append_u32(out, segment.segment_index);
        zkdl::append_u64(out, segment.logical_offset);
        zkdl::append_u32(out, segment.actual_length);
        zkdl::append_u32(out, segment.padded_length);
        zkdl::append_u32(out, segment.padding_length);
        append_digest(out, segment.frame_digest);
        zkdl::append_g1(out, segment.commitment);
    }
}

inline void append_commitment(zkdl::WireSizeCounter& out,
                              const CommitmentWire& commitment) {
    zkdl::append_string(out, commitment.tensor_id);
    zkdl::append_u32(out, commitment.visibility);
    zkdl::append_u32(out, commitment.domain_size);
    zkdl::append_g1(out, commitment.commitment);
    zkdl::append_u32(out, commitment.segment_binding_version);
    zkdl::append_u32(out, commitment.segment_size);
    zkdl::append_u32(out, commitment.segment_count);
    append_digest(out, commitment.srs_manifest_digest);
    append_digest(out, commitment.basis_digest);
    append_digest(out, commitment.segment_commitment_digest);
    const std::uint64_t count =
        static_cast<std::uint64_t>(commitment.segment_commitments.size());
    zkdl::append_u64(out, count);
    // Each legacy segment record is index(4), offset(8), three lengths(12),
    // frame digest(32), and compressed G1(48).  Count the whole vector in
    // constant time rather than walking tens of thousands of points twice.
    constexpr std::uint64_t kSegmentWireBytes = 104U;
    if (count > std::numeric_limits<std::uint64_t>::max() / kSegmentWireBytes)
        throw std::overflow_error("segment commitment wire size overflows uint64");
    out.add(count * kSegmentWireBytes);
}

inline bool read_commitment(const std::vector<byte>& in, size_t& offset,
                           CommitmentWire& commitment) {
    std::uint64_t count = 0U;
    if (!zkdl::read_string(in, offset, commitment.tensor_id) ||
        !zkdl::read_u32(in, offset, commitment.visibility) ||
        !zkdl::read_u32(in, offset, commitment.domain_size) ||
        !zkdl::read_g1(in, offset, commitment.commitment) ||
        !zkdl::read_u32(in, offset, commitment.segment_binding_version) ||
        !zkdl::read_u32(in, offset, commitment.segment_size) ||
        !zkdl::read_u32(in, offset, commitment.segment_count) ||
        !read_digest(in, offset, commitment.srs_manifest_digest) ||
        !read_digest(in, offset, commitment.basis_digest) ||
        !read_digest(in, offset, commitment.segment_commitment_digest) ||
        !zkdl::read_u64(in, offset, count) || count == 0U ||
        count > kMaxRelationOpenings) return false;
    commitment.segment_commitments.resize(static_cast<std::size_t>(count));
    for (auto& segment : commitment.segment_commitments) {
        if (!zkdl::read_u32(in, offset, segment.segment_index) ||
            !zkdl::read_u64(in, offset, segment.logical_offset) ||
            !zkdl::read_u32(in, offset, segment.actual_length) ||
            !zkdl::read_u32(in, offset, segment.padded_length) ||
            !zkdl::read_u32(in, offset, segment.padding_length) ||
            !read_digest(in, offset, segment.frame_digest) ||
            !zkdl::read_g1(in, offset, segment.commitment)) return false;
    }
    return true;
}

template <typename Output>
inline void append_statement(Output& out, const RelationStatement& statement) {
    zkdl::append_u32(out, statement.segment_binding_version);
    append_digest(out, statement.segment_commitments_digest);
    append_digest(out, statement.statement_digest);
    append_digest(out, statement.model_digest);
    append_digest(out, statement.table_digest);
    append_digest(out, statement.ledger_digest);
    append_digest(out, statement.srs_digest);
    append_digest(out, statement.registry_digest);
    append_digest(out, statement.public_output_digest);
    zkdl::append_u64(out, static_cast<std::uint64_t>(statement.public_output.size()));
    out.insert(out.end(), statement.public_output.begin(), statement.public_output.end());
}

inline bool read_statement(const std::vector<byte>& in, size_t& offset,
                           RelationStatement& statement) {
    std::uint64_t bytes = 0U;
    if (!zkdl::read_u32(in, offset, statement.segment_binding_version) ||
        !read_digest(in, offset, statement.segment_commitments_digest) ||
        !read_digest(in, offset, statement.statement_digest) ||
        !read_digest(in, offset, statement.model_digest) ||
        !read_digest(in, offset, statement.table_digest) ||
        !read_digest(in, offset, statement.ledger_digest) ||
        !read_digest(in, offset, statement.srs_digest) ||
        !read_digest(in, offset, statement.registry_digest) ||
        !read_digest(in, offset, statement.public_output_digest) ||
        !zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    statement.public_output.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                                   in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    return true;
}

inline bool segment_commitment_wire_shape_valid(
    const CommitmentWire& commitment, std::string* error = nullptr) {
    const auto fail = [&](const char* message) {
        if (error) *error = message;
        return false;
    };
    if (commitment.tensor_id.empty() || commitment.visibility == 0U ||
        commitment.domain_size == 0U || commitment.segment_binding_version !=
            RelationStatement::kSegmentBindingVersion || commitment.segment_size == 0U ||
        (commitment.segment_size & (commitment.segment_size - 1U)) != 0U ||
        commitment.segment_count == 0U ||
        commitment.segment_count != commitment.segment_commitments.size() ||
        static_cast<std::uint64_t>(commitment.segment_size) * commitment.segment_count !=
            commitment.domain_size || commitment.srs_manifest_digest == std::array<byte, 32U>{} ||
        commitment.basis_digest == std::array<byte, 32U>{} ||
        commitment.segment_commitment_digest == std::array<byte, 32U>{})
        return fail("segment commitment wire header is not canonical");
    for (std::size_t index = 0U; index < commitment.segment_commitments.size(); ++index) {
        const auto& segment = commitment.segment_commitments[index];
        const std::uint64_t expected_offset =
            static_cast<std::uint64_t>(index) * commitment.segment_size;
        std::uint32_t expected_actual = 0U;
        if (!checked_relation_u32(
                std::min<std::uint64_t>(commitment.segment_size,
                                        commitment.domain_size - expected_offset),
                expected_actual, "segment commitment actual length", error))
            return false;
        if (segment.segment_index != index || segment.logical_offset != expected_offset ||
            segment.padded_length != commitment.segment_size ||
            segment.actual_length != expected_actual ||
            segment.padding_length != segment.padded_length - segment.actual_length ||
            segment.frame_digest == std::array<byte, 32U>{} ||
            !zkdl::cpu_pedersen_strict_g1(
                zkdl::cpu_g1_from_native(segment.commitment), error))
            return fail("segment commitment wire order/layout/point is not canonical");
    }
    return true;
}

inline std::array<byte, 32U> segment_commitment_digest(
    const CommitmentWire& commitment) {
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.segment-commitment-vector.v1");
    zkdl::append_string(encoded, commitment.tensor_id);
    zkdl::append_u32(encoded, commitment.visibility);
    zkdl::append_u32(encoded, commitment.domain_size);
    zkdl::append_u32(encoded, commitment.segment_binding_version);
    zkdl::append_u32(encoded, commitment.segment_size);
    zkdl::append_u32(encoded, commitment.segment_count);
    append_digest(encoded, commitment.srs_manifest_digest);
    append_digest(encoded, commitment.basis_digest);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitment.segment_commitments.size()));
    for (const auto& segment : commitment.segment_commitments) {
        zkdl::append_u32(encoded, segment.segment_index);
        zkdl::append_u64(encoded, segment.logical_offset);
        zkdl::append_u32(encoded, segment.actual_length);
        zkdl::append_u32(encoded, segment.padded_length);
        zkdl::append_u32(encoded, segment.padding_length);
        append_digest(encoded, segment.frame_digest);
        zkdl::append_g1(encoded, segment.commitment);
    }
    return zkdl::sha256(encoded);
}

inline std::array<byte, 32U> relation_commitment_vector_digest(
    const RelationStatement& statement,
    const std::vector<CommitmentWire>& commitments) {
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.relation-segment-binding.v1");
    zkdl::append_u32(encoded, statement.segment_binding_version);
    append_digest(encoded, statement.srs_digest);
    append_digest(encoded, statement.registry_digest);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitments.size()));
    for (const auto& commitment : commitments) {
        zkdl::append_string(encoded, commitment.tensor_id);
        zkdl::append_u32(encoded, commitment.visibility);
        zkdl::append_u32(encoded, commitment.domain_size);
        zkdl::append_u32(encoded, commitment.segment_binding_version);
        zkdl::append_u32(encoded, commitment.segment_size);
        zkdl::append_u32(encoded, commitment.segment_count);
        append_digest(encoded, commitment.segment_commitment_digest);
    }
    return zkdl::sha256(encoded);
}

inline bool relation_segment_commitment_valid(const RelationStatement& statement, const CommitmentWire& commitment, std::string* error = nullptr) {
    if (statement.segment_binding_version != RelationStatement::kSegmentBindingVersion || !segment_commitment_wire_shape_valid(commitment, error) || segment_commitment_digest(commitment) != commitment.segment_commitment_digest) {
        if (error && error->empty()) *error = "segment commitment digest mismatch";
        return false;
    }
    if (commitment.srs_manifest_digest != statement.srs_digest) {
        if (error) *error = "segment commitment SRS manifest mismatch";
        return false;
    }
    zkdl::CpuG1Point segment_sum = zkdl::CpuG1Point::zero();
    for (const auto& segment : commitment.segment_commitments) segment_sum = zkdl::cpu_g1_add(segment_sum, zkdl::cpu_g1_from_native(segment.commitment));
    if (!zkdl::cpu_g1_equal(segment_sum, zkdl::cpu_g1_from_native(commitment.commitment))) {
        if (error) *error = "segment commitment aggregate is not the ordered segment sum";
        return false;
    }
    return true;
}

inline bool relation_segment_commitments_valid(
    const RelationStatement& statement,
    const std::vector<CommitmentWire>& commitments,
    std::string* error = nullptr) {
    if (statement.segment_binding_version != RelationStatement::kSegmentBindingVersion) {
        if (error) *error = "relation segment binding version mismatch";
        return false;
    }
    if (statement.segment_commitments_digest == std::array<byte, 32U>{}) {
        if (error) *error = "relation statement lacks segment commitment digest";
        return false;
    }
    for (const auto& commitment : commitments)
        if (!relation_segment_commitment_valid(statement, commitment, error)) return false;
    if (relation_commitment_vector_digest(statement, commitments) !=
        statement.segment_commitments_digest) {
        if (error) *error = "relation segment commitment vector digest mismatch";
        return false;
    }
    return true;
}

template <typename Output>
inline void append_constraint_proof(Output& out, const ConstraintProof& proof) {
    append_constraint(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.opening_tensor_indices.size()));
    for (const auto index : proof.opening_tensor_indices) zkdl::append_u32(out, index);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.opening_commitment_indices.size()));
    for (const std::uint32_t index : proof.opening_commitment_indices) zkdl::append_u32(out, index);
    std::vector<byte> sumcheck;
    zkdl::serialize_zk_sumcheck(proof.zk_sumcheck, sumcheck);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck.size()));
    out.insert(out.end(), sumcheck.begin(), sumcheck.end());
    std::vector<byte> terminal;
    zkdl::serialize_zk_terminal_relation(proof.terminal_relation, terminal);
    zkdl::append_u64(out, static_cast<std::uint64_t>(terminal.size()));
    out.insert(out.end(), terminal.begin(), terminal.end());
}

inline bool read_constraint_proof(
    const std::vector<byte>& in, size_t& offset, ConstraintProof& proof,
    std::uint64_t max_terminal_body_bytes =
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes) {
    std::uint64_t count = 0U, proof_bytes = 0U;
    if (!read_constraint(in, offset, proof.descriptor) ||
        !zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.opening_tensor_indices.resize(static_cast<std::size_t>(count));
    for (auto& index : proof.opening_tensor_indices)
        if (!zkdl::read_u32(in, offset, index)) return false;
    if (!zkdl::read_u64(in, offset, count) || count == 0U || count > kMaxRelationOpenings) return false;
    proof.opening_commitment_indices.resize(static_cast<size_t>(count));
    for (auto& index : proof.opening_commitment_indices)
        if (!zkdl::read_u32(in, offset, index)) return false;
    if (!zkdl::read_u64(in, offset, proof_bytes) || proof_bytes == 0U ||
        proof_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
        proof_bytes > in.size() - offset) return false;
    std::vector<byte> encoded(in.begin() + static_cast<std::ptrdiff_t>(offset),
                              in.begin() + static_cast<std::ptrdiff_t>(offset + proof_bytes));
    offset += static_cast<size_t>(proof_bytes);
    if (!zkdl::deserialize_zk_sumcheck(encoded, proof.zk_sumcheck, nullptr)) return false;
    if (!zkdl::read_u64(in, offset, proof_bytes) || proof_bytes == 0U ||
        proof_bytes > max_terminal_body_bytes ||
        proof_bytes > in.size() - offset) return false;
    encoded.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                   in.begin() + static_cast<std::ptrdiff_t>(offset + proof_bytes));
    offset += static_cast<size_t>(proof_bytes);
    if (!zkdl::deserialize_zk_terminal_relation(
            encoded, proof.terminal_relation, nullptr,
            max_terminal_body_bytes)) return false;
    return true;
}

inline std::vector<byte> PaddingProofRecord::serialize() const {
    if (version != kVersion ||
        group_index == std::numeric_limits<std::uint32_t>::max() ||
        group_key == std::array<byte, 32U>{{}} || domain_size == 0U ||
        (domain_size & (domain_size - 1U)) != 0U ||
        member_tensor_ids.empty() || member_tensor_ids.size() > kMaxMembers ||
        member_tensor_ids.size() != member_commitment_indices.size()) {
        throw std::invalid_argument("invalid Qpad proof record");
    }
    std::set<std::string> member_ids;
    std::set<std::uint32_t> commitment_indices;
    for (std::size_t index = 0U; index < member_tensor_ids.size(); ++index)
        if (member_tensor_ids[index].empty() ||
            !member_ids.insert(member_tensor_ids[index]).second ||
            !commitment_indices.insert(member_commitment_indices[index]).second)
            throw std::invalid_argument("invalid Qpad group member map");
    std::vector<byte> out{'P', 'A', 'D', '2'};
    zkdl::append_u32(out, version);
    zkdl::append_u32(out, group_index);
    append_digest(out, group_key);
    zkdl::append_u32(out, domain_size);
    zkdl::append_u64(out, static_cast<std::uint64_t>(member_tensor_ids.size()));
    for (std::size_t index = 0U; index < member_tensor_ids.size(); ++index) {
        zkdl::append_string(out, member_tensor_ids[index]);
        zkdl::append_u32(out, member_commitment_indices[index]);
    }
    std::vector<byte> relation_bytes;
    append_constraint_proof(relation_bytes, relation);
    zkdl::append_u64(out, static_cast<std::uint64_t>(relation_bytes.size()));
    out.insert(out.end(), relation_bytes.begin(), relation_bytes.end());
    return out;
}

inline bool PaddingProofRecord::deserialize(const std::vector<byte>& bytes,
                                            PaddingProofRecord& record,
                                            std::string* error,
                                            std::uint64_t max_record_body_bytes,
                                            std::uint64_t max_terminal_body_bytes) {
    PaddingProofRecord parsed;
    std::uint32_t version = 0U;
    std::uint64_t nested_size = 0U, member_count = 0U;
    std::size_t offset = 0U;
    if (bytes.size() > max_record_body_bytes || bytes.size() < 4U ||
        bytes[0] != 'P' || bytes[1] != 'A' ||
        bytes[2] != 'D' || bytes[3] != '2') {
        if (error) *error = "invalid Qpad proof framing or authenticated body bound";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, version) || version != PaddingProofRecord::kVersion ||
        !zkdl::read_u32(bytes, offset, parsed.group_index) ||
        !read_digest(bytes, offset, parsed.group_key) ||
        !zkdl::read_u32(bytes, offset, parsed.domain_size) ||
        !zkdl::read_u64(bytes, offset, member_count) || member_count == 0U ||
        member_count > PaddingProofRecord::kMaxMembers ||
        member_count > (bytes.size() - offset) / 12U) {
        if (error) *error = "truncated or invalid Qpad group header";
        return false;
    }
    parsed.member_tensor_ids.resize(static_cast<std::size_t>(member_count));
    parsed.member_commitment_indices.resize(static_cast<std::size_t>(member_count));
    std::set<std::string> member_ids;
    std::set<std::uint32_t> commitment_indices;
    for (std::size_t index = 0U; index < parsed.member_tensor_ids.size(); ++index)
        if (!zkdl::read_string(bytes, offset, parsed.member_tensor_ids[index]) ||
            parsed.member_tensor_ids[index].empty() ||
            !member_ids.insert(parsed.member_tensor_ids[index]).second ||
            !zkdl::read_u32(bytes, offset, parsed.member_commitment_indices[index]) ||
            !commitment_indices.insert(parsed.member_commitment_indices[index]).second) {
            if (error) *error = "invalid Qpad group member map";
            return false;
        }
    if (!zkdl::read_u64(bytes, offset, nested_size) || nested_size == 0U ||
        nested_size > max_record_body_bytes ||
        nested_size != bytes.size() - offset) {
        if (error) *error = "truncated or trailing Qpad proof record";
        return false;
    }
    std::vector<byte> relation_bytes(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
    std::size_t relation_offset = 0U;
    if (!read_constraint_proof(relation_bytes, relation_offset, parsed.relation,
                               max_terminal_body_bytes) ||
        relation_offset != relation_bytes.size() ||
        parsed.group_index == std::numeric_limits<std::uint32_t>::max() ||
        parsed.group_key == std::array<byte, 32U>{{}} ||
        parsed.domain_size == 0U ||
        (parsed.domain_size & (parsed.domain_size - 1U)) != 0U) {
        if (error) *error = "invalid Qpad relation payload";
        return false;
    }
    parsed.version = version;
    record = std::move(parsed);
    return true;
}

template <typename Output>
inline void append_primitive_binding(
    Output& out, const OperationProof::PrimitiveBinding& binding) {
    zkdl::append_u32(out, binding.a_commitment_index);
    zkdl::append_u32(out, binding.b_commitment_index);
    zkdl::append_u32(out, binding.c_commitment_index);
    zkdl::serialize_opening(binding.opening_a, out);
    zkdl::serialize_opening(binding.opening_b, out);
    zkdl::serialize_opening(binding.opening_c, out);
}

inline bool read_primitive_binding(
    const std::vector<byte>& in, size_t& offset,
    OperationProof::PrimitiveBinding& binding) {
    return zkdl::read_u32(in, offset, binding.a_commitment_index) &&
           zkdl::read_u32(in, offset, binding.b_commitment_index) &&
           zkdl::read_u32(in, offset, binding.c_commitment_index) &&
           zkdl::deserialize_opening(in, offset, binding.opening_a) &&
           zkdl::deserialize_opening(in, offset, binding.opening_b) &&
           zkdl::deserialize_opening(in, offset, binding.opening_c);
}

template <typename Output>
inline void append_operation_proof(Output& out, const OperationProof& proof) {
    append_operation(out, proof.descriptor);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.primitive_proofs.size()));
    for (const auto& primitive : proof.primitive_proofs) {
        zkdl::append_u64(out, static_cast<std::uint64_t>(primitive.size()));
        out.insert(out.end(), primitive.begin(), primitive.end());
    }
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.primitive_bindings.size()));
    for (const auto& binding : proof.primitive_bindings)
        append_primitive_binding(out, binding);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.route_proof.size()));
    out.insert(out.end(), proof.route_proof.begin(), proof.route_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.fac_proof.size()));
    out.insert(out.end(), proof.fac_proof.begin(), proof.fac_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.batch_proof.size()));
    out.insert(out.end(), proof.batch_proof.begin(), proof.batch_proof.end());
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.constraints.size()));
    for (const auto& constraint : proof.constraints) append_constraint_proof(out, constraint);
}

inline bool read_operation_proof(
    const std::vector<byte>& in, size_t& offset, OperationProof& proof,
    std::uint64_t max_terminal_body_bytes =
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes) {
    std::uint64_t count = 0U, bytes = 0U;
    if (!read_operation_strict(in, offset, proof.descriptor) ||
        !zkdl::read_u64(in, offset, count) || count > 8U) return false;
    proof.primitive_proofs.resize(static_cast<size_t>(count));
    for (auto& primitive : proof.primitive_proofs) {
        if (!zkdl::read_u64(in, offset, bytes) || bytes == 0U ||
            bytes > HiddenMatMulProof::kMaxProofBytes || bytes > in.size() - offset)
            return false;
        primitive.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                         in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
        offset += static_cast<size_t>(bytes);
    }
    if (!zkdl::read_u64(in, offset, count) || count > 8U) return false;
    proof.primitive_bindings.resize(static_cast<size_t>(count));
    for (auto& binding : proof.primitive_bindings)
        if (!read_primitive_binding(in, offset, binding)) return false;
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.route_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                             in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.route_proof.empty()) {
        SparseRouteProof route;
        if (!SparseRouteProof::deserialize(proof.route_proof, route)) return false;
    }
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.fac_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                           in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.fac_proof.empty()) {
        FacLine8Proof fac;
        if (!FacLine8Proof::deserialize(proof.fac_proof, fac)) return false;
    }
    if (!zkdl::read_u64(in, offset, bytes) || bytes > in.size() - offset) return false;
    proof.batch_proof.assign(in.begin() + static_cast<std::ptrdiff_t>(offset),
                             in.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
    offset += static_cast<size_t>(bytes);
    if (!proof.batch_proof.empty()) {
        ConstraintProof batch;
        if (!deserialize_constraint_payload(proof.batch_proof, batch, nullptr,
                                            max_terminal_body_bytes)) return false;
    }
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationConstraints ||
        (count == 0U && proof.primitive_proofs.empty() && proof.route_proof.empty() &&
         proof.fac_proof.empty() && proof.batch_proof.empty() &&
         proof.descriptor.kind != RelationKind::PublicOutput)) return false;
    proof.constraints.resize(static_cast<size_t>(count));
    for (auto& constraint : proof.constraints)
        if (!read_constraint_proof(in, offset, constraint,
                                   max_terminal_body_bytes)) return false;
    return true;
}

template <typename Output>
inline void append_native_fr_vector(Output& out,
                                    const std::vector<Fr_t>& values) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(values.size()));
    for (const auto& value : values) zkdl::append_fr(out, value);
}

inline void append_native_fr_vector(zkdl::WireSizeCounter& out,
                                    const std::vector<Fr_t>& values) {
    const std::uint64_t count = static_cast<std::uint64_t>(values.size());
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire))
        throw std::overflow_error("native field vector wire size overflows uint64");
    zkdl::append_u64(out, count);
    out.add(count * sizeof(FrWire));
}

inline bool read_native_fr_vector(const std::vector<byte>& in, size_t& offset,
                                  std::vector<Fr_t>& values) {
    std::uint64_t count = 0U;
    if (!zkdl::read_u64(in, offset, count) || count > kMaxRelationPointBits) return false;
    values.resize(static_cast<size_t>(count));
    for (auto& value : values)
        if (!zkdl::read_fr(in, offset, value)) return false;
    return true;
}

template <typename Output>
inline void serialize_hidden_matmul_opening(const HiddenMatMulOpening& opening, Output& out) {
    if (opening.segments.empty() || opening.segments.size() > kMaxRelationSegments) throw std::invalid_argument("hidden segmented opening count exceeds cap");
    std::uint32_t segment_count = 0U;
    if (!checked_relation_u32(opening.segments.size(), segment_count,
                              "hidden segmented opening count"))
        throw std::invalid_argument("hidden segmented opening count exceeds uint32");
    zkdl::append_g1(out, opening.tau);
    zkdl::append_u32(out, segment_count);
    for (const auto& segment : opening.segments) zkdl::serialize_hyrax_opening_v3(segment, out);
}

inline bool deserialize_hidden_matmul_opening(const std::vector<byte>& bytes, std::size_t& offset, HiddenMatMulOpening& opening) {
    std::uint32_t count = 0U;
    if (!zkdl::read_g1(bytes, offset, opening.tau) || !zkdl::read_u32(bytes, offset, count) || count == 0U || count > kMaxRelationSegments) return false;
    opening.segments.resize(count);
    for (auto& segment : opening.segments) if (!zkdl::deserialize_hyrax_opening_v3(bytes, offset, segment)) return false;
    return true;
}

inline bool hidden_matmul_opening_shape_valid(const HiddenMatMulOpening& opening, std::uint64_t domain) {
    if (opening.segments.empty() || opening.segments.size() > kMaxRelationSegments || domain == 0U || domain > (1U << kMaxRelationPointBits)) return false;
    const std::uint32_t size = opening.segments.front().generator_size;
    if (size == 0U || (size & (size - 1U)) != 0U || size > (1U << 18U) || domain != static_cast<std::uint64_t>(size) * opening.segments.size()) return false;
    for (std::size_t index = 0U; index < opening.segments.size(); ++index) { const auto& segment = opening.segments[index]; if (segment.generator_size != size || segment.logical_domain != domain || segment.segment_index != index || segment.logical_offset != index * static_cast<std::uint64_t>(size) || segment.actual_length != size || segment.padded_length != size) return false; }
    return true;
}

inline std::vector<byte> HiddenMatMulProof::serialize() const {
    std::string shape_error;
    if (version != kVersion || !shape.valid(&shape_error) ||
        primitive_digest == std::array<byte, 32U>{{}}) {
        throw std::invalid_argument(shape_error.empty()
                                        ? "invalid hidden matrix proof"
                                        : shape_error);
    }
    std::vector<byte> out{'Z', 'M', 'H', '2'};
    zkdl::append_u32(out, version);
    zkdl::matmul_append_shape(out, shape);
    zkdl::append_u32(out, a_commitment_index);
    zkdl::append_u32(out, b_commitment_index);
    zkdl::append_u32(out, c_commitment_index);
    append_digest(out, primitive_digest);
    append_native_fr_vector(out, row_point);
    append_native_fr_vector(out, column_point);
    append_native_fr_vector(out, batch_point);
    std::vector<byte> nested;
    zkdl::serialize_zk_sumcheck(sumcheck, nested);
    zkdl::append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
    serialize_hidden_matmul_opening(opening_a, out);
    serialize_hidden_matmul_opening(opening_b, out);
    serialize_hidden_matmul_opening(opening_c, out);
    nested.clear();
    zkdl::serialize_pedersen_product(terminal_product, nested);
    zkdl::append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
    nested.clear();
    zkdl::serialize_pedersen_scalar_equality(terminal_binding, nested);
    zkdl::append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
    if (out.size() > kMaxProofBytes)
        throw std::invalid_argument("hidden matrix proof exceeds bounded wire size");
    return out;
}

inline bool HiddenMatMulProof::deserialize(const std::vector<byte>& bytes,
                                           HiddenMatMulProof& proof,
                                           std::string* error) {
    HiddenMatMulProof parsed;
    std::uint32_t version = 0U;
    std::uint64_t nested_size = 0U;
    std::size_t offset = 0U;
    if (bytes.size() > kMaxProofBytes) {
        if (error) *error = "hidden matrix proof exceeds bounded wire size";
        return false;
    }
    if (bytes.size() < 4U || bytes[0] != 'Z' || bytes[1] != 'M' ||
        bytes[2] != 'H' || bytes[3] != '2') {
        if (error) *error = "invalid hidden matrix proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, version) || version != kVersion ||
        !zkdl::matmul_read_shape(bytes, offset, parsed.shape) ||
        !zkdl::read_u32(bytes, offset, parsed.a_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.b_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.c_commitment_index) ||
        !read_digest(bytes, offset, parsed.primitive_digest) ||
        !read_native_fr_vector(bytes, offset, parsed.row_point) ||
        !read_native_fr_vector(bytes, offset, parsed.column_point) ||
        !read_native_fr_vector(bytes, offset, parsed.batch_point) ||
        !zkdl::read_u64(bytes, offset, nested_size) || nested_size == 0U ||
        nested_size > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
        nested_size > bytes.size() - offset) {
        if (error) *error = "truncated hidden matrix proof header";
        return false;
    }
    std::vector<byte> nested(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                             bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_size));
    offset += static_cast<std::size_t>(nested_size);
    if (!zkdl::deserialize_zk_sumcheck(nested, parsed.sumcheck, error) ||
        !deserialize_hidden_matmul_opening(bytes, offset, parsed.opening_a) ||
        !deserialize_hidden_matmul_opening(bytes, offset, parsed.opening_b) ||
        !deserialize_hidden_matmul_opening(bytes, offset, parsed.opening_c) ||
        !zkdl::read_u64(bytes, offset, nested_size) || nested_size == 0U ||
        nested_size > 1U << 22U || nested_size > bytes.size() - offset) {
        if (error) *error = "truncated hidden matrix proof body";
        return false;
    }
    nested.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                  bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_size));
    offset += static_cast<std::size_t>(nested_size);
    if (!zkdl::deserialize_pedersen_product(nested, parsed.terminal_product, error) ||
        !zkdl::read_u64(bytes, offset, nested_size) || nested_size == 0U ||
        nested_size > 4096U || nested_size != bytes.size() - offset) {
        if (error) *error = "truncated hidden matrix terminal product";
        return false;
    }
    nested.assign(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.end());
    if (!zkdl::deserialize_pedersen_scalar_equality(nested,
                                                     parsed.terminal_binding, error))
        return false;
    parsed.version = version;
    std::string shape_error;
    if (!parsed.shape.valid(&shape_error) ||
        parsed.row_point.size() != zkdl::MatMulShape::log2_ceil(parsed.shape.m) ||
        parsed.column_point.size() != zkdl::MatMulShape::log2_ceil(parsed.shape.k) ||
        parsed.batch_point.size() != zkdl::MatMulShape::log2_ceil(parsed.shape.num) ||
        parsed.sumcheck.degree != 2U ||
        parsed.sumcheck.domain_size != parsed.shape.n ||
        !hidden_matmul_opening_shape_valid(parsed.opening_a, parsed.shape.a_size()) ||
        !hidden_matmul_opening_shape_valid(parsed.opening_b, parsed.shape.b_size()) ||
        !hidden_matmul_opening_shape_valid(parsed.opening_c, parsed.shape.c_size())) {
        if (error) *error = shape_error.empty()
                                  ? "hidden matrix proof shape is not canonical"
                                  : shape_error;
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline std::vector<byte> FacLine8Proof::serialize() const {
    std::vector<byte> out;
    out.reserve(1024U);
    out.push_back('F'); out.push_back('A'); out.push_back('C'); out.push_back('3');
    zkdl::append_u32(out, version);
    zkdl::append_u64(out, static_cast<std::uint64_t>(selector_proofs.size()));
    for (const auto& route : selector_proofs) {
        const std::vector<byte> encoded = route.serialize();
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    std::vector<byte> aggregate;
    zkdl::serialize_pedersen_scalar_equality(aggregate_binding, aggregate);
    zkdl::append_u64(out, static_cast<std::uint64_t>(aggregate.size()));
    out.insert(out.end(), aggregate.begin(), aggregate.end());
    return out;
}

inline bool FacLine8Proof::deserialize(const std::vector<byte>& bytes,
                                       FacLine8Proof& proof,
                                       std::string* error) {
    FacLine8Proof parsed;
    std::uint64_t count = 0U, nested_bytes = 0U;
    size_t offset = 0U;
    constexpr std::uint64_t kMaxWireItems = kMaxRelationOpenings;
    if (bytes.size() < 4U || bytes[0] != 'F' || bytes[1] != 'A' ||
        bytes[2] != 'C' || bytes[3] != '3') {
        if (error) *error = "invalid FAC4DNN Line8 proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, parsed.version) || parsed.version != kVersion ||
        !zkdl::read_u64(bytes, offset, count) || count == 0U || count > kMaxWireItems) {
        if (error) *error = "invalid FAC4DNN Line8 descriptor/header";
        return false;
    }
    parsed.selector_proofs.resize(static_cast<std::size_t>(count));
    for (auto& route : parsed.selector_proofs) {
        if (!zkdl::read_u64(bytes, offset, nested_bytes) || nested_bytes == 0U ||
            nested_bytes > bytes.size() - offset ||
            nested_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes) {
            if (error) *error = "invalid FAC4DNN selector route length";
            return false;
        }
        std::vector<byte> encoded(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_bytes));
        offset += static_cast<std::size_t>(nested_bytes);
        if (!SparseRouteProof::deserialize(encoded, route, error)) return false;
    }
    if (!zkdl::read_u64(bytes, offset, nested_bytes) || nested_bytes == 0U ||
        nested_bytes > bytes.size() - offset ||
        nested_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes) {
        if (error) *error = "invalid FAC4DNN aggregate binding length";
        return false;
    }
    std::vector<byte> aggregate(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + nested_bytes));
    offset += static_cast<std::size_t>(nested_bytes);
    if (!zkdl::deserialize_pedersen_scalar_equality(
            aggregate, parsed.aggregate_binding, error)) return false;
    if (offset != bytes.size()) {
        if (error) *error = "trailing FAC4DNN Line8 proof bytes";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline std::vector<byte> SparseRouteProof::serialize() const {
    std::vector<byte> out;
    out.reserve(1024U);
    out.push_back('S'); out.push_back('R'); out.push_back('P'); out.push_back('1');
    zkdl::append_u32(out, version);
    zkdl::append_u32(out, source_commitment_index);
    zkdl::append_u32(out, output_commitment_index);
    zkdl::append_u32(out, edge_domain);
    zkdl::append_u32(out, node_domain);
    zkdl::append_u32(out, feature_domain);
    zkdl::append_u32(out, source_feature_domain);
    append_native_fr_vector(out, edge_point);
    append_native_fr_vector(out, feature_point);
    append_native_fr_vector(out, source_feature_point);
    append_native_fr_vector(out, node_point);
    std::vector<byte> sumcheck_bytes;
    zkdl::serialize_zk_sumcheck(sumcheck, sumcheck_bytes);
    zkdl::append_u64(out, static_cast<std::uint64_t>(sumcheck_bytes.size()));
    out.insert(out.end(), sumcheck_bytes.begin(), sumcheck_bytes.end());
    zkdl::serialize_hyrax_opening_v3(source_opening, out);
    zkdl::serialize_hyrax_opening_v3(output_opening, out);
    std::vector<byte> terminal_bytes;
    zkdl::serialize_pedersen_scalar_equality(terminal_binding, terminal_bytes);
    zkdl::append_u64(out, static_cast<std::uint64_t>(terminal_bytes.size()));
    out.insert(out.end(), terminal_bytes.begin(), terminal_bytes.end());
    return out;
}

inline bool SparseRouteProof::deserialize(const std::vector<byte>& bytes,
                                          SparseRouteProof& proof,
                                          std::string* error) {
    SparseRouteProof parsed;
    size_t offset = 0U;
    std::uint64_t sumcheck_bytes = 0U;
    if (bytes.size() < 4U || bytes[0] != 'S' || bytes[1] != 'R' ||
        bytes[2] != 'P' || bytes[3] != '1') {
        if (error) *error = "invalid sparse route proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, parsed.version) || parsed.version != kVersion ||
        !zkdl::read_u32(bytes, offset, parsed.source_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.output_commitment_index) ||
        !zkdl::read_u32(bytes, offset, parsed.edge_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.node_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.feature_domain) ||
        !zkdl::read_u32(bytes, offset, parsed.source_feature_domain) ||
        !read_native_fr_vector(bytes, offset, parsed.edge_point) ||
        !read_native_fr_vector(bytes, offset, parsed.feature_point) ||
        !read_native_fr_vector(bytes, offset, parsed.source_feature_point) ||
        !read_native_fr_vector(bytes, offset, parsed.node_point) ||
        !zkdl::read_u64(bytes, offset, sumcheck_bytes) ||
        sumcheck_bytes == 0U ||
        sumcheck_bytes > zkdl::ZkSumcheckProof::kMaxNestedProofBytes ||
        sumcheck_bytes > bytes.size() - offset) {
        if (error) *error = "truncated sparse route proof header";
        return false;
    }
    std::vector<byte> sumcheck_bytes_vector(
        bytes.begin() + static_cast<std::ptrdiff_t>(offset),
        bytes.begin() + static_cast<std::ptrdiff_t>(offset + sumcheck_bytes));
    offset += static_cast<size_t>(sumcheck_bytes);
    std::uint64_t terminal_bytes = 0U;
    if (!zkdl::deserialize_zk_sumcheck(sumcheck_bytes_vector, parsed.sumcheck, error) ||
        !zkdl::deserialize_hyrax_opening_v3(bytes, offset, parsed.source_opening) ||
        !zkdl::deserialize_hyrax_opening_v3(bytes, offset, parsed.output_opening) ||
        !zkdl::read_u64(bytes, offset, terminal_bytes) || terminal_bytes == 0U ||
        terminal_bytes > 4096U || terminal_bytes != bytes.size() - offset) {
        if (error) *error = "truncated or trailing sparse route proof";
        return false;
    }
    const std::vector<byte> terminal(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                                     bytes.end());
    if (!zkdl::deserialize_pedersen_scalar_equality(terminal, parsed.terminal_binding, error))
        return false;
    const auto power_of_two = [](std::uint32_t value) {
        return value != 0U && (value & (value - 1U)) == 0U;
    };
    if (!power_of_two(parsed.edge_domain) || !power_of_two(parsed.node_domain) ||
        !power_of_two(parsed.feature_domain) || !power_of_two(parsed.source_feature_domain) ||
        parsed.edge_point.size() >= 32U || parsed.feature_point.size() >= 32U ||
        parsed.source_feature_point.size() >= 32U || parsed.node_point.size() >= 32U ||
        parsed.edge_point.size() != log2_exact(parsed.edge_domain) ||
        parsed.node_point.size() != log2_exact(parsed.node_domain) ||
        parsed.feature_point.size() != log2_exact(parsed.feature_domain) ||
        parsed.source_feature_point.size() != log2_exact(parsed.source_feature_domain) ||
        parsed.source_feature_point.size() + parsed.node_point.size() > kMaxRelationPointBits ||
        parsed.feature_point.size() + parsed.edge_point.size() > kMaxRelationPointBits ||
        parsed.sumcheck.degree != 2U ||
        parsed.sumcheck.domain_size != std::max<std::uint64_t>(
            2U, static_cast<std::uint64_t>(parsed.node_domain) * parsed.source_feature_domain) ||
        parsed.source_opening.logical_domain !=
            static_cast<std::uint64_t>(parsed.node_domain) * parsed.source_feature_domain ||
        parsed.output_opening.logical_domain !=
            static_cast<std::uint64_t>(parsed.edge_domain) * parsed.feature_domain ||
        parsed.source_opening.generator_size == 0U ||
        parsed.output_opening.generator_size == 0U ||
        parsed.source_opening.generator_size > parsed.source_opening.logical_domain ||
        parsed.output_opening.generator_size > parsed.output_opening.logical_domain ||
        parsed.source_opening.logical_domain % parsed.source_opening.generator_size != 0U ||
        parsed.output_opening.logical_domain % parsed.output_opening.generator_size != 0U) {
        if (error) *error = "sparse route proof shape is not canonical";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

template <typename Output>
inline void append_proof_wire(Output& out, const Proof& proof) {
    if (proof.version != Proof::kVersion || !proof.hiding_commitment_records.empty())
        throw std::invalid_argument("obsolete version or legacy hiding commitment records are not part of formal GATR wire");
    out.push_back('G'); out.push_back('A'); out.push_back('T'); out.push_back('R');
    zkdl::append_u32(out, proof.version);
    append_statement(out, proof.statement);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.commitments.size()));
    for (const auto& commitment : proof.commitments) append_commitment(out, commitment);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.padding_records.size()));
    for (const auto& padding : proof.padding_records) {
        const std::vector<byte> encoded = padding.serialize();
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.hiding_commitment_records.size()));
    for (const auto& record : proof.hiding_commitment_records) append_hiding_record(out, record);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.operations.size()));
    for (const auto& operation : proof.operations) append_operation_proof(out, operation);
    zkdl::append_u64(out, static_cast<std::uint64_t>(proof.sorted_lookup_proofs.size()));
    for (const auto& lookup : proof.sorted_lookup_proofs) {
        const std::vector<byte> encoded = lookup.serialize();
        zkdl::append_u64(out, static_cast<std::uint64_t>(encoded.size()));
        out.insert(out.end(), encoded.begin(), encoded.end());
    }
    zkdl::append_u32(out, proof.public_output_commitment);
    serialize_hidden_matmul_opening(proof.public_output_opening, out);
    std::vector<byte> public_binding;
    zkdl::serialize_pedersen_scalar_value(proof.public_output_binding, public_binding);
    zkdl::append_u64(out, public_binding.size());
    out.insert(out.end(), public_binding.begin(), public_binding.end());
    append_digest(out, proof.transcript_digest);
}

inline std::uint64_t Proof::serialized_size() const {
    zkdl::WireSizeCounter count;
    append_proof_wire(count, *this);
    return count.size();
}

inline std::vector<byte> Proof::serialize() const {
    const std::uint64_t exact_size = serialized_size();
    if (exact_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::length_error("proof wire size exceeds addressable vector capacity");
    std::vector<byte> out;
    if (exact_size > static_cast<std::uint64_t>(out.max_size()))
        throw std::length_error("proof wire size exceeds vector max_size");
    out.reserve(static_cast<std::size_t>(exact_size));
    zkdl::BoundedWireWriter writer(out, exact_size);
    append_proof_wire(writer, *this);
    if (writer.size() != exact_size)
        throw std::length_error("proof wire serializer disagrees with its counted size");
    return out;
}

inline bool proof_deserialize_impl(
    const std::vector<byte>& bytes, Proof& proof,
    const std::vector<std::vector<std::string>>* expected_qpad_groups,
    std::uint64_t max_qpad_record_bytes,
    std::uint64_t max_terminal_body_bytes,
    std::uint64_t max_sorted_lookup_relation_body_bytes,
    std::string* error) {
    const std::uint64_t max_items = 1U << 20;
    if (expected_qpad_groups != nullptr) {
        if (expected_qpad_groups->size() > max_items ||
            max_terminal_body_bytes == 0U ||
            (!expected_qpad_groups->empty() && max_qpad_record_bytes == 0U)) {
            if (error) *error = "authenticated relation context has invalid item/body bounds";
            return false;
        }
        std::set<std::string> expected_ids;
        for (const auto& group : *expected_qpad_groups) {
            if (group.empty() || group.size() > PaddingProofRecord::kMaxMembers) {
                if (error) *error = "authenticated Qpad context has an empty or oversized group";
                return false;
            }
            for (const auto& id : group)
                if (id.empty() || !expected_ids.insert(id).second) {
                    if (error) *error = "authenticated Qpad context has empty or duplicate tensor ids";
                    return false;
                }
        }
    }
    size_t offset = 0U;
    Proof parsed;
    std::uint32_t version = 0U;
    std::uint64_t count = 0U;
    if (bytes.size() < 4U || bytes[0] != 'G' || bytes[1] != 'A' ||
        bytes[2] != 'T' || bytes[3] != 'R') {
        if (error) *error = "invalid GAT relation proof magic";
        return false;
    }
    offset = 4U;
    if (!zkdl::read_u32(bytes, offset, version) || version != Proof::kVersion ||
        !read_statement(bytes, offset, parsed.statement) ||
        !zkdl::read_u64(bytes, offset, count) || count == 0U || count > max_items) {
        if (error) *error = "invalid GAT relation proof header";
        return false;
    }
    parsed.version = version;
    parsed.commitments.resize(static_cast<size_t>(count));
    std::set<std::string> commitment_ids;
    for (auto& commitment : parsed.commitments)
        if (!read_commitment(bytes, offset, commitment)) {
            if (error) *error = "truncated GAT commitment section";
            return false;
        } else if (commitment.tensor_id.empty() || !commitment_ids.insert(commitment.tensor_id).second) {
            if (error) *error = "duplicate/empty GAT commitment id";
            return false;
        }
    const std::uint64_t qpad_count_limit = expected_qpad_groups == nullptr
        ? kMaxRelationOpenings
        : static_cast<std::uint64_t>(expected_qpad_groups->size());
    if (!zkdl::read_u64(bytes, offset, count) || count > qpad_count_limit ||
        (expected_qpad_groups != nullptr && count != expected_qpad_groups->size())) {
        if (error) *error = expected_qpad_groups == nullptr
            ? "invalid Qpad proof record section"
            : "Qpad proof group count does not match the authenticated tensor plan";
        return false;
    }
    const std::uint64_t nested_qpad_cap =
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes;
    const std::uint64_t qpad_body_limit = expected_qpad_groups == nullptr
        ? nested_qpad_cap
        : max_qpad_record_bytes;
    parsed.padding_records.reserve(static_cast<std::size_t>(count));
    for (std::size_t index = 0U; index < static_cast<std::size_t>(count); ++index) {
        std::uint64_t padding_bytes = 0U;
        if (!zkdl::read_u64(bytes, offset, padding_bytes) || padding_bytes == 0U ||
            padding_bytes > qpad_body_limit ||
            padding_bytes > bytes.size() - offset) {
            if (error) *error = expected_qpad_groups == nullptr
                ? "truncated Qpad proof record"
                : "Qpad record body exceeds the authenticated typed plan bound";
            return false;
        }
        const std::vector<byte> encoded(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + padding_bytes));
        offset += static_cast<std::size_t>(padding_bytes);
        PaddingProofRecord padding;
        if (!PaddingProofRecord::deserialize(encoded, padding, error,
                                             qpad_body_limit,
                                             max_terminal_body_bytes)) return false;
        if (expected_qpad_groups != nullptr &&
            padding.member_tensor_ids != (*expected_qpad_groups)[index]) {
            if (error) *error = "Qpad group member order differs from the authenticated tensor plan";
            return false;
        }
        if (expected_qpad_groups != nullptr)
            for (std::size_t member = 0U;
                 member < padding.member_tensor_ids.size(); ++member) {
                const auto commitment = std::find_if(
                    parsed.commitments.begin(), parsed.commitments.end(),
                    [&](const CommitmentWire& wire) {
                        return wire.tensor_id == padding.member_tensor_ids[member];
                    });
                if (commitment == parsed.commitments.end() ||
                    static_cast<std::size_t>(std::distance(
                        parsed.commitments.begin(), commitment)) !=
                        padding.member_commitment_indices[member]) {
                    if (error) *error = "Qpad member commitment index differs from canonical commitment order";
                    return false;
                }
            }
        parsed.padding_records.push_back(std::move(padding));
    }
    if (!zkdl::read_u64(bytes, offset, count) || count != 0U) {
        if (error) *error = "invalid GAT hiding commitment section";
        return false;
    }
    if (!zkdl::read_u64(bytes, offset, count) || count == 0U || count > max_items) {
        if (error) *error = "GAT relation proof has no operation section";
        return false;
    }
    parsed.operations.resize(static_cast<size_t>(count));
    std::set<std::string> operation_ids;
    for (auto& operation : parsed.operations)
        if (!read_operation_proof(bytes, offset, operation,
                                  max_terminal_body_bytes)) {
            if (error) *error = "truncated GAT operation section";
            return false;
        } else if (!operation_ids.insert(operation.descriptor.id).second) {
            if (error) *error = "duplicate GAT operation id";
            return false;
        }
    if (!zkdl::read_u64(bytes, offset, count) || count > (1U << 16U) ||
        (expected_qpad_ids != nullptr && count != 0U &&
         max_sorted_lookup_relation_body_bytes == 0U)) {
        if (error) *error = "invalid authenticated sorted-lookup proof section";
        return false;
    }
    parsed.sorted_lookup_proofs.resize(static_cast<std::size_t>(count));
    for (auto& lookup : parsed.sorted_lookup_proofs) {
        std::uint64_t lookup_bytes = 0U;
        if (!zkdl::read_u64(bytes, offset, lookup_bytes) || lookup_bytes == 0U ||
            lookup_bytes > zkdl::AuthenticatedSortedLookupProof::kMaxProofBytes ||
            lookup_bytes > bytes.size() - offset) {
            if (error) *error = "truncated authenticated sorted-lookup proof";
            return false;
        }
        const std::vector<byte> encoded(
            bytes.begin() + static_cast<std::ptrdiff_t>(offset),
            bytes.begin() + static_cast<std::ptrdiff_t>(offset + lookup_bytes));
        offset += static_cast<std::size_t>(lookup_bytes);
        if (!zkdl::AuthenticatedSortedLookupProof::deserialize(
                encoded, lookup, error, max_terminal_body_bytes,
                expected_qpad_ids == nullptr
                    ? zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes
                    : max_sorted_lookup_relation_body_bytes)) return false;
    }
    std::uint64_t public_binding_size = 0U;
    if (!zkdl::read_u32(bytes, offset, parsed.public_output_commitment) || !deserialize_hidden_matmul_opening(bytes, offset, parsed.public_output_opening) || !zkdl::read_u64(bytes, offset, public_binding_size) || public_binding_size == 0U || public_binding_size > 4096U || public_binding_size > bytes.size() - offset) { if (error) *error = "truncated public output segmented opening"; return false; }
    const std::vector<byte> public_binding(bytes.begin() + static_cast<std::ptrdiff_t>(offset), bytes.begin() + static_cast<std::ptrdiff_t>(offset + public_binding_size));
    offset += public_binding_size;
    if (!zkdl::deserialize_pedersen_scalar_value(public_binding, parsed.public_output_binding, error) || !read_digest(bytes, offset, parsed.transcript_digest) || offset != bytes.size()) { if (error) *error = "truncated or trailing GAT relation proof bytes"; return false; }
    if (parsed.public_output_commitment >= parsed.commitments.size() ||
        parsed.commitments[parsed.public_output_commitment].tensor_id != "public_Y" || !hidden_matmul_opening_shape_valid(parsed.public_output_opening, parsed.commitments[parsed.public_output_commitment].domain_size)) {
        if (error) *error = "GAT proof public output commitment is missing or not public_Y";
        return false;
    }
    proof = std::move(parsed);
    return true;
}

inline bool Proof::deserialize(const std::vector<byte>& bytes, Proof& proof,
                               std::string* error) {
    return proof_deserialize_impl(bytes, proof, nullptr,
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes,
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes,
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes, error);
}

inline bool Proof::deserialize(
    const std::vector<byte>& bytes, Proof& proof,
    const std::vector<std::vector<std::string>>& expected_qpad_groups,
    std::uint64_t max_qpad_record_bytes,
    std::uint64_t max_terminal_body_bytes,
    std::uint64_t max_sorted_lookup_relation_body_bytes,
    std::string* error) {
    return proof_deserialize_impl(bytes, proof, &expected_qpad_groups,
                                  max_qpad_record_bytes,
                                  max_terminal_body_bytes,
                                  max_sorted_lookup_relation_body_bytes, error);
}

inline void absorb_descriptor(zkdl::Transcript& transcript,
                              const OperationDescriptor& descriptor) {
    std::vector<byte> encoded;
    append_operation(encoded, descriptor);
    transcript.absorb("gat.relation.operation", encoded);
}

inline void absorb_statement(zkdl::Transcript& transcript,
                             const RelationStatement& statement,
                             const std::vector<CommitmentWire>& commitments) {
    std::vector<byte> encoded;
    append_statement(encoded, statement);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitments.size()));
    for (const auto& commitment : commitments) append_commitment(encoded, commitment);
    transcript.absorb("gat.relation.statement", encoded);
}

/*
 * Hidden generic Sumcheck round engine.  This deliberately lives beside the
 * relation descriptor rather than in a second proof implementation: all
 * commitments, Sigma equations and Fiat--Shamir challenges are delegated to
 * proof.cuh/cpu-backend.cuh's authenticated Pedersen primitives.  The engine
 * is consumed by ConstraintProof together with the terminal DAG/opening
 * relation; legacy relation families below remain fail-closed until their
 * value-revealing wires are migrated to this native hidden interface.
 */
struct ZkSumcheckProverResult {
    zkdl::ZkSumcheckProof proof;
    std::vector<Fr_t> challenges;
    std::vector<FrWire> folded_terminal_values;
    FrWire final_q_blinder{};
};

struct ZkSumcheckVerifierResult {
    std::vector<Fr_t> challenges;
    G1Jacobian_t final_claim_commitment{};
};

inline bool zksc_digest_nonzero(const std::array<byte, 32U>& digest) {
    return digest != std::array<byte, 32U>{{}};
}

inline bool zksc_statement_resources_valid(const RelationStatement& statement,
                                           std::string* error = nullptr) {
    if (!zksc_digest_nonzero(statement.registry_digest) ||
        !zksc_digest_nonzero(statement.srs_digest)) {
        if (error) *error = "ZkSumcheck requires authenticated registry/SRS digests";
        return false;
    }
    return true;
}

inline std::array<byte, 32U> zksc_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round_index, std::uint32_t commitment_index,
    std::uint32_t related_index) {
    std::vector<byte> encoded;
    append_constraint(encoded, descriptor);
    zkdl::append_string(encoded, operation_context);
    zkdl::append_string(encoded, role);
    zkdl::append_u32(encoded, round_index);
    zkdl::append_u32(encoded, commitment_index);
    zkdl::append_u32(encoded, related_index);
    append_digest(encoded, statement.statement_digest);
    append_digest(encoded, statement.model_digest);
    append_digest(encoded, statement.table_digest);
    append_digest(encoded, statement.ledger_digest);
    append_digest(encoded, statement.registry_digest);
    append_digest(encoded, statement.srs_digest);
    return zkdl::sha256(encoded);
}

inline zkdl::PedersenProofContext zksc_make_context(
    std::uint32_t operation, std::uint32_t degree, std::uint32_t basis_size,
    std::uint32_t commitment_index, std::uint32_t related_index,
    const std::array<byte, 32U>& basis_digest,
    const std::array<byte, 32U>& descriptor_digest,
    const RelationStatement& statement) {
    zkdl::PedersenProofContext context;
    context.commitment_index = commitment_index;
    context.operation = operation;
    context.commitment_a_index = commitment_index;
    context.commitment_b_index = related_index;
    context.commitment_c_index = zkdl::PedersenProofContext::kUnsetIndex;
    context.evaluation_commitment_index = zkdl::PedersenProofContext::kUnsetIndex;
    if (operation == zkdl::PedersenProofContext::kOperationVectorLinear) {
        context.evaluation_commitment_index = related_index;
        context.commitment_b_index = zkdl::PedersenProofContext::kUnsetIndex;
    }
    context.degree = degree;
    context.basis_size = basis_size;
    context.basis_digest = basis_digest;
    context.descriptor_digest = descriptor_digest;
    context.registry_digest = statement.registry_digest;
    context.srs_digest = statement.srs_digest;
    return context;
}

inline std::uint32_t zksc_round_commitment_index(std::uint32_t round) {
    constexpr std::uint32_t kStride = 4U;
    if (round > (std::numeric_limits<std::uint32_t>::max() / kStride) - 2U)
        throw std::invalid_argument("ZkSumcheck round commitment index overflow");
    return round * kStride;
}

inline std::uint32_t zksc_endpoint_index(std::uint32_t round) {
    return zksc_round_commitment_index(round) + 1U;
}

inline std::uint32_t zksc_evaluation_index(std::uint32_t round) {
    return zksc_round_commitment_index(round) + 2U;
}

inline std::array<byte, 32U> zksc_proof_descriptor_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context) {
    return zksc_role_digest(descriptor, statement, operation_context,
                            "initial-public-claim", 0U, zksc_endpoint_index(0U),
                            zkdl::PedersenProofContext::kUnsetIndex);
}

inline std::array<byte, 32U> zksc_vector_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t related_index) {
    return zksc_role_digest(descriptor, statement, operation_context, role, round,
                            zksc_round_commitment_index(round), related_index);
}

inline std::array<byte, 32U> zksc_scalar_role_digest(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t commitment_index,
    std::uint32_t related_index) {
    return zksc_role_digest(descriptor, statement, operation_context, role, round,
                            commitment_index, related_index);
}

inline FrWire zksc_eval_coefficients(const std::vector<FrWire>& coefficients,
                                     const FrWire& point) {
    if (coefficients.empty()) throw std::invalid_argument("empty ZkSumcheck coefficient vector");
    FrWire result{};
    for (std::size_t index = coefficients.size(); index-- > 0U;) {
        result = zkdl::fr_add(zkdl::fr_mul(result, point), coefficients[index]);
    }
    return result;
}

inline bool zksc_interpolate_coefficients(const std::vector<FrWire>& evaluations,
                                          std::vector<FrWire>& coefficients,
                                          std::string* error = nullptr) {
    if (evaluations.empty() || evaluations.size() > 9U) {
        if (error) *error = "ZkSumcheck interpolation degree is out of bounds";
        return false;
    }
    coefficients.assign(evaluations.size(), FrWire{});
    try {
        for (std::size_t i = 0U; i < evaluations.size(); ++i) {
            std::vector<FrWire> basis(1U, zkdl::fr_from_u32(1U));
            FrWire denominator = zkdl::fr_from_u32(1U);
            const FrWire xi = zkdl::fr_from_u32(static_cast<std::uint32_t>(i));
            for (std::size_t j = 0U; j < evaluations.size(); ++j) {
                if (i == j) continue;
                const FrWire xj = zkdl::fr_from_u32(static_cast<std::uint32_t>(j));
                std::vector<FrWire> next(basis.size() + 1U, FrWire{});
                const FrWire negative_xj = zkdl::fr_neg(xj);
                for (std::size_t k = 0U; k < basis.size(); ++k) {
                    next[k] = zkdl::fr_add(next[k], zkdl::fr_mul(basis[k], negative_xj));
                    next[k + 1U] = zkdl::fr_add(next[k + 1U], basis[k]);
                }
                basis.swap(next);
                denominator = zkdl::fr_mul(denominator, zkdl::fr_sub(xi, xj));
            }
            const FrWire scale = zkdl::fr_mul(evaluations[i], zkdl::cpu_fr_inv(denominator));
            for (std::size_t k = 0U; k < coefficients.size(); ++k)
                coefficients[k] = zkdl::fr_add(coefficients[k],
                                               zkdl::fr_mul(scale, basis[k]));
        }
        for (std::size_t i = 0U; i < evaluations.size(); ++i)
            if (!same_wire(zksc_eval_coefficients(coefficients,
                                                   zkdl::fr_from_u32(static_cast<std::uint32_t>(i))),
                           evaluations[i])) {
                if (error) *error = "ZkSumcheck interpolation failed its evaluation check";
                return false;
            }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    return true;
}

inline void zksc_append_nested(std::vector<byte>& out,
                               const std::vector<byte>& nested) {
    zkdl::append_u64(out, static_cast<std::uint64_t>(nested.size()));
    out.insert(out.end(), nested.begin(), nested.end());
}

inline std::vector<byte> zksc_round_pre_bytes(
    const zkdl::ZkSumcheckRoundProof& round) {
    std::vector<byte> out;
    zkdl::append_u32(out, round.version);
    zkdl::append_u32(out, round.round_index);
    zkdl::append_g1(out, round.coefficient_commitment);
    zkdl::append_g1(out, round.endpoint_commitment);
    std::vector<byte> endpoint;
    zkdl::serialize_pedersen_vector_linear(round.endpoint_linear, endpoint);
    zksc_append_nested(out, endpoint);
    zkdl::append_u32(out, static_cast<std::uint32_t>(round.link_kind));
    if (round.link_kind == zkdl::ZkSumcheckLinkKind::InitialPublicClaim) {
        std::vector<byte> initial;
        zkdl::serialize_pedersen_scalar_value(round.initial_claim_binding, initial);
        zksc_append_nested(out, initial);
        zkdl::append_u64(out, 0U);
    } else if (round.link_kind == zkdl::ZkSumcheckLinkKind::PreviousEvaluation) {
        zkdl::append_u64(out, 0U);
        std::vector<byte> previous;
        zkdl::serialize_pedersen_scalar_equality(round.previous_claim_binding, previous);
        zksc_append_nested(out, previous);
    } else {
        throw std::invalid_argument("unknown ZkSumcheck link branch");
    }
    return out;
}

inline std::vector<byte> zksc_round_query_bytes(
    const zkdl::ZkSumcheckRoundProof& round) {
    std::vector<byte> out;
    zkdl::append_u32(out, round.round_index);
    zkdl::append_g1(out, round.evaluation_commitment);
    std::vector<byte> evaluation;
    zkdl::serialize_pedersen_vector_linear(round.evaluation_linear, evaluation);
    zksc_append_nested(out, evaluation);
    return out;
}

inline void zksc_absorb_header(zkdl::Transcript& transcript,
                               const zkdl::ZkSumcheckProof& proof) {
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, proof.version);
    zkdl::append_string(encoded, proof.label);
    zkdl::append_u32(encoded, proof.degree);
    zkdl::append_u32(encoded, proof.domain_size);
    zkdl::append_u32(encoded, proof.segment_size);
    zkdl::append_u32(encoded, proof.segment_count);
    zkdl::append_pedersen_context(encoded, proof.initial_claim_context);
    encoded.insert(encoded.end(), proof.descriptor_digest.begin(), proof.descriptor_digest.end());
    encoded.insert(encoded.end(), proof.registry_digest.begin(), proof.registry_digest.end());
    encoded.insert(encoded.end(), proof.srs_digest.begin(), proof.srs_digest.end());
    encoded.insert(encoded.end(), proof.basis_digest.begin(), proof.basis_digest.end());
    zkdl::append_g1(encoded, proof.blinding_generator);
    zkdl::append_g1(encoded, proof.u_generator);
    transcript.absorb("gat.zksumcheck.header", encoded);
}

inline zkdl::PedersenProofContext zksc_make_vector_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t related_index,
    const std::array<byte, 32U>& basis_digest, std::uint32_t degree) {
    return zksc_make_context(
        zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
        zksc_round_commitment_index(round), related_index, basis_digest,
        zksc_vector_role_digest(descriptor, statement, operation_context, role,
                                round, related_index), statement);
}

inline zkdl::PedersenProofContext zksc_make_scalar_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::string& role,
    std::uint32_t round, std::uint32_t commitment_index,
    std::uint32_t related_index, std::uint32_t operation,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator) {
    return zksc_make_context(
        operation, 1U, 0U, commitment_index, related_index,
        zkdl::cpu_ipa_basis_digest({}, blinding_generator, u_generator, 0U),
        zksc_scalar_role_digest(descriptor, statement, operation_context, role,
                                round, commitment_index, related_index), statement);
}

inline bool zksc_validate_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, std::string* error) {
    if (!descriptor.valid(error) || descriptor.degree != degree ||
        degree == 0U || degree > 8U || tensor_values.empty() ||
        domain_size == 0U || (domain_size & (domain_size - 1U)) != 0U ||
        domain_size > (1U << 30U) || generators.size() < static_cast<std::size_t>(degree + 1U) ||
        !zkdl::fr_canonical(initial_claim) || operation_context.empty() ||
        !zksc_statement_resources_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid ZkSumcheck prover inputs";
        return false;
    }
    if (!zkdl::cpu_pedersen_basis_valid(
            std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + degree + 1U),
            blinding_generator, u_generator, degree + 1U, error)) return false;
    for (const auto& tensor : tensor_values) {
        if (tensor.size() != domain_size) {
            if (error) *error = "ZkSumcheck tensor/domain mismatch";
            return false;
        }
        for (const auto& value : tensor)
            if (!zkdl::fr_canonical(value)) {
                if (error) *error = "ZkSumcheck tensor value is not canonical";
                return false;
            }
    }
    return true;
}

inline bool zksc_validate_provider_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& providers,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, std::string* error,
    std::uint32_t shared_segment_size = 0U) {
    if (!descriptor.valid(error) || descriptor.degree != degree || degree == 0U ||
        degree > 8U || providers.empty() || domain_size == 0U ||
        (domain_size & (domain_size - 1U)) != 0U ||
        domain_size > (1U << kMaxRelationPointBits) ||
        generators.size() < static_cast<std::size_t>(degree + 1U) ||
        operation_context.empty() || !zkdl::fr_canonical(initial_claim) ||
        !zksc_statement_resources_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid segmented ZkSumcheck prover inputs";
        return false;
    }
    ZkRelationSegmentLayout expected_layout;
    const bool layout_valid = shared_segment_size == 0U
        ? zktr_segment_layout(domain_size, generators.size(), expected_layout, error)
        : zktr_segment_layout_for_size(domain_size, generators.size(),
                                       shared_segment_size, expected_layout, error);
    if (!layout_valid) return false;
    for (const auto& provider : providers) {
        if (!provider.valid(error) || provider.logical_domain() != domain_size ||
            provider.segment_size() != expected_layout.segment_size ||
            provider.segment_count() != expected_layout.segment_count) {
            if (error) *error = "segmented ZkSumcheck provider/shared-layout mismatch";
            return false;
        }
    }
    const std::vector<zkdl::CpuG1Point> round_generators(
        generators.begin(), generators.begin() + degree + 1U);
    return zkdl::cpu_pedersen_basis_valid(
        round_generators, blinding_generator, u_generator, degree + 1U, error);
}

using ZkSumcheckProgressCallback = std::function<void(
    const char*, std::uint32_t, std::uint32_t, std::uint64_t, std::uint64_t)>;

struct RouteSumcheckProgressReporter {
    using Clock = std::chrono::steady_clock;
    const std::string& operation;
    std::uint32_t total_rounds = 0U;
    Clock::time_point started = Clock::now();
    Clock::time_point last_report = started;

    void update(const char* phase, std::uint32_t round,
                std::uint32_t tensor_index, std::uint64_t completed,
                std::uint64_t total) {
        const auto now = Clock::now();
        if (now - last_report < std::chrono::seconds(15)) return;
        const double elapsed = std::chrono::duration<double>(now - started).count();
        std::clog << "zkdl_native_progress phase=route_sumcheck status=PROGRESS"
                  << " operation_id=" << operation
                  << " step=" << phase
                  << " round=" << round << '/' << total_rounds;
        if (tensor_index != std::numeric_limits<std::uint32_t>::max())
            std::clog << " tensor_index=" << tensor_index;
        std::clog << " completed=" << completed << '/' << total
                  << " elapsed_wall_seconds=" << elapsed << '\n' << std::flush;
        last_report = now;
    }

    void update_route_work(const char* step, std::uint64_t completed,
                           std::uint64_t total) {
        const auto now = Clock::now();
        if (now - last_report < std::chrono::seconds(15)) return;
        const double elapsed = std::chrono::duration<double>(now - started).count();
        std::clog << "zkdl_native_progress phase=route_work status=PROGRESS"
                  << " operation_id=" << operation
                  << " step=" << step
                  << " completed=" << completed << '/' << total
                  << " elapsed_wall_seconds=" << elapsed << '\n' << std::flush;
        last_report = now;
    }
};

struct RelationSegmentProgressReporter {
    using Clock = std::chrono::steady_clock;
    std::string phase;
    Clock::time_point started = Clock::now();
    Clock::time_point last_report = started;

    explicit RelationSegmentProgressReporter(std::string phase_name)
        : phase(std::move(phase_name)) {}

    void update(const std::string& family, std::uint32_t completed,
                std::uint32_t total) {
        const auto now = Clock::now();
        if (now - last_report < std::chrono::seconds(15)) return;
        const double elapsed = std::chrono::duration<double>(now - started).count();
        std::clog << "zkdl_native_progress phase=" << phase
                  << " status=PROGRESS family_id=" << family
                  << " segment=" << completed << '/' << total
                  << " elapsed_wall_seconds=" << elapsed << '\n' << std::flush;
        last_report = now;
    }
};

struct ZkSumcheckRoundEvaluationScratch {
    std::vector<std::vector<FrWire>> row_values;
    std::vector<std::vector<FrWire>> chunk_evaluations;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    std::vector<std::vector<zkdl_blst::blst_fr>> montgomery_row_state;
    std::vector<std::vector<zkdl_blst::blst_fr>> montgomery_chunk_evaluations;
    std::vector<unsigned char> montgomery_chunk_valid;
#endif

    ZkSumcheckRoundEvaluationScratch(std::size_t chunk_count,
                                     std::size_t tensor_count,
                                     std::size_t evaluation_count,
                                     bool use_montgomery = false) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (use_montgomery) {
            if (tensor_count > std::numeric_limits<std::size_t>::max() / 3U)
                throw std::length_error("Sumcheck Montgomery row scratch shape overflows");
            montgomery_row_state.assign(
                chunk_count,
                std::vector<zkdl_blst::blst_fr>(tensor_count * 3U));
            montgomery_chunk_evaluations.assign(
                chunk_count,
                std::vector<zkdl_blst::blst_fr>(evaluation_count));
            montgomery_chunk_valid.assign(chunk_count, 1U);
            return;
        }
#else
        (void)use_montgomery;
#endif
        row_values.assign(chunk_count, std::vector<FrWire>(tensor_count));
        chunk_evaluations.assign(
            chunk_count, std::vector<FrWire>(evaluation_count));
    }
};

struct ZkSumcheckMontgomeryConstraintPlan {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    struct Term {
        zkdl_blst::blst_fr coefficient{};
        const std::vector<std::uint32_t>* refs = nullptr;
    };
    zkdl_blst::blst_fr constant{};
    std::array<zkdl_blst::blst_fr, 9U> evaluation_points{};
    std::vector<Term> terms;
    std::size_t tensor_count = 0U;
    std::uint32_t degree = 0U;
    bool enabled = false;
#else
    bool enabled = false;
#endif
};

inline bool zksc_montgomery_constraint_plan_peak_bytes(
    std::uint64_t term_count, bool enabled,
    std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (!enabled) return true;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    constexpr std::uint64_t fixed_construction_scratch =
        sizeof(ZkSumcheckMontgomeryConstraintPlan::Term) +
        sizeof(zkdl::CpuIpaPreparedScalar) + sizeof(zkdl_blst::blst_scalar) +
        sizeof(std::array<byte, 32U>);
    const std::uint64_t fixed_plan_bytes =
        sizeof(ZkSumcheckMontgomeryConstraintPlan);
    if (term_count > (std::numeric_limits<std::uint64_t>::max() -
                      fixed_plan_bytes) /
                         sizeof(ZkSumcheckMontgomeryConstraintPlan::Term))
        return false;
    const std::uint64_t resident_bytes = fixed_plan_bytes + term_count *
        sizeof(ZkSumcheckMontgomeryConstraintPlan::Term);
    if (fixed_construction_scratch >
        std::numeric_limits<std::uint64_t>::max() - resident_bytes)
        return false;
    peak_bytes = resident_bytes + fixed_construction_scratch;
#else
    (void)term_count;
#endif
    return true;
}

inline bool zksc_folded_state_within_budget(
    const std::vector<SegmentedTensorProvider>& providers,
    const ConstraintDescriptor& descriptor,
    std::uint64_t extra_live_bytes = 0U) noexcept {
    std::uint64_t plan_peak_bytes = 0U;
    return zksc_montgomery_constraint_plan_peak_bytes(
               static_cast<std::uint64_t>(descriptor.terms.size()), true,
               plan_peak_bytes) &&
           zksc_folded_state_within_budget(
               providers, plan_peak_bytes, extra_live_bytes);
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
inline bool zksc_compile_montgomery_constraint_plan(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count,
    std::uint32_t degree, ZkSumcheckMontgomeryConstraintPlan& plan,
    std::string* error = nullptr) {
    plan = ZkSumcheckMontgomeryConstraintPlan{};
    if (!descriptor.valid(error) || descriptor.degree != degree ||
        tensor_count == 0U || degree > 8U) {
        if (error && error->empty())
            *error = "Sumcheck Montgomery descriptor plan shape is invalid";
        return false;
    }
    for (const auto& term : descriptor.terms)
        for (const std::uint32_t ref : term.refs)
            if (ref >= tensor_count) {
                if (error) *error = "Sumcheck descriptor tensor reference is out of bounds";
                return false;
            }
    try {
        plan.constant = zkdl::cpu_ipa_frwire_to_montgomery_checked(
            descriptor.constant);
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
            plan.evaluation_points[evaluation] =
                zkdl::cpu_ipa_frwire_to_montgomery_checked(
                    zkdl::fr_from_u32(evaluation));
        plan.terms.reserve(descriptor.terms.size());
        for (const auto& term : descriptor.terms) {
            ZkSumcheckMontgomeryConstraintPlan::Term compiled;
            compiled.coefficient = zkdl::cpu_ipa_frwire_to_montgomery_checked(
                term.coefficient);
            compiled.refs = &term.refs;
            plan.terms.push_back(compiled);
        }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    plan.tensor_count = tensor_count;
    plan.degree = degree;
    plan.enabled = true;
    return true;
}

inline zkdl_blst::blst_fr zksc_eval_constraint_montgomery(
    const ZkSumcheckMontgomeryConstraintPlan& plan,
    const zkdl_blst::blst_fr* values) {
    zkdl_blst::blst_fr result = plan.constant;
    for (const auto& term : plan.terms) {
        zkdl_blst::blst_fr product = term.coefficient;
        for (const std::uint32_t ref : *term.refs) {
            zkdl_blst::blst_fr next{};
            zkdl_blst::blst_fr_mul(&next, &product, &values[ref]);
            product = next;
        }
        zkdl_blst::blst_fr next{};
        zkdl_blst::blst_fr_add(&next, &result, &product);
        result = next;
    }
    return result;
}
#endif

inline bool zksc_accumulate_round_evaluation_rows(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& current,
    std::uint32_t degree, std::size_t row_begin, std::size_t row_end,
    std::vector<FrWire>& evaluations,
    ZkSumcheckRoundEvaluationScratch& scratch,
    std::string* error = nullptr,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan = nullptr) {
    if (row_begin > row_end || current.empty() || current.front().size() < 2U ||
        (current.front().size() & 1U) != 0U || degree > 8U ||
        evaluations.size() != degree + 1U ||
        row_end > current.front().size() / 2U) {
        if (error) *error = "segmented ZkSumcheck current round shape is invalid";
        return false;
    }
    for (const auto& tensor : current)
        if (tensor.size() != current.front().size()) {
            if (error) *error = "segmented ZkSumcheck current tensor shapes differ";
            return false;
        }
    const std::size_t row_count = row_end - row_begin;
    if (row_count == 0U) return true;

    std::size_t chunk_limit = 1U;
    if (row_count >= 4096U) {
        const std::size_t available =
            static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U;
        chunk_limit = std::min(row_count, available);
    }
    const std::size_t grain = row_count / chunk_limit +
                              (row_count % chunk_limit != 0U ? 1U : 0U);
    const std::size_t active_chunks = row_count / grain +
                                      (row_count % grain != 0U ? 1U : 0U);
    if (active_chunks == 0U ||
        (montgomery_plan == nullptr &&
         (active_chunks > scratch.row_values.size() ||
          active_chunks > scratch.chunk_evaluations.size()))) {
        if (error) *error = "segmented ZkSumcheck row scratch shape is invalid";
        return false;
    }

#if defined(ZKDL_USE_BLST_G1_CODEC)
    if (montgomery_plan != nullptr) {
        if (!montgomery_plan->enabled || montgomery_plan->degree != degree ||
            montgomery_plan->tensor_count != current.size() ||
            active_chunks > scratch.montgomery_row_state.size() ||
            active_chunks > scratch.montgomery_chunk_evaluations.size() ||
            active_chunks > scratch.montgomery_chunk_valid.size()) {
            if (error) *error = "segmented ZkSumcheck Montgomery scratch/plan shape is invalid";
            return false;
        }
        const std::size_t row_state_size = current.size() * 3U;
        for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
            if (scratch.montgomery_row_state[chunk].size() != row_state_size ||
                scratch.montgomery_chunk_evaluations[chunk].size() != evaluations.size()) {
                if (error) *error = "segmented ZkSumcheck Montgomery scratch shape is invalid";
                return false;
            }
            std::fill(scratch.montgomery_chunk_evaluations[chunk].begin(),
                      scratch.montgomery_chunk_evaluations[chunk].end(),
                      zkdl_blst::blst_fr{});
            scratch.montgomery_chunk_valid[chunk] = 1U;
        }
        zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
            const std::size_t chunk = local_row / grain;
            if (scratch.montgomery_chunk_valid[chunk] == 0U) return;
            const std::size_t row = row_begin + local_row;
            auto& row_state = scratch.montgomery_row_state[chunk];
            auto& partial = scratch.montgomery_chunk_evaluations[chunk];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                const FrWire& left_wire = current[tensor][2U * row];
                const FrWire& right_wire = current[tensor][2U * row + 1U];
                if (!zkdl::fr_canonical(left_wire) ||
                    !zkdl::fr_canonical(right_wire)) {
                    scratch.montgomery_chunk_valid[chunk] = 0U;
                    return;
                }
                const auto left = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(left_wire);
                const auto right = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(right_wire);
                row_state[tensor] = left;
                zkdl_blst::blst_fr_sub(
                    &row_state[current.size() + tensor], &right, &left);
            }
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
                const auto& x = montgomery_plan->evaluation_points[evaluation];
                for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                    zkdl_blst::blst_fr scaled{};
                    zkdl_blst::blst_fr_mul(
                        &scaled, &x, &row_state[current.size() + tensor]);
                    zkdl_blst::blst_fr_add(
                        &row_state[2U * current.size() + tensor],
                        &row_state[tensor], &scaled);
                }
                const auto value = zksc_eval_constraint_montgomery(
                    *montgomery_plan, row_state.data() + 2U * current.size());
                zkdl_blst::blst_fr next{};
                zkdl_blst::blst_fr_add(
                    &next, &partial[evaluation], &value);
                partial[evaluation] = next;
            }
        });
        for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
            if (scratch.montgomery_chunk_valid[chunk] == 0U) {
                if (error) *error = "ZkSumcheck tensor value is not canonical";
                return false;
            }
        }
        for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
                evaluations[evaluation] = zkdl::fr_add(
                    evaluations[evaluation],
                    zkdl::cpu_ipa_montgomery_to_frwire(
                        scratch.montgomery_chunk_evaluations[chunk][evaluation]));
        return true;
    }
#else
    (void)montgomery_plan;
#endif

    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        if (scratch.row_values[chunk].size() != current.size() ||
            scratch.chunk_evaluations[chunk].size() != evaluations.size()) {
            if (error) *error = "segmented ZkSumcheck row scratch shape is invalid";
            return false;
        }
    std::array<FrWire, 9U> evaluation_points{};
    for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
        evaluation_points[evaluation] = zkdl::fr_from_u32(evaluation);
    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        std::fill(scratch.chunk_evaluations[chunk].begin(),
                  scratch.chunk_evaluations[chunk].end(), FrWire{});

    zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
        const std::size_t chunk = local_row / grain;
        const std::size_t row = row_begin + local_row;
        auto& values = scratch.row_values[chunk];
        auto& partial = scratch.chunk_evaluations[chunk];
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
            const FrWire& x = evaluation_points[evaluation];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                const FrWire& left = current[tensor][2U * row];
                const FrWire& right = current[tensor][2U * row + 1U];
                values[tensor] = zkdl::fr_add(
                    left, zkdl::fr_mul(x, zkdl::fr_sub(right, left)));
            }
            partial[evaluation] = zkdl::fr_add(
                partial[evaluation], eval_constraint(descriptor, values));
        }
    });

    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
            evaluations[evaluation] = zkdl::fr_add(
                evaluations[evaluation],
                scratch.chunk_evaluations[chunk][evaluation]);
    return true;
}

inline bool zksc_accumulate_round_evaluations(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& current,
    std::uint32_t degree, std::vector<FrWire>& evaluations,
    std::string* error = nullptr,
    std::uint32_t round_index = 0U,
    const ZkSumcheckProgressCallback* progress = nullptr,
    const char* progress_phase = "sumcheck_cached_rows",
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan = nullptr) {
    if (current.empty() || current.front().size() < 2U ||
        (current.front().size() & 1U) != 0U || degree > 8U ||
        evaluations.size() != degree + 1U) {
        if (error) *error = "segmented ZkSumcheck current round shape is invalid";
        return false;
    }
    for (const auto& tensor : current)
        if (tensor.size() != current.front().size()) {
            if (error) *error = "segmented ZkSumcheck current tensor shapes differ";
            return false;
        }
    const std::size_t row_count = current.front().size() / 2U;
    constexpr std::size_t kProgressRows = 1U << 18U;
    const std::size_t scratch_rows = progress == nullptr
        ? row_count : std::min(row_count, kProgressRows);
    const std::size_t scratch_chunks = scratch_rows < 4096U ? 1U :
        std::min(scratch_rows,
                 static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    ZkSumcheckRoundEvaluationScratch scratch(
        scratch_chunks, current.size(), evaluations.size(),
        montgomery_plan != nullptr);
    for (std::size_t row_begin = 0U; row_begin < row_count;) {
        const std::size_t row_end = row_begin + std::min(
            row_count - row_begin, progress == nullptr ? row_count : kProgressRows);
        if (!zksc_accumulate_round_evaluation_rows(
                descriptor, current, degree, row_begin, row_end,
                evaluations, scratch, error, montgomery_plan)) return false;
        row_begin = row_end;
        if (progress != nullptr)
            (*progress)(progress_phase, round_index,
                        std::numeric_limits<std::uint32_t>::max(), row_end,
                        row_count);
    }
    return true;
}

inline bool zksc_segmented_round_evaluations(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& providers,
    std::uint32_t degree, std::uint32_t round_index,
    const std::vector<FrWire>& prior_challenges,
    std::vector<FrWire>& evaluations, std::string* error = nullptr,
    const ZkSumcheckProgressCallback* progress = nullptr,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan = nullptr) {
    if (providers.empty() || prior_challenges.size() != round_index ||
        providers.front().logical_domain() == 0U) {
        if (error) *error = "segmented ZkSumcheck round state is invalid";
        return false;
    }
    const std::uint32_t logical_domain = providers.front().logical_domain();
    const std::uint32_t segment_size = providers.front().segment_size();
    const std::uint32_t segment_count = providers.front().segment_count();
    std::uint32_t low_bits = 0U;
    for (std::uint32_t value = segment_size; value > 1U; value >>= 1U) ++low_bits;
    std::uint32_t total_bits = 0U;
    for (std::uint32_t value = logical_domain; value > 1U; value >>= 1U) ++total_bits;
    if (round_index >= total_bits || round_index > zkdl::ZkSumcheckProof::kMaxRounds) {
        if (error) *error = "segmented ZkSumcheck round index is out of bounds";
        return false;
    }
    evaluations.assign(degree + 1U, FrWire{});
    try {
        if (round_index < low_bits) {
            for (std::uint32_t segment = 0U; segment < segment_count; ++segment) {
                std::vector<std::vector<FrWire>> current(providers.size());
                for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
                    if (!providers[tensor].read_segment(segment, current[tensor], error)) return false;
                    for (std::uint32_t previous = 0U; previous < round_index; ++previous)
                        fold_values(current[tensor], prior_challenges[previous]);
                }
                if (!zksc_accumulate_round_evaluations(
                        descriptor, current, degree, evaluations, error,
                        round_index, progress,
                        "sumcheck_provider_rows", montgomery_plan)) return false;
                if (progress != nullptr)
                    (*progress)("sumcheck_provider_segments", round_index,
                                std::numeric_limits<std::uint32_t>::max(),
                                segment + 1U, segment_count);
            }
        } else {
            std::vector<std::vector<FrWire>> current(providers.size(),
                                                      std::vector<FrWire>(segment_count));
            for (std::uint32_t segment = 0U; segment < segment_count; ++segment) {
                for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
                    std::vector<FrWire> block;
                    if (!providers[tensor].read_segment(segment, block, error)) return false;
                    for (std::uint32_t previous = 0U; previous < low_bits; ++previous)
                        fold_values(block, prior_challenges[previous]);
                    current[tensor][segment] = block.front();
                }
                if (progress != nullptr)
                    (*progress)("sumcheck_provider_segments", round_index,
                                std::numeric_limits<std::uint32_t>::max(),
                                segment + 1U, segment_count);
            }
            for (std::uint32_t previous = low_bits; previous < round_index; ++previous)
                for (auto& tensor : current)
                    fold_values(tensor, prior_challenges[previous]);
            if (!zksc_accumulate_round_evaluations(
                    descriptor, current, degree, evaluations, error,
                    round_index, nullptr, "sumcheck_cached_rows",
                    montgomery_plan)) return false;
        }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    return true;
}

inline bool zksc_segmented_fold_state_initialize(
    const std::vector<SegmentedTensorProvider>& providers,
    const FrWire& challenge,
    ZkSumcheckFoldedState& state,
    std::string* error = nullptr,
    std::uint32_t progress_round = 0U,
    const ZkSumcheckProgressCallback* progress = nullptr) {
    if (providers.empty() || !zkdl::fr_canonical(challenge) ||
        providers.front().logical_domain() < 2U) {
        if (error) *error = "segmented ZkSumcheck fold-state initialization is invalid";
        return false;
    }
    const std::uint32_t logical_domain = providers.front().logical_domain();
    const std::uint32_t segment_size = providers.front().segment_size();
    const std::uint32_t segment_count = providers.front().segment_count();
    if ((logical_domain & 1U) != 0U || (segment_size & 1U) != 0U) {
        if (error) *error = "segmented ZkSumcheck fold-state domain is not pair aligned";
        return false;
    }
    try {
        state.clear();
        state.resize(providers.size());
        for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
            if (!providers[tensor].valid(error) ||
                providers[tensor].logical_domain() != logical_domain ||
                providers[tensor].segment_size() != segment_size ||
                providers[tensor].segment_count() != segment_count) {
                if (error && error->empty()) *error = "segmented ZkSumcheck fold-state provider layout mismatch";
                return false;
            }
            state[tensor].reserve(logical_domain / 2U);
        }
#if defined(ZKDL_USE_BLST_G1_CODEC)
        const zkdl_blst::blst_fr challenge_montgomery =
            zkdl::cpu_ipa_frwire_to_montgomery_unchecked(challenge);
#endif
        for (std::uint32_t segment = 0U; segment < segment_count; ++segment) {
            for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
                std::vector<FrWire> block;
                if (!providers[tensor].read_segment(segment, block, error) ||
                    block.size() != segment_size) return false;
#if defined(ZKDL_USE_BLST_G1_CODEC)
                std::vector<zkdl_blst::blst_fr> folded_values;
                folded_values.reserve(segment_size / 2U);
                for (std::size_t index = 0U; index < block.size() / 2U; ++index) {
                    if (!zkdl::fr_canonical(block[2U * index]) ||
                        !zkdl::fr_canonical(block[2U * index + 1U])) {
                        if (error) *error = "segmented ZkSumcheck provider scalar is not canonical";
                        return false;
                    }
                    const zkdl_blst::blst_fr left =
                        zkdl::cpu_ipa_frwire_to_montgomery_unchecked(block[2U * index]);
                    const zkdl_blst::blst_fr right =
                        zkdl::cpu_ipa_frwire_to_montgomery_unchecked(block[2U * index + 1U]);
                    zkdl_blst::blst_fr difference{}, scaled{}, folded{};
                    zkdl_blst::blst_fr_sub(&difference, &right, &left);
                    zkdl_blst::blst_fr_mul(&scaled, &challenge_montgomery, &difference);
                    zkdl_blst::blst_fr_add(&folded, &left, &scaled);
                    folded_values.push_back(folded);
                }
                state[tensor].insert(state[tensor].end(), folded_values.begin(), folded_values.end());
#else
                fold_values(block, challenge);
                state[tensor].insert(state[tensor].end(), block.begin(), block.end());
#endif
            }
            if (progress != nullptr)
                (*progress)("fold_state_init_segments", progress_round,
                            std::numeric_limits<std::uint32_t>::max(),
                            segment + 1U, segment_count);
        }
        for (const auto& tensor : state)
            if (tensor.size() != logical_domain / 2U) {
                if (error) *error = "segmented ZkSumcheck fold-state coverage mismatch";
                return false;
            }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    return true;
}

inline bool zksc_fold_values_in_place(std::vector<ZkSumcheckFoldedScalar>& values,
                                      const FrWire& challenge,
                                      std::string* error = nullptr,
                                      std::uint32_t progress_round = 0U,
                                      std::uint32_t tensor_index = 0U,
                                      const ZkSumcheckProgressCallback* progress = nullptr) {
    if (values.size() < 2U || (values.size() & 1U) != 0U ||
        !zkdl::fr_canonical(challenge)) {
        if (error) *error = "segmented ZkSumcheck in-place fold shape is invalid";
        return false;
    }
    const std::size_t folded_size = values.size() / 2U;
    constexpr std::size_t kProgressRows = 1U << 18U;
#if defined(ZKDL_USE_BLST_G1_CODEC)
    const zkdl_blst::blst_fr challenge_montgomery =
        zkdl::cpu_ipa_frwire_to_montgomery_unchecked(challenge);
#else
    const FrWire one_minus = zkdl::fr_sub(zkdl::fr_from_u32(1U), challenge);
#endif
    for (std::size_t begin = 0U; begin < folded_size;) {
        const std::size_t end = progress == nullptr
            ? folded_size : std::min(folded_size, begin + kProgressRows);
        for (std::size_t index = begin; index < end; ++index) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
            const zkdl_blst::blst_fr left = values[2U * index];
            const zkdl_blst::blst_fr right = values[2U * index + 1U];
            zkdl_blst::blst_fr difference{}, scaled{}, folded{};
            zkdl_blst::blst_fr_sub(&difference, &right, &left);
            zkdl_blst::blst_fr_mul(&scaled, &challenge_montgomery, &difference);
            zkdl_blst::blst_fr_add(&folded, &left, &scaled);
            values[index] = folded;
#else
            const FrWire left = values[2U * index];
            const FrWire right = values[2U * index + 1U];
            values[index] = zkdl::fr_add(zkdl::fr_mul(one_minus, left),
                                         zkdl::fr_mul(challenge, right));
#endif
        }
        begin = end;
        if (progress != nullptr)
            (*progress)("fold_state_rows", progress_round, tensor_index,
                        end, folded_size);
    }
    values.resize(folded_size);
    return true;
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
// Spilled folded state remains canonical wire storage.  Keep its legacy wire
// fold separate from the cached in-memory BLST Montgomery representation.
inline bool zksc_fold_values_in_place(std::vector<FrWire>& values,
                                      const FrWire& challenge,
                                      std::string* error = nullptr,
                                      std::uint32_t progress_round = 0U,
                                      std::uint32_t tensor_index = 0U,
                                      const ZkSumcheckProgressCallback* progress = nullptr) {
    if (values.size() < 2U || (values.size() & 1U) != 0U ||
        !zkdl::fr_canonical(challenge)) {
        if (error) *error = "segmented ZkSumcheck wire fold shape is invalid";
        return false;
    }
    const std::size_t folded_size = values.size() / 2U;
    constexpr std::size_t kProgressRows = 1U << 18U;
    const FrWire one_minus = zkdl::fr_sub(zkdl::fr_from_u32(1U), challenge);
    for (std::size_t begin = 0U; begin < folded_size;) {
        const std::size_t end = progress == nullptr
            ? folded_size : std::min(folded_size, begin + kProgressRows);
        for (std::size_t index = begin; index < end; ++index) {
            const FrWire left = values[2U * index];
            const FrWire right = values[2U * index + 1U];
            values[index] = zkdl::fr_add(zkdl::fr_mul(one_minus, left),
                                         zkdl::fr_mul(challenge, right));
        }
        begin = end;
        if (progress != nullptr)
            (*progress)("fold_state_rows", progress_round, tensor_index,
                        end, folded_size);
    }
    values.resize(folded_size);
    return true;
}
#endif

inline bool zksc_spilled_round_evaluations(
    const ConstraintDescriptor& descriptor,
    const ZkSumcheckFoldedSpillFile& state,
    std::uint32_t degree, std::vector<FrWire>& evaluations,
    std::string* error = nullptr,
    std::uint32_t round_index = 0U,
    const ZkSumcheckProgressCallback* progress = nullptr,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan = nullptr) {
    const std::uint64_t tensor_count = state.tensor_count();
    const std::uint64_t state_rows = state.logical_rows();
    if (tensor_count == 0U || state_rows < 2U || (state_rows & 1U) != 0U ||
        degree > 8U ||
        !state.validate_complete(state.operation_identity(), state.logical_domain(),
                                 tensor_count, state_rows, round_index, error)) {
        if (error && error->empty()) *error = "Sumcheck spill round evaluation state is invalid";
        return false;
    }
    const std::uint64_t row_count = state_rows / 2U;
    const std::uint64_t max_io_rows = std::min<std::uint64_t>(
        row_count, kZkSumcheckSpillIoRows);
    std::uint64_t input_values = 0U, input_bytes = 0U;
    if (max_io_rows == 0U ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (2U * max_io_rows) ||
        tensor_count * (2U * max_io_rows) >
            std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire)) {
        if (error) *error = "Sumcheck spill evaluation block size overflows";
        return false;
    }
    input_values = tensor_count * (2U * max_io_rows);
    input_bytes = input_values * sizeof(FrWire);
    ZkSumcheckAlignedBuffer input;
    if (!input.allocate(input_bytes, error)) return false;
    const std::size_t scratch_chunks = row_count < 4096U ? 1U :
        std::min<std::uint64_t>(row_count,
            static_cast<std::uint64_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    ZkSumcheckRoundEvaluationScratch scratch(
        scratch_chunks, static_cast<std::size_t>(tensor_count), degree + 1U,
        montgomery_plan != nullptr);
    evaluations.assign(degree + 1U, FrWire{});
    for (std::uint64_t row_begin = 0U; row_begin < row_count;) {
        const std::uint64_t rows = std::min<std::uint64_t>(
            row_count - row_begin, kZkSumcheckSpillIoRows);
        const std::uint64_t offset = row_begin * 2U * tensor_count;
        const std::uint64_t count = rows * 2U * tensor_count;
        if (!state.read_values(offset, count, input.wires(), input_values,
                               state.operation_identity(), state.logical_domain(),
                               tensor_count, state_rows, round_index, error)) return false;
        std::size_t chunk_limit = 1U;
        if (rows >= 4096U) chunk_limit = std::min<std::uint64_t>(
            rows, static_cast<std::uint64_t>(zkdl::cpu_parallel_worker_count()) + 1U);
        const std::size_t grain = static_cast<std::size_t>(rows / chunk_limit +
            (rows % chunk_limit != 0U ? 1U : 0U));
        const std::size_t active_chunks = static_cast<std::size_t>(rows / grain +
            (rows % grain != 0U ? 1U : 0U));
        if (active_chunks == 0U ||
            (montgomery_plan == nullptr &&
             (active_chunks > scratch.row_values.size() ||
              active_chunks > scratch.chunk_evaluations.size()))) {
            if (error) *error = "Sumcheck spill row evaluation scratch shape is invalid";
            return false;
        }
        const FrWire* values = input.wires();
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (montgomery_plan != nullptr) {
            if (!montgomery_plan->enabled || montgomery_plan->degree != degree ||
                montgomery_plan->tensor_count != tensor_count ||
                active_chunks > scratch.montgomery_row_state.size() ||
                active_chunks > scratch.montgomery_chunk_evaluations.size() ||
                active_chunks > scratch.montgomery_chunk_valid.size()) {
                if (error) *error = "Sumcheck spill Montgomery scratch/plan shape is invalid";
                return false;
            }
            for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
                std::fill(scratch.montgomery_chunk_evaluations[chunk].begin(),
                          scratch.montgomery_chunk_evaluations[chunk].end(),
                          zkdl_blst::blst_fr{});
                scratch.montgomery_chunk_valid[chunk] = 1U;
            }
            zkdl::cpu_parallel_for(static_cast<std::size_t>(rows), grain,
                [&](std::size_t local_row) {
                    const std::size_t chunk = local_row / grain;
                    if (scratch.montgomery_chunk_valid[chunk] == 0U) return;
                    auto& row_state = scratch.montgomery_row_state[chunk];
                    auto& partial = scratch.montgomery_chunk_evaluations[chunk];
                    const std::size_t base =
                        2U * local_row * static_cast<std::size_t>(tensor_count);
                    for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor) {
                        const FrWire& left_wire = values[base + tensor];
                        const FrWire& right_wire = values[base + tensor_count + tensor];
                        if (!zkdl::fr_canonical(left_wire) ||
                            !zkdl::fr_canonical(right_wire)) {
                            scratch.montgomery_chunk_valid[chunk] = 0U;
                            return;
                        }
                        const auto left = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(
                            left_wire);
                        const auto right = zkdl::cpu_ipa_frwire_to_montgomery_unchecked(
                            right_wire);
                        row_state[tensor] = left;
                        zkdl_blst::blst_fr_sub(
                            &row_state[static_cast<std::size_t>(tensor_count) + tensor],
                            &right, &left);
                    }
                    for (std::uint32_t evaluation = 0U;
                         evaluation <= degree; ++evaluation) {
                        const auto& x = montgomery_plan->evaluation_points[evaluation];
                        for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor) {
                            zkdl_blst::blst_fr scaled{};
                            zkdl_blst::blst_fr_mul(
                                &scaled, &x,
                                &row_state[static_cast<std::size_t>(tensor_count) + tensor]);
                            zkdl_blst::blst_fr_add(
                                &row_state[2U * static_cast<std::size_t>(tensor_count) + tensor],
                                &row_state[tensor], &scaled);
                        }
                        const auto value = zksc_eval_constraint_montgomery(
                            *montgomery_plan,
                            row_state.data() + 2U * static_cast<std::size_t>(tensor_count));
                        zkdl_blst::blst_fr next{};
                        zkdl_blst::blst_fr_add(&next, &partial[evaluation], &value);
                        partial[evaluation] = next;
                    }
                });
            for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
                if (scratch.montgomery_chunk_valid[chunk] == 0U) {
                    if (error) *error = "ZkSumcheck tensor value is not canonical";
                    return false;
                }
            }
            for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
                for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
                    evaluations[evaluation] = zkdl::fr_add(
                        evaluations[evaluation],
                        zkdl::cpu_ipa_montgomery_to_frwire(
                            scratch.montgomery_chunk_evaluations[chunk][evaluation]));
        } else
#endif
        {
            if (active_chunks > scratch.row_values.size() ||
                active_chunks > scratch.chunk_evaluations.size()) {
                if (error) *error = "Sumcheck spill row evaluation scratch shape is invalid";
                return false;
            }
            for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
                std::fill(scratch.chunk_evaluations[chunk].begin(),
                          scratch.chunk_evaluations[chunk].end(), FrWire{});
            std::array<FrWire, 9U> evaluation_points{};
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
                evaluation_points[evaluation] = zkdl::fr_from_u32(evaluation);
            zkdl::cpu_parallel_for(static_cast<std::size_t>(rows), grain,
                [&](std::size_t local_row) {
                    const std::size_t chunk = local_row / grain;
                    auto& row_values = scratch.row_values[chunk];
                    auto& partial = scratch.chunk_evaluations[chunk];
                    const std::size_t base =
                        2U * local_row * static_cast<std::size_t>(tensor_count);
                    for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
                        const FrWire x = evaluation_points[evaluation];
                        for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor) {
                            const FrWire& left = values[base + tensor];
                            const FrWire& right = values[base + tensor_count + tensor];
                            row_values[tensor] = zkdl::fr_add(
                                left, zkdl::fr_mul(x, zkdl::fr_sub(right, left)));
                        }
                        partial[evaluation] = zkdl::fr_add(
                            partial[evaluation], eval_constraint(descriptor, row_values));
                    }
                });
            for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
                for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
                    evaluations[evaluation] = zkdl::fr_add(
                        evaluations[evaluation], scratch.chunk_evaluations[chunk][evaluation]);
        }
        row_begin += rows;
        if (progress != nullptr)
            (*progress)("sumcheck_spill_eval_rows", round_index,
                        std::numeric_limits<std::uint32_t>::max(), row_begin,
                        row_count);
    }
    return true;
}

inline bool zksc_spilled_fold_state_initialize(
    const std::vector<SegmentedTensorProvider>& providers,
    const FrWire& challenge, const std::string& operation_identity,
    const std::string& directory, std::uint32_t round_index,
    ZkSumcheckFoldedSpillFile& state, std::string* error = nullptr,
    const ZkSumcheckProgressCallback* progress = nullptr) {
    if (providers.empty() || !zkdl::fr_canonical(challenge) ||
        providers.front().logical_domain() < 2U || directory.empty()) {
        if (error) *error = "Sumcheck spill initialization inputs are invalid";
        return false;
    }
    const std::uint32_t domain = providers.front().logical_domain();
    const std::uint32_t segment_size = providers.front().segment_size();
    const std::uint32_t segment_count = providers.front().segment_count();
    if ((domain & 1U) != 0U || (segment_size & 1U) != 0U ||
        static_cast<std::uint64_t>(domain) !=
            static_cast<std::uint64_t>(segment_size) * segment_count ||
        static_cast<std::uint64_t>(providers.size()) >
            std::numeric_limits<std::uint64_t>::max() / (segment_size / 2U)) {
        if (error) *error = "Sumcheck spill initialization provider layout is invalid";
        return false;
    }
    std::uint64_t tile_values = static_cast<std::uint64_t>(providers.size()) *
                                (segment_size / 2U);
    if (tile_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire)) {
        if (error) *error = "Sumcheck spill initialization tile size overflows";
        return false;
    }
    const std::uint64_t tile_bytes = tile_values * sizeof(FrWire);
    std::uint64_t peak = 0U;
    if (!zksc_spilled_state_ram_peak_bytes(providers.size(), domain,
                                           segment_size, peak) ||
        peak > kZkSumcheckSpillRamBudgetBytes) {
        if (error) *error = "Sumcheck spill initialization exceeds its bounded RAM envelope";
        return false;
    }
    ZkSumcheckAlignedBuffer tile;
    if (!tile.allocate(tile_bytes, error) ||
        !ZkSumcheckFoldedSpillFile::create(
            directory, operation_identity, domain, providers.size(), domain / 2U,
            round_index, state, error)) return false;
    auto* tile_values_ptr = tile.wires();
    for (const auto& provider : providers)
        if (!provider.valid(error) || provider.logical_domain() != domain ||
            provider.segment_size() != segment_size ||
            provider.segment_count() != segment_count) {
            if (error && error->empty()) *error = "Sumcheck spill provider layout mismatch";
            return false;
        }
    for (std::uint32_t segment = 0U; segment < segment_count; ++segment) {
        for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
            std::vector<FrWire> block;
            if (!providers[tensor].read_segment(segment, block, error) ||
                block.size() != segment_size ||
                !zksc_fold_values_in_place(block, challenge, error,
                                           round_index, static_cast<std::uint32_t>(tensor),
                                           progress)) return false;
            for (std::uint32_t row = 0U; row < segment_size / 2U; ++row)
                tile_values_ptr[static_cast<std::uint64_t>(row) * providers.size() + tensor] =
                    block[row];
        }
        if (!state.append_values(tile_values_ptr, tile_values, error)) return false;
        if (progress != nullptr)
            (*progress)("sumcheck_spill_init_segments", round_index,
                        std::numeric_limits<std::uint32_t>::max(), segment + 1U,
                        segment_count);
    }
    return state.finish(error);
}

inline bool zksc_spilled_fold_state(
    const FrWire& challenge, ZkSumcheckFoldedSpillFile& state,
    std::uint32_t next_round, std::string* error = nullptr,
    const ZkSumcheckProgressCallback* progress = nullptr) {
    if (!zkdl::fr_canonical(challenge) || state.logical_rows() < 2U ||
        (state.logical_rows() & 1U) != 0U ||
        state.round_index() + 1U != next_round ||
        !state.validate_complete(state.operation_identity(), state.logical_domain(),
                                  state.tensor_count(), state.logical_rows(),
                                  state.round_index(), error)) {
        if (error && error->empty()) *error = "Sumcheck spill fold input state is invalid";
        return false;
    }
    const std::uint64_t tensor_count = state.tensor_count();
    const std::uint64_t output_rows = state.logical_rows() / 2U;
    std::uint64_t input_values = 0U, output_values = 0U;
    if (tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (2U * kZkSumcheckSpillIoRows) ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           kZkSumcheckSpillIoRows) {
        if (error) *error = "Sumcheck spill fold block size overflows";
        return false;
    }
    const std::uint64_t max_io_rows = std::min<std::uint64_t>(
        output_rows, kZkSumcheckSpillIoRows);
    if (max_io_rows == 0U ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           (2U * max_io_rows) ||
        tensor_count > std::numeric_limits<std::uint64_t>::max() /
                           max_io_rows) {
        if (error) *error = "Sumcheck spill fold block size overflows";
        return false;
    }
    input_values = tensor_count * (2U * max_io_rows);
    output_values = tensor_count * max_io_rows;
    ZkSumcheckAlignedBuffer input, output;
    if (input_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        output_values > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        !input.allocate(input_values * sizeof(FrWire), error) ||
        !output.allocate(output_values * sizeof(FrWire), error)) return false;
    ZkSumcheckFoldedSpillFile next;
    if (!ZkSumcheckFoldedSpillFile::create(
            state.directory(), state.operation_identity(), state.logical_domain(),
            tensor_count, output_rows, next_round, next, error)) return false;
    const FrWire one_minus = zkdl::fr_sub(zkdl::fr_from_u32(1U), challenge);
    for (std::uint64_t row_begin = 0U; row_begin < output_rows;) {
        const std::uint64_t rows = std::min<std::uint64_t>(
            output_rows - row_begin, kZkSumcheckSpillIoRows);
        const std::uint64_t input_offset = row_begin * 2U * tensor_count;
        const std::uint64_t count = rows * 2U * tensor_count;
        if (!state.read_values(input_offset, count, input.wires(), input_values,
                               state.operation_identity(), state.logical_domain(),
                               tensor_count, state.logical_rows(), state.round_index(),
                               error)) return false;
        const FrWire* input_values_ptr = input.wires();
        FrWire* output_values_ptr = output.wires();
        zkdl::cpu_parallel_for(static_cast<std::size_t>(rows), 128U,
            [&](std::size_t local_row) {
                const std::size_t input_base = 2U * local_row * tensor_count;
                const std::size_t output_base = local_row * tensor_count;
                for (std::size_t tensor = 0U; tensor < tensor_count; ++tensor) {
                    const FrWire left = input_values_ptr[input_base + tensor];
                    const FrWire right = input_values_ptr[input_base + tensor_count + tensor];
                    output_values_ptr[output_base + tensor] = zkdl::fr_add(
                        zkdl::fr_mul(one_minus, left), zkdl::fr_mul(challenge, right));
                }
            });
        if (!next.append_values(output_values_ptr, rows * tensor_count, error)) return false;
        row_begin += rows;
        if (progress != nullptr)
            (*progress)("sumcheck_spill_fold_rows", next_round,
                        std::numeric_limits<std::uint32_t>::max(), row_begin,
                        output_rows);
    }
    if (!next.finish(error)) return false;
    state = std::move(next);
    return true;
}

inline bool zksc_spilled_terminal_values(
    const ZkSumcheckFoldedSpillFile& state, std::uint32_t expected_round,
    std::vector<FrWire>& terminal, std::string* error = nullptr) {
    if (state.logical_rows() != 1U ||
        !state.validate_complete(state.operation_identity(), state.logical_domain(),
                                 state.tensor_count(), 1U, expected_round, error)) {
        if (error && error->empty()) *error = "Sumcheck spill terminal state shape is invalid";
        return false;
    }
    ZkSumcheckAlignedBuffer values;
    std::uint64_t count = state.tensor_count();
    if (count > std::numeric_limits<std::uint64_t>::max() / sizeof(FrWire) ||
        !values.allocate(count * sizeof(FrWire), error) ||
        !state.read_values(0U, count, values.wires(), values.size() / sizeof(FrWire),
                           state.operation_identity(), state.logical_domain(), count,
                           1U, expected_round, error)) return false;
    terminal.assign(values.wires(), values.wires() + count);
    return true;
}

#if defined(ZKDL_USE_BLST_G1_CODEC)
inline bool zksc_accumulate_montgomery_folded_round_evaluation_rows(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<zkdl_blst::blst_fr>>& current,
    std::uint32_t degree, std::size_t row_begin, std::size_t row_end,
    std::vector<FrWire>& evaluations,
    ZkSumcheckRoundEvaluationScratch& scratch, std::string* error,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan) {
    if (row_begin > row_end || current.empty() || current.front().size() < 2U ||
        (current.front().size() & 1U) != 0U || degree > 8U ||
        evaluations.size() != degree + 1U ||
        row_end > current.front().size() / 2U) {
        if (error) *error = "segmented ZkSumcheck Montgomery cached round shape is invalid";
        return false;
    }
    for (const auto& tensor : current)
        if (tensor.size() != current.front().size()) {
            if (error) *error = "segmented ZkSumcheck Montgomery cached tensor shapes differ";
            return false;
        }
    const std::size_t row_count = row_end - row_begin;
    if (row_count == 0U) return true;
    std::size_t chunk_limit = 1U;
    if (row_count >= 4096U)
        chunk_limit = std::min(row_count,
            static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    const std::size_t grain = row_count / chunk_limit +
                              (row_count % chunk_limit != 0U ? 1U : 0U);
    const std::size_t active_chunks = row_count / grain +
                                      (row_count % grain != 0U ? 1U : 0U);
    if (active_chunks == 0U) {
        if (error) *error = "segmented ZkSumcheck Montgomery cached scratch is empty";
        return false;
    }
    if (montgomery_plan != nullptr) {
        if (!montgomery_plan->enabled || montgomery_plan->degree != degree ||
            montgomery_plan->tensor_count != current.size() ||
            active_chunks > scratch.montgomery_row_state.size() ||
            active_chunks > scratch.montgomery_chunk_evaluations.size() ||
            active_chunks > scratch.montgomery_chunk_valid.size()) {
            if (error) *error = "segmented ZkSumcheck Montgomery cached plan/scratch shape is invalid";
            return false;
        }
        const std::size_t row_state_size = current.size() * 3U;
        for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
            if (scratch.montgomery_row_state[chunk].size() != row_state_size ||
                scratch.montgomery_chunk_evaluations[chunk].size() != evaluations.size()) {
                if (error) *error = "segmented ZkSumcheck Montgomery cached scratch shape is invalid";
                return false;
            }
            std::fill(scratch.montgomery_chunk_evaluations[chunk].begin(),
                      scratch.montgomery_chunk_evaluations[chunk].end(),
                      zkdl_blst::blst_fr{});
            scratch.montgomery_chunk_valid[chunk] = 1U;
        }
        zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
            const std::size_t chunk = local_row / grain;
            const std::size_t row = row_begin + local_row;
            auto& row_state = scratch.montgomery_row_state[chunk];
            auto& partial = scratch.montgomery_chunk_evaluations[chunk];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                const auto& left = current[tensor][2U * row];
                const auto& right = current[tensor][2U * row + 1U];
                row_state[tensor] = left;
                zkdl_blst::blst_fr_sub(
                    &row_state[current.size() + tensor], &right, &left);
            }
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
                const auto& x = montgomery_plan->evaluation_points[evaluation];
                for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                    zkdl_blst::blst_fr scaled{};
                    zkdl_blst::blst_fr_mul(
                        &scaled, &x, &row_state[current.size() + tensor]);
                    zkdl_blst::blst_fr_add(
                        &row_state[2U * current.size() + tensor],
                        &row_state[tensor], &scaled);
                }
                const auto value = zksc_eval_constraint_montgomery(
                    *montgomery_plan, row_state.data() + 2U * current.size());
                zkdl_blst::blst_fr next{};
                zkdl_blst::blst_fr_add(&next, &partial[evaluation], &value);
                partial[evaluation] = next;
            }
        });
        for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
            for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
                evaluations[evaluation] = zkdl::fr_add(
                    evaluations[evaluation],
                    zkdl::cpu_ipa_montgomery_to_frwire(
                        scratch.montgomery_chunk_evaluations[chunk][evaluation]));
        return true;
    }
    if (active_chunks > scratch.row_values.size() ||
        active_chunks > scratch.chunk_evaluations.size()) {
        if (error) *error = "segmented ZkSumcheck cached wire scratch shape is invalid";
        return false;
    }
    std::array<FrWire, 9U> evaluation_points{};
    for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
        evaluation_points[evaluation] = zkdl::fr_from_u32(evaluation);
    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk) {
        if (scratch.row_values[chunk].size() != current.size() ||
            scratch.chunk_evaluations[chunk].size() != evaluations.size()) {
            if (error) *error = "segmented ZkSumcheck cached wire scratch shape is invalid";
            return false;
        }
        std::fill(scratch.chunk_evaluations[chunk].begin(),
                  scratch.chunk_evaluations[chunk].end(), FrWire{});
    }
    zkdl::cpu_parallel_for(row_count, grain, [&](std::size_t local_row) {
        const std::size_t chunk = local_row / grain;
        const std::size_t row = row_begin + local_row;
        auto& values = scratch.row_values[chunk];
        auto& partial = scratch.chunk_evaluations[chunk];
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation) {
            const FrWire& x = evaluation_points[evaluation];
            for (std::size_t tensor = 0U; tensor < current.size(); ++tensor) {
                const FrWire left = zkdl::cpu_ipa_montgomery_to_frwire(current[tensor][2U * row]);
                const FrWire right = zkdl::cpu_ipa_montgomery_to_frwire(current[tensor][2U * row + 1U]);
                values[tensor] = zkdl::fr_add(
                    left, zkdl::fr_mul(x, zkdl::fr_sub(right, left)));
            }
            partial[evaluation] = zkdl::fr_add(
                partial[evaluation], eval_constraint(descriptor, values));
        }
    });
    for (std::size_t chunk = 0U; chunk < active_chunks; ++chunk)
        for (std::uint32_t evaluation = 0U; evaluation <= degree; ++evaluation)
            evaluations[evaluation] = zkdl::fr_add(
                evaluations[evaluation], scratch.chunk_evaluations[chunk][evaluation]);
    return true;
}

inline bool zksc_montgomery_folded_round_evaluations(
    const ConstraintDescriptor& descriptor,
    const ZkSumcheckFoldedState& state, std::uint32_t degree,
    std::vector<FrWire>& evaluations, std::string* error,
    std::uint32_t round_index,
    const ZkSumcheckProgressCallback* progress,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan) {
    if (state.empty() || state.front().size() < 2U ||
        (state.front().size() & 1U) != 0U) {
        if (error) *error = "segmented ZkSumcheck cached round state is invalid";
        return false;
    }
    evaluations.assign(degree + 1U, FrWire{});
    const std::size_t row_count = state.front().size() / 2U;
    constexpr std::size_t kProgressRows = 1U << 18U;
    const std::size_t scratch_rows = progress == nullptr
        ? row_count : std::min(row_count, kProgressRows);
    const std::size_t scratch_chunks = scratch_rows < 4096U ? 1U :
        std::min(scratch_rows,
                 static_cast<std::size_t>(zkdl::cpu_parallel_worker_count()) + 1U);
    ZkSumcheckRoundEvaluationScratch scratch(
        scratch_chunks, state.size(), evaluations.size(), montgomery_plan != nullptr);
    for (std::size_t row_begin = 0U; row_begin < row_count;) {
        const std::size_t row_end = row_begin + std::min(
            row_count - row_begin, progress == nullptr ? row_count : kProgressRows);
        if (!zksc_accumulate_montgomery_folded_round_evaluation_rows(
                descriptor, state, degree, row_begin, row_end, evaluations,
                scratch, error, montgomery_plan)) return false;
        row_begin = row_end;
        if (progress != nullptr)
            (*progress)("sumcheck_cached_rows", round_index,
                        std::numeric_limits<std::uint32_t>::max(), row_end,
                        row_count);
    }
    return true;
}
#endif

inline bool zksc_folded_round_evaluations(
    const ConstraintDescriptor& descriptor,
    const ZkSumcheckFoldedState& state,
    std::uint32_t degree, std::vector<FrWire>& evaluations,
    std::string* error = nullptr,
    std::uint32_t round_index = 0U,
    const ZkSumcheckProgressCallback* progress = nullptr,
    const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan = nullptr) {
#if defined(ZKDL_USE_BLST_G1_CODEC)
    return zksc_montgomery_folded_round_evaluations(
        descriptor, state, degree, evaluations, error, round_index, progress,
        montgomery_plan);
#else
    if (state.empty() || state.front().size() < 2U ||
        (state.front().size() & 1U) != 0U) {
        if (error) *error = "segmented ZkSumcheck cached round state is invalid";
        return false;
    }
    evaluations.assign(degree + 1U, FrWire{});
    return zksc_accumulate_round_evaluations(
        descriptor, state, degree, evaluations, error, round_index, progress,
        "sumcheck_cached_rows", montgomery_plan);
#endif
}

inline bool zksc_segmented_terminal_values(
    const std::vector<SegmentedTensorProvider>& providers,
    const std::vector<FrWire>& challenges, std::vector<FrWire>& terminal,
    std::string* error = nullptr,
    const ZkSumcheckProgressCallback* progress = nullptr) {
    if (providers.empty() || challenges.empty()) {
        if (error) *error = "segmented ZkSumcheck terminal state is empty";
        return false;
    }
    const std::uint32_t logical_domain = providers.front().logical_domain();
    const std::uint32_t segment_size = providers.front().segment_size();
    const std::uint32_t segment_count = providers.front().segment_count();
    std::uint32_t low_bits = 0U, total_bits = 0U;
    for (std::uint32_t value = segment_size; value > 1U; value >>= 1U) ++low_bits;
    for (std::uint32_t value = logical_domain; value > 1U; value >>= 1U) ++total_bits;
    if (challenges.size() != total_bits) {
        if (error) *error = "segmented ZkSumcheck terminal challenge count mismatch";
        return false;
    }
    std::vector<std::vector<FrWire>> current(providers.size(),
                                              std::vector<FrWire>(segment_count));
    for (std::uint32_t segment = 0U; segment < segment_count; ++segment) {
        for (std::size_t tensor = 0U; tensor < providers.size(); ++tensor) {
            std::vector<FrWire> block;
            if (!providers[tensor].read_segment(segment, block, error)) return false;
            for (std::uint32_t previous = 0U; previous < low_bits; ++previous)
                fold_values(block, challenges[previous]);
            if (block.empty() || segment >= current[tensor].size()) {
                if (error) *error = "segmented ZkSumcheck terminal block shape mismatch";
                return false;
            }
            current[tensor][segment] = block.front();
        }
        if (progress != nullptr)
            (*progress)("sumcheck_terminal_segments", total_bits - 1U,
                        std::numeric_limits<std::uint32_t>::max(),
                        segment + 1U, segment_count);
    }
    for (std::uint32_t previous = low_bits; previous < total_bits; ++previous)
        for (auto& tensor : current) fold_values(tensor, challenges[previous]);
    terminal.clear();
    terminal.reserve(current.size());
    for (const auto& tensor : current) terminal.push_back(tensor.front());
    return true;
}

inline bool prove_zk_sumcheck_segmented(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& tensor_providers,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, zkdl::Transcript& parent_transcript,
    ZkSumcheckProverResult& result, std::string* error = nullptr,
    std::uint32_t shared_segment_size = 0U,
    bool reuse_folded_state = false,
    const char* progress_phase = nullptr,
    const char* progress_operation_id = nullptr,
    const std::string* spill_directory = nullptr,
    bool use_montgomery_evaluator = true,
    std::uint64_t extra_live_bytes = 0U) {
    try {
        if (!zksc_validate_provider_inputs(
                descriptor, tensor_providers, degree, domain_size, generators,
                blinding_generator, u_generator, initial_claim, statement,
                operation_context, error, shared_segment_size)) return false;
        const std::uint32_t bits = log2_exact(domain_size);
        if (bits == std::numeric_limits<std::uint32_t>::max() ||
            bits == 0U || bits > zkdl::ZkSumcheckProof::kMaxRounds) {
            if (error) *error = "ZkSumcheck domain has no supported round decomposition";
            return false;
        }
        ZkSumcheckMontgomeryConstraintPlan montgomery_plan;
        const ZkSumcheckMontgomeryConstraintPlan* montgomery_plan_ptr = nullptr;
        std::uint64_t montgomery_plan_peak_bytes = 0U;
#if defined(ZKDL_USE_BLST_G1_CODEC)
        if (use_montgomery_evaluator) {
            if (!zksc_compile_montgomery_constraint_plan(
                    descriptor, tensor_providers.size(), degree,
                    montgomery_plan, error)) return false;
            montgomery_plan_ptr = &montgomery_plan;
        }
#else
        (void)use_montgomery_evaluator;
#endif
        if (!zksc_montgomery_constraint_plan_peak_bytes(
                static_cast<std::uint64_t>(descriptor.terms.size()),
                use_montgomery_evaluator, montgomery_plan_peak_bytes)) {
            if (error) *error = "Sumcheck Montgomery descriptor plan memory bound overflows";
            return false;
        }
        std::uint64_t montgomery_and_extra_peak_bytes = 0U;
        if (montgomery_plan_peak_bytes >
                std::numeric_limits<std::uint64_t>::max() - extra_live_bytes) {
            if (error) *error = "Sumcheck extra live-memory bound overflows";
            return false;
        }
        montgomery_and_extra_peak_bytes =
            montgomery_plan_peak_bytes + extra_live_bytes;
        RouteSumcheckProgressReporter route_progress_reporter{
            operation_context, bits};
        NativeOperationProgressReporter native_progress(
            progress_phase == nullptr ? std::string() : progress_phase,
            progress_phase == nullptr ? std::string() :
                (progress_operation_id == nullptr ? operation_context : progress_operation_id));
        ZkSumcheckProgressCallback route_progress_callback;
        ZkSumcheckProgressCallback native_progress_callback;
        if (reuse_folded_state) {
            route_progress_callback = [&route_progress_reporter](
                const char* phase, std::uint32_t round,
                std::uint32_t tensor_index, std::uint64_t completed,
                std::uint64_t total) {
                route_progress_reporter.update(
                    phase, round + 1U, tensor_index, completed, total);
            };
        } else if (progress_phase != nullptr) {
            native_progress_callback = [&native_progress, bits, tensor_count = tensor_providers.size()](
                const char* phase, std::uint32_t round,
                std::uint32_t tensor_index, std::uint64_t completed,
                std::uint64_t total) {
                native_progress.update(
                    phase, completed, total, tensor_index,
                    static_cast<std::uint32_t>(tensor_count), round + 1U, bits);
            };
        }
        const ZkSumcheckProgressCallback* progress = reuse_folded_state
            ? &route_progress_callback
            : progress_phase != nullptr ? &native_progress_callback : nullptr;
        bool use_spilled_folded_state = false;
        if (!reuse_folded_state && spill_directory != nullptr) {
            std::uint64_t folded_peak_bytes = 0U;
            if (spill_directory->empty() ||
                !zksc_folded_state_peak_bytes(
                    tensor_providers.size(), tensor_providers.front().logical_domain(),
                    tensor_providers.front().segment_size(), folded_peak_bytes)) {
                if (error) *error = "Sumcheck spill context or folded-state capacity is invalid";
                return false;
            }
            if (montgomery_and_extra_peak_bytes >
                    std::numeric_limits<std::uint64_t>::max() - folded_peak_bytes) {
                if (error) *error = "Sumcheck Montgomery folded-state capacity overflows";
                return false;
            }
            if (folded_peak_bytes + montgomery_and_extra_peak_bytes >
                kZkSumcheckFoldedStateBudgetBytes) {
                std::uint64_t spill_disk_bytes = 0U, spill_ram_bytes = 0U;
                if (!zksc_spilled_state_disk_peak_bytes(
                        tensor_providers.size(), tensor_providers.front().logical_domain(),
                        spill_disk_bytes) ||
                    !zksc_spilled_state_ram_peak_bytes(
                        tensor_providers.size(), tensor_providers.front().logical_domain(),
                        tensor_providers.front().segment_size(), spill_ram_bytes) ||
                    spill_ram_bytes > kZkSumcheckSpillRamBudgetBytes) {
                    if (error) *error = "Sumcheck spill resource envelope is invalid";
                    return false;
                }
                if (montgomery_and_extra_peak_bytes > kZkSumcheckSpillRamBudgetBytes ||
                    spill_ram_bytes > kZkSumcheckSpillRamBudgetBytes -
                                          montgomery_and_extra_peak_bytes) {
                    if (error) *error = "Sumcheck spill plan exceeds its RAM budget";
                    return false;
                }
                use_spilled_folded_state = true;
            }
        }
        result = ZkSumcheckProverResult{};
        auto& proof = result.proof;
        proof.label = operation_context;
        proof.degree = degree;
        proof.domain_size = domain_size;
        proof.segment_size = tensor_providers.front().segment_size();
        proof.segment_count = tensor_providers.front().segment_count();
        proof.descriptor_digest = zksc_proof_descriptor_digest(
            descriptor, statement, operation_context);
        proof.registry_digest = statement.registry_digest;
        proof.srs_digest = statement.srs_digest;
        const std::vector<zkdl::CpuG1Point> round_generators(
            generators.begin(), generators.begin() + degree + 1U);
        proof.basis_digest = zkdl::cpu_ipa_basis_digest(
            round_generators, blinding_generator, u_generator, degree + 1U);
        proof.blinding_generator = zkdl::cpu_g1_to_native(blinding_generator);
        proof.u_generator = zkdl::cpu_g1_to_native(u_generator);
        proof.initial_claim_context = zksc_make_scalar_context(
            descriptor, statement, operation_context, "initial-public-claim", 0U,
            zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        proof.initial_claim_context.descriptor_digest = proof.descriptor_digest;
        proof.initial_claim_context.registry_digest = proof.registry_digest;
        proof.initial_claim_context.srs_digest = proof.srs_digest;

        zksc_absorb_header(parent_transcript, proof);
        FrWire previous_value = initial_claim;
        FrWire previous_blinder{};
        std::vector<FrWire> prior_challenges;
        prior_challenges.reserve(bits);
        ZkSumcheckFoldedState folded_state;
        ZkSumcheckFoldedSpillFile spilled_folded_state;
        for (std::uint32_t round_index = 0U; round_index < bits; ++round_index) {
            std::vector<FrWire> evaluations;
            const bool evaluated = reuse_folded_state && round_index != 0U
                ? zksc_folded_round_evaluations(
                      descriptor, folded_state, degree, evaluations, error,
                      round_index, progress, montgomery_plan_ptr)
                : use_spilled_folded_state && round_index != 0U
                ? zksc_spilled_round_evaluations(
                      descriptor, spilled_folded_state, degree, evaluations, error,
                      round_index, progress, montgomery_plan_ptr)
                : zksc_segmented_round_evaluations(
                      descriptor, tensor_providers, degree, round_index,
                      prior_challenges, evaluations, error, progress,
                      montgomery_plan_ptr);
            if (!evaluated) return false;
            std::vector<FrWire> coefficients;
            if (!zksc_interpolate_coefficients(evaluations, coefficients, error)) return false;
            const FrWire endpoint_value = zksc_eval_coefficients(
                coefficients, zkdl::fr_from_u32(0U));
            const FrWire endpoint_at_one = zksc_eval_coefficients(
                coefficients, zkdl::fr_from_u32(1U));
            const FrWire endpoint = zkdl::fr_add(endpoint_value, endpoint_at_one);
            const FrWire r_c = zkdl::cpu_fresh_csprng_scalar();
            const FrWire r_e = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point coefficient_point = zkdl::cpu_g1_add(
                zkdl::cpu_g1_sum(round_generators, coefficients),
                zkdl::cpu_g1_mul(blinding_generator, r_c));
            const zkdl::CpuG1Point endpoint_commitment = zkdl::cpu_pedersen_scalar_commit(
                endpoint, r_e, blinding_generator, u_generator);
            const std::array<byte, 32U> endpoint_digest = zksc_vector_role_digest(
                descriptor, statement, operation_context, "endpoint", round_index,
                zksc_endpoint_index(round_index));
            const auto endpoint_context = zksc_make_context(
                zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
                zksc_round_commitment_index(round_index), zksc_endpoint_index(round_index),
                proof.basis_digest, endpoint_digest, statement);
            std::vector<FrWire> endpoint_weights(degree + 1U, zkdl::fr_from_u32(1U));
            endpoint_weights[0] = zkdl::fr_from_u32(2U);
            auto endpoint_linear = zkdl::cpu_open_pedersen_vector_linear(
                round_generators, coefficients, r_c, endpoint_weights, r_e,
                coefficient_point, endpoint_commitment, blinding_generator,
                u_generator, endpoint_context);

            zkdl::ZkSumcheckRoundProof round;
            round.round_index = round_index;
            round.coefficient_commitment = zkdl::cpu_g1_to_native(coefficient_point);
            round.endpoint_commitment = zkdl::cpu_g1_to_native(endpoint_commitment);
            round.endpoint_linear = endpoint_linear.proof;
            if (round_index == 0U) {
                round.link_kind = zkdl::ZkSumcheckLinkKind::InitialPublicClaim;
                const auto initial_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "initial-public-claim", 0U,
                    zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
                    zkdl::PedersenProofContext::kOperationScalarValue,
                    blinding_generator, u_generator);
                round.initial_claim_binding = zkdl::cpu_open_pedersen_scalar_value_public(
                    initial_claim, r_e, endpoint_commitment, blinding_generator,
                    u_generator, initial_context).proof;
            } else {
                round.link_kind = zkdl::ZkSumcheckLinkKind::PreviousEvaluation;
                const auto previous_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "previous-evaluation",
                    round_index, zksc_endpoint_index(round_index),
                    zksc_evaluation_index(round_index - 1U),
                    zkdl::PedersenProofContext::kOperationScalarEquality,
                    blinding_generator, u_generator);
                round.previous_claim_binding = zkdl::cpu_open_pedersen_scalar_equality(
                    previous_value, r_e, previous_blinder, endpoint_commitment,
                    zkdl::cpu_g1_from_native(proof.rounds.back().evaluation_commitment),
                    blinding_generator, u_generator, previous_context).proof;
            }
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.pre",
                zksc_round_pre_bytes(round));
            const Fr_t challenge = parent_transcript.challenge(
                "gat.zksumcheck." + operation_context + ".round." +
                std::to_string(round_index) + ".challenge");
            const FrWire challenge_wire = zkdl::fr_wire(challenge);
            std::vector<FrWire> query_weights(degree + 1U, zkdl::fr_from_u32(1U));
            for (std::uint32_t power = 1U; power <= degree; ++power)
                query_weights[power] = zkdl::fr_mul(query_weights[power - 1U], challenge_wire);
            const FrWire query_value = zksc_eval_coefficients(coefficients, challenge_wire);
            const FrWire r_q = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point query_commitment = zkdl::cpu_pedersen_scalar_commit(
                query_value, r_q, blinding_generator, u_generator);
            const auto query_context = zksc_make_vector_context(
                descriptor, statement, operation_context, "evaluation", round_index,
                zksc_evaluation_index(round_index), proof.basis_digest, degree);
            auto query_linear = zkdl::cpu_open_pedersen_vector_linear(
                round_generators, coefficients, r_c, query_weights, r_q,
                coefficient_point, query_commitment, blinding_generator,
                u_generator, query_context);
            round.evaluation_commitment = zkdl::cpu_g1_to_native(query_commitment);
            round.evaluation_linear = query_linear.proof;
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.query",
                zksc_round_query_bytes(round));
            proof.rounds.push_back(round);
            previous_value = query_value;
            previous_blinder = r_q;
            if (reuse_folded_state) {
                if (round_index == 0U) {
                    if (!zksc_segmented_fold_state_initialize(
                            tensor_providers, challenge_wire, folded_state,
                            error, round_index + 1U, progress)) return false;
                } else {
                    for (std::size_t tensor = 0U; tensor < folded_state.size(); ++tensor) {
                        std::uint32_t tensor_index = 0U;
                        if (!checked_relation_u32(tensor, tensor_index,
                                                  "route Sumcheck tensor index", error) ||
                            !zksc_fold_values_in_place(
                                folded_state[tensor], challenge_wire, error,
                                round_index + 1U, tensor_index, progress))
                            return false;
                    }
                }
            } else if (use_spilled_folded_state) {
                if (round_index == 0U) {
                    if (!zksc_spilled_fold_state_initialize(
                            tensor_providers, challenge_wire, operation_context,
                            *spill_directory, round_index + 1U,
                            spilled_folded_state, error, progress)) return false;
                } else if (!zksc_spilled_fold_state(
                               challenge_wire, spilled_folded_state,
                               round_index + 1U, error, progress)) return false;
            }
            prior_challenges.push_back(challenge_wire);
            result.challenges.push_back(challenge);
        }
        proof.final_claim_commitment = proof.rounds.back().evaluation_commitment;
        if (reuse_folded_state) {
            result.folded_terminal_values.clear();
            result.folded_terminal_values.reserve(folded_state.size());
            for (const auto& tensor : folded_state) {
                if (tensor.size() != 1U) {
                    if (error) *error = "segmented ZkSumcheck cached terminal shape is invalid";
                    return false;
                }
                result.folded_terminal_values.push_back(
                    zksc_folded_scalar_to_wire(tensor.front()));
            }
        } else if (use_spilled_folded_state) {
            if (!zksc_spilled_terminal_values(
                    spilled_folded_state, bits, result.folded_terminal_values, error))
                return false;
        } else if (!zksc_segmented_terminal_values(
                       tensor_providers, prior_challenges,
                       result.folded_terminal_values, error, progress)) return false;
        result.final_q_blinder = previous_blinder;
        std::vector<byte> final_bytes;
        zkdl::append_g1(final_bytes, proof.final_claim_commitment);
        parent_transcript.absorb("gat.zksumcheck." + operation_context + ".final", final_bytes);
        std::string shape_error;
        if (!zkdl::zk_sumcheck_shape_valid(proof, &shape_error)) {
            if (error) *error = shape_error.empty() ? "invalid generated ZkSumcheck proof" : shape_error;
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool prove_zk_sumcheck(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, zkdl::Transcript& parent_transcript,
    ZkSumcheckProverResult& result, std::string* error = nullptr,
    const char* progress_phase = nullptr,
    const char* progress_operation_id = nullptr,
    bool reuse_folded_state = false,
    bool use_montgomery_evaluator = true,
    std::uint64_t extra_live_bytes = 0U) {
    std::vector<SegmentedTensorProvider> providers;
    if (!make_segmented_tensor_providers(tensor_values, generators.size(), providers, error))
        return false;
    return prove_zk_sumcheck_segmented(
        descriptor, providers, degree, domain_size, generators,
        blinding_generator, u_generator, initial_claim, statement,
        operation_context, parent_transcript, result, error, 0U,
        reuse_folded_state, progress_phase, progress_operation_id, nullptr,
        use_montgomery_evaluator, extra_live_bytes);
}

inline bool verify_zk_sumcheck(
    const zkdl::ZkSumcheckProof& proof, const ConstraintDescriptor& descriptor,
    std::uint32_t degree, std::uint32_t domain_size,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    const FrWire& initial_claim, const RelationStatement& statement,
    const std::string& operation_context, zkdl::Transcript& parent_transcript,
    ZkSumcheckVerifierResult& result, std::string* error = nullptr,
    std::uint32_t shared_segment_size = 0U) {
    try {
        if (!descriptor.valid(error) || proof.label != operation_context ||
            proof.degree != degree || proof.domain_size != domain_size ||
            !zkdl::zk_sumcheck_shape_valid(proof, error) ||
            !zksc_statement_resources_valid(statement, error) ||
            !zkdl::fr_canonical(initial_claim) || generators.size() < degree + 1U) {
            if (error && error->empty()) *error = "invalid ZkSumcheck verifier inputs";
            return false;
        }
        ZkRelationSegmentLayout expected_layout;
        const bool layout_valid = shared_segment_size == 0U
            ? zktr_segment_layout(domain_size, generators.size(), expected_layout, error)
            : zktr_segment_layout_for_size(domain_size, generators.size(),
                                           shared_segment_size, expected_layout, error);
        if (!layout_valid || proof.segment_size != expected_layout.segment_size ||
            proof.segment_count != expected_layout.segment_count) {
            if (error && error->empty()) *error = "ZkSumcheck shared segment layout mismatch";
            return false;
        }
        const auto strict_external = [&](const G1Jacobian_t& wire,
                                         const zkdl::CpuG1Point& expected,
                                         const char* label) {
            std::array<byte, 48U> encoded{};
            std::string point_error;
            if (!zkdl_g1_wire::encode(wire, encoded, &point_error) ||
                !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(wire), expected)) {
                if (error) *error = std::string("ZkSumcheck ") + label + " basis mismatch";
                return false;
            }
            return true;
        };
        if (!strict_external(proof.blinding_generator, blinding_generator, "H") ||
            !strict_external(proof.u_generator, u_generator, "U")) return false;
        const std::vector<zkdl::CpuG1Point> round_generators(
            generators.begin(), generators.begin() + degree + 1U);
        if (!zkdl::cpu_pedersen_basis_valid(round_generators, blinding_generator,
                                            u_generator, degree + 1U, error) ||
            proof.basis_digest != zkdl::cpu_ipa_basis_digest(
                round_generators, blinding_generator, u_generator, degree + 1U) ||
            proof.descriptor_digest != zksc_proof_descriptor_digest(
                descriptor, statement, operation_context) ||
            proof.registry_digest != statement.registry_digest ||
            proof.srs_digest != statement.srs_digest) {
            if (error && error->empty()) *error = "ZkSumcheck descriptor/resource basis mismatch";
            return false;
        }
        const auto expected_initial_context = zksc_make_scalar_context(
            descriptor, statement, operation_context, "initial-public-claim", 0U,
            zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        if (!zkdl::cpu_pedersen_context_equal(
                proof.initial_claim_context, expected_initial_context)) {
            if (error) *error = "ZkSumcheck initial claim context mismatch";
            return false;
        }
        result = ZkSumcheckVerifierResult{};
        zksc_absorb_header(parent_transcript, proof);
        if (proof.rounds.size() > kMaxRelationPointBits) {
            if (error) *error = "ZkSumcheck proof round count exceeds native point cap";
            return false;
        }
        for (std::size_t round_index = 0U; round_index < proof.rounds.size(); ++round_index) {
            std::uint32_t round_index_u32 = 0U;
            if (!checked_relation_u32(round_index, round_index_u32,
                                      "ZkSumcheck round index", error))
                return false;
            const auto& round = proof.rounds[round_index];
            const auto endpoint_context = zksc_make_context(
                zkdl::PedersenProofContext::kOperationVectorLinear, degree, degree + 1U,
                zksc_round_commitment_index(round_index_u32),
                zksc_endpoint_index(round_index_u32),
                proof.basis_digest,
                zksc_vector_role_digest(descriptor, statement, operation_context,
                                        "endpoint", round_index_u32,
                                        zksc_endpoint_index(round_index_u32)),
                statement);
            std::vector<FrWire> endpoint_weights(degree + 1U, zkdl::fr_from_u32(1U));
            endpoint_weights[0] = zkdl::fr_from_u32(2U);
            if (!zkdl::cpu_verify_pedersen_vector_linear(
                    round.endpoint_linear,
                    zkdl::cpu_g1_from_native(round.coefficient_commitment),
                    zkdl::cpu_g1_from_native(round.endpoint_commitment), round_generators,
                    endpoint_weights, blinding_generator, u_generator, endpoint_context,
                    error)) return false;
            if (round_index == 0U) {
                const auto initial_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "initial-public-claim", 0U,
                    zksc_endpoint_index(0U), zkdl::PedersenProofContext::kUnsetIndex,
                    zkdl::PedersenProofContext::kOperationScalarValue,
                    blinding_generator, u_generator);
                if (!zkdl::cpu_verify_pedersen_scalar_value(
                        round.initial_claim_binding, initial_claim,
                        zkdl::cpu_g1_from_native(round.endpoint_commitment),
                        blinding_generator, u_generator, initial_context, error)) return false;
            } else {
                const auto previous_context = zksc_make_scalar_context(
                    descriptor, statement, operation_context, "previous-evaluation",
                    round_index_u32,
                    zksc_endpoint_index(round_index_u32),
                    zksc_evaluation_index(round_index_u32 - 1U),
                    zkdl::PedersenProofContext::kOperationScalarEquality,
                    blinding_generator, u_generator);
                if (!zkdl::cpu_verify_pedersen_scalar_equality(
                        round.previous_claim_binding,
                        zkdl::cpu_g1_from_native(round.endpoint_commitment),
                        zkdl::cpu_g1_from_native(
                            proof.rounds[round_index - 1U].evaluation_commitment),
                        blinding_generator, u_generator, previous_context, error)) return false;
            }
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.pre",
                zksc_round_pre_bytes(round));
            const Fr_t challenge = parent_transcript.challenge(
                "gat.zksumcheck." + operation_context + ".round." +
                std::to_string(round_index) + ".challenge");
            const FrWire challenge_wire = zkdl::fr_wire(challenge);
            std::vector<FrWire> query_weights(degree + 1U, zkdl::fr_from_u32(1U));
            for (std::uint32_t power = 1U; power <= degree; ++power)
                query_weights[power] = zkdl::fr_mul(query_weights[power - 1U], challenge_wire);
            const auto query_context = zksc_make_vector_context(
                descriptor, statement, operation_context, "evaluation",
                round_index_u32,
                zksc_evaluation_index(round_index_u32),
                proof.basis_digest, degree);
            if (!zkdl::cpu_verify_pedersen_vector_linear(
                    round.evaluation_linear,
                    zkdl::cpu_g1_from_native(round.coefficient_commitment),
                    zkdl::cpu_g1_from_native(round.evaluation_commitment), round_generators,
                    query_weights, blinding_generator, u_generator, query_context,
                    error)) return false;
            parent_transcript.absorb(
                "gat.zksumcheck." + operation_context + ".round.query",
                zksc_round_query_bytes(round));
            result.challenges.push_back(challenge);
        }
        result.final_claim_commitment = proof.final_claim_commitment;
        std::vector<byte> final_bytes;
        zkdl::append_g1(final_bytes, proof.final_claim_commitment);
        parent_transcript.absorb("gat.zksumcheck." + operation_context + ".final", final_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

/*
 * Hidden terminal relation proof.
 *
 * This seam is intentionally independent from ConstraintProof.  It commits
 * each descriptor tensor's MLE evaluation as T_i, opens the canonical tensor
 * commitment with Hyrax v4, and reduces every higher-degree monomial through
 * the deterministic Pedersen product DAG.  No terminal value or blinder is
 * serialized; the returned prover result keeps those scalars local for the
 * caller that owns the canonical witness.
 */
struct ZkTerminalRelationProverResult {
    zkdl::ZkTerminalRelationProof proof;
    std::vector<FrWire> terminal_values;
    std::vector<FrWire> terminal_blinders;
    std::vector<FrWire> product_blinders;
    FrWire residual_blinder{};
};

struct ZkTerminalRelationVerifierResult {
    zkdl::CpuG1Point residual_commitment;
    std::vector<zkdl::CpuG1Point> terminal_commitments;
};

struct ZkTerminalProductPlan {
    std::uint32_t monomial_index = 0U;
    std::uint32_t factor_step = 0U;
    std::uint32_t left_commitment_index = 0U;
    std::uint32_t right_commitment_index = 0U;
    std::uint32_t output_commitment_index = 0U;
    // A repeated terminal factor (x*x) gets a deterministic role alias for
    // the existing product proof. The alias is never an independent claim:
    // prover and verifier copy the already authenticated/opened ScalarState.
    std::uint32_t right_alias_source_index =
        std::numeric_limits<std::uint32_t>::max();
};

constexpr std::uint32_t kZkTerminalEvaluationIndexBase = 0x40000000U;
constexpr std::uint32_t kZkTerminalProductIndexBase = 0x80000000U;
constexpr std::uint32_t kZkTerminalProductAliasIndexBase = 0x90000000U;
constexpr std::uint32_t kZkTerminalResidualIndex = 0xF0000000U;
constexpr std::uint32_t kZkTerminalExternalIndex = 0xF0000001U;

static_assert(kZkTerminalProductIndexBase +
                  zkdl::ZkTerminalRelationProof::kMaxProductNodes <=
              kZkTerminalProductAliasIndexBase,
              "terminal product and alias index ranges overlap");
static_assert(kZkTerminalProductAliasIndexBase +
                  zkdl::ZkTerminalRelationProof::kMaxProductNodes <
              kZkTerminalResidualIndex,
              "terminal alias index range overlaps reserved residual roles");

inline std::uint32_t zktr_terminal_evaluation_index(std::size_t tensor_index) {
    std::uint32_t checked_index = 0U;
    if (tensor_index >= 0x40000000U ||
        !checked_relation_u32(tensor_index, checked_index,
                              "terminal evaluation index"))
        throw std::invalid_argument("terminal evaluation commitment index overflow");
    return kZkTerminalEvaluationIndexBase + checked_index;
}

inline std::uint32_t zktr_product_index(std::size_t node_index) {
    std::uint32_t checked_index = 0U;
    if (node_index >= 0x70000000U ||
        !checked_relation_u32(node_index, checked_index,
                              "terminal product index"))
        throw std::invalid_argument("terminal product commitment index overflow");
    return kZkTerminalProductIndexBase + checked_index;
}

inline bool zktr_product_alias_index(std::size_t node_index,
                                     std::uint32_t& alias_index,
                                     std::string* error = nullptr) {
    if (node_index >= zkdl::ZkTerminalRelationProof::kMaxProductNodes ||
        !checked_relation_u32(node_index, alias_index,
                              "terminal product alias index")) {
        if (error && error->empty())
            *error = "terminal product alias index exceeds its reserved range";
        return false;
    }
    alias_index += kZkTerminalProductAliasIndexBase;
    if (alias_index >= kZkTerminalResidualIndex) {
        if (error) *error = "terminal product alias collides with a reserved role";
        return false;
    }
    return true;
}

inline bool zktr_all_statement_digests_valid(const RelationStatement& statement,
                                             std::string* error = nullptr) {
    const std::array<byte, 32U> zero{{}};
    if (statement.statement_digest == zero || statement.model_digest == zero ||
        statement.table_digest == zero || statement.ledger_digest == zero ||
        statement.registry_digest == zero || statement.srs_digest == zero) {
        if (error) *error = "terminal relation requires authenticated statement resources";
        return false;
    }
    return true;
}

struct SegmentCommitmentMaterial {
    std::vector<FrWire> blinders;
    std::vector<zkdl::CpuG1Point> commitments;
};

struct QpadGroupPlan;
struct QpadGroupProverContext;
class QpadAggregateReadState;
class SegmentCommitmentValidationReceipt;
inline bool prove_padding_group(
    const QpadGroupPlan&, const QpadGroupProverContext&,
    zkdl::Transcript&, PaddingProofRecord&, std::string*);
struct QpadAggregateReceiptFactory;

inline void zktr_update_fr_wire_digest(
    zkdl::Sha256& hash, const FrWire* values, std::size_t count) {
    std::array<byte, 4096U> encoded{};
    std::size_t used = 0U;
    for (std::size_t index = 0U; index < count; ++index) {
        const auto& value = values[index];
        for (const std::uint32_t limb : value) {
            for (std::size_t octet = 0U; octet < 4U; ++octet)
                encoded[used++] = static_cast<byte>(limb >> (8U * octet));
        }
        if (used == encoded.size()) {
            hash.update(encoded.data(), used);
            used = 0U;
        }
    }
    if (used != 0U) hash.update(encoded.data(), used);
}

inline std::array<byte, 32U> zktr_segment_scalar_digest(
    const FrWire* values, std::size_t count) {
    zkdl::Sha256 hash;
    zktr_update_fr_wire_digest(hash, values, count);
    return hash.final();
}

inline std::array<byte, 32U> zktr_segment_scalar_digest(
    const std::vector<FrWire>& values) {
    return zktr_segment_scalar_digest(values.data(), values.size());
}

inline std::array<byte, 32U> zktr_segment_receipt_header_digest(
    const std::string& tensor_id, std::uint32_t visibility,
    const SegmentedTensorProvider& provider, const CommitmentWire& commitment) {
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.segment-msm-receipt.v1");
    zkdl::append_string(encoded, tensor_id);
    zkdl::append_u32(encoded, visibility);
    zkdl::append_u32(encoded, provider.logical_domain());
    zkdl::append_u32(encoded, provider.segment_size());
    zkdl::append_u32(encoded, provider.segment_count());
    zkdl::append_u32(encoded, commitment.segment_binding_version);
    zkdl::append_u32(encoded, commitment.domain_size);
    zkdl::append_u32(encoded, commitment.segment_size);
    zkdl::append_u32(encoded, commitment.segment_count);
    append_digest(encoded, commitment.srs_manifest_digest);
    append_digest(encoded, commitment.basis_digest);
    append_digest(encoded, commitment.segment_commitment_digest);
    return zkdl::sha256(encoded);
}

inline std::array<byte, 32U> zktr_segment_receipt_material_digest_fields(
    const std::string& tensor_id, std::uint32_t visibility,
    std::uint32_t domain_size, const std::array<byte, 32U>& basis_digest,
    std::uint32_t segment, const zkdl::CpuG1Point& commitment,
    const FrWire& blinder) {
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.segment-msm-material.v1");
    zkdl::append_string(encoded, tensor_id);
    zkdl::append_u32(encoded, visibility);
    zkdl::append_u32(encoded, domain_size);
    zkdl::append_u32(encoded, segment);
    append_digest(encoded, basis_digest);
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(commitment));
    append_fr_wire(encoded, blinder);
    return zkdl::sha256(encoded);
}

inline std::array<byte, 32U> zktr_segment_receipt_material_digest(
    const std::string& tensor_id, std::uint32_t visibility,
    std::uint32_t domain_size, const std::array<byte, 32U>& basis_digest,
    const SegmentCommitmentMaterial& material, std::uint32_t segment) {
    return zktr_segment_receipt_material_digest_fields(
        tensor_id, visibility, domain_size, basis_digest, segment,
        material.commitments.at(segment), material.blinders.at(segment));
}

class SegmentCommitmentValidationReceipt {
public:
    SegmentCommitmentValidationReceipt() = default;
    bool valid() const noexcept { return valid_; }
    std::size_t segment_count() const noexcept { return scalar_digests_.size(); }

    bool matches_context(
        const std::string& tensor_id, std::uint32_t visibility,
        const SegmentedTensorProvider& provider,
        const std::vector<zkdl::CpuG1Point>& generators,
        const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
        const zkdl::CpuIpaBasisContext* authenticated_basis,
        const CommitmentWire& commitment,
        const SegmentCommitmentMaterial& material,
        std::string* error = nullptr) const {
        if (!valid_ || authenticated_basis == nullptr ||
            !segment_commitment_wire_shape_valid(commitment, error) ||
            !authenticated_basis->owns_exact_generators(generators, h, u) ||
            !basis_.owns_exact_generators(generators, h, u) ||
            !basis_.owns_exact_generators(authenticated_basis->generators(),
                                          authenticated_basis->h(),
                                          authenticated_basis->u()) ||
            tensor_id != commitment.tensor_id || visibility != commitment.visibility ||
            provider.logical_domain() != commitment.domain_size ||
            provider.segment_size() != commitment.segment_size ||
            provider.segment_count() != commitment.segment_count ||
            material.blinders.size() != scalar_digests_.size() ||
            material.commitments.size() != scalar_digests_.size() ||
            commitment.segment_commitments.size() != scalar_digests_.size() ||
            !zkdl::cpu_g1_equal(
                aggregate_commitment_,
                zkdl::cpu_g1_from_native(commitment.commitment)) ||
            header_digest_ != zktr_segment_receipt_header_digest(
                tensor_id, visibility, provider, commitment)) {
            if (error) *error = "segment MSM receipt does not match authenticated basis/provider/wire";
            return false;
        }
        for (std::uint32_t segment = 0U; segment < scalar_digests_.size(); ++segment) {
            if (!zkdl::fr_canonical(material.blinders[segment]) ||
                !zkdl::cpu_g1_equal(
                    material.commitments[segment],
                    zkdl::cpu_g1_from_native(
                        commitment.segment_commitments[segment].commitment)) ||
                zktr_segment_receipt_material_digest(
                    tensor_id, visibility, commitment.domain_size,
                    commitment.basis_digest, material, segment) !=
                    material_digests_[segment]) {
                if (error) *error = "segment MSM receipt material binding mismatch";
                return false;
            }
        }
        return true;
    }

    bool matches_segment(std::uint32_t segment,
                         const std::vector<FrWire>& values,
                         const SegmentCommitmentMaterial& material,
                         std::string* error = nullptr) const {
        if (!valid_ || segment >= scalar_digests_.size() ||
            segment >= material.blinders.size() ||
            segment >= material.commitments.size() ||
            zktr_segment_receipt_material_digest(
                tensor_id_, visibility_, domain_size_, basis_digest_, material,
                segment) != material_digests_[segment]) {
            if (error) *error = "segment MSM receipt witness/material digest mismatch";
            return false;
        }
        const bool values_match = derived_segment_validator_
            ? derived_segment_validator_(segment, values, material, error)
            : zktr_segment_scalar_digest(values) == scalar_digests_[segment];
        if (!values_match) {
            if (error && error->empty())
                *error = "segment MSM receipt witness/material digest mismatch";
            return false;
        }
        return true;
    }

    bool matches_segment(std::uint32_t segment,
                         const std::vector<FrWire>& values,
                         const FrWire& segment_blinder,
                         const zkdl::CpuG1Point& segment_commitment,
                         std::string* error = nullptr) const {
        if (!valid_ || segment >= scalar_digests_.size() ||
            !zkdl::fr_canonical(segment_blinder) ||
            zktr_segment_receipt_material_digest_fields(
                tensor_id_, visibility_, domain_size_, basis_digest_, segment,
                segment_commitment, segment_blinder) != material_digests_[segment] ||
            zktr_segment_scalar_digest(values) != scalar_digests_[segment]) {
            if (error) *error = "segment MSM receipt witness/material digest mismatch";
            return false;
        }
        return true;
    }

    bool matches_opening_context(
        const zkdl::CpuIpaBasisContext& authenticated_basis,
        const std::vector<zkdl::CpuG1Point>& generators,
        const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
        std::uint32_t domain_size, std::uint32_t segment_size,
        std::uint32_t segment_count, const zkdl::CpuG1Point& aggregate,
        const SegmentCommitmentMaterial& material,
        std::string* error = nullptr) const {
        if (!valid_ ||
            !authenticated_basis.owns_exact_generators(generators, h, u) ||
            !basis_.owns_exact_generators(generators, h, u) ||
            !basis_.owns_exact_generators(authenticated_basis.generators(),
                                          authenticated_basis.h(),
                                          authenticated_basis.u()) ||
            domain_size != domain_size_ || segment_size != segment_size_ ||
            segment_count != segment_count_ || material.blinders.size() != segment_count_ ||
            material.commitments.size() != segment_count_ ||
            !zkdl::cpu_g1_equal(aggregate, aggregate_commitment_)) {
            if (error) *error = "segment MSM receipt does not match opening context";
            return false;
        }
        for (std::uint32_t segment = 0U; segment < segment_count_; ++segment) {
            if (zktr_segment_receipt_material_digest(
                    tensor_id_, visibility_, domain_size_, basis_digest_, material,
                    segment) != material_digests_[segment]) {
                if (error) *error = "segment MSM receipt opening material mismatch";
                return false;
            }
        }
        return true;
    }

private:
    friend struct SegmentCommitmentValidationReceiptFactory;
    friend struct QpadAggregateReceiptFactory;
    bool valid_ = false;
    zkdl::CpuIpaBasisContext basis_;
    std::string tensor_id_;
    std::uint32_t visibility_ = 0U;
    std::uint32_t domain_size_ = 0U;
    std::uint32_t segment_size_ = 0U;
    std::uint32_t segment_count_ = 0U;
    std::array<byte, 32U> basis_digest_{};
    std::array<byte, 32U> header_digest_{};
    zkdl::CpuG1Point aggregate_commitment_;
    std::vector<std::array<byte, 32U>> scalar_digests_;
    std::vector<std::array<byte, 32U>> material_digests_;
    std::function<bool(std::uint32_t, const std::vector<FrWire>&,
                       const SegmentCommitmentMaterial&, std::string*)>
        derived_segment_validator_;
};

// Holds the one-pass aggregate-Y scratch receipt state for a Qpad group.
class QpadAggregateReadState {
private:
    friend struct QpadAggregateReceiptFactory;
    friend bool prove_padding_group(
        const QpadGroupPlan&, const QpadGroupProverContext&,
        zkdl::Transcript&, PaddingProofRecord&, std::string*);
    bool source_members_validated_ = false;
    std::uint32_t last_segment_ = std::numeric_limits<std::uint32_t>::max();
    bool last_source_members_valid_ = false;
    std::array<byte, 32U> last_aggregate_digest_{};
    std::vector<std::array<byte, 32U>> aggregate_segment_digests_;
};

struct QpadAggregateReceiptFactory {
private:
    static SegmentCommitmentValidationReceipt mint(
        const zkdl::CpuIpaBasisContext& basis,
        const std::string& tensor_id, std::uint32_t visibility,
        const SegmentedTensorProvider& provider,
        const CommitmentWire& commitment,
        const SegmentCommitmentMaterial& material,
        const std::shared_ptr<QpadAggregateReadState>& read_state);
    friend bool prove_padding_group(
        const QpadGroupPlan&, const QpadGroupProverContext&,
        zkdl::Transcript&, PaddingProofRecord&, std::string*);
};

inline SegmentCommitmentValidationReceipt QpadAggregateReceiptFactory::mint(
    const zkdl::CpuIpaBasisContext& basis,
    const std::string& tensor_id, std::uint32_t visibility,
    const SegmentedTensorProvider& provider,
    const CommitmentWire& commitment,
    const SegmentCommitmentMaterial& material,
    const std::shared_ptr<QpadAggregateReadState>& read_state) {
    if (!basis.valid() || !read_state || !read_state->source_members_validated_ ||
        read_state->aggregate_segment_digests_.size() != commitment.segment_count ||
        tensor_id.empty() || visibility == 0U ||
        !provider.valid() || !segment_commitment_wire_shape_valid(commitment) ||
        commitment.tensor_id != tensor_id || commitment.visibility != visibility ||
        provider.logical_domain() != commitment.domain_size ||
        provider.segment_size() != commitment.segment_size ||
        provider.segment_count() != commitment.segment_count ||
        material.blinders.size() != commitment.segment_count ||
        material.commitments.size() != commitment.segment_count ||
        commitment.segment_commitments.size() != commitment.segment_count ||
        segment_commitment_digest(commitment) !=
            commitment.segment_commitment_digest)
        throw std::invalid_argument("cannot mint invalid Qpad aggregate receipt");
    zkdl::CpuG1Point segment_sum = zkdl::CpuG1Point::zero();
    for (std::uint32_t segment = 0U; segment < commitment.segment_count; ++segment) {
        if (!zkdl::fr_canonical(material.blinders[segment]) ||
            !zkdl::cpu_g1_equal(
                material.commitments[segment],
                zkdl::cpu_g1_from_native(
                    commitment.segment_commitments[segment].commitment)))
            throw std::invalid_argument("Qpad aggregate receipt material differs from its wire");
        segment_sum = zkdl::cpu_g1_add(segment_sum, material.commitments[segment]);
    }
    if (!zkdl::cpu_g1_equal(
            segment_sum, zkdl::cpu_g1_from_native(commitment.commitment)))
        throw std::invalid_argument("Qpad aggregate wire is not the segment commitment sum");
    SegmentCommitmentValidationReceipt receipt;
    receipt.basis_ = basis;
    receipt.tensor_id_ = tensor_id;
    receipt.visibility_ = visibility;
    receipt.domain_size_ = commitment.domain_size;
    receipt.segment_size_ = commitment.segment_size;
    receipt.segment_count_ = commitment.segment_count;
    receipt.basis_digest_ = commitment.basis_digest;
    receipt.aggregate_commitment_ =
        zkdl::cpu_g1_from_native(commitment.commitment);
    receipt.header_digest_ = zktr_segment_receipt_header_digest(
        tensor_id, visibility, provider, commitment);
    receipt.scalar_digests_.assign(commitment.segment_count, {});
    receipt.material_digests_.reserve(commitment.segment_count);
    for (std::uint32_t segment = 0U;
         segment < commitment.segment_count; ++segment)
        receipt.material_digests_.push_back(
            zktr_segment_receipt_material_digest(
                tensor_id, visibility, commitment.domain_size,
                commitment.basis_digest, material, segment));
    receipt.derived_segment_validator_ =
        [read_state](std::uint32_t segment, const std::vector<FrWire>& values,
                     const SegmentCommitmentMaterial&, std::string* error) {
            if (!read_state->source_members_validated_ ||
                !read_state->last_source_members_valid_ ||
                read_state->last_segment_ != segment ||
                segment >= read_state->aggregate_segment_digests_.size() ||
                read_state->aggregate_segment_digests_[segment] !=
                    read_state->last_aggregate_digest_ ||
                read_state->last_aggregate_digest_ !=
                    zktr_segment_scalar_digest(values)) {
                if (error) *error = "Qpad aggregate receipt lacks freshly validated member segments";
                return false;
            }
            return true;
        };
    receipt.valid_ = true;
    return receipt;
}

struct SegmentCommitmentValidationReceiptFactory {
private:
    static SegmentCommitmentValidationReceipt mint(
        const zkdl::CpuIpaBasisContext& basis,
        const std::string& tensor_id, std::uint32_t visibility,
        const SegmentedTensorProvider& provider,
        const CommitmentWire& commitment,
        const SegmentCommitmentMaterial& material,
        std::vector<std::array<byte, 32U>>&& scalar_digests) {
        if (scalar_digests.size() != commitment.segment_count ||
            !basis.valid())
            throw std::invalid_argument("cannot mint incomplete segment MSM receipt");
        SegmentCommitmentValidationReceipt receipt;
        receipt.basis_ = basis;
        receipt.tensor_id_ = tensor_id;
        receipt.visibility_ = commitment.visibility;
        receipt.domain_size_ = commitment.domain_size;
        receipt.segment_size_ = commitment.segment_size;
        receipt.segment_count_ = commitment.segment_count;
        receipt.basis_digest_ = commitment.basis_digest;
        receipt.aggregate_commitment_ = zkdl::cpu_g1_from_native(commitment.commitment);
        receipt.header_digest_ = zktr_segment_receipt_header_digest(
            tensor_id, visibility, provider, commitment);
        receipt.scalar_digests_ = std::move(scalar_digests);
        receipt.material_digests_.reserve(receipt.scalar_digests_.size());
        for (std::uint32_t segment = 0U;
             segment < receipt.scalar_digests_.size(); ++segment)
            receipt.material_digests_.push_back(
                zktr_segment_receipt_material_digest(
                    tensor_id, commitment.visibility, commitment.domain_size,
                    commitment.basis_digest, material, segment));
        receipt.valid_ = true;
        return receipt;
    }

    friend bool zktr_build_segment_commitment_material(
        const std::string&, std::uint32_t, const SegmentedTensorProvider&,
        const std::vector<zkdl::CpuG1Point>&, const zkdl::CpuG1Point&,
        const zkdl::CpuG1Point&, const FrWire&, const RelationStatement&,
        CommitmentWire&, SegmentCommitmentMaterial&, std::string*, double*,
        std::uint32_t, const SegmentCommitmentProgressCallback*,
        const zkdl::CpuIpaBasisContext*, zkdl::CpuG1MsmSession*,
        SegmentCommitmentValidationReceipt*);
    friend bool zktr_commitment_wire_from_material(
        std::string, std::uint32_t, const SegmentedTensorProvider&,
        const std::vector<zkdl::CpuG1Point>&, const zkdl::CpuG1Point&,
        const zkdl::CpuG1Point&, const RelationStatement&,
        const SegmentCommitmentMaterial&, CommitmentWire&, std::string*,
        std::uint32_t, const SegmentCommitmentProgressCallback*,
        const zkdl::CpuIpaBasisContext*, zkdl::CpuG1MsmSession*,
        SegmentCommitmentValidationReceipt*);
};

inline bool registered_model_segment_blinders(const std::string& tensor_id, const ZkRelationSegmentLayout& layout, const FrWire& total, const std::array<byte, 32U>& srs_digest, std::vector<FrWire>& shares, std::string* error = nullptr) {
    if (tensor_id.empty() || !zkdl::fr_canonical(total) || same_wire(total, FrWire{}) || layout.segment_count == 0U || layout.segment_count > kMaxRelationSegments || srs_digest == std::array<byte, 32U>{}) { if (error) *error = "registered model segment blinder inputs are invalid"; return false; }
    if (layout.segment_count == 1U) { shares = {total}; return true; }
    std::vector<byte> key, prefix;
    append_fr_wire(key, total);
    zkdl::append_string(prefix, "zkdl.registered-model.segment-blinders.v1");
    zkdl::append_string(prefix, tensor_id);
    zkdl::append_u32(prefix, layout.logical_domain);
    zkdl::append_u32(prefix, layout.segment_size);
    zkdl::append_u32(prefix, layout.segment_count);
    append_digest(prefix, srs_digest);
    for (std::uint32_t nonce = 0U; nonce < (1U << 16U); ++nonce) {
        shares.assign(layout.segment_count, FrWire{});
        FrWire sum{};
        for (std::uint32_t segment = 1U; segment < layout.segment_count; ++segment) {
            bool derived = false;
            for (std::uint32_t attempt = 0U; attempt < (1U << 16U); ++attempt) {
                auto message = prefix;
                zkdl::append_u32(message, nonce);
                zkdl::append_u32(message, segment);
                zkdl::append_u32(message, attempt);
                std::array<byte, 32U> digest{};
                unsigned length = 0U;
                if (!HMAC(EVP_sha256(), key.data(), static_cast<int>(key.size()), message.data(), message.size(), digest.data(), &length) || length != digest.size()) { if (error) *error = "registered model secret PRF failed"; return false; }
                FrWire candidate{};
                for (std::size_t limb = 0U; limb < candidate.size(); ++limb) for (std::size_t octet = 0U; octet < 4U; ++octet) candidate[limb] |= static_cast<std::uint32_t>(digest[4U * limb + octet]) << (8U * octet);
                if (!zkdl::fr_canonical(candidate) || same_wire(candidate, FrWire{})) continue;
                shares[segment] = candidate;
                sum = zkdl::fr_add(sum, candidate);
                derived = true;
                break;
            }
            if (!derived) { if (error) *error = "registered model secret PRF rejection cap exceeded"; return false; }
        }
        shares[0U] = zkdl::fr_sub(total, sum);
        if (!same_wire(shares[0U], FrWire{})) return true;
    }
    if (error) *error = "registered model nonzero segment share cap exceeded";
    return false;
}

/* Build the ordered segment vector exactly once at the outer commitment
 * boundary.  The legacy one-G1 handle is deliberately the unweighted sum of
 * these points.  The terminal relation receives this material and never
 * samples a second set of segment blinders. */
inline bool zktr_basis_valid_for_prefix(
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    std::uint32_t prefix_size,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    std::string* error) {
    if (authenticated_basis != nullptr) {
        if (!authenticated_basis->owns_exact_generators(generators, h, u) ||
            !zkdl::cpu_pedersen_basis_valid(*authenticated_basis, h, u,
                                            prefix_size, error)) {
            if (error && error->empty())
                *error = "relation IPA basis context does not own the requested generators/H/U";
            return false;
        }
        return true;
    }
    if (prefix_size > generators.size()) return false;
    const std::vector<zkdl::CpuG1Point> prefix(
        generators.begin(), generators.begin() + prefix_size);
    return zkdl::cpu_pedersen_basis_valid(prefix, h, u, prefix_size, error);
}

inline std::array<byte, 32U> zktr_basis_digest_for_prefix(
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u,
    std::uint32_t prefix_size, std::uint32_t padded_size,
    const zkdl::CpuIpaBasisContext* authenticated_basis) {
    if (authenticated_basis != nullptr) {
        if (!authenticated_basis->owns_exact_generators(generators, h, u))
            throw std::invalid_argument("relation IPA basis handle does not own the supplied generators/H/U");
        return zkdl::cpu_ipa_basis_digest(*authenticated_basis, prefix_size,
                                          padded_size);
    }
    if (prefix_size == generators.size())
        return zkdl::cpu_ipa_basis_digest(generators, h, u, padded_size);
    const std::vector<zkdl::CpuG1Point> prefix(
        generators.begin(), generators.begin() + prefix_size);
    return zkdl::cpu_ipa_basis_digest(prefix, h, u, padded_size);
}

inline zkdl::CpuG1Point zktr_cpu_g1_sum(
    const std::vector<zkdl::CpuG1Point>& generators, const FrWire* scalars,
    std::size_t count, const zkdl::CpuIpaBasisContext* authenticated_basis,
    zkdl::CpuG1MsmSession* msm_session) {
    if (msm_session != nullptr) {
        if (authenticated_basis == nullptr)
            throw std::invalid_argument(
                "prepared segment MSM requires an authenticated IPA basis");
        return msm_session->sum(*authenticated_basis, generators,
                                authenticated_basis->h(),
                                authenticated_basis->u(), scalars, count);
    }
    return zkdl::cpu_g1_sum(generators.data(), scalars, count);
}

inline bool zktr_build_segment_commitment_material(
    const std::string& tensor_id, std::uint32_t visibility,
    const SegmentedTensorProvider& provider,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator, const FrWire& tensor_blinder,
    const RelationStatement& statement, CommitmentWire& commitment,
    SegmentCommitmentMaterial& material, std::string* error = nullptr,
    double* commitment_crypto_time_seconds = nullptr,
    std::uint32_t shared_segment_size = 0U,
    const SegmentCommitmentProgressCallback* segment_progress = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    SegmentCommitmentValidationReceipt* validation_receipt = nullptr) {
    try {
        double commitment_crypto_seconds = 0.0;
        ZkRelationSegmentLayout layout;
        const bool layout_valid = shared_segment_size == 0U
            ? zktr_segment_layout(provider.logical_domain(), generators.size(), layout, error)
            : zktr_segment_layout_for_size(provider.logical_domain(), generators.size(),
                                           shared_segment_size, layout, error);
        if (tensor_id.empty() || visibility == 0U ||
            !provider.valid(error) || !layout_valid)
            return false;
        if (provider.logical_domain() != layout.logical_domain ||
            provider.segment_size() != layout.segment_size ||
            provider.segment_count() != layout.segment_count) {
            if (error) *error = "segment commitment provider differs from authenticated shared layout";
            return false;
        }
        if (!zkdl::fr_canonical(tensor_blinder) ||
            !zktr_basis_valid_for_prefix(generators, blinding_generator,
                                         u_generator, layout.segment_size,
                                         authenticated_basis, error))
            return false;
        if (validation_receipt != nullptr && authenticated_basis == nullptr) {
            if (error) *error = "segment MSM receipt requires authenticated IPA basis";
            return false;
        }
        std::uint32_t generator_count = 0U;
        if (!checked_relation_u32(generators.size(), generator_count,
                                  "segment commitment generator count", error))
            return false;
        const auto frame_digest = zktr_basis_digest_for_prefix(
            generators, blinding_generator, u_generator, layout.segment_size,
            zkdl::cpu_ipa_padded_size(layout.segment_size), authenticated_basis);
        const auto basis_digest = zktr_basis_digest_for_prefix(
            generators, blinding_generator, u_generator, generator_count,
            zkdl::cpu_ipa_padded_size(generator_count), authenticated_basis);
        if (statement.srs_digest == std::array<byte, 32U>{} ||
            frame_digest == std::array<byte, 32U>{} ||
            basis_digest == std::array<byte, 32U>{}) {
            if (error) *error = "segment commitment lacks authenticated SRS/basis digest";
            return false;
        }
        material = SegmentCommitmentMaterial{};
        material.blinders.resize(layout.segment_count);
        material.commitments.reserve(layout.segment_count);
        if (layout.segment_count > 1U && visibility == 2U) {
            if (!registered_model_segment_blinders(tensor_id, layout, tensor_blinder, statement.srs_digest, material.blinders, error)) return false;
        } else if (layout.segment_count > 1U && visibility != 1U) {
            FrWire sum{};
            for (std::uint32_t segment = 1U; segment < layout.segment_count; ++segment) {
                material.blinders[segment] = zkdl::cpu_fresh_csprng_scalar();
                sum = zkdl::fr_add(sum, material.blinders[segment]);
            }
            material.blinders[0U] = zkdl::fr_sub(tensor_blinder, sum);
        } else {
            material.blinders[0U] = tensor_blinder;
        }
        commitment = CommitmentWire{};
        commitment.tensor_id = tensor_id;
        commitment.visibility = visibility;
        commitment.domain_size = layout.logical_domain;
        commitment.segment_binding_version = RelationStatement::kSegmentBindingVersion;
        commitment.segment_size = layout.segment_size;
        commitment.segment_count = layout.segment_count;
        commitment.srs_manifest_digest = statement.srs_digest;
        commitment.basis_digest = basis_digest;
        commitment.segment_commitments.reserve(layout.segment_count);
        zkdl::CpuG1Point legacy = zkdl::CpuG1Point::zero();
        std::vector<std::array<byte, 32U>> scalar_digests;
        if (validation_receipt != nullptr)
            scalar_digests.reserve(layout.segment_count);
        for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
            std::vector<FrWire> values;
            if (!provider.read_segment(segment, values, error)) return false;
            if (validation_receipt != nullptr)
                scalar_digests.push_back(zktr_segment_scalar_digest(values));
            const std::uint64_t offset = static_cast<std::uint64_t>(segment) * layout.segment_size;
            std::uint32_t actual = 0U;
            if (!checked_relation_u32(
                    std::min<std::uint64_t>(layout.segment_size,
                                            layout.logical_domain - offset),
                    actual, "segment commitment actual length", error))
                return false;
            const auto crypto_start = std::chrono::steady_clock::now();
            const zkdl::CpuG1Point xi = zkdl::cpu_g1_add(
                zktr_cpu_g1_sum(generators, values.data(), layout.segment_size,
                                authenticated_basis, msm_session),
                zkdl::cpu_g1_mul(blinding_generator,
                                 material.blinders[segment]));
            const G1Jacobian_t encoded_commitment = zkdl::cpu_g1_to_native(xi);
            const zkdl::CpuG1Point next_legacy = zkdl::cpu_g1_add(legacy, xi);
            commitment_crypto_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - crypto_start).count();
            if (xi.is_zero() && visibility != 1U) {
                if (error) *error = "private segment commitment point is zero";
                return false;
            }
            SegmentCommitmentWire wire;
            wire.segment_index = segment;
            wire.logical_offset = offset;
            wire.actual_length = actual;
            wire.padded_length = layout.segment_size;
            wire.padding_length = layout.segment_size - actual;
            wire.frame_digest = frame_digest;
            wire.commitment = encoded_commitment;
            commitment.segment_commitments.push_back(std::move(wire));
            material.commitments.push_back(xi);
            legacy = next_legacy;
            if (segment_progress != nullptr)
                (*segment_progress)(segment + 1U, layout.segment_count);
        }
        const auto finalization_start = std::chrono::steady_clock::now();
        commitment.commitment = zkdl::cpu_g1_to_native(legacy);
        commitment_crypto_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - finalization_start).count();
        commitment.segment_commitment_digest = segment_commitment_digest(commitment);
        if (!segment_commitment_wire_shape_valid(commitment, error)) return false;
        if (validation_receipt != nullptr)
            *validation_receipt = SegmentCommitmentValidationReceiptFactory::mint(
                *authenticated_basis, tensor_id, visibility, provider, commitment,
                material, std::move(scalar_digests));
        if (commitment_crypto_time_seconds != nullptr)
            *commitment_crypto_time_seconds += commitment_crypto_seconds;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

/* Re-encode an already authenticated outer material without sampling.  This
 * is used by derived lookup wires: setup owns the segment shares/Xi vector,
 * while proof assembly may only replay the exact vector against the provider
 * and authenticated IPA basis. */
inline bool zktr_commitment_wire_from_material(
    std::string tensor_id, std::uint32_t visibility,
    const SegmentedTensorProvider& provider,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator, const RelationStatement& statement,
    const SegmentCommitmentMaterial& material, CommitmentWire& commitment,
    std::string* error = nullptr,
    std::uint32_t shared_segment_size = 0U,
    const SegmentCommitmentProgressCallback* segment_progress = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    SegmentCommitmentValidationReceipt* validation_receipt = nullptr) {
    try {
        ZkRelationSegmentLayout layout;
        const bool layout_valid = shared_segment_size == 0U
            ? zktr_segment_layout(provider.logical_domain(), generators.size(), layout, error)
            : zktr_segment_layout_for_size(provider.logical_domain(), generators.size(),
                                           shared_segment_size, layout, error);
        if (tensor_id.empty() || visibility == 0U || !provider.valid(error) ||
            !layout_valid)
            return false;
        if (provider.logical_domain() != layout.logical_domain ||
            provider.segment_size() != layout.segment_size ||
            provider.segment_count() != layout.segment_count) {
            if (error) *error = "segment commitment replay provider differs from authenticated shared layout";
            return false;
        }
        if (material.blinders.size() != layout.segment_count ||
            material.commitments.size() != layout.segment_count ||
            !zktr_basis_valid_for_prefix(generators, blinding_generator,
                                         u_generator, layout.segment_size,
                                         authenticated_basis, error))
            return false;
        std::uint32_t generator_count = 0U;
        if (!checked_relation_u32(generators.size(), generator_count,
                                  "segment commitment generator count", error))
            return false;
        const auto frame_digest = zktr_basis_digest_for_prefix(
            generators, blinding_generator, u_generator, layout.segment_size,
            zkdl::cpu_ipa_padded_size(layout.segment_size), authenticated_basis);
        const auto basis_digest = zktr_basis_digest_for_prefix(
            generators, blinding_generator, u_generator, generator_count,
            zkdl::cpu_ipa_padded_size(generator_count), authenticated_basis);
        if (statement.srs_digest == std::array<byte, 32U>{} ||
            frame_digest == std::array<byte, 32U>{} ||
            basis_digest == std::array<byte, 32U>{}) {
            if (error) *error = "segment commitment lacks authenticated SRS/basis digest";
            return false;
        }
        commitment = CommitmentWire{};
        commitment.tensor_id = tensor_id;
        commitment.visibility = visibility;
        commitment.domain_size = layout.logical_domain;
        commitment.segment_binding_version = RelationStatement::kSegmentBindingVersion;
        commitment.segment_size = layout.segment_size;
        commitment.segment_count = layout.segment_count;
        commitment.srs_manifest_digest = statement.srs_digest;
        commitment.basis_digest = basis_digest;
        commitment.segment_commitments.reserve(layout.segment_count);
        zkdl::CpuG1Point legacy = zkdl::CpuG1Point::zero();
        std::vector<std::array<byte, 32U>> scalar_digests;
        if (validation_receipt != nullptr)
            scalar_digests.reserve(layout.segment_count);
        for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
            if (!zkdl::fr_canonical(material.blinders[segment])) {
                if (error) *error = "segment commitment material has a non-canonical blinder";
                return false;
            }
            std::vector<FrWire> values;
            if (!provider.read_segment(segment, values, error)) return false;
            if (validation_receipt != nullptr)
                scalar_digests.push_back(zktr_segment_scalar_digest(values));
            const std::uint64_t offset = static_cast<std::uint64_t>(segment) * layout.segment_size;
            std::uint32_t actual = 0U;
            if (!checked_relation_u32(
                    std::min<std::uint64_t>(layout.segment_size,
                                            layout.logical_domain - offset),
                    actual, "segment commitment actual length", error))
                return false;
            const zkdl::CpuG1Point xi = zkdl::cpu_g1_add(
                zktr_cpu_g1_sum(generators, values.data(), layout.segment_size,
                                authenticated_basis, msm_session),
                zkdl::cpu_g1_mul(blinding_generator, material.blinders[segment]));
            if ((xi.is_zero() && visibility != 1U) || !zkdl::cpu_g1_equal(xi, material.commitments[segment])) {
                if (error) *error = "segment commitment material differs from provider witness";
                return false;
            }
            SegmentCommitmentWire wire;
            wire.segment_index = segment;
            wire.logical_offset = offset;
            wire.actual_length = actual;
            wire.padded_length = layout.segment_size;
            wire.padding_length = layout.segment_size - actual;
            wire.frame_digest = frame_digest;
            wire.commitment = zkdl::cpu_g1_to_native(xi);
            commitment.segment_commitments.push_back(std::move(wire));
            legacy = zkdl::cpu_g1_add(legacy, xi);
            if (segment_progress != nullptr) (*segment_progress)(segment + 1U, layout.segment_count);
        }
        commitment.commitment = zkdl::cpu_g1_to_native(legacy);
        commitment.segment_commitment_digest = segment_commitment_digest(commitment);
        if (!segment_commitment_wire_shape_valid(commitment, error)) return false;
        if (validation_receipt != nullptr) {
            if (authenticated_basis == nullptr) {
                if (error) *error = "segment MSM receipt requires authenticated IPA basis";
                return false;
            }
            *validation_receipt = SegmentCommitmentValidationReceiptFactory::mint(
                *authenticated_basis, tensor_id, visibility, provider, commitment,
                material, std::move(scalar_digests));
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool zktr_segment_material_matches(
    const CommitmentWire& commitment, const SegmentCommitmentMaterial& material,
    std::string* error = nullptr) {
    if (!segment_commitment_wire_shape_valid(commitment, error) ||
        commitment.segment_commitments.size() != material.commitments.size() ||
        material.commitments.size() != material.blinders.size()) {
        if (error && error->empty()) *error = "segment commitment material shape mismatch";
        return false;
    }
    for (std::size_t index = 0U; index < material.commitments.size(); ++index) {
        if (!zkdl::cpu_g1_equal(
                material.commitments[index],
                zkdl::cpu_g1_from_native(commitment.segment_commitments[index].commitment))) {
            if (error) *error = "segment commitment material differs from statement vector";
            return false;
        }
    }
    return true;
}

inline bool zktr_material_from_wire(
    const CommitmentWire& commitment,
    const std::map<std::string, std::vector<FrWire>>& segment_blinding_scalars,
    SegmentCommitmentMaterial& material, std::string* error = nullptr) {
    if (!segment_commitment_wire_shape_valid(commitment, error)) return false;
    const auto found = segment_blinding_scalars.find(commitment.tensor_id);
    if (found == segment_blinding_scalars.end() ||
        found->second.size() != commitment.segment_commitments.size()) {
        if (error) *error = "segment blinder shares are missing for commitment wire";
        return false;
    }
    material.blinders = found->second;
    material.commitments.clear();
    material.commitments.reserve(commitment.segment_commitments.size());
    for (const auto& segment : commitment.segment_commitments)
        material.commitments.push_back(zkdl::cpu_g1_from_native(segment.commitment));
    return true;
}

inline std::vector<FrWire> zktr_segment_point(const std::vector<FrWire>& point,
                                              std::uint32_t segment_size) {
    std::uint32_t low_bits = 0U;
    for (std::uint32_t value = segment_size; value > 1U; value >>= 1U) ++low_bits;
    if (low_bits > point.size()) throw std::invalid_argument("segmented relation point is too short");
    return std::vector<FrWire>(point.end() - static_cast<std::ptrdiff_t>(low_bits), point.end());
}

inline std::vector<FrWire> zktr_segment_prefix_point(const std::vector<FrWire>& point,
                                                     std::uint32_t segment_size) {
    std::uint32_t low_bits = 0U;
    for (std::uint32_t value = segment_size; value > 1U; value >>= 1U) ++low_bits;
    if (low_bits > point.size()) throw std::invalid_argument("segmented relation point is too short");
    return std::vector<FrWire>(point.begin(), point.end() - static_cast<std::ptrdiff_t>(low_bits));
}

inline zkdl::CpuG1Point zktr_segmented_commitment(
    const SegmentedTensorProvider& provider,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const FrWire& randomness,
    std::string* error = nullptr) {
    ZkRelationSegmentLayout layout;
    if (!provider.valid(error) ||
        !zktr_segment_layout(provider.logical_domain(), generators.size(), layout, error) ||
        provider.segment_size() != layout.segment_size ||
        provider.segment_count() != layout.segment_count)
        return zkdl::CpuG1Point::zero();
    zkdl::CpuG1Point result = zkdl::CpuG1Point::zero();
    const std::vector<zkdl::CpuG1Point> segment_generators(
        generators.begin(), generators.begin() + layout.segment_size);
    for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
        std::vector<FrWire> segment_values;
        if (!provider.read_segment(segment, segment_values, error))
            return zkdl::CpuG1Point::zero();
        result = zkdl::cpu_g1_add(result, zkdl::cpu_g1_sum(segment_generators, segment_values));
    }
    return zkdl::cpu_g1_add(result, zkdl::cpu_g1_mul(blinding_generator, randomness));
}

inline zkdl::CpuG1Point zktr_segmented_commitment(
    const std::vector<FrWire>& values, const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator, const FrWire& randomness,
    std::string* error = nullptr) {
    SegmentedTensorProvider provider;
    if (!SegmentedTensorProvider::from_values(values, generators.size(), provider, error))
        return zkdl::CpuG1Point::zero();
    return zktr_segmented_commitment(provider, generators, blinding_generator, randomness, error);
}

inline bool zktr_validate_common_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& tensor_providers,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    const std::string& operation_context,
    std::string* error = nullptr,
    std::uint32_t shared_segment_size = 0U,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr) {
    if (!descriptor.valid(error) || tensor_providers.empty() ||
        tensor_providers.size() != canonical_commitments.size() ||
        tensor_providers.size() != tensor_blinders.size() ||
        tensor_providers.size() != commitment_indices.size() ||
        tensor_providers.size() > zkdl::ZkTerminalRelationProof::kMaxTerminalCommitments ||
        evaluation_point.size() >= 31U || evaluation_point.empty() ||
        operation_context.empty() || !zktr_all_statement_digests_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid terminal relation inputs";
        return false;
    }
    const std::uint64_t domain64 = std::uint64_t(1U) << evaluation_point.size();
    ZkRelationSegmentLayout layout;
    const bool layout_valid = shared_segment_size == 0U
        ? zktr_segment_layout(domain64, generators.size(), layout, error)
        : zktr_segment_layout_for_size(domain64, generators.size(),
                                       shared_segment_size, layout, error);
    if (!layout_valid) return false;
    if (!zktr_basis_valid_for_prefix(
            generators, blinding_generator, u_generator, layout.basis_domain,
            authenticated_basis, error)) return false;
    std::set<std::uint32_t> seen_indices;
    for (std::size_t index = 0U; index < tensor_providers.size(); ++index) {
        if (commitment_indices[index] >= kZkTerminalEvaluationIndexBase ||
            !seen_indices.insert(commitment_indices[index]).second ||
            !tensor_providers[index].valid(error) ||
            tensor_providers[index].logical_domain() != domain64 ||
            tensor_providers[index].segment_size() != layout.segment_size ||
            tensor_providers[index].segment_count() != layout.segment_count ||
            !zkdl::fr_canonical(tensor_blinders[index]) ||
            !zkdl::cpu_pedersen_strict_g1(canonical_commitments[index], error)) {
            if (error) *error = "invalid canonical terminal tensor input or shared segment layout";
            return false;
        }
        // Segment payloads are validated when the terminal-opening loop consumes
        // each segment below; do not decode and discard them here.
    }
    for (const auto& coordinate : evaluation_point)
        if (!zkdl::fr_canonical(coordinate)) {
            if (error) *error = "terminal evaluation point is not canonical";
            return false;
        }
    for (const auto& term : descriptor.terms)
        for (const auto ref : term.refs)
            if (ref >= tensor_providers.size()) {
                if (error) *error = "terminal monomial references an unknown tensor";
                return false;
            }
    return true;
}

inline bool zktr_validate_common_inputs(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    const std::string& operation_context,
    std::string* error = nullptr) {
    std::vector<SegmentedTensorProvider> providers;
    if (!make_segmented_tensor_providers(tensor_values, generators.size(), providers, error))
        return false;
    return zktr_validate_common_inputs(
        descriptor, providers, canonical_commitments, tensor_blinders,
        commitment_indices, evaluation_point, generators, blinding_generator,
        u_generator, statement, operation_context, error);
}

inline bool zktr_validate_public_inputs(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement, const std::string& operation_context,
    std::string* error = nullptr,
    std::uint32_t shared_segment_size = 0U,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr) {
    if (!descriptor.valid(error) || tensor_count == 0U ||
        tensor_count != canonical_commitments.size() ||
        tensor_count != commitment_indices.size() ||
        tensor_count > zkdl::ZkTerminalRelationProof::kMaxTerminalCommitments ||
        evaluation_point.size() >= 31U || evaluation_point.empty() ||
        operation_context.empty() || !zktr_all_statement_digests_valid(statement, error)) {
        if (error && error->empty()) *error = "invalid terminal relation verifier inputs";
        return false;
    }
    const std::uint64_t domain64 = std::uint64_t(1U) << evaluation_point.size();
    ZkRelationSegmentLayout layout;
    const bool layout_valid = shared_segment_size == 0U
        ? zktr_segment_layout(domain64, generators.size(), layout, error)
        : zktr_segment_layout_for_size(domain64, generators.size(),
                                       shared_segment_size, layout, error);
    if (!layout_valid) return false;
    if (!zktr_basis_valid_for_prefix(
            generators, blinding_generator, u_generator, layout.basis_domain,
            authenticated_basis, error)) return false;
    std::set<std::uint32_t> seen_indices;
    for (std::size_t index = 0U; index < tensor_count; ++index)
        if (commitment_indices[index] >= kZkTerminalEvaluationIndexBase ||
            !seen_indices.insert(commitment_indices[index]).second ||
            !zkdl::cpu_pedersen_strict_g1(canonical_commitments[index], error)) {
            if (error && error->empty()) *error = "invalid canonical terminal commitment map";
            return false;
        }
    for (const auto& coordinate : evaluation_point)
        if (!zkdl::fr_canonical(coordinate)) {
            if (error) *error = "terminal evaluation point is not canonical";
            return false;
        }
    for (const auto& term : descriptor.terms)
        for (const auto ref : term.refs)
            if (ref >= tensor_count) {
                if (error) *error = "terminal monomial references an unknown tensor";
                return false;
            }
    return true;
}

inline bool zktr_build_product_plan(
    const ConstraintDescriptor& descriptor, std::size_t tensor_count,
    const std::vector<std::uint32_t>& commitment_indices,
    std::vector<ZkTerminalProductPlan>& plans,
    std::vector<std::uint32_t>& monomial_commitment_indices,
    std::string* error = nullptr) {
    if (commitment_indices.size() != tensor_count) {
        if (error) *error = "terminal product plan tensor/index shape mismatch";
        return false;
    }
    plans.clear();
    monomial_commitment_indices.clear();
    monomial_commitment_indices.reserve(descriptor.terms.size());
    for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
        const auto& refs = descriptor.terms[monomial].refs;
        if (refs.empty()) {
            if (error) *error = "terminal product plan contains an empty monomial";
            return false;
        }
        for (const auto ref : refs)
            if (ref >= tensor_count) {
                if (error) *error = "terminal product plan reference is out of bounds";
                return false;
            }
        if (refs.size() == 1U) {
            monomial_commitment_indices.push_back(
                zktr_terminal_evaluation_index(refs.front()));
            continue;
        }
        std::uint32_t left = zktr_terminal_evaluation_index(refs[0U]);
        for (std::size_t factor = 1U; factor < refs.size(); ++factor) {
            const std::size_t node_index = plans.size();
            if (node_index >= zkdl::ZkTerminalRelationProof::kMaxProductNodes) {
                if (error) *error = "terminal product DAG exceeds its bounded node cap";
                return false;
            }
            std::uint32_t monomial_index = 0U, factor_step = 0U;
            if (!checked_relation_u32(monomial, monomial_index,
                                      "terminal product monomial index", error) ||
                !checked_relation_u32(factor, factor_step,
                                      "terminal product factor index", error))
                return false;
            const std::uint32_t right = zktr_terminal_evaluation_index(refs[factor]);
            std::uint32_t right_index = right;
            std::uint32_t alias_source =
                std::numeric_limits<std::uint32_t>::max();
            if (right_index == left) {
                alias_source = right_index;
                if (!zktr_product_alias_index(node_index, right_index, error))
                    return false;
            }
            const std::uint32_t output = zktr_product_index(node_index);
            if (left == right_index || left == output || right_index == output) {
                if (error) *error = "terminal product role alias/index collision";
                return false;
            }
            plans.push_back(ZkTerminalProductPlan{
                monomial_index, factor_step, left, right_index, output,
                alias_source});
            left = output;
        }
        monomial_commitment_indices.push_back(left);
    }
    return true;
}

inline zkdl::PedersenProofContext zktr_make_link_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const std::array<byte, 32U>& descriptor_digest) {
    const std::uint32_t operation =
        link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality
            ? zkdl::PedersenProofContext::kOperationScalarEquality
            : zkdl::PedersenProofContext::kOperationScalarValue;
    const std::uint32_t related =
        link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality
            ? kZkTerminalExternalIndex
            : zkdl::PedersenProofContext::kUnsetIndex;
    const auto basis_digest = zkdl::cpu_ipa_basis_digest(
        {}, zkdl::CpuG1Point::zero(), zkdl::CpuG1Point::zero(), 0U);
    // The basis is replaced by the caller after construction; this helper is
    // kept parameter-free only for role/index derivation.
    return zksc_make_context(
        operation, 1U, 0U, kZkTerminalResidualIndex, related, basis_digest,
        descriptor_digest, statement);
}

inline zkdl::PedersenProofContext zktr_make_link_context(
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const std::array<byte, 32U>& descriptor_digest,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator) {
    (void)descriptor;
    (void)operation_context;
    auto context = zktr_make_link_context(
        descriptor, statement, operation_context, link_kind, descriptor_digest);
    context.basis_digest = zkdl::cpu_ipa_basis_digest(
        {}, blinding_generator, u_generator, 0U);
    return context;
}

inline zkdl::PedersenProofContext zktr_make_product_context(
    const RelationStatement& statement, const std::array<byte, 32U>& descriptor_digest,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    std::uint32_t left, std::uint32_t right, std::uint32_t output) {
    auto context = zksc_make_context(
        zkdl::PedersenProofContext::kOperationProduct, 2U, 0U, left, right,
        zkdl::cpu_ipa_basis_digest({}, blinding_generator, u_generator, 0U),
        descriptor_digest, statement);
    context.commitment_c_index = output;
    return context;
}

inline zkdl::CpuIpaOpeningMetadata zktr_make_opening_metadata(
    const std::array<byte, 32U>& descriptor_digest,
    const RelationStatement& statement, const std::vector<FrWire>& point,
    std::uint32_t commitment_index, std::uint32_t evaluation_commitment_index) {
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.evaluation_commitment_index = evaluation_commitment_index;
    metadata.outer_point = point;
    metadata.descriptor_digest = descriptor_digest;
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    return metadata;
}

inline void zktr_absorb_header(
    zkdl::Transcript& transcript, const zkdl::ZkTerminalRelationProof& proof,
    const ConstraintDescriptor& descriptor, const RelationStatement& statement,
    const std::string& operation_context, const std::vector<FrWire>& point,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const zkdl::CpuG1Point& blinding_generator, const zkdl::CpuG1Point& u_generator,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& expected_public_value) {
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, proof.version);
    zkdl::append_string(encoded, operation_context);
    append_constraint(encoded, descriptor);
    zkdl::append_pedersen_context(encoded, proof.context);
    append_digest(encoded, statement.statement_digest);
    append_digest(encoded, statement.model_digest);
    append_digest(encoded, statement.table_digest);
    append_digest(encoded, statement.ledger_digest);
    append_digest(encoded, statement.registry_digest);
    append_digest(encoded, statement.srs_digest);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(point.size()));
    for (const auto& coordinate : point) zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(commitment_indices.size()));
    for (std::size_t index = 0U; index < commitment_indices.size(); ++index) {
        zkdl::append_u32(encoded, commitment_indices[index]);
        zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(canonical_commitments[index]));
    }
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(blinding_generator));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(u_generator));
    zkdl::append_u32(encoded, static_cast<std::uint32_t>(link_kind));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(external_claim_commitment));
    zkdl::append_fr(encoded, zkdl::fr_value(expected_public_value));
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(proof.terminal_commitments.size()));
    for (const auto& commitment : proof.terminal_commitments)
        zkdl::append_g1(encoded, commitment);
    // Bind the weighted terminal Xi aggregate independently of the legacy
    // unweighted handle.  The segment vector is already in the statement
    // transcript; this value is the exact C* used by the terminal relation.
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(proof.terminal_segment_openings.size()));
    for (const auto& openings : proof.terminal_segment_openings) {
        zkdl::CpuG1Point aggregate = zkdl::CpuG1Point::zero();
        if (!openings.empty() && openings.front().padded_length != 0U) {
            const std::vector<FrWire> high_point = zktr_segment_prefix_point(
                point, openings.front().padded_length);
            const std::vector<FrWire> weights = zkdl::cpu_ipa_mle_weights(high_point);
            if (weights.size() == openings.size()) {
                for (std::size_t index = 0U; index < openings.size(); ++index)
                    aggregate = zkdl::cpu_g1_add(
                        aggregate, zkdl::cpu_g1_mul(
                            zkdl::cpu_g1_from_native(openings[index].xi), weights[index]));
            }
        }
        zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(aggregate));
    }
    transcript.absorb("gat.zkterminal.header", encoded);
}

inline void zktr_absorb_components(
    zkdl::Transcript& transcript, const zkdl::ZkTerminalRelationProof& proof,
    const std::string& operation_context) {
    for (std::size_t index = 0U; index < proof.terminal_openings.size(); ++index) {
        const auto& segment_openings = proof.terminal_segment_openings[index];
        for (std::size_t segment = 0U; segment < segment_openings.size(); ++segment) {
            std::vector<byte> encoded;
            zkdl::serialize_hyrax_opening_v3(segment_openings[segment], encoded);
            transcript.absorb("gat.zkterminal." + operation_context + ".opening." +
                              std::to_string(index) + ".segment." +
                              std::to_string(segment), encoded);
        }
    }
    for (std::size_t index = 0U; index < proof.product_nodes.size(); ++index) {
        std::vector<byte> encoded;
        zkdl::serialize_zk_product_node(proof.product_nodes[index], encoded);
        transcript.absorb("gat.zkterminal." + operation_context + ".product." +
                          std::to_string(index), encoded);
    }
    std::vector<byte> link;
    if (proof.claim_link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality)
        zkdl::serialize_pedersen_scalar_equality(proof.external_equality_binding, link);
    else
        zkdl::serialize_pedersen_scalar_value(proof.public_value_binding, link);
    transcript.absorb("gat.zkterminal." + operation_context + ".link", link);
}

inline bool zktr_link_kind_valid(zkdl::ZkTerminalClaimLinkKind link_kind) {
    return link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality ||
           link_kind == zkdl::ZkTerminalClaimLinkKind::PublicValue;
}

inline bool prove_zk_terminal_relation_segmented(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& tensor_providers,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& external_claim_blinder,
    const FrWire& expected_public_value,
    const std::string& operation_context,
    zkdl::Transcript& parent_transcript,
    ZkTerminalRelationProverResult& result,
    std::string* error = nullptr,
    const std::vector<SegmentCommitmentMaterial>* segment_materials = nullptr,
    std::uint32_t shared_segment_size = 0U,
    NativeOperationProgressReporter* progress = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::vector<const SegmentCommitmentValidationReceipt*>*
        segment_receipts = nullptr) {
    try {
        NativeOperationProgressReporter local_progress("terminal_relation", operation_context);
        if (progress == nullptr) progress = &local_progress;
        if (!zktr_link_kind_valid(link_kind) ||
            !zktr_validate_common_inputs(
                descriptor, tensor_providers, canonical_commitments, tensor_blinders,
                commitment_indices, evaluation_point, generators, blinding_generator,
                u_generator, statement, operation_context, error,
                shared_segment_size, authenticated_basis) ||
            !zkdl::fr_canonical(expected_public_value)) {
            if (error && error->empty()) *error = "invalid terminal relation prover inputs";
            return false;
        }
        if (authenticated_basis != nullptr &&
            !zktr_basis_valid_for_prefix(
                generators, blinding_generator, u_generator,
                static_cast<std::uint32_t>(generators.size()),
                authenticated_basis, error)) return false;
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::fr_canonical(external_claim_blinder) ||
                zkdl::fr_compare(external_claim_blinder, FrWire{}) == 0 ||
                !zkdl::cpu_pedersen_strict_g1(external_claim_commitment, error)) {
                if (error && error->empty()) *error = "invalid external terminal claim";
                return false;
            }
        } else if (!external_claim_commitment.is_zero() ||
                   zkdl::fr_compare(external_claim_blinder, FrWire{}) != 0) {
            if (error) *error = "public terminal link carries an external claim witness";
            return false;
        }

        std::vector<ZkTerminalProductPlan> plans;
        std::vector<std::uint32_t> monomial_indices;
        if (!zktr_build_product_plan(descriptor, tensor_providers.size(), commitment_indices,
                                     plans, monomial_indices, error)) return false;

        result = ZkTerminalRelationProverResult{};
        auto& proof = result.proof;
        proof.claim_link_kind = link_kind;
        proof.context = zktr_make_link_context(
            descriptor, statement, operation_context, link_kind,
            zksc_proof_descriptor_digest(descriptor, statement, operation_context),
            blinding_generator, u_generator);
        const std::array<byte, 32U> descriptor_digest = proof.context.descriptor_digest;
        result.terminal_values.reserve(tensor_providers.size());
        result.terminal_blinders.reserve(tensor_providers.size());
        proof.terminal_commitments.reserve(tensor_providers.size());
        proof.terminal_openings.reserve(tensor_providers.size());
        proof.terminal_segment_openings.reserve(tensor_providers.size());

        struct ScalarState {
            FrWire value{};
            FrWire blinder{};
            zkdl::CpuG1Point commitment;
        };
        std::map<std::uint32_t, ScalarState> states;
        for (std::size_t tensor_index = 0U;
             tensor_index < tensor_providers.size(); ++tensor_index) {
            std::uint32_t tensor_index_u32 = 0U;
            if (!checked_relation_u32(tensor_index, tensor_index_u32,
                                      "terminal relation tensor index", error)) return false;
            ZkRelationSegmentLayout layout;
            const bool layout_valid = shared_segment_size == 0U
                ? zktr_segment_layout(tensor_providers[tensor_index].logical_domain(),
                                      generators.size(), layout, error)
                : zktr_segment_layout_for_size(
                      tensor_providers[tensor_index].logical_domain(),
                      generators.size(), shared_segment_size, layout, error);
            if (!layout_valid ||
                tensor_providers[tensor_index].segment_size() != layout.segment_size ||
                tensor_providers[tensor_index].segment_count() != layout.segment_count) {
                if (error && error->empty())
                    *error = "terminal provider differs from authenticated shared segment layout";
                return false;
            }
            const std::vector<FrWire> low_point =
                zktr_segment_point(evaluation_point, layout.segment_size);
            const std::vector<FrWire> low_weights =
                zkdl::cpu_ipa_mle_weights(low_point);
            std::vector<zkdl::CpuG1Point> segment_generators;
            if (authenticated_basis == nullptr)
                segment_generators.assign(
                    generators.begin(), generators.begin() + layout.segment_size);
            const std::vector<FrWire> high_point =
                zktr_segment_prefix_point(evaluation_point, layout.segment_size);
            const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
            if (high_weights.size() != layout.segment_count) {
                if (error) *error = "segmented terminal high-bit MLE shape mismatch";
                return false;
            }
            for (const auto& weight : high_weights)
                if (zkdl::fr_compare(weight, FrWire{}) == 0) {
                    if (error) *error = "segmented terminal high-bit binding weight is zero";
                    return false;
                }
            SegmentCommitmentMaterial singleton_material;
            const SegmentCommitmentMaterial* fixed_material = nullptr;
            if (segment_materials != nullptr && tensor_index < segment_materials->size()) {
                fixed_material = &(*segment_materials)[tensor_index];
            } else if (layout.segment_count == 1U) {
                singleton_material.blinders = {tensor_blinders[tensor_index]};
                singleton_material.commitments = {canonical_commitments[tensor_index]};
                fixed_material = &singleton_material;
            } else {
                if (error) *error = "segmented terminal relation lacks statement segment material";
                return false;
            }
            if (fixed_material->blinders.size() != layout.segment_count ||
                fixed_material->commitments.size() != layout.segment_count) {
                if (error) *error = "segmented terminal material count mismatch";
                return false;
            }
            const std::vector<FrWire>& segment_blinders = fixed_material->blinders;
            const SegmentCommitmentValidationReceipt* validation_receipt =
                segment_receipts != nullptr && tensor_index < segment_receipts->size()
                    ? (*segment_receipts)[tensor_index] : nullptr;
            if (validation_receipt != nullptr && !validation_receipt->valid())
                validation_receipt = nullptr;
            if (validation_receipt != nullptr &&
                (authenticated_basis == nullptr ||
                 !validation_receipt->matches_opening_context(
                     *authenticated_basis, generators, blinding_generator,
                     u_generator, layout.logical_domain, layout.segment_size,
                     layout.segment_count, canonical_commitments[tensor_index],
                     *fixed_material, error))) return false;
            FrWire value{};
            FrWire tau_blinder{};
            zkdl::CpuG1Point aggregate_tau = zkdl::CpuG1Point::zero();
            zkdl::CpuG1Point aggregate_xi = zkdl::CpuG1Point::zero();
            std::vector<zkdl::HyraxOpeningProofV3> segment_openings;
            segment_openings.reserve(layout.segment_count);
            const std::uint32_t terminal_index = zktr_terminal_evaluation_index(tensor_index);
            for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
                const std::uint64_t offset = static_cast<std::uint64_t>(segment) * layout.segment_size;
                std::uint32_t actual = 0U;
                if (!checked_relation_u32(
                        std::min<std::uint64_t>(layout.segment_size,
                                                layout.logical_domain - offset),
                        actual, "terminal segment actual length", error))
                    return false;
                std::vector<FrWire> segment_values;
                if (!tensor_providers[tensor_index].read_segment(
                        segment, segment_values, error)) return false;
                const FrWire segment_value = zkdl::cpu_ipa_inner_product(
                    segment_values, low_weights);
                const FrWire segment_tau_blinder = zkdl::cpu_fresh_csprng_scalar();
                const zkdl::CpuG1Point segment_xi = fixed_material->commitments[segment];
                zkdl::CpuG1Point checked_xi = segment_xi;
                if (validation_receipt != nullptr) {
                    if (!validation_receipt->matches_segment(
                            segment, segment_values, *fixed_material, error))
                        return false;
                } else {
                    checked_xi = zkdl::cpu_g1_add(
                        zktr_cpu_g1_sum(generators, segment_values.data(),
                                        layout.segment_size, authenticated_basis,
                                        msm_session),
                        zkdl::cpu_g1_mul(blinding_generator,
                                         segment_blinders[segment]));
                    if (!zkdl::cpu_g1_equal(checked_xi, segment_xi)) {
                        if (error) *error = "segment commitment material does not match witness values";
                        return false;
                    }
                }
                const zkdl::CpuG1Point segment_tau = zkdl::cpu_pedersen_scalar_commit(
                    segment_value, segment_tau_blinder, blinding_generator, u_generator);
                auto metadata = zktr_make_opening_metadata(
                    descriptor_digest, statement, evaluation_point,
                    commitment_indices[tensor_index], terminal_index);
                metadata.logical_domain = layout.logical_domain;
                metadata.segment_index = segment;
                metadata.logical_offset = offset;
                metadata.actual_length = actual;
                metadata.padded_length = layout.segment_size;
                const auto opening = authenticated_basis != nullptr
                    ? zkdl::cpu_open_hyrax_v3_authenticated(
                          *authenticated_basis, segment_values, low_point,
                          segment_blinders[segment], segment_tau_blinder,
                          segment_tau, metadata, checked_xi, msm_session)
                    : zkdl::cpu_open_hyrax_v3_with_checked_xi(
                          segment_generators, segment_values, low_point,
                          blinding_generator, u_generator, segment_blinders[segment],
                          segment_tau_blinder, segment_tau, metadata, checked_xi);
                if (!zkdl::cpu_g1_equal(
                        zkdl::cpu_g1_from_native(opening.proof.xi), segment_xi)) {
                    if (error) *error = "segment Hyrax xi differs from authenticated statement vector";
                    return false;
                }
                segment_openings.push_back(opening.proof);
                value = zkdl::fr_add(value, zkdl::fr_mul(high_weights[segment], segment_value));
                tau_blinder = zkdl::fr_add(
                    tau_blinder, zkdl::fr_mul(high_weights[segment], segment_tau_blinder));
                aggregate_tau = zkdl::cpu_g1_add(
                    aggregate_tau, zkdl::cpu_g1_mul(segment_tau, high_weights[segment]));
                aggregate_xi = zkdl::cpu_g1_add(
                    aggregate_xi, zkdl::cpu_g1_mul(segment_xi, high_weights[segment]));
                progress->update("terminal_opening_segments", segment + 1U,
                                 layout.segment_count, tensor_index_u32,
                                 static_cast<std::uint32_t>(tensor_providers.size()));
            }
            const zkdl::CpuG1Point terminal_commitment = aggregate_tau;
            if (!zkdl::cpu_pedersen_strict_g1(terminal_commitment, error) ||
                terminal_commitment.is_zero()) {
                if (error && error->empty()) *error = "terminal scalar commitment is invalid";
                return false;
            }
            proof.terminal_commitments.push_back(zkdl::cpu_g1_to_native(terminal_commitment));
            proof.terminal_openings.push_back(segment_openings.front());
            proof.terminal_segment_openings.push_back(std::move(segment_openings));
            result.terminal_values.push_back(value);
            result.terminal_blinders.push_back(tau_blinder);
            states.emplace(terminal_index, ScalarState{value, tau_blinder, terminal_commitment});
        }

        result.product_blinders.reserve(plans.size());
        for (std::size_t plan_index = 0U; plan_index < plans.size(); ++plan_index) {
            const auto& plan = plans[plan_index];
            if (plan.right_alias_source_index !=
                std::numeric_limits<std::uint32_t>::max()) {
                std::uint32_t expected_alias = 0U;
                const auto source = states.find(plan.right_alias_source_index);
                if (plan.right_alias_source_index != plan.left_commitment_index ||
                    !zktr_product_alias_index(plan_index, expected_alias, error) ||
                    plan.right_commitment_index != expected_alias ||
                    source == states.end() ||
                    states.find(plan.right_commitment_index) != states.end()) {
                    if (error && error->empty())
                        *error = "terminal square-product alias does not match its opened source claim";
                    return false;
                }
                if (!states.emplace(plan.right_commitment_index,
                                    source->second).second) {
                    if (error) *error = "terminal square-product alias index collision";
                    return false;
                }
            }
            const auto left = states.find(plan.left_commitment_index);
            const auto right = states.find(plan.right_commitment_index);
            if (left == states.end() || right == states.end()) {
                if (error) *error = "terminal product plan references an unavailable claim";
                return false;
            }
            const FrWire output_value = zkdl::fr_mul(left->second.value, right->second.value);
            const FrWire output_blinder = zkdl::cpu_fresh_csprng_scalar();
            const zkdl::CpuG1Point output_commitment = zkdl::cpu_pedersen_scalar_commit(
                output_value, output_blinder, blinding_generator, u_generator);
            const auto product_context = zktr_make_product_context(
                statement, descriptor_digest, blinding_generator, u_generator,
                plan.left_commitment_index, plan.right_commitment_index,
                plan.output_commitment_index);
            const auto product = zkdl::cpu_open_pedersen_product(
                left->second.value, right->second.value, left->second.blinder,
                right->second.blinder, output_blinder, left->second.commitment,
                right->second.commitment, output_commitment, blinding_generator,
                u_generator, product_context);
            zkdl::ZkProductNodeProof node;
            node.monomial_index = plan.monomial_index;
            node.factor_step = plan.factor_step;
            node.left_commitment_index = plan.left_commitment_index;
            node.right_commitment_index = plan.right_commitment_index;
            node.output_commitment_index = plan.output_commitment_index;
            node.output_commitment = zkdl::cpu_g1_to_native(output_commitment);
            node.product_proof = product.proof;
            proof.product_nodes.push_back(node);
            result.product_blinders.push_back(output_blinder);
            states.emplace(plan.output_commitment_index,
                           ScalarState{output_value, output_blinder, output_commitment});
            progress->update("terminal_product_nodes", plan_index + 1U, plans.size());
        }

        FrWire residual_value = descriptor.constant;
        FrWire residual_blinder{};
        zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_mul(
            u_generator, descriptor.constant);
        for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
            const auto state = states.find(monomial_indices[monomial]);
            if (state == states.end()) {
                if (error) *error = "terminal monomial claim is unavailable";
                return false;
            }
            const FrWire coefficient = descriptor.terms[monomial].coefficient;
            residual_value = zkdl::fr_add(
                residual_value, zkdl::fr_mul(coefficient, state->second.value));
            residual_blinder = zkdl::fr_add(
                residual_blinder, zkdl::fr_mul(coefficient, state->second.blinder));
            residual_commitment = zkdl::cpu_g1_add(
                residual_commitment,
                zkdl::cpu_g1_mul(state->second.commitment, coefficient));
        }
        if (zkdl::fr_compare(residual_blinder, FrWire{}) == 0 ||
            !zkdl::fr_canonical(residual_blinder) ||
            !zkdl::cpu_g1_equal(
                residual_commitment,
                zkdl::cpu_pedersen_scalar_commit(
                    residual_value, residual_blinder, blinding_generator, u_generator))) {
            if (error) *error = "BLOCKED: terminal residual blinder/commitment is invalid";
            return false;
        }
        proof.residual_commitment = zkdl::cpu_g1_to_native(residual_commitment);
        result.residual_blinder = residual_blinder;
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            const zkdl::CpuG1Point expected_external = zkdl::cpu_pedersen_scalar_commit(
                residual_value, external_claim_blinder, blinding_generator, u_generator);
            if (!zkdl::cpu_g1_equal(expected_external, external_claim_commitment)) {
                if (error) *error = "external terminal claim commitment mismatch";
                return false;
            }
            proof.external_claim_commitment =
                zkdl::cpu_g1_to_native(external_claim_commitment);
            proof.external_equality_binding = zkdl::cpu_open_pedersen_scalar_equality(
                residual_value, residual_blinder, external_claim_blinder,
                residual_commitment, external_claim_commitment, blinding_generator,
                u_generator, proof.context).proof;
        } else {
            if (!same_wire(residual_value, expected_public_value)) {
                if (error) *error = "terminal residual does not match public value";
                return false;
            }
            proof.external_claim_commitment = proof.residual_commitment;
            proof.public_value_binding = zkdl::cpu_open_pedersen_scalar_value_public(
                residual_value, residual_blinder, residual_commitment,
                blinding_generator, u_generator, proof.context).proof;
        }
        std::string shape_error;
        if (!zkdl::zk_terminal_relation_shape_valid(proof, &shape_error)) {
            if (error) *error = shape_error.empty() ? "invalid terminal relation proof" : shape_error;
            return false;
        }
        zktr_absorb_header(parent_transcript, proof, descriptor, statement, operation_context,
                           evaluation_point, commitment_indices, canonical_commitments,
                           blinding_generator, u_generator, link_kind,
                           external_claim_commitment, expected_public_value);
        zktr_absorb_components(parent_transcript, proof, operation_context);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool prove_zk_terminal_relation(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::vector<FrWire>>& tensor_values,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<FrWire>& tensor_blinders,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& external_claim_blinder,
    const FrWire& expected_public_value,
    const std::string& operation_context,
    zkdl::Transcript& parent_transcript,
    ZkTerminalRelationProverResult& result,
    std::string* error = nullptr) {
    std::vector<SegmentedTensorProvider> providers;
    if (!make_segmented_tensor_providers(tensor_values, generators.size(), providers, error))
        return false;
    return prove_zk_terminal_relation_segmented(
        descriptor, providers, canonical_commitments, tensor_blinders,
        commitment_indices, evaluation_point, generators, blinding_generator,
        u_generator, statement, link_kind, external_claim_commitment,
        external_claim_blinder, expected_public_value, operation_context,
        parent_transcript, result, error);
}

inline bool verify_zk_terminal_relation(
    const zkdl::ZkTerminalRelationProof& proof,
    const ConstraintDescriptor& descriptor,
    const std::vector<zkdl::CpuG1Point>& canonical_commitments,
    const std::vector<std::uint32_t>& commitment_indices,
    const std::vector<FrWire>& evaluation_point,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::ZkTerminalClaimLinkKind link_kind,
    const zkdl::CpuG1Point& external_claim_commitment,
    const FrWire& expected_public_value,
    const std::string& operation_context,
    zkdl::Transcript& parent_transcript,
    ZkTerminalRelationVerifierResult& result,
    std::string* error = nullptr,
    const std::vector<CommitmentWire>* commitment_wires = nullptr,
    std::uint32_t shared_segment_size = 0U,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr) {
    try {
        if (!zktr_link_kind_valid(link_kind) ||
            !zktr_validate_public_inputs(
                descriptor, canonical_commitments.size(), canonical_commitments,
                commitment_indices, evaluation_point, generators, blinding_generator,
                u_generator, statement, operation_context, error,
                shared_segment_size, authenticated_basis) ||
            !zkdl::fr_canonical(expected_public_value) ||
            proof.claim_link_kind != link_kind ||
            !zkdl::zk_terminal_relation_shape_valid(proof, error)) {
            if (error && error->empty()) *error = "invalid terminal relation verifier inputs";
            return false;
        }
        if (authenticated_basis != nullptr &&
            !zktr_basis_valid_for_prefix(
                generators, blinding_generator, u_generator,
                static_cast<std::uint32_t>(generators.size()),
                authenticated_basis, error)) return false;
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::cpu_pedersen_strict_g1(external_claim_commitment, error) ||
                external_claim_commitment.is_zero() ||
                !zkdl::cpu_g1_equal(
                    external_claim_commitment,
                    zkdl::cpu_g1_from_native(proof.external_claim_commitment))) {
                if (error && error->empty()) *error = "external terminal claim mismatch";
                return false;
            }
        } else if (!external_claim_commitment.is_zero()) {
            if (error) *error = "public terminal verifier carries an external claim";
            return false;
        }

        const std::array<byte, 32U> descriptor_digest =
            zksc_proof_descriptor_digest(descriptor, statement, operation_context);
        const auto expected_context = zktr_make_link_context(
            descriptor, statement, operation_context, link_kind, descriptor_digest,
            blinding_generator, u_generator);
        if (!zkdl::cpu_pedersen_context_equal(proof.context, expected_context) ||
            proof.context.descriptor_digest != descriptor_digest ||
            proof.context.registry_digest != statement.registry_digest ||
            proof.context.srs_digest != statement.srs_digest) {
            if (error) *error = "terminal relation link context mismatch";
            return false;
        }
        if (proof.terminal_commitments.size() != canonical_commitments.size() ||
            proof.terminal_openings.size() != canonical_commitments.size()) {
            if (error) *error = "terminal relation commitment/opening count mismatch";
            return false;
        }
        if (commitment_wires != nullptr &&
            !relation_segment_commitments_valid(statement, *commitment_wires, error))
            return false;
        if (commitment_wires != nullptr) {
            for (std::size_t tensor_index = 0U;
                 tensor_index < commitment_indices.size(); ++tensor_index) {
                const std::uint32_t index = commitment_indices[tensor_index];
                if (index >= commitment_wires->size() ||
                    !segment_commitment_wire_shape_valid((*commitment_wires)[index], error)) {
                    if (error && error->empty()) *error = "terminal segment commitment wire index is invalid";
                    return false;
                }
                const auto& wire = (*commitment_wires)[index];
                zkdl::CpuG1Point legacy = zkdl::CpuG1Point::zero();
                for (const auto& segment : wire.segment_commitments)
                    legacy = zkdl::cpu_g1_add(
                        legacy, zkdl::cpu_g1_from_native(segment.commitment));
                if (!zkdl::cpu_g1_equal(
                        legacy, zkdl::cpu_g1_from_native(wire.commitment)) ||
                    !zkdl::cpu_g1_equal(
                        canonical_commitments[tensor_index],
                        zkdl::cpu_g1_from_native(wire.commitment))) {
                    if (error) *error = "legacy canonical handle is not the ordered segment sum";
                    return false;
                }
            }
        }
        zktr_absorb_header(parent_transcript, proof, descriptor, statement, operation_context,
                           evaluation_point, commitment_indices, canonical_commitments,
                           blinding_generator, u_generator, link_kind,
                           external_claim_commitment, expected_public_value);

        std::map<std::uint32_t, zkdl::CpuG1Point> states;
        result = ZkTerminalRelationVerifierResult{};
        result.terminal_commitments.reserve(proof.terminal_commitments.size());
        for (std::size_t tensor_index = 0U;
             tensor_index < proof.terminal_commitments.size(); ++tensor_index) {
            const zkdl::CpuG1Point expected_terminal_commitment = zkdl::cpu_g1_from_native(
                proof.terminal_commitments[tensor_index]);
            const std::uint32_t terminal_index =
                zktr_terminal_evaluation_index(tensor_index);
            const auto& segment_openings = proof.terminal_segment_openings[tensor_index];
            const auto& first_opening = segment_openings.front();
            ZkRelationSegmentLayout layout;
            const bool layout_valid = shared_segment_size == 0U
                ? zktr_segment_layout(first_opening.logical_domain, generators.size(),
                                      layout, error)
                : zktr_segment_layout_for_size(first_opening.logical_domain,
                                               generators.size(),
                                               shared_segment_size, layout, error);
            if (!layout_valid || segment_openings.size() != layout.segment_count) {
                if (error && error->empty())
                    *error = "terminal opening count differs from authenticated shared segment layout";
                return false;
            }
            const CommitmentWire* statement_commitment = nullptr;
            if (commitment_wires != nullptr) {
                if (commitment_indices[tensor_index] >= commitment_wires->size()) {
                    if (error) *error = "terminal statement commitment index is out of bounds";
                    return false;
                }
                statement_commitment = &(*commitment_wires)[commitment_indices[tensor_index]];
                if (statement_commitment->segment_count != layout.segment_count ||
                    statement_commitment->segment_size != layout.segment_size ||
                    statement_commitment->domain_size != layout.logical_domain ||
                    statement_commitment->segment_commitments.size() != layout.segment_count) {
                    if (error) *error = "terminal statement segment layout mismatch";
                    return false;
                }
            }
            const std::vector<FrWire> low_point =
                zktr_segment_point(evaluation_point, layout.segment_size);
            std::vector<zkdl::CpuG1Point> segment_generators;
            if (authenticated_basis == nullptr)
                segment_generators.assign(
                    generators.begin(), generators.begin() + layout.segment_size);
            const std::vector<FrWire> high_point =
                zktr_segment_prefix_point(evaluation_point, layout.segment_size);
            const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
            if (high_weights.size() != layout.segment_count) {
                if (error) *error = "segmented terminal high-bit MLE shape mismatch";
                return false;
            }
            for (const auto& weight : high_weights)
                if (zkdl::fr_compare(weight, FrWire{}) == 0) {
                    if (error) *error = "segmented terminal high-bit binding weight is zero";
                    return false;
                }
            zkdl::CpuG1Point aggregate_tau = zkdl::CpuG1Point::zero();
            zkdl::CpuG1Point aggregate_xi = zkdl::CpuG1Point::zero();
            for (std::size_t segment = 0U; segment < segment_openings.size(); ++segment) {
                const auto& opening = segment_openings[segment];
                auto metadata = zktr_make_opening_metadata(
                    descriptor_digest, statement, evaluation_point,
                    commitment_indices[tensor_index], terminal_index);
                metadata.logical_domain = layout.logical_domain;
                if (!checked_relation_u32(segment, metadata.segment_index,
                                          "terminal opening segment index", error))
                    return false;
                metadata.logical_offset = static_cast<std::uint64_t>(segment) * layout.segment_size;
                if (!checked_relation_u32(
                        std::min<std::uint64_t>(
                            layout.segment_size,
                            layout.logical_domain - metadata.logical_offset),
                        metadata.actual_length,
                        "terminal opening actual length", error))
                    return false;
                metadata.padded_length = layout.segment_size;
                if (opening.commitment_index != commitment_indices[tensor_index] ||
                    opening.evaluation_commitment_index != terminal_index ||
                    opening.generator_size != layout.segment_size ||
                    opening.outer_point.size() != evaluation_point.size() ||
                    (commitment_wires != nullptr &&
                     (commitment_indices[tensor_index] >= commitment_wires->size() ||
                      segment >= (*commitment_wires)[commitment_indices[tensor_index]]
                          .segment_commitments.size() ||
                      opening.segment_index != (*commitment_wires)[commitment_indices[tensor_index]]
                          .segment_commitments[segment].segment_index ||
                      opening.logical_domain != (*commitment_wires)[commitment_indices[tensor_index]]
                          .domain_size ||
                      opening.logical_offset != (*commitment_wires)[commitment_indices[tensor_index]]
                          .segment_commitments[segment].logical_offset ||
                      opening.actual_length != (*commitment_wires)[commitment_indices[tensor_index]]
                          .segment_commitments[segment].actual_length ||
                      opening.padded_length != (*commitment_wires)[commitment_indices[tensor_index]]
                          .segment_commitments[segment].padded_length ||
                      !zkdl::cpu_g1_equal(
                          zkdl::cpu_g1_from_native(opening.xi),
                          zkdl::cpu_g1_from_native((*commitment_wires)[commitment_indices[tensor_index]]
                              .segment_commitments[segment].commitment)))) ||
                    !(authenticated_basis != nullptr
                          ? zkdl::cpu_verify_hyrax_v3_authenticated(
                                opening, zkdl::cpu_g1_from_native(opening.xi),
                                zkdl::cpu_g1_from_native(opening.tau),
                                *authenticated_basis, low_point, metadata, error,
                                msm_session)
                          : zkdl::cpu_verify_hyrax_v3(
                                opening, zkdl::cpu_g1_from_native(opening.xi),
                                zkdl::cpu_g1_from_native(opening.tau),
                                segment_generators, low_point, blinding_generator,
                                u_generator, metadata, error))) return false;
                aggregate_tau = zkdl::cpu_g1_add(
                    aggregate_tau, zkdl::cpu_g1_mul(
                        zkdl::cpu_g1_from_native(opening.tau), high_weights[segment]));
                aggregate_xi = zkdl::cpu_g1_add(
                    aggregate_xi, zkdl::cpu_g1_mul(
                        zkdl::cpu_g1_from_native(opening.xi), high_weights[segment]));
            }
            zkdl::CpuG1Point expected_weighted_xi = aggregate_xi;
            if (statement_commitment != nullptr) {
                expected_weighted_xi = zkdl::CpuG1Point::zero();
                for (std::size_t segment = 0U;
                     segment < segment_openings.size(); ++segment)
                    expected_weighted_xi = zkdl::cpu_g1_add(
                        expected_weighted_xi,
                        zkdl::cpu_g1_mul(
                            zkdl::cpu_g1_from_native(
                                statement_commitment->segment_commitments[segment].commitment),
                            high_weights[segment]));
            }
            if (!zkdl::cpu_g1_equal(aggregate_xi, expected_weighted_xi) ||
                !zkdl::cpu_g1_equal(aggregate_tau, expected_terminal_commitment) ||
                first_opening.logical_domain != proof.terminal_openings[tensor_index].logical_domain ||
                first_opening.segment_index != proof.terminal_openings[tensor_index].segment_index) {
                if (error) *error = "terminal weighted segment commitments do not match statement vector";
                return false;
            }
            if (!states.emplace(terminal_index, expected_terminal_commitment).second) {
                if (error) *error = "duplicate terminal evaluation commitment index";
                return false;
            }
            result.terminal_commitments.push_back(expected_terminal_commitment);
        }

        std::vector<ZkTerminalProductPlan> plans;
        std::vector<std::uint32_t> monomial_indices;
        if (!zktr_build_product_plan(descriptor, canonical_commitments.size(),
                                     commitment_indices, plans, monomial_indices, error))
            return false;
        if (plans.size() != proof.product_nodes.size()) {
            if (error) *error = "terminal product DAG node count mismatch";
            return false;
        }
        for (std::size_t node_index = 0U; node_index < plans.size(); ++node_index) {
            const auto& plan = plans[node_index];
            const auto& node = proof.product_nodes[node_index];
            if (node.monomial_index != plan.monomial_index ||
                node.factor_step != plan.factor_step ||
                node.left_commitment_index != plan.left_commitment_index ||
                node.right_commitment_index != plan.right_commitment_index ||
                node.output_commitment_index != plan.output_commitment_index) {
                if (error) *error = "terminal product DAG order/ref mismatch";
                return false;
            }
            if (plan.right_alias_source_index !=
                std::numeric_limits<std::uint32_t>::max()) {
                std::uint32_t expected_alias = 0U;
                const auto source = states.find(plan.right_alias_source_index);
                if (plan.right_alias_source_index != plan.left_commitment_index ||
                    !zktr_product_alias_index(node_index, expected_alias, error) ||
                    plan.right_commitment_index != expected_alias ||
                    source == states.end() ||
                    states.find(plan.right_commitment_index) != states.end()) {
                    if (error && error->empty())
                        *error = "terminal square-product alias does not match its authenticated source claim";
                    return false;
                }
                if (!states.emplace(plan.right_commitment_index,
                                    source->second).second) {
                    if (error) *error = "terminal square-product alias index collision";
                    return false;
                }
            }
            const auto left = states.find(plan.left_commitment_index);
            const auto right = states.find(plan.right_commitment_index);
            if (left == states.end() || right == states.end()) {
                if (error) *error = "terminal product DAG references an unknown claim";
                return false;
            }
            const zkdl::CpuG1Point output = zkdl::cpu_g1_from_native(node.output_commitment);
            const auto product_context = zktr_make_product_context(
                statement, descriptor_digest, blinding_generator, u_generator,
                plan.left_commitment_index, plan.right_commitment_index,
                plan.output_commitment_index);
            if (!zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_a), left->second) ||
                !zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_b), right->second) ||
                !zkdl::cpu_g1_equal(
                    zkdl::cpu_g1_from_native(node.product_proof.commitment_c), output) ||
                !zkdl::cpu_verify_pedersen_product(
                    node.product_proof, left->second, right->second, output,
                    blinding_generator, u_generator, product_context, error))
                return false;
            if (!states.emplace(plan.output_commitment_index, output).second) {
                if (error) *error = "duplicate terminal product output index";
                return false;
            }
        }

        zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_mul(
            u_generator, descriptor.constant);
        for (std::size_t monomial = 0U; monomial < descriptor.terms.size(); ++monomial) {
            const auto state = states.find(monomial_indices[monomial]);
            if (state == states.end()) {
                if (error) *error = "terminal residual references an unknown monomial claim";
                return false;
            }
            residual_commitment = zkdl::cpu_g1_add(
                residual_commitment,
                zkdl::cpu_g1_mul(state->second, descriptor.terms[monomial].coefficient));
        }
        if (!zkdl::cpu_g1_equal(
                residual_commitment,
                zkdl::cpu_g1_from_native(proof.residual_commitment))) {
            if (error) *error = "terminal residual commitment equation mismatch";
            return false;
        }
        if (link_kind == zkdl::ZkTerminalClaimLinkKind::ExternalEquality) {
            if (!zkdl::cpu_verify_pedersen_scalar_equality(
                    proof.external_equality_binding, residual_commitment,
                    external_claim_commitment, blinding_generator, u_generator,
                    proof.context, error)) return false;
        } else if (!zkdl::cpu_verify_pedersen_scalar_value(
                       proof.public_value_binding, expected_public_value,
                       residual_commitment, blinding_generator, u_generator,
                       proof.context, error)) {
            return false;
        }
        zktr_absorb_components(parent_transcript, proof, operation_context);
        result.residual_commitment = residual_commitment;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline void fold_values(std::vector<FrWire>& values, const FrWire& challenge) {
    if (values.empty() || (values.size() & 1U) != 0U)
        throw std::invalid_argument("relation fold requires an even non-empty domain");
    std::vector<FrWire> folded(values.size() / 2U);
    const FrWire one_minus = zkdl::fr_sub(zkdl::fr_from_u32(1U), challenge);
    for (size_t i = 0; i < folded.size(); ++i)
        folded[i] = zkdl::fr_add(zkdl::fr_mul(one_minus, values[2U * i]),
                                 zkdl::fr_mul(challenge, values[2U * i + 1U]));
    values.swap(folded);
}

inline bool pointwise_shape_domain(const ConstraintDescriptor& descriptor, std::uint32_t domain, std::string* error = nullptr) {
    if (!descriptor.valid(error) || domain == 0U || (domain & (domain - 1U)) != 0U || descriptor.degree > 6U) return false;
    if (descriptor.logical_shape.empty()) return true;
    std::uint64_t product = 1U;
    for (const auto axis : descriptor.padded_shape) {
        if (product > domain / axis) { if (error) *error = "pointwise active shape exceeds authenticated domain"; return false; }
        product *= axis;
    }
    if (product != domain) { if (error) *error = "pointwise active shape/domain mismatch"; return false; }
    return true;
}

inline bool pointwise_active_slot(const ConstraintDescriptor& descriptor, std::uint64_t index) {
    for (std::size_t axis = descriptor.padded_shape.size(); axis > 0U; --axis) {
        if (index % descriptor.padded_shape[axis - 1U] >= descriptor.logical_shape[axis - 1U]) return false;
        index /= descriptor.padded_shape[axis - 1U];
    }
    return true;
}

inline std::vector<FrWire> pointwise_challenges(const ConstraintDescriptor& descriptor, std::uint32_t domain, std::size_t tensor_count, const std::string& context, zkdl::Transcript& transcript) {
    std::vector<byte> encoded;
    append_constraint(encoded, descriptor);
    zkdl::append_u32(encoded, domain);
    zkdl::append_u64(encoded, tensor_count);
    transcript.absorb(context + ".pointwise-eqmask-v2.plan", encoded);
    std::vector<FrWire> rho;
    for (std::uint32_t bit = 0U; bit < log2_exact(domain); ++bit) rho.push_back(zkdl::fr_wire(transcript.challenge(context + ".pointwise-eqmask-v2.rho." + std::to_string(bit))));
    return rho;
}

inline void pointwise_absorb_provider_commitments(const std::string& context, std::uint32_t domain, const std::vector<std::string>& tags, const std::vector<zkdl::CpuG1Point>& canonical, const std::vector<std::vector<zkdl::CpuG1Point>>& segments, zkdl::Transcript& transcript, NativeOperationProgressReporter* progress = nullptr) {
    if (tags.size() != canonical.size() || tags.size() != segments.size()) throw std::invalid_argument("pointwise provider commitment count mismatch");
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, domain);
    zkdl::append_u64(encoded, tags.size());
    for (std::size_t index = 0U; index < tags.size(); ++index) {
        zkdl::append_string(encoded, tags[index]);
        zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(canonical[index]));
        zkdl::append_u64(encoded, segments[index].size());
        for (std::size_t segment = 0U; segment < segments[index].size(); ++segment) {
            std::uint32_t segment_index = 0U;
            if (!checked_relation_u32(segment, segment_index,
                                      "pointwise commitment segment index") ||
                segment >= kMaxRelationSegments)
                throw std::invalid_argument("pointwise commitment segment count exceeds cap");
            zkdl::append_u32(encoded, segment_index);
            zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(segments[index][segment]));
            if (progress != nullptr) progress->update("provider_commitment_transcript_segments", segment + 1U, segments[index].size(), static_cast<std::uint32_t>(index), static_cast<std::uint32_t>(segments.size()));
        }
    }
    transcript.absorb(context + ".pointwise-eqmask-v2.provider-commitments", encoded);
}

inline ConstraintDescriptor pointwise_sumcheck_descriptor(const ConstraintDescriptor& descriptor, std::size_t tensor_count) {
    std::uint32_t first_aux_index = 0U, second_aux_index = 0U;
    if (!checked_relation_u32(tensor_count, first_aux_index,
                              "pointwise auxiliary tensor index") ||
        tensor_count == std::numeric_limits<std::size_t>::max() ||
        !checked_relation_u32(tensor_count + 1U, second_aux_index,
                              "pointwise auxiliary tensor index"))
        throw std::invalid_argument("pointwise tensor count exceeds uint32 index space");
    ConstraintDescriptor expanded = descriptor;
    expanded.degree += 2U;
    expanded.constant = FrWire{};
    for (auto& term : expanded.terms) { term.refs.push_back(first_aux_index); term.refs.push_back(second_aux_index); }
    if (descriptor.constant != FrWire{}) expanded.terms.push_back(Monomial{descriptor.constant, {first_aux_index, second_aux_index}});
    return expanded;
}

inline bool pointwise_eqmask_weight_table_peak_bytes(
    std::uint32_t bit_count, std::uint64_t& peak_bytes) noexcept;

struct PointwiseEqmaskWeightTable {
    static constexpr std::uint32_t kChunkBits = 8U;
    static constexpr std::uint32_t kMaxBits = kMaxRelationPointBits;
    static constexpr std::size_t kMaxChunks =
        (kMaxBits + kChunkBits - 1U) / kChunkBits;
    struct Chunk {
        std::uint32_t bit_offset = 0U;
        std::uint32_t bit_count = 0U;
        std::uint32_t entry_count = 0U;
        std::uint64_t entry_offset = 0U;
    };

    std::array<Chunk, kMaxChunks> chunks{};
    std::unique_ptr<FrWire[]> weights;
    std::uint32_t bits = 0U;
    std::uint32_t chunk_count = 0U;
    std::uint64_t entry_count = 0U;
    std::uint64_t peak_live_bytes = 0U;

    bool build(const std::vector<FrWire>& rho, std::string* error = nullptr) {
        clear();
        if (rho.size() > kMaxBits) {
            if (error) *error = "pointwise eqmask challenge count exceeds the native point cap";
            return false;
        }
        for (const auto& challenge : rho) {
            if (!zkdl::fr_canonical(challenge)) {
                if (error) *error = "pointwise eqmask challenge is not canonical";
                return false;
            }
        }
        bits = static_cast<std::uint32_t>(rho.size());
        std::uint64_t payload_entries = 0U;
        for (std::uint32_t offset = 0U; offset < bits; offset += kChunkBits) {
            const std::uint32_t chunk_bits =
                std::min<std::uint32_t>(kChunkBits, bits - offset);
            const std::uint32_t entries = 1U << chunk_bits;
            if (payload_entries > std::numeric_limits<std::uint64_t>::max() - entries) {
                if (error) *error = "pointwise eqmask weight-table entry count overflows";
                clear();
                return false;
            }
            chunks[chunk_count++] = Chunk{offset, chunk_bits, entries, payload_entries};
            payload_entries += entries;
        }
        std::uint64_t checked_peak_bytes = 0U;
        if (!pointwise_eqmask_weight_table_peak_bytes(bits, checked_peak_bytes) ||
            payload_entries > std::numeric_limits<std::size_t>::max() / sizeof(FrWire)) {
            if (error) *error = "pointwise eqmask weight-table memory bound overflows";
            clear();
            return false;
        }
        try {
            if (payload_entries != 0U)
                weights.reset(new FrWire[static_cast<std::size_t>(payload_entries)]);
        } catch (const std::exception& exception) {
            if (error) *error = std::string("pointwise eqmask weight-table allocation failed: ") +
                                exception.what();
            clear();
            return false;
        }
        entry_count = payload_entries;
        peak_live_bytes = checked_peak_bytes;
        const FrWire one = zkdl::fr_from_u32(1U);
        for (std::uint32_t chunk_index = 0U;
             chunk_index < chunk_count; ++chunk_index) {
            const Chunk& chunk = chunks[chunk_index];
            FrWire* table = weights.get() + chunk.entry_offset;
            table[0U] = one;
            for (std::uint32_t local_bit = 0U;
                 local_bit < chunk.bit_count; ++local_bit) {
                const FrWire& challenge = rho[chunk.bit_offset + local_bit];
                const FrWire zero_weight = zkdl::fr_sub(one, challenge);
                const std::uint32_t half = 1U << local_bit;
                for (std::uint32_t prefix = 0U; prefix < half; ++prefix) {
                    const FrWire prior = table[prefix];
                    table[prefix] = zkdl::fr_mul(prior, zero_weight);
                    table[half + prefix] = zkdl::fr_mul(prior, challenge);
                }
            }
        }
        return true;
    }

    FrWire evaluate(std::uint64_t index) const {
        if (bits >= 64U || index >= (std::uint64_t{1U} << bits))
            throw std::out_of_range("pointwise eqmask lookup index exceeds its domain");
        FrWire value = zkdl::fr_from_u32(1U);
        for (std::uint32_t chunk_index = 0U;
             chunk_index < chunk_count; ++chunk_index) {
            const Chunk& chunk = chunks[chunk_index];
            const std::uint64_t mask = chunk.entry_count - 1U;
            const std::uint64_t entry = (index >> chunk.bit_offset) & mask;
            value = zkdl::fr_mul(
                value, weights[chunk.entry_offset + entry]);
        }
        return value;
    }

    void clear() noexcept {
        weights.reset();
        chunks = {};
        bits = 0U;
        chunk_count = 0U;
        entry_count = 0U;
        peak_live_bytes = 0U;
    }
};

inline bool pointwise_eqmask_weight_table_peak_bytes(
    std::uint32_t bit_count, std::uint64_t& peak_bytes) noexcept {
    peak_bytes = 0U;
    if (bit_count > PointwiseEqmaskWeightTable::kMaxBits) return false;
    std::uint64_t entries = 0U;
    for (std::uint32_t offset = 0U; offset < bit_count;
         offset += PointwiseEqmaskWeightTable::kChunkBits) {
        const std::uint32_t chunk_bits = std::min<std::uint32_t>(
            PointwiseEqmaskWeightTable::kChunkBits, bit_count - offset);
        const std::uint64_t chunk_entries = std::uint64_t{1U} << chunk_bits;
        if (entries > std::numeric_limits<std::uint64_t>::max() - chunk_entries)
            return false;
        entries += chunk_entries;
    }
    constexpr std::uint64_t allocator_reserve = 64U;
    constexpr std::uint64_t construction_scratch = 4U * sizeof(FrWire);
    const std::uint64_t metadata_bytes = sizeof(PointwiseEqmaskWeightTable) +
        2U * sizeof(void*) + sizeof(SegmentedTensorProvider::RangeReader) +
        allocator_reserve;
    if (entries > (std::numeric_limits<std::uint64_t>::max() - metadata_bytes -
                   construction_scratch) / sizeof(FrWire))
        return false;
    peak_bytes = metadata_bytes + entries * sizeof(FrWire) + construction_scratch;
    return true;
}

inline bool pointwise_providers(
    const ConstraintDescriptor& descriptor,
    const std::vector<SegmentedTensorProvider>& input,
    const std::vector<FrWire>& rho,
    std::vector<SegmentedTensorProvider>& output,
    std::string* error = nullptr,
    const PointwiseEqmaskWeightTable* eqmask_table = nullptr,
    NativeOperationProgressReporter* progress = nullptr) {
    if (input.empty() || !pointwise_shape_domain(descriptor, input.front().logical_domain(), error) || rho.size() != log2_exact(input.front().logical_domain())) return false;
    if (eqmask_table != nullptr && eqmask_table->bits != rho.size()) {
        if (error) *error = "pointwise eqmask table challenge count mismatch";
        return false;
    }
    output = input;
    const std::uint32_t domain = input.front().logical_domain(), segment_size = input.front().segment_size();
    if (eqmask_table != nullptr) {
        output.emplace_back(domain, segment_size,
            [domain, eqmask_table, progress](
                std::uint64_t offset, std::uint32_t count,
                std::vector<FrWire>& values, std::string* read_error) {
                if (offset > domain || count > domain - offset) {
                    if (read_error) *read_error = "pointwise eqmask range is out of bounds";
                    return false;
                }
                values.resize(count);
                for (std::uint32_t slot = 0U; slot < count; ++slot) {
                    const std::uint64_t index = offset + slot;
                    values[slot] = eqmask_table->evaluate(index);
                    if (progress != nullptr &&
                        ((slot & 4095U) == 4095U || slot + 1U == count))
                        progress->update("eqmask_8bit_weight_table", index + 1U,
                                         domain);
                }
                return true;
            });
    } else {
        output.emplace_back(domain, segment_size,
            [rho, domain](std::uint64_t offset, std::uint32_t count,
                          std::vector<FrWire>& values, std::string* read_error) {
                if (offset > domain || count > domain - offset) {
                    if (read_error) *read_error = "pointwise eqmask range is out of bounds";
                    return false;
                }
                values.assign(count, zkdl::fr_from_u32(1U));
                for (std::uint32_t slot = 0U; slot < count; ++slot) {
                    const std::uint64_t index = offset + slot;
                    for (std::size_t bit = 0U; bit < rho.size(); ++bit)
                        values[slot] = zkdl::fr_mul(
                            values[slot], ((index >> bit) & 1U)
                                ? rho[bit]
                                : zkdl::fr_sub(zkdl::fr_from_u32(1U), rho[bit]));
                }
                return true;
            });
    }
    output.emplace_back(domain, segment_size, [descriptor](std::uint64_t offset, std::uint32_t count, std::vector<FrWire>& values, std::string*) {
        values.resize(count);
        for (std::uint32_t slot = 0U; slot < count; ++slot) values[slot] = zkdl::fr_from_u32(pointwise_active_slot(descriptor, offset + slot) ? 1U : 0U);
        return true;
    });
    return true;
}

inline FrWire pointwise_prefix_mle(const std::vector<FrWire>& point, std::size_t offset, std::uint64_t domain, std::uint64_t active) {
    if (active == domain) return zkdl::fr_from_u32(1U);
    FrWire total{}, prefix = zkdl::fr_from_u32(1U);
    std::uint32_t bits = 0U;
    for (auto value = domain; value > 1U; value >>= 1U) ++bits;
    for (std::uint32_t bit = bits; bit > 0U; --bit) {
        const FrWire coordinate = point.at(offset + bit - 1U);
        if ((active >> (bit - 1U)) & 1U) { total = zkdl::fr_add(total, zkdl::fr_mul(prefix, zkdl::fr_sub(zkdl::fr_from_u32(1U), coordinate))); prefix = zkdl::fr_mul(prefix, coordinate); }
        else prefix = zkdl::fr_mul(prefix, zkdl::fr_sub(zkdl::fr_from_u32(1U), coordinate));
    }
    return total;
}

inline FrWire pointwise_terminal_weight(const ConstraintDescriptor& descriptor, const std::vector<FrWire>& rho, const std::vector<FrWire>& msb_point) {
    const std::vector<FrWire> point(msb_point.rbegin(), msb_point.rend());
    if (rho.size() != point.size()) throw std::invalid_argument("pointwise challenge/terminal point mismatch");
    FrWire weight = zkdl::fr_from_u32(1U);
    for (std::size_t bit = 0U; bit < rho.size(); ++bit) weight = zkdl::fr_mul(weight, zkdl::fr_add(zkdl::fr_mul(rho[bit], point[bit]), zkdl::fr_mul(zkdl::fr_sub(zkdl::fr_from_u32(1U), rho[bit]), zkdl::fr_sub(zkdl::fr_from_u32(1U), point[bit]))));
    std::size_t offset = 0U;
    for (std::size_t axis = descriptor.padded_shape.size(); axis > 0U; --axis) {
        weight = zkdl::fr_mul(weight, pointwise_prefix_mle(point, offset, descriptor.padded_shape[axis - 1U], descriptor.logical_shape[axis - 1U]));
        for (auto value = descriptor.padded_shape[axis - 1U]; value > 1U; value >>= 1U) ++offset;
    }
    return weight;
}

inline ConstraintDescriptor pointwise_terminal_descriptor(const ConstraintDescriptor& descriptor, const std::vector<FrWire>& rho, const std::vector<FrWire>& point) {
    ConstraintDescriptor terminal = descriptor;
    const FrWire weight = pointwise_terminal_weight(descriptor, rho, point);
    terminal.constant = zkdl::fr_mul(terminal.constant, weight);
    for (auto& term : terminal.terms) term.coefficient = zkdl::fr_mul(term.coefficient, weight);
    return terminal;
}

inline bool sorted_marker_pieces_valid(const std::vector<ConstraintDescriptor>& pieces,
                                      const std::vector<FrWire>& piece_rhos,
                                      std::uint32_t expected_count,
                                      std::string* error = nullptr) {
    const auto fail = [&](const char* detail) {
        if (error) *error = detail;
        return false;
    };
    if (pieces.size() != 2U || piece_rhos.size() != 2U ||
        pieces[0].id != "marker-count" || pieces[0].degree != 1U ||
        pieces[0].constant != zkdl::fr_neg(zkdl::fr_from_u32(expected_count)) ||
        pieces[0].terms.size() != 1U || pieces[0].terms[0].coefficient != zkdl::fr_from_u32(1U) ||
        pieces[0].terms[0].refs != std::vector<std::uint32_t>{0U} ||
        pieces[1].id != "marker-padding" || pieces[1].degree != 2U ||
        pieces[1].constant != FrWire{} || pieces[1].terms.size() != 2U ||
        pieces[1].terms[0].coefficient != zkdl::fr_from_u32(1U) ||
        pieces[1].terms[0].refs != std::vector<std::uint32_t>{0U} ||
        pieces[1].terms[1].coefficient != zkdl::fr_neg(zkdl::fr_from_u32(1U)) ||
        pieces[1].terms[1].refs != std::vector<std::uint32_t>{0U, 1U})
        return fail("sorted-lookup marker relation plan is not canonical");
    for (const auto& rho : piece_rhos)
        if (!zkdl::fr_canonical(rho) || rho == FrWire{})
            return fail("sorted-lookup marker relation challenge is invalid");
    return true;
}

inline ConstraintDescriptor sorted_marker_mixed_sumcheck_descriptor(
    const ConstraintDescriptor& relation,
    const std::vector<ConstraintDescriptor>& pieces,
    const std::vector<FrWire>& piece_rhos,
    std::size_t input_count) {
    std::uint32_t eq_index = 0U, active_index = 0U;
    if (!checked_relation_u32(input_count, eq_index, "marker eq-mask tensor index") ||
        input_count == std::numeric_limits<std::size_t>::max() ||
        !checked_relation_u32(input_count + 1U, active_index, "marker active-mask tensor index"))
        throw std::invalid_argument("sorted-lookup marker tensor count exceeds uint32 index space");
    ConstraintDescriptor mixed;
    mixed.id = relation.id + "/marker-count-hybrid-v1";
    mixed.degree = 4U;
    mixed.logical_shape = relation.logical_shape;
    mixed.padded_shape = relation.padded_shape;
    Monomial count_term = pieces[0].terms[0U];
    count_term.coefficient = zkdl::fr_mul(piece_rhos[0U], count_term.coefficient);
    mixed.terms.push_back(std::move(count_term));
    for (const auto& source : pieces[1U].terms) {
        Monomial padding_term = source;
        padding_term.coefficient = zkdl::fr_mul(piece_rhos[1U], source.coefficient);
        padding_term.refs.push_back(eq_index);
        padding_term.refs.push_back(active_index);
        mixed.terms.push_back(std::move(padding_term));
    }
    return mixed;
}

inline ConstraintDescriptor sorted_marker_mixed_terminal_descriptor(
    const ConstraintDescriptor& relation,
    const std::vector<ConstraintDescriptor>& pieces,
    const std::vector<FrWire>& piece_rhos,
    const std::vector<FrWire>& eq_rho,
    const std::vector<FrWire>& point) {
    ConstraintDescriptor terminal;
    terminal.id = relation.id + "/marker-count-hybrid-v1";
    terminal.degree = 4U;
    terminal.logical_shape = relation.logical_shape;
    terminal.padded_shape = relation.padded_shape;
    const FrWire padding_weight = pointwise_terminal_weight(relation, eq_rho, point);
    Monomial count_term = pieces[0U].terms[0U];
    count_term.coefficient = zkdl::fr_mul(piece_rhos[0U], count_term.coefficient);
    terminal.terms.push_back(std::move(count_term));
    for (const auto& source : pieces[1U].terms) {
        Monomial padding_term = source;
        padding_term.coefficient = zkdl::fr_mul(
            zkdl::fr_mul(piece_rhos[1U], padding_weight), source.coefficient);
        terminal.terms.push_back(std::move(padding_term));
    }
    return terminal;
}

inline bool verify_constraint(const ConstraintProof& proof,
                             const ConstraintDescriptor& expected,
                             const std::vector<std::string>& expected_tensor_refs,
                             const std::vector<CommitmentWire>& commitments,
                             const std::vector<zkdl::CpuG1Point>& generators,
                             const zkdl::CpuG1Point& ipa_blinding_generator,
                             const zkdl::CpuG1Point& ipa_u_generator,
                             const RelationStatement& statement,
                             zkdl::Transcript& transcript,
                             std::string* error = nullptr,
                             const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
                             zkdl::CpuG1MsmSession* msm_session = nullptr) {
    std::vector<byte> expected_bytes, actual_bytes;
    append_constraint(expected_bytes, expected);
    append_constraint(actual_bytes, proof.descriptor);
    if (expected_bytes != actual_bytes) {
        if (error) *error = "relation constraint descriptor mismatch";
        return false;
    }
    std::uint32_t expected_count = 0U;
    if (!checked_relation_u32(expected_tensor_refs.size(), expected_count,
                              "relation expected tensor count", error))
        return false;
    std::vector<std::uint32_t> expected_indices(expected_count);
    for (std::size_t index = 0U; index < expected_indices.size(); ++index)
        if (!checked_relation_u32(index, expected_indices[index],
                                  "relation expected tensor index", error))
            return false;
    if (expected_indices.empty() || proof.opening_tensor_indices != expected_indices ||
        proof.opening_commitment_indices.size() != expected_indices.size()) {
        if (error) *error = "relation constraint terminal tensor map is not canonical";
        return false;
    }
    std::uint32_t domain = 0U;
    std::vector<zkdl::CpuG1Point> canonical_commitments;
    canonical_commitments.reserve(expected_indices.size());
    for (std::size_t i = 0U; i < expected_indices.size(); ++i) {
        const std::uint32_t commitment_index = proof.opening_commitment_indices[i];
        if (commitment_index >= commitments.size() ||
            commitments[commitment_index].tensor_id != expected_tensor_refs[i]) {
            if (error) *error = "relation terminal commitment reference mismatch";
            return false;
        }
        if (domain == 0U) domain = commitments[commitment_index].domain_size;
        if (commitments[commitment_index].domain_size != domain || domain == 0U ||
            log2_exact(domain) == std::numeric_limits<std::uint32_t>::max()) {
            if (error) *error = "relation terminal tensors do not share an authenticated MLE domain";
            return false;
        }
        canonical_commitments.push_back(
            zkdl::cpu_g1_from_native(commitments[commitment_index].commitment));
    }
    ZkRelationSegmentLayout domain_layout;
    if (!zktr_segment_layout(domain, generators.size(), domain_layout, error)) {
        if (error && error->empty()) *error = "relation verifier has no authenticated segmented domain";
        return false;
    }
    const std::vector<zkdl::CpuG1Point>& terminal_generators = generators;
    const std::string operation_context = "gat.relation." + expected.id;
    if (!pointwise_shape_domain(expected, domain, error)) return false;
    const auto rho = pointwise_challenges(expected, domain, expected_tensor_refs.size(), operation_context, transcript);
    const ConstraintDescriptor weighted = pointwise_sumcheck_descriptor(expected, expected_tensor_refs.size());
    ZkSumcheckVerifierResult sumcheck_result;
    if (!verify_zk_sumcheck(proof.zk_sumcheck, weighted, weighted.degree, domain,
                            terminal_generators, ipa_blinding_generator,
                            ipa_u_generator, FrWire{}, statement, operation_context,
                            transcript, sumcheck_result, error)) return false;
    if (sumcheck_result.challenges.size() != log2_exact(domain)) {
        if (error) *error = "ZkSumcheck terminal point/domain mismatch";
        return false;
    }
    std::vector<FrWire> terminal_point;
    terminal_point.reserve(sumcheck_result.challenges.size());
    for (const auto& challenge : sumcheck_result.challenges)
        terminal_point.push_back(zkdl::fr_wire(challenge));
    std::reverse(terminal_point.begin(), terminal_point.end());
    const ConstraintDescriptor terminal_descriptor = pointwise_terminal_descriptor(expected, rho, terminal_point);
    ZkTerminalRelationVerifierResult terminal_result;
    if (!verify_zk_terminal_relation(
            proof.terminal_relation, terminal_descriptor, canonical_commitments,
            proof.opening_commitment_indices, terminal_point, terminal_generators,
            ipa_blinding_generator, ipa_u_generator, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            zkdl::cpu_g1_from_native(sumcheck_result.final_claim_commitment),
            FrWire{}, operation_context, transcript, terminal_result, error,
            &commitments, 0U, authenticated_basis, msm_session)) return false;
    return true;
}

inline ConstraintDescriptor batched_family_descriptor(
    const OperationDescriptor& operation, zkdl::Transcript& transcript,
    std::string* error = nullptr) {
    ConstraintDescriptor descriptor;
    descriptor.id = operation.id + "/batch/" + operation.batch_family;
    descriptor.degree = 1U;
    if (!operation.constraints.empty()) { descriptor.logical_shape = operation.constraints.front().logical_shape; descriptor.padded_shape = operation.constraints.front().padded_shape; }
    for (const auto& constraint : operation.constraints)
        descriptor.degree = std::max(descriptor.degree, constraint.degree);
    if (operation.batch_family.empty() || operation.constraints.empty() ||
        operation.constraints.size() > kMaxRelationConstraints) {
        if (error) *error = "relation batched family has an invalid base constraint stack";
        return ConstraintDescriptor{};
    }
    FrWire weight = zkdl::fr_from_u32(1U);
    for (std::size_t index = 0U; index < operation.constraints.size(); ++index) {
        const Fr_t challenge = transcript.challenge(
            "gat.relation." + operation.id + ".batch." + operation.batch_family +
            ".rho." + std::to_string(index));
        weight = zkdl::fr_wire(challenge);
        const auto& constraint = operation.constraints[index];
        descriptor.constant = zkdl::fr_add(
            descriptor.constant, zkdl::fr_mul(weight, constraint.constant));
        for (const auto& term : constraint.terms) {
            Monomial combined = term;
            combined.coefficient = zkdl::fr_mul(weight, term.coefficient);
            descriptor.terms.push_back(std::move(combined));
        }
    }
    if (descriptor.terms.empty() || descriptor.terms.size() > kMaxRelationTerms ||
        !descriptor.valid(error)) return ConstraintDescriptor{};
    return descriptor;
}

inline bool fill_constraint_openings(
    const ConstraintDescriptor& descriptor,
    const std::vector<std::string>& operation_refs,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, ConstraintProof& proof,
    std::string* error = nullptr,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts = nullptr,
    const std::string* spill_directory = nullptr) {
    if (operation_refs.empty()) {
        if (error) *error = "relation batched proof has no operation tensor references";
        return false;
    }
    std::vector<SegmentedTensorProvider> tensor_providers;
    std::vector<zkdl::CpuG1Point> canonical_commitments;
    std::vector<FrWire> tensor_blinders;
    std::vector<std::uint32_t> commitment_indices;
    tensor_providers.reserve(operation_refs.size());
    canonical_commitments.reserve(operation_refs.size());
    tensor_blinders.reserve(operation_refs.size());
    commitment_indices.reserve(operation_refs.size());
    std::uint32_t domain = 0U;
    for (const auto& ref : operation_refs) {
        const auto found = tensor_map.find(ref);
        if (found == tensor_map.end() || found->second == nullptr) {
            if (error) *error = "relation batched proof tensor reference is missing: " + ref;
            return false;
        }
        const TensorWitness& tensor = *found->second;
        if (domain == 0U) domain = tensor.domain_size;
        if (tensor.domain_size != domain || domain == 0U ||
            (domain & (domain - 1U)) != 0U) {
            if (error) *error = "relation proof tensor references do not share an authenticated MLE domain";
            return false;
        }
        const auto commitment = std::find_if(
            commitments.begin(), commitments.end(),
            [&](const CommitmentWire& value) { return value.tensor_id == ref; });
        if (commitment == commitments.end() || commitment->domain_size != domain) {
            if (error) *error = "relation proof commitment reference is missing or has the wrong domain: " + ref;
            return false;
        }
        SegmentedTensorProvider provider;
        if (!tensor.segmented(generators.size(), provider, error)) return false;
        tensor_providers.push_back(std::move(provider));
        canonical_commitments.push_back(zkdl::cpu_g1_from_native(commitment->commitment));
        tensor_blinders.push_back(relation_blinding_scalar(tensor, blinders));
        std::uint32_t commitment_index = 0U;
        if (!checked_relation_u32(
                static_cast<std::size_t>(std::distance(commitments.begin(), commitment)),
                commitment_index, "relation tensor commitment index", error))
            return false;
        commitment_indices.push_back(commitment_index);
    }
    try {
        if (domain == 0U || log2_exact(domain) == std::numeric_limits<std::uint32_t>::max() ||
            log2_exact(domain) > kMaxRelationPointBits) {
            if (error) *error = "relation proof domain exceeds the native MLE point cap";
            return false;
        }
        proof = ConstraintProof{};
        proof.descriptor = descriptor;
        proof.opening_tensor_indices.resize(operation_refs.size());
        std::iota(proof.opening_tensor_indices.begin(), proof.opening_tensor_indices.end(), 0U);
        proof.opening_commitment_indices = commitment_indices;
        ZkRelationSegmentLayout domain_layout;
        if (!zktr_segment_layout(domain, generators.size(), domain_layout, error)) return false;
        const std::vector<zkdl::CpuG1Point>& terminal_generators = generators;
        ZkSumcheckProverResult sumcheck_result;
        const std::string operation_context = "gat.relation." + descriptor.id;
        NativeOperationProgressReporter constraint_progress("constraint_proof", descriptor.id);
        const auto rho = pointwise_challenges(descriptor, domain, operation_refs.size(), operation_context, transcript);
        const ConstraintDescriptor weighted = pointwise_sumcheck_descriptor(descriptor, operation_refs.size());
        PointwiseEqmaskWeightTable eqmask_table;
        if (!eqmask_table.build(rho, error)) return false;
        std::vector<SegmentedTensorProvider> weighted_providers;
        if (!pointwise_providers(descriptor, tensor_providers, rho, weighted_providers,
                                 error, &eqmask_table, &constraint_progress)) return false;
        if (!prove_zk_sumcheck_segmented(
                weighted, weighted_providers, weighted.degree, domain, terminal_generators,
                ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                operation_context, transcript, sumcheck_result, error, 0U,
                zksc_folded_state_within_budget(
                    weighted_providers, weighted, eqmask_table.peak_live_bytes),
                "constraint_sumcheck", descriptor.id.c_str(),
                spill_directory, true, eqmask_table.peak_live_bytes)) return false;
        weighted_providers.clear();
        eqmask_table.clear();
        proof.zk_sumcheck = std::move(sumcheck_result.proof);
        std::vector<FrWire> terminal_point;
        terminal_point.reserve(sumcheck_result.challenges.size());
        for (const auto& challenge : sumcheck_result.challenges)
            terminal_point.push_back(zkdl::fr_wire(challenge));
        std::reverse(terminal_point.begin(), terminal_point.end());
        std::vector<SegmentCommitmentMaterial> terminal_materials;
        std::vector<const SegmentCommitmentValidationReceipt*> terminal_receipts;
        terminal_materials.reserve(operation_refs.size());
        if (validation_receipts != nullptr)
            terminal_receipts.reserve(operation_refs.size());
        if (segment_blinding_scalars != nullptr) {
            for (std::size_t ref_index = 0U; ref_index < operation_refs.size();
                 ++ref_index) {
                const auto& ref = operation_refs[ref_index];
                const auto commitment = std::find_if(
                    commitments.begin(), commitments.end(),
                    [&](const CommitmentWire& value) { return value.tensor_id == ref; });
                if (commitment == commitments.end()) {
                    if (error) *error = "terminal segment material commitment is missing: " + ref;
                    return false;
                }
                SegmentCommitmentMaterial material;
                if (!zktr_material_from_wire(
                        *commitment, *segment_blinding_scalars, material, error)) return false;
                if (validation_receipts != nullptr) {
                    const auto receipt = validation_receipts->find(ref);
                    if (receipt == validation_receipts->end() ||
                        !receipt->second.matches_context(
                            ref, commitment->visibility,
                            tensor_providers[ref_index], generators,
                            ipa_blinding_generator, ipa_u_generator,
                            authenticated_basis, *commitment, material, error)) {
                        if (error && error->empty())
                            *error = "terminal segment MSM receipt is missing or invalid";
                        return false;
                    }
                    terminal_receipts.push_back(&receipt->second);
                }
                terminal_materials.push_back(std::move(material));
            }
        }
        const ConstraintDescriptor terminal_descriptor = pointwise_terminal_descriptor(descriptor, rho, terminal_point);
        ZkTerminalRelationProverResult terminal_result;
        if (!prove_zk_terminal_relation_segmented(
                terminal_descriptor, tensor_providers, canonical_commitments, tensor_blinders,
                commitment_indices, terminal_point, terminal_generators,
                ipa_blinding_generator, ipa_u_generator, statement,
                zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
                zkdl::cpu_g1_from_native(sumcheck_result.proof.final_claim_commitment),
                sumcheck_result.final_q_blinder, FrWire{}, operation_context,
                transcript, terminal_result, error,
                segment_blinding_scalars != nullptr ? &terminal_materials : nullptr,
                0U, &constraint_progress, authenticated_basis, msm_session,
                validation_receipts != nullptr ? &terminal_receipts : nullptr)) return false;
        proof.terminal_relation = std::move(terminal_result.proof);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline std::vector<byte> serialize_constraint_payload(const ConstraintProof& proof) {
    std::vector<byte> encoded{'B', 'C', 'H', '2'};
    append_constraint_proof(encoded, proof);
    return encoded;
}

inline bool deserialize_constraint_payload(
    const std::vector<byte>& bytes, ConstraintProof& proof,
    std::string* error,
    std::uint64_t max_terminal_body_bytes) {
    if (bytes.size() < 4U || bytes[0] != 'B' || bytes[1] != 'C' ||
        bytes[2] != 'H' || bytes[3] != '2') {
        if (error) *error = "invalid batched relation proof magic";
        return false;
    }
    size_t offset = 4U;
    if (!read_constraint_proof(bytes, offset, proof,
                               max_terminal_body_bytes) ||
        offset != bytes.size()) {
        if (error) *error = "truncated or trailing batched relation proof";
        return false;
    }
    return true;
}

inline bool prove_sparse_route(
    const OperationDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, SparseRouteProof& route,
    std::string* error, SparseRouteProverSecrets* secrets = nullptr,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr);

inline bool verify_sparse_route(
    const OperationDescriptor& descriptor,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, const SparseRouteProof& route,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr);

inline OperationDescriptor fac_line8_selector_descriptor(
    const OperationDescriptor& operation, const FacBindingDescriptor& binding,
    std::size_t index) {
    OperationDescriptor selector;
    selector.id = operation.id + "/fac4dnn-line8/selector/" + std::to_string(index);
    selector.kind = RelationKind::SparseRoute;
    selector.degree = 2U;
    selector.source_refs = {binding.canonical_ref, binding.view_ref};
    selector.tensor_refs = {binding.canonical_ref, binding.view_ref};
    selector.mapping = operation.mapping + ";fac4dnn-public-selector";
    selector.padding = operation.padding + ";fac4dnn-selector-masked";
    selector.terminal_claim = operation.terminal_claim;
    selector.terminal_point = operation.terminal_point;
    selector.public_route = true;
    selector.requires_reindex = true;
    selector.selector_edge_domain = binding.view_outer_domain;
    selector.selector_node_domain = binding.canonical_outer_domain;
    selector.selector_feature_domain = binding.view_feature_domain;
    selector.selector_source_feature_domain = binding.canonical_feature_domain;
    selector.selector_src = binding.source_outer_index;
    selector.selector_dst = binding.source_outer_index;
    selector.selector_active = binding.selector_active;
    selector.selector_feature_map = binding.source_feature_index;
    selector.selector_source_ref = binding.canonical_ref;
    selector.selector_output_ref = binding.view_ref;
    return selector;
}

inline bool fac_line8_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(
        commitments.begin(), commitments.end(), [&](const CommitmentWire& commitment) {
            return commitment.tensor_id == tensor_id;
        });
    if (found == commitments.end() || found->domain_size == 0U ||
        found->domain_size > (std::uint32_t(1U) << kMaxRelationPointBits)) {
        if (error) *error = "FAC4DNN canonical commitment is missing or out of bounds: " + tensor_id;
        return false;
    }
    return checked_relation_u32(
        static_cast<std::size_t>(std::distance(commitments.begin(), found)),
        index, "FAC4DNN commitment index", error);
}

inline bool fac_line8_absorb_header(
    const OperationDescriptor& operation, const RelationStatement& statement,
    const std::vector<CommitmentWire>& commitments,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator, zkdl::Transcript& transcript,
    std::array<byte, 32U>& digest, std::string* error) {
    if (!operation.valid(error) || operation.fac_bindings.empty() ||
        operation.fac_bindings.size() > kMaxRelationOpenings ||
        !zktr_all_statement_digests_valid(statement, error)) return false;
    std::vector<byte> encoded;
    zkdl::append_u32(encoded, FacLine8Proof::kVersion);
    append_statement(encoded, statement);
    append_operation(encoded, operation);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(operation.fac_bindings.size()));
    for (const auto& binding : operation.fac_bindings) {
        std::uint32_t canonical_index = 0U, view_index = 0U;
        if (!fac_line8_commitment_index(
                commitments, binding.canonical_ref, canonical_index, error) ||
            !fac_line8_commitment_index(
                commitments, binding.view_ref, view_index, error)) return false;
        zkdl::append_string(encoded, binding.canonical_ref);
        zkdl::append_string(encoded, binding.view_ref);
        zkdl::append_u32(encoded, canonical_index);
        zkdl::append_u32(encoded, view_index);
        append_commitment(encoded, commitments[canonical_index]);
        append_commitment(encoded, commitments[view_index]);
    }
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(ipa_blinding_generator));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(ipa_u_generator));
    transcript.absorb("gat.fac4dnn.line8.header", encoded);
    digest = zkdl::sha256(encoded);
    return digest != std::array<byte, 32U>{};
}

inline std::vector<FrWire> fac_line8_rho_challenges(
    const OperationDescriptor& operation, zkdl::Transcript& transcript) {
    std::vector<FrWire> rho;
    rho.reserve(operation.fac_bindings.size());
    for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index)
        rho.push_back(zkdl::cpu_ipa_nonzero_challenge(
            transcript, "gat.fac4dnn.line8." + operation.id + ".rho." +
                            std::to_string(index)));
    return rho;
}

inline std::array<byte, 32U> fac_line8_aggregate_digest(
    const std::array<byte, 32U>& header_digest,
    const std::vector<FrWire>& rho,
    const std::vector<SparseRouteProof>& routes,
    NativeOperationProgressReporter* progress = nullptr) {
    std::vector<byte> encoded(header_digest.begin(), header_digest.end());
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(routes.size()));
    for (std::size_t index = 0U; index < routes.size(); ++index) {
        zkdl::append_fr(encoded, zkdl::fr_value(rho[index]));
        const std::vector<byte> route_bytes = routes[index].serialize();
        append_digest(encoded, zkdl::sha256(route_bytes));
        zkdl::append_g1(encoded, routes[index].terminal_binding.commitment_1);
        zkdl::append_g1(encoded, routes[index].terminal_binding.commitment_2);
        zkdl::append_g1(encoded, routes[index].sumcheck.final_claim_commitment);
        if (progress != nullptr) progress->update("aggregate_selector_digest", index + 1U, routes.size());
    }
    return zkdl::sha256(encoded);
}

inline zkdl::PedersenProofContext fac_line8_aggregate_context(
    const std::array<byte, 32U>& digest, const RelationStatement& statement,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator) {
    constexpr std::uint32_t kAggregateResidualIndex = 0xF0000100U;
    constexpr std::uint32_t kAggregateSumcheckIndex = 0xF0000101U;
    return zksc_make_context(
        zkdl::PedersenProofContext::kOperationScalarEquality, 1U, 0U,
        kAggregateResidualIndex, kAggregateSumcheckIndex,
        zkdl::cpu_ipa_basis_digest({}, ipa_blinding_generator,
                                    ipa_u_generator, 0U),
        digest, statement);
}

inline bool fac_line8_aggregate_commitments(
    const std::vector<SparseRouteProof>& routes, const std::vector<FrWire>& rho,
    zkdl::CpuG1Point& residual_commitment,
    zkdl::CpuG1Point& sumcheck_commitment, std::string* error,
    NativeOperationProgressReporter* progress = nullptr) {
    if (routes.empty() || routes.size() != rho.size()) {
        if (error) *error = "FAC4DNN aggregate terminal shape mismatch";
        return false;
    }
    residual_commitment = zkdl::CpuG1Point{};
    sumcheck_commitment = zkdl::CpuG1Point{};
    for (std::size_t index = 0U; index < routes.size(); ++index) {
        std::string point_error;
        if (!zkdl::pedersen_wire_g1_valid(
                routes[index].terminal_binding.commitment_1, &point_error) ||
            !zkdl::pedersen_wire_g1_valid(
                routes[index].terminal_binding.commitment_2, &point_error)) {
            if (error) *error = point_error.empty()
                ? "FAC4DNN aggregate contains an invalid terminal commitment"
                : point_error;
            return false;
        }
        residual_commitment = zkdl::cpu_g1_add(
            residual_commitment,
            zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(
                    routes[index].terminal_binding.commitment_1), rho[index]));
        sumcheck_commitment = zkdl::cpu_g1_add(
            sumcheck_commitment,
            zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(
                    routes[index].terminal_binding.commitment_2), rho[index]));
        if (progress != nullptr) progress->update("aggregate_selector_commitments", index + 1U, routes.size());
    }
    return true;
}

inline bool prove_fac_line8(
    const OperationDescriptor& operation,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, FacLine8Proof& proof,
    std::string* error = nullptr,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr) {
    try {
        NativeOperationProgressReporter fac_progress("fac_line8", operation.id);
        proof = FacLine8Proof{};
        std::array<byte, 32U> header_digest{};
        if (!fac_line8_absorb_header(
                operation, statement, commitments, ipa_blinding_generator,
                ipa_u_generator, transcript, header_digest, error)) return false;
        proof.selector_proofs.reserve(operation.fac_bindings.size());
        std::vector<SparseRouteProverSecrets> terminal_secrets;
        terminal_secrets.reserve(operation.fac_bindings.size());
        for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
            const OperationDescriptor selector = fac_line8_selector_descriptor(
                operation, operation.fac_bindings[index], index);
            SparseRouteProof route;
            SparseRouteProverSecrets secrets;
            std::string selector_error;
            if (!prove_sparse_route(
                    selector, tensor_map, blinders, generators,
                    ipa_blinding_generator, ipa_u_generator, commitments,
                    statement, transcript, route, &selector_error, &secrets,
                    segment_blinding_scalars, authenticated_basis, msm_session)) {
                if (error) *error = "FAC4DNN selector proof failed operation_id=" + operation.id +
                    " binding_index=" + std::to_string(index) + ": " + selector_error;
                return false;
            }
            const std::vector<byte> encoded = route.serialize();
            transcript.absorb(
                "gat.fac4dnn.line8." + operation.id + ".selector." +
                    std::to_string(index), encoded);
            proof.selector_proofs.push_back(std::move(route));
            terminal_secrets.push_back(secrets);
            fac_progress.update("selector_binding_proofs", index + 1U,
                                operation.fac_bindings.size(),
                                static_cast<std::uint32_t>(index),
                                static_cast<std::uint32_t>(operation.fac_bindings.size()));
        }
        const std::vector<FrWire> rho = fac_line8_rho_challenges(operation, transcript);
        zkdl::CpuG1Point residual_commitment, sumcheck_commitment;
        if (!fac_line8_aggregate_commitments(
                proof.selector_proofs, rho, residual_commitment,
                sumcheck_commitment, error, &fac_progress)) return false;
        FrWire aggregate_value{};
        FrWire residual_blinder{};
        FrWire sumcheck_blinder{};
        for (std::size_t index = 0U; index < terminal_secrets.size(); ++index) {
            aggregate_value = zkdl::fr_add(
                aggregate_value,
                zkdl::fr_mul(rho[index], terminal_secrets[index].terminal_value));
            residual_blinder = zkdl::fr_add(
                residual_blinder,
                zkdl::fr_mul(rho[index], terminal_secrets[index].residual_blinder));
            sumcheck_blinder = zkdl::fr_add(
                sumcheck_blinder,
                zkdl::fr_mul(rho[index], terminal_secrets[index].sumcheck_blinder));
            fac_progress.update("aggregate_selector_blinders", index + 1U,
                                terminal_secrets.size());
        }
        if (residual_blinder == FrWire{} || sumcheck_blinder == FrWire{}) {
            if (error) *error =
                "FAC4DNN aggregate blinder is zero; retry with fresh proof randomness";
            return false;
        }
        const auto aggregate_digest = fac_line8_aggregate_digest(
            header_digest, rho, proof.selector_proofs, &fac_progress);
        proof.aggregate_binding = zkdl::cpu_open_pedersen_scalar_equality(
            aggregate_value, residual_blinder, sumcheck_blinder,
            residual_commitment, sumcheck_commitment,
            ipa_blinding_generator, ipa_u_generator,
            fac_line8_aggregate_context(
                aggregate_digest, statement, ipa_blinding_generator,
                ipa_u_generator)).proof;
        std::vector<byte> aggregate_bytes;
        zkdl::serialize_pedersen_scalar_equality(
            proof.aggregate_binding, aggregate_bytes);
        transcript.absorb(
            "gat.fac4dnn.line8." + operation.id + ".aggregate",
            aggregate_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = "FAC4DNN proof failed operation_id=" + operation.id + ": " + exception.what();
        return false;
    }
}

inline bool verify_fac_line8(
    const OperationDescriptor& operation,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement,
    zkdl::Transcript& transcript, const FacLine8Proof& proof,
    std::string* error = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr) {
    try {
        if (proof.version != FacLine8Proof::kVersion ||
            operation.fac_bindings.empty() ||
            proof.selector_proofs.size() != operation.fac_bindings.size()) {
            if (error) *error = "FAC4DNN selector proof count/version mismatch";
            return false;
        }
        std::array<byte, 32U> header_digest{};
        if (!fac_line8_absorb_header(
                operation, statement, commitments, ipa_blinding_generator,
                ipa_u_generator, transcript, header_digest, error)) return false;
        for (std::size_t index = 0U; index < operation.fac_bindings.size(); ++index) {
            const OperationDescriptor selector = fac_line8_selector_descriptor(
                operation, operation.fac_bindings[index], index);
            if (!verify_sparse_route(
                    selector, commitments, generators, ipa_blinding_generator,
                    ipa_u_generator, statement, transcript,
                    proof.selector_proofs[index], error,
                    authenticated_basis, msm_session)) return false;
            const std::vector<byte> encoded = proof.selector_proofs[index].serialize();
            transcript.absorb(
                "gat.fac4dnn.line8." + operation.id + ".selector." +
                    std::to_string(index), encoded);
        }
        const std::vector<FrWire> rho = fac_line8_rho_challenges(operation, transcript);
        zkdl::CpuG1Point residual_commitment, sumcheck_commitment;
        if (!fac_line8_aggregate_commitments(
                proof.selector_proofs, rho, residual_commitment,
                sumcheck_commitment, error)) return false;
        const auto aggregate_digest = fac_line8_aggregate_digest(
            header_digest, rho, proof.selector_proofs);
        if (!zkdl::cpu_verify_pedersen_scalar_equality(
                proof.aggregate_binding, residual_commitment,
                sumcheck_commitment, ipa_blinding_generator, ipa_u_generator,
                fac_line8_aggregate_context(
                    aggregate_digest, statement, ipa_blinding_generator,
                    ipa_u_generator), error)) return false;
        std::vector<byte> aggregate_bytes;
        zkdl::serialize_pedersen_scalar_equality(
            proof.aggregate_binding, aggregate_bytes);
        transcript.absorb(
            "gat.fac4dnn.line8." + operation.id + ".aggregate",
            aggregate_bytes);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool descriptor_vector_valid(const std::vector<OperationDescriptor>& operations,
                                   std::string* error = nullptr) {
    std::set<std::string> ids;
    for (const auto& operation : operations) {
        if (!operation.valid(error) || !ids.insert(operation.id).second) {
            if (error && error->empty()) *error = "duplicate relation operation id";
            return false;
        }
    }
    return !operations.empty();
}

/* FAC4DNN Line8 and kind-2 matrix primitives are formal hidden relations.
 * Their proof functions below replay committed-coefficient Sumcheck,
 * terminal relation and canonical Hyrax openings.  The historical kind-1
 * primitive remains explicitly fail-closed. */
inline bool formal_relation_components_supported(
    const std::vector<OperationDescriptor>& operations,
    std::string* error = nullptr) {
    for (const auto& operation : operations) {
        for (const auto& primitive : operation.primitives)
            if (primitive.kind != 2U ||
                (operation.kind != RelationKind::Projection &&
                 operation.kind != RelationKind::AttentionInnerProduct)) {
                if (error) *error =
                    "BLOCKED: legacy/value-revealing relation primitive is not admitted";
                return false;
            }
    }
    return true;
}

inline const TensorDescriptor* find_tensor_descriptor(
    const std::vector<TensorDescriptor>& descriptors, const std::string& id) {
    const auto found = std::find_if(descriptors.begin(), descriptors.end(),
                                    [&](const TensorDescriptor& descriptor) {
                                        return descriptor.id == id;
                                    });
    return found == descriptors.end() ? nullptr : &*found;
}

inline bool commitment_equal(const CommitmentWire& left,
                             const CommitmentWire& right) noexcept {
    return left.tensor_id == right.tensor_id && left.visibility == right.visibility &&
           left.domain_size == right.domain_size &&
           zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(left.commitment),
                              zkdl::cpu_g1_from_native(right.commitment)) &&
           left.segment_binding_version == right.segment_binding_version &&
           left.segment_size == right.segment_size &&
           left.segment_count == right.segment_count &&
           left.srs_manifest_digest == right.srs_manifest_digest &&
           left.basis_digest == right.basis_digest &&
           left.segment_commitment_digest == right.segment_commitment_digest &&
           left.segment_commitments.size() == right.segment_commitments.size() &&
           std::equal(left.segment_commitments.begin(), left.segment_commitments.end(),
                      right.segment_commitments.begin(),
                      [](const SegmentCommitmentWire& a, const SegmentCommitmentWire& b) {
                          return a.segment_index == b.segment_index &&
                                 a.logical_offset == b.logical_offset &&
                                 a.actual_length == b.actual_length &&
                                 a.padded_length == b.padded_length &&
                                 a.padding_length == b.padding_length &&
                                 a.frame_digest == b.frame_digest &&
                                 zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(a.commitment),
                                                    zkdl::cpu_g1_from_native(b.commitment));
                      });
}

inline bool verify_tensor_contract(const VerifierContext& context,
                                   const std::vector<CommitmentWire>& commitments,
                                   std::string* error = nullptr) {
    std::set<std::string> descriptor_ids;
    for (const auto& descriptor : context.tensor_descriptors) {
        if (!descriptor.valid(error) || !descriptor_ids.insert(descriptor.id).second) {
            if (error && error->empty()) *error = "duplicate/invalid verifier tensor descriptor";
            return false;
        }
    }
    if (!context.tensor_descriptors.empty()) {
        for (const auto& commitment : commitments) {
            if (commitment.tensor_id.rfind("sorted_lookup/", 0U) == 0U) continue;
            const TensorDescriptor* descriptor = find_tensor_descriptor(
                context.tensor_descriptors, commitment.tensor_id);
            if (descriptor == nullptr || descriptor->domain_size != commitment.domain_size ||
                descriptor->visibility != commitment.visibility) {
                if (error) *error = "commitment does not match public tensor shape/padding contract";
                return false;
            }
        }
        for (const auto& descriptor : context.tensor_descriptors) {
            const auto found = std::find_if(commitments.begin(), commitments.end(),
                                            [&](const CommitmentWire& commitment) {
                                                return commitment.tensor_id == descriptor.id;
                                            });
            if (found == commitments.end()) {
                if (error) *error = "proof omits a tensor from the public shape contract: " + descriptor.id;
                return false;
            }
        }
    }
    for (const auto& operation : context.operations) {
        if (operation.batch_family == "indexed_public_table") {
            const TensorDescriptor* table = find_tensor_descriptor(
                context.tensor_descriptors, operation.public_table_ref);
            const auto table_values = context.public_tensor_values.find(
                operation.public_table_ref);
            if (table == nullptr || table->visibility != 1U ||
                (table_values == context.public_tensor_values.end() && context.public_tensor_providers.count(operation.public_table_ref) == 0U)) {
                if (error) *error = "indexed table relation lacks an authenticated public table descriptor/value map";
                return false;
            }
            if (table->padded_shape.size() != 1U ||
                operation.public_table_selector.size() != table->domain_size ||
                table->logical_shape.size() != 1U ||
                table->logical_shape.front() == 0U ||
                table->logical_shape.front() > table->domain_size) {
                if (error) *error = "indexed table Q_table selector does not match authenticated table shape";
                return false;
            }
            for (std::size_t index = 0U; index < operation.public_table_selector.size(); ++index) {
                const FrWire expected = index < table->logical_shape.front()
                    ? zkdl::fr_from_u32(1U) : zkdl::fr_from_u32(0U);
                if (operation.public_table_selector[index] != expected) {
                    if (error) *error = "indexed table Q_table selector is not canonical for the authenticated table";
                    return false;
                }
            }
        }
        std::uint32_t aligned_domain = 0U;
        for (const auto& ref : operation.tensor_refs) {
            const TensorDescriptor* descriptor = find_tensor_descriptor(
                context.tensor_descriptors, ref);
            if (descriptor == nullptr) continue;
            if (aligned_domain == 0U) aligned_domain = descriptor->domain_size;
            const bool has_route = operation.kind == RelationKind::SparseRoute ||
                                   !operation.selector_source_ref.empty();
            if (!has_route &&
                descriptor->domain_size != aligned_domain) {
                if (error) *error = "operation tensor views violate authenticated MLE padding/domain contract";
                return false;
            }
        }
        for (const auto& primitive : operation.primitives) {
            const std::array<std::uint64_t, 3U> expected = {
                primitive.matmul_shape.a_size(), primitive.matmul_shape.b_size(),
                primitive.matmul_shape.c_size()};
            for (std::size_t i = 0U; i < primitive.tensor_refs.size(); ++i) {
                const TensorDescriptor* descriptor = find_tensor_descriptor(
                    context.tensor_descriptors, primitive.tensor_refs[i]);
                if (descriptor == nullptr || descriptor->domain_size != expected[i]) {
                    if (error) *error = "zkMatMul primitive shape is not bound to public tensor descriptor";
                    return false;
                }
            }
        }
    }
    return true;
}

inline bool verify_model_registry(
    const std::vector<CommitmentWire>& commitments,
    const std::vector<CommitmentWire>& registered_model_commitments,
    std::string* error = nullptr) {
    std::set<std::string> ids;
    for (const auto& registered : registered_model_commitments) {
        if (registered.visibility != 2U || registered.tensor_id.empty() ||
            !ids.insert(registered.tensor_id).second) {
            if (error) *error = "authenticated model registry contains duplicate/non-model commitment";
            return false;
        }
        const auto found = std::find_if(commitments.begin(), commitments.end(),
                                       [&](const CommitmentWire& commitment) {
                                           return commitment.tensor_id == registered.tensor_id;
                                       });
        if (found == commitments.end() || !commitment_equal(*found, registered)) {
            if (error) *error = "model tensor commitment does not match authenticated registry/VK map";
            return false;
        }
    }
    for (const auto& commitment : commitments) {
        if (commitment.visibility != 2U) continue;
        const auto found = std::find_if(registered_model_commitments.begin(),
                                       registered_model_commitments.end(),
                                       [&](const CommitmentWire& registered) {
                                           return registered.tensor_id == commitment.tensor_id;
                                       });
        if (found == registered_model_commitments.end()) {
            if (error) *error = "model tensor is not present in authenticated registry/VK map";
            return false;
        }
    }
    return true;
}

inline bool verify_hiding_records(
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CommitmentRecord>& records,
    const std::vector<zkdl::CpuG1Point>& generators,
    std::string* error = nullptr) {
    std::set<std::string> labels;
    for (const auto& record : records) {
        if (!zkdl::CpuHidingCommitment::verify_record(record, error) ||
            record.label.empty() || !labels.insert(record.label).second) {
            if (error && error->empty()) *error = "invalid/duplicate native hiding commitment record";
            return false;
        }
        const auto found = std::find_if(commitments.begin(), commitments.end(),
                                       [&](const CommitmentWire& commitment) {
                                           return commitment.tensor_id == record.label;
                                       });
        const std::uint32_t expected_record_visibility =
            found == commitments.end() ? 0U : (found->visibility == 2U ? 1U :
                                               (found->visibility == 3U ? 2U : 0U));
        if (found == commitments.end() || expected_record_visibility == 0U ||
            record.visibility != expected_record_visibility ||
            record.points.size() != found->domain_size ||
            generators.size() < found->domain_size ||
            !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(record.commitment),
                                zkdl::cpu_g1_from_native(found->commitment))) {
            if (error) *error = "hiding commitment record is not bound to the canonical tensor commitment";
            return false;
        }
        for (std::size_t point = 0U; point < record.points.size(); ++point)
            if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(record.points[point]),
                                    generators[point])) {
                if (error) *error = "hiding commitment record generator vector is not authenticated";
                return false;
            }
    }
    for (const auto& commitment : commitments) {
        if (commitment.visibility == 1U) continue;
        if (!labels.count(commitment.tensor_id)) {
            if (error) *error = "private/model tensor is missing a native hiding commitment record";
            return false;
        }
    }
    return true;
}

inline bool hiding_records_equal(const std::vector<zkdl::CommitmentRecord>& left,
                                 const std::vector<zkdl::CommitmentRecord>& right) {
    if (left.size() != right.size()) return false;
    std::map<std::string, std::vector<byte>> lhs, rhs;
    for (const auto& record : left) {
        std::vector<byte> encoded;
        append_hiding_record(encoded, record);
        if (!lhs.emplace(record.label, std::move(encoded)).second) return false;
    }
    for (const auto& record : right) {
        std::vector<byte> encoded;
        append_hiding_record(encoded, record);
        if (!rhs.emplace(record.label, std::move(encoded)).second) return false;
    }
    return lhs == rhs;
}

inline void absorb_hiding_records(
    zkdl::Transcript& transcript,
    const std::vector<zkdl::CommitmentRecord>& records) {
    std::map<std::string, const zkdl::CommitmentRecord*> ordered;
    for (const auto& record : records) ordered.emplace(record.label, &record);
    std::vector<byte> encoded;
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(ordered.size()));
    for (const auto& entry : ordered) append_hiding_record(encoded, *entry.second);
    transcript.absorb("gat.relation.hiding-records", encoded);
}

inline bool verify_public_tensor_values(const VerifierContext& context, const std::vector<CommitmentWire>& commitments, std::string* error = nullptr) {
    const auto verify_provider = [&](const std::string& id, const SegmentedTensorProvider& provider) {
        const auto found = std::find_if(commitments.begin(), commitments.end(), [&](const CommitmentWire& commitment) { return commitment.tensor_id == id; });
        if (found == commitments.end() || (found->visibility != 1U && context.graph_public_tensor_ids.count(id) == 0U) || !provider.valid(error) || provider.logical_domain() != found->domain_size) { if (error) *error = "public tensor provider does not match commitment domain"; return false; }
        CommitmentWire expected;
        SegmentCommitmentMaterial material;
        if (!zktr_build_segment_commitment_material(
                id, 1U, provider, context.ipa_generators,
                context.ipa_blinding_generator, context.ipa_u_generator, FrWire{},
                context.statement, expected, material, error, nullptr, 0U, nullptr,
                &context.ipa_generators)) return false;
        if (expected.segment_size != found->segment_size || expected.segment_count != found->segment_count || expected.basis_digest != found->basis_digest || expected.srs_manifest_digest != found->srs_manifest_digest || expected.segment_commitments.size() != found->segment_commitments.size()) { if (error) *error = "public tensor segmented layout differs from authenticated material"; return false; }
        for (std::size_t index = 0U; index < expected.segment_commitments.size(); ++index) { const auto& wanted = expected.segment_commitments[index]; const auto& actual = found->segment_commitments[index]; if (wanted.segment_index != actual.segment_index || wanted.logical_offset != actual.logical_offset || wanted.actual_length != actual.actual_length || wanted.padded_length != actual.padded_length || wanted.padding_length != actual.padding_length || wanted.frame_digest != actual.frame_digest || !zkdl::cpu_g1_equal(material.commitments[index], zkdl::cpu_g1_from_native(actual.commitment))) { if (error) *error = "public tensor segment commitment is not graph/table-derived"; return false; } }
        return true;
    };
    for (const auto& entry : context.public_tensor_providers) if (!verify_provider(entry.first, entry.second)) return false;
    for (const auto& entry : context.public_tensor_values) {
        if (context.public_tensor_providers.count(entry.first) != 0U) { if (error) *error = "public tensor has duplicate dense/provider material"; return false; }
        SegmentedTensorProvider provider;
        if (!SegmentedTensorProvider::from_values(entry.second, context.ipa_generators.size(), provider, error) || !verify_provider(entry.first, provider)) return false;
    }
    return true;
}

inline bool primitive_tensor_values(
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::string& id, std::uint64_t expected_size,
    std::vector<Fr_t>& values, std::string* error) {
    const auto found = tensor_map.find(id);
    if (found == tensor_map.end() || found->second == nullptr ||
        expected_size > std::numeric_limits<std::uint32_t>::max() ||
        found->second->domain_size != expected_size ||
        found->second->values.size() != expected_size) {
        if (error) *error = "zkMatMul primitive tensor shape/value mismatch: " + id;
        return false;
    }
    values.clear();
    values.reserve(found->second->values.size());
    for (const auto& value : found->second->values) values.push_back(zkdl::fr_value(value));
    return true;
}

inline bool hidden_matmul_shape_valid(const zkdl::MatMulShape& shape,
                                      std::string* error = nullptr) {
    std::string shape_error;
    if (!shape.valid(&shape_error) || shape.num != 1U ||
        !zkdl::MatMulShape::power_of_two(shape.m) ||
        !zkdl::MatMulShape::power_of_two(shape.n) ||
        !zkdl::MatMulShape::power_of_two(shape.k) || shape.n < 2U) {
        if (error) *error = shape_error.empty()
                                  ? "hidden matrix shape must use power-of-two dimensions"
                                  : shape_error;
        return false;
    }
    const std::uint64_t max_domain = std::uint64_t(1U) << kMaxRelationPointBits;
    if (shape.a_size() == 0U || shape.b_size() == 0U || shape.c_size() == 0U ||
        shape.a_size() > max_domain || shape.b_size() > max_domain ||
        shape.c_size() > max_domain) {
        if (error) *error = "hidden matrix tensor domain exceeds authenticated point cap";
        return false;
    }
    try {
        if (zkdl::MatMulShape::log2_ceil(shape.num) +
                zkdl::MatMulShape::log2_ceil(shape.m) +
                zkdl::MatMulShape::log2_ceil(shape.n) > kMaxRelationPointBits ||
            zkdl::MatMulShape::log2_ceil(shape.num) +
                zkdl::MatMulShape::log2_ceil(shape.n) +
                zkdl::MatMulShape::log2_ceil(shape.k) > kMaxRelationPointBits ||
            zkdl::MatMulShape::log2_ceil(shape.num) +
                zkdl::MatMulShape::log2_ceil(shape.m) +
                zkdl::MatMulShape::log2_ceil(shape.k) > kMaxRelationPointBits) {
            if (error) *error = "hidden matrix MLE point exceeds native point cap";
            return false;
        }
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
    return true;
}

inline bool checked_matmul_shape_domains_u32(
    const zkdl::MatMulShape& shape, std::array<std::uint32_t, 3U>& domains,
    std::string* error = nullptr) {
    if (!shape.valid(error)) return false;
    std::array<std::uint64_t, 3U> wide_domains{};
    if (!shape.tensor_sizes(wide_domains)) {
        if (error) *error = "zkMatMul tensor shape product overflows uint64";
        return false;
    }
    for (std::size_t index = 0U; index < wide_domains.size(); ++index) {
        if (wide_domains[index] == 0U ||
            wide_domains[index] > (std::uint64_t(1U) << kMaxRelationPointBits) ||
            !checked_relation_u32(wide_domains[index], domains[index],
                                  "zkMatMul tensor domain", error)) {
            if (error && error->empty())
                *error = "zkMatMul tensor domain cannot be represented as uint32";
            return false;
        }
    }
    return true;
}

inline bool checked_hidden_matmul_primitive_index(
    std::size_t primitive_index, std::uint32_t& primitive_index_u32,
    std::string* error = nullptr) {
    if (!checked_relation_u32(primitive_index, primitive_index_u32,
                              "hidden matrix primitive index", error))
        return false;
    return true;
}

inline bool hidden_matmul_product_context_indices(
    std::size_t primitive_index, std::array<std::uint32_t, 3U>& indices,
    std::string* error = nullptr) {
    std::uint32_t primitive_index_u32 = 0U;
    if (!checked_hidden_matmul_primitive_index(primitive_index,
                                              primitive_index_u32, error) ||
        primitive_index_u32 > (0x0fffffffU - 2U) / 3U) {
        if (error && error->empty())
            *error = "hidden matrix product context index exceeds reserved range";
        return false;
    }
    const std::uint32_t base = 0x20000000U + primitive_index_u32 * 3U;
    indices = {{base, base + 1U, base + 2U}};
    return true;
}

inline ConstraintDescriptor hidden_matmul_constraint_descriptor(
    const OperationDescriptor& operation, std::size_t primitive_index) {
    ConstraintDescriptor descriptor;
    descriptor.id = operation.id + "/hidden-matmul/" +
                    std::to_string(primitive_index);
    descriptor.degree = 2U;
    descriptor.terms.push_back(Monomial{zkdl::fr_from_u32(1U), {0U, 1U}});
    descriptor.terms.push_back(Monomial{fr_from_signed_i64(-1), {2U}});
    return descriptor;
}

inline std::array<byte, 32U> hidden_matmul_primitive_digest(
    const OperationDescriptor& operation, const PrimitiveDescriptor& primitive,
    std::size_t primitive_index) {
    std::uint32_t primitive_index_u32 = 0U;
    if (!checked_hidden_matmul_primitive_index(primitive_index,
                                              primitive_index_u32))
        throw std::invalid_argument("hidden matrix primitive index exceeds uint32");
    std::vector<byte> encoded;
    zkdl::append_string(encoded, operation.id);
    zkdl::append_u32(encoded, primitive_index_u32);
    append_primitive(encoded, primitive);
    return zkdl::sha256(encoded);
}

inline bool hidden_matmul_derive_points(
    const OperationDescriptor& operation, std::size_t primitive_index,
    const zkdl::MatMulShape& shape, zkdl::Transcript& transcript,
    std::vector<Fr_t>& row_point, std::vector<Fr_t>& column_point,
    std::vector<Fr_t>& batch_point, std::string* error = nullptr) {
    try {
        std::uint32_t primitive_index_u32 = 0U;
        if (!checked_hidden_matmul_primitive_index(primitive_index,
                                                  primitive_index_u32, error))
            return false;
        const std::string prefix = "gat.relation." + operation.id +
                                   ".hidden-matmul." +
                                   std::to_string(primitive_index_u32);
        row_point.clear();
        column_point.clear();
        batch_point.clear();
        for (std::uint32_t bit = 0U;
             bit < zkdl::MatMulShape::log2_ceil(shape.m); ++bit)
            row_point.push_back(transcript.challenge(prefix + ".row." +
                                                     std::to_string(bit)));
        for (std::uint32_t bit = 0U;
             bit < zkdl::MatMulShape::log2_ceil(shape.k); ++bit)
            column_point.push_back(transcript.challenge(prefix + ".column." +
                                                        std::to_string(bit)));
        for (std::uint32_t bit = 0U;
             bit < zkdl::MatMulShape::log2_ceil(shape.num); ++bit)
            batch_point.push_back(transcript.challenge(prefix + ".batch." +
                                                       std::to_string(bit)));
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline std::vector<FrWire> hidden_matmul_index_point(std::uint32_t index,
                                                      std::uint32_t bits) {
    std::vector<FrWire> point;
    point.reserve(bits);
    for (std::uint32_t bit = 0U; bit < bits; ++bit)
        point.push_back(zkdl::fr_from_u32((index >> bit) & 1U));
    return point;
}

inline std::vector<FrWire> hidden_matmul_concat_points(
    const std::vector<FrWire>& first, const std::vector<FrWire>& second,
    const std::vector<FrWire>& third) {
    std::vector<FrWire> point;
    point.reserve(first.size() + second.size() + third.size());
    point.insert(point.end(), first.begin(), first.end());
    point.insert(point.end(), second.begin(), second.end());
    point.insert(point.end(), third.begin(), third.end());
    return point;
}

inline std::vector<FrWire> hidden_matmul_wire_points(const std::vector<Fr_t>& values) {
    std::vector<FrWire> point;
    point.reserve(values.size());
    for (const auto& value : values) point.push_back(zkdl::fr_wire(value));
    return point;
}

inline bool hidden_matmul_line_values(const SegmentedTensorProvider& provider, const zkdl::MatMulShape& shape, const std::vector<Fr_t>& row_point, const std::vector<Fr_t>& column_point, const std::vector<Fr_t>& batch_point, bool left_tensor, std::vector<FrWire>& line, std::string* error, NativeOperationProgressReporter* progress = nullptr) {
    if (!provider.valid(error) || provider.logical_domain() != (left_tensor ? shape.a_size() : shape.b_size())) return false;
    const auto row_weights = zkdl::cpu_ipa_mle_weights(hidden_matmul_wire_points(row_point)), column_weights = zkdl::cpu_ipa_mle_weights(hidden_matmul_wire_points(column_point)), batch_weights = zkdl::cpu_ipa_mle_weights(hidden_matmul_wire_points(batch_point));
    line.assign(shape.n, FrWire{});
    for (std::uint32_t segment = 0U; segment < provider.segment_count(); ++segment) {
        std::vector<FrWire> values;
        if (!provider.read_segment(segment, values, error)) return false;
        for (std::size_t slot = 0U; slot < values.size(); ++slot) {
            const auto index = static_cast<std::uint64_t>(segment) * provider.segment_size() + slot;
            const auto inner = left_tensor ? index % shape.n : (index / shape.k) % shape.n;
            const auto outer = left_tensor ? (index / shape.n) % shape.m : index % shape.k;
            const auto batch = index / (left_tensor ? static_cast<std::uint64_t>(shape.m) * shape.n : static_cast<std::uint64_t>(shape.n) * shape.k);
            const FrWire weight = zkdl::fr_mul(batch_weights.at(batch), left_tensor ? row_weights.at(outer) : column_weights.at(outer));
            line[inner] = zkdl::fr_add(line[inner], zkdl::fr_mul(values[slot], weight));
        }
        if (progress != nullptr) progress->update(left_tensor ? "hidden_matmul_A_segments" : "hidden_matmul_B_segments", segment + 1U, provider.segment_count());
    }
    return true;
}

inline zkdl::CpuIpaOpeningMetadata hidden_matmul_opening_metadata(
    const OperationDescriptor& operation, const PrimitiveDescriptor& primitive,
    std::size_t primitive_index, const std::string& role,
    std::uint32_t commitment_index, const std::vector<FrWire>& point,
    const RelationStatement& statement) {
    std::uint32_t primitive_index_u32 = 0U;
    if (!checked_hidden_matmul_primitive_index(primitive_index,
                                              primitive_index_u32))
        throw std::invalid_argument("hidden matrix primitive index exceeds uint32");
    if (primitive_index_u32 > (0x0fffffffU / 4U))
        throw std::invalid_argument("hidden matrix opening index exceeds reserved range");
    zkdl::CpuIpaOpeningMetadata metadata;
    metadata.commitment_index = commitment_index;
    metadata.evaluation_commitment_index =
        0x60000000U + primitive_index_u32 * 4U;
    metadata.registry_digest = statement.registry_digest;
    metadata.srs_digest = statement.srs_digest;
    std::vector<byte> encoded;
    zkdl::append_string(encoded, operation.id);
    zkdl::append_u32(encoded, primitive_index_u32);
    append_primitive(encoded, primitive);
    zkdl::append_string(encoded, role);
    zkdl::append_u64(encoded, static_cast<std::uint64_t>(point.size()));
    for (const auto& coordinate : point)
        zkdl::append_fr(encoded, zkdl::fr_value(coordinate));
    metadata.descriptor_digest = zkdl::sha256(encoded);
    return metadata;
}

inline bool hidden_matmul_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t expected_domain, std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(
        commitments.begin(), commitments.end(), [&](const CommitmentWire& commitment) {
            return commitment.tensor_id == tensor_id;
        });
    if (found == commitments.end() || found->domain_size != expected_domain) {
        if (error) *error = "hidden matrix commitment/domain mismatch: " + tensor_id;
        return false;
    }
    return checked_relation_u32(
        static_cast<std::size_t>(std::distance(commitments.begin(), found)),
        index, "hidden matrix commitment index", error);
}

inline FrWire hidden_matmul_nonzero_scalar() {
    return zkdl::cpu_fresh_csprng_scalar();
}

struct HiddenMatMulOpeningSecrets {
    HiddenMatMulOpening proof;
    FrWire value{};
    FrWire tau_blinder{};
};

inline bool prove_hidden_segmented_opening(const SegmentedTensorProvider& provider, const CommitmentWire& commitment, const SegmentCommitmentMaterial& material, const std::vector<FrWire>& point, const std::vector<zkdl::CpuG1Point>& generators, const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u, zkdl::CpuIpaOpeningMetadata metadata, HiddenMatMulOpeningSecrets& result, std::string* error, NativeOperationProgressReporter* progress = nullptr, const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr, const SegmentCommitmentValidationReceipt* validation_receipt = nullptr, zkdl::CpuG1MsmSession* msm_session = nullptr) {
    ZkRelationSegmentLayout layout;
    if (!provider.valid(error) || !zktr_segment_layout(provider.logical_domain(), generators.size(), layout, error) || !zktr_segment_material_matches(commitment, material, error) || commitment.domain_size != provider.logical_domain() || commitment.segment_size != layout.segment_size || commitment.segment_count != layout.segment_count || provider.segment_size() != layout.segment_size || provider.segment_count() != layout.segment_count || point.size() != log2_exact(layout.logical_domain)) return false;
    const auto low = zktr_segment_point(point, layout.segment_size), high = zktr_segment_prefix_point(point, layout.segment_size);
    const auto low_weights = zkdl::cpu_ipa_mle_weights(low), high_weights = zkdl::cpu_ipa_mle_weights(high);
    if (!zktr_basis_valid_for_prefix(generators, h, u, layout.segment_size,
                                     authenticated_basis, error)) return false;
    const auto frame = zktr_basis_digest_for_prefix(
        generators, h, u, layout.segment_size,
        zkdl::cpu_ipa_padded_size(layout.segment_size), authenticated_basis);
    if (validation_receipt != nullptr &&
        !validation_receipt->matches_context(
            commitment.tensor_id, commitment.visibility, provider, generators,
            h, u, authenticated_basis, commitment, material, error)) return false;
    std::vector<zkdl::CpuG1Point> raw_basis;
    if (authenticated_basis == nullptr)
        raw_basis.assign(generators.begin(), generators.begin() + layout.segment_size);
    result = HiddenMatMulOpeningSecrets{};
    zkdl::CpuG1Point tau = zkdl::CpuG1Point::zero();
    for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
        const auto& wire = commitment.segment_commitments[segment];
        std::vector<FrWire> values;
        if (wire.frame_digest != frame || !provider.read_segment(segment, values, error)) return false;
        zkdl::CpuG1Point xi = material.commitments[segment];
        if (validation_receipt != nullptr) {
            if (!validation_receipt->matches_segment(segment, values, material, error))
                return false;
        } else {
            xi = zkdl::cpu_g1_add(
                zktr_cpu_g1_sum(generators, values.data(), layout.segment_size,
                                authenticated_basis, msm_session),
                zkdl::cpu_g1_mul(h, material.blinders[segment]));
            if (!zkdl::cpu_g1_equal(xi, material.commitments[segment])) { if (error) *error = "hidden segment Xi differs from canonical witness commitment"; return false; }
        }
        const FrWire value = zkdl::cpu_ipa_inner_product(values, low_weights), blinder = hidden_matmul_nonzero_scalar();
        const auto segment_tau = zkdl::cpu_pedersen_scalar_commit(value, blinder, h, u);
        metadata.outer_point = point;
        metadata.logical_domain = layout.logical_domain;
        metadata.segment_index = segment;
        metadata.logical_offset = wire.logical_offset;
        metadata.actual_length = wire.actual_length;
        metadata.padded_length = wire.padded_length;
        const auto opening = authenticated_basis != nullptr
            ? zkdl::cpu_open_hyrax_v3_authenticated(
                  *authenticated_basis, values, low, material.blinders[segment],
                  blinder, segment_tau, metadata, xi, msm_session)
            : zkdl::cpu_open_hyrax_v3_with_checked_xi(
                  raw_basis, values, low, h, u, material.blinders[segment],
                  blinder, segment_tau, metadata, xi);
        if (!zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(opening.proof.xi), xi)) return false;
        result.proof.segments.push_back(opening.proof);
        result.value = zkdl::fr_add(result.value, zkdl::fr_mul(high_weights[segment], value));
        result.tau_blinder = zkdl::fr_add(result.tau_blinder, zkdl::fr_mul(high_weights[segment], blinder));
        tau = zkdl::cpu_g1_add(tau, zkdl::cpu_g1_mul(segment_tau, high_weights[segment]));
        if (progress != nullptr) progress->update("hidden_matmul_opening_segments", segment + 1U, layout.segment_count);
    }
    result.proof.tau = zkdl::cpu_g1_to_native(tau);
    return hidden_matmul_opening_shape_valid(result.proof, layout.logical_domain);
}

inline bool verify_hidden_segmented_opening(const HiddenMatMulOpening& opening, const CommitmentWire& commitment, const std::vector<FrWire>& point, const std::vector<zkdl::CpuG1Point>& generators, const zkdl::CpuG1Point& h, const zkdl::CpuG1Point& u, zkdl::CpuIpaOpeningMetadata metadata, std::string* error, const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr, zkdl::CpuG1MsmSession* msm_session = nullptr) {
    ZkRelationSegmentLayout layout;
    if (!segment_commitment_wire_shape_valid(commitment, error) || !zktr_segment_layout(commitment.domain_size, generators.size(), layout, error) || !hidden_matmul_opening_shape_valid(opening, layout.logical_domain) || commitment.segment_size != layout.segment_size || commitment.segment_count != layout.segment_count || point.size() != log2_exact(layout.logical_domain) || opening.segments.size() != layout.segment_count) return false;
    const auto low = zktr_segment_point(point, layout.segment_size), high = zktr_segment_prefix_point(point, layout.segment_size);
    const auto high_weights = zkdl::cpu_ipa_mle_weights(high);
    if (!zktr_basis_valid_for_prefix(generators, h, u, layout.segment_size,
                                     authenticated_basis, error)) return false;
    const auto frame = zktr_basis_digest_for_prefix(
        generators, h, u, layout.segment_size,
        zkdl::cpu_ipa_padded_size(layout.segment_size), authenticated_basis);
    std::vector<zkdl::CpuG1Point> raw_basis;
    if (authenticated_basis == nullptr)
        raw_basis.assign(generators.begin(), generators.begin() + layout.segment_size);
    zkdl::CpuG1Point tau = zkdl::CpuG1Point::zero();
    for (std::uint32_t segment = 0U; segment < layout.segment_count; ++segment) {
        const auto& wire = commitment.segment_commitments[segment];
        const auto& proof = opening.segments[segment];
        metadata.outer_point = point;
        metadata.logical_domain = layout.logical_domain;
        metadata.segment_index = segment;
        metadata.logical_offset = wire.logical_offset;
        metadata.actual_length = wire.actual_length;
        metadata.padded_length = wire.padded_length;
        const bool verified = authenticated_basis != nullptr
            ? zkdl::cpu_verify_hyrax_v3_authenticated(
                  proof, zkdl::cpu_g1_from_native(wire.commitment),
                  zkdl::cpu_g1_from_native(proof.tau), *authenticated_basis,
                  low, metadata, error, msm_session)
            : zkdl::cpu_verify_hyrax_v3(
                  proof, zkdl::cpu_g1_from_native(wire.commitment),
                  zkdl::cpu_g1_from_native(proof.tau), raw_basis, low, h, u,
                  metadata, error);
        if (wire.frame_digest != frame || !verified) return false;
        tau = zkdl::cpu_g1_add(tau, zkdl::cpu_g1_mul(zkdl::cpu_g1_from_native(proof.tau), high_weights[segment]));
    }
    if (!zkdl::cpu_g1_equal(tau, zkdl::cpu_g1_from_native(opening.tau))) { if (error) *error = "hidden segment tau aggregation mismatch"; return false; }
    return true;
}

inline bool prove_hidden_matmul(
    const OperationDescriptor& operation, const PrimitiveDescriptor& primitive,
    std::size_t primitive_index,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement, zkdl::Transcript& transcript,
    HiddenMatMulProof& result, std::string* error, const std::map<std::string, std::vector<FrWire>>& segment_blinding_scalars,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts = nullptr) {
    try {
        const std::string progress_operation_id = operation.id + "/primitive/" + std::to_string(primitive_index);
        NativeOperationProgressReporter hidden_progress("hidden_matmul", progress_operation_id);
        std::uint32_t primitive_index_u32 = 0U;
        std::array<std::uint32_t, 3U> product_indices{};
        if (!checked_hidden_matmul_primitive_index(primitive_index,
                                                  primitive_index_u32, error) ||
            !hidden_matmul_product_context_indices(primitive_index,
                                                   product_indices, error))
            return false;
        if (primitive.kind != 2U ||
            !hidden_matmul_shape_valid(primitive.matmul_shape, error) ||
            primitive.tensor_refs.size() != 3U) return false;
        std::array<std::uint32_t, 3U> shape_domains{};
        if (!checked_matmul_shape_domains_u32(primitive.matmul_shape, shape_domains, error)) return false;
        const auto a = tensor_map.find(primitive.tensor_refs[0]);
        const auto b = tensor_map.find(primitive.tensor_refs[1]);
        const auto c = tensor_map.find(primitive.tensor_refs[2]);
        if (a == tensor_map.end() || b == tensor_map.end() || c == tensor_map.end() ||
            a->second == nullptr || b->second == nullptr || c->second == nullptr ||
            a->second->domain_size != primitive.matmul_shape.a_size() ||
            b->second->domain_size != primitive.matmul_shape.b_size() ||
            c->second->domain_size != primitive.matmul_shape.c_size()) {
            if (error) *error = "hidden matrix witness shape is not canonical";
            return false;
        }
        if (generators.size() < primitive.matmul_shape.n) { if (error) *error = "hidden matrix inner dimension exceeds authenticated basis"; return false; }
        SegmentedTensorProvider a_provider, b_provider, c_provider;
        if (!a->second->segmented(generators.size(), a_provider, error) || !b->second->segmented(generators.size(), b_provider, error) || !c->second->segmented(generators.size(), c_provider, error)) return false;
        std::uint32_t a_index = 0U, b_index = 0U, c_index = 0U;
        if (!hidden_matmul_commitment_index(commitments, primitive.tensor_refs[0],
                                            shape_domains[0U],
                                            a_index, error) ||
            !hidden_matmul_commitment_index(commitments, primitive.tensor_refs[1],
                                            shape_domains[1U],
                                            b_index, error) ||
            !hidden_matmul_commitment_index(commitments, primitive.tensor_refs[2],
                                            shape_domains[2U],
                                            c_index, error)) return false;

        std::vector<Fr_t> row_point, column_point, batch_point;
        if (!hidden_matmul_derive_points(operation, primitive_index, primitive.matmul_shape,
                                         transcript, row_point, column_point, batch_point,
                                         error)) return false;
        std::vector<FrWire> a_line, b_line;
        if (!hidden_matmul_line_values(a_provider, primitive.matmul_shape, row_point, column_point, batch_point, true, a_line, error, &hidden_progress) || !hidden_matmul_line_values(b_provider, primitive.matmul_shape, row_point, column_point, batch_point, false, b_line, error, &hidden_progress)) return false;
        const auto row = hidden_matmul_wire_points(row_point), column = hidden_matmul_wire_points(column_point), batch = hidden_matmul_wire_points(batch_point);
        const auto output_point = hidden_matmul_concat_points(batch, row, column);
        FrWire c_value{};
        if (!c_provider.evaluate_mle(output_point, c_value, error, &hidden_progress)) return false;
        std::vector<FrWire> c_selector(primitive.matmul_shape.n, FrWire{});
        c_selector[0U] = c_value;
        const ConstraintDescriptor constraint = hidden_matmul_constraint_descriptor(
            operation, primitive_index);
        const std::string operation_context = "gat.relation." + operation.id +
                                               "/hidden-matmul/" +
                                               std::to_string(primitive_index);
        const std::vector<zkdl::CpuG1Point> inner_generators(
            generators.begin(), generators.begin() + primitive.matmul_shape.n);
        ZkSumcheckProverResult sumcheck;
        if (!prove_zk_sumcheck(constraint, {a_line, b_line, c_selector}, 2U,
                               primitive.matmul_shape.n, inner_generators,
                               ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                               operation_context, transcript, sumcheck, error,
                               "hidden_matmul_sumcheck", progress_operation_id.c_str(),
                               true, true)) return false;
        std::vector<FrWire> inner_point;
        for (const auto& challenge : sumcheck.challenges)
            inner_point.push_back(zkdl::fr_wire(challenge));
        std::reverse(inner_point.begin(), inner_point.end());
        const std::vector<FrWire> full_a = hidden_matmul_concat_points(
            batch, row, inner_point);
        const std::vector<FrWire> full_b = hidden_matmul_concat_points(
            batch, inner_point, column);
        const std::vector<FrWire> full_c = output_point;
        const std::array<byte, 32U> primitive_digest = hidden_matmul_primitive_digest(
            operation, primitive, primitive_index);
        const auto open_one = [&](const SegmentedTensorProvider& provider, std::uint32_t index, const std::vector<FrWire>& point, const char* role, HiddenMatMulOpeningSecrets& opening) {
            SegmentCommitmentMaterial material;
            if (!zktr_material_from_wire(commitments[index], segment_blinding_scalars, material, error)) return false;
            const auto metadata = hidden_matmul_opening_metadata(operation, primitive, primitive_index, role, index, point, statement);
            const SegmentCommitmentValidationReceipt* receipt = nullptr;
            if (validation_receipts != nullptr) {
                const auto found_receipt = validation_receipts->find(primitive.tensor_refs[
                    role[0] == 'A' ? 0U : role[0] == 'B' ? 1U : 2U]);
                if (found_receipt == validation_receipts->end()) {
                    if (error) *error = "hidden matrix opening receipt is missing";
                    return false;
                }
                receipt = &found_receipt->second;
            }
            return prove_hidden_segmented_opening(
                provider, commitments[index], material, point, generators,
                ipa_blinding_generator, ipa_u_generator, metadata, opening,
                error, &hidden_progress, authenticated_basis, receipt,
                msm_session);
        };
        HiddenMatMulOpeningSecrets opening_a, opening_b, opening_c;
        if (!open_one(a_provider, a_index, full_a, "A", opening_a) || !open_one(b_provider, b_index, full_b, "B", opening_b) || !open_one(c_provider, c_index, full_c, "C", opening_c)) return false;
        const FrWire a_value = opening_a.value, b_value = opening_b.value;
        const FrWire eq0 = [&]() {
            FrWire value = zkdl::fr_from_u32(1U);
            for (const auto& coordinate : inner_point)
                value = zkdl::fr_mul(value, zkdl::fr_sub(
                    zkdl::fr_from_u32(1U), coordinate));
            return value;
        }();
        const FrWire product_value = zkdl::fr_mul(a_value, b_value);
        const FrWire product_blinder = hidden_matmul_nonzero_scalar();
        const zkdl::CpuG1Point product_commitment = zkdl::cpu_pedersen_scalar_commit(
            product_value, product_blinder, ipa_blinding_generator, ipa_u_generator);
        const auto product_context = zktr_make_product_context(
            statement, sumcheck.proof.descriptor_digest, ipa_blinding_generator,
            ipa_u_generator, product_indices[0U], product_indices[1U],
            product_indices[2U]);
        const zkdl::PedersenProductProverResult product = zkdl::cpu_open_pedersen_product(
            a_value, b_value, opening_a.tau_blinder, opening_b.tau_blinder,
            product_blinder, zkdl::cpu_g1_from_native(opening_a.proof.tau),
            zkdl::cpu_g1_from_native(opening_b.proof.tau), product_commitment,
            ipa_blinding_generator, ipa_u_generator, product_context);
        const FrWire residual_value = zkdl::fr_sub(
            product_value, zkdl::fr_mul(opening_c.value, eq0));
        const FrWire residual_blinder = zkdl::fr_sub(
            product_blinder, zkdl::fr_mul(opening_c.tau_blinder, eq0));
        if (zkdl::fr_compare(residual_blinder, FrWire{}) == 0) {
            if (error) *error = "hidden matrix residual blinder is zero";
            return false;
        }
        const zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_add(
            product_commitment,
            zkdl::cpu_g1_neg(zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(opening_c.proof.tau), eq0)));
        const auto equality_context = zktr_make_link_context(
            constraint, statement, operation_context,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            sumcheck.proof.descriptor_digest, ipa_blinding_generator,
            ipa_u_generator);
        const zkdl::PedersenScalarEqualityProverResult equality =
            zkdl::cpu_open_pedersen_scalar_equality(
                residual_value, residual_blinder, sumcheck.final_q_blinder,
                residual_commitment,
                zkdl::cpu_g1_from_native(sumcheck.proof.final_claim_commitment),
                ipa_blinding_generator, ipa_u_generator, equality_context);
        result = HiddenMatMulProof{};
        result.shape = primitive.matmul_shape;
        result.a_commitment_index = a_index;
        result.b_commitment_index = b_index;
        result.c_commitment_index = c_index;
        result.primitive_digest = primitive_digest;
        result.row_point = row_point;
        result.column_point = column_point;
        result.batch_point = batch_point;
        result.sumcheck = std::move(sumcheck.proof);
        result.opening_a = opening_a.proof;
        result.opening_b = opening_b.proof;
        result.opening_c = opening_c.proof;
        result.terminal_product = product.proof;
        result.terminal_binding = equality.proof;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = "hidden matrix prover failed operation_id=" + operation.id + " primitive_index=" + std::to_string(primitive_index) + ": " + exception.what();
        return false;
    }
}

inline bool verify_hidden_matmul(
    const OperationDescriptor& operation, const PrimitiveDescriptor& primitive,
    std::size_t primitive_index, const HiddenMatMulProof& proof,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement, zkdl::Transcript& transcript,
    std::string* error = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr) {
    try {
        std::uint32_t primitive_index_u32 = 0U;
        std::array<std::uint32_t, 3U> product_indices{};
        if (!checked_hidden_matmul_primitive_index(primitive_index,
                                                  primitive_index_u32, error) ||
            !hidden_matmul_product_context_indices(primitive_index,
                                                   product_indices, error))
            return false;
        if (primitive.kind != 2U || !hidden_matmul_shape_valid(primitive.matmul_shape, error) ||
            primitive.tensor_refs.size() != 3U || proof.version != HiddenMatMulProof::kVersion ||
            proof.shape != primitive.matmul_shape ||
            proof.primitive_digest != hidden_matmul_primitive_digest(
                operation, primitive, primitive_index)) {
            if (error && error->empty()) *error = "hidden matrix primitive statement mismatch";
            return false;
        }
        std::array<std::uint32_t, 3U> shape_domains{};
        if (!checked_matmul_shape_domains_u32(primitive.matmul_shape, shape_domains, error)) return false;
        std::vector<Fr_t> row_point, column_point, batch_point;
        if (!hidden_matmul_derive_points(operation, primitive_index, primitive.matmul_shape,
                                         transcript, row_point, column_point, batch_point,
                                         error)) return false;
        const auto same_points = [](const std::vector<Fr_t>& left,
                                    const std::vector<Fr_t>& right) {
            if (left.size() != right.size()) return false;
            for (std::size_t i = 0U; i < left.size(); ++i)
                if (zkdl::fr_compare(zkdl::fr_wire(left[i]), zkdl::fr_wire(right[i])) != 0)
                    return false;
            return true;
        };
        if (!same_points(proof.row_point, row_point) ||
            !same_points(proof.column_point, column_point) ||
            !same_points(proof.batch_point, batch_point)) {
            if (error) *error = "hidden matrix challenge point mismatch";
            return false;
        }
        const std::uint32_t a_domain = shape_domains[0U];
        const std::uint32_t b_domain = shape_domains[1U];
        const std::uint32_t c_domain = shape_domains[2U];
        if (generators.size() < primitive.matmul_shape.n) { if (error) *error = "hidden matrix verifier inner dimension exceeds authenticated basis"; return false; }
        if (proof.a_commitment_index >= commitments.size() ||
            proof.b_commitment_index >= commitments.size() ||
            proof.c_commitment_index >= commitments.size() ||
            commitments[proof.a_commitment_index].tensor_id != primitive.tensor_refs[0] ||
            commitments[proof.b_commitment_index].tensor_id != primitive.tensor_refs[1] ||
            commitments[proof.c_commitment_index].tensor_id != primitive.tensor_refs[2] ||
            commitments[proof.a_commitment_index].domain_size != a_domain ||
            commitments[proof.b_commitment_index].domain_size != b_domain ||
            commitments[proof.c_commitment_index].domain_size != c_domain) {
            if (error) *error = "hidden matrix proof commitment index mismatch";
            return false;
        }
        const ConstraintDescriptor constraint = hidden_matmul_constraint_descriptor(
            operation, primitive_index);
        const std::string operation_context = "gat.relation." + operation.id +
                                               "/hidden-matmul/" +
                                               std::to_string(primitive_index);
        const std::vector<zkdl::CpuG1Point> inner_generators(
            generators.begin(), generators.begin() + primitive.matmul_shape.n);
        ZkSumcheckVerifierResult sumcheck;
        if (!verify_zk_sumcheck(proof.sumcheck, constraint, 2U, primitive.matmul_shape.n,
                                inner_generators, ipa_blinding_generator,
                                ipa_u_generator, FrWire{}, statement, operation_context,
                                transcript, sumcheck, error)) return false;
        std::vector<FrWire> inner_point;
        for (const auto& challenge : sumcheck.challenges)
            inner_point.push_back(zkdl::fr_wire(challenge));
        std::reverse(inner_point.begin(), inner_point.end());
        const std::vector<FrWire> row = hidden_matmul_wire_points(row_point);
        const std::vector<FrWire> column = hidden_matmul_wire_points(column_point);
        const std::vector<FrWire> batch = hidden_matmul_wire_points(batch_point);
        const std::vector<FrWire> full_a = hidden_matmul_concat_points(
            batch, row, inner_point);
        const std::vector<FrWire> full_b = hidden_matmul_concat_points(
            batch, inner_point, column);
        const std::vector<FrWire> full_c = hidden_matmul_concat_points(
            batch, row, column);
        const auto verify_opening = [&](const HiddenMatMulOpening& opening, std::uint32_t index, std::uint32_t domain, const std::vector<FrWire>& point, const char* role) {
            const auto metadata = hidden_matmul_opening_metadata(operation, primitive, primitive_index, role, index, point, statement);
            return domain == commitments[index].domain_size &&
                   verify_hidden_segmented_opening(
                       opening, commitments[index], point, generators,
                       ipa_blinding_generator, ipa_u_generator, metadata, error,
                       authenticated_basis, msm_session);
        };
        if (!verify_opening(proof.opening_a, proof.a_commitment_index, a_domain,
                            full_a, "A") ||
            !verify_opening(proof.opening_b, proof.b_commitment_index, b_domain,
                            full_b, "B") ||
            !verify_opening(proof.opening_c, proof.c_commitment_index, c_domain,
                            full_c, "C")) return false;
        FrWire eq0 = zkdl::fr_from_u32(1U);
        for (const auto& coordinate : inner_point)
            eq0 = zkdl::fr_mul(eq0, zkdl::fr_sub(zkdl::fr_from_u32(1U), coordinate));
        const auto product_context = zktr_make_product_context(
            statement, proof.sumcheck.descriptor_digest, ipa_blinding_generator,
            ipa_u_generator, product_indices[0U], product_indices[1U],
            product_indices[2U]);
        const zkdl::CpuG1Point product_commitment = zkdl::cpu_g1_from_native(
            proof.terminal_product.commitment_c);
        if (!zkdl::cpu_verify_pedersen_product(
                proof.terminal_product,
                zkdl::cpu_g1_from_native(proof.opening_a.tau),
                zkdl::cpu_g1_from_native(proof.opening_b.tau), product_commitment,
                ipa_blinding_generator, ipa_u_generator, product_context, error)) return false;
        const zkdl::CpuG1Point residual_commitment = zkdl::cpu_g1_add(
            product_commitment,
            zkdl::cpu_g1_neg(zkdl::cpu_g1_mul(
                zkdl::cpu_g1_from_native(proof.opening_c.tau), eq0)));
        const auto equality_context = zktr_make_link_context(
            constraint, statement, operation_context,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            proof.sumcheck.descriptor_digest, ipa_blinding_generator,
            ipa_u_generator);
        return zkdl::cpu_verify_pedersen_scalar_equality(
            proof.terminal_binding, residual_commitment,
            zkdl::cpu_g1_from_native(sumcheck.final_claim_commitment),
            ipa_blinding_generator, ipa_u_generator, equality_context, error);
    } catch (const std::exception& exception) {
        if (error) *error = std::string("hidden matrix verifier failed: ") + exception.what();
        return false;
    }
}

inline bool prove_matmul_primitive(
    const PrimitiveDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    std::vector<byte>& encoded, std::string* error) {
    (void)descriptor;
    (void)tensor_map;
    (void)generators;
    (void)ipa_blinding_generator;
    (void)ipa_u_generator;
    (void)encoded;
    if (error) *error =
        "BLOCKED: legacy value-revealing zkMatMul prover is disabled; use HiddenMatMulProof";
    return false;
}
#if 0
    if (descriptor.kind != 1U || descriptor.tensor_refs.size() != 3U) {
        if (error) *error = "unsupported native relation primitive";
        return false;
    }
    const zkdl::MatMulShape& shape = descriptor.matmul_shape;
    std::string shape_error;
    if (!shape.valid(&shape_error)) {
        if (error) *error = "zkMatMul primitive shape rejected: " + shape_error;
        return false;
    }
    const std::uint32_t generator_count = std::max(shape.n, std::max(shape.k, 1U));
    if (generators.size() < generator_count) {
        if (error) *error = "authenticated SRS generator domain is too small for zkMatMul";
        return false;
    }
    std::vector<Fr_t> a_values, b_values, c_values;
    if (!primitive_tensor_values(tensor_map, descriptor.tensor_refs[0], shape.a_size(),
                                 a_values, error) ||
        !primitive_tensor_values(tensor_map, descriptor.tensor_refs[1], shape.b_size(),
                                 b_values, error) ||
        !primitive_tensor_values(tensor_map, descriptor.tensor_refs[2], shape.c_size(),
                                 c_values, error)) return false;
    std::vector<G1Jacobian_t> native_generators;
    native_generators.reserve(generator_count);
    for (std::uint32_t i = 0U; i < generator_count; ++i)
        native_generators.push_back(zkdl::cpu_g1_to_native(generators[i]));
    try {
        Commitment gen_a(shape.n, native_generators.data());
        Commitment gen_b(shape.k, native_generators.data());
        Commitment gen_c(shape.k, native_generators.data());
        FrTensor a(shape.a_size(), a_values.data());
        FrTensor b(shape.b_size(), b_values.data());
        FrTensor c(shape.c_size(), c_values.data());
        const zkdl::MatMulProof proof = zkdl::matmul_prove(
            a, b, c, shape, gen_a, gen_b, gen_c,
            nullptr, ipa_blinding_generator, ipa_u_generator);
        encoded = proof.serialize();
        if (encoded.empty()) {
            if (error) *error = "zkMatMul primitive prover returned an empty proof";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = std::string("zkMatMul primitive prover failed: ") + exception.what();
        return false;
    }
}
#endif

bool verify_matmul_primitive(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator, bool gpu_backend,
    std::string* error) {
    (void)descriptor;
    (void)encoded;
    (void)generators;
    (void)ipa_blinding_generator;
    (void)ipa_u_generator;
    (void)gpu_backend;
    if (error) *error =
        "BLOCKED: legacy value-revealing zkMatMul verifier is disabled; use HiddenMatMulProof";
    return false;
}
#if 0
    if (descriptor.kind != 1U || descriptor.tensor_refs.size() != 3U || encoded.empty()) {
        if (error) *error = "invalid native relation primitive proof";
        return false;
    }
    const zkdl::MatMulShape& shape = descriptor.matmul_shape;
    std::string shape_error;
    if (!shape.valid(&shape_error)) {
        if (error) *error = "zkMatMul primitive shape rejected: " + shape_error;
        return false;
    }
    const std::uint32_t generator_count = std::max(shape.n, std::max(shape.k, 1U));
    if (generators.size() < generator_count) {
        if (error) *error = "authenticated SRS generator domain is too small for zkMatMul";
        return false;
    }
    zkdl::MatMulProof proof;
    if (!zkdl::MatMulProof::deserialize(encoded, proof, error)) return false;
    if (proof.shape != shape || proof.c_binding_mode != zkdl::MatMulProof::kCCommitment) {
        if (error) *error = "zkMatMul primitive proof shape/binding mismatch";
        return false;
    }
    zkdl::MatMulCpuCommitmentView a_view, b_view, c_view;
    a_view.generators.assign(generators.begin(), generators.begin() + shape.n);
    b_view.generators.assign(generators.begin(), generators.begin() + shape.k);
    c_view.generators.assign(generators.begin(), generators.begin() + shape.k);
    a_view.commitments.reserve(proof.commitment_a.size());
    b_view.commitments.reserve(proof.commitment_b.size());
    c_view.commitments.reserve(proof.commitment_c.size());
    for (const auto& point : proof.commitment_a) a_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    for (const auto& point : proof.commitment_b) b_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    for (const auto& point : proof.commitment_c) c_view.commitments.push_back(zkdl::cpu_g1_from_native(point));
    if (gpu_backend) {
        if (proof.commitment_a.size() != shape.num * shape.m ||
            proof.commitment_b.size() != shape.num * shape.n ||
            proof.commitment_c.size() != shape.num * shape.m) {
            if (error) *error = "zkMatMul GPU commitment tensor shape mismatch";
            return false;
        }
        std::vector<G1Jacobian_t> native_generators_a(
            generators.size() >= shape.n ? shape.n : 0U);
        std::vector<G1Jacobian_t> native_generators_b(
            generators.size() >= shape.k ? shape.k : 0U);
        for (std::uint32_t index = 0U; index < shape.n; ++index)
            native_generators_a[index] = zkdl::cpu_g1_to_native(generators[index]);
        for (std::uint32_t index = 0U; index < shape.k; ++index)
            native_generators_b[index] = zkdl::cpu_g1_to_native(generators[index]);
        Commitment gen_a(shape.n, native_generators_a.data());
        Commitment gen_b(shape.k, native_generators_b.data());
        G1TensorJacobian com_a(static_cast<std::uint32_t>(proof.commitment_a.size()),
                               proof.commitment_a.data());
        G1TensorJacobian com_b(static_cast<std::uint32_t>(proof.commitment_b.size()),
                               proof.commitment_b.data());
        G1TensorJacobian com_c(static_cast<std::uint32_t>(proof.commitment_c.size()),
                               proof.commitment_c.data());
        return zkdl::matmul_verify_gpu(proof, shape, gen_a, com_a, gen_b, com_b,
                                       &gen_b, &com_c, nullptr, error);
    }
    return zkdl::matmul_verify_cpu(proof, shape, a_view, b_view, &c_view, nullptr,
                                   ipa_blinding_generator, ipa_u_generator, error);
}
#endif

inline bool primitive_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(commitments.begin(), commitments.end(),
                                    [&](const CommitmentWire& value) {
                                        return value.tensor_id == tensor_id;
                                    });
    if (found == commitments.end()) {
        if (error) *error = "zkMatMul canonical commitment is missing: " + tensor_id;
        return false;
    }
    return checked_relation_u32(
        static_cast<std::size_t>(std::distance(commitments.begin(), found)),
        index, "primitive commitment index", error);
}

inline std::vector<FrWire> primitive_opening_point(
    const zkdl::EvaluationOpeningProof& opening) {
    std::vector<FrWire> point;
    point.reserve(opening.inner_point.size() + opening.outer_point.size());
    for (const auto& value : opening.inner_point) point.push_back(zkdl::fr_wire(value));
    for (const auto& value : opening.outer_point) point.push_back(zkdl::fr_wire(value));
    return point;
}

inline bool prove_primitive_binding(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    OperationProof::PrimitiveBinding& binding, std::string* error) {
    zkdl::MatMulProof primitive;
    if (!zkdl::MatMulProof::deserialize(encoded, primitive, error) ||
        primitive.shape != descriptor.matmul_shape ||
        primitive.c_binding_mode != zkdl::MatMulProof::kCCommitment ||
        !primitive.has_c_opening) {
        if (error && error->empty()) *error = "zkMatMul primitive has no canonical C opening";
        return false;
    }
    std::array<std::uint32_t, 3U> shape_domains{};
    if (!checked_matmul_shape_domains_u32(descriptor.matmul_shape, shape_domains, error)) return false;
    const std::uint32_t a_domain = shape_domains[0U];
    const std::uint32_t b_domain = shape_domains[1U];
    const std::uint32_t c_domain = shape_domains[2U];
    const auto a = tensor_map.find(descriptor.tensor_refs[0]);
    const auto b = tensor_map.find(descriptor.tensor_refs[1]);
    const auto c = tensor_map.find(descriptor.tensor_refs[2]);
    if (a == tensor_map.end() || b == tensor_map.end() || c == tensor_map.end() ||
        a->second == nullptr || b->second == nullptr || c->second == nullptr ||
        a->second->domain_size != a_domain || b->second->domain_size != b_domain ||
        c->second->domain_size != c_domain || generators.size() <
            std::max(a_domain, std::max(b_domain, c_domain))) {
        if (error) *error = "zkMatMul canonical binding tensor shape is invalid";
        return false;
    }
    if (!primitive_commitment_index(commitments, descriptor.tensor_refs[0],
                                    binding.a_commitment_index, error) ||
        !primitive_commitment_index(commitments, descriptor.tensor_refs[1],
                                    binding.b_commitment_index, error) ||
        !primitive_commitment_index(commitments, descriptor.tensor_refs[2],
                                    binding.c_commitment_index, error)) return false;
    const std::vector<FrWire> point_a = primitive_opening_point(primitive.opening_a);
    const std::vector<FrWire> point_b = primitive_opening_point(primitive.opening_b);
    const std::vector<FrWire> point_c = primitive_opening_point(primitive.opening_c);
    auto metadata_for = [&](const std::string& suffix, std::uint32_t index) {
        OperationDescriptor opening_descriptor;
        opening_descriptor.id = descriptor.tensor_refs[0] + "/" + suffix;
        if (suffix == "B") opening_descriptor.id = descriptor.tensor_refs[1] + "/B";
        if (suffix == "C") opening_descriptor.id = descriptor.tensor_refs[2] + "/C";
        return relation_opening_metadata(opening_descriptor, index, {}, statement);
    };
    binding.opening_a = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + a_domain),
        a->second->values, point_a, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*a->second, blinders), metadata_for("A", binding.a_commitment_index));
    binding.opening_b = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + b_domain),
        b->second->values, point_b, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*b->second, blinders), metadata_for("B", binding.b_commitment_index));
    binding.opening_c = zkdl::cpu_open(
        std::vector<zkdl::CpuG1Point>(generators.begin(), generators.begin() + c_domain),
        c->second->values, point_c, ipa_blinding_generator, ipa_u_generator,
        relation_blinding_scalar(*c->second, blinders), metadata_for("C", binding.c_commitment_index));
    if (!same_wire(zkdl::fr_wire(binding.opening_a.value),
                   zkdl::fr_wire(primitive.opening_a.value)) ||
        !same_wire(zkdl::fr_wire(binding.opening_b.value),
                   zkdl::fr_wire(primitive.opening_b.value)) ||
        !same_wire(zkdl::fr_wire(binding.opening_c.value),
                   zkdl::fr_wire(primitive.opening_c.value))) {
        if (error) *error = "zkMatMul row commitment opening is not bound to canonical tensor opening";
        return false;
    }
    return true;
}

inline bool verify_primitive_binding(
    const PrimitiveDescriptor& descriptor, const std::vector<byte>& encoded,
    const OperationProof::PrimitiveBinding& binding,
    const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const RelationStatement& statement, std::string* error) {
    zkdl::MatMulProof primitive;
    if (!zkdl::MatMulProof::deserialize(encoded, primitive, error) ||
        primitive.shape != descriptor.matmul_shape ||
        primitive.c_binding_mode != zkdl::MatMulProof::kCCommitment ||
        !primitive.has_c_opening) {
        if (error && error->empty()) *error = "zkMatMul primitive binding mode is not canonical";
        return false;
    }
    std::array<std::uint32_t, 3U> shape_domains{};
    if (!checked_matmul_shape_domains_u32(descriptor.matmul_shape, shape_domains, error)) return false;
    const std::array<std::uint32_t, 3U> indices = {
        binding.a_commitment_index, binding.b_commitment_index, binding.c_commitment_index};
    const std::array<std::uint64_t, 3U> domains = {
        descriptor.matmul_shape.a_size(), descriptor.matmul_shape.b_size(), descriptor.matmul_shape.c_size()};
    const std::array<const zkdl::EvaluationOpeningProof*, 3U> openings = {
        &binding.opening_a, &binding.opening_b, &binding.opening_c};
    const std::array<const zkdl::EvaluationOpeningProof*, 3U> primitive_openings = {
        &primitive.opening_a, &primitive.opening_b, &primitive.opening_c};
    auto metadata_for = [&](std::size_t slot) {
        OperationDescriptor opening_descriptor;
        opening_descriptor.id = descriptor.tensor_refs[slot] + "/" +
                                (slot == 0U ? "A" : (slot == 1U ? "B" : "C"));
        return relation_opening_metadata(opening_descriptor, indices[slot], {}, statement);
    };
    for (std::size_t i = 0U; i < indices.size(); ++i) {
        if (indices[i] >= commitments.size() ||
            commitments[indices[i]].tensor_id != descriptor.tensor_refs[i] ||
            commitments[indices[i]].domain_size != domains[i] ||
            openings[i]->commitment_index != indices[i] ||
            openings[i]->generator_size != shape_domains[i] || !openings[i]->outer_point.empty() ||
            openings[i]->inner_point.size() != primitive_openings[i]->inner_point.size() +
                                               primitive_openings[i]->outer_point.size()) {
            if (error) *error = "zkMatMul canonical commitment binding metadata mismatch";
            return false;
        }
        const std::vector<FrWire> point = primitive_opening_point(*primitive_openings[i]);
        for (std::size_t j = 0U; j < point.size(); ++j)
            if (!same_wire(zkdl::fr_wire(openings[i]->inner_point[j]), point[j])) {
                if (error) *error = "zkMatMul canonical binding point differs from row opening";
                return false;
            }
        if (!zkdl::cpu_verify_opening(*openings[i],
                                      zkdl::cpu_g1_from_native(commitments[indices[i]].commitment),
                                      std::vector<zkdl::CpuG1Point>(generators.begin(),
                                                                    generators.begin() + shape_domains[i]),
                                      ipa_blinding_generator, ipa_u_generator,
                                      metadata_for(i), error) ||
            !same_wire(zkdl::fr_wire(openings[i]->value),
                       zkdl::fr_wire(primitive_openings[i]->value))) {
            if (error && error->empty()) *error = "zkMatMul canonical row reduction opening failed";
            return false;
        }
    }
    return true;
}

inline bool route_tensor_shapes_valid(
    const OperationDescriptor& descriptor,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const TensorWitness*& source, const TensorWitness*& output,
    std::string* error) {
    if ((descriptor.kind != RelationKind::SparseRoute &&
         descriptor.kind != RelationKind::Alpha) ||
        descriptor.selector_edge_domain == 0U || descriptor.selector_node_domain == 0U ||
        descriptor.selector_feature_domain == 0U ||
        descriptor.selector_source_feature_domain == 0U ||
        descriptor.selector_src.size() != descriptor.selector_edge_domain ||
        descriptor.selector_dst.size() != descriptor.selector_edge_domain ||
        descriptor.selector_active.size() != descriptor.selector_edge_domain ||
        descriptor.selector_feature_map.size() != descriptor.selector_feature_domain ||
        descriptor.selector_source_ref.empty() || descriptor.selector_output_ref.empty()) {
        if (error) *error = "sparse route descriptor is incomplete";
        return false;
    }
    const auto source_it = tensor_map.find(descriptor.selector_source_ref);
    const auto output_it = tensor_map.find(descriptor.selector_output_ref);
    if (source_it == tensor_map.end() || output_it == tensor_map.end() ||
        source_it->second == nullptr || output_it->second == nullptr) {
        if (error) *error = "sparse route canonical source/output tensor is missing";
        return false;
    }
    const std::uint64_t source_size = static_cast<std::uint64_t>(descriptor.selector_node_domain) *
                                      descriptor.selector_source_feature_domain;
    const std::uint64_t output_size = static_cast<std::uint64_t>(descriptor.selector_edge_domain) *
                                      descriptor.selector_feature_domain;
    if (source_size > std::numeric_limits<std::uint32_t>::max() ||
        output_size > std::numeric_limits<std::uint32_t>::max() ||
        source_it->second->domain_size != source_size ||
        output_it->second->domain_size != output_size) {
        if (error) *error = "sparse route source/output domains do not match public selector shape";
        return false;
    }
    for (std::uint32_t edge = 0U; edge < descriptor.selector_edge_domain; ++edge)
        if (descriptor.selector_active[edge] > 1U ||
            (!descriptor.selector_active[edge] &&
             (descriptor.selector_src[edge] != 0U || descriptor.selector_dst[edge] != 0U)) ||
            (descriptor.selector_active[edge] &&
             (descriptor.selector_src[edge] >= descriptor.selector_node_domain ||
              descriptor.selector_dst[edge] >= descriptor.selector_node_domain))) {
        if (error) *error = "sparse route public src/dst index is out of bounds";
            return false;
        }
    for (const std::uint32_t source : descriptor.selector_feature_map)
        if (source >= descriptor.selector_source_feature_domain) {
            if (error) *error = "sparse route public feature map is out of bounds";
            return false;
        }
    source = source_it->second;
    output = output_it->second;
    return true;
}

inline std::vector<FrWire> public_route_selector_mle_blocked(
    const OperationDescriptor& descriptor, const std::vector<FrWire>& edge_point,
    bool destination, std::uint32_t max_block_bits,
    const RouteWorkProgressCallback* progress = nullptr) {
    const std::uint32_t edge_domain = descriptor.selector_edge_domain;
    const std::uint32_t node_domain = descriptor.selector_node_domain;
    if (edge_point.size() >= kMaxRelationPointBits || max_block_bits > 18U ||
        edge_domain != (std::uint32_t(1U) << edge_point.size()) ||
        descriptor.selector_src.size() != edge_domain ||
        descriptor.selector_active.size() != edge_domain ||
        (destination && descriptor.selector_dst.size() != edge_domain)) {
        throw std::invalid_argument("public sparse selector MLE shape mismatch");
    }
    const std::uint32_t low_bits = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(edge_point.size()), max_block_bits);
    const std::size_t block_size = std::size_t(1U) << low_bits;
    std::vector<FrWire> low_point(edge_point.begin(), edge_point.begin() + low_bits);
    std::vector<FrWire> high_point(edge_point.begin() + low_bits, edge_point.end());
    // fold_values consumes point[0] at edge-index parity; cpu_ipa_mle_weights is MSB-first.
    std::reverse(low_point.begin(), low_point.end());
    std::reverse(high_point.begin(), high_point.end());
    const std::vector<FrWire> low_weights = zkdl::cpu_ipa_mle_weights(low_point);
    const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
    std::vector<FrWire> selector(node_domain);
    for (std::size_t segment = 0U; segment < high_weights.size(); ++segment) {
        for (std::size_t offset = 0U; offset < block_size; ++offset) {
            const std::size_t edge = segment * block_size + offset;
            if (!descriptor.selector_active[edge]) continue;
            const std::uint32_t node = destination ? descriptor.selector_dst[edge] : descriptor.selector_src[edge];
            if (node >= node_domain) throw std::invalid_argument("public sparse selector node index is out of bounds");
            selector[node] = zkdl::fr_add(selector[node], zkdl::fr_mul(high_weights[segment], low_weights[offset]));
        }
        if (progress != nullptr)
            (*progress)("selector_edge_blocks", (segment + 1U) * block_size,
                        edge_domain);
    }
    return selector;
}

inline std::vector<FrWire> public_route_selector_mle(
    const OperationDescriptor& descriptor, const std::vector<FrWire>& edge_point,
    bool destination = false,
    const RouteWorkProgressCallback* progress = nullptr) {
    return public_route_selector_mle_blocked(
        descriptor, edge_point, destination, 18U, progress);
}

inline std::vector<FrWire> public_route_feature_selector_mle(
    const OperationDescriptor& descriptor,
    const std::vector<FrWire>& feature_point,
    std::uint32_t max_block_bits,
    const RouteWorkProgressCallback* progress = nullptr) {
    const std::uint32_t output_domain = descriptor.selector_feature_domain;
    const std::uint32_t source_domain = descriptor.selector_source_feature_domain;
    if (feature_point.size() >= kMaxRelationPointBits || max_block_bits > 18U ||
        output_domain != (std::uint32_t(1U) << feature_point.size()) ||
        descriptor.selector_feature_map.size() != output_domain) {
        throw std::invalid_argument("public sparse feature selector MLE shape mismatch");
    }
    const std::uint32_t low_bits = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(feature_point.size()), max_block_bits);
    const std::size_t block_size = std::size_t(1U) << low_bits;
    std::vector<FrWire> low_point(feature_point.begin(), feature_point.begin() + low_bits);
    std::vector<FrWire> high_point(feature_point.begin() + low_bits, feature_point.end());
    std::reverse(low_point.begin(), low_point.end());
    std::reverse(high_point.begin(), high_point.end());
    const std::vector<FrWire> low_weights = zkdl::cpu_ipa_mle_weights(low_point);
    const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
    const FrWire zero{};
    std::vector<FrWire> selector(source_domain, zero);
    for (std::size_t segment = 0U; segment < high_weights.size(); ++segment) {
        for (std::size_t offset = 0U; offset < block_size; ++offset) {
            const std::size_t output = segment * block_size + offset;
            const std::uint32_t source = descriptor.selector_feature_map[output];
            if (source >= source_domain) throw std::invalid_argument("public sparse feature selector source index is out of bounds");
            const FrWire weight = zkdl::fr_mul(high_weights[segment], low_weights[offset]);
            selector[source] = zkdl::fr_add(selector[source], weight);
        }
        if (progress != nullptr)
            (*progress)("selector_feature_blocks", (segment + 1U) * block_size,
                        output_domain);
    }
    return selector;
}

inline std::vector<FrWire> public_route_feature_selector_mle(
    const OperationDescriptor& descriptor,
    const std::vector<FrWire>& feature_point,
    const RouteWorkProgressCallback* progress = nullptr) {
    return public_route_feature_selector_mle(
        descriptor, feature_point, 18U, progress);
}

inline bool route_commitment_index(
    const std::vector<CommitmentWire>& commitments, const std::string& tensor_id,
    std::uint32_t& index, std::string* error) {
    const auto found = std::find_if(commitments.begin(), commitments.end(),
                                    [&](const CommitmentWire& commitment) {
                                        return commitment.tensor_id == tensor_id;
                                    });
    if (found == commitments.end()) {
        if (error) *error = "sparse route commitment is missing: " + tensor_id;
        return false;
    }
    return checked_relation_u32(
        static_cast<std::size_t>(std::distance(commitments.begin(), found)),
        index, "sparse route commitment index", error);
}

#include "gat_adapter/gat_sparse_route.cuh"

inline bool qpad_selector_provider_unhashed(
    const TensorDescriptor& descriptor, std::size_t basis_size,
    SegmentedTensorProvider& provider, std::string* error = nullptr,
    std::uint64_t* provider_read_count = nullptr) {
    if (!descriptor.valid(error) || descriptor.logical_shape.size() !=
        descriptor.padded_shape.size()) {
        if (error && error->empty()) *error = "invalid Qpad tensor descriptor";
        return false;
    }
    bool padded = false;
    for (std::size_t axis = 0U; axis < descriptor.logical_shape.size(); ++axis)
        padded = padded || descriptor.logical_shape[axis] != descriptor.padded_shape[axis];
    if (!padded || descriptor.padding_rule.find("zero") == std::string::npos ||
        descriptor.padding_rule.find("mask") == std::string::npos) {
        if (error) *error = "padded tensor lacks an authenticated zero/mask Qpad rule";
        return false;
    }
    std::uint32_t segment_size = 0U, segment_count = 0U;
    if (!SegmentedTensorProvider::shape(
            descriptor.domain_size, basis_size, segment_size, segment_count, error)) return false;
    const FrWire one = zkdl::fr_from_u32(1U);
    provider = SegmentedTensorProvider(
        descriptor.domain_size, segment_size,
        [descriptor, one, provider_read_count](std::uint64_t offset, std::uint32_t count,
                     std::vector<FrWire>& output, std::string* read_error) {
            if (offset > descriptor.domain_size ||
                count > descriptor.domain_size - offset) {
                if (read_error) *read_error = "Qpad selector range is out of bounds";
                return false;
            }
            if (provider_read_count != nullptr &&
                *provider_read_count != std::numeric_limits<std::uint64_t>::max())
                ++*provider_read_count;
            output.assign(count, FrWire{});
            const std::uint64_t row_width = descriptor.padded_shape.back();
            const std::uint64_t end = offset + count;
            std::uint64_t cursor = offset;
            std::size_t output_offset = 0U;
            while (cursor < end) {
                const std::uint64_t column = cursor % row_width;
                const std::uint64_t take = std::min<std::uint64_t>(
                    end - cursor, row_width - column);
                std::uint64_t outer_index = cursor / row_width;
                bool inside_outer = true;
                for (std::size_t axis = descriptor.padded_shape.size() - 1U;
                     axis > 0U; --axis) {
                    const std::size_t outer_axis = axis - 1U;
                    const std::uint64_t coordinate =
                        outer_index % descriptor.padded_shape[outer_axis];
                    outer_index /= descriptor.padded_shape[outer_axis];
                    if (coordinate >= descriptor.logical_shape[outer_axis])
                        inside_outer = false;
                }
                const std::uint64_t logical_width = descriptor.logical_shape.back();
                const std::uint64_t zero_count = inside_outer && column < logical_width
                    ? std::min<std::uint64_t>(take, logical_width - column) : 0U;
                std::fill_n(output.begin() + static_cast<std::ptrdiff_t>(
                                output_offset + static_cast<std::size_t>(zero_count)),
                            static_cast<std::size_t>(take - zero_count), one);
                cursor += take;
                output_offset += static_cast<std::size_t>(take);
            }
            return true;
        });
    (void)segment_count;
    return true;
}

inline bool qpad_record_required(const std::vector<std::uint64_t>& logical_shape,
                                const std::vector<std::uint64_t>& padded_shape,
                                bool is_public) {
    if (is_public || logical_shape.empty() ||
        logical_shape.size() != padded_shape.size()) return false;
    for (std::size_t axis = 0U; axis < logical_shape.size(); ++axis)
        if (logical_shape[axis] != padded_shape[axis]) return true;
    return false;
}

struct QpadGroupPlan {
    std::uint32_t group_index = 0U;
    std::array<byte, 32U> group_key{};
    std::uint32_t domain_size = 0U;
    std::uint32_t segment_size = 0U;
    std::uint32_t segment_count = 0U;
    std::vector<const TensorDescriptor*> members;
    std::vector<std::uint32_t> descriptor_indices;
};

struct QpadGroupProverContext {
    const std::map<std::string, const TensorWitness*>* tensor_map = nullptr;
    const std::map<std::string, FrWire>* blinders = nullptr;
    const std::vector<zkdl::CpuG1Point>* generators = nullptr;
    const zkdl::CpuG1Point* blinding_generator = nullptr;
    const zkdl::CpuG1Point* u_generator = nullptr;
    const std::vector<CommitmentWire>* commitments = nullptr;
    const RelationStatement* statement = nullptr;
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr;
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr;
    zkdl::CpuG1MsmSession* msm_session = nullptr;
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts = nullptr;
    const std::string* spill_directory = nullptr;
};

inline bool qpad_selector_group_identity(
    const TensorDescriptor& descriptor, std::size_t basis_size,
    std::vector<byte>& identity, ZkRelationSegmentLayout& layout,
    std::string* error = nullptr) {
    SegmentedTensorProvider selector_oracle;
    if (!descriptor.valid(error) ||
        !qpad_record_required(descriptor.logical_shape, descriptor.padded_shape,
                              descriptor.visibility == 1U) ||
        !qpad_selector_provider_unhashed(
            descriptor, basis_size, selector_oracle, error) ||
        !zktr_segment_layout(descriptor.domain_size, basis_size, layout, error) ||
        layout.logical_domain != descriptor.domain_size) {
        if (error && error->empty())
            *error = "Qpad group member lacks a canonical padded tensor/layout";
        return false;
    }
    identity.clear();
    zkdl::append_string(identity, "zkdl.qpad.selector-identity.v2");
    zkdl::append_u32(identity, descriptor.domain_size);
    zkdl::append_u64(identity, descriptor.logical_shape.size());
    for (const auto axis : descriptor.logical_shape) zkdl::append_u64(identity, axis);
    zkdl::append_u64(identity, descriptor.padded_shape.size());
    for (const auto axis : descriptor.padded_shape) zkdl::append_u64(identity, axis);
    zkdl::append_u32(identity, layout.segment_size);
    zkdl::append_u32(identity, layout.segment_count);
    return true;
}

inline std::array<byte, 32U> qpad_group_key_digest(
    const std::vector<byte>& selector_identity,
    const RelationStatement& statement,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const ZkRelationSegmentLayout& layout,
    const zkdl::CpuIpaBasisContext* authenticated_basis) {
    std::uint32_t generator_count = 0U;
    if (!checked_relation_u32(generators.size(), generator_count,
                              "Qpad group generator count"))
        throw std::invalid_argument("Qpad group generator count exceeds uint32");
    const auto frame_digest = zktr_basis_digest_for_prefix(
        generators, blinding_generator, u_generator, layout.segment_size,
        zkdl::cpu_ipa_padded_size(layout.segment_size), authenticated_basis);
    const auto basis_digest = zktr_basis_digest_for_prefix(
        generators, blinding_generator, u_generator, generator_count,
        zkdl::cpu_ipa_padded_size(generator_count), authenticated_basis);
    if (statement.srs_digest == std::array<byte, 32U>{} ||
        statement.registry_digest == std::array<byte, 32U>{} ||
        statement.segment_commitments_digest == std::array<byte, 32U>{} ||
        frame_digest == std::array<byte, 32U>{} ||
        basis_digest == std::array<byte, 32U>{})
        throw std::invalid_argument("Qpad group key lacks authenticated statement/basis identity");
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.qpad.same-mask-rlc-group.v2");
    zkdl::append_u64(encoded, selector_identity.size());
    encoded.insert(encoded.end(), selector_identity.begin(), selector_identity.end());
    append_digest(encoded, statement.srs_digest);
    append_digest(encoded, statement.registry_digest);
    append_digest(encoded, statement.segment_commitments_digest);
    append_digest(encoded, basis_digest);
    append_digest(encoded, frame_digest);
    zkdl::append_u32(encoded, layout.segment_size);
    zkdl::append_u32(encoded, layout.segment_count);
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(blinding_generator));
    zkdl::append_g1(encoded, zkdl::cpu_g1_to_native(u_generator));
    return zkdl::sha256(encoded);
}

inline bool qpad_build_group_plans(
    const std::vector<TensorDescriptor>& descriptors,
    std::size_t basis_size,
    const RelationStatement* statement,
    const std::vector<zkdl::CpuG1Point>* generators,
    const zkdl::CpuG1Point* blinding_generator,
    const zkdl::CpuG1Point* u_generator,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    std::vector<QpadGroupPlan>& groups,
    std::string* error = nullptr) {
    struct Bucket {
        std::array<byte, 32U> selector_identity{};
        std::array<byte, 32U> group_key{};
        std::uint32_t domain_size = 0U;
        std::uint32_t segment_size = 0U;
        std::uint32_t segment_count = 0U;
        std::vector<const TensorDescriptor*> members;
        std::vector<std::uint32_t> descriptor_indices;
    };
    if (basis_size == 0U ||
        ((statement == nullptr) != (generators == nullptr)) ||
        ((statement == nullptr) != (blinding_generator == nullptr)) ||
        ((statement == nullptr) != (u_generator == nullptr)) ||
        (statement != nullptr && generators->size() != basis_size)) {
        if (error) *error = "Qpad group planner lacks a canonical common basis context";
        return false;
    }
    groups.clear();
    std::vector<Bucket> buckets;
    std::map<std::array<byte, 32U>, std::size_t> bucket_index;
    std::set<std::string> required_ids;
    try {
        for (std::size_t descriptor_index = 0U;
             descriptor_index < descriptors.size(); ++descriptor_index) {
            const auto& descriptor = descriptors[descriptor_index];
            if (!qpad_record_required(descriptor.logical_shape,
                                      descriptor.padded_shape,
                                      descriptor.visibility == 1U)) continue;
            if (!required_ids.insert(descriptor.id).second) {
                if (error) *error = "Qpad group planner found a duplicate required tensor id";
                return false;
            }
            std::vector<byte> identity_bytes;
            ZkRelationSegmentLayout layout;
            if (!qpad_selector_group_identity(
                    descriptor, basis_size, identity_bytes, layout, error)) return false;
            const auto selector_identity = zkdl::sha256(identity_bytes);
            std::array<byte, 32U> group_key = selector_identity;
            if (statement != nullptr) {
                group_key = qpad_group_key_digest(
                    identity_bytes, *statement, *generators,
                    *blinding_generator, *u_generator, layout,
                    authenticated_basis);
            }
            auto found = bucket_index.find(group_key);
            std::size_t index = 0U;
            if (found == bucket_index.end()) {
                index = buckets.size();
                bucket_index.emplace(group_key, index);
                Bucket bucket;
                bucket.selector_identity = selector_identity;
                bucket.group_key = group_key;
                bucket.domain_size = descriptor.domain_size;
                bucket.segment_size = layout.segment_size;
                bucket.segment_count = layout.segment_count;
                buckets.push_back(std::move(bucket));
            } else {
                index = found->second;
                const auto& bucket = buckets[index];
                if (bucket.selector_identity != selector_identity ||
                    bucket.domain_size != descriptor.domain_size ||
                    bucket.segment_size != layout.segment_size ||
                    bucket.segment_count != layout.segment_count) {
                    if (error) *error = "Qpad group-key collision across selector/layout identities";
                    return false;
                }
            }
            buckets[index].members.push_back(&descriptor);
            std::uint32_t canonical_index = 0U;
            if (!checked_relation_u32(descriptor_index, canonical_index,
                                      "Qpad descriptor index", error)) return false;
            buckets[index].descriptor_indices.push_back(canonical_index);
        }
        for (const auto& bucket : buckets) {
            for (std::size_t begin = 0U; begin < bucket.members.size();
                 begin += PaddingProofRecord::kMaxMembers) {
                QpadGroupPlan group;
                if (!checked_relation_u32(groups.size(), group.group_index,
                                          "Qpad group index", error) ||
                    group.group_index >= 0x10000000U) {
                    if (error && error->empty()) *error = "Qpad group index exceeds reserved role range";
                    return false;
                }
                group.group_key = bucket.group_key;
                group.domain_size = bucket.domain_size;
                group.segment_size = bucket.segment_size;
                group.segment_count = bucket.segment_count;
                const std::size_t end = std::min(
                    bucket.members.size(), begin + PaddingProofRecord::kMaxMembers);
                group.members.assign(bucket.members.begin() +
                                         static_cast<std::ptrdiff_t>(begin),
                                     bucket.members.begin() +
                                         static_cast<std::ptrdiff_t>(end));
                group.descriptor_indices.assign(
                    bucket.descriptor_indices.begin() +
                        static_cast<std::ptrdiff_t>(begin),
                    bucket.descriptor_indices.begin() +
                        static_cast<std::ptrdiff_t>(end));
                groups.push_back(std::move(group));
            }
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = std::string("Qpad group planning failed: ") + exception.what();
        return false;
    }
}

inline std::uint32_t qpad_group_aggregate_index(std::size_t group_index) {
    std::uint32_t checked_index = 0U;
    if (group_index >= 0x10000000U ||
        !checked_relation_u32(group_index, checked_index,
                              "Qpad group aggregate index"))
        throw std::invalid_argument("Qpad group aggregate index overflow");
    return 0x30000000U + checked_index;
}

inline ConstraintDescriptor qpad_constraint_descriptor(
    const QpadGroupPlan& group) {
    ConstraintDescriptor descriptor;
    descriptor.id = "gat.padding-group/" +
        zkdl::digest_hex(group.group_key) + "/" +
        std::to_string(group.group_index) + "/zero-mask-rlc-v2";
    descriptor.degree = 2U;
    descriptor.terms.push_back(
        Monomial{zkdl::fr_from_u32(1U), {0U, 1U}});
    return descriptor;
}

inline FrWire qpad_selector_mle_from_point(
    const TensorDescriptor& descriptor,
    const std::vector<FrWire>& msb_point) {
    std::vector<FrWire> low_bit_point(msb_point.rbegin(), msb_point.rend());
    if (low_bit_point.size() != log2_exact(descriptor.domain_size))
        throw std::invalid_argument("Qpad selector terminal point/domain mismatch");
    FrWire logical_indicator = zkdl::fr_from_u32(1U);
    std::size_t offset = 0U;
    for (std::size_t axis = descriptor.padded_shape.size(); axis > 0U; --axis) {
        const std::size_t current = axis - 1U;
        logical_indicator = zkdl::fr_mul(
            logical_indicator,
            pointwise_prefix_mle(low_bit_point, offset,
                                 descriptor.padded_shape[current],
                                 descriptor.logical_shape[current]));
        for (std::uint64_t value = descriptor.padded_shape[current];
             value > 1U; value >>= 1U) ++offset;
    }
    if (offset != low_bit_point.size())
        throw std::invalid_argument("Qpad selector axes do not cover its terminal point");
    return zkdl::fr_sub(zkdl::fr_from_u32(1U), logical_indicator);
}

inline ConstraintDescriptor qpad_terminal_descriptor(
    const QpadGroupPlan& group, const std::vector<FrWire>& rho,
    const std::vector<FrWire>& msb_point) {
    if (group.members.empty())
        throw std::invalid_argument("Qpad terminal descriptor has no group members");
    const ConstraintDescriptor sumcheck = qpad_constraint_descriptor(group);
    const FrWire selector_value = qpad_selector_mle_from_point(
        *group.members.front(), msb_point);
    const FrWire eqmask_value = pointwise_terminal_weight(
        sumcheck, rho, msb_point);
    ConstraintDescriptor terminal;
    terminal.id = sumcheck.id + "/terminal";
    terminal.degree = 1U;
    terminal.terms.push_back(Monomial{
        zkdl::fr_mul(eqmask_value, selector_value), {0U}});
    return terminal;
}

inline bool qpad_absorb_group_header(
    const QpadGroupPlan& group,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript,
    const std::string& operation_context,
    std::vector<std::uint32_t>& member_commitment_indices,
    std::string* error = nullptr) {
    if (group.members.empty() ||
        group.members.size() != group.descriptor_indices.size() ||
        group.members.size() > PaddingProofRecord::kMaxMembers ||
        statement.segment_commitments_digest == std::array<byte, 32U>{}) {
        if (error) *error = "Qpad group transcript header is incomplete";
        return false;
    }
    std::vector<byte> encoded;
    zkdl::append_string(encoded, "zkdl.qpad.same-mask-rlc.transcript.v2");
    zkdl::append_u32(encoded, group.group_index);
    append_digest(encoded, group.group_key);
    zkdl::append_u32(encoded, group.domain_size);
    zkdl::append_u32(encoded, group.segment_size);
    zkdl::append_u32(encoded, group.segment_count);
    append_digest(encoded, statement.segment_commitments_digest);
    append_digest(encoded, statement.statement_digest);
    zkdl::append_u64(encoded, group.members.size());
    member_commitment_indices.clear();
    member_commitment_indices.reserve(group.members.size());
    std::set<std::uint32_t> seen_commitments;
    for (std::size_t member = 0U; member < group.members.size(); ++member) {
        const TensorDescriptor& descriptor = *group.members[member];
        const auto found = std::find_if(
            commitments.begin(), commitments.end(),
            [&](const CommitmentWire& wire) {
                return wire.tensor_id == descriptor.id;
            });
        std::uint32_t commitment_index = 0U;
        if (!descriptor.valid(error) || found == commitments.end() ||
            found->domain_size != descriptor.domain_size ||
            !checked_relation_u32(
                static_cast<std::size_t>(std::distance(commitments.begin(), found)),
                commitment_index, "Qpad member commitment index", error) ||
            !seen_commitments.insert(commitment_index).second) {
            if (error && error->empty()) *error = "Qpad group member commitment map is invalid";
            return false;
        }
        zkdl::append_u32(encoded, group.descriptor_indices[member]);
        zkdl::append_string(encoded, descriptor.id);
        zkdl::append_u32(encoded, descriptor.visibility);
        zkdl::append_u32(encoded, descriptor.domain_size);
        zkdl::append_u64(encoded, descriptor.logical_shape.size());
        for (const auto axis : descriptor.logical_shape)
            zkdl::append_u64(encoded, axis);
        zkdl::append_u64(encoded, descriptor.padded_shape.size());
        for (const auto axis : descriptor.padded_shape)
            zkdl::append_u64(encoded, axis);
        zkdl::append_string(encoded, descriptor.mle_order);
        zkdl::append_string(encoded, descriptor.stack_axis);
        zkdl::append_string(encoded, descriptor.mapping);
        zkdl::append_string(encoded, descriptor.padding_rule);
        zkdl::append_string(encoded, descriptor.registry_id);
        zkdl::append_u32(encoded, commitment_index);
        append_digest(encoded, found->segment_commitment_digest);
        zkdl::append_g1(encoded, found->commitment);
        member_commitment_indices.push_back(commitment_index);
    }
    transcript.absorb(operation_context + ".rlc-v2.group-header", encoded);
    return true;
}

inline bool qpad_group_randomizers(
    const QpadGroupPlan& group,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement,
    zkdl::Transcript& transcript,
    const std::string& operation_context,
    std::vector<std::uint32_t>& member_commitment_indices,
    std::vector<FrWire>& alphas,
    std::string* error = nullptr) {
    if (!qpad_absorb_group_header(
            group, commitments, statement, transcript, operation_context,
            member_commitment_indices, error)) return false;
    alphas.clear();
    alphas.reserve(group.members.size());
    for (std::size_t member = 0U; member < group.members.size(); ++member) {
        bool sampled = false;
        for (std::uint32_t attempt = 0U; attempt < 16U; ++attempt) {
            const FrWire alpha = zkdl::fr_wire(transcript.challenge(
                operation_context + ".rlc-v2.alpha." +
                std::to_string(member) + ".attempt." +
                std::to_string(attempt)));
            if (!zkdl::fr_canonical(alpha)) {
                if (error) *error = "Qpad Fiat-Shamir coefficient is not canonical";
                return false;
            }
            if (alpha != FrWire{}) {
                alphas.push_back(alpha);
                sampled = true;
                break;
            }
        }
        if (!sampled) {
            if (error) *error = "Qpad Fiat-Shamir failed to derive a nonzero coefficient";
            return false;
        }
    }
    return true;
}

inline bool prove_padding_record(
    const TensorDescriptor& descriptor, std::size_t record_index,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::map<std::string, FrWire>& blinders,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator,
    const std::vector<CommitmentWire>& commitments,
    const RelationStatement& statement, zkdl::Transcript& transcript,
    PaddingProofRecord& record, std::string* error = nullptr,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts = nullptr,
    const std::string* spill_directory = nullptr,
    QpadSelectorXiCache* selector_cache = nullptr) {
    try {
        PreparedQpadSelector prepared_selector;
        if (!QpadSelectorCommitmentBuilder::prepare(
                descriptor, generators.size(), prepared_selector, error))
            return false;
        const SegmentedTensorProvider& selector_provider =
            prepared_selector.provider();
        const auto tensor = tensor_map.find(descriptor.id);
        if (tensor == tensor_map.end() || tensor->second == nullptr ||
            tensor->second->domain_size != descriptor.domain_size) {
            if (error) *error = "Qpad tensor witness is missing or has the wrong domain";
            return false;
        }
        const auto commitment = std::find_if(
            commitments.begin(), commitments.end(), [&](const CommitmentWire& value) {
                return value.tensor_id == descriptor.id &&
                       value.domain_size == descriptor.domain_size;
            });
        if (commitment == commitments.end()) {
            if (error) *error = "Qpad tensor commitment is missing";
            return false;
        }
        std::uint32_t commitment_index = 0U;
        if (!checked_relation_u32(
                static_cast<std::size_t>(std::distance(commitments.begin(), commitment)),
                commitment_index, "Qpad tensor commitment index", error))
            return false;
        SegmentedTensorProvider tensor_provider;
        if (!tensor->second->segmented(generators.size(), tensor_provider, error)) return false;
        CommitmentWire selector_wire;
        SegmentCommitmentMaterial selector_material;
        SegmentCommitmentValidationReceipt selector_receipt;
        zkdl::CpuG1Point selector_commitment;
        if (segment_blinding_scalars != nullptr) {
            const auto build_with_cache = [&](QpadSelectorXiCache& cache) {
                return QpadSelectorCommitmentBuilder::build(
                    prepared_selector, generators, ipa_blinding_generator,
                    ipa_u_generator, statement, authenticated_basis, msm_session,
                    cache, selector_wire,
                    selector_material, &selector_receipt, error);
            };
            bool selector_built = false;
            if (selector_cache != nullptr) {
                selector_built = build_with_cache(*selector_cache);
            } else {
                QpadSelectorXiCache local_selector_cache;
                selector_built = build_with_cache(local_selector_cache);
            }
            if (!selector_built) return false;
            selector_commitment = zkdl::cpu_g1_from_native(selector_wire.commitment);
        } else {
            if (!QpadSelectorCommitmentBuilder::compute_digest(
                    prepared_selector, error)) return false;
            selector_commitment = zktr_segmented_commitment(
                selector_provider, generators, ipa_blinding_generator, FrWire{}, error);
        }
        if (!prepared_selector.selector_digest_ready()) {
            if (error) *error = "Qpad selector builder did not produce its digest";
            return false;
        }
        if (selector_commitment.is_zero()) return false;
        const ConstraintDescriptor constraint = qpad_constraint_descriptor(descriptor.id);
        const std::string operation_context = "gat.relation.padding." + descriptor.id;
        const std::vector<SegmentedTensorProvider> providers = {
            tensor_provider, selector_provider};
        const auto rho = pointwise_challenges(constraint, descriptor.domain_size, providers.size(), operation_context, transcript);
        const ConstraintDescriptor weighted = pointwise_sumcheck_descriptor(constraint, providers.size());
        PointwiseEqmaskWeightTable eqmask_table;
        if (!eqmask_table.build(rho, error)) return false;
        NativeOperationProgressReporter qpad_record_progress(
            "qpad_record", descriptor.id);
        std::vector<SegmentedTensorProvider> weighted_providers;
        if (!pointwise_providers(constraint, providers, rho, weighted_providers,
                                 error, &eqmask_table, &qpad_record_progress)) return false;
        const std::uint64_t selector_cache_live_bytes = selector_cache != nullptr
            ? selector_cache->peak_live_bytes() : 0U;
        if (selector_cache_live_bytes >
            std::numeric_limits<std::uint64_t>::max() -
                eqmask_table.peak_live_bytes) {
            if (error) *error = "Qpad selector cache live-memory bound overflows";
            return false;
        }
        const std::uint64_t qpad_auxiliary_live_bytes =
            eqmask_table.peak_live_bytes + selector_cache_live_bytes;
        ZkSumcheckProverResult sumcheck;
        if (!prove_zk_sumcheck_segmented(
                weighted, weighted_providers, weighted.degree, descriptor.domain_size, generators,
                ipa_blinding_generator,
                ipa_u_generator, FrWire{}, statement, operation_context,
                transcript, sumcheck, error, 0U,
                zksc_folded_state_within_budget(
                    weighted_providers, weighted, qpad_auxiliary_live_bytes),
                "qpad_sumcheck", descriptor.id.c_str(), spill_directory, true,
                qpad_auxiliary_live_bytes)) return false;
        weighted_providers.clear();
        eqmask_table.clear();
        std::vector<FrWire> point;
        for (const auto& challenge : sumcheck.challenges)
            point.push_back(zkdl::fr_wire(challenge));
        std::reverse(point.begin(), point.end());
        const std::vector<zkdl::CpuG1Point> canonical_commitments = {
            zkdl::cpu_g1_from_native(commitment->commitment), selector_commitment};
        const std::vector<FrWire> canonical_blinders = {
            relation_blinding_scalar(*tensor->second, blinders), FrWire{}};
        const std::vector<std::uint32_t> commitment_indices = {
            commitment_index, qpad_selector_index(record_index)};
        std::vector<SegmentCommitmentMaterial> terminal_materials;
        std::vector<const SegmentCommitmentValidationReceipt*> terminal_receipts;
        if (segment_blinding_scalars != nullptr) {
            SegmentCommitmentMaterial tensor_material;
            if (!zktr_material_from_wire(
                    *commitment, *segment_blinding_scalars, tensor_material, error)) return false;
            if (!zkdl::cpu_g1_equal(selector_commitment,
                                    zkdl::cpu_g1_from_native(selector_wire.commitment))) {
                if (error) *error = "Qpad selector segment material mismatch";
                return false;
            }
            if (validation_receipts != nullptr) {
                const auto tensor_receipt = validation_receipts->find(descriptor.id);
                SegmentedTensorProvider committed_tensor_provider;
                if (tensor_receipt == validation_receipts->end() ||
                    !tensor->second->segmented(generators.size(),
                                               committed_tensor_provider, error) ||
                    !tensor_receipt->second.matches_context(
                        descriptor.id, commitment->visibility,
                        committed_tensor_provider, generators,
                        ipa_blinding_generator, ipa_u_generator,
                        authenticated_basis, *commitment, tensor_material, error) ||
                    !selector_receipt.valid() ||
                    !selector_receipt.matches_context(
                        "qpad/" + descriptor.id, 1U, selector_provider,
                        generators, ipa_blinding_generator, ipa_u_generator,
                        authenticated_basis, selector_wire, selector_material,
                        error)) return false;
                terminal_receipts.push_back(&tensor_receipt->second);
                terminal_receipts.push_back(&selector_receipt);
            }
            terminal_materials.push_back(std::move(tensor_material));
            terminal_materials.push_back(std::move(selector_material));
        }
        const ConstraintDescriptor terminal_descriptor = pointwise_terminal_descriptor(constraint, rho, point);
        ZkTerminalRelationProverResult terminal;
        if (!prove_zk_terminal_relation_segmented(
                terminal_descriptor, providers, canonical_commitments, canonical_blinders,
                commitment_indices, point, generators,
                ipa_blinding_generator, ipa_u_generator, statement,
                zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
                zkdl::cpu_g1_from_native(sumcheck.proof.final_claim_commitment),
                sumcheck.final_q_blinder, FrWire{}, operation_context,
                transcript, terminal, error,
                segment_blinding_scalars != nullptr ? &terminal_materials : nullptr,
                0U, &qpad_record_progress, authenticated_basis, msm_session,
                validation_receipts != nullptr ? &terminal_receipts : nullptr)) return false;
        record = PaddingProofRecord{};
        record.tensor_id = descriptor.id;
        record.commitment_index = commitment_index;
        record.domain_size = descriptor.domain_size;
        record.selector_digest = prepared_selector.selector_digest();
        record.relation.descriptor = constraint;
        record.relation.opening_tensor_indices = {0U, 1U};
        record.relation.opening_commitment_indices = commitment_indices;
        record.relation.zk_sumcheck = std::move(sumcheck.proof);
        record.relation.terminal_relation = std::move(terminal.proof);
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = std::string("Qpad prover failed: ") + exception.what();
        return false;
    }
}

inline bool verify_padding_record(
    const PaddingProofRecord& record, const TensorDescriptor& descriptor,
    std::size_t record_index, const std::vector<CommitmentWire>& commitments,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& ipa_blinding_generator,
    const zkdl::CpuG1Point& ipa_u_generator, const RelationStatement& statement,
    zkdl::Transcript& transcript, std::string* error = nullptr,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    QpadSelectorXiCache* selector_cache = nullptr) {
    try {
        PreparedQpadSelector prepared_selector;
        if (!QpadSelectorCommitmentBuilder::prepare(
                descriptor, generators.size(), prepared_selector, error) ||
            record.version != PaddingProofRecord::kVersion ||
            record.tensor_id != descriptor.id || record.domain_size != descriptor.domain_size) {
            if (error && error->empty()) *error = "Qpad proof descriptor/digest mismatch";
            return false;
        }
        const SegmentedTensorProvider& selector_provider =
            prepared_selector.provider();
        if (record.commitment_index >= commitments.size() ||
            commitments[record.commitment_index].tensor_id != descriptor.id ||
            commitments[record.commitment_index].domain_size != descriptor.domain_size) {
            if (error) *error = "Qpad proof commitment index mismatch";
            return false;
        }
        const CommitmentWire& input_wire = commitments[record.commitment_index];
        if (!relation_segment_commitment_valid(statement, input_wire, error)) return false;
        if (record.relation.opening_tensor_indices != std::vector<std::uint32_t>{0U, 1U} ||
            record.relation.opening_commitment_indices != std::vector<std::uint32_t>{
                record.commitment_index, qpad_selector_index(record_index)}) {
            if (error) *error = "Qpad proof terminal commitment map is not canonical";
            return false;
        }
        CommitmentWire selector_wire;
        SegmentCommitmentMaterial selector_material;
        const auto build_with_cache = [&](QpadSelectorXiCache& cache) {
            return QpadSelectorCommitmentBuilder::build(
                prepared_selector, generators, ipa_blinding_generator,
                ipa_u_generator, statement, authenticated_basis, msm_session,
                cache, selector_wire, selector_material, nullptr, error);
        };
        bool selector_built = false;
        if (selector_cache != nullptr) {
            selector_built = build_with_cache(*selector_cache);
        } else {
            QpadSelectorXiCache local_selector_cache;
            selector_built = build_with_cache(local_selector_cache);
        }
        if (!selector_built) return false;
        if (!prepared_selector.selector_digest_ready() ||
            record.selector_digest != prepared_selector.selector_digest()) {
            if (error) *error = "Qpad proof descriptor/digest mismatch";
            return false;
        }
        const zkdl::CpuG1Point selector_commitment = zkdl::cpu_g1_from_native(selector_wire.commitment);
        if (selector_commitment.is_zero()) return false;
        const auto& terminal_proof = record.relation.terminal_relation;
        if (terminal_proof.terminal_segment_openings.size() != 2U) {
            if (error) *error = "Qpad terminal opening groups do not match its two canonical tensors";
            return false;
        }
        const auto opening_group_matches_wire = [&](std::size_t tensor_index, const CommitmentWire& wire, std::uint32_t commitment_index) {
            const auto& openings = terminal_proof.terminal_segment_openings[tensor_index];
            if (openings.size() != wire.segment_commitments.size()) {
                if (error) *error = "Qpad terminal opening count differs from segment commitment layout";
                return false;
            }
            for (std::size_t segment_index = 0U; segment_index < openings.size(); ++segment_index) {
                const auto& opening = openings[segment_index];
                const auto& segment = wire.segment_commitments[segment_index];
                if (opening.commitment_index != commitment_index || opening.evaluation_commitment_index != zktr_terminal_evaluation_index(tensor_index) || opening.generator_size != wire.segment_size || opening.segment_index != segment.segment_index || opening.logical_domain != wire.domain_size || opening.logical_offset != segment.logical_offset || opening.actual_length != segment.actual_length || opening.padded_length != segment.padded_length || !zkdl::cpu_g1_equal(zkdl::cpu_g1_from_native(opening.xi), zkdl::cpu_g1_from_native(segment.commitment))) {
                    if (error) *error = "Qpad terminal opening does not match its segment commitment wire";
                    return false;
                }
            }
            return true;
        };
        if (!opening_group_matches_wire(0U, input_wire, record.commitment_index) || !opening_group_matches_wire(1U, selector_wire, qpad_selector_index(record_index))) return false;
        const std::vector<zkdl::CpuG1Point> canonical_commitments = {
            zkdl::cpu_g1_from_native(commitments[record.commitment_index].commitment),
            selector_commitment};
        const ConstraintDescriptor constraint = qpad_constraint_descriptor(descriptor.id);
        std::vector<byte> expected_descriptor, actual_descriptor;
        append_constraint(expected_descriptor, constraint);
        append_constraint(actual_descriptor, record.relation.descriptor);
        if (expected_descriptor != actual_descriptor) {
            if (error) *error = "Qpad relation descriptor mismatch";
            return false;
        }
        const std::string operation_context = "gat.relation.padding." + descriptor.id;
        const auto rho = pointwise_challenges(constraint, descriptor.domain_size, canonical_commitments.size(), operation_context, transcript);
        const ConstraintDescriptor weighted = pointwise_sumcheck_descriptor(constraint, canonical_commitments.size());
        ZkSumcheckVerifierResult sumcheck;
        if (!verify_zk_sumcheck(
                record.relation.zk_sumcheck, weighted, weighted.degree, descriptor.domain_size,
                generators, ipa_blinding_generator, ipa_u_generator, FrWire{}, statement,
                operation_context, transcript, sumcheck, error)) return false;
        std::vector<FrWire> point;
        for (const auto& challenge : sumcheck.challenges)
            point.push_back(zkdl::fr_wire(challenge));
        std::reverse(point.begin(), point.end());
        const ConstraintDescriptor terminal_descriptor = pointwise_terminal_descriptor(constraint, rho, point);
        ZkTerminalRelationVerifierResult terminal;
        return verify_zk_terminal_relation(
            record.relation.terminal_relation, terminal_descriptor, canonical_commitments,
            record.relation.opening_commitment_indices, point, generators,
            ipa_blinding_generator, ipa_u_generator, statement,
            zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
            zkdl::cpu_g1_from_native(sumcheck.final_claim_commitment), FrWire{},
            operation_context, transcript, terminal, error, nullptr, 0U,
            authenticated_basis, msm_session);
    } catch (const std::exception& exception) {
        if (error) *error = std::string("Qpad verifier failed: ") + exception.what();
        return false;
    }
}

inline bool prove_padding_records(
    const ProverContext& context,
    const std::map<std::string, const TensorWitness*>& tensor_map,
    const std::vector<CommitmentWire>& commitments, zkdl::Transcript& transcript,
    std::vector<PaddingProofRecord>& records, std::string* error = nullptr,
    const std::map<std::string, std::vector<FrWire>>* segment_blinding_scalars = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::map<std::string, SegmentCommitmentValidationReceipt>*
        validation_receipts = nullptr,
    std::size_t total_records = std::numeric_limits<std::size_t>::max()) {
    records.clear();
    if (total_records == std::numeric_limits<std::size_t>::max()) {
        total_records = 0U;
        for (const auto& descriptor : context.tensor_descriptors)
            if (qpad_record_required(descriptor.logical_shape,
                                     descriptor.padded_shape,
                                     descriptor.visibility == 1U)) ++total_records;
    }
    QpadSelectorXiCache selector_cache;
    NativeOperationProgressReporter record_progress("qpad_records", "padding_records");
    std::size_t index = 0U;
    for (const auto& descriptor : context.tensor_descriptors) {
        if (!qpad_record_required(descriptor.logical_shape,
                                  descriptor.padded_shape,
                                  descriptor.visibility == 1U)) continue;
        PaddingProofRecord record;
        if (!prove_padding_record(
                descriptor, index, tensor_map, context.blinding_scalars,
                context.ipa_generators, context.ipa_blinding_generator,
                context.ipa_u_generator, commitments, context.statement,
                transcript, record, error, segment_blinding_scalars,
                &context.ipa_generators, msm_session,
                validation_receipts,
                context.sumcheck_spill_directory.empty()
                    ? nullptr : &context.sumcheck_spill_directory,
                &selector_cache)) return false;
        const std::vector<byte> encoded = record.serialize();
        transcript.absorb("gat.relation.padding", encoded);
        records.push_back(std::move(record));
        ++index;
        record_progress.update("records", index, total_records);
    }
    return true;
}

inline bool verify_padding_records(
    const VerifierContext& context, const Proof& proof,
    zkdl::Transcript& transcript, std::string* error = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr) {
    std::vector<const TensorDescriptor*> expected;
    for (const auto& descriptor : context.tensor_descriptors) {
        if (qpad_record_required(descriptor.logical_shape,
                                 descriptor.padded_shape,
                                 descriptor.visibility == 1U))
            expected.push_back(&descriptor);
    }
    if (proof.padding_records.size() != expected.size()) {
        if (error) *error = "Qpad proof record count does not match authenticated tensor families";
        return false;
    }
    QpadSelectorXiCache selector_cache;
    for (std::size_t index = 0U; index < expected.size(); ++index) {
        if (!verify_padding_record(proof.padding_records[index], *expected[index], index,
                                   proof.commitments, context.ipa_generators,
                                   context.ipa_blinding_generator,
                                   context.ipa_u_generator, context.statement,
                                   transcript, error,
                                   &context.ipa_generators,
                                   msm_session, &selector_cache)) return false;
        const std::vector<byte> encoded = proof.padding_records[index].serialize();
        transcript.absorb("gat.relation.padding", encoded);
    }
    return true;
}

// Sorted-lookup proving/verifying is defined below the relation entry points
// so the wire implementation stays together.  Keep the declarations here so
// both entry points compile without relying on declaration order.
inline bool prove_authenticated_sorted_lookup(
    const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
    AuthenticatedSortedLookupProviderSet providers,
    const std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>& commitment_refs,
    const std::vector<zkdl::CpuG1Point>& outer_commitments,
    const std::vector<FrWire>& outer_blinders,
    const std::vector<SegmentCommitmentMaterial>& outer_materials,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::AuthenticatedSortedLookupProof& proof,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr,
    const std::vector<SegmentCommitmentValidationReceipt>* outer_receipts = nullptr,
    const std::vector<CommitmentWire>* all_commitments = nullptr,
    const std::string* spill_directory = nullptr);

inline bool verify_authenticated_sorted_lookup(
    const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
    const SegmentedTensorProvider& authenticated_table_keys,
    const SegmentedTensorProvider& authenticated_table_active,
    const std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>& commitment_refs,
    const std::vector<zkdl::CpuG1Point>& outer_commitments,
    const std::vector<CommitmentWire>& outer_wires,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    const zkdl::AuthenticatedSortedLookupProof& proof,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis = nullptr,
    zkdl::CpuG1MsmSession* msm_session = nullptr);

struct RelationProofProgress {
    using Clock = std::chrono::steady_clock;
    Clock::time_point started = Clock::now();
    std::string phase = "entry_validation";
    std::string operation;
    bool completed = false;

    static void emit(const char* phase_name, const char* status,
                     Clock::time_point start, const std::string& detail = std::string()) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        std::clog << "zkdl_native_progress phase=" << phase_name << " status=" << status
                  << " elapsed_wall_seconds=" << elapsed;
        if (!detail.empty()) std::clog << ' ' << detail;
        std::clog << '\n' << std::flush;
    }

    RelationProofProgress(std::size_t operation_count, std::size_t tensor_count,
                          std::size_t lookup_count) {
        emit("relation", "START", started,
             "operations=" + std::to_string(operation_count) +
             " tensors=" + std::to_string(tensor_count) +
             " lookups=" + std::to_string(lookup_count));
    }

    ~RelationProofProgress() {
        if (!completed) {
            std::string detail = "failed_phase=" + phase;
            if (!operation.empty()) detail += " operation_id=" + operation;
            emit("relation", "FAIL", started, detail);
        }
    }

    void set_phase(const std::string& value) {
        phase = value;
        operation.clear();
    }

    void set_operation(const std::string& value) {
        phase = "operation_proof";
        operation = value;
    }

    void finish(std::size_t commitment_count, std::size_t operation_count,
                std::size_t lookup_count) {
        completed = true;
        emit("relation", "PASS", started,
             "commitments=" + std::to_string(commitment_count) +
             " operations=" + std::to_string(operation_count) +
             " sorted_lookups=" + std::to_string(lookup_count));
    }
};

inline bool prove_impl(const ProverContext& context, Proof& proof, std::string* error, ProverContext* consume_owner) {
    try {
        if (consume_owner != nullptr && static_cast<const ProverContext*>(consume_owner) != &context) throw std::invalid_argument("consuming prover context owner mismatch");
        const std::size_t operation_count = context.operations.size();
        RelationProofProgress progress(operation_count, context.tensors.size(),
                                       context.sorted_lookup_descriptors.size());
        const auto emit_progress = [&](const char* phase, const char* status,
                                       std::chrono::steady_clock::time_point phase_start,
                                       const std::string& detail = std::string()) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - phase_start).count();
            std::clog << "zkdl_native_progress phase=" << phase << " status=" << status
                      << " elapsed_wall_seconds=" << elapsed;
            if (!detail.empty()) std::clog << ' ' << detail;
            std::clog << '\n' << std::flush;
        };
        progress.set_phase("descriptor_validation");
        if (!descriptor_vector_valid(context.operations, error)) return false;
        if (!formal_relation_components_supported(context.operations, error)) return false;
        progress.set_phase("backend_validation");
        if (context.gpu_backend) {
            if (error) *error =
                "BLOCKED: formal zkDL GPU GAT proving is unavailable until every relation family "
                "and terminal opening runs on native CUDA cryptographic kernels";
            return false;
        }
        if (context.tensors.empty() || context.generators.empty() ||
            context.ipa_generators.empty() || context.ipa_blinding_generator.is_zero() ||
            context.ipa_u_generator.is_zero() ||
            !same_g1_basis(context.generators, context.ipa_generators)) {
            if (error) *error = "BLOCKED: relation prover requires authenticated independent IPA G/H/U basis";
            return false;
        }
        if (!context.hiding_commitment_records.empty()) {
            if (error) *error = "BLOCKED: formal GATR uses IPA knowledge openings, not hiding records";
            return false;
        }
        std::map<std::string, const TensorWitness*> tensor_map;
        progress.set_phase("tensor_validation");
        for (const auto& tensor : context.tensors) {
            if (!tensor.metadata_valid(error) || !tensor_map.emplace(tensor.id, &tensor).second) {
                if (error && error->empty()) *error = "duplicate relation tensor id";
                return false;
            }
        }
        if (!relation_blinding_scalars_unique(context.tensors,
                                              context.blinding_scalars, error)) return false;
        progress.set_phase("context_validation");
        proof = Proof{};
        proof.statement = context.statement;
        proof.statement.segment_binding_version = RelationStatement::kSegmentBindingVersion;
        // This map is proof-local: the outer commitment stage fills it once,
        // and every ordinary terminal/sparse/FAC path receives the same
        // immutable shares explicitly.  No prover-context mutation or dead
        // material vector is used as hidden state.
        std::map<std::string, std::vector<FrWire>> segment_blinding_scalars;
        std::map<std::string, SegmentCommitmentValidationReceipt>
            segment_commitment_receipts;
        if (context.sorted_lookup_descriptors.size() != context.sorted_lookup_providers.size() ||
            context.sorted_lookup_descriptors.size() != context.sorted_lookup_commitment_refs.size() ||
            context.sorted_lookup_descriptors.size() != context.sorted_lookup_outer_commitments.size() ||
            context.sorted_lookup_descriptors.size() != context.sorted_lookup_outer_blinders.size() ||
            context.sorted_lookup_descriptors.size() != context.sorted_lookup_outer_segment_blinders.size() ||
            context.sorted_lookup_descriptors.size() != context.sorted_lookup_outer_segment_commitments.size() ||
            context.sorted_lookup_descriptors.size() > (1U << 16U)) {
            throw std::invalid_argument("sorted-lookup context descriptor/provider count mismatch");
        }
        const std::array<byte, 32> expected_public_digest =
            zkdl::sha256(proof.statement.public_output);
        if (proof.statement.public_output.empty() ||
            (proof.statement.public_output_digest != std::array<byte, 32>{} &&
             proof.statement.public_output_digest != expected_public_digest)) {
            throw std::invalid_argument("public Y digest is not canonical");
        }
        proof.statement.public_output_digest = expected_public_digest;
        proof.hiding_commitment_records.clear();
        const auto commitment_phase_start = std::chrono::steady_clock::now();
        progress.set_phase("commitment");
        emit_progress("commitment", "START", commitment_phase_start,
                      "tensor_commitments=" + std::to_string(context.tensors.size()) +
                      " lookup_outer_groups=" + std::to_string(context.sorted_lookup_descriptors.size()));
        RelationSegmentProgressReporter commitment_progress("commitment");
        const auto msm_prepare_start = std::chrono::steady_clock::now();
        zkdl::CpuG1MsmSession msm_session(context.ipa_generators);
        msm_session.set_hyrax_gpu_fold_session(context.hyrax_gpu_fold_session);
        const double msm_prepare_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - msm_prepare_start).count();
        if (context.relation_commitment_time_seconds != nullptr)
            *context.relation_commitment_time_seconds += msm_prepare_seconds;
        if (context.commitment_open_time_seconds != nullptr)
            *context.commitment_open_time_seconds += msm_prepare_seconds;
        for (const auto& tensor : context.tensors) {
            SegmentedTensorProvider provider;
            if (!tensor.segmented(context.ipa_generators.size(), provider, error)) return false;
            CommitmentWire commitment;
            SegmentCommitmentMaterial material;
            SegmentCommitmentValidationReceipt validation_receipt;
            const FrWire randomness = relation_blinding_scalar(tensor, context.blinding_scalars);
            const auto commitment_start = std::chrono::steady_clock::now();
            double commitment_crypto_seconds = 0.0;
            const double provider_io_before = context.provider_file_io_decode_time_seconds != nullptr
                ? *context.provider_file_io_decode_time_seconds : 0.0;
            const SegmentCommitmentProgressCallback segment_progress =
                [&tensor, &commitment_progress](std::uint32_t completed,
                                                std::uint32_t total) {
                    commitment_progress.update(tensor.id, completed, total);
                };
            const bool commitment_built = zktr_build_segment_commitment_material(
                    tensor.id, tensor.visibility, provider, context.ipa_generators,
                    context.ipa_blinding_generator, context.ipa_u_generator, randomness,
                    proof.statement, commitment, material, error,
                    &commitment_crypto_seconds, 0U, &segment_progress,
                    &context.ipa_generators, &msm_session, &validation_receipt);
            const double commitment_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - commitment_start).count();
            if (context.commitment_open_time_seconds != nullptr)
                *context.commitment_open_time_seconds += commitment_crypto_seconds;
            if (context.relation_commitment_time_seconds != nullptr)
                *context.relation_commitment_time_seconds += commitment_seconds;
            if (context.relation_commitment_file_io_decode_time_seconds != nullptr &&
                context.provider_file_io_decode_time_seconds != nullptr) {
                const double provider_io_delta = *context.provider_file_io_decode_time_seconds - provider_io_before;
                if (provider_io_delta < 0.0 || provider_io_delta > commitment_seconds) {
                    if (error) *error = "native tensor explicit provider I/O timing exceeded its commitment interval";
                    return false;
                }
                *context.relation_commitment_file_io_decode_time_seconds += provider_io_delta;
            }
            if (!commitment_built) return false;
            segment_blinding_scalars[tensor.id] = material.blinders;
            if (!validation_receipt.valid()) {
                if (error) *error = "tensor commitment did not produce its MSM validation receipt";
                return false;
            }
            segment_commitment_receipts.emplace(tensor.id,
                                                std::move(validation_receipt));
            proof.commitments.push_back(commitment);
        }
        std::vector<std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>> sorted_refs =
            context.sorted_lookup_commitment_refs;
        std::vector<std::vector<zkdl::CpuG1Point>> sorted_outer =
            context.sorted_lookup_outer_commitments;
        std::vector<std::vector<FrWire>> sorted_blinders = context.sorted_lookup_outer_blinders;
        std::vector<std::vector<SegmentCommitmentMaterial>> sorted_materials;
        sorted_materials.resize(sorted_refs.size());
        std::vector<std::vector<SegmentCommitmentValidationReceipt>> sorted_receipts(
            sorted_refs.size());
        const auto sorted_outer_provider = [](
            const AuthenticatedSortedLookupProviderSet& providers,
            const zkdl::AuthenticatedSortedLookupCommitmentRef& ref,
            SegmentedTensorProvider& provider) {
            if (ref.role == zkdl::kSortedLookupCommitmentQueryX) provider = providers.query_x;
            else if (ref.role == zkdl::kSortedLookupCommitmentQueryY) provider = providers.query_y;
            else if (ref.role == zkdl::kSortedLookupCommitmentQueryKey) provider = providers.query_keys;
            else if (ref.role == zkdl::kSortedLookupCommitmentQueryPrefix) provider = providers.query_prefix;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergeKey) provider = providers.merge_keys;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergeDelta) provider = providers.merge_delta;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergeMarker) provider = providers.merge_marker;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergeInverse) provider = providers.merge_inverse;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergePrefix) provider = providers.merge_prefix;
            else if (ref.role == zkdl::kSortedLookupCommitmentMergeBit &&
                     ref.column_index < providers.merge_bits.size())
                provider = providers.merge_bits[ref.column_index];
            else return false;
            return provider.valid(nullptr);
        };
        std::size_t sorted_id_offset = 0U;
        for (std::size_t lookup_index = 0U;
             lookup_index < sorted_refs.size(); ++lookup_index) {
            if (sorted_refs[lookup_index].size() != sorted_outer[lookup_index].size() ||
                sorted_refs[lookup_index].size() != sorted_blinders[lookup_index].size() ||
                sorted_refs[lookup_index].size() !=
                    context.sorted_lookup_outer_segment_blinders[lookup_index].size() ||
                sorted_refs[lookup_index].size() !=
                    context.sorted_lookup_outer_segment_commitments[lookup_index].size())
                throw std::invalid_argument("sorted-lookup outer commitment map mismatch");
            for (std::size_t role_index = 0U;
                 role_index < sorted_refs[lookup_index].size(); ++role_index) {
                auto& ref = sorted_refs[lookup_index][role_index];
                if (!checked_relation_u32(proof.commitments.size(),
                                          ref.commitment_index,
                                          "sorted-lookup commitment index", error))
                    return false;
                if (!ref.valid(error) || !zkdl::cpu_pedersen_strict_g1(
                        sorted_outer[lookup_index][role_index], error) ||
                    !zkdl::fr_canonical(sorted_blinders[lookup_index][role_index]) ||
                    zkdl::fr_compare(sorted_blinders[lookup_index][role_index], FrWire{}) == 0)
                    return false;
                CommitmentWire commitment;
                if (sorted_id_offset >= context.sorted_lookup_commitment_ids.size())
                    throw std::invalid_argument("sorted-lookup commitment id map is incomplete");
                commitment.tensor_id = context.sorted_lookup_commitment_ids[sorted_id_offset++];
                commitment.visibility = 3U;
                commitment.domain_size = ref.logical_domain;
                SegmentedTensorProvider outer_provider;
                if (!sorted_outer_provider(context.sorted_lookup_providers[lookup_index], ref,
                                           outer_provider))
                    throw std::invalid_argument("sorted-lookup outer provider role is not canonical");
                SegmentCommitmentMaterial material;
                SegmentCommitmentValidationReceipt validation_receipt;
                material.blinders = context.sorted_lookup_outer_segment_blinders[lookup_index][role_index];
                material.commitments = context.sorted_lookup_outer_segment_commitments[lookup_index][role_index];
                if (!zktr_commitment_wire_from_material(
                        commitment.tensor_id, 3U, outer_provider, context.ipa_generators,
                        context.ipa_blinding_generator, context.ipa_u_generator,
                        proof.statement, material, commitment, error,
                        context.sorted_lookup_descriptors[lookup_index].segment_size,
                        nullptr, &context.ipa_generators, &msm_session, &validation_receipt) ||
                    !zkdl::cpu_g1_equal(
                        sorted_outer[lookup_index][role_index],
                        zkdl::cpu_g1_from_native(commitment.commitment)))
                    return false;
                if (!validation_receipt.valid()) {
                    if (error) *error = "sorted-lookup commitment did not produce its MSM validation receipt";
                    return false;
                }
                sorted_receipts[lookup_index].push_back(
                    std::move(validation_receipt));
                sorted_materials[lookup_index].push_back(std::move(material));
                proof.commitments.push_back(std::move(commitment));
            }
        }
        if (sorted_id_offset != context.sorted_lookup_commitment_ids.size())
            throw std::invalid_argument("sorted-lookup commitment id map has extra entries");
        proof.statement.segment_commitments_digest = relation_commitment_vector_digest(
            proof.statement, proof.commitments);
        if (!verify_model_registry(proof.commitments,
                                   context.registered_model_commitments, error)) return false;
        emit_progress("commitment", "PASS", commitment_phase_start,
                      "commitments=" + std::to_string(proof.commitments.size()));
        zkdl::Transcript transcript("zkdl-gat-relation-v1");
        absorb_statement(transcript, proof.statement, proof.commitments);
        absorb_hiding_records(transcript, proof.hiding_commitment_records);
        std::size_t qpad_record_count = 0U;
        for (const auto& descriptor : context.tensor_descriptors)
            if (qpad_record_required(descriptor.logical_shape,
                                     descriptor.padded_shape,
                                     descriptor.visibility == 1U)) ++qpad_record_count;
        const auto qpad_phase_start = std::chrono::steady_clock::now();
        progress.set_phase("qpad_records");
        emit_progress("qpad_records", "START", qpad_phase_start,
                      "total=" + std::to_string(qpad_record_count));
        if (!prove_padding_records(context, tensor_map, proof.commitments,
                                   transcript, proof.padding_records, error,
                                   &segment_blinding_scalars, &msm_session,
                                   &segment_commitment_receipts,
                                   qpad_record_count)) return false;
        emit_progress("qpad_records", "PASS", qpad_phase_start,
                      "records=" + std::to_string(proof.padding_records.size()));
        const auto operation_phase_start = std::chrono::steady_clock::now();
        auto last_operation_progress = operation_phase_start;
        progress.set_phase("operation_proofs");
        emit_progress("operation_proofs", "START", operation_phase_start,
                      "total=" + std::to_string(operation_count));
        proof.operations.reserve(operation_count);
        for (std::size_t operation_index = 0U;
             operation_index < operation_count; ++operation_index) {
            const auto& operation = context.operations[operation_index];
            const std::string operation_id = operation.id;
            progress.set_operation(operation_id);
            const bool has_route = operation.kind == RelationKind::SparseRoute ||
                                   !operation.selector_source_ref.empty();
            const std::uint64_t source_domain = has_route
                ? std::uint64_t(operation.selector_node_domain) * operation.selector_source_feature_domain
                : 0U;
            const std::uint64_t output_domain = has_route
                ? std::uint64_t(operation.selector_edge_domain) * operation.selector_feature_domain
                : 0U;
            emit_progress("operation", "START", operation_phase_start,
                          "index=" + std::to_string(operation_index + 1U) +
                          " total=" + std::to_string(operation_count) +
                          " kind=" + std::to_string(static_cast<unsigned int>(operation.kind)) +
                          " id=" + operation.id +
                          " route=" + std::to_string(has_route) +
                          " source_domain=" + std::to_string(source_domain) +
                          " output_domain=" + std::to_string(output_domain) +
                          " fac_bindings=" + std::to_string(operation.fac_bindings.size()) +
                          " primitives=" + std::to_string(operation.primitives.size()) +
                          " constraints=" + std::to_string(operation.constraints.size()));
            absorb_descriptor(transcript, operation);
            OperationProof operation_proof;
            if (operation.primitives.size() > 8U) {
                throw std::invalid_argument("relation operation carries too many native primitives");
            }
            if (has_route) {
                SparseRouteProof route;
                if (!prove_sparse_route(operation, tensor_map, context.blinding_scalars,
                                        context.ipa_generators,
                                        context.ipa_blinding_generator,
                                        context.ipa_u_generator,
                                        proof.commitments, context.statement,
                                        transcript, route, error, nullptr,
                                        &segment_blinding_scalars,
                                        &context.ipa_generators,
                                        &msm_session)) return false;
                operation_proof.route_proof = route.serialize();
                transcript.absorb("gat.relation.route", operation_proof.route_proof);
            } else if (!operation_proof.route_proof.empty()) {
                throw std::invalid_argument("non-route operation carries a sparse route proof");
            }
            if (!operation.fac_bindings.empty()) {
                FacLine8Proof fac;
                if (!prove_fac_line8(operation, tensor_map, context.blinding_scalars,
                                     context.ipa_generators,
                                     context.ipa_blinding_generator,
                                     context.ipa_u_generator, proof.commitments,
                                     context.statement,
                                     transcript, fac, error,
                                     &segment_blinding_scalars,
                                     &context.ipa_generators,
                                     &msm_session)) return false;
                operation_proof.fac_proof = fac.serialize();
                transcript.absorb("gat.relation.fac", operation_proof.fac_proof);
            } else if (!operation_proof.fac_proof.empty()) {
                throw std::invalid_argument("operation has an unexpected FAC4DNN proof");
            }
            if (!operation.batch_family.empty() &&
                operation.batch_family != "authenticated_sorted_lookup") {
                const ConstraintDescriptor batched = batched_family_descriptor(
                    operation, transcript, error);
                if (batched.id.empty()) return false;
                ConstraintProof payload;
                if (!fill_constraint_openings(batched, operation.tensor_refs, tensor_map,
                                              context.blinding_scalars,
                                              context.ipa_generators,
                                              context.ipa_blinding_generator,
                                              context.ipa_u_generator,
                                              proof.commitments,
                                              context.statement,
                                              transcript, payload, error,
                                              &segment_blinding_scalars,
                                              &context.ipa_generators, &msm_session,
                                              &segment_commitment_receipts,
                                              context.sumcheck_spill_directory.empty()
                                                  ? nullptr : &context.sumcheck_spill_directory)) return false;
                operation_proof.batch_proof = serialize_constraint_payload(payload);
                transcript.absorb("gat.relation.batch", operation_proof.batch_proof);
            } else if (operation.batch_family != "authenticated_sorted_lookup" &&
                       !operation_proof.batch_proof.empty()) {
                throw std::invalid_argument("operation has an unexpected batched relation proof");
            }
            for (const auto& primitive : operation.primitives) {
                if (primitive.kind != 2U) {
                    if (error) *error =
                        "BLOCKED: legacy/value-revealing matrix primitive reached production prover";
                    return false;
                }
                HiddenMatMulProof hidden;
                const std::size_t primitive_index = operation_proof.primitive_proofs.size();
                if (!prove_hidden_matmul(
                        operation, primitive, primitive_index, tensor_map,
                        context.blinding_scalars, context.ipa_generators,
                        context.ipa_blinding_generator, context.ipa_u_generator,
                        proof.commitments, context.statement, transcript, hidden, error,
                        segment_blinding_scalars, &context.ipa_generators,
                        &msm_session, &segment_commitment_receipts)) {
                    if (error) {
                        const std::string detail = *error;
                        *error = "hidden-matmul proof failed phase=operation_proofs operation_id=" +
                            operation.id + " primitive_index=" + std::to_string(primitive_index) +
                            (detail.empty() ? std::string{} : ": " + detail);
                    }
                    return false;
                }
                std::vector<byte> primitive_bytes = hidden.serialize();
                operation_proof.primitive_proofs.push_back(std::move(primitive_bytes));
                transcript.absorb("gat.relation.primitive",
                                  operation_proof.primitive_proofs.back());
            }
            if (!operation.batch_family.empty() &&
                operation.batch_family != "authenticated_sorted_lookup") {
                if (!operation_proof.constraints.empty())
                    throw std::invalid_argument("batched relation carries repeated constraint proofs");
            }
            for (const auto& descriptor : operation.constraints) {
                if (!operation.batch_family.empty() &&
                    operation.batch_family != "authenticated_sorted_lookup") break;
                ConstraintProof constraint;
                if (!fill_constraint_openings(
                        descriptor, operation.tensor_refs, tensor_map,
                        context.blinding_scalars, context.ipa_generators,
                        context.ipa_blinding_generator, context.ipa_u_generator,
                        proof.commitments, context.statement, transcript,
                        constraint, error, &segment_blinding_scalars,
                        &context.ipa_generators, &msm_session,
                        &segment_commitment_receipts,
                        context.sumcheck_spill_directory.empty()
                            ? nullptr : &context.sumcheck_spill_directory)) return false;
                operation_proof.constraints.push_back(std::move(constraint));
            }
            if (consume_owner != nullptr) operation_proof.descriptor = std::move(consume_owner->operations[operation_index]); else operation_proof.descriptor = operation;
            proof.operations.push_back(std::move(operation_proof));
            progress.set_phase("operation_proofs");
            const auto operation_now = std::chrono::steady_clock::now();
            if (operation_now - last_operation_progress >= std::chrono::seconds(15) ||
                operation_index + 1U == operation_count) {
                emit_progress("operation_proofs", "PROGRESS", operation_phase_start,
                              "completed=" + std::to_string(operation_index + 1U) +
                              " total=" + std::to_string(operation_count) +
                              " current_id=" + operation_id);
                last_operation_progress = operation_now;
            }
        }
        emit_progress("operation_proofs", "PASS", operation_phase_start,
                      "completed=" + std::to_string(operation_count));
        const auto lookup_phase_start = std::chrono::steady_clock::now();
        progress.set_phase("sorted_lookup_proofs");
        emit_progress("sorted_lookup_proofs", "START", lookup_phase_start,
                      "total=" + std::to_string(context.sorted_lookup_descriptors.size()));
        for (std::size_t lookup_index = 0U;
             lookup_index < context.sorted_lookup_descriptors.size(); ++lookup_index) {
            progress.phase = "sorted_lookup_proof";
            progress.operation = context.sorted_lookup_descriptors[lookup_index].operation_id;
            emit_progress("sorted_lookup", "START", lookup_phase_start,
                          "index=" + std::to_string(lookup_index + 1U) +
                          " total=" + std::to_string(context.sorted_lookup_descriptors.size()) +
                          " operation_id=" + context.sorted_lookup_descriptors[lookup_index].operation_id);
            zkdl::AuthenticatedSortedLookupProof lookup;
            if (!prove_authenticated_sorted_lookup(
                    context.sorted_lookup_descriptors[lookup_index],
                    context.sorted_lookup_providers[lookup_index],
                    sorted_refs[lookup_index], sorted_outer[lookup_index],
                    sorted_blinders[lookup_index], sorted_materials[lookup_index],
                    context.ipa_generators,
                    context.ipa_blinding_generator, context.ipa_u_generator,
                    context.statement, lookup, error,
                    &context.ipa_generators, &msm_session,
                    &sorted_receipts[lookup_index], &proof.commitments,
                    context.sumcheck_spill_directory.empty()
                        ? nullptr : &context.sumcheck_spill_directory)) {
                if (error) {
                    const std::string detail = *error;
                    *error = "sorted-lookup proof failed phase=sorted_lookup operation_id=" +
                        context.sorted_lookup_descriptors[lookup_index].operation_id +
                        (detail.empty() ? std::string{} : ": " + detail);
                }
                return false;
            }
            const std::vector<byte> encoded = lookup.serialize();
            transcript.absorb("gat.relation.sorted_lookup", encoded);
            proof.sorted_lookup_proofs.push_back(std::move(lookup));
            progress.set_phase("sorted_lookup_proofs");
        }
        emit_progress("sorted_lookup_proofs", "PASS", lookup_phase_start,
                      "completed=" + std::to_string(context.sorted_lookup_descriptors.size()));
        const auto output_phase_start = std::chrono::steady_clock::now();
        progress.set_phase("public_output");
        emit_progress("public_output", "START", output_phase_start);
        const auto public_it = std::find_if(context.tensors.begin(), context.tensors.end(),
                                            [](const TensorWitness& tensor) {
                                                return tensor.id == "public_Y" && tensor.visibility == 1U;
                                            });
        if (public_it == context.tensors.end()) throw std::invalid_argument("public output tensor is missing");
        if (context.statement.public_output.empty() ||
            context.statement.public_output != encode_public_field_vector(public_it->values)) {
            throw std::invalid_argument("public output witness is not the verifier-supplied canonical Y");
        }
        if (!checked_relation_u32(
                static_cast<std::size_t>(
                    std::distance(context.tensors.begin(), public_it)),
                proof.public_output_commitment,
                "public output commitment index", error))
            return false;
        const std::uint32_t output_domain = public_it->domain_size;
        const std::uint32_t output_bits = log2_exact(output_domain);
        std::vector<FrWire> output_point;
        for (std::uint32_t i = 0U; i < output_bits; ++i) output_point.push_back(zkdl::fr_wire(transcript.challenge("gat.public_output.point." + std::to_string(i))));
        SegmentedTensorProvider output_provider;
        SegmentCommitmentMaterial output_material;
        if (!public_it->segmented(context.ipa_generators.size(), output_provider, error) || !zktr_material_from_wire(proof.commitments[proof.public_output_commitment], segment_blinding_scalars, output_material, error)) return false;
        auto output_metadata = relation_opening_metadata(public_output_opening_descriptor(), proof.public_output_commitment, output_point, context.statement);
        output_metadata.evaluation_commitment_index = 0x70000000U;
        HiddenMatMulOpeningSecrets output_opening;
        NativeOperationProgressReporter output_progress("public_output_opening", "public_Y");
        const auto output_receipt = segment_commitment_receipts.find("public_Y");
        if (output_receipt == segment_commitment_receipts.end() ||
            !prove_hidden_segmented_opening(
                output_provider, proof.commitments[proof.public_output_commitment],
                output_material, output_point, context.ipa_generators,
                context.ipa_blinding_generator, context.ipa_u_generator,
                output_metadata, output_opening, error, &output_progress,
                &context.ipa_generators, &output_receipt->second, &msm_session))
            return false;
        proof.public_output_opening = std::move(output_opening.proof);
        auto output_context = zksc_make_context(zkdl::PedersenProofContext::kOperationScalarValue, 1U, 0U, proof.public_output_commitment, zkdl::PedersenProofContext::kUnsetIndex, zkdl::cpu_ipa_basis_digest({}, context.ipa_blinding_generator, context.ipa_u_generator, 0U), output_metadata.descriptor_digest, context.statement);
        proof.public_output_binding = zkdl::cpu_open_pedersen_scalar_value_public(output_opening.value, output_opening.tau_blinder, zkdl::cpu_g1_from_native(proof.public_output_opening.tau), context.ipa_blinding_generator, context.ipa_u_generator, output_context).proof;
        std::vector<byte> public_bytes;
        serialize_hidden_matmul_opening(proof.public_output_opening, public_bytes);
        zkdl::serialize_pedersen_scalar_value(proof.public_output_binding, public_bytes);
        transcript.absorb("gat.public_output.opening", public_bytes);
        proof.transcript_digest = transcript.digest();
        emit_progress("public_output", "PASS", output_phase_start,
                      "proof_operations=" + std::to_string(proof.operations.size()) +
                      " sorted_lookups=" + std::to_string(proof.sorted_lookup_proofs.size()));
        progress.finish(proof.commitments.size(), proof.operations.size(),
                        proof.sorted_lookup_proofs.size());
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }
}

inline bool prove(const ProverContext& context, Proof& proof, std::string* error = nullptr) {
    return prove_impl(context, proof, error, nullptr);
}

inline bool prove_consuming(ProverContext& context, Proof& proof, std::string* error = nullptr) {
    return prove_impl(context, proof, error, &context);
}

inline bool verify(
    const Proof& proof, const VerifierContext& context,
    std::string* error = nullptr,
    std::uint64_t max_terminal_body_bytes =
        zkdl::ZkTerminalRelationProof::kMaxNestedProofBytes) {
    if (proof.version != Proof::kVersion || !same_bytes(proof.statement.statement_digest,
                                                         context.statement.statement_digest) ||
        !same_bytes(proof.statement.model_digest, context.statement.model_digest) ||
        !same_bytes(proof.statement.table_digest, context.statement.table_digest) ||
        !same_bytes(proof.statement.ledger_digest, context.statement.ledger_digest) ||
        !same_bytes(proof.statement.srs_digest, context.statement.srs_digest) ||
        !same_bytes(proof.statement.registry_digest, context.statement.registry_digest) ||
        proof.statement.public_output_digest != zkdl::sha256(proof.statement.public_output) ||
        (context.statement.public_output_digest != std::array<byte, 32>{} &&
         proof.statement.public_output_digest != context.statement.public_output_digest) ||
        proof.statement.public_output != context.statement.public_output) {
        if (error) *error = "relation proof statement/resource identity mismatch";
        return false;
    }
    if (context.ipa_generators.empty() || context.ipa_blinding_generator.is_zero() ||
        context.ipa_u_generator.is_zero() ||
        !same_g1_basis(context.generators, context.ipa_generators)) {
        if (error) *error = "BLOCKED: independent verifier lacks authenticated IPA G/H/U basis";
        return false;
    }
    if (!descriptor_vector_valid(context.operations, error) ||
        !formal_relation_components_supported(context.operations, error)) return false;
    if (context.gpu_backend) {
        if (error) *error =
            "BLOCKED: formal zkDL GPU GAT verification is unavailable until every relation family "
            "and terminal opening runs on native CUDA cryptographic kernels";
        return false;
    }
    if (proof.operations.size() != context.operations.size() || proof.commitments.empty()) return false;
    zkdl::CpuG1MsmSession verifier_msm_session(context.ipa_generators);
    if (!relation_segment_commitments_valid(proof.statement, proof.commitments, error) ||
        (context.statement.segment_binding_version != 0U &&
         context.statement.segment_binding_version != proof.statement.segment_binding_version) ||
        (context.statement.segment_commitments_digest != std::array<byte, 32U>{} &&
         context.statement.segment_commitments_digest != proof.statement.segment_commitments_digest))
        return false;
    if (!verify_tensor_contract(context, proof.commitments, error) ||
        !verify_model_registry(proof.commitments,
                               context.registered_model_commitments, error) ||
        !verify_public_tensor_values(context, proof.commitments, error)) return false;
    if (!proof.hiding_commitment_records.empty() ||
        !context.hiding_commitment_records.empty()) {
        if (error) *error = "legacy hiding commitment records are rejected by formal GATR";
        return false;
    }
    std::set<std::string> commitment_ids;
    for (const auto& commitment : proof.commitments) {
        ZkRelationSegmentLayout commitment_layout;
        if (commitment.tensor_id.empty() || !commitment_ids.insert(commitment.tensor_id).second ||
            commitment.domain_size == 0U || log2_exact(commitment.domain_size) == std::numeric_limits<std::uint32_t>::max() ||
            !zktr_segment_layout(commitment.domain_size, context.ipa_generators.size(),
                                 commitment_layout, error)) {
            if (error) *error = "relation proof commitment section is invalid";
            return false;
        }
        const auto visibility = context.tensor_visibility.find(commitment.tensor_id);
        if (visibility != context.tensor_visibility.end() &&
            commitment.visibility != visibility->second) {
            if (error) *error = "relation proof commitment visibility does not match public context";
            return false;
        }
    }
    std::set<std::string> expected_commitment_ids;
    if (!context.tensor_ids.empty()) {
        expected_commitment_ids.insert(context.tensor_ids.begin(), context.tensor_ids.end());
    } else {
        for (const auto& operation : context.operations)
            expected_commitment_ids.insert(operation.tensor_refs.begin(), operation.tensor_refs.end());
        expected_commitment_ids.insert("public_Y");
    }
    expected_commitment_ids.insert(context.sorted_lookup_commitment_ids.begin(),
                                  context.sorted_lookup_commitment_ids.end());
    if (commitment_ids != expected_commitment_ids) {
        if (error) *error = "relation proof commitment section contains an unplanned or missing tensor";
        return false;
    }
    zkdl::Transcript transcript("zkdl-gat-relation-v1");
    absorb_statement(transcript, proof.statement, proof.commitments);
    absorb_hiding_records(transcript, proof.hiding_commitment_records);
    if (!verify_padding_records(context, proof, transcript, error,
                                &verifier_msm_session)) return false;
    for (size_t op_index = 0U; op_index < context.operations.size(); ++op_index) {
        const auto& expected_operation = context.operations[op_index];
        const auto& operation = proof.operations[op_index];
        bool descriptor_bytes_match = false;
        {
            std::vector<byte> expected_bytes, actual_bytes;
            append_operation(expected_bytes, expected_operation);
            append_operation(actual_bytes, operation.descriptor);
            descriptor_bytes_match = expected_bytes == actual_bytes;
        }
        if (!descriptor_bytes_match ||
            (expected_operation.batch_family != "" &&
             expected_operation.batch_family != "authenticated_sorted_lookup"
                 ? !operation.constraints.empty()
                 : operation.constraints.size() != expected_operation.constraints.size())) {
            if (error) *error = "relation operation descriptor/count mismatch";
            return false;
        }
        absorb_descriptor(transcript, expected_operation);
        if (operation.primitive_proofs.size() != expected_operation.primitives.size()) {
            if (error) *error = "relation primitive proof count mismatch";
            return false;
        }
        if (!operation.primitive_bindings.empty()) {
            if (error) *error = "legacy/value-revealing primitive bindings are not part of formal proof";
            return false;
        }
        const bool has_route = expected_operation.kind == RelationKind::SparseRoute ||
                               !expected_operation.selector_source_ref.empty();
        if (has_route) {
            if (operation.route_proof.empty()) {
                if (error) *error = "sparse route operation is missing its native relation proof";
                return false;
            }
            SparseRouteProof route;
            if (!SparseRouteProof::deserialize(operation.route_proof, route, error) ||
                !verify_sparse_route(expected_operation, proof.commitments,
                                     context.ipa_generators,
                                     context.ipa_blinding_generator,
                                     context.ipa_u_generator, context.statement,
                                     transcript, route, error,
                                     &context.ipa_generators,
                                     &verifier_msm_session)) return false;
            transcript.absorb("gat.relation.route", operation.route_proof);
        } else if (!operation.route_proof.empty()) {
            if (error) *error = "non-route operation carries a sparse route proof";
            return false;
        }
        if (!expected_operation.fac_bindings.empty()) {
            if (operation.fac_proof.empty()) {
                if (error) *error = "FAC4DNN operation is missing its native view-binding proof";
                return false;
            }
            FacLine8Proof fac;
            if (!FacLine8Proof::deserialize(operation.fac_proof, fac, error) ||
                !verify_fac_line8(expected_operation, proof.commitments,
                                  context.ipa_generators,
                                  context.ipa_blinding_generator,
                                  context.ipa_u_generator, context.statement,
                                  transcript, fac, error,
                                  &context.ipa_generators,
                                  &verifier_msm_session)) return false;
            transcript.absorb("gat.relation.fac", operation.fac_proof);
        } else if (!operation.fac_proof.empty()) {
            if (error) *error = "operation carries an unexpected FAC4DNN proof";
            return false;
        }
        if (!expected_operation.batch_family.empty() &&
            expected_operation.batch_family != "authenticated_sorted_lookup") {
            if (operation.batch_proof.empty()) {
                if (error) *error = "batched relation operation is missing its family proof";
                return false;
            }
            ConstraintProof payload;
            if (!deserialize_constraint_payload(operation.batch_proof, payload, error,
                                                max_terminal_body_bytes)) return false;
            const ConstraintDescriptor batched = batched_family_descriptor(
                expected_operation, transcript, error);
            if (batched.id.empty() ||
                !verify_constraint(payload, batched, expected_operation.tensor_refs,
                                   proof.commitments, context.ipa_generators,
                                   context.ipa_blinding_generator,
                                   context.ipa_u_generator, context.statement,
                                   transcript, error,
                                   &context.ipa_generators,
                                   &verifier_msm_session)) return false;
            transcript.absorb("gat.relation.batch", operation.batch_proof);
        } else if (expected_operation.batch_family != "authenticated_sorted_lookup" &&
                   !operation.batch_proof.empty()) {
            if (error) *error = "operation carries an unexpected batched family proof";
            return false;
        }
        for (size_t primitive_index = 0U;
             primitive_index < expected_operation.primitives.size(); ++primitive_index) {
            if (expected_operation.primitives[primitive_index].kind != 2U) {
                if (error) *error =
                    "BLOCKED: legacy/value-revealing matrix primitive reached production verifier";
                return false;
            }
            HiddenMatMulProof hidden;
            if (!HiddenMatMulProof::deserialize(
                    operation.primitive_proofs[primitive_index], hidden, error) ||
                !verify_hidden_matmul(
                    expected_operation, expected_operation.primitives[primitive_index],
                    primitive_index, hidden, proof.commitments,
                    context.ipa_generators, context.ipa_blinding_generator,
                    context.ipa_u_generator, context.statement, transcript, error,
                    &context.ipa_generators, &verifier_msm_session))
                return false;
            transcript.absorb("gat.relation.primitive",
                              operation.primitive_proofs[primitive_index]);
        }
        if (expected_operation.batch_family.empty()) {
            for (size_t constraint_index = 0U;
                 constraint_index < expected_operation.constraints.size(); ++constraint_index)
                if (!verify_constraint(operation.constraints[constraint_index],
                                       expected_operation.constraints[constraint_index],
                                       expected_operation.tensor_refs,
                                       proof.commitments, context.ipa_generators,
                                       context.ipa_blinding_generator,
                                       context.ipa_u_generator,
                                       context.statement,
                                       transcript, error,
                                       &context.ipa_generators,
                                       &verifier_msm_session)) return false;
        }
    }
    if (proof.sorted_lookup_proofs.size() != context.sorted_lookup_descriptors.size() ||
        context.sorted_lookup_descriptors.size() != context.sorted_lookup_table_keys.size() ||
        context.sorted_lookup_descriptors.size() != context.sorted_lookup_table_active.size()) {
        if (error) *error = "sorted-lookup proof count/context descriptor mismatch";
        return false;
    }
    std::size_t sorted_id_offset = 0U;
    for (std::size_t lookup_index = 0U;
         lookup_index < context.sorted_lookup_descriptors.size(); ++lookup_index) {
        const auto& lookup = proof.sorted_lookup_proofs[lookup_index];
        std::vector<zkdl::CpuG1Point> outer_commitments;
        std::vector<CommitmentWire> outer_wires;
        outer_commitments.reserve(lookup.commitments.size());
        outer_wires.reserve(lookup.commitments.size());
        for (const auto& ref : lookup.commitments) {
            if (ref.commitment_index >= proof.commitments.size() ||
                proof.commitments[ref.commitment_index].tensor_id.rfind("sorted_lookup/", 0U) != 0U ||
                sorted_id_offset >= context.sorted_lookup_commitment_ids.size() ||
                proof.commitments[ref.commitment_index].tensor_id !=
                    context.sorted_lookup_commitment_ids[sorted_id_offset++])
                return false;
            outer_commitments.push_back(zkdl::cpu_g1_from_native(
                proof.commitments[ref.commitment_index].commitment));
            outer_wires.push_back(proof.commitments[ref.commitment_index]);
        }
        std::vector<byte> encoded;
        if (!verify_authenticated_sorted_lookup(
                context.sorted_lookup_descriptors[lookup_index],
                context.sorted_lookup_table_keys[lookup_index],
                context.sorted_lookup_table_active[lookup_index],
                lookup.commitments, outer_commitments, outer_wires,
                context.ipa_generators,
                context.ipa_blinding_generator, context.ipa_u_generator,
                context.statement, lookup, error,
                &context.ipa_generators,
                &verifier_msm_session)) return false;
        encoded = lookup.serialize();
        transcript.absorb("gat.relation.sorted_lookup", encoded);
    }
    if (sorted_id_offset != context.sorted_lookup_commitment_ids.size()) {
        if (error) *error = "sorted-lookup commitment id map is incomplete";
        return false;
    }
    if (proof.public_output_commitment >= proof.commitments.size() || proof.commitments[proof.public_output_commitment].tensor_id != "public_Y" || proof.commitments[proof.public_output_commitment].visibility != 1U) { if (error) *error = "public output segmented commitment is missing"; return false; }
    const auto& output_commitment = proof.commitments[proof.public_output_commitment];
    const std::uint32_t domain = output_commitment.domain_size, bits = log2_exact(domain);
    std::vector<FrWire> point;
    for (std::uint32_t i = 0U; i < bits; ++i) point.push_back(zkdl::fr_wire(transcript.challenge("gat.public_output.point." + std::to_string(i))));
    std::vector<FrWire> public_values;
    if (!decode_public_field_vector(context.statement.public_output, public_values) || public_values.size() != domain) { if (error) *error = "public Y is not a canonical field vector matching commitment domain"; return false; }
    SegmentedTensorProvider provider;
    if (!SegmentedTensorProvider::from_values(public_values, context.ipa_generators.size(), provider, error)) return false;
    VerifierContext output_public;
    output_public.statement = context.statement;
    output_public.ipa_generators = context.ipa_generators;
    output_public.ipa_blinding_generator = context.ipa_blinding_generator;
    output_public.ipa_u_generator = context.ipa_u_generator;
    output_public.public_tensor_providers.emplace("public_Y", provider);
    if (!verify_public_tensor_values(output_public, proof.commitments, error)) return false;
    auto metadata = relation_opening_metadata(public_output_opening_descriptor(), proof.public_output_commitment, point, context.statement);
    metadata.evaluation_commitment_index = 0x70000000U;
    if (!verify_hidden_segmented_opening(
            proof.public_output_opening, output_commitment, point,
            context.ipa_generators, context.ipa_blinding_generator,
            context.ipa_u_generator, metadata, error,
            &context.ipa_generators, &verifier_msm_session)) return false;
    FrWire expected_public_y{};
    if (!provider.evaluate_mle(point, expected_public_y, error)) return false;
    const auto output_context = zksc_make_context(zkdl::PedersenProofContext::kOperationScalarValue, 1U, 0U, proof.public_output_commitment, zkdl::PedersenProofContext::kUnsetIndex, zkdl::cpu_ipa_basis_digest({}, context.ipa_blinding_generator, context.ipa_u_generator, 0U), metadata.descriptor_digest, context.statement);
    if (!zkdl::cpu_verify_pedersen_scalar_value(proof.public_output_binding, expected_public_y, zkdl::cpu_g1_from_native(proof.public_output_opening.tau), context.ipa_blinding_generator, context.ipa_u_generator, output_context, error)) return false;
    std::vector<byte> opening_bytes;
    serialize_hidden_matmul_opening(proof.public_output_opening, opening_bytes);
    zkdl::serialize_pedersen_scalar_value(proof.public_output_binding, opening_bytes);
    transcript.absorb("gat.public_output.opening", opening_bytes);
    if (transcript.digest() != proof.transcript_digest) { if (error) *error = "GAT relation transcript digest mismatch"; return false; }
    return true;
}

/*
 * Dedicated authenticated sorted lookup primitive.  This intentionally does
 * not route through ConstraintProof: Q and S are separate committed domains
 * and only their committed terminal product is linked to the public table
 * product.  The two entry points are parameter-oriented so the adapter can
 * pass its existing outer commitment handles without another wire type.
 */
inline bool prove_authenticated_sorted_lookup(
    const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
    AuthenticatedSortedLookupProviderSet providers,
    const std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>& commitment_refs,
    const std::vector<zkdl::CpuG1Point>& outer_commitments,
    const std::vector<FrWire>& outer_blinders,
    const std::vector<SegmentCommitmentMaterial>& outer_materials,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    zkdl::AuthenticatedSortedLookupProof& proof,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    zkdl::CpuG1MsmSession* msm_session,
    const std::vector<SegmentCommitmentValidationReceipt>* outer_receipts,
    const std::vector<CommitmentWire>* all_commitments,
    const std::string* spill_directory) {
    try {
        NativeOperationProgressReporter lookup_progress("sorted_lookup_proof", descriptor.operation_id);
        if (!descriptor.valid(error) || descriptor.table_digest != statement.table_digest ||
            !zktr_all_statement_digests_valid(statement, error) ||
            !providers.valid(descriptor, error) || commitment_refs.empty() ||
            commitment_refs.size() != outer_commitments.size() ||
            commitment_refs.size() != outer_blinders.size() ||
            commitment_refs.size() != outer_materials.size() || generators.empty() ||
            (outer_receipts != nullptr &&
             outer_receipts->size() != commitment_refs.size()) ||
            (outer_receipts != nullptr && all_commitments == nullptr) ||
            !zkdl::cpu_pedersen_strict_g1(blinding_generator, error) ||
            !zkdl::cpu_pedersen_strict_g1(u_generator, error)) {
            if (error && error->empty()) *error = "invalid authenticated sorted-lookup prover inputs";
            return false;
        }
        if (authenticated_basis != nullptr &&
            !zktr_basis_valid_for_prefix(
                generators, blinding_generator, u_generator,
                static_cast<std::uint32_t>(generators.size()),
                authenticated_basis, error)) return false;
        const auto expected_domain = [&](std::uint32_t role) {
            return role == zkdl::kSortedLookupCommitmentQueryKey ||
                           role == zkdl::kSortedLookupCommitmentQueryBit ||
                           role == zkdl::kSortedLookupCommitmentQueryPrefix ||
                           role == zkdl::kSortedLookupCommitmentQueryX ||
                           role == zkdl::kSortedLookupCommitmentQueryY
                       ? descriptor.query_domain
                       : descriptor.merge_domain;
        };
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> outer_by_role;
        for (std::size_t index = 0U; index < commitment_refs.size(); ++index) {
            const auto& ref = commitment_refs[index];
            if (!ref.valid(error) || ref.logical_domain != expected_domain(ref.role) ||
                ref.segment_size != descriptor.segment_size ||
                ref.segment_count != expected_domain(ref.role) / descriptor.segment_size ||
                !outer_by_role.emplace(std::make_pair(ref.role, ref.column_index), index).second ||
                !zkdl::fr_canonical(outer_blinders[index]) ||
                zkdl::fr_compare(outer_blinders[index], FrWire{}) == 0 ||
                !zkdl::cpu_pedersen_strict_g1(outer_commitments[index], error)) {
                if (error && error->empty()) *error = "invalid sorted-lookup outer commitment handle";
                return false;
            }
        }
        const auto role_for_tag = [&](const std::string& tag) {
            if (tag == "query_x") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryX), 0U);
            if (tag == "query_y") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryY), 0U);
            if (tag == "query_key") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryKey), 0U);
            if (tag == "query_prefix") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryPrefix), 0U);
            if (tag == "merge_key") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeKey), 0U);
            if (tag == "merge_delta") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeDelta), 0U);
            if (tag == "merge_marker") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeMarker), 0U);
            if (tag == "merge_inverse") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeInverse), 0U);
            if (tag == "merge_prefix") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergePrefix), 0U);
            const std::string merge_bit = "merge_bit_";
            if (tag.compare(0U, merge_bit.size(), merge_bit) == 0) {
                std::uint32_t bit = std::numeric_limits<std::uint32_t>::max();
                (void)parse_relation_u32_suffix(tag.substr(merge_bit.size()), bit);
                return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeBit),
                                      bit);
            }
            return std::make_pair(static_cast<std::uint32_t>(0U), 0U);
        };
        SegmentedTensorProvider query_product_prefix, merge_product_prefix;
        const auto provider_for_tag = [&](const std::string& tag,
                                          SegmentedTensorProvider& provider) {
            if (tag == "query_x") provider = providers.query_x;
            else if (tag == "query_y") provider = providers.query_y;
            else if (tag == "query_key") provider = providers.query_keys;
            else if (tag == "query_active") provider = providers.query_active;
            else if (tag == "query_prefix") provider = providers.query_prefix;
            else if (tag == "query_product_prefix") provider = query_product_prefix;
            else if (tag == "query_prefix_next") provider = providers.query_prefix_next;
            else if (tag == "query_first") provider = providers.query_first;
            else if (tag == "query_last") provider = providers.query_last;
            else if (tag == "merge_key") provider = providers.merge_keys;
            else if (tag == "merge_active") provider = providers.merge_active;
            else if (tag == "merge_delta") provider = providers.merge_delta;
            else if (tag == "merge_marker") provider = providers.merge_marker;
            else if (tag == "merge_inverse") provider = providers.merge_inverse;
            else if (tag == "merge_prefix") provider = providers.merge_prefix;
            else if (tag == "merge_product_prefix") provider = merge_product_prefix;
            else if (tag == "merge_prefix_next") provider = providers.merge_prefix_next;
            else if (tag == "merge_previous_key") provider = providers.merge_previous_key;
            else if (tag == "merge_nonzero") provider = providers.merge_nonzero;
            else if (tag == "merge_first") provider = providers.merge_first;
            else if (tag == "merge_last") provider = providers.merge_last;
            else {
                const std::string key = "merge_bit_";
                const std::string delta = "merge_delta_bit_";
                if (tag.compare(0U, key.size(), key) == 0) {
                    std::uint32_t bit = 0U;
                    if (!parse_relation_u32_suffix(tag.substr(key.size()), bit))
                        return false;
                    if (bit >= providers.merge_bits.size()) return false;
                    provider = providers.merge_bits[bit];
                } else if (tag.compare(0U, delta.size(), delta) == 0) {
                    std::uint32_t bit = 0U;
                    if (!parse_relation_u32_suffix(tag.substr(delta.size()), bit))
                        return false;
                    if (bit >= providers.merge_delta_bits.size()) return false;
                    provider = providers.merge_delta_bits[bit];
                } else return false;
            }
            return provider.valid(error);
        };

        const unsigned __int128 query_base = static_cast<unsigned __int128>(1U) << descriptor.y_bits;
        for (std::uint32_t segment = 0U; segment < descriptor.query_segment_count; ++segment) {
            std::vector<FrWire> query_key_block, query_x_block, query_y_block;
            if (!providers.query_keys.read_segment(segment, query_key_block, error) ||
                !providers.query_x.read_segment(segment, query_x_block, error) ||
                !providers.query_y.read_segment(segment, query_y_block, error)) return false;
            for (std::size_t index = 0U; index < query_key_block.size(); ++index) {
                const std::uint64_t global = static_cast<std::uint64_t>(segment) * descriptor.segment_size + index;
                if (global >= descriptor.query_active_length) break;
                unsigned __int128 x = 0U, y = 0U, key = 0U;
                if (!sorted_lookup_u128_from_field(query_x_block[index], descriptor.x_bits, x, error) ||
                    !sorted_lookup_u128_from_field(query_y_block[index], descriptor.y_bits, y, error) ||
                    !sorted_lookup_u128_from_field(query_key_block[index], descriptor.x_bits + descriptor.y_bits, key, error) ||
                    key != x * query_base + y) {
                    if (error && error->empty()) *error = "sorted-lookup query key encoding mismatch";
                    return false;
                }
            }
            lookup_progress.update("query_validation_segments", segment + 1U,
                                   descriptor.query_segment_count);
        }

        zkdl::Transcript transcript("zkdl-authenticated-sorted-lookup-v1");
        std::vector<byte> header;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, header);
        zkdl::append_u64(header, static_cast<std::uint64_t>(commitment_refs.size()));
        for (std::size_t index = 0U; index < commitment_refs.size(); ++index) {
            const auto& ref = commitment_refs[index];
            zkdl::append_u32(header, ref.role);
            zkdl::append_u32(header, ref.column_index);
            zkdl::append_u32(header, ref.commitment_index);
            zkdl::append_u32(header, ref.logical_domain);
            zkdl::append_u32(header, ref.segment_size);
            zkdl::append_u32(header, ref.segment_count);
            zkdl::append_g1(header, zkdl::cpu_g1_to_native(outer_commitments[index]));
        }
        transcript.absorb("gat.sorted_lookup.header", header);
        const FrWire beta = zkdl::cpu_ipa_nonzero_challenge(
            transcript, "gat.sorted_lookup.beta");
        const FrWire gamma = zkdl::cpu_ipa_nonzero_challenge(
            transcript, "gat.sorted_lookup.gamma");
        std::vector<byte> prefix_identity;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, prefix_identity);
        append_digest(prefix_identity, statement.statement_digest);
        append_digest(prefix_identity, statement.model_digest);
        append_digest(prefix_identity, statement.table_digest);
        append_digest(prefix_identity, statement.ledger_digest);
        append_digest(prefix_identity, statement.registry_digest);
        append_digest(prefix_identity, statement.srs_digest);
        append_digest(prefix_identity, statement.public_output_digest);
        const std::array<byte, 32U> prefix_base_digest = zkdl::sha256(prefix_identity);
        std::vector<byte> query_prefix_identity = prefix_identity;
        zkdl::append_string(query_prefix_identity, "query-product-prefix");
        std::vector<byte> merge_prefix_identity = prefix_identity;
        zkdl::append_string(merge_prefix_identity, "merge-product-prefix");
        const std::array<byte, 32U> query_prefix_digest = zkdl::sha256(query_prefix_identity);
        const std::array<byte, 32U> merge_prefix_digest = zkdl::sha256(merge_prefix_identity);
        const std::uint32_t key_bits = descriptor.x_bits + descriptor.y_bits;
        const SegmentCommitmentProgressCallback segment_commitment_progress =
            [&lookup_progress](std::uint32_t completed, std::uint32_t total) {
                lookup_progress.update("relation_commitment_segments", completed, total);
            };
        if (!make_authenticated_sorted_lookup_prefix_product_provider(
                providers.query_keys, providers.query_active, beta, gamma,
                query_product_prefix, error, &lookup_progress,
                query_prefix_digest, key_bits) ||
            !make_authenticated_sorted_lookup_prefix_next_provider(
                query_product_prefix, providers.query_keys, providers.query_active,
                descriptor.query_active_length, beta, gamma,
                providers.query_prefix_next, error) ||
            !make_authenticated_sorted_lookup_prefix_product_provider(
                providers.merge_keys, providers.merge_active, beta, gamma,
                merge_product_prefix, error, &lookup_progress,
                merge_prefix_digest, key_bits) ||
            !make_authenticated_sorted_lookup_prefix_next_provider(
                merge_product_prefix, providers.merge_keys, providers.merge_active,
                descriptor.merge_active_length, beta, gamma,
                providers.merge_prefix_next, error) ||
            !make_authenticated_sorted_lookup_shifted_key_provider(
                providers.merge_keys, providers.merge_active,
                providers.merge_previous_key, error) ||
            !make_authenticated_sorted_lookup_nonzero_provider(
                providers.merge_delta, providers.merge_nonzero, error)) {
            return false;
        }
        providers.query_bits.clear();
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit) {
            SegmentedTensorProvider derived;
            if (!make_authenticated_sorted_lookup_bit_provider(
                    providers.query_keys, descriptor.x_bits + descriptor.y_bits, bit,
                    derived, error)) return false;
            providers.query_bits.push_back(std::move(derived));
        }
        providers.merge_bits.clear();
        providers.merge_delta_bits.clear();
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit) {
            SegmentedTensorProvider key_bit, delta_bit;
            if (!make_authenticated_sorted_lookup_bit_provider(
                    providers.merge_keys, descriptor.x_bits + descriptor.y_bits, bit,
                    key_bit, error) ||
                !make_authenticated_sorted_lookup_bit_provider(
                    providers.merge_delta, descriptor.x_bits + descriptor.y_bits, bit,
                    delta_bit, error)) return false;
            providers.merge_bits.push_back(std::move(key_bit));
            providers.merge_delta_bits.push_back(std::move(delta_bit));
        }
        if (!providers.valid_after_derived_providers(descriptor, error)) return false;
        FrWire table_product{};
        if (!authenticated_sorted_lookup_public_table_product(
                providers.table_keys, providers.table_active, descriptor.table_active_length,
                beta, gamma, table_product, error, &lookup_progress)) {
            if (error && !error->empty())
                *error = "sorted-lookup phase=public_table_product: " + *error;
            else if (error)
                *error = "sorted-lookup phase=public_table_product failed";
            return false;
        }

        SegmentedTensorProvider query_endpoint_provider, merge_endpoint_provider;
        FrWire query_endpoint{}, merge_endpoint{};
        const bool lookup_endpoints_ready =
            authenticated_sorted_lookup_prefix_product_endpoint(
                providers.query_keys, providers.query_active, descriptor.query_active_length,
                beta, gamma, query_endpoint, error, &lookup_progress,
                query_prefix_digest, &query_product_prefix) &&
            authenticated_sorted_lookup_prefix_product_endpoint(
                providers.merge_keys, providers.merge_active, descriptor.merge_active_length,
                beta, gamma, merge_endpoint, error, &lookup_progress,
                merge_prefix_digest, &merge_product_prefix) &&
            make_authenticated_sorted_lookup_constant_provider(
                descriptor.query_domain, descriptor.segment_size, query_endpoint,
                query_endpoint_provider, error) &&
            make_authenticated_sorted_lookup_constant_provider(
                descriptor.merge_domain, descriptor.segment_size, merge_endpoint,
                merge_endpoint_provider, error);
        if (!lookup_endpoints_ready) {
            if (error && !error->empty())
                *error = "sorted-lookup phase=prefix_endpoint: " + *error;
            else if (error)
                *error = "sorted-lookup phase=prefix_endpoint failed";
            return false;
        }

        struct State {
            zkdl::CpuG1Point commitment;
            FrWire blinder{};
            std::uint32_t outer_index = std::numeric_limits<std::uint32_t>::max();
            SegmentCommitmentMaterial material;
            const SegmentCommitmentValidationReceipt* receipt = nullptr;
        };
        std::map<std::string, State> states;
        const auto validate_outer_commitment =
            [&](std::size_t index, const SegmentedTensorProvider& provider,
                const SegmentCommitmentValidationReceipt*& receipt) {
                receipt = nullptr;
                if (index >= commitment_refs.size() ||
                    index >= outer_materials.size()) {
                    if (error) *error = "sorted-lookup outer commitment role index is out of bounds";
                    return false;
                }
                if (outer_receipts == nullptr || all_commitments == nullptr) {
                    CommitmentWire checked;
                    return zktr_commitment_wire_from_material(
                               "sorted_lookup/material-check", 3U, provider,
                               generators, blinding_generator, u_generator,
                               statement, outer_materials[index], checked, error,
                               descriptor.segment_size,
                               &segment_commitment_progress, authenticated_basis,
                               msm_session) &&
                           zkdl::cpu_g1_equal(
                               zkdl::cpu_g1_from_native(checked.commitment),
                               outer_commitments[index]);
                }
                const auto& ref = commitment_refs[index];
                if (ref.commitment_index >= all_commitments->size()) {
                    if (error) *error = "sorted-lookup outer commitment index is out of bounds";
                    return false;
                }
                const CommitmentWire& checked =
                    (*all_commitments)[ref.commitment_index];
                std::uint32_t generator_count = 0U;
                if (!checked_relation_u32(
                        generators.size(), generator_count,
                        "sorted-lookup authenticated generator count", error) ||
                    !segment_commitment_wire_shape_valid(checked, error) ||
                    checked.visibility != 3U ||
                    checked.domain_size != ref.logical_domain ||
                    checked.segment_size != ref.segment_size ||
                    checked.segment_count != ref.segment_count ||
                    checked.srs_manifest_digest != statement.srs_digest ||
                    checked.basis_digest != zktr_basis_digest_for_prefix(
                        generators, blinding_generator, u_generator,
                        generator_count,
                        zkdl::cpu_ipa_padded_size(generator_count),
                        authenticated_basis) ||
                    !zkdl::cpu_g1_equal(
                        zkdl::cpu_g1_from_native(checked.commitment),
                        outer_commitments[index]) ||
                    !(*outer_receipts)[index].matches_context(
                        checked.tensor_id, checked.visibility, provider,
                        generators, blinding_generator, u_generator,
                        authenticated_basis, checked, outer_materials[index],
                        error))
                    return false;
                receipt = &(*outer_receipts)[index];
                return true;
            };
        auto state_for = [&](const std::string& tag,
                                   const SegmentedTensorProvider& provider,
                                   State& state) mutable {
            const auto existing = states.find(tag);
            if (existing != states.end()) {
                state = existing->second;
                return true;
            }
            const auto role = role_for_tag(tag);
            if (role.first != 0U) {
                const auto outer = outer_by_role.find(role);
                if (outer == outer_by_role.end()) return false;
                const std::size_t index = outer->second;
                const SegmentCommitmentValidationReceipt* receipt = nullptr;
                if (!validate_outer_commitment(index, provider, receipt)) return false;
                state = State{outer_commitments[index], outer_blinders[index],
                              commitment_refs[index].commitment_index,
                              outer_materials[index], receipt};
            } else {
                FrWire blinder = zkdl::cpu_fresh_csprng_scalar();
                while (blinder == FrWire{}) blinder = zkdl::cpu_fresh_csprng_scalar();
                CommitmentWire internal_wire;
                SegmentCommitmentMaterial internal_material;
                if (!zktr_build_segment_commitment_material(
                        "sorted_lookup/internal/" + tag, 3U, provider, generators,
                        blinding_generator, u_generator, blinder, statement,
                        internal_wire, internal_material, error, nullptr,
                        descriptor.segment_size, &segment_commitment_progress,
                        authenticated_basis, msm_session)) return false;
                const auto commitment = zkdl::cpu_g1_from_native(internal_wire.commitment);
                state = State{commitment, blinder,
                              std::numeric_limits<std::uint32_t>::max(),
                              std::move(internal_material), nullptr};
            }
            states.emplace(tag, state);
            return true;
        };

        const auto make_piece = [](const std::string& id, std::uint32_t degree,
                                   const FrWire& constant,
                                   const std::vector<Monomial>& terms) {
            ConstraintDescriptor piece;
            piece.id = id;
            piece.degree = degree;
            piece.constant = constant;
            piece.terms = terms;
            return piece;
        };
        const auto weighted_descriptor = [&](std::uint32_t kind,
                                             const std::vector<std::string>& tags,
                                             const std::vector<ConstraintDescriptor>& pieces,
                                             std::vector<FrWire>* piece_rhos) {
            std::vector<byte> plan;
            zkdl::append_u32(plan, kind);
            zkdl::append_u64(plan, static_cast<std::uint64_t>(tags.size()));
            for (const auto& tag : tags) zkdl::append_string(plan, tag);
            zkdl::append_u64(plan, static_cast<std::uint64_t>(pieces.size()));
            for (const auto& piece : pieces) append_constraint(plan, piece);
            const std::string context = "gat.sorted_lookup.relation." + std::to_string(kind);
            transcript.absorb(context + ".plan", plan);
            ConstraintDescriptor result;
            result.id = "authenticated_sorted_lookup/relation/" + std::to_string(kind);
            result.degree = 1U;
            if (piece_rhos != nullptr) { piece_rhos->clear(); piece_rhos->reserve(pieces.size()); }
            for (const auto& piece : pieces) result.degree = std::max(result.degree, piece.degree);
            for (std::size_t index = 0U; index < pieces.size(); ++index) {
                const FrWire rho = zkdl::cpu_ipa_nonzero_challenge(
                    transcript, context + ".rho." + std::to_string(index));
                if (piece_rhos != nullptr) piece_rhos->push_back(rho);
                result.constant = zkdl::fr_add(result.constant,
                                               zkdl::fr_mul(rho, pieces[index].constant));
                for (const auto& term : pieces[index].terms) {
                    Monomial weighted = term;
                    weighted.coefficient = zkdl::fr_mul(rho, term.coefficient);
                    result.terms.push_back(std::move(weighted));
                }
            }
            return result;
        };

        std::map<std::uint32_t, ConstraintDescriptor> relation_descriptors;
        std::map<std::string, FrWire> terminal_values, terminal_blinders;
        std::map<std::string, zkdl::CpuG1Point> terminal_commitments;
        proof = zkdl::AuthenticatedSortedLookupProof{};
        proof.descriptor = descriptor;
        proof.commitments = commitment_refs;
        auto run_relation = [&](std::uint32_t kind,
                                      const std::vector<std::string>& tags,
                                      const std::vector<SegmentedTensorProvider>& relation_providers,
                                      const std::vector<ConstraintDescriptor>& pieces) mutable {
            std::string failure_phase = "layout";
            const auto run_relation_body = [&]() -> bool {
            if (tags.empty() || tags.size() != relation_providers.size()) return false;
            ZkRelationSegmentLayout relation_layout;
            if (!zktr_segment_layout_for_size(
                    relation_providers.front().logical_domain(), generators.size(),
                    descriptor.segment_size, relation_layout, error)) return false;
            for (const auto& provider : relation_providers)
                if (!provider.valid(error) ||
                    provider.logical_domain() != relation_layout.logical_domain ||
                    provider.segment_size() != relation_layout.segment_size ||
                    provider.segment_count() != relation_layout.segment_count) {
                    if (error) *error = "sorted-lookup relation provider differs from authenticated shared layout";
                    return false;
                }
            std::vector<zkdl::CpuG1Point> canonical;
            std::vector<FrWire> blinders;
            std::vector<std::uint32_t> indices, outer_indices;
            std::vector<SegmentCommitmentMaterial> terminal_materials;
            std::vector<const SegmentCommitmentValidationReceipt*> terminal_receipts;
            terminal_materials.reserve(tags.size());
            terminal_receipts.reserve(tags.size());
            failure_phase = "state_commitment";
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                lookup_progress.update("relation_provider_states", index + 1U,
                                       tags.size(), kind, 5U);
                State state;
                if (!state_for(tags[index], relation_providers[index], state)) return false;
                canonical.push_back(state.commitment);
                blinders.push_back(state.blinder);
                std::uint32_t relation_index = 0U;
                if (!checked_relation_u32(index, relation_index,
                                          "sorted-lookup relation commitment index", error))
                    return false;
                indices.push_back(relation_index);
                outer_indices.push_back(state.outer_index);
                terminal_materials.push_back(state.material);
                terminal_receipts.push_back(state.receipt);
            }
            const std::string context = "gat.sorted_lookup.relation." + std::to_string(kind);
            std::vector<std::vector<zkdl::CpuG1Point>> provider_segments;
            for (const auto& material : terminal_materials) provider_segments.push_back(material.commitments);
            failure_phase = "commitment_absorb";
            pointwise_absorb_provider_commitments(
                context, relation_providers.front().logical_domain(), tags, canonical,
                provider_segments, transcript, &lookup_progress);
            failure_phase = "challenge_plan";
            std::vector<FrWire> piece_rhos;
            const ConstraintDescriptor relation = weighted_descriptor(kind, tags, pieces, &piece_rhos);
            if (!relation.valid(error)) return false;
            relation_descriptors[kind] = relation;
            const auto rho = pointwise_challenges(relation, relation_providers.front().logical_domain(), relation_providers.size(), context, transcript);
            const bool marker_membership = kind == static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership);
            if (marker_membership && !sorted_marker_pieces_valid(pieces, piece_rhos, descriptor.table_unique_rows, error)) return false;
            const ConstraintDescriptor weighted = marker_membership
                ? sorted_marker_mixed_sumcheck_descriptor(relation, pieces, piece_rhos, relation_providers.size())
                : pointwise_sumcheck_descriptor(relation, relation_providers.size());
            const FrWire initial_claim = marker_membership
                ? zkdl::fr_mul(piece_rhos[0U], zkdl::fr_from_u32(descriptor.table_unique_rows))
                : FrWire{};
            PointwiseEqmaskWeightTable eqmask_table;
            if (!eqmask_table.build(rho, error)) return false;
            std::vector<SegmentedTensorProvider> weighted_providers;
            failure_phase = "sumcheck_provider";
            if (!pointwise_providers(relation, relation_providers, rho,
                                     weighted_providers, error, &eqmask_table,
                                     &lookup_progress)) return false;
            ZkSumcheckProverResult sumcheck;
            failure_phase = "sumcheck";
            if (!prove_zk_sumcheck_segmented(
                    weighted, weighted_providers, weighted.degree,
                    relation_providers.front().logical_domain(), generators,
                    blinding_generator, u_generator, initial_claim, statement, context,
                    transcript, sumcheck, error, descriptor.segment_size,
                    zksc_folded_state_within_budget(
                        weighted_providers, weighted, eqmask_table.peak_live_bytes),
                    "sorted_lookup_sumcheck", descriptor.operation_id.c_str(),
                    spill_directory, true, eqmask_table.peak_live_bytes)) {
                return false;
            }
            weighted_providers.clear();
            eqmask_table.clear();
            std::vector<FrWire> point;
            for (const auto& challenge : sumcheck.challenges)
                point.push_back(zkdl::fr_wire(challenge));
            std::reverse(point.begin(), point.end());
            const ConstraintDescriptor terminal_descriptor = marker_membership
                ? sorted_marker_mixed_terminal_descriptor(relation, pieces, piece_rhos, rho, point)
                : pointwise_terminal_descriptor(relation, rho, point);
            ZkTerminalRelationProverResult terminal;
            failure_phase = "terminal";
            if (!prove_zk_terminal_relation_segmented(
                    terminal_descriptor, relation_providers, canonical, blinders, indices, point,
                    generators, blinding_generator, u_generator, statement,
                    zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
                    zkdl::cpu_g1_from_native(sumcheck.proof.final_claim_commitment),
                    sumcheck.final_q_blinder, FrWire{}, context, transcript,
                    terminal, error, &terminal_materials,
                    descriptor.segment_size, &lookup_progress,
                    authenticated_basis, msm_session, &terminal_receipts)) return false;
            zkdl::AuthenticatedSortedLookupRelationProof wire;
            wire.kind = static_cast<zkdl::AuthenticatedSortedLookupRelationKind>(kind);
            wire.logical_domain = relation_providers.front().logical_domain();
            wire.segment_size = relation_providers.front().segment_size();
            wire.segment_count = relation_providers.front().segment_count();
            wire.relation_digest = zksc_proof_descriptor_digest(relation, statement, context);
            wire.commitment_indices = indices;
            wire.provider_tags = tags;
            wire.outer_commitment_indices = outer_indices;
            wire.sumcheck = sumcheck.proof;
            wire.terminal_relation = terminal.proof;
            failure_phase = "relation_wire";
            if (!wire.valid(error)) return false;
            std::vector<byte> encoded;
            serialize_authenticated_sorted_lookup_relation(wire, encoded);
            transcript.absorb(context + ".proof", encoded);
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                if (tags[index] == "query_endpoint" || tags[index] == "merge_endpoint") {
                    terminal_values[tags[index]] = terminal.terminal_values[index];
                    terminal_blinders[tags[index]] = terminal.terminal_blinders[index];
                    terminal_commitments[tags[index]] = zkdl::cpu_g1_from_native(
                        terminal.proof.terminal_commitments[index]);
                }
            }
            proof.relations.push_back(std::move(wire));
            return true;
            };
            bool completed = false;
            try {
                completed = run_relation_body();
            } catch (const std::exception& exception) {
                if (error) *error = exception.what();
            }
            if (!completed && error) {
                const std::string detail = *error;
                *error = "sorted-lookup relation kind=" + std::to_string(kind) +
                         " phase=" + failure_phase +
                         (detail.empty() ? " failed" : ": " + detail);
            }
            return completed;
        };

        const auto authenticate_outer_count_scratch = [&](std::uint32_t role,
                                                           const SegmentedTensorProvider& provider,
                                                           const char* role_name) {
            const auto outer = outer_by_role.find(std::make_pair(role, 0U));
            if (outer == outer_by_role.end()) {
                if (error) *error = std::string("sorted-lookup outer count scratch role is missing: ") + role_name;
                return false;
            }
            const std::size_t index = outer->second;
            const SegmentCommitmentValidationReceipt* receipt = nullptr;
            if (!validate_outer_commitment(index, provider, receipt)) {
                if (error && error->empty())
                    *error = std::string("sorted-lookup outer count scratch commitment mismatch: ") + role_name;
                return false;
            }
            return true;
        };
        if (!authenticate_outer_count_scratch(
                zkdl::kSortedLookupCommitmentQueryPrefix,
                providers.query_prefix, "query_prefix") ||
            !authenticate_outer_count_scratch(
                zkdl::kSortedLookupCommitmentMergePrefix,
                providers.merge_prefix, "merge_prefix")) return false;

        const auto index_map = [](const std::vector<std::string>& tags) {
            std::map<std::string, std::uint32_t> result;
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                std::uint32_t tag_index = 0U;
                if (!checked_relation_u32(index, tag_index,
                                         "sorted-lookup tag index"))
                    throw std::invalid_argument("sorted-lookup tag count exceeds uint32");
                result.emplace(tags[index], tag_index);
            }
            return result;
        };
        const auto monomial = [](const FrWire& coefficient,
                                 std::initializer_list<std::uint32_t> refs) {
            Monomial result;
            result.coefficient = coefficient;
            result.refs.assign(refs.begin(), refs.end());
            return result;
        };

        const std::vector<std::string> query_tags = {
            "query_x", "query_y", "query_key", "query_active", "query_product_prefix",
            "query_prefix_next", "query_first", "query_last", "query_endpoint"};
        const auto query_indices = index_map(query_tags);
        const FrWire query_base_wire = sorted_lookup_field_from_u128(query_base);
        std::vector<ConstraintDescriptor> query_pieces;
        query_pieces.push_back(make_piece(
            "query-key-encoding", 1U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(query_base_wire), {query_indices.at("query_x")}),
             monomial(zkdl::fr_from_u32(std::numeric_limits<std::uint32_t>::max()),
                      {query_indices.at("query_y")})}));
        // The last coefficient above is replaced with -1 below so the
        // expression remains independent of the field's signed encoding.
        query_pieces.back().terms.back().coefficient = zkdl::fr_neg(zkdl::fr_from_u32(1U));
        query_pieces.push_back(make_piece(
            "query-x-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_x")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_x"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-y-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_y")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_y"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-key-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_key"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_active"), query_indices.at("query_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-prefix-recurrence", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_prefix_next")}),
             monomial(zkdl::fr_neg(beta),
                      {query_indices.at("query_product_prefix"), query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(gamma), {query_indices.at("query_product_prefix")})}));
        query_pieces.push_back(make_piece(
            "query-prefix-initial", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_first"), query_indices.at("query_product_prefix")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_first")})}));
        query_pieces.push_back(make_piece(
            "query-endpoint", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_last"), query_indices.at("query_prefix_next")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_last"), query_indices.at("query_endpoint")})}));
        std::vector<SegmentedTensorProvider> query_relation_providers = {
            providers.query_x, providers.query_y, providers.query_keys,
            providers.query_active, query_product_prefix, providers.query_prefix_next,
            providers.query_first, providers.query_last, query_endpoint_provider};
        if (!run_relation(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::QueryBinding),
                          query_tags, query_relation_providers, query_pieces)) return false;

        const std::vector<std::string> recurrence_tags = {
            "merge_product_prefix", "merge_prefix_next", "merge_key", "merge_active", "merge_first"};
        const auto recurrence_indices = index_map(recurrence_tags);
        std::vector<ConstraintDescriptor> recurrence_pieces;
        recurrence_pieces.push_back(make_piece(
            "merge-prefix-recurrence", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {recurrence_indices.at("merge_prefix_next")}),
             monomial(zkdl::fr_neg(beta),
                      {recurrence_indices.at("merge_product_prefix"), recurrence_indices.at("merge_key")}),
             monomial(zkdl::fr_neg(gamma), {recurrence_indices.at("merge_product_prefix")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-prefix-initial", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {recurrence_indices.at("merge_first"), recurrence_indices.at("merge_product_prefix")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_first")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {recurrence_indices.at("merge_active"), recurrence_indices.at("merge_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_active")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-key-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {recurrence_indices.at("merge_key")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_key"), recurrence_indices.at("merge_active")})}));
        std::vector<SegmentedTensorProvider> recurrence_providers = {
            merge_product_prefix, providers.merge_prefix_next, providers.merge_keys,
            providers.merge_active, providers.merge_first};
        if (!run_relation(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MultisetRecurrence),
                          recurrence_tags, recurrence_providers, recurrence_pieces)) return false;

        std::vector<std::string> sorted_tags = {
            "merge_key", "merge_previous_key", "merge_delta", "merge_active"};
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit)
            sorted_tags.push_back("merge_bit_" + std::to_string(bit));
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit)
            sorted_tags.push_back("merge_delta_bit_" + std::to_string(bit));
        sorted_tags.insert(sorted_tags.end(), {"merge_inverse", "merge_nonzero", "merge_marker", "merge_first"});
        const auto sorted_indices = index_map(sorted_tags);
        const auto key_ref = [&](std::uint32_t bit) { return sorted_indices.at("merge_bit_" + std::to_string(bit)); };
        const auto delta_bit_ref = [&](std::uint32_t bit) { return sorted_indices.at("merge_delta_bit_" + std::to_string(bit)); };
        std::vector<ConstraintDescriptor> sorted_pieces;
        sorted_pieces.push_back(make_piece(
            "sorted-delta", 1U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_key")}),
             monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_previous_key")})}));
        std::vector<Monomial> key_reconstruction = {
            monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_key")})};
        std::vector<Monomial> delta_reconstruction = {
            monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")})};
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit) {
            const FrWire weight = sorted_lookup_field_from_u128(static_cast<unsigned __int128>(1U) << bit);
            key_reconstruction.push_back(monomial(zkdl::fr_neg(weight), {key_ref(bit)}));
            delta_reconstruction.push_back(monomial(zkdl::fr_neg(weight), {delta_bit_ref(bit)}));
            sorted_pieces.push_back(make_piece(
                "sorted-key-bit-boolean-" + std::to_string(bit), 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {key_ref(bit), key_ref(bit)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {key_ref(bit)})}));
            sorted_pieces.push_back(make_piece(
                "sorted-delta-bit-boolean-" + std::to_string(bit), 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {delta_bit_ref(bit), delta_bit_ref(bit)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {delta_bit_ref(bit)})}));
        }
        sorted_pieces.push_back(make_piece("sorted-key-reconstruction", 1U, FrWire{}, key_reconstruction));
        sorted_pieces.push_back(make_piece("sorted-delta-reconstruction", 1U, FrWire{}, delta_reconstruction));
        sorted_pieces.push_back(make_piece(
            "sorted-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_active"), sorted_indices.at("merge_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_active")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-inverse", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta"), sorted_indices.at("merge_inverse")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-positive-delta", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {sorted_indices.at("merge_delta"), sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-marker", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_marker")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_first")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")}),
             monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_first"), sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-nonzero-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_nonzero"), sorted_indices.at("merge_nonzero")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")})}));
        for (const auto& tag : {std::string("merge_delta"), std::string("merge_inverse"),
                                std::string("merge_marker"), std::string("merge_previous_key")}) {
            sorted_pieces.push_back(make_piece(
                "sorted-padding-" + tag, 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at(tag)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                          {sorted_indices.at(tag), sorted_indices.at("merge_active")})}));
        }
        std::vector<SegmentedTensorProvider> sorted_providers;
        for (const auto& tag : sorted_tags) {
            SegmentedTensorProvider provider;
            if (!provider_for_tag(tag, provider)) return false;
            sorted_providers.push_back(std::move(provider));
        }
        if (!run_relation(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::SortedCanonical),
                          sorted_tags, sorted_providers, sorted_pieces)) return false;

        const std::vector<std::string> marker_tags = {"merge_marker", "merge_active"};
        const auto marker_indices = index_map(marker_tags);
        std::vector<ConstraintDescriptor> marker_pieces;
        marker_pieces.push_back(make_piece(
            "marker-count", 1U, zkdl::fr_neg(zkdl::fr_from_u32(descriptor.table_unique_rows)),
            {monomial(zkdl::fr_from_u32(1U), {marker_indices.at("merge_marker")})}));
        marker_pieces.push_back(make_piece(
            "marker-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {marker_indices.at("merge_marker")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {marker_indices.at("merge_marker"), marker_indices.at("merge_active")})}));
        if (!run_relation(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership),
                          marker_tags, {providers.merge_marker, providers.merge_active}, marker_pieces)) return false;

        const std::vector<std::string> endpoint_tags = {
            "merge_prefix_next", "merge_last", "merge_endpoint"};
        const auto endpoint_indices = index_map(endpoint_tags);
        std::vector<ConstraintDescriptor> endpoint_pieces = {make_piece(
            "merge-endpoint", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {endpoint_indices.at("merge_last"), endpoint_indices.at("merge_prefix_next")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {endpoint_indices.at("merge_last"), endpoint_indices.at("merge_endpoint")})})};
        if (!run_relation(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::EndpointEquality),
                          endpoint_tags, {providers.merge_prefix_next, providers.merge_last, merge_endpoint_provider},
                          endpoint_pieces)) return false;

        if (terminal_values.find("query_endpoint") == terminal_values.end() ||
            terminal_values.find("merge_endpoint") == terminal_values.end() ||
            terminal_blinders["query_endpoint"] == FrWire{} ||
            terminal_blinders["merge_endpoint"] == FrWire{}) {
            if (error) *error = "sorted-lookup endpoint terminal claims are missing";
            return false;
        }
        FrWire table_blinder = zkdl::cpu_fresh_csprng_scalar();
        while (table_blinder == FrWire{}) table_blinder = zkdl::cpu_fresh_csprng_scalar();
        const zkdl::CpuG1Point table_commitment = zkdl::cpu_pedersen_scalar_commit(
            table_product, table_blinder, blinding_generator, u_generator);
        const auto table_context = zksc_make_scalar_context(
            relation_descriptors.at(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership)),
            statement, "gat.sorted_lookup.table_product", "table_product", 0U, 0xE1000002U,
            zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        try {
            proof.table_product_value_proof = zkdl::cpu_open_pedersen_scalar_value_public(
                table_product, table_blinder, table_commitment, blinding_generator,
                u_generator, table_context).proof;
        } catch (const std::exception& exception) {
            if (error) *error = std::string("sorted-lookup phase=table_product_opening: ") +
                                exception.what();
            return false;
        }
        proof.table_product_commitment = zkdl::cpu_g1_to_native(table_commitment);
        proof.query_endpoint_commitment = zkdl::cpu_g1_to_native(terminal_commitments.at("query_endpoint"));
        proof.merge_endpoint_commitment = zkdl::cpu_g1_to_native(terminal_commitments.at("merge_endpoint"));
        const auto endpoint_context = zktr_make_product_context(
            statement, descriptor.descriptor_digest, blinding_generator, u_generator,
            0xE1000001U, 0xE1000002U, 0xE1000003U);
        zkdl::PedersenProductProverResult endpoint_product;
        try {
            endpoint_product = zkdl::cpu_open_pedersen_product(
                terminal_values.at("query_endpoint"), table_product,
                terminal_blinders.at("query_endpoint"), table_blinder,
                terminal_blinders.at("merge_endpoint"), terminal_commitments.at("query_endpoint"),
                table_commitment, terminal_commitments.at("merge_endpoint"),
                blinding_generator, u_generator, endpoint_context);
        } catch (const std::exception& exception) {
            if (error) *error = std::string("sorted-lookup phase=endpoint_product_opening: ") +
                                exception.what();
            return false;
        }
        proof.endpoint_product = endpoint_product.proof;
        std::vector<byte> table_value_bytes, endpoint_product_bytes;
        zkdl::serialize_pedersen_scalar_value(proof.table_product_value_proof, table_value_bytes);
        zkdl::serialize_pedersen_product(proof.endpoint_product, endpoint_product_bytes);
        transcript.absorb("gat.sorted_lookup.table_product", table_value_bytes);
        transcript.absorb("gat.sorted_lookup.endpoint_product", endpoint_product_bytes);

        std::vector<byte> layout, order;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, layout);
        for (const auto& ref : commitment_refs) {
            zkdl::append_u32(layout, ref.role);
            zkdl::append_u32(layout, ref.column_index);
            zkdl::append_u32(layout, ref.commitment_index);
            zkdl::append_u32(layout, ref.logical_domain);
            zkdl::append_u32(layout, ref.segment_size);
            zkdl::append_u32(layout, ref.segment_count);
            zkdl::append_u32(order, ref.role);
            zkdl::append_u32(order, ref.column_index);
            zkdl::append_u32(order, ref.commitment_index);
        }
        proof.layout_digest = zkdl::sha256(layout);
        proof.commitment_order_digest = zkdl::sha256(order);
        proof.transcript_digest = transcript.digest();
        if (!proof.valid(error)) return false;
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = "sorted-lookup proof failed operation_id=" +
            descriptor.operation_id + " phase=sorted_lookup: " + exception.what();
        return false;
    }
}

inline bool verify_authenticated_sorted_lookup(
    const zkdl::AuthenticatedSortedLookupDescriptor& descriptor,
    const SegmentedTensorProvider& authenticated_table_keys,
    const SegmentedTensorProvider& authenticated_table_active,
    const std::vector<zkdl::AuthenticatedSortedLookupCommitmentRef>& commitment_refs,
    const std::vector<zkdl::CpuG1Point>& outer_commitments,
    const std::vector<CommitmentWire>& outer_wires,
    const std::vector<zkdl::CpuG1Point>& generators,
    const zkdl::CpuG1Point& blinding_generator,
    const zkdl::CpuG1Point& u_generator,
    const RelationStatement& statement,
    const zkdl::AuthenticatedSortedLookupProof& proof,
    std::string* error,
    const zkdl::CpuIpaBasisContext* authenticated_basis,
    zkdl::CpuG1MsmSession* msm_session) {
    try {
        if (!descriptor.valid(error) || descriptor.table_digest != statement.table_digest ||
            !zktr_all_statement_digests_valid(statement, error) ||
            !authenticated_table_keys.valid(error) || !authenticated_table_active.valid(error) ||
            authenticated_table_keys.logical_domain() != descriptor.table_domain ||
            authenticated_table_active.logical_domain() != descriptor.table_domain ||
            authenticated_table_keys.segment_size() != descriptor.segment_size ||
            authenticated_table_active.segment_size() != descriptor.segment_size ||
            authenticated_table_keys.segment_count() != descriptor.table_segment_count ||
            authenticated_table_active.segment_count() != descriptor.table_segment_count ||
            commitment_refs.empty() || commitment_refs.size() != outer_commitments.size() ||
            commitment_refs.size() != outer_wires.size() ||
            !zkdl::cpu_pedersen_strict_g1(blinding_generator, error) ||
            !zkdl::cpu_pedersen_strict_g1(u_generator, error) ||
            !proof.valid(error)) {
            if (error && error->empty()) *error = "invalid authenticated sorted-lookup verifier inputs";
            return false;
        }
        if (authenticated_basis != nullptr &&
            !zktr_basis_valid_for_prefix(
                generators, blinding_generator, u_generator,
                static_cast<std::uint32_t>(generators.size()),
                authenticated_basis, error)) return false;
        std::vector<byte> expected_descriptor, actual_descriptor;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, expected_descriptor);
        serialize_authenticated_sorted_lookup_descriptor(proof.descriptor, actual_descriptor);
        if (expected_descriptor != actual_descriptor || proof.commitments.size() != commitment_refs.size()) {
            if (error) *error = "sorted-lookup proof descriptor/outer map mismatch";
            return false;
        }
        const auto expected_domain = [&](std::uint32_t role) {
            return role == zkdl::kSortedLookupCommitmentQueryKey ||
                           role == zkdl::kSortedLookupCommitmentQueryBit ||
                           role == zkdl::kSortedLookupCommitmentQueryPrefix ||
                           role == zkdl::kSortedLookupCommitmentQueryX ||
                           role == zkdl::kSortedLookupCommitmentQueryY
                       ? descriptor.query_domain
                       : descriptor.merge_domain;
        };
        std::map<std::pair<std::uint32_t, std::uint32_t>, std::size_t> outer_by_role;
        for (std::size_t index = 0U; index < commitment_refs.size(); ++index) {
            const auto& expected = commitment_refs[index];
            const auto& actual = proof.commitments[index];
            const auto& wire = outer_wires[index];
            zkdl::CpuG1Point legacy = zkdl::CpuG1Point::zero();
            if (!segment_commitment_wire_shape_valid(wire, error) ||
                segment_commitment_digest(wire) != wire.segment_commitment_digest ||
                wire.visibility != 3U || wire.domain_size != expected.logical_domain ||
                wire.segment_size != expected.segment_size ||
                wire.segment_count != expected.segment_count ||
                wire.srs_manifest_digest != statement.srs_digest) {
                if (error && error->empty()) *error =
                    "sorted-lookup outer segment wire is not authenticated";
                return false;
            }
            for (const auto& segment : wire.segment_commitments)
                legacy = zkdl::cpu_g1_add(
                    legacy, zkdl::cpu_g1_from_native(segment.commitment));
            if (!expected.valid(error) || !actual.valid(error) ||
                expected.role != actual.role || expected.column_index != actual.column_index ||
                expected.commitment_index != actual.commitment_index ||
                expected.logical_domain != actual.logical_domain ||
                expected.segment_size != actual.segment_size ||
                expected.segment_count != actual.segment_count ||
                expected.logical_domain != expected_domain(expected.role) ||
                expected.segment_size != descriptor.segment_size ||
                expected.segment_count != expected_domain(expected.role) / descriptor.segment_size ||
                !zkdl::cpu_g1_equal(legacy,
                                    zkdl::cpu_g1_from_native(wire.commitment)) ||
                !zkdl::cpu_g1_equal(outer_commitments[index],
                                    zkdl::cpu_g1_from_native(wire.commitment)) ||
                !outer_by_role.emplace(std::make_pair(expected.role, expected.column_index), index).second ||
                !zkdl::cpu_pedersen_strict_g1(outer_commitments[index], error)) {
                if (error && error->empty()) *error = "invalid sorted-lookup verifier commitment handle";
                return false;
            }
        }
        const auto role_for_tag = [&](const std::string& tag) {
            if (tag == "query_x") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryX), 0U);
            if (tag == "query_y") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryY), 0U);
            if (tag == "query_key") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryKey), 0U);
            if (tag == "query_prefix") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentQueryPrefix), 0U);
            if (tag == "merge_key") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeKey), 0U);
            if (tag == "merge_delta") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeDelta), 0U);
            if (tag == "merge_marker") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeMarker), 0U);
            if (tag == "merge_inverse") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeInverse), 0U);
            if (tag == "merge_prefix") return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergePrefix), 0U);
            const std::string key = "merge_bit_";
            if (tag.compare(0U, key.size(), key) == 0) {
                std::uint32_t bit = std::numeric_limits<std::uint32_t>::max();
                (void)parse_relation_u32_suffix(tag.substr(key.size()), bit);
                return std::make_pair(static_cast<std::uint32_t>(zkdl::kSortedLookupCommitmentMergeBit),
                                      bit);
            }
            return std::make_pair(static_cast<std::uint32_t>(0U), 0U);
        };
        const auto expected_outer = [&](const std::string& tag) -> std::uint32_t {
            const auto role = role_for_tag(tag);
            if (role.first == 0U) return std::numeric_limits<std::uint32_t>::max();
            const auto found = outer_by_role.find(role);
            return found == outer_by_role.end()
                       ? std::numeric_limits<std::uint32_t>::max()
                       : commitment_refs[found->second].commitment_index;
        };
        const auto index_map = [](const std::vector<std::string>& tags) {
            std::map<std::string, std::uint32_t> result;
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                std::uint32_t tag_index = 0U;
                if (!checked_relation_u32(index, tag_index,
                                         "sorted-lookup tag index"))
                    throw std::invalid_argument("sorted-lookup tag count exceeds uint32");
                result.emplace(tags[index], tag_index);
            }
            return result;
        };
        const auto monomial = [](const FrWire& coefficient,
                                 std::initializer_list<std::uint32_t> refs) {
            Monomial result;
            result.coefficient = coefficient;
            result.refs.assign(refs.begin(), refs.end());
            return result;
        };
        const auto make_piece = [](const std::string& id, std::uint32_t degree,
                                   const FrWire& constant,
                                   const std::vector<Monomial>& terms) {
            ConstraintDescriptor piece;
            piece.id = id;
            piece.degree = degree;
            piece.constant = constant;
            piece.terms = terms;
            return piece;
        };
        zkdl::Transcript transcript("zkdl-authenticated-sorted-lookup-v1");
        std::vector<byte> header;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, header);
        zkdl::append_u64(header, static_cast<std::uint64_t>(commitment_refs.size()));
        for (std::size_t index = 0U; index < commitment_refs.size(); ++index) {
            const auto& ref = commitment_refs[index];
            zkdl::append_u32(header, ref.role);
            zkdl::append_u32(header, ref.column_index);
            zkdl::append_u32(header, ref.commitment_index);
            zkdl::append_u32(header, ref.logical_domain);
            zkdl::append_u32(header, ref.segment_size);
            zkdl::append_u32(header, ref.segment_count);
            zkdl::append_g1(header, zkdl::cpu_g1_to_native(outer_commitments[index]));
        }
        transcript.absorb("gat.sorted_lookup.header", header);
        const FrWire beta = zkdl::cpu_ipa_nonzero_challenge(transcript, "gat.sorted_lookup.beta");
        const FrWire gamma = zkdl::cpu_ipa_nonzero_challenge(transcript, "gat.sorted_lookup.gamma");
        FrWire table_product{};
        if (!authenticated_sorted_lookup_public_table_product(
                authenticated_table_keys, authenticated_table_active,
                descriptor.table_active_length, beta, gamma, table_product, error)) return false;
        unsigned __int128 previous_key = 0U;
        bool have_previous = false;
        for (std::uint32_t segment = 0U; segment < descriptor.table_segment_count; ++segment) {
            std::vector<FrWire> keys, active;
            if (!authenticated_table_keys.read_segment(segment, keys, error) ||
                !authenticated_table_active.read_segment(segment, active, error)) return false;
            for (std::size_t index = 0U; index < keys.size(); ++index) {
                const std::uint64_t global = static_cast<std::uint64_t>(segment) * descriptor.segment_size + index;
                if (global >= descriptor.table_domain) break;
                const bool is_active = global < descriptor.table_active_length;
                if ((active[index] == zkdl::fr_from_u32(1U)) != is_active) {
                    if (error) *error = "authenticated table active mask is not canonical";
                    return false;
                }
                unsigned __int128 current = 0U;
                if (!sorted_lookup_u128_from_field(keys[index], descriptor.x_bits + descriptor.y_bits,
                                                   current, error)) return false;
                if (is_active) {
                    if (have_previous && current <= previous_key) {
                        if (error) *error = "authenticated table is not strictly sorted";
                        return false;
                    }
                    previous_key = current;
                    have_previous = true;
                } else if (current != 0U) {
                    if (error) *error = "authenticated table padding is nonzero";
                    return false;
                }
            }
        }

        const auto weighted_descriptor = [&](std::uint32_t kind,
                                             const std::vector<std::string>& tags,
                                             const std::vector<ConstraintDescriptor>& pieces,
                                             std::vector<FrWire>* piece_rhos) {
            std::vector<byte> plan;
            zkdl::append_u32(plan, kind);
            zkdl::append_u64(plan, static_cast<std::uint64_t>(tags.size()));
            for (const auto& tag : tags) zkdl::append_string(plan, tag);
            zkdl::append_u64(plan, static_cast<std::uint64_t>(pieces.size()));
            for (const auto& piece : pieces) append_constraint(plan, piece);
            const std::string context = "gat.sorted_lookup.relation." + std::to_string(kind);
            transcript.absorb(context + ".plan", plan);
            ConstraintDescriptor result;
            result.id = "authenticated_sorted_lookup/relation/" + std::to_string(kind);
            result.degree = 1U;
            if (piece_rhos != nullptr) { piece_rhos->clear(); piece_rhos->reserve(pieces.size()); }
            for (const auto& piece : pieces) result.degree = std::max(result.degree, piece.degree);
            for (std::size_t index = 0U; index < pieces.size(); ++index) {
                const FrWire rho = zkdl::cpu_ipa_nonzero_challenge(
                    transcript, context + ".rho." + std::to_string(index));
                if (piece_rhos != nullptr) piece_rhos->push_back(rho);
                result.constant = zkdl::fr_add(result.constant,
                                               zkdl::fr_mul(rho, pieces[index].constant));
                for (const auto& term : pieces[index].terms) {
                    Monomial weighted = term;
                    weighted.coefficient = zkdl::fr_mul(rho, term.coefficient);
                    result.terms.push_back(std::move(weighted));
                }
            }
            return result;
        };
        std::map<std::uint32_t, ConstraintDescriptor> relation_descriptors;
        std::map<std::string, zkdl::CpuG1Point> internal_commitments;
        std::map<std::string, std::vector<zkdl::CpuG1Point>> internal_segment_commitments;
        zkdl::CpuG1Point query_endpoint_commitment = zkdl::CpuG1Point::zero();
        zkdl::CpuG1Point merge_endpoint_commitment = zkdl::CpuG1Point::zero();
        const auto verify_relation = [&](std::size_t relation_index,
                                         std::uint32_t kind,
                                         const std::vector<std::string>& tags,
                                         const std::vector<ConstraintDescriptor>& pieces) {
            if (relation_index >= proof.relations.size()) return false;
            const auto& wire = proof.relations[relation_index];
            if (static_cast<std::uint32_t>(wire.kind) != kind || wire.provider_tags != tags ||
                wire.commitment_indices.size() != tags.size() ||
                wire.outer_commitment_indices.size() != tags.size()) return false;
            std::vector<zkdl::CpuG1Point> canonical;
            std::vector<std::vector<zkdl::CpuG1Point>> provider_segments;
            std::vector<std::uint32_t> indices;
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                if (wire.commitment_indices[index] != index ||
                    wire.outer_commitment_indices[index] != expected_outer(tags[index])) return false;
                const auto role = role_for_tag(tags[index]);
                zkdl::CpuG1Point commitment = zkdl::CpuG1Point::zero();
                if (role.first != 0U) {
                    const auto found = outer_by_role.find(role);
                    if (found == outer_by_role.end()) return false;
                    commitment = outer_commitments[found->second];
                } else {
                    if (index >= wire.terminal_relation.terminal_segment_openings.size() ||
                        wire.terminal_relation.terminal_segment_openings[index].empty()) return false;
                    for (const auto& opening : wire.terminal_relation.terminal_segment_openings[index])
                        commitment = zkdl::cpu_g1_add(
                            commitment, zkdl::cpu_g1_from_native(opening.xi));
                    const auto existing = internal_commitments.find(tags[index]);
                    if (existing != internal_commitments.end() &&
                        !zkdl::cpu_g1_equal(existing->second, commitment)) return false;
                    internal_commitments[tags[index]] = commitment;
                }
                if (index >= wire.terminal_relation.terminal_segment_openings.size()) return false;
                const auto& openings = wire.terminal_relation.terminal_segment_openings[index];
                ZkRelationSegmentLayout public_layout;
                if (wire.segment_size != descriptor.segment_size ||
                    !zktr_segment_layout_for_size(
                        wire.logical_domain, generators.size(),
                        descriptor.segment_size, public_layout, error) ||
                    wire.segment_count != public_layout.segment_count ||
                    openings.size() != public_layout.segment_count) {
                    if (error && error->empty())
                        *error = "sorted-lookup relation wire differs from authenticated shared layout";
                    return false;
                }
                std::vector<zkdl::CpuG1Point> fixed_segments;
                for (std::size_t segment = 0U; segment < openings.size(); ++segment) {
                    if (openings[segment].segment_index != segment || openings[segment].logical_domain != wire.logical_domain || openings[segment].logical_offset != segment * static_cast<std::uint64_t>(public_layout.segment_size) || openings[segment].padded_length != public_layout.segment_size) return false;
                    fixed_segments.push_back(zkdl::cpu_g1_from_native(openings[segment].xi));
                }
                if (role.first == 0U) {
                    const auto existing = internal_segment_commitments.find(tags[index]);
                    if (existing != internal_segment_commitments.end() && !same_g1_basis(existing->second, fixed_segments)) return false;
                    internal_segment_commitments[tags[index]] = fixed_segments;
                }
                provider_segments.push_back(std::move(fixed_segments));
                canonical.push_back(commitment);
                std::uint32_t relation_index = 0U;
                if (!checked_relation_u32(index, relation_index,
                                          "sorted-lookup relation commitment index", error))
                    return false;
                indices.push_back(relation_index);
            }
            const std::string context = "gat.sorted_lookup.relation." + std::to_string(kind);
            pointwise_absorb_provider_commitments(context, wire.logical_domain, tags, canonical, provider_segments, transcript);
            std::vector<FrWire> piece_rhos;
            const ConstraintDescriptor relation = weighted_descriptor(kind, tags, pieces, &piece_rhos);
            if (!relation.valid(error) || wire.relation_digest != zksc_proof_descriptor_digest(relation, statement, context)) return false;
            relation_descriptors[kind] = relation;
            if (!pointwise_shape_domain(relation, wire.logical_domain, error)) return false;
            const auto rho = pointwise_challenges(relation, wire.logical_domain, tags.size(), context, transcript);
            const bool marker_membership = kind == static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership);
            if (marker_membership && !sorted_marker_pieces_valid(pieces, piece_rhos, descriptor.table_unique_rows, error)) return false;
            const ConstraintDescriptor weighted = marker_membership
                ? sorted_marker_mixed_sumcheck_descriptor(relation, pieces, piece_rhos, tags.size())
                : pointwise_sumcheck_descriptor(relation, tags.size());
            const FrWire initial_claim = marker_membership
                ? zkdl::fr_mul(piece_rhos[0U], zkdl::fr_from_u32(descriptor.table_unique_rows))
                : FrWire{};
            ZkSumcheckVerifierResult sumcheck;
            if (!verify_zk_sumcheck(
                    wire.sumcheck, weighted, weighted.degree, wire.logical_domain,
                    generators, blinding_generator, u_generator, initial_claim, statement,
                    context, transcript, sumcheck, error,
                    descriptor.segment_size)) return false;
            std::vector<FrWire> point;
            for (const auto& challenge : sumcheck.challenges)
                point.push_back(zkdl::fr_wire(challenge));
            std::reverse(point.begin(), point.end());
            // The terminal verifier below checks each Hyrax opening, while
            // this lookup-specific check binds outer openings to the
            // authenticated ordered Xi vector and its high-point weights.
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                const auto role = role_for_tag(tags[index]);
                if (role.first == 0U) continue;
                const auto found = outer_by_role.find(role);
                if (found == outer_by_role.end() ||
                    found->second >= outer_wires.size() ||
                    index >= wire.terminal_relation.terminal_segment_openings.size())
                    return false;
                const auto& statement_wire = outer_wires[found->second];
                const auto& openings = wire.terminal_relation.terminal_segment_openings[index];
                if (openings.size() != statement_wire.segment_count) return false;
                const std::vector<FrWire> high_point = zktr_segment_prefix_point(
                    point, statement_wire.segment_size);
                const std::vector<FrWire> high_weights = zkdl::cpu_ipa_mle_weights(high_point);
                if (high_weights.size() != openings.size()) return false;
                zkdl::CpuG1Point actual_weighted = zkdl::CpuG1Point::zero();
                zkdl::CpuG1Point expected_weighted = zkdl::CpuG1Point::zero();
                for (std::size_t segment = 0U; segment < openings.size(); ++segment) {
                    const auto& opening = openings[segment];
                    const auto& declared = statement_wire.segment_commitments[segment];
                    if (opening.segment_index != declared.segment_index ||
                        opening.logical_domain != statement_wire.domain_size ||
                        opening.logical_offset != declared.logical_offset ||
                        opening.actual_length != declared.actual_length ||
                        opening.padded_length != declared.padded_length ||
                        !zkdl::cpu_g1_equal(
                            zkdl::cpu_g1_from_native(opening.xi),
                            zkdl::cpu_g1_from_native(declared.commitment)))
                        return false;
                    actual_weighted = zkdl::cpu_g1_add(
                        actual_weighted,
                        zkdl::cpu_g1_mul(
                            zkdl::cpu_g1_from_native(opening.xi), high_weights[segment]));
                    expected_weighted = zkdl::cpu_g1_add(
                        expected_weighted,
                        zkdl::cpu_g1_mul(
                            zkdl::cpu_g1_from_native(declared.commitment), high_weights[segment]));
                }
                if (!zkdl::cpu_g1_equal(actual_weighted, expected_weighted)) return false;
            }
            ZkTerminalRelationVerifierResult terminal;
            if (!verify_zk_terminal_relation(
                    wire.terminal_relation,
                    marker_membership
                        ? sorted_marker_mixed_terminal_descriptor(relation, pieces, piece_rhos, rho, point)
                        : pointwise_terminal_descriptor(relation, rho, point), canonical, indices, point,
                    generators, blinding_generator, u_generator, statement,
                    zkdl::ZkTerminalClaimLinkKind::ExternalEquality,
                    zkdl::cpu_g1_from_native(sumcheck.final_claim_commitment), FrWire{},
                    context, transcript, terminal, error, nullptr,
                    descriptor.segment_size, authenticated_basis,
                    msm_session)) return false;
            std::vector<byte> encoded;
            serialize_authenticated_sorted_lookup_relation(wire, encoded);
            transcript.absorb(context + ".proof", encoded);
            for (std::size_t index = 0U; index < tags.size(); ++index) {
                if (tags[index] == "query_endpoint")
                    query_endpoint_commitment = terminal.terminal_commitments[index];
                if (tags[index] == "merge_endpoint")
                    merge_endpoint_commitment = terminal.terminal_commitments[index];
            }
            return true;
        };

        const unsigned __int128 query_base = static_cast<unsigned __int128>(1U) << descriptor.y_bits;
        const std::vector<std::string> query_tags = {
            "query_x", "query_y", "query_key", "query_active", "query_product_prefix",
            "query_prefix_next", "query_first", "query_last", "query_endpoint"};
        const auto query_indices = index_map(query_tags);
        const FrWire query_base_wire = sorted_lookup_field_from_u128(query_base);
        std::vector<ConstraintDescriptor> query_pieces;
        query_pieces.push_back(make_piece(
            "query-key-encoding", 1U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(query_base_wire), {query_indices.at("query_x")}),
             monomial(zkdl::fr_from_u32(std::numeric_limits<std::uint32_t>::max()),
                      {query_indices.at("query_y")})}));
        // The last coefficient above is replaced with -1 below so the
        // expression remains independent of the field's signed encoding.
        query_pieces.back().terms.back().coefficient = zkdl::fr_neg(zkdl::fr_from_u32(1U));
        query_pieces.push_back(make_piece(
            "query-x-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_x")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_x"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-y-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_y")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_y"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-key-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_key"), query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_active"), query_indices.at("query_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_active")})}));
        query_pieces.push_back(make_piece(
            "query-prefix-recurrence", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {query_indices.at("query_prefix_next")}),
             monomial(zkdl::fr_neg(beta),
                      {query_indices.at("query_product_prefix"), query_indices.at("query_key")}),
             monomial(zkdl::fr_neg(gamma), {query_indices.at("query_product_prefix")})}));
        query_pieces.push_back(make_piece(
            "query-prefix-initial", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_first"), query_indices.at("query_product_prefix")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_first")})}));
        query_pieces.push_back(make_piece(
            "query-endpoint", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {query_indices.at("query_last"), query_indices.at("query_prefix_next")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {query_indices.at("query_last"), query_indices.at("query_endpoint")})}));
        if (!verify_relation(0U,
                             static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::QueryBinding),
                             query_tags, query_pieces)) return false;

        const std::vector<std::string> recurrence_tags = {
            "merge_product_prefix", "merge_prefix_next", "merge_key", "merge_active", "merge_first"};
        const auto recurrence_indices = index_map(recurrence_tags);
        std::vector<ConstraintDescriptor> recurrence_pieces;
        recurrence_pieces.push_back(make_piece(
            "merge-prefix-recurrence", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {recurrence_indices.at("merge_prefix_next")}),
             monomial(zkdl::fr_neg(beta),
                      {recurrence_indices.at("merge_product_prefix"), recurrence_indices.at("merge_key")}),
             monomial(zkdl::fr_neg(gamma), {recurrence_indices.at("merge_product_prefix")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-prefix-initial", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {recurrence_indices.at("merge_first"), recurrence_indices.at("merge_product_prefix")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_first")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {recurrence_indices.at("merge_active"), recurrence_indices.at("merge_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_active")})}));
        recurrence_pieces.push_back(make_piece(
            "merge-key-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {recurrence_indices.at("merge_key")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {recurrence_indices.at("merge_key"), recurrence_indices.at("merge_active")})}));
        if (!verify_relation(1U,
                             static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MultisetRecurrence),
                             recurrence_tags, recurrence_pieces)) return false;

        std::vector<std::string> sorted_tags = {
            "merge_key", "merge_previous_key", "merge_delta", "merge_active"};
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit)
            sorted_tags.push_back("merge_bit_" + std::to_string(bit));
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit)
            sorted_tags.push_back("merge_delta_bit_" + std::to_string(bit));
        sorted_tags.insert(sorted_tags.end(), {"merge_inverse", "merge_nonzero", "merge_marker", "merge_first"});
        const auto sorted_indices = index_map(sorted_tags);
        const auto key_ref = [&](std::uint32_t bit) { return sorted_indices.at("merge_bit_" + std::to_string(bit)); };
        const auto delta_bit_ref = [&](std::uint32_t bit) { return sorted_indices.at("merge_delta_bit_" + std::to_string(bit)); };
        std::vector<ConstraintDescriptor> sorted_pieces;
        sorted_pieces.push_back(make_piece(
            "sorted-delta", 1U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_key")}),
             monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_previous_key")})}));
        std::vector<Monomial> key_reconstruction = {
            monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_key")})};
        std::vector<Monomial> delta_reconstruction = {
            monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")})};
        for (std::uint32_t bit = 0U; bit < descriptor.x_bits + descriptor.y_bits; ++bit) {
            const FrWire weight = sorted_lookup_field_from_u128(static_cast<unsigned __int128>(1U) << bit);
            key_reconstruction.push_back(monomial(zkdl::fr_neg(weight), {key_ref(bit)}));
            delta_reconstruction.push_back(monomial(zkdl::fr_neg(weight), {delta_bit_ref(bit)}));
            sorted_pieces.push_back(make_piece(
                "sorted-key-bit-boolean-" + std::to_string(bit), 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {key_ref(bit), key_ref(bit)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {key_ref(bit)})}));
            sorted_pieces.push_back(make_piece(
                "sorted-delta-bit-boolean-" + std::to_string(bit), 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {delta_bit_ref(bit), delta_bit_ref(bit)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {delta_bit_ref(bit)})}));
        }
        sorted_pieces.push_back(make_piece("sorted-key-reconstruction", 1U, FrWire{}, key_reconstruction));
        sorted_pieces.push_back(make_piece("sorted-delta-reconstruction", 1U, FrWire{}, delta_reconstruction));
        sorted_pieces.push_back(make_piece(
            "sorted-active-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_active"), sorted_indices.at("merge_active")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_active")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-inverse", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta"), sorted_indices.at("merge_inverse")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-positive-delta", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_delta")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {sorted_indices.at("merge_delta"), sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-marker", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_marker")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_first")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")}),
             monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_first"), sorted_indices.at("merge_nonzero")})}));
        sorted_pieces.push_back(make_piece(
            "sorted-nonzero-boolean", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at("merge_nonzero"), sorted_indices.at("merge_nonzero")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)), {sorted_indices.at("merge_nonzero")})}));
        for (const auto& tag : {std::string("merge_delta"), std::string("merge_inverse"),
                                std::string("merge_marker"), std::string("merge_previous_key")}) {
            sorted_pieces.push_back(make_piece(
                "sorted-padding-" + tag, 2U, FrWire{},
                {monomial(zkdl::fr_from_u32(1U), {sorted_indices.at(tag)}),
                 monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                          {sorted_indices.at(tag), sorted_indices.at("merge_active")})}));
        }
        if (!verify_relation(2U,
                             static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::SortedCanonical),
                             sorted_tags, sorted_pieces)) return false;

        const std::vector<std::string> marker_tags = {"merge_marker", "merge_active"};
        const auto marker_indices = index_map(marker_tags);
        std::vector<ConstraintDescriptor> marker_pieces;
        marker_pieces.push_back(make_piece(
            "marker-count", 1U, zkdl::fr_neg(zkdl::fr_from_u32(descriptor.table_unique_rows)),
            {monomial(zkdl::fr_from_u32(1U), {marker_indices.at("merge_marker")})}));
        marker_pieces.push_back(make_piece(
            "marker-padding", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U), {marker_indices.at("merge_marker")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {marker_indices.at("merge_marker"), marker_indices.at("merge_active")})}));
        if (!verify_relation(3U,
                             static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership),
                             marker_tags, marker_pieces)) return false;

        const std::vector<std::string> endpoint_tags = {
            "merge_prefix_next", "merge_last", "merge_endpoint"};
        const auto endpoint_indices = index_map(endpoint_tags);
        std::vector<ConstraintDescriptor> endpoint_pieces = {make_piece(
            "merge-endpoint", 2U, FrWire{},
            {monomial(zkdl::fr_from_u32(1U),
                      {endpoint_indices.at("merge_last"), endpoint_indices.at("merge_prefix_next")}),
             monomial(zkdl::fr_neg(zkdl::fr_from_u32(1U)),
                      {endpoint_indices.at("merge_last"), endpoint_indices.at("merge_endpoint")})})};
        if (!verify_relation(4U,
                             static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::EndpointEquality),
                             endpoint_tags, endpoint_pieces)) return false;

        if (!zkdl::cpu_g1_equal(query_endpoint_commitment,
                                zkdl::cpu_g1_from_native(proof.query_endpoint_commitment)) ||
            !zkdl::cpu_g1_equal(merge_endpoint_commitment,
                                zkdl::cpu_g1_from_native(proof.merge_endpoint_commitment)) ||
            relation_descriptors.find(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership)) ==
                relation_descriptors.end()) {
            if (error) *error = "sorted-lookup endpoint or marker descriptor mismatch";
            return false;
        }
        const auto table_context = zksc_make_scalar_context(
            relation_descriptors.at(static_cast<std::uint32_t>(zkdl::AuthenticatedSortedLookupRelationKind::MarkerMembership)),
            statement, "gat.sorted_lookup.table_product", "table_product", 0U, 0xE1000002U,
            zkdl::PedersenProofContext::kUnsetIndex,
            zkdl::PedersenProofContext::kOperationScalarValue,
            blinding_generator, u_generator);
        if (!zkdl::cpu_verify_pedersen_scalar_value(
                proof.table_product_value_proof, table_product,
                zkdl::cpu_g1_from_native(proof.table_product_commitment),
                blinding_generator, u_generator, table_context, error)) return false;
        const auto endpoint_context = zktr_make_product_context(
            statement, descriptor.descriptor_digest, blinding_generator, u_generator,
            0xE1000001U, 0xE1000002U, 0xE1000003U);
        if (!zkdl::cpu_verify_pedersen_product(
                proof.endpoint_product, query_endpoint_commitment,
                zkdl::cpu_g1_from_native(proof.table_product_commitment),
                merge_endpoint_commitment, blinding_generator, u_generator,
                endpoint_context, error)) return false;
        std::vector<byte> table_value_bytes, endpoint_product_bytes;
        zkdl::serialize_pedersen_scalar_value(proof.table_product_value_proof, table_value_bytes);
        zkdl::serialize_pedersen_product(proof.endpoint_product, endpoint_product_bytes);
        transcript.absorb("gat.sorted_lookup.table_product", table_value_bytes);
        transcript.absorb("gat.sorted_lookup.endpoint_product", endpoint_product_bytes);
        std::vector<byte> layout, order;
        serialize_authenticated_sorted_lookup_descriptor(descriptor, layout);
        for (const auto& ref : commitment_refs) {
            zkdl::append_u32(layout, ref.role);
            zkdl::append_u32(layout, ref.column_index);
            zkdl::append_u32(layout, ref.commitment_index);
            zkdl::append_u32(layout, ref.logical_domain);
            zkdl::append_u32(layout, ref.segment_size);
            zkdl::append_u32(layout, ref.segment_count);
            zkdl::append_u32(order, ref.role);
            zkdl::append_u32(order, ref.column_index);
            zkdl::append_u32(order, ref.commitment_index);
        }
        if (proof.layout_digest != zkdl::sha256(layout) ||
            proof.commitment_order_digest != zkdl::sha256(order) ||
            proof.transcript_digest != transcript.digest()) {
            if (error) *error = "sorted-lookup layout or transcript digest mismatch";
            return false;
        }
        return true;
    } catch (const std::exception& exception) {
        if (error) *error = exception.what();
        return false;
    }

}  // verify_authenticated_sorted_lookup
}  // namespace relation
}  // namespace zkdl_gat

#endif
