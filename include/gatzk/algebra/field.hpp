#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mcl/bn.hpp>
#include <mutex>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>



namespace gatzk::algebra
{
    // 确保 mcl 库的 BLS12‑381 曲线配对初始化只执行一次
    inline void ensure_mcl_field_ready()
    {
        static std::once_flag once;
        std::call_once(once, []() {
            mcl::bn::initPairing(mcl::BLS12_381);
        });
    }

    // 域元素
    class FieldElement
    {
    public:
        FieldElement()
        {
            ensure_mcl_field_ready();
            // 将内部的mcl::Fr对象清零，表示域元素 0
            value_.clear(); 
        }

        explicit FieldElement(std::uint64_t raw)
        {
            ensure_mcl_field_ready();
            // 判断raw是否在long long的表示范围内（即不会溢出带符号整数）
            if (raw <= static_cast<std::uint64_t>(std::numeric_limits<long long>::max()))
            {
                value_ = static_cast<long long>(raw);
                return;
            }
            // 若raw超出long long范围，则将其转换为十进制字符串，再通过setStr解析为域元素
            value_.setStr(std::to_string(raw), 10);
        }

        // 返回域元素 0
        static FieldElement zero()
        {
            return FieldElement(0);
        }

        // 返回域元素 1
        static FieldElement one()
        {
            return FieldElement(1);
        }
        // 从原生mcl::Fr类型创建域元素
        static FieldElement from_native(const mcl::Fr& value)
        {
            FieldElement out;
            out.value_ = value;
            return out;
        }

        // 从有符号整数创建域元素
        static FieldElement from_signed(std::int64_t value)
        {
            FieldElement out;
            out.value_ = static_cast<long long>(value);
            return out;
        }

        // 从小端字节序的字节数组构造域元素
        static FieldElement from_little_endian_mod(const std::uint8_t* bytes, std::size_t size)
        {
            FieldElement out;
            out.value_.setLittleEndianMod(bytes, size);
            return out;
        }

        // 返回域中指定阶的单位根
        static FieldElement root_of_unity(std::size_t size)
        {
            // 阶必须是非零的2的幂
            if (size == 0 || (size & (size - 1U)) != 0U)
            {
                throw std::runtime_error("domain size must be a non-zero power of two");
            }
            ensure_mcl_field_ready();
            const mcl::Vint exponent = (mcl::Fr::getOp().mp - 1) / static_cast<int>(size);
            // BLS12‑381标量域的一个常见原根是5，作为生成元
            const mcl::Fr generator = 5;
            FieldElement out;
            mcl::Fr::pow(out.value_, generator, exponent);
            return out;
        }

        std::uint64_t value() const
        {
            return value_.getUint64();
        }
        bool is_zero() const
        {
            return value_.isZero();
        }

        FieldElement operator+(const FieldElement& other) const
        {
            FieldElement out;
            mcl::Fr::add(out.value_, value_, other.value_);
            return out;
        }

        FieldElement operator-(const FieldElement& other) const
        {
            FieldElement out;
            mcl::Fr::sub(out.value_, value_, other.value_);
            return out;
        }

        FieldElement operator*(const FieldElement& other) const
        {
            FieldElement out;
            mcl::Fr::mul(out.value_, value_, other.value_);
            return out;
        }

        FieldElement operator/(const FieldElement& other) const
        {
            return *this * other.inv();
        }

        FieldElement& operator+=(const FieldElement& other)
        {
            mcl::Fr::add(value_, value_, other.value_);
            return *this;
        }

        FieldElement& operator-=(const FieldElement& other)
        {
            mcl::Fr::sub(value_, value_, other.value_);
            return *this;
        }

        FieldElement& operator*=(const FieldElement& other)
        {
            mcl::Fr::mul(value_, value_, other.value_);
            return *this;
        }

        bool operator==(const FieldElement& other) const
        {
            return value_ == other.value_;
        }

        FieldElement pow(std::uint64_t exponent) const
        {
            FieldElement out;
            mcl::Fr::pow(out.value_, value_, exponent);
            return out;
        }

        // 计算当前元素的乘法逆元
        FieldElement inv() const
        {
            if (is_zero())
            {
                throw std::runtime_error("attempted to invert zero in FieldElement");
            }
            FieldElement out;
            // 调用 mcl::Fr::inv 计算逆元
            mcl::Fr::inv(out.value_, value_);
            return out;
        }

        // 将域元素转换为十进制字符串表示
        std::string to_string() const
        {
            return value_.getStr(10);
        }

        const mcl::Fr& native() const
        {
            return value_;
        }

        // 将域元素以小端字节序写入字节缓冲区，返回实际写入的字节数
        std::size_t write_little_endian(std::uint8_t* bytes, std::size_t size) const
        {
            if (bytes == nullptr)
            {
                throw std::runtime_error("FieldElement::write_little_endian received null output buffer");
            }
            std::memset(bytes, 0, size);
            return value_.getLittleEndian(bytes, size);
        }

    private:
        mcl::Fr value_;
    };

    // 一元负号运算符重载（非成员函数）：返回域元素的相反数
    inline FieldElement operator-(const FieldElement& value)
    {
        // 零减去 value 得到相反数
        return FieldElement::zero() - value;
    }

    inline std::ostream& operator<<(std::ostream& stream, const FieldElement& value)
    {
        stream << value.to_string();
        return stream;
    }

    // 批量计算多个域元素的乘法逆元，使用前缀积和后缀积的方法
    inline std::vector<FieldElement> batch_invert(const std::vector<FieldElement>& values)
    {
        std::vector<FieldElement> out(values.size(), FieldElement::zero());
        if (values.empty())
        {
            return out;
        }

        // 前缀积数组，存储从第一个元素到当前元素的累积乘积
        std::vector<mcl::Fr> prefixes(values.size());
        mcl::Fr accumulator = 1;
        for (std::size_t i = 0; i < values.size(); ++i)
        {
            if (values[i].is_zero())
            {
                throw std::runtime_error("batch_invert encountered zero");
            }
            prefixes[i] = accumulator;
            mcl::Fr::mul(accumulator, accumulator, values[i].native());
        }

        // 计算所有元素总乘积的逆元
        mcl::Fr suffix_inverse;
        mcl::Fr::inv(suffix_inverse, accumulator);
        for (std::size_t i = values.size(); i-- > 0;)
        {
            mcl::Fr inverse;
            mcl::Fr::mul(inverse, suffix_inverse, prefixes[i]);
            out[i] = FieldElement::from_native(inverse);
            mcl::Fr::mul(suffix_inverse, suffix_inverse, values[i].native());
        }
        return out;
    }

    // 计算大于等于给定值的最小 2 的幂
    inline std::size_t next_power_of_two(std::size_t value)
    {
        std::size_t out = 1;
        while (out < value)
        {
            out <<= 1U;
        }
        return out;
    }
}