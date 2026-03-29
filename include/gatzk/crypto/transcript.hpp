#pragma once

#include <map>
#include <string>

#include "gatzk/algebra/field.hpp"
#include "gatzk/crypto/curve.hpp"

namespace gatzk::crypto {

class Transcript {
  public:
    explicit Transcript(std::string label);

    void absorb_scalar(const std::string& label, const algebra::FieldElement& value);
    void absorb_commitment(const std::string& label, const G1Point& point);
    algebra::FieldElement challenge(const std::string& label);

    const std::map<std::string, algebra::FieldElement>& issued_challenges() const;

  private:
    std::string state_;
    std::map<std::string, algebra::FieldElement> challenges_;
};

}  // namespace gatzk::crypto
