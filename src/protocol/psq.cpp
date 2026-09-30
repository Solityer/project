#include "gatzk/protocol/psq.hpp"

#include <stdexcept>

namespace gatzk::protocol
{
    namespace
    {
        std::vector<algebra::FieldElement> build_selector_prefix_state(const std::vector<algebra::FieldElement> &values,
                                                                       const std::vector<algebra::FieldElement> &q_new,
                                                                       bool enable_single_graph_selector_shortcut)
        {
            if (values.size() != q_new.size())
            {
                throw std::runtime_error("prefix state input size mismatch");
            }
            if (values.empty())
            {
                return {};
            }
            std::vector<algebra::FieldElement> out(values.size(), algebra::FieldElement::zero());
            out[0] = values[0];
            const auto one = algebra::FieldElement::one();
            for (std::size_t i = 1; i < values.size(); ++i)
            {
                const auto &selector = q_new[i];
                const auto &value = values[i];
                if (enable_single_graph_selector_shortcut)
                {
                    if (selector.is_zero())
                    {
                        out[i] = out[i - 1] + value;
                        continue;
                    }
                    if (selector == one)
                    {
                        out[i] = value;
                        continue;
                    }
                }
                out[i] = selector * value + (one - selector) * (out[i - 1] + value);
            }
            return out;
        }

    }

    std::vector<algebra::FieldElement> build_group_prefix_state(const std::vector<algebra::FieldElement> &values,
                                                                const std::vector<algebra::FieldElement> &q_new, bool enable_single_graph_selector_shortcut)
    {
        return build_selector_prefix_state(values, q_new, enable_single_graph_selector_shortcut);
    }

    std::vector<algebra::FieldElement> build_max_counter_state(const std::vector<algebra::FieldElement> &s_max, const std::vector<algebra::FieldElement> &q_new,
                                                               bool enable_single_graph_selector_shortcut)
    {
        if (s_max.size() != q_new.size())
        {
            throw std::runtime_error("max counter input size mismatch");
        }
        return build_selector_prefix_state(s_max, q_new, enable_single_graph_selector_shortcut);
    }
}
