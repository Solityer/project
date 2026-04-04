#include "gatzk/protocol/schema.hpp"

namespace gatzk::protocol {

const std::vector<std::string>& proof_block_order() {
    static const std::vector<std::string> order = {
        "M_pub",
        "Com_dyn",
        "S_route",
        "Eval_ext",
        "Eval_dom",
        "Com_quot",
        "Open_dom",
        "W_ext",
        "Pi_bind",
    };
    return order;
}

}  // namespace gatzk::protocol
