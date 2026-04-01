# Multi-Head Migration Plan

## Current delta
- `src/model/gat.cpp` is already aligned to the official single-layer GAT forward: hidden `8` heads, output `1` head, self-loop participation, hidden-head ELU, concat to `d_cat=64`, output attention head with identity activation.
- `src/protocol/trace.cpp`, `src/protocol/quotients.cpp`, and `src/protocol/verifier.cpp` still contain legacy single-head witness and quotient wiring such as `P_H_prime`, `P_E_src`, `P_Y_lin`, `P_Y`, `t_FH`, `t_edge`, `t_in`, `t_d`, `t_N`.
- `src/protocol/challenges.cpp` already has multi-head label scaffolding, but `replay_challenges()` still hard-stops on real multi-head formal context.

## File-level migration route
1. `src/protocol/trace.cpp`
- Replace the single-head witness namespace with per-hidden-head objects plus explicit concat/output objects.
- Materialize `H_cat`, `H_cat_star`, `H_C`, `Y'_star`, `Y'_star_edge`, `widehat_y_star`, `Y_star`, `Y_star_edge`, `PSQ_out`.

2. `src/protocol/quotients.cpp`
- Split the legacy five-domain quotient wiring into the seven-domain note layout.
- Add `t_cat` and `t_C`; route hidden heads to `H_in/H_d_h`, concat to `H_cat`, output head to `H_C`.

3. `src/protocol/challenges.cpp`
- Remove the legacy single-head replay guard.
- Rebuild the Fiat-Shamir transcript strictly in the note order for hidden heads, concat stage, and output stage.

4. `src/protocol/verifier.cpp`
- Stop replaying legacy single-head bundles.
- Check `M_pub`, block order, seven-domain openings, quotient identities, and output-stage bindings under the note layout.

5. `src/protocol/prover.cpp`
- Keep `M_pub` and fixed proof block order in sync with the final multi-head formal objects.
- Export formal artifacts for `cat` and `C` domains once trace generation is upgraded.

## Risks
- The current formal `cora_full` run still aborts in `build_trace()` because the multi-head trace object system is incomplete.
- The verifier cannot accept a real multi-head proof until `replay_challenges()` and quotient evaluation stop depending on the legacy single-head names.
