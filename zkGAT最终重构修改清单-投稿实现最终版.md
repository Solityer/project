# zkGAT 最终代码重构修改清单

本文档用于交给 Codex 或工程实现者执行。目标是把当前项目整理为安全类期刊/会议可投稿的 zkGAT 主实验代码。最终主实验不保留 `seed -> tau`、`deterministic tau`、`test tau` 作为证明路径；所有最终结果必须来自 secure SRS + masked opening 主路径。


## A. 总体目标

1. 主实验路径只保留 `trusted_srs` 或 `mpc_srs`。
2. Prover 与 verifier 不得持有、读取、序列化、打印或由 seed 派生 KZG 陷门 `tau`。
3. KZG commitment 必须基于 SRS powers 或 Lagrange-basis SRS，而不是 `polynomial.evaluate(tau)`。
4. 私有 witness 多项式必须使用 domain-vanishing blinding：`P_zk = P + Z_D * R_P`。
5. Proof object 不得包含未声明为 public 的裸模型参数、裸中间激活、裸注意力分数、裸 softmax 权重、裸 lookup witness 或裸私有随机点评值。模型参数承诺必须使用隐藏承诺，不能公开裸参数或裸参数 opening。
6. 主实验 runner、benchmark exporter、README、论文表格只能读取 secure 主路径结果。
7. 所有安全测试和 negative tests 必须进入 CI 或最小回归脚本。


## A0. 阶段化执行顺序

Codex 必须按以下阶段执行，不得跳过前置验收：

### Phase 1：主路径清理与静态扫描

- 删除主路径中的 `tau` 字段、`seed -> tau`、`evaluate(tau)` 和旧 tau setup。
- 建立 CI 静态扫描脚本。
- 验收：主路径编译失败前不得继续下一阶段；扫描必须能拦截旧关键词。

### Phase 2：SRS manifest 与 key split

- 实现 `KZGProverKey` / `KZGVerifierKey`。
- 实现 `srs_manifest.json` 加载、hash 校验和 setup metadata gate。
- 验收：prover 不含 tau，verifier 只加载 VK 和 manifest。

### Phase 3：SRS-based commitment / opening

- 实现 coefficient-basis MSM commitment。
- 实现 Lagrange-basis MSM commitment。
- 实现 secure single-point / batch opening。
- 验收：commit/open 全部不调用 tau。

### Phase 4：ProofSchema 与 visibility registry

- 为全部多项式登记 domain、visibility、layer/head、opening policy。
- 实现 serializer tag 检查。
- 验收：unknown/duplicate/missing tag 全部拒绝。

### Phase 5：blinding 与 masked evaluation

- 实现 `BlindedPolynomial`。
- 实现 CSPRNG blinding。
- 改造 quotient、external evaluation、domain opening。
- 验收：`private_plain_value_count == 0`。

### Phase 6：prover / verifier / proof object 全链路改造

- 改造 context、prove、verify、proof size、manifest。
- 验收：Cora secure run verify pass。

### Phase 7：安全测试与泄露审计

- 完成 tamper tests、leakage audit、SRS mismatch tests。
- 验收：全部 tamper case 被拒绝。

### Phase 8：主实验、baseline、公平性核查与消融

- 重新跑五个数据集 CPU/GPU full。
- 重新导出 baseline proof-scope 表。
- 跑代表性消融与完整附录消融。
- 验收：所有结果来自 secure 主路径。

### Phase 9：README、论文表格和最终归档

- 顶层 README 只保留 secure 命令。
- 删除旧 tau/SRS 叙述和无关路径。
- 输出 paper-ready tables。

## B. 删除 / 禁用旧路径

### B1. 全局搜索并删除主路径中的可见 tau

搜索以下关键词：

- `tau =`
- `.tau`
- `key.tau`
- `evaluate(tau)`
- `evaluate(key.tau)`
- `seed + 17`
- `KZG::setup(seed)`
- `commit_tau_evaluation`
- `tau_evaluation`

处理规则：

1. `include/gatzk/crypto/kzg.hpp` 的主路径 key 结构不得包含 `tau`。
2. `src/crypto/kzg.cpp` 的主路径不得计算 `tau = FieldElement(seed + c)`。
3. `commit()`、`commit_batch()` 不得调用 `polynomial.evaluate(key.tau)`。
4. `commit_tau_evaluation()` 和 `commit_tau_evaluation_batch()` 必须从最终主分支删除；不得移动到 `tests/`、`dev_only/` 或任何最终提交路径。若需要历史对照，只能保留在私有分支。
5. 主 benchmark、正式 CLI、README 示例命令不得引用旧 setup。

### B2. 删除无关复现路径

用户要求最终代码不保留与主实验无关的复现路径。因此：

1. 删除或不纳入最终发布分支：旧 tau SRS runner、artifact_repro runner、debug-only benchmark、旧 README 复现实验路径。
2. 若开发阶段需要备份，只保留在私有 tag，不进入最终投稿代码。
3. 最终仓库顶层只保留 secure 主实验入口、baseline 对比入口、安全测试入口和结果导出入口。

## C. 新 SRS 与 KZG 接口设计

### C1. 新数据结构

新增或替换为以下结构：

```cpp
struct KZGProverKey {
    std::string srs_id;
    std::string srs_type; // trusted_srs or mpc_srs
    std::size_t degree_bound;
    std::vector<G1Point> powers_g1; // [tau^i]_1
    std::unordered_map<std::string, std::vector<G1Point>> lagrange_srs_by_domain;
};

struct KZGVerifierKey {
    std::string srs_id;
    std::string srs_type;
    G2Point g2_one;
    G2Point g2_tau;
    std::shared_ptr<PreparedG2> g2_one_prepared;
    std::shared_ptr<PreparedG2> g2_tau_prepared;
};

struct KZGKeys {
    KZGProverKey pk;
    KZGVerifierKey vk;
};
```

禁止字段：

```cpp
FieldElement tau;
FieldElement tau_evaluation;
```

若某些内部算法需要缓存评值，必须命名为 `claimed_value`、`masked_value`、`public_value`，不能再叫 `tau_evaluation`。

### C2. Setup 接口

替换：

```cpp
KZGKeyPair KZG::setup(std::uint64_t seed);
```

为：

```cpp
// 独立 SRS 生成工具使用，不链接进 prover/verifier/benchmark 主程序
SRSFiles generate_srs_offline(const SRSConfig& cfg);

// 主实验程序只允许加载 SRS 文件
KZGProverKey load_prover_srs(const std::string& srs_path);
KZGVerifierKey load_verifier_srs(const std::string& srs_path);
```

SRS 生成工具必须独立于 prover/verifier/benchmark 主程序；最终主实验程序只能加载 SRS 文件，不得现场生成、恢复或派生 tau。

`SRSConfig` 至少包含：

- `curve_id`
- `srs_id`
- `srs_type`
- `degree_bound`
- `domain_sizes`
- `ceremony_manifest_path` 或 `srs_path`

### C3. Commitment 接口

替换直接 tau-evaluation 承诺：

```cpp
Commitment KZG::commit(const std::string& name, const Polynomial& polynomial, const KZGKeyPair& key);
```

为：

```cpp
Commitment KZG::commit_coefficient_basis(
    const std::string& name,
    const Polynomial& coeff_poly,
    const KZGProverKey& pk);

Commitment KZG::commit_lagrange_basis(
    const std::string& name,
    const Polynomial& eval_poly,
    const std::string& domain_name,
    const KZGProverKey& pk);

std::vector<Commitment> KZG::commit_batch_secure(
    const std::vector<NamedPolynomial>& polys,
    const KZGProverKey& pk,
    CommitBatchProfile* profile = nullptr);
```

实现规则：

1. coefficient basis 使用 `multi_scalar_mul(p_i, pk.powers_g1[i])`。
2. evaluation basis 使用 `multi_scalar_mul(P(omega_i), pk.lagrange_srs_by_domain[domain][i])`。
3. 不允许退回 `P(tau)`。
4. 所有 commitment 的 `serialized_size()` 仍只统计承诺点，不统计 SRS。

### C4. Opening 接口

替换依赖 `key.tau` 的 opening 实现。新增：

```cpp
OpeningProof KZG::open_batch_secure(
    const std::vector<CommittedPolynomial>& polys,
    const std::vector<FieldElement>& points,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const FieldElement& folding_challenge,
    const KZGProverKey& pk);

bool KZG::verify_batch_secure(
    const std::vector<Commitment>& commitments,
    const std::vector<FieldElement>& points,
    const std::vector<std::vector<FieldElement>>& claimed_values,
    const FieldElement& folding_challenge,
    const OpeningProof& proof,
    const KZGVerifierKey& vk);
```

要求：

1. Prover 通过 quotient polynomial 或 equivalent SRS-based witness 生成 opening proof。
2. Verifier 只用 pairing equation 和 `VK_KZG` 验证。
3. Opening proof 不得依赖 verifier/prover 知道 tau。

## D. Zero-Knowledge blinding 改造

### D1. 多项式分类

在 protocol schema 中新增：

```cpp
enum class PolynomialVisibility {
    Public,
    PublicOutput,
    PrivateWitness
};
```

每个动态多项式必须登记 visibility。

默认规则：

- public：拓扑、selector、index、公开 metadata、公开最终 logits。
- private：模型参数相关、中间激活、attention score、lookup query/multiplicity、route query、PSQ、状态机、folding vector、所有非公开 witness。

### D2. BlindedPolynomial 对象

新增：

```cpp
struct BlindedPolynomial {
    std::string name;
    Polynomial semantic_poly;   // P
    Polynomial blinding_poly;   // R_P, may be zero for public
    Polynomial committed_poly;  // P + Z_D * R_P
    PolynomialVisibility visibility;
    std::string domain_name;
};
```

主证明路径只对 `committed_poly` 承诺与 opening。

模型参数也必须按 private witness 处理。模型参数承诺进入 `VK_model` 时必须是隐藏承诺；不得公开裸参数，不得为模型参数生成裸 opening，不得把模型参数随机点评值写入 proof。投影矩阵、注意力向量和偏置只能通过固定的 ZK binding 子证明参与验证。

### D3. Blinding 生成

新增 `ZKPolicy`：

```cpp
struct ZKPolicy {
    std::string privacy_policy;
    std::size_t blinding_degree;
    bool blind_private_polynomials;
    bool public_logits;
    bool public_labels_only;
};
```

实现要求：

1. 每个 private polynomial 使用独立 randomness。
2. randomness 来自 CSPRNG，不来自实验 seed。
3. randomness 不写入 proof、不写入 manifest、不写入日志。
4. `proof_size_bytes` 统计 blinding 带来的新增承诺/opening。

### D4. Eval_ext / Eval_dom 字段重构

将字段统一为：

```cpp
struct EvaluationValue {
    std::string label;
    FieldElement value;
    EvaluationKind kind; // PublicValue or MaskedPrivateValue
};
```

禁止 serializer 输出未标记的裸 `FieldElement`。

Verifier 规则：

1. 若 `visibility=PrivateWitness`，只接受 `MaskedPrivateValue`。
2. 若 `visibility=Public` 或 `PublicOutput`，可以接受 `PublicValue`。
3. 若字段标记与 `privacy_policy` 不一致，拒绝。

## E. ProtocolContext / Proof / Metadata 修改

### E1. ProtocolContext

当前结构中不要直接保存旧 `KZGKeyPair`。改为：

```cpp
struct ProtocolContext {
    AppConfig config;
    GraphDataset dataset;
    LocalGraph local;
    ModelParameters model;
    StaticTables tables;
    WorkDomains domains;
    KZGProverKey pk;
    KZGVerifierKey vk;
    ZKPolicy zk_policy;
    PublicMetadata public_metadata;
    ...
};
```

### E2. PublicMetadata 增加字段

必须增加：

- `proof_task`
- `srs_id`
- `srs_type`
- `srs_degree_bound`
- `privacy_policy`
- `input_privacy`
- `output_mode`
- `opening_mode`
- `binding_mode`

`canonical_public_metadata()` 必须吸入这些字段，并且顺序固定在 model/table/quant 之前。上述字段必须进入 Fiat--Shamir transcript；缺失、顺序错误、值不匹配或出现未声明字段时，verifier 必须拒绝。

### E3. Proof 对象重构

`Proof` 中将：

- `dynamic_commitments` 改为 `dynamic_commitments_zk`
- `quotient_commitments` 改为 `quotient_commitments_zk`
- `domain_openings` 改为 `domain_openings_zk`
- `external_evaluations` 改为 `external_evaluations_zk`
- `witness_scalars` 若无法证明不泄露，必须删除或改为 masked/public-tagged scalar

新增：

```cpp
std::vector<LeakageAuditEntry> leakage_audit;
```

或由 verifier 从 schema 生成，不一定放入 proof。

## F. Prover 修改

### F1. Build context

1. 加载 SRS：`trusted_srs` / `mpc_srs`。
2. 验证 `srs_id`、degree bound、domain sizes。
3. 初始化 `ZKPolicy`。
4. 禁止任何 seed 派生 tau。

### F2. Trace 生成

Trace 语义保持不变。GAT forward、hidden heads、output heads、PSQ、lookup、route 不要重写。

修改点：trace 输出后，统一进入 `materialize_blinded_polynomials()`：

```cpp
std::vector<BlindedPolynomial> materialize_blinded_polynomials(
    const TraceArtifacts& trace,
    const ZKPolicy& policy,
    const WorkDomains& domains);
```

### F3. Commitment

用 `KZG::commit_batch_secure()` 替换所有旧 `commit_batch()`。

### F4. Quotient

Quotient assembly 必须使用 committed/blinded polynomials。如果某个 quotient identity 因 blinding 产生 correction term，必须显式加入 quotient 计算，不得默默使用裸 polynomial。

### F5. Opening

全部 opening 使用 secure opening。外点和域内 opening 均不得打开裸 private polynomial。

## G. Verifier 修改

### G1. Metadata gate

验证：

- `protocol_name == zkgat`
- `proof_task == node_classification_full_gat`
- `srs_type in {trusted_srs, mpc_srs}`
- `srs_id` 与本地 VK 一致
- `privacy_policy == model_and_trace_hidden`
- `input_privacy == public_graph_public_features`
- `output_mode == public_logits`
- `opening_mode == domain_grouped_batch_opening`
- `binding_mode == zk_subproof_for_crpc`
- `domain_cfg/dim_cfg/encoding/padding_rule/degree_bound` 一致

### G2. Schema gate

严格检查：

- block order
- family length
- layer/head id
- output head count
- no extra fields
- no duplicate tags
- no unknown tags

### G3. Opening gate

使用 `verify_batch_secure()`；不得调用任何含 tau 的接口。

### G4. Leakage audit gate

新增：

```cpp
bool verify_no_private_plain_values(const Proof& proof, const ProofSchema& schema, const ZKPolicy& policy);
```

拒绝条件：

- private polynomial 的 `EvaluationKind` 是 `PublicValue`
- 出现模型参数裸值
- 出现完整中间矩阵
- 出现完整 attention alpha
- 出现完整 lookup witness
- 出现未声明 scalar

### G5. Negative tests

Verifier 必须对以下篡改拒绝：

1. final logits tamper
2. metadata tamper
3. srs_id tamper
4. srs_type tamper
5. privacy_policy tamper
6. commitment order tamper
7. dynamic commitment point tamper
8. quotient commitment tamper
9. domain opening value tamper
10. external masked value tamper
11. challenge tamper
12. edge order tamper
13. route query tamper
14. lookup query tamper
15. PSQ state tamper
16. output average relation tamper
17. hidden profile mismatch
18. K_out mismatch
19. proof_task mismatch
20. added unknown field

## H. CLI / Config / Runner 修改

### H1. Config

新增配置字段：

```ini
srs_type = trusted_srs
srs_id = <hash-or-manifest-id>
srs_path = artifacts/srs/<srs_id>
privacy_policy = model_and_trace_hidden
input_privacy = public_graph_public_features
output_mode = public_logits
opening_mode = domain_grouped_batch_opening
binding_mode = zk_subproof_for_crpc
proof_task = node_classification_full_gat
```

删除或禁止主实验字段：

```ini
seed = <used for tau>
allow_synthetic_model = true   # 主实验禁止
```

注意：普通随机种子仍可用于 benchmark order、sampling、CSPRNG test fixture，但不得用于 KZG trapdoor。

### H2. CLI

主命令建议：

```bash
./zkgat prove --config configs/cora_secure.cfg --srs artifacts/srs/main --out runs/cora_secure
./zkgat verify --proof runs/cora_secure/proof.bin --vk artifacts/srs/main/vk.bin
./zkgat bench --suite paper_secure --datasets all --devices cpu,gpu
./zkgat security-tests --suite tamper_all
```

### H3. Exporter

结果表必须含：

- dataset
- backend
- srs_type
- srs_id
- privacy_policy
- input_privacy
- output_mode
- opening_mode
- binding_mode
- proof_task
- commitment_time_ms
- prove_time_ms
- verify_time_ms
- proof_size_bytes
- verifier_reads_witness=false
- private_plain_value_count=0
- reference_consistency_passed=true
- tamper_tests_passed=true

所有论文结果只能从 `results/paper_secure/` 自动导出。缺少上述任一字段的 run 不得进入论文表格。导出器必须 hard fail，不允许只打印 warning 后继续生成论文表格。

## I. Baseline 对比修改

不需要把 ezkl、zkDL、zkGPT 改成 zkGAT 的 SRS。需要补 proof-scope 对齐表。

每个 baseline 必须记录：

- full_gat_forward_proved
- final_logits_bound
- full_dataset
- setup_assumption
- proof_size_definition
- verifier_reads_witness
- witness_included_in_proof
- public_input_layout
- task_type
- dataset
- model_arch
- model_params

对比表不允许只给时间和 proof size，而不说明 proof scope。

## J. 消融实验要求

主文保留代表性消融，附录保留完整消融。

CPU：

- cpu_full
- cpu_w_o_fast_msm
- cpu_w_o_parallel_fft
- cpu_w_o_fft_backend_upgrade
- cpu_w_o_fft_kernel_upgrade
- cpu_w_o_trace_layout_upgrade
- cpu_w_o_fast_verify_pairing

GPU：

- gpu_full
- gpu_w_o_cuda_hotspots
- gpu_w_o_fast_msm
- gpu_w_o_parallel_fft
- gpu_w_o_fft_backend_upgrade
- gpu_w_o_fft_kernel_upgrade
- gpu_w_o_trace_layout_upgrade
- gpu_w_o_fast_verify_pairing

所有消融必须基于 secure 主路径重新运行。工程优化不能改变 proof relation、transcript、proof schema、opening equation 或 leakage policy。

## K. 最终验收标准

代码通过以下检查后，才能写入论文主实验：

1. 全仓主路径无 `seed -> tau`。
2. Prover/verifier 主结构无 `tau` 字段。
3. `commit_batch_secure()` 不调用 `evaluate(tau)`。
4. Private polynomial 全部 blinding。
5. Proof object 中 private plain value count 为 0。
6. 五个数据集 CPU/GPU full 均 verify pass。
7. 所有 tamper tests 均 reject。
8. Baseline proof-scope 表完整。
9. 消融实验来自同一 secure codebase。
10. README 主命令可从干净 checkout 复现实验表。
11. `PublicMetadata` 必须包含并吸入 `input_privacy`、`output_mode`、`opening_mode`、`binding_mode`。
12. 导出器必须拒绝缺少 secure 字段或安全测试未通过的结果行。



## L. CI 静态扫描 gate

新增脚本：

```bash
python3 tools/security/static_scan_no_tau.py
```

扫描范围：

- `include/`
- `src/`
- `configs/`
- `scripts/`
- `comparison/` 中进入主实验的代码
- README 与 benchmark runner

禁止模式：

- `FieldElement tau`
- `.tau`
- `key.tau`
- `evaluate(tau`
- `evaluate(key.tau`
- `seed + 17`
- `KZG::setup(seed`
- `tau_evaluation`
- `commit_tau_evaluation`
- `deterministic_tau`
- `test_tau`
- `seeded_tau`

允许例外：无。最终主实验分支中不得保留 `seed -> tau`、`deterministic_tau`、`seeded_tau`、`test_tau` 或任何证明者可见陷门代码。若开发阶段需要历史备份，只能保留在私有分支，不能进入最终提交代码。

验收：CI 中任何禁止模式出现在主路径，构建失败。

## M. SRS manifest 与文件格式任务

新增目录格式：

```text
artifacts/srs/<srs_id>/
  srs_manifest.json
  pk_coef.bin
  pk_lagrange_FH.bin
  pk_lagrange_edge.bin
  pk_lagrange_in.bin
  pk_lagrange_d_h.bin
  pk_lagrange_cat.bin
  pk_lagrange_C.bin
  pk_lagrange_N.bin
  vk.bin
```

`SRSManifest` 结构必须包含：

```cpp
struct SRSManifest {
    std::string srs_id;
    std::string curve_id;
    std::string field_modulus_id;
    std::string srs_type; // trusted_srs or mpc_srs
    std::string ceremony_transcript_hash;
    std::size_t degree_bound;
    std::map<std::string, std::size_t> domain_sizes;
    std::string pk_coef_hash;
    std::map<std::string, std::string> pk_lagrange_hashes;
    std::string vk_hash;
    std::string encoding;
    std::string generator_version;
    std::string security_level;
};
```

加载规则：

1. hash mismatch 立即失败；
2. degree bound 小于协议要求立即失败；
3. domain size 不匹配立即失败；
4. `srs_type` 非 `trusted_srs/mpc_srs` 立即失败；
5. `srs_id` 必须写入 `PublicMetadata` 并进入 transcript。

## N. RNG 与 secret-handling

1. Blinding randomness 必须来自 CSPRNG，例如系统随机源或经审计的 DRBG。
2. 实验 seed 只能控制 dataset order、benchmark repetition、非安全采样；不得进入 SRS trapdoor 或 blinding randomness。
3. Blinding randomness 不得写入 proof、manifest、日志、异常消息或 debug dump。
4. 临时 secret buffer 使用后应清零；若库不支持清零，至少封装 `SecureBuffer` 并禁止复制。
5. panic / exception 中不得打印 secret scalar。
6. CI 增加日志扫描，确认不存在 `blinding`, `randomness`, `secret`, `tau` 的值输出。

## O. masked 外点评值绑定实现任务

主实验固定使用以下绑定方式，不允许 Codex 自行选择其他模式：

```cpp
enum class BindingMode {
    DomainQuotientMode,
    ZKSubproofMode,
    PublicOutputMode
};
```

固定规则：

1. CRPC 投影绑定使用 `ZKSubproofMode`。
2. 源注意力、目标注意力绑定使用 `ZKSubproofMode`。
3. 压缩特征绑定使用 `ZKSubproofMode`。
4. 拼接、lookup、route、PSQ、max、range、inverse、normalization、output average 使用 `DomainQuotientMode`。
5. 最终 logits 使用 `PublicOutputMode`。

每个 ZK binding 子证明必须显式记录绑定对象名称、来源承诺、折叠承诺、masked evaluation、挑战点和绑定模式。证明对象只允许携带辅助承诺、masked opening 与 KZG opening proof。Verifier 必须检查绑定模式与固定规则一致，并拒绝任何裸折叠向量、裸模型参数评值、裸中间激活或裸 `mu`。

验收：

- 每个 `Pi_bind` 都有 `binding_mode`。
- Verifier 拒绝未声明 binding mode 或与固定规则不一致的 proof。
- Private binding 不输出裸折叠向量、裸模型参数评值、裸中间激活或裸 `mu`。
- `ProofSchema` 中每个 external evaluation 都标记 `MaskedPrivateValue` 或 `PublicValue`。

## P. KZG opening 方程测试

新增 selftests：

1. `secure_kzg_single_open_selftest`
2. `secure_kzg_same_point_batch_open_selftest`
3. `secure_kzg_multi_point_batch_open_selftest`
4. `secure_lagrange_commit_open_selftest`
5. `secure_opening_tamper_selftest`

每个 selftest 必须验证：

- 正确 opening 通过；
- 篡改 claimed value 拒绝；
- 篡改 commitment 拒绝；
- 篡改 point 拒绝；
- 篡改 folding challenge 拒绝；
- 不存在 tau 访问。

## Q. reference forward 一致性测试

新增命令：

```bash
./zkgat reference-check --config configs/cora_secure.cfg --srs artifacts/srs/main --out runs/cora_secure
```

输出字段：

- `reference_fp32_logits_hash`
- `reference_quant_logits_hash`
- `secure_trace_logits_hash`
- `proof_public_output_hash`
- `reference_consistency_passed`
- `model_arch`
- `model_params`
- `quant_config`
- `static_tables`

验收：

- public logits 模式下三个量化相关 hash 必须一致；
- label-only 模式下必须额外验证 argmax proof；
- 不通过的 run 不进入论文结果。

## R. baseline 公平性核查

每个 baseline 结果必须补齐：

- same checkpoint
- same quantization
- same graph preprocessing
- same self-loop rule
- same edge sort rule
- same public output policy
- same full dataset / sampled subset flag
- verifier reads witness
- witness included in proof
- final logits binding
- full GAT forward coverage
- setup assumption
- proof size definition
- hardware and device

若任一字段缺失，baseline 只能进入 qualitative discussion，不能进入主对比表。

## S. 消融实验验收规则

每个 ablation run 必须满足：

1. verify pass；
2. proof relation 不变；
3. transcript schema 不变；
4. proof scope 不变；
5. `srs_id` 不变；
6. `privacy_policy` 不变；
7. `private_plain_value_count == 0`；
8. public metadata hash 只能因 backend flag 变化；
9. proof size definition 不变；
10. reference consistency pass。

未满足条件的 ablation 视为非法，不得进入正文或附录。

## T. README 与 paper branch 清理

最终投稿分支要求：

1. 顶层 README 只保留 secure 主实验命令、security tests、baseline 对比、结果导出。
2. 删除旧 tau SRS、artifact_repro、debug-only benchmark 的公开入口。
3. 删除或迁移所有旧结果表，避免论文误读。
4. `configs/*_secure.cfg` 是唯一主实验配置。
5. `results/paper_secure/` 是唯一论文结果来源。
6. README 必须说明：主实验不使用 `seed -> tau`，所有结果来自 secure SRS + masked opening。


## U. 与现有项目代码的对齐原则

1. 本次重构不是重写 GAT 模型层。现有多层多头 GAT 前向语义、隐藏层 ELU、隐藏层拼接、输出层注意力头、输出偏置、输出头平均、节点分类任务、边排序、route、PSQ、lookup 和量化规则都必须保持一致。
2. 主要改动集中在 crypto、protocol、proof schema、metadata、verifier gate、runner、security tests 和 result exporter。除非为了接入 secure SRS、masked opening 或 visibility registry，不得修改 GAT forward 的数学语义。
3. `src/model`、trace construction、route、PSQ、lookup 相关代码优先做适配层改造，不做语义替换。任何会改变 logits、attention 分组、输出平均或 public metadata hash 的改动，都必须先通过 reference forward 一致性测试。
4. 主实验固定为公开图结构和公开节点特征，隐藏模型参数和中间推理状态，公开最终 logits。Codex 不得把当前主实验改成隐藏输入图或隐藏节点特征版本。
5. ezkl、zkDL、zkGPT 仍作为外部 baseline。不要把 baseline 改写成 zkGAT SRS；只补 proof-scope、公平性字段、同 checkpoint、同量化、同预处理和同输出绑定检查。
6. 旧 `seed -> tau` 结果、旧 deterministic SRS 结果和 debug benchmark 结果不得进入最终论文表格。所有最终表格只允许由 secure 主路径重新导出。
