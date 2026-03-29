# Project Goal

正式主线只维护以下目标：

1. 以 `GAT-ZKML_终稿_修订版.md` 作为唯一协议规范；
2. 以全量图作为正式运行对象：
   - `cora`
   - `citeseer`
   - `pubmed`
3. CPU 路径必须支持 `build -> trace -> prove -> verify -> benchmark` 全闭环；
4. 保留可继续迭代的 GPU backend 接入点，但不改动 pairing / verifier / mcl 曲线主路径；
5. 主入口配置固定为：
   - `configs/cora_full.cfg`
   - `configs/citeseer_full.cfg`
   - `configs/pubmed_full.cfg`

不再把 `toy / smoke / cora_benchmark` 视为正式入口，也不再以这些口径维护主文档和主命令。
