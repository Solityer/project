# zkGPT PPI stalled launch evidence

Captured: 2026-10-21 13:42 CST (host clock; Asia/Shanghai)
Scope: read-only inspection of the user's detached zkGPT formal launch. No source edits, tests, dataset run, or scratch copy.

## Launch and process state

- launch_id: `8ac696be4d4f471d9f6dc8be72501d04`
- session_id: `369464a5dc5de9442f551459b92f9f12`
- requested datasets: `ppi`, `ogbn_arxiv`, serially
- launch metadata records supervisor PID 2060845 / start ticks 47977582 and child PID 2060848 / start ticks 47977586, both with cwd `/home/pzh/project/comparison/zkGPT`.
- At capture, exact `ps -p 2060845,2060848` returned no processes. A process-table search for zkGPT formal launch/prover names found no live zkGPT run, and the only launch directory on disk is the launch above.
- `status.json` remains stale at `RUNNING`; `session_status.json` remains `RUNNING`, `active_stage=cell:ppi`, with only `prepare` completed and no recorded failure.
- No signal was sent: neither recorded process identity exists now. The existing `stop_formal.py` validates the exact live supervisor and child identity before using pidfd SIGTERM, so it cannot safely signal this stale record. Status files were left untouched.

## Last recorded PPI stage

The complete console has 153 lines. Its final line, also the final line of the PPI prover stderr, is:

```
[local-cell-ppi] [ppi/prover] gat_adapter_prover_stage: native-initSubset complete elapsed_ms=2.76565e+06
```

The immediately preceding PPI prover records are two layer-112 GKR shape-census progress lines at 15,004 ms and 30,006 ms, then `native-initSubset begin`. No subsequent PPI stage line was recorded. In particular, there is no visible canonical-validation or Lasso stage marker after initSubset; absent logging alone does not establish whether an unlogged internal operation began.

- console last-modified: 2026-10-20 23:20:00 CST
- PPI prover stderr last-modified: 2026-10-20 23:20:00 CST; 392 bytes
- PPI prover stdout: empty
- PPI run directory: 24 KB, containing registration metadata/secret and prover logs only
- No PPI proof, verifier receipt, result, or exit-code artifact is present in that run directory. OGBN-Arxiv did not begin; session active stage remained PPI.
- Existing logs remain at the paths listed below; this snapshot indexes them and does not copy any large files.

## Source and build identity

Session source identity:
- native/source-tree SHA-256: `78dd3a11a0cff4e7794cd1f31aef2490e0ad144d44f3dcd189a3e484e9fbb2dc`
- runtime source SHA-256: `a31635dbca153c4cfd94b8a4629f44ef26a0e2429f081af7e1ad36e55b384421`

The current source-tree/runtime hashes match those session values. The recorded CPU prover SHA-256 `dc45f7575176f0821ca7901f06459aa74d8ab834762c344c120f668c80eca349` and verifier SHA-256 `c04d5ec6a6baa7050f99f9bb80227a0f86e6a7f3e0ed166a8f28b73259fb0062` match both the session's preflight fingerprints and current `build/cpu/build_provenance.json`. Thus the preserved session and present build report the same source and binary identities.

## Capacity and current host snapshot

Immediately before PPI, the session's local resource recheck recorded:
- host available 112,538,566,656 B; estimated required 24,959,661,043 B
- filesystem available 1,660,145,020,928 B; estimated required 854,734,389,804 B
- PPI capacity admission was READY; its estimated disk peak was about 854.65 GB.

At capture (after the recorded process had disappeared):
- `free -b`: 121,823,707,136 B available RAM; swap unused
- `vmstat 1 2`: final one-second sample about 10% user, 0% system, 89% idle, 0% I/O wait
- `df -h /`: 1.6 T available
- `du -sh` of the small persistent PPI run directory: 24 KB. Transient/unlinked scratch from the old process is no longer measurable; no recursive scan of large scratch was performed.

## Boot journal evidence and limits

`journalctl --list-boots` records the previous boot's last entry at 2026-10-20 23:30 CST and current boot beginning 2026-10-21 12:28 CST. The previous-boot kernel journal around 23:23 contains NVIDIA driver GPU-memory allocation failures (`NV_ERR_NO_MEMORY`). The inspected interval does not record a host-kernel OOM kill or an explicit shutdown/reboot cause. This timing does not establish that the GPU errors caused the zkGPT process loss or the boot transition.

## Original log and metadata paths

- Detached launch record: `baseline-local/formal-detached/8ac696be4d4f471d9f6dc8be72501d04/status.json`
- Full launch console: `baseline-local/formal-detached/8ac696be4d4f471d9f6dc8be72501d04/console.log`
- Session status/context: `results/formal/369464a5dc5de9442f551459b92f9f12/session_status.json`, `local_formal_context.json`, `session.json`
- PPI cell logs: `baseline-local/formal-owner-logs/369464a5dc5de9442f551459b92f9f12/cell-ppi.stderr.log` and `cell-ppi.stdout.log`
- Native run logs: `baseline-local/runs/formal/369464a5dc5de9442f551459b92f9f12/ppi/prover.stderr.log`, `prover.stdout.log`
- Build provenance: `build/cpu/build_provenance.json`

