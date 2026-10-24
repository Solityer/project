# Latest PPI failure and scoped stop evidence

- Timestamp observed: 2026-10-22 16:19 Asia/Shanghai.
- Launch: `94e69096789b489daa73f9a60dafda44` (`--datasets ppi ogbn_arxiv`).
- Session: `fba2a7ab38486b6549b7aac7a351e0f8`.
- Supervisor PID/PGID: `2872709`; start ticks: `7204011`.
- Runner PID/PGID: `2872712`; start ticks: `7204016`.
- Active OGBN prover PID/PGID: `2927230`; command/cwd recorded from `/proc` and `ps` before stop. At observation: elapsed about 12,895 s, RSS about 32.9 GB, CPU 92.8%.
- PPI status: `FAILED`; no proof generated, no verifier called. Native proof elapsed 12,821,774 ms before failing.
- PPI error: native masked GKR proof rejected because native scratch allocated blocks exceeded live-prefix bound after reclamation; allocated `147095588864`, bound `147095547904` (40,960-byte difference), active `147095538208`, page_live/file `147095538240`, tail `2624`.
- Runner continued the selected queue to `ogbn_arxiv`; OGBN had reached native proving and was actively consuming CPU. Its then-current stderr and session/launch state are copied above.
- Resource observation before stop: filesystem `/` had about 1.2 TiB free; memory 121 GiB total, 98 GiB available, 15 GiB swap with 44 MiB used.
- Original logs and result artifacts remain in place under `baseline-local/runs/formal/<session>/` and `baseline-local/formal-detached/<launch>/`.

- Stop command: `python3 scripts/stop_formal.py --launch-id 94e69096789b489daa73f9a60dafda44`.
- Stop result: `ZKGPT_FORMAL_STOP_COMPLETE`, launch status `INTERRUPTED`, supervisor received SIGTERM, runner exit code 130.
- Post-stop `ps -p 2872709,2872712,2927230` returned no process rows; launch/session status snapshots after stop are preserved.
