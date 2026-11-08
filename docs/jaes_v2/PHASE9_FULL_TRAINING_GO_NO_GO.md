# Phase 9 — Full training decision

**NO-GO for full foundation.** The two Phase 8 runs completed as historical exploratory baselines. Phase 9 has a small arithmetic probe and audited data but lacks a frozen engine, end-to-end validation parity, 100-update resource evidence and replicated guard-feasible benefit. C2 remains blocked on identifiable cue-survival data.

| Measured item | Evidence |
| --- | --- |
| GPU | RTX 3050 Laptop, 4,096 MiB VRAM, driver 595.71. GPU execution for the TF2.15 training contract is unverified. |
| Disk | Snapshot C 551,919,616 bytes, D 23,301,349,376 bytes. Keep at least 15 GiB free on D. C/pagefile pressure previously caused process launch failure. |
| Short bounded probe | 35.62 s, 687.14 MiB process peak, 17.45 s startup/dataset phase. This is two single-view gradient cases with fresh Adam, not an optimizer throughput benchmark. |
| Historical final-invocation checkpoint storage | ERM 450,937,590 bytes, channel 1,090,721,524 bytes. Checkpoint retention differed due to interrupted invocations. |
| Historical channel final invocation | Three updates took 24.81 s. A linear 100/500-update scenario at this isolated rate gives about 14/69 min of updates alone. No claim about sustained throughput, validation, RAM peak or total job runtime. |

Medium 640 yields 40 updates/epoch, medium-plus 2,560 yields 160, and natural full 25,380 yields 1,587 with batch 16. At the isolated channel rate, update-only scenarios are about 1.9 h/7.7 h/76.6 h for 21 epochs respectively. Epoch-level validation, restart overhead, data I/O and shorter early stops could materially change totals. Foundation natural sampling is imbalanced and changes the objective exposure relative to balanced medium. A storage preflight is a projection from checkpoint cadence, separate from resource approval.

Next resource evidence, after the author restores C/pagefile headroom and engineering parity passes, is one fresh 100-update benchmark that includes paired preprocessing, validation, checkpoint export and resume. Extend to 500 only if peak commit/RSS, throughput, free disk and restart behavior are stable. Log process RSS, available system commit, GPU allocation if applicable, bytes per checkpoint and timing components. Set precommitted hard stops for D reserve, available system commit, memory, stall and elapsed time. Recheck versions and hashes before each start. Keep failed pilot artifacts and stop on OOM.
