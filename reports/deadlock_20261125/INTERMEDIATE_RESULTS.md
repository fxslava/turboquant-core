# TurboQuant intermediate results — 2026-11-25

Native decode G passed the device canaries and three fresh compiled reviews, followed by a separate measured-result review. Its comment-free export is now synchronized into both canonical trees. Batch reshape performance remains a deployment blocker. This is an intermediate report, not deployment approval.

| Gate | Result |
| --- | --- |
| Baseline checkpoints | Core `9822714d7ebeebc9d2b2a9907fc1aea3d90d1fe1`; vLLM `964190003f9e29e590b957cc7ef84e9c099cbf36` |
| Native exact `208×6` | G **33,514** raw ticks; unchanged B 34,700; historical champion 34,701; exact BF16 parity |
| Native ragged `4×1`, three splits | G **51,925** raw ticks; all 1,024 BF16 words exact. Both initial candidates failed this case by 125 ULP |
| Native ragged `2×3`, two splits | G **53,242** raw ticks; B 53,771; all 1,536 BF16 words exact |
| Native raw-query, grid 14 | G **50,827** raw ticks; B 51,454; all 4,608 BF16 words and all 4,608 FP32 query scratch words exact |
| Native runtime safety | All four G cases passed input/workspace/guard checks and synchronization accounting, without exceptions |
| Hang reproduction | No 900-second hang reproduced. The original remote incident ELF and logs are unavailable; its `+0x2270` cannot be attributed using a different local ELF |
| Batch current repeat | **64,460** raw ticks; first run 64,459; numerical and guard checks passed |
| Exact Frozen03 matched replay | **63,654** raw ticks; historical raw result 63,629. Every role trace matches the archive after a constant 13-tick shift; another 12 ticks occur after the final role frame |
| Batch performance | Current writer is **806 ticks** behind the exact Frozen03 image under the matched guarded fixture. No normalization or waiver has been applied |
| Analyzer | Zero new diagnostics against independent baselines; inherited diagnostics and call-closure limitations remain recorded |
| Source synchronization | Production checker passes: **34 C++ files + one analyzer configuration** have matching inventories and byte hashes |
| Isolated batch P/r2 | Both compiler paths and three fresh compiled reviews passed. The first six-tile model passed exact parity and guards at **65,266 ticks**, failing the performance gate. The remaining three probes were withheld |
| Isolated batch P/r3 | The partial narrow device ELF matches Frozen03 **byte for byte**. Fresh review rejected the split on generated host cleanup-state and unused-companion teardown defects before any model run |
| Isolated batch P/r5 | Passed 22 lifecycle and 36 dispatch CPU cases plus three reviews. Real startup stopped before any kernel because a private constructor queried an uninitialized SDK cache |
| Current isolated batch P/r6 | Passed three fresh compiled reviews and real SDK startup. The six-tile primary passed exact K/V/scales, input and guard checks at **63,649 raw ticks** (191.53 seconds): **20 ticks above the historical cap**, five below the matched modern Frozen03 replay. Remaining three canaries are withheld. Planned shared inventory remains 36 C++ files + one analyzer configuration |
| Host validation before final promotion | **905 disjoint pytest cases + 9 CTest cases passed** |
| Final synchronized native gate | **606 + 14 pytest, 7 vLLM C++ checks, 9 core CTests, five strict syntax checks passed**; fresh real-Torch host library verified schemas, Meta/FakeTensor and CPU tiling with zero device traps |

The winning decode change disables the automatic GEMV interpretation for the existing NZ-staged attention operands. Public entry signatures, tiling, workspace and allocation contracts are unchanged. The proposed zero-GM fill replacement was rejected; the suspected barrier and Fill-order defects were not confirmed.

The verified E8 reader/dequantizer integration preserves the external 4,352-byte plane and internal 1,056-byte padded stride. The unverified E8 encoder remains excluded. No full NPU device-plugin build has been attempted.

The first isolated batch restoration candidate finished compiling but was rejected before CAModel. CANN 9.2.1 generated MIX type/ratio metadata for an orphaned name of its second entry in both layouts. CPU dispatch tests could not cover that registration defect. P/r2 separated the source groups and passed both simulator and modern ASC builds, all three compiled reviews, and the first model's full numerical and safety checks. Its raw 65,266 ticks exceeded the matched Frozen03 result by 1,612 ticks inside role execution. The entire narrow Vector image is byte-identical to Frozen03 but relocated by 68,352 bytes; relocation causality is not yet proven. P/r2 is rejected on performance and has not been promoted.

P/r3 separated the simulator modules and produced an entire narrow device ELF identical to Frozen03. Its compiler build was stopped after detecting duplicate dependency goals, and fresh review found generated host cleanup symbols that could coalesce across libraries plus missing cleanup for an unused companion. No P/r3 model ran. P/r4 retains the exact device sources and repairs simulator-only symbol and registration ownership with ordinary linker exports and a typed host cleanup adapter. All lifecycle cases must pass against the real generated host wrappers using a CPU fake backend before new model admission. No SDK, generated-source, or device ELF patch is permitted. No Wave-2/Wave-3 reshape mutant has been promoted.

The installed profiler contract permits delayed callbacks after unregister, so P/r5 made simulator modules explicitly resident for process lifetime, with work quiesced before exit. All 22 lifecycle and 36 dispatch controls passed, including negative controls and staged installation. Its real startup exposed a separate loader-order fault before any device launch: the private module queried the runtime SoC string before the runtime constructor initialized it. The retained core and two independent audits identified the exact storage and call chain. P/r6 adds ordinary explicit SDK-provider dependencies without changing kernel or serving inputs. A guarded real startup-only debugger run reached the first instruction of main with distinct registered handles and actual downstream ACL pointers resolving to CAModel; it then stopped and reaped the process tree before any kernel launch. Three fresh compiled reviews passed. The six-tile primary completed naturally in 191.53 seconds with exact parity, guards and event retirement. All 766,265 historical role-trace lines match after an eight-tick shift; a further 12 ticks occur after role retirement. This explains the measured 20-tick difference but does not waive the literal cap. An unchanged original-host comparison has been prepared and is undergoing independent operational review; neither historical comparison probe has run. No tick subtraction or normalization is authorized.

The vLLM cache-spec class consolidation and six regression cases are committed as `e19e9ecbe`. Native G source changes are synchronized and committed as core `d6c63d8` and vLLM `9a0c30f8f`. Batch performance, the final report and clean task-owned working trees remain outstanding.

Exact mandatory host commands:

```powershell
cd C:\vllm-ascend
python tests/ut/attention/run_turboquant_device_free.py -q
python -m pytest --noconftest -q tests/host/turboquant/test_source_sync.py tests/host/turboquant/test_build_selection.py
python scripts/check_turboquant_sync.py --core C:/Ascend/turboquant-core
```

Measured runtime receipts are indexed in `G_RUNTIME_AUDIT_INDEX.json`; batch matched controls are recorded in `d1/ingress/FROZEN03_ARCHIVE_REPLAY_AUDIT.json` and `d1/ingress/FROZEN03_CURRENT_REPEAT_AUDIT.json`. Device canaries stayed within six total live tiles per launch and used the 900-second watchdog.

Latest performance evidence: [P/r6 batch timing audit](slots/P/r6/canary/P6_BATCH_TIMING_AUDIT.json). The original-host comparison is performance-only and does not replace the guarded numerical canary. Its reference must reproduce raw 63,629 before the candidate can run.

The fresh R12 operational review rejected the optional comparison runner for acceptance-logic and evidence-validation gaps. Its artifact, six-tile, locking and watchdog controls passed. R13 is being prepared separately; no optional comparison model has run and no kernel has changed.

The collection-only R16 protocol passed independent operational review. Its actual unmodified-host Frozen03 control completed naturally in 199.54 seconds, child exit 0, process tree drained, and reported **63,630 raw ticks**, one above the archived 63,629. The candidate has not run under the old host. A new independent actual-reference/methodology review is checking the complete 428-reference collection and exact archival trace comparison. The exact archived-counter reproduction gate remains unmet; no counter normalization or candidate promotion has occurred.

The actual original-host P/r6 candidate completed naturally in **193.64 seconds**, exit 0 and drained, at **63,637 raw ticks**. This fails both the absolute 63,629 cap (+8) and the measured 63,630 control (+7). No promotion or supplemental probe was admitted. The fresh independent pair review rejected an architectural exception: all captured event bodies and spans match, but the necessary cause of the extra boundary ticks is unproven. Both raw timing failures remain recorded; P6 is rejected for promotion. Preliminary alignment places all added ticks before or after an otherwise identical captured role schedule.

Latest P7 progress: the isolated candidate changes only `cmake/SimTargets.cmake` and adds `cmake/LinkSimPrefillMix.py.in`; 165 other source files are unchanged. All 167 hashes verified in both Windows and WSL. A normal link-only probe restores the Frozen03 narrow entry/VF addresses; unchanged Lloyd constants and memory arenas match the prior combined image. Fresh retained-object CMake merge/finalize/host/pack steps completed successfully in 6.90 seconds, without BiSheng or a full plugin build. CPU controls on the actual generated wrappers passed 36 dispatch-boundary and 16 lifecycle cases. Three fresh Sol 6.2 compiled reviewers are reading the immutable bundle, with final verdicts sequenced. P7 has no runtime parity or timing result yet and is not admitted for simulation or promotion. Canonical shared sources remain 34 C++ files plus one analyzer configuration.

Paused at the user's request after all agents finished. All three P7 compiled reviews and the R17 collection-only review passed. P7 runtime remains unrun and unadmitted; the guarded binder is prepared but incomplete. No active jobs remain. See `PAUSE_CHECKPOINT.json` and `RESUME.md`.

Finalized P6 under the explicit user timing waiver on 2026-11-26: all four runtime cases passed, fresh Sol release PASS, 36 C++ plus AKA synced, final606+14+7Cpp+9CTest/5Torchsyntax/30CANNsyntax passed and analyzer zero-new. See finalize_p6_20261126/FINAL_HANDOFF.md. P7 remains isolated.
