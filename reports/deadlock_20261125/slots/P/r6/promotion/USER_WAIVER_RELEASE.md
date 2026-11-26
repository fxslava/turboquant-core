# P/r6 release under the user timing waiver

The P/r6 modern guarded result remains **63,649 raw ticks**, **20 above the 63,629 cap**. The release receipt records `raw_cap_met=false`, no normalization, and the user's waiver of this exact modern delta. The historical results keep their original verdicts. P/r7 stays isolated.

## Evidence

The root operator captured the exact user request in `finalize_p6_20261126/USER_WAIVER.json`, SHA-256 `ca0e507c5a16a7f76788f8a2c654b6416fad89f847b5f997969d24b20e2bafd9`. Every artifact reference uses its absolute real WSL path, byte count and SHA-256 as `{path, bytes, sha256}`.

| Waiver field | Value |
| --- | --- |
| `schema`, `status`, `candidate` | `p6-user-waiver-v1`, `EXPLICIT_USER_AUTHORIZATION`, `P/r6` |
| `source_manifest_sha256` | `da54b14f843583cdd0c8e262bc799430afeb89b59406e3aeb3e0deecc34a0064` |
| `compiled_index_sha256` | `d57fb0c27ebc63b8615583c83426ca1e852d8a45d4a6a48a0a7bd945788e5875` |
| `measurement` | Exact modern matched run `result.json`, SHA-256 `c301c5e4d7117f810b3beaa57eb6d46ab1538a4f103f15f25133c02f4d9dbcd6` |
| `modern_audit` | Exact modern matched trace audit, SHA-256 `3f30c2255871d617bf3a825f796deb8f4d1d040c16eaa17235e60d1e1fbef330` |
| `original_cap`, `observed_raw_ticks`, `waived_delta_ticks` | Integers 63629, 63649, 20 |
| `normalization`, `user_request` | `NONE`, literal user authorization |

Use the existing four `functional` case records with their exact R11 results, trace audits and `parity_guards_and_trace=PASS`. The matched case uses the same result and trace audit pinned by the waiver. The release envelope has `status=PASS_P6_USER_WAIVED_RELEASE`, `candidate=P/r6` and the same source and compiled SHA fields as the waiver. Compute the required `performance` object using `promote_p6_user_waiver.accepted_performance(waiver_ref, matched_ref)`.

The fresh independent Sol review must cover that waiver, all four actual functional records, and the exact bytes to deploy. In `winning_review`, include `status=PASS`, `review_kind=fresh_independent_sol_final`, the source and compiled SHA fields, `deployed_bytes_plan_sha256` equal to `08185028c3c4af7ef474f0f739e07ffafb6958d8f0fa091dced3686d8cf7b3e7`, `adapter` as a full reference to the new script, and `receipt` as a full reference to the fresh review. Include `functional_sha256` and `performance_sha256` using the script's `digest_value` helper, which hashes key-sorted compact JSON.

## Apply and validate

```powershell
$p = '/mnt/c/Ascend/turboquant-core/reports/deadlock_20261125/slots/P/r6/promotion'
$out = '/home/dmin/work/deadlock_20261125/slotP/r6/promotion-final-r1'
wsl -d Ubuntu-22.04 -u dmin -- python3 -B "$p/promote_p6_user_waiver.py" verify-release --winning-release $release --winning-release-sha256 $release_sha256
wsl -d Ubuntu-22.04 -u dmin -- python3 -B "$p/promote_p6_user_waiver.py" apply --winning-release $release --winning-release-sha256 $release_sha256 --output $out
```

Apply reconstructs the original seven core and eight mirror entries and verifies the original byte ledger. It writes and checks the shared source mirror before the deferred consumer patch files. The original CMake patch adds the wide translation unit to the existing `vllm_ascend_turboquant` target. The syntax and analyzer checkers enumerate it alongside narrow. The README uses both sources in its standalone command. The existing CMake target and native install component stay the same.

`setup.py` already consumes the CMake native install component and resulting native manifest for both wheel and editable installation. The existing modern ASC build emits a depfile for each translation unit. Both retained P/r2 depfiles list the unchanged shared `routine_reshape_batch_pipeline.hpp`. The deferred CMake target adds the wide source to that same compiler path. The `setup.py` tracked SHA-256 remains `27032a2cae757dba5b2a473cc26285004f0abc91ab0154c44b3f7f641edae75a`. The existing Windows host runner exercises `setup.py` through `tests/host/turboquant/test_build_selection.py`. The focused gate can run as follows inside the vLLM checkout.

```powershell
python -B -m pytest --noconftest -q -p no:cacheprovider tests/host/turboquant/test_build_selection.py
```

Run the sealed post-promotion host and deployed CANN syntax and AKA plans against the emitted `PROMOTION_APPLIED.json`. Commit only the approved product paths after those gates, with DCO sign-off.
