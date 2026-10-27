# Deviations from the testbed protocol

[`testbed/PROTOCOL.md`](../../testbed/PROTOCOL.md) says that deviations are listed in `DEVIATIONS.md`. That file did not exist at the freeze. Adding it to `testbed/` afterwards would change the frozen inputs, so the list is kept here. Additional file 1, Sect. S11.11, gives the same list.

## Amendment before the freeze (part of the protocol)

**A1** (21 October 2026, protocol Sect. 10) allowed one full repeat of calibration phase 3 at μ = 4 requests per second.

- The first attempt at μ = 4 failed one of 16 checks; the repeat passed all 16.
- Both attempts are in the data archive.
- μ = 4 is an operational working scale, not a proven maximum.

## Deviations

1. **Guaranteed CPU.**
   - *Protocol (Sect. 3):* workers with a guaranteed CPU share of 100%.
   - *Actual:* the provider's standard STD3-4-8 nodes do not confirm dedicated vCPUs. Steal time since boot was at most 0.12% (node snapshot), which does not exclude short pauses.
2. **Identical workers.**
   - The four workers had the same type and size, but their measured speed differed:
     - w4 was about 16–19% faster, by CPU time per request and by the single-thread benchmark;
     - the CPU time per request on w2 changed over time.
   - The rotation of policies over workers balances a constant worker effect, but not these changes.
3. **Deviations file.** The protocol refers to `DEVIATIONS.md` inside the frozen folder. It was not created before the freeze; this file replaces it, and the frozen inputs are unchanged.
4. **Provider and environment.** The protocol does not fix them; they are documented in [`README.md`](README.md#cluster). VK Cloud was chosen after two other providers could not supply the requested nodes.
5. **Retrieval of results.**
   - The bulk copy (`kubectl cp` in `testbed/ops/cloud_stage.sh`) broke off in runs 1–3.
   - `ops/finish_fetch.sh`, outside the frozen inputs, retrieved the results with per-file SHA-256 verification against the driver pod (`results/run*/sha256_driver.txt`).
   - Measurements and analysis were not changed.
6. **Attribution of one removed pod.** A derived quantity was corrected after the freeze ([`README.md`](README.md#corrections-after-the-freeze)). K1–K3 (protocol H1–H3) are unaffected.
7. **Diagnostics not planned in the protocol.**
   - The node snapshot (`ops/node_snapshot.sh`).
   - Tables X01–X06 and figures F08, F10, and F11.
   - They are descriptive and exclude no request from the protocol's measures.
