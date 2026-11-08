# Reference results

Jolt and QAFold expose different statement-hiding guarantees; see README.md. Setup is separate. Jolt values come from rounded log tables. No averaging or new timings.

| Backend | Case | Prover s | Verifier s | Peak GiB |
|---|---|---:|---:|---:|
| Jolt | 8B_8bit | 94.007 | 0.4320 | 16.11 |
| Jolt | 8B_16bit | 181.752 | 0.4270 | 31.22 |
| Jolt | 70B_8bit | 643.676 | 0.4500 | 132.84 |
| Jolt | 70B_16bit | 1054.300 | 0.4500 | 265.01 |
| QAFold | 8b8 | 40.673 | 0.1567 | 23.91 |
| QAFold | 8b16 | 72.416 | 0.1458 | 37.94 |
| QAFold | 70b8 | 199.537 | 0.8649 | 130.16 |
| QAFold | 70b16 | 351.720 | 0.8283 | 196.45 |
