**Table 3.** Mirror/rotation consistency and transformed correctness at 1x.

| Model | E5 | CM | EM | CR | ER |
|---|---:|---:|---:|---:|---:|
| A | 0.0221 (0.0232, 0.0210) | 0.00174 (0.00149, 0.00199) | 0.0227 (0.0236, 0.0218) | 0.00303 (0.00278, 0.00328) | 0.0238 (0.0247, 0.0230) |
| Blank | 0.0426 | 0.00e+00 | 0.0426 | 0.00e+00 | 0.0426 |
| Copy-last | 0.0641 | 0.00e+00 | 0.0641 | 0.00e+00 | 0.0641 |

Learned entries: mean (seed 0, seed 1). Errors are pixel MSE on 1,000 test videos. Timing columns are seconds; inference excludes host-to-device transfer and includes XPU synchronization.
