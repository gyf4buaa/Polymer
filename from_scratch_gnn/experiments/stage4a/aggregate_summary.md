# Stage 4A — global information augmentation

Formal source commit: `64cbe8e540fd4347c929b6364d0d598488cf96f4`.
G0 reuses the five historical Stage 3A.1 / Stage 3C R0 runs; D and M each add five frozen five-fold OOF runs.

## Overall OOF wMAE

| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 | 0.0226486259–0.0228896667 |
| D | 0.0229013665 | 0.0230870474 | 0.0230705655 | 0.0230880763 | 0.0230255862 | 0.0230345284 ± 0.0000786390 | 0.0229013665–0.0230880763 |
| M | 0.0259070865 | 0.0261050336 | 0.0257674891 | 0.0257695495 | 0.0258301908 | 0.0258758699 ± 0.0001402029 | 0.0257674891–0.0261050336 |

## Paired overall deltas (left minus right)

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| D-G0 | +0.0001284943 | +0.0004384215 | +0.0001808988 | +0.0003550693 | +0.0001427240 | +0.0002491216 ± 0.0001392705 | 0/5 |
| M-G0 | +0.0031342143 | +0.0034564077 | +0.0028778224 | +0.0030365425 | +0.0029473286 | +0.0030904631 ± 0.0002260755 | 0/5 |
| D-M | -0.0030057200 | -0.0030179862 | -0.0026969236 | -0.0026814732 | -0.0028046045 | -0.0028413415 ± 0.0001627787 | 5/5 |

## Per-target MAE

| Target | G0 mean ± SD | D mean ± SD | M mean ± SD | D−G0 delta mean ± SD | M−G0 delta mean ± SD |
|---|---:|---:|---:|---:|---:|
| Tg | 51.7968233185 ± 0.9200317144; 50.7396148061–53.2698233561 | 51.4390009508 ± 0.9473472544; 49.8552254054–52.3784384407 | 56.8695953429 ± 0.8501726845; 55.7853009730–57.7819200209 | -0.3578223677 ± 0.4970783857 | +5.072772024 ± 1.335256927 |
| FFV | 0.0058948628 ± 0.0001283411; 0.0057822356–0.0061021919 | 0.0062351798 ± 0.0002323543; 0.0059931834–0.0065573018 | 0.0073537757 ± 0.0001339283; 0.0072248972–0.0075735554 | +0.0003403169831 ± 0.0002898530846 | +0.001458912832 ± 0.0001767912177 |
| Tc | 0.0246617909 ± 0.0003660684; 0.0241218720–0.0250413770 | 0.0248372575 ± 0.0002554596; 0.0244552121–0.0251740042 | 0.0276782102 ± 0.0003630837; 0.0273976589–0.0281511287 | +0.0001754666259 ± 0.0002880590792 | +0.003016419328 ± 0.0005942636855 |
| Density | 0.0246015926 ± 0.0006658114; 0.0236123141–0.0252954472 | 0.0259022557 ± 0.0008308891; 0.0248113767–0.0270879778 | 0.0348146568 ± 0.0009801374; 0.0338773990–0.0362385425 | +0.001300663143 ± 0.0005120351416 | +0.0102130642 ± 0.001190419142 |
| Rg | 1.5773881146 ± 0.0216399134; 1.5559213601–1.6102015654 | 1.5680595939 ± 0.0296361115; 1.5238572094–1.6018230163 | 1.6237012509 ± 0.0126682973; 1.6091275857–1.6425824506 | -0.009328520687 ± 0.04580063027 | +0.04631313637 ± 0.03166833398 |

## Parameters and execution

- G0/D/M parameter counts: `{'D': {'G0_total_trainable': 1243657, 'D_total_trainable': 1253897, 'projection_trainable': 10240, 'new_parameters_vs_G0': 10240}, 'M': {'G0_total_trainable': 1243657, 'M_total_trainable': 2292233, 'projection_trainable': 1048576, 'new_parameters_vs_G0': 1048576}, 'G0': {'total_trainable': 1243657, 'projection_trainable': 0, 'new_parameters_vs_G0': 0}}`.
- Morgan adds 1,048,576 trainable parameters; a Morgan score gain cannot be attributed to fingerprint information alone because model capacity also increases.
- New formal runs: 10; G0 retrains: 0.
- Training time from run metadata: 1.59 h; formal queue wall time: 3477.8 s.
- RTX device(s): NVIDIA GeForce RTX 4070; concurrency: 2.
- Maximum VRAM: allocated 153.9 MiB, reserved 192.0 MiB; nvidia-smi 4803.0 MiB.
- Mean GPU utilization across folds: 68.03170054945055; max sampled utilization: 84.0; mean epoch time: 1.5276791834882133 s.
- Throughput pilot: `{'single_lane': {'wall_seconds': 20.208273355994606, 'jobs': {'D': {'return_code': 0, 'training_seconds': 6.945626651009661, 'process_wall_seconds': 10.098727917007636, 'seconds_per_epoch': 1.3053034087992272, 'peak_vram_allocated_mb': 124.71142578125, 'projection_l2_norm': 2.7508881092071533, 'verification_status': 'passed', 'log_path': '/tmp/stage4a_pilot_logs/measured_serial_D_seed_43.log'}, 'M': {'return_code': 0, 'training_seconds': 7.08922681499098, 'process_wall_seconds': 10.108826838986715, 'seconds_per_epoch': 1.3299802166002337, 'peak_vram_allocated_mb': 141.97607421875, 'projection_l2_norm': 18.62412452697754, 'verification_status': 'passed', 'log_path': '/tmp/stage4a_pilot_logs/measured_serial_M_seed_43.log'}}, 'resource_samples': {'mean_cpu_percent': 9.036456921162863, 'max_cpu_percent': 20.863309352517987, 'mean_ram_used_mb': 1812.673828125, 'max_ram_used_mb': 2209.13671875, 'mean_gpu_util_percent': 19.73076923076923, 'max_gpu_util_percent': 37.0, 'max_gpu_memory_used_mb': 4413.0, 'gpu_total_memory_mb': 12282.0}}, 'two_lane': {'wall_seconds': 11.77111080600298, 'jobs': {'D': {'return_code': 0, 'training_seconds': 8.681872183995438, 'process_wall_seconds': 11.77072083701205, 'seconds_per_epoch': 1.6438111651979852, 'peak_vram_allocated_mb': 124.71142578125, 'projection_l2_norm': 2.5452792644500732, 'verification_status': 'passed', 'log_path': '/tmp/stage4a_pilot_logs/measured_parallel_D_seed_43.log'}, 'M': {'return_code': 0, 'training_seconds': 8.681056563000311, 'process_wall_seconds': 11.77055072800431, 'seconds_per_epoch': 1.6440222770004767, 'peak_vram_allocated_mb': 141.97607421875, 'projection_l2_norm': 23.00569725036621, 'verification_status': 'passed', 'log_path': '/tmp/stage4a_pilot_logs/measured_parallel_M_seed_43.log'}}, 'resource_samples': {'mean_cpu_percent': 16.687950700073262, 'max_cpu_percent': 30.62438057482656, 'mean_ram_used_mb': 2901.4828125, 'max_ram_used_mb': 3532.8671875, 'mean_gpu_util_percent': 54.0, 'max_gpu_util_percent': 81.0, 'max_gpu_memory_used_mb': 4754.0, 'gpu_total_memory_mb': 12282.0}}, 'speedup': 1.7167685946587878, 'wall_time_reduction_percent': 41.75103137888233}`.
- CPU/RAM bottleneck: `No pilot CPU/RAM bottleneck: two-lane max CPU 30.6%, peak WSL RAM 3533 MiB; formal max CPU 22.3%, peak WSL RAM 3503 MiB.`; engineering failures: `[{'stage': 'pre-formal CUDA smoke', 'issue': 'Linux RDKit descriptor byte hash differed from the macOS raw matrix despite matching version and summary statistics.', 'remediation': 'Transferred the audited feature cache; both descriptor and Morgan matrices pass committed manifest hashes.', 'formal_impact': 'No formal runs were attempted before the cache matched.'}, {'stage': 'pre-formal CUDA smoke', 'issue': 'Initial fold adapter dropped rows, fold_ids, and output_dir when calling the frozen trainer.', 'remediation': 'Fixed in the final source commit and added a regression test.', 'formal_impact': 'No formal runs were attempted before the fix.'}]`.

## Diagnostics

- D projection mean L2 norm by seed: `{'42': 5.360447883605957, '43': 4.645516490936279, '44': 4.969697761535644, '45': 4.817241573333741, '46': 5.226688194274902}`.
- M projection mean L2 norm by seed: `{'42': 30.290189361572267, '43': 36.305258178710936, '44': 35.745721435546876, '45': 32.8565170288086, '46': 32.75241584777832}`.
- M average bits on: `{'42': 37.95359337764957, '43': 37.95359337764957, '44': 37.95359337764957, '45': 37.95359337764957, '46': 37.95359337764957}`; duplicate fingerprint rows: `{'42': 1133, '43': 1133, '44': 1133, '45': 1133, '46': 1133}`.
- D training-fold means/stds and the complete descriptor audit are retained in each run's scaler files and feature manifest.
- Diagnostics are descriptive only and were not used for feature selection or tuning.

## Conclusion

- D stable improvement: **False**.
- M stable improvement: **False**.
- D+M combination warranted: **False**.
- Next model to carry: **G0**.
- Stage 4B was not run.
