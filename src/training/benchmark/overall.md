# Punteggi per prova e set

Generato da `src/scripts/benchmark_stats.py`. Solo numeri aggregati. Leggere prima le avvertenze in [README.md](README.md).

| run | set | images | macro_f1 | weighted_f1 | subject | background | perspective | individuals | text_classification | text_language |
|---|---|---|---|---|---|---|---|---|---|---|
| P11 LoRA+teste 256 | coverage | 2532 | 0.7808 | 0.9054 | 0.8409 | 0.9114 | 0.6247 | 0.9811 | 0.8060 | 0.5207 |
| P11 LoRA+teste 256 | base (prompt-dev + prompt-test) | 2363 | 0.7863 | 0.9126 | 0.8476 | 0.9140 | 0.6283 | 0.9900 | 0.8171 | 0.5210 |
| P11 LoRA+teste 256 | prompt-test | 1193 | 0.7909 | 0.9151 | 0.8605 | 0.8939 | 0.6197 | 0.9900 | 0.8050 | 0.5765 |
| P11 LoRA+teste 256 | prompt-dev, prime 600 | 600 | 0.7665 | 0.9081 | 0.8075 | 0.9181 | 0.5936 | 0.9924 | 0.8462 | 0.4413 |
| P12 LoRA+teste+pooling 512 | coverage | 2532 | 0.8056 | 0.9162 | 0.8296 | 0.8746 | 0.7444 | 0.9687 | 0.8050 | 0.6115 |
| P12 LoRA+teste+pooling 512 | base (prompt-dev + prompt-test) | 2363 | 0.8000 | 0.9221 | 0.8325 | 0.8800 | 0.7471 | 0.9770 | 0.7603 | 0.6030 |
| P12 LoRA+teste+pooling 512 | prompt-test | 1193 | 0.7913 | 0.9211 | 0.8319 | 0.8715 | 0.7517 | 0.9690 | 0.7359 | 0.5876 |
| P12 LoRA+teste+pooling 512 | prompt-dev, prime 600 | 600 | 0.7981 | 0.9235 | 0.8401 | 0.8755 | 0.7252 | 0.9924 | 0.7207 | 0.6348 |
| Nathan vlh finale (final-abt) | prompt-test | 1193 | 0.8214 | 0.9166 | 0.8463 | 0.8810 | 0.7550 | 0.9594 | 0.7665 | 0.7203 |
| Ensemble P11+Nathan media | prompt-test | 1193 | 0.8181 | 0.9257 | 0.8668 | 0.9034 | 0.7516 | 0.9849 | 0.7953 | 0.6068 |
| Ensemble P11+Nathan, lingua da Nathan | prompt-test | 1193 | 0.8371 | 0.9266 | 0.8668 | 0.9034 | 0.7516 | 0.9849 | 0.7953 | 0.7203 |
| Teste su Apertus congelato: ce+gradnorm | set di 450 immagini | 450 | 0.5371 | 0.6047 | 0.4623 | 0.7565 | 0.2283 | 0.9017 | 0.5065 | 0.3674 |
| Teste su Apertus congelato: ce | set di 450 immagini | 450 | 0.5416 | 0.6118 | 0.4657 | 0.8170 | 0.2297 | 0.8811 | 0.4931 | 0.3630 |
| Teste su Apertus congelato: focal+gradnorm | set di 450 immagini | 450 | 0.5006 | 0.5770 | 0.4412 | 0.6422 | 0.2028 | 0.8811 | 0.4719 | 0.3643 |
| Teste su Apertus congelato: focal | set di 450 immagini | 450 | 0.5283 | 0.6004 | 0.4483 | 0.7639 | 0.2670 | 0.8853 | 0.4994 | 0.3058 |
| Teste con pesi per classe: ce+winv | set di 450 immagini | 450 | 0.5602 | 0.6332 | 0.4575 | 0.8064 | 0.2394 | 0.9175 | 0.5030 | 0.4375 |
| Teste con pesi per classe: ce+winv_seed1 | set di 450 immagini | 450 | 0.5514 | 0.6236 | 0.4846 | 0.7773 | 0.2323 | 0.9252 | 0.5034 | 0.3856 |
| Teste con pesi per classe: ce+winv_seed2 | set di 450 immagini | 450 | 0.5543 | 0.6278 | 0.4787 | 0.7719 | 0.2379 | 0.9252 | 0.5384 | 0.3734 |
| Teste con pesi per classe: ce+winv_seed3 | set di 450 immagini | 450 | 0.5619 | 0.6269 | 0.4867 | 0.7984 | 0.2084 | 0.9175 | 0.5308 | 0.4295 |
| Teste con pesi per classe: ce+winv_seed4 | set di 450 immagini | 450 | 0.5590 | 0.6346 | 0.5070 | 0.8191 | 0.2501 | 0.9057 | 0.5218 | 0.3501 |
| Teste con pesi per classe: ce+wsqrt | set di 450 immagini | 450 | 0.5340 | 0.6128 | 0.4500 | 0.7621 | 0.2252 | 0.9017 | 0.4939 | 0.3711 |
| Teste con pesi per classe: ce+wsqrt_seed1 | set di 450 immagini | 450 | 0.5325 | 0.6097 | 0.4172 | 0.7533 | 0.2012 | 0.9214 | 0.5275 | 0.3741 |
| Teste con pesi per classe: ce+wsqrt_seed2 | set di 450 immagini | 450 | 0.5448 | 0.6179 | 0.4660 | 0.7790 | 0.2309 | 0.9057 | 0.5042 | 0.3827 |
| Teste con pesi per classe: ce+wsqrt_seed3 | set di 450 immagini | 450 | 0.5430 | 0.6175 | 0.4481 | 0.7938 | 0.2103 | 0.9136 | 0.5333 | 0.3591 |
| Teste con pesi per classe: ce+wsqrt_seed4 | set di 450 immagini | 450 | 0.5354 | 0.6101 | 0.4357 | 0.7822 | 0.2266 | 0.8726 | 0.5049 | 0.3905 |
| Teste con pesi per classe: ce | set di 450 immagini | 450 | 0.5328 | 0.5985 | 0.4514 | 0.7608 | 0.2324 | 0.8769 | 0.4903 | 0.3851 |
| Teste con pesi per classe: ce_seed1 | set di 450 immagini | 450 | 0.5349 | 0.6087 | 0.4559 | 0.7794 | 0.2581 | 0.8726 | 0.4865 | 0.3568 |
| Teste con pesi per classe: ce_seed2 | set di 450 immagini | 450 | 0.5355 | 0.6136 | 0.4664 | 0.7638 | 0.2228 | 0.9136 | 0.5237 | 0.3228 |
| Teste con pesi per classe: ce_seed3 | set di 450 immagini | 450 | 0.5456 | 0.6091 | 0.4637 | 0.7917 | 0.2373 | 0.8976 | 0.4807 | 0.4026 |
| Teste con pesi per classe: ce_seed4 | set di 450 immagini | 450 | 0.5451 | 0.6175 | 0.4915 | 0.7914 | 0.2296 | 0.9057 | 0.5248 | 0.3277 |
| Teste con pesi per classe: focal+winv | set di 450 immagini | 450 | 0.5498 | 0.6217 | 0.4644 | 0.7646 | 0.2449 | 0.9214 | 0.4893 | 0.4144 |
| Teste con pesi per classe: focal+winv_seed1 | set di 450 immagini | 450 | 0.5398 | 0.6089 | 0.4266 | 0.7807 | 0.2449 | 0.9220 | 0.4964 | 0.3685 |
| Teste con pesi per classe: focal+winv_seed2 | set di 450 immagini | 450 | 0.5475 | 0.6185 | 0.4648 | 0.7738 | 0.2247 | 0.9096 | 0.5188 | 0.3930 |
| Teste con pesi per classe: focal+winv_seed3 | set di 450 immagini | 450 | 0.5568 | 0.6269 | 0.4586 | 0.7783 | 0.2246 | 0.9403 | 0.5190 | 0.4202 |
| Teste con pesi per classe: focal+winv_seed4 | set di 450 immagini | 450 | 0.5397 | 0.6136 | 0.4592 | 0.7754 | 0.2379 | 0.9214 | 0.4884 | 0.3558 |
| Teste con pesi per classe: focal+wsqrt | set di 450 immagini | 450 | 0.5324 | 0.6085 | 0.4363 | 0.7799 | 0.2030 | 0.9444 | 0.4964 | 0.3343 |
| Teste con pesi per classe: focal+wsqrt_seed1 | set di 450 immagini | 450 | 0.5361 | 0.6088 | 0.4538 | 0.7649 | 0.2225 | 0.9096 | 0.4860 | 0.3798 |
| Teste con pesi per classe: focal+wsqrt_seed2 | set di 450 immagini | 450 | 0.5514 | 0.6177 | 0.4568 | 0.8092 | 0.2525 | 0.8769 | 0.5130 | 0.4003 |
| Teste con pesi per classe: focal+wsqrt_seed3 | set di 450 immagini | 450 | 0.5350 | 0.6079 | 0.4484 | 0.7814 | 0.2127 | 0.9096 | 0.5119 | 0.3460 |
| Teste con pesi per classe: focal+wsqrt_seed4 | set di 450 immagini | 450 | 0.5374 | 0.6133 | 0.4512 | 0.7730 | 0.2422 | 0.8811 | 0.5146 | 0.3622 |
| Teste con pesi per classe: focal | set di 450 immagini | 450 | 0.5463 | 0.6083 | 0.4540 | 0.7735 | 0.2384 | 0.9252 | 0.5061 | 0.3807 |
| Teste con pesi per classe: focal_seed1 | set di 450 immagini | 450 | 0.5186 | 0.5890 | 0.4558 | 0.7052 | 0.2378 | 0.9136 | 0.4651 | 0.3341 |
| Teste con pesi per classe: focal_seed2 | set di 450 immagini | 450 | 0.5241 | 0.5974 | 0.4282 | 0.7709 | 0.2095 | 0.9175 | 0.4723 | 0.3464 |
| Teste con pesi per classe: focal_seed3 | set di 450 immagini | 450 | 0.5468 | 0.6101 | 0.4472 | 0.7907 | 0.2165 | 0.9347 | 0.4956 | 0.3964 |
| Teste con pesi per classe: focal_seed4 | set di 450 immagini | 450 | 0.5262 | 0.5992 | 0.4541 | 0.7534 | 0.2212 | 0.8683 | 0.5007 | 0.3597 |
| Baseline pixel, 450: ce+gradnorm | set di 450 immagini | 450 | 0.2418 | 0.3352 | 0.1720 | 0.4079 | 0.1370 | 0.4864 | 0.1028 | 0.1447 |
| Baseline pixel, 450: ce | set di 450 immagini | 450 | 0.2417 | 0.3357 | 0.1700 | 0.4026 | 0.1302 | 0.4953 | 0.1028 | 0.1491 |
| Baseline pixel, 450: focal+gradnorm | set di 450 immagini | 450 | 0.2571 | 0.3444 | 0.1861 | 0.4347 | 0.1294 | 0.5294 | 0.1111 | 0.1517 |
| Baseline pixel, 450: focal | set di 450 immagini | 450 | 0.2560 | 0.3452 | 0.1880 | 0.4505 | 0.1371 | 0.5040 | 0.1060 | 0.1502 |
| Baseline pixel, coverage: ce+gradnorm | coverage | 2532 | 0.3575 | 0.6738 | 0.2634 | 0.5847 | 0.3038 | 0.6157 | 0.1913 | 0.1863 |
| Baseline pixel, coverage: ce+gradnorm | base (prompt-dev + prompt-test) | 2363 | 0.3603 | 0.6813 | 0.2671 | 0.5910 | 0.3025 | 0.6168 | 0.1912 | 0.1932 |
| Baseline pixel, coverage: ce+gradnorm | prompt-test | 1193 | 0.3597 | 0.6796 | 0.2664 | 0.5621 | 0.3175 | 0.6319 | 0.1942 | 0.1860 |
| Baseline pixel, coverage: ce+gradnorm | prompt-dev, prime 600 | 600 | 0.3399 | 0.6808 | 0.2487 | 0.6215 | 0.2388 | 0.5750 | 0.1714 | 0.1839 |
| Baseline pixel, coverage: ce | coverage | 2532 | 0.3620 | 0.6756 | 0.2755 | 0.5889 | 0.2982 | 0.6234 | 0.1970 | 0.1893 |
| Baseline pixel, coverage: ce | base (prompt-dev + prompt-test) | 2363 | 0.3645 | 0.6830 | 0.2794 | 0.5932 | 0.2962 | 0.6249 | 0.1978 | 0.1954 |
| Baseline pixel, coverage: ce | prompt-test | 1193 | 0.3604 | 0.6792 | 0.2704 | 0.5600 | 0.3012 | 0.6371 | 0.2069 | 0.1869 |
| Baseline pixel, coverage: ce | prompt-dev, prime 600 | 600 | 0.3477 | 0.6852 | 0.2652 | 0.6233 | 0.2384 | 0.5961 | 0.1706 | 0.1923 |
| Baseline pixel, coverage: focal+gradnorm | coverage | 2532 | 0.3662 | 0.6785 | 0.2749 | 0.6057 | 0.3081 | 0.6044 | 0.2085 | 0.1953 |
| Baseline pixel, coverage: focal+gradnorm | base (prompt-dev + prompt-test) | 2363 | 0.3692 | 0.6858 | 0.2784 | 0.6089 | 0.3063 | 0.6103 | 0.2086 | 0.2029 |
| Baseline pixel, coverage: focal+gradnorm | prompt-test | 1193 | 0.3666 | 0.6825 | 0.2706 | 0.5791 | 0.3205 | 0.6147 | 0.2179 | 0.1968 |
| Baseline pixel, coverage: focal+gradnorm | prompt-dev, prime 600 | 600 | 0.3460 | 0.6852 | 0.2693 | 0.5997 | 0.2410 | 0.5961 | 0.1788 | 0.1912 |
| Baseline pixel, coverage: focal | coverage | 2532 | 0.3669 | 0.6797 | 0.2827 | 0.6045 | 0.3050 | 0.6016 | 0.2151 | 0.1926 |
| Baseline pixel, coverage: focal | base (prompt-dev + prompt-test) | 2363 | 0.3689 | 0.6871 | 0.2877 | 0.6056 | 0.3036 | 0.6008 | 0.2158 | 0.1995 |
| Baseline pixel, coverage: focal | prompt-test | 1193 | 0.3697 | 0.6855 | 0.2813 | 0.5829 | 0.3199 | 0.6126 | 0.2251 | 0.1961 |
| Baseline pixel, coverage: focal | prompt-dev, prime 600 | 600 | 0.3450 | 0.6849 | 0.2806 | 0.5959 | 0.2461 | 0.5930 | 0.1699 | 0.1846 |
| Zero-shot 70B: describe, codes | coverage | 2532 | 0.5829 | 0.7385 | 0.5455 | 0.6313 | 0.2741 | 0.8986 | 0.6184 | 0.5292 |
| Zero-shot 70B: describe, codes | base (prompt-dev + prompt-test) | 2363 | 0.5871 | 0.7416 | 0.5447 | 0.6300 | 0.2764 | 0.9090 | 0.6252 | 0.5372 |
| Zero-shot 70B: describe, codes | prompt-test | 1193 | 0.5897 | 0.7399 | 0.5428 | 0.6105 | 0.2763 | 0.9217 | 0.6179 | 0.5691 |
| Zero-shot 70B: describe, codes | prompt-dev, prime 600 | 600 | 0.5873 | 0.7455 | 0.5737 | 0.6285 | 0.2645 | 0.9015 | 0.7018 | 0.4534 |
| Zero-shot 70B: shared, json | coverage | 2532 | 0.3036 | 0.6159 | 0.0760 | 0.3973 | 0.1167 | 0.6606 | 0.2750 | 0.2960 |
| Zero-shot 70B: shared, json | base (prompt-dev + prompt-test) | 2363 | 0.2988 | 0.6184 | 0.0768 | 0.4006 | 0.1153 | 0.6571 | 0.2760 | 0.2671 |
| Zero-shot 70B: shared, json | prompt-test | 1193 | 0.2940 | 0.6210 | 0.0864 | 0.4177 | 0.1138 | 0.6123 | 0.2790 | 0.2548 |
| Zero-shot 70B: shared, json | prompt-dev, prime 600 | 600 | 0.3122 | 0.6216 | 0.0703 | 0.3690 | 0.1115 | 0.6938 | 0.2802 | 0.3482 |
| Zero-shot 70B: split, codes | prompt-dev, prime 600 | 600 | 0.4543 | 0.6825 | 0.4270 | 0.5206 | 0.1352 | 0.7832 | 0.5284 | 0.3316 |
| Zero-shot 8B: describe, codes | coverage | 2532 | 0.5282 | 0.7047 | 0.4708 | 0.5798 | 0.2026 | 0.9036 | 0.5273 | 0.4853 |
| Zero-shot 8B: describe, codes | base (prompt-dev + prompt-test) | 2363 | 0.5125 | 0.7077 | 0.4731 | 0.5757 | 0.2047 | 0.9109 | 0.4942 | 0.4167 |
| Zero-shot 8B: describe, codes | prompt-test | 1193 | 0.5019 | 0.7029 | 0.4645 | 0.5450 | 0.1978 | 0.9213 | 0.4762 | 0.4068 |
| Zero-shot 8B: describe, codes | prompt-dev, prime 600 | 600 | 0.5213 | 0.7191 | 0.4738 | 0.5745 | 0.2277 | 0.8953 | 0.5110 | 0.4458 |
| Zero-shot 8B: describe, names | set di 12 immagini | 12 | 0.4628 | 0.7709 | 0.5750 | 0.5524 | 0.6444 | 0.4737 | 0.1562 | 0.3750 |
| Zero-shot 8B: full, codes | prompt-dev, prime 600 | 600 | 0.1639 | 0.4625 | 0.0559 | 0.1386 | 0.0431 | 0.4810 | 0.1530 | 0.1117 |
| Zero-shot 8B: full, json | prompt-dev, prime 600 | 600 | 0.3261 | 0.6288 | 0.0573 | 0.3513 | 0.1653 | 0.7470 | 0.2153 | 0.4204 |
| Zero-shot 8B: minimal, codes | prompt-dev, prime 600 | 600 | 0.1383 | 0.3477 | 0.0604 | 0.0771 | 0.0847 | 0.4816 | 0.0357 | 0.0904 |
| Zero-shot 8B: minimal, json | coverage | 2532 | 0.3163 | 0.6320 | 0.0761 | 0.4113 | 0.0871 | 0.7344 | 0.1917 | 0.3972 |
| Zero-shot 8B: minimal, json | base (prompt-dev + prompt-test) | 2363 | 0.3103 | 0.6357 | 0.0789 | 0.4155 | 0.0873 | 0.7327 | 0.1903 | 0.3572 |
| Zero-shot 8B: minimal, json | prompt-test | 1193 | 0.2912 | 0.6309 | 0.0733 | 0.4207 | 0.0858 | 0.6647 | 0.1961 | 0.3064 |
| Zero-shot 8B: minimal, json | prompt-dev, prime 600 | 600 | 0.3274 | 0.6486 | 0.0651 | 0.4029 | 0.0912 | 0.7951 | 0.1883 | 0.4216 |
| Zero-shot 8B: shared, json | coverage | 2532 | 0.2293 | 0.4675 | 0.0650 | 0.2676 | 0.0979 | 0.5811 | 0.2133 | 0.1508 |
| Zero-shot 8B: shared, json | base (prompt-dev + prompt-test) | 2363 | 0.2304 | 0.4719 | 0.0662 | 0.2646 | 0.0967 | 0.5852 | 0.2167 | 0.1534 |
| Zero-shot 8B: shared, json | prompt-test | 1193 | 0.2250 | 0.4692 | 0.0672 | 0.2690 | 0.0897 | 0.5623 | 0.2265 | 0.1352 |
| Zero-shot 8B: shared, json | prompt-dev, prime 600 | 600 | 0.2496 | 0.4827 | 0.0805 | 0.2427 | 0.1294 | 0.5961 | 0.2213 | 0.2273 |
| Zero-shot 8B: split, codes | prompt-dev, prime 600 | 600 | 0.4822 | 0.6869 | 0.4482 | 0.6034 | 0.1309 | 0.8961 | 0.4863 | 0.3283 |
| Zero-shot 8B: split, names | prompt-dev, prime 600 | 600 | 0.4527 | 0.6334 | 0.4385 | 0.5222 | 0.1028 | 0.8804 | 0.4130 | 0.3591 |
