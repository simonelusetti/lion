# Few-NERD replication: context sensitivity

Pinned `DFKI-SLT/few-nerd`, supervised split, revision `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b`. Exact file names and SHA-256 values are in [provenance.json](tables/fewnerd/provenance.json). Coarse IO converted to BIO: B on non-O type change/after O, I on continuation. Contiguous same-coarse-type entities merge because IO lacks boundaries; fine tags are not used.

Seed 0: 10,531 source training sentences → 2,106 scorer fit + 527 scorer validation + 3,949 T1 + 3,949 T2. Classifiers use exactly T1∪T2 (7,898 sentences), excluding scorer split A. Native validation has 2,438 sentences. Test: 1,870 sentences, 46,435 words. CoNLL used 14,041 source / 10,531 classifier sentences. Calibration/held-out proportions remain 70%/30% of the opposite selector half.

Test gold spans seen anywhere in T1∪T2: 1,162/4,823 (24.09%); seen with same type: 22.39%. This is a span-weighted statistic; C below is focal-word weighted.

Before training: Part 1 3.88 h, Part 2 17.1 min; total 4.17 h. Test not halved under the six-hour rule. Actual Part 1 3.00 h; Part 2 12.5 min; pipeline wall-clock 3.50 h. Fresh fills were generated once, with token IDs saved from the start; all diagnostics reuse these four draws per row, mask and fold.

Calibration γ = 0.089514, the new scorer's token error rate on its own validation split. Measurement S retains the step-1 definition: KL(original || filled), γ=.01 over native BIO tags (CoNLL nine, Few-NERD seventeen). A flip changes argmax. Folds and draws are averaged within each focal word; they are not extra observations. Baselines show mean ± sample SD over seeds 0/1/2 plus row SE. Selector/random always use common rows, bins and weights. * marks fewer than 50 paired rows; these cells do not contribute ratios. Failed-gate groups remain in every analysis as UNCERTIFIED.

Checkpoint selection is raw native-dev token macro-F1, including O and all BIO tags, exactly the existing matched baseline criterion; this differs from reported span F1. No augmented arms are trained.

Few-NERD group status: O: gate passed, entity: gate passed. These statuses apply to every table below.

## Pipeline: cuts, gate and label-flip reference

| Fold | Group | CoNLL | Few-NERD |
| --- | --- | --- | --- |
| g1 | O | ρ=0.00; pass; ΔCE=0.0264; abs CE=0.0964 ± 0.5923; n=1560; UCB=0.1210 | ρ=0.30; pass; ΔCE=0.0268; abs CE=0.2361 ± 0.6786; n=1185; UCB=0.2686 |
| g1 | entity | ρ=0.80; pass; ΔCE=0.0305; abs CE=0.4005 ± 0.7943; n=1277; UCB=0.4370 | ρ=0.75; pass; ΔCE=0.0291; abs CE=1.0687 ± 1.1493; n=1035; UCB=1.1274 |
| g2 | O | ρ=0.00; pass; ΔCE=0.0148; abs CE=0.0739 ± 0.4548; n=1551; UCB=0.0929 | ρ=0.25; pass; ΔCE=0.0534; abs CE=0.2776 ± 0.7695; n=1185; UCB=0.3143 |
| g2 | entity | ρ=0.85; pass; ΔCE=0.0256; abs CE=0.3971 ± 0.8277; n=1246; UCB=0.4356 | ρ=0.75; pass; ΔCE=0.0550; abs CE=1.0651 ± 1.1108; n=1052; UCB=1.1214 |

The gate rejects when the held-out CE-increase lower bound exceeds ε=.05; ρ=1 passes by identity. Absolute CE UCB is mean + 1.645 SD/√n. It is an exact label-error reference only for fill-ins conditioned on selection; these are the calibration-style fill-ins. Calibration uses Learn then Test, δ=.1 and the unchanged 0.05 cut grid.

## Raw test span F1

| Model | Seed | CoNLL F1 | Few-NERD F1 | Few precision | Few recall |
| --- | --- | --- | --- | --- | --- |
| baseline | 0 | 0.8408 | 0.5945 | 0.5393 | 0.6622 |
| baseline | 1 | 0.8459 | 0.6080 | 0.5480 | 0.6828 |
| baseline | 2 | 0.8413 | 0.6040 | 0.5409 | 0.6836 |
| scorer | -1 | 0.8161 | 0.5403 | 0.4926 | 0.5982 |
| baseline | mean ± SD | 0.8427 ± 0.0029 | 0.6021 ± 0.0070 | — | — |

All F1 values use the existing exact BIO span implementation. The scorer is one fitted model; baseline summaries use three seeds.

## Step 1: all draws

| Model | Group | CoNLL n | CoNLL S sel / rand | CoNLL flips % sel / rand | CoNLL raw ratio | CoNLL reference | Few n | Few S sel / rand | Few flips % sel / rand | Few raw ratio | Few reference |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | 38303 | 0.0573 ± 0.0109 (SE 0.0015) / 0.0572 ± 0.0108 (SE 0.0014) | 1.6824 ± 0.3400 (SE 0.0417) / 1.6773 ± 0.3209 (SE 0.0415) | 1.0019 ± 0.0059 (SE 0.0116) | 0.1070 | 37705 | 0.1080 ± 0.0121 (SE 0.0019) / 0.1052 ± 0.0112 (SE 0.0019) | 3.1851 ± 0.4431 (SE 0.0740) / 3.0540 ± 0.3756 (SE 0.0731) | 1.0257 ± 0.0115 (SE 0.0094) | 0.2914 |
| baseline | entity | 8132 | 0.1024 ± 0.0007 (SE 0.0042) / 0.1283 ± 0.0005 (SE 0.0046) | 4.9844 ± 0.0441 (SE 0.1482) / 5.9200 ± 0.1000 (SE 0.1587) | 0.7982 ± 0.0020 (SE 0.0379) | 0.4363 | 8730 | 0.1079 ± 0.0008 (SE 0.0034) / 0.1314 ± 0.0009 (SE 0.0035) | 8.2737 ± 0.2491 (SE 0.1756) / 9.8940 ± 0.1862 (SE 0.1911) | 0.8209 ± 0.0021 (SE 0.0299) | 1.1244 |
| scorer | O | 38303 | 0.0365 (SE 0.0009) / 0.0365 (SE 0.0009) | 0.7989 (SE 0.0254) / 0.8133 (SE 0.0254) | 1.0002 (SE 0.0115) | 0.1070 | 37705 | 0.0615 (SE 0.0010) / 0.0590 (SE 0.0010) | 0.7701 (SE 0.0273) / 0.6859 (SE 0.0261) | 1.0423 (SE 0.0087) | 0.2914 |
| scorer | entity | 8132 | 0.0861 (SE 0.0034) / 0.1064 (SE 0.0037) | 5.7197 (SE 0.1810) / 6.6235 (SE 0.1886) | 0.8091 (SE 0.0369) | 0.4363 | 8730 | 0.0915 (SE 0.0029) / 0.1084 (SE 0.0028) | 9.9184 (SE 0.2253) / 11.8585 (SE 0.2392) | 0.8447 (SE 0.0312) | 1.1244 |

Raw ratios divide selector S by random S within seed. Label-flip references average the two fold UCBs for the corresponding group.

## 1b-A: restoration and non-null draws

| Model | Group | CoNLL paired n | CoNLL excluded sel / rand | CoNLL dropped sel / rand | CoNLL changed sel / rand | CoNLL restored % sel / rand | CoNLL null % sel / rand | CoNLL non-null S sel / rand | CoNLL non-null flips % sel / rand | Few paired n | Few excluded sel / rand | Few dropped sel / rand | Few changed sel / rand | Few restored % sel / rand | Few null % sel / rand | Few non-null S sel / rand | Few non-null flips % sel / rand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | 38278 | 25 / 23 | 22.7823 / 22.7823 | 22.2501 / 22.2495 | 2.3359 / 2.3386 | 0.0822 / 0.0790 | 0.0573 ± 0.0109 (SE 0.0015) / 0.0572 ± 0.0108 (SE 0.0014) | 1.6835 ± 0.3403 (SE 0.0417) / 1.6784 ± 0.3211 (SE 0.0416) | 37702 | 1 / 3 | 22.0971 / 22.0971 | 20.7269 / 20.8712 | 6.2008 / 5.5479 | 0.0199 / 0.0146 | 0.1080 ± 0.0121 (SE 0.0019) / 0.1052 ± 0.0112 (SE 0.0019) | 3.1858 ± 0.4428 (SE 0.0740) / 3.0542 ± 0.3756 (SE 0.0731) |
| baseline | entity | 7524 | 514 / 542 | 3.5599 / 3.5599 | 2.0553 / 2.2879 | 42.2654 / 35.7333 | 18.3165 / 14.4399 | 0.1255 ± 0.0011 (SE 0.0050) / 0.1546 ± 0.0009 (SE 0.0056) | 6.0999 ± 0.0615 (SE 0.1788) / 7.1030 ± 0.1157 (SE 0.1921) | 8669 | 53 / 13 | 8.0513 / 8.0513 | 4.2113 / 4.9029 | 47.6938 / 39.1043 | 6.9273 / 2.2924 | 0.1116 ± 0.0007 (SE 0.0035) / 0.1331 ± 0.0009 (SE 0.0035) | 8.6342 ± 0.2787 (SE 0.1822) / 10.0354 ± 0.1880 (SE 0.1933) |
| scorer | O | 38278 | 25 / 23 | 22.7823 / 22.7823 | 22.2501 / 22.2495 | 2.3359 / 2.3386 | 0.0822 / 0.0790 | 0.0365 (SE 0.0009) / 0.0365 (SE 0.0009) | 0.7994 (SE 0.0255) / 0.8138 (SE 0.0255) | 37702 | 1 / 3 | 22.0971 / 22.0971 | 20.7269 / 20.8712 | 6.2008 / 5.5479 | 0.0199 / 0.0146 | 0.0615 (SE 0.0010) / 0.0590 (SE 0.0010) | 0.7702 (SE 0.0273) / 0.6860 (SE 0.0261) |
| scorer | entity | 7524 | 514 / 542 | 3.5599 / 3.5599 | 2.0553 / 2.2879 | 42.2654 / 35.7333 | 18.3165 / 14.4399 | 0.1069 (SE 0.0041) / 0.1290 (SE 0.0046) | 7.1161 (SE 0.2234) / 7.9817 (SE 0.2286) | 8669 | 53 / 13 | 8.0513 / 8.0513 | 4.2113 / 4.9029 | 47.6938 / 39.1043 | 6.9273 / 2.2924 | 0.0945 (SE 0.0030) / 0.1098 (SE 0.0029) | 10.3121 (SE 0.2335) / 12.0146 (SE 0.2424) |

Restoration counts identical words among dropped positions, case-sensitive. Null swaps include zero-drop draws. Each non-null statistic averages eligible draws per fold, then available folds per word, on common non-null row support; exclusions are mask-specific all-null rows before pairing.

## 1b-B: changed-word counts

| Model | Group | Bin | CoNLL n | CoNLL S sel / rand | CoNLL flips % sel / rand | Few n | Few S sel / rand | Few flips % sel / rand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | 0 | 37* | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 8* | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) |
| baseline | entity | 0 | 1590 | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 354 | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) |
| baseline | O | 1 | 338 | 0.1631 ± 0.0665 (SE 0.0249) / 0.1507 ± 0.0662 (SE 0.0221) | 7.9265 ± 2.9993 (SE 1.1961) / 7.4745 ± 2.8645 (SE 1.1371) | 43* | 0.0650 ± 0.0115 (SE 0.0382) / 0.0221 ± 0.0044 (SE 0.0106) | 2.9716 ± 0.2238 (SE 1.8147) / 0.5168 ± 0.2238 (SE 0.5168) |
| baseline | entity | 1 | 3741 | 0.1410 ± 0.0002 (SE 0.0079) / 0.1789 ± 0.0013 (SE 0.0092) | 6.9222 ± 0.1240 (SE 0.2858) / 8.3170 ± 0.1807 (SE 0.3151) | 1396 | 0.0428 ± 0.0004 (SE 0.0054) / 0.0564 ± 0.0013 (SE 0.0078) | 4.1309 ± 0.7506 (SE 0.3708) / 5.3078 ± 0.3369 (SE 0.4245) |
| baseline | O | 2 | 335 | 0.2090 ± 0.0425 (SE 0.0285) / 0.1962 ± 0.0456 (SE 0.0255) | 8.3085 ± 1.0988 (SE 1.0923) / 7.8151 ± 1.1275 (SE 1.0462) | 110 | 0.0735 ± 0.0179 (SE 0.0275) / 0.0777 ± 0.0151 (SE 0.0222) | 2.7273 ± 0.8194 (SE 1.2603) / 3.3712 ± 1.0800 (SE 1.2084) |
| baseline | entity | 2 | 2711 | 0.1032 ± 0.0026 (SE 0.0085) / 0.1402 ± 0.0020 (SE 0.0108) | 5.5945 ± 0.0795 (SE 0.3251) / 6.3829 ± 0.2132 (SE 0.3552) | 2401 | 0.0842 ± 0.0008 (SE 0.0069) / 0.0842 ± 0.0002 (SE 0.0070) | 7.3447 ± 0.9164 (SE 0.3862) / 7.2493 ± 0.2555 (SE 0.3708) |
| baseline | O | 3 | 592 | 0.1853 ± 0.0343 (SE 0.0244) / 0.1811 ± 0.0299 (SE 0.0245) | 5.5767 ± 1.0387 (SE 0.6600) / 5.4031 ± 0.9521 (SE 0.6448) | 216 | 0.0583 ± 0.0105 (SE 0.0153) / 0.0554 ± 0.0070 (SE 0.0121) | 1.9676 ± 0.2970 (SE 0.6488) / 1.7168 ± 0.4846 (SE 0.5489) |
| baseline | entity | 3 | 2123 | 0.0929 ± 0.0037 (SE 0.0099) / 0.1081 ± 0.0038 (SE 0.0109) | 4.7869 ± 0.2160 (SE 0.3471) / 4.9831 ± 0.3135 (SE 0.3473) | 2835 | 0.0991 ± 0.0009 (SE 0.0072) / 0.1003 ± 0.0011 (SE 0.0057) | 8.3064 ± 0.6143 (SE 0.3810) / 8.7306 ± 0.3626 (SE 0.3749) |
| baseline | O | 4 | 797 | 0.1928 ± 0.0285 (SE 0.0228) / 0.1890 ± 0.0272 (SE 0.0222) | 5.2523 ± 0.9934 (SE 0.5881) / 5.3273 ± 0.9758 (SE 0.5998) | 381 | 0.0900 ± 0.0102 (SE 0.0167) / 0.0637 ± 0.0048 (SE 0.0119) | 2.8325 ± 0.5133 (SE 0.6406) / 2.0195 ± 0.4604 (SE 0.5654) |
| baseline | entity | 4 | 1569 | 0.1147 ± 0.0033 (SE 0.0149) / 0.1104 ± 0.0025 (SE 0.0116) | 5.3095 ± 0.0836 (SE 0.4283) / 5.8654 ± 0.4030 (SE 0.4654) | 2621 | 0.1190 ± 0.0012 (SE 0.0076) / 0.1403 ± 0.0010 (SE 0.0089) | 9.7895 ± 0.5094 (SE 0.4360) / 10.5796 ± 0.4000 (SE 0.4435) |
| baseline | O | 5–9 | 7552 | 0.0799 ± 0.0140 (SE 0.0043) / 0.0804 ± 0.0137 (SE 0.0044) | 2.2884 ± 0.4081 (SE 0.1234) / 2.2546 ± 0.3824 (SE 0.1224) | 4716 | 0.0748 ± 0.0080 (SE 0.0041) / 0.0709 ± 0.0083 (SE 0.0038) | 2.4980 ± 0.2570 (SE 0.1792) / 2.2388 ± 0.2227 (SE 0.1684) |
| baseline | entity | 5–9 | 1308 | 0.1105 ± 0.0023 (SE 0.0130) / 0.1067 ± 0.0020 (SE 0.0115) | 5.3538 ± 0.1991 (SE 0.4472) / 5.4133 ± 0.2942 (SE 0.4439) | 3943 | 0.1425 ± 0.0011 (SE 0.0068) / 0.1531 ± 0.0015 (SE 0.0065) | 10.4334 ± 0.5378 (SE 0.3311) / 11.2837 ± 0.3192 (SE 0.3361) |
| baseline | O | 10+ | 29084 | 0.0433 ± 0.0086 (SE 0.0013) / 0.0434 ± 0.0087 (SE 0.0013) | 1.2422 ± 0.2707 (SE 0.0362) / 1.2574 ± 0.2557 (SE 0.0367) | 34139 | 0.1116 ± 0.0126 (SE 0.0021) / 0.1089 ± 0.0115 (SE 0.0021) | 3.2676 ± 0.4612 (SE 0.0794) / 3.1400 ± 0.3928 (SE 0.0785) |
| baseline | entity | 10+ | 116 | 0.0151 ± 0.0003 (SE 0.0051) / 0.0890 ± 0.0025 (SE 0.0535) | 0.9339 ± 0.6584 (SE 0.5930) / 2.6700 ± 0.2696 (SE 1.3300) | 894 | 0.1816 ± 0.0009 (SE 0.0156) / 0.1911 ± 0.0021 (SE 0.0153) | 11.5834 ± 0.0802 (SE 0.7159) / 12.5994 ± 0.0775 (SE 0.7251) |
| scorer | O | 0 | 37* | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 8* | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | entity | 0 | 1590 | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 354 | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | O | 1 | 338 | 0.0469 (SE 0.0087) / 0.0394 (SE 0.0063) | 1.1095 (SE 0.4277) / 0.8876 (SE 0.3519) | 43* | 0.0331 (SE 0.0154) / 0.0228 (SE 0.0090) | 0.3876 (SE 0.3876) / 0.0000 (SE 0.0000) |
| scorer | entity | 1 | 3741 | 0.1236 (SE 0.0064) / 0.1536 (SE 0.0075) | 8.5104 (SE 0.3708) / 9.8369 (SE 0.3821) | 1396 | 0.0362 (SE 0.0042) / 0.0464 (SE 0.0062) | 5.3695 (SE 0.5262) / 6.0649 (SE 0.5623) |
| scorer | O | 2 | 335 | 0.1158 (SE 0.0151) / 0.1077 (SE 0.0132) | 3.5075 (SE 0.6108) / 3.6816 (SE 0.6100) | 110 | 0.0441 (SE 0.0143) / 0.0513 (SE 0.0132) | 1.9318 (SE 0.8486) / 2.3106 (SE 1.1511) |
| scorer | entity | 2 | 2711 | 0.0825 (SE 0.0064) / 0.1113 (SE 0.0085) | 5.7174 (SE 0.3734) / 6.6335 (SE 0.4009) | 2401 | 0.0679 (SE 0.0051) / 0.0708 (SE 0.0058) | 7.6982 (SE 0.4728) / 8.5746 (SE 0.4859) |
| scorer | O | 3 | 592 | 0.1084 (SE 0.0142) / 0.1106 (SE 0.0144) | 2.9842 (SE 0.4635) / 3.2517 (SE 0.4752) | 216 | 0.0583 (SE 0.0154) / 0.0517 (SE 0.0100) | 0.6366 (SE 0.2983) / 1.1960 (SE 0.4102) |
| scorer | entity | 3 | 2123 | 0.0752 (SE 0.0077) / 0.0940 (SE 0.0092) | 4.7751 (SE 0.3946) / 5.6210 (SE 0.4304) | 2835 | 0.0828 (SE 0.0059) / 0.0840 (SE 0.0047) | 9.7619 (SE 0.4974) / 11.2684 (SE 0.5070) |
| scorer | O | 4 | 797 | 0.1293 (SE 0.0152) / 0.1253 (SE 0.0145) | 3.7275 (SE 0.4828) / 3.5550 (SE 0.4496) | 381 | 0.0601 (SE 0.0115) / 0.0486 (SE 0.0093) | 1.6076 (SE 0.5345) / 0.4921 (SE 0.3018) |
| scorer | entity | 4 | 1569 | 0.0889 (SE 0.0118) / 0.0889 (SE 0.0098) | 5.3856 (SE 0.5053) / 5.2475 (SE 0.4823) | 2621 | 0.1038 (SE 0.0065) / 0.1166 (SE 0.0072) | 11.2727 (SE 0.5488) / 12.6494 (SE 0.5624) |
| scorer | O | 5–9 | 7552 | 0.0569 (SE 0.0025) / 0.0578 (SE 0.0026) | 1.2133 (SE 0.0807) / 1.2624 (SE 0.0804) | 4716 | 0.0447 (SE 0.0024) / 0.0425 (SE 0.0024) | 0.8022 (SE 0.0922) / 0.7554 (SE 0.0892) |
| scorer | entity | 5–9 | 1308 | 0.0932 (SE 0.0111) / 0.0790 (SE 0.0083) | 5.5619 (SE 0.5141) / 5.4632 (SE 0.5084) | 3943 | 0.1233 (SE 0.0059) / 0.1253 (SE 0.0053) | 12.4493 (SE 0.4251) / 13.1700 (SE 0.4147) |
| scorer | O | 10+ | 29084 | 0.0272 (SE 0.0008) / 0.0271 (SE 0.0008) | 0.5708 (SE 0.0219) / 0.5649 (SE 0.0214) | 34139 | 0.0633 (SE 0.0011) / 0.0607 (SE 0.0011) | 0.7527 (SE 0.0282) / 0.6760 (SE 0.0274) |
| scorer | entity | 10+ | 116 | 0.0147 (SE 0.0065) / 0.0572 (SE 0.0350) | 0.5388 (SE 0.3870) / 2.3707 (SE 1.2979) | 894 | 0.1544 (SE 0.0137) / 0.1542 (SE 0.0120) | 15.7392 (SE 0.9648) / 15.8184 (SE 0.9448) |

Bins count changed words per draw. Statistics use common rows with draw support in the same bin for both masks.

## 1b-B: distance to nearest change

| Model | Group | Bin | CoNLL n | CoNLL S sel / rand | CoNLL flips % sel / rand | Few n | Few S sel / rand | Few flips % sel / rand |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | 1 | 38276 | 0.0573 ± 0.0109 (SE 0.0015) / 0.0572 ± 0.0108 (SE 0.0014) | 1.6830 ± 0.3409 (SE 0.0417) / 1.6777 ± 0.3217 (SE 0.0415) | 35769 | 0.1137 ± 0.0126 (SE 0.0021) / 0.1103 ± 0.0115 (SE 0.0020) | 3.2292 ± 0.4284 (SE 0.0773) / 3.0907 ± 0.3657 (SE 0.0763) |
| baseline | entity | 1 | 1264 | 0.4247 ± 0.0084 (SE 0.0268) / 0.4834 ± 0.0067 (SE 0.0296) | 14.8635 ± 0.2989 (SE 0.7587) / 16.6513 ± 0.4485 (SE 0.8032) | 1891 | 0.3945 ± 0.0041 (SE 0.0170) / 0.3591 ± 0.0039 (SE 0.0157) | 21.2035 ± 0.2060 (SE 0.7073) / 21.0192 ± 0.6651 (SE 0.7060) |
| baseline | O | 2 | 73 | 0.0393 ± 0.0069 (SE 0.0248) / 0.0432 ± 0.0188 (SE 0.0298) | 1.8265 ± 0.4943 (SE 1.4379) / 1.3699 ± 1.1416 (SE 1.0153) | 2678 | 0.0464 ± 0.0052 (SE 0.0040) / 0.0502 ± 0.0069 (SE 0.0046) | 2.0792 ± 0.3792 (SE 0.2289) / 2.0984 ± 0.4430 (SE 0.2283) |
| baseline | entity | 2 | 917 | 0.0655 ± 0.0018 (SE 0.0068) / 0.0790 ± 0.0025 (SE 0.0090) | 5.1194 ± 0.2263 (SE 0.4861) / 5.9297 ± 0.5934 (SE 0.5566) | 1262 | 0.0661 ± 0.0014 (SE 0.0042) / 0.0815 ± 0.0007 (SE 0.0058) | 7.5960 ± 0.9479 (SE 0.5080) / 9.4845 ± 0.1935 (SE 0.5742) |
| baseline | O | 3 | 0* | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) | 261 | 0.0250 ± 0.0031 (SE 0.0046) / 0.0310 ± 0.0050 (SE 0.0058) | 1.9902 ± 0.3414 (SE 0.6837) / 1.6071 ± 0.3327 (SE 0.5228) |
| baseline | entity | 3 | 508 | 0.0435 ± 0.0010 (SE 0.0054) / 0.0445 ± 0.0002 (SE 0.0063) | 4.2186 ± 0.4735 (SE 0.6120) / 4.3772 ± 0.1799 (SE 0.6609) | 818 | 0.0371 ± 0.0002 (SE 0.0032) / 0.0423 ± 0.0015 (SE 0.0036) | 6.3298 ± 0.1743 (SE 0.5644) / 6.5539 ± 0.6162 (SE 0.5755) |
| baseline | O | 4–5 | 0* | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) | 86 | 0.0170 ± 0.0004 (SE 0.0080) / 0.0240 ± 0.0025 (SE 0.0099) | 2.7778 ± 0.1119 (SE 1.4638) / 3.4884 ± 1.1628 (SE 1.6337) |
| baseline | entity | 4–5 | 1077 | 0.0418 ± 0.0012 (SE 0.0052) / 0.0384 ± 0.0027 (SE 0.0051) | 3.5464 ± 0.4493 (SE 0.3616) / 3.4239 ± 0.7072 (SE 0.3667) | 1421 | 0.0235 ± 0.0002 (SE 0.0021) / 0.0331 ± 0.0006 (SE 0.0031) | 4.5088 ± 0.3929 (SE 0.3474) / 5.6582 ± 0.1414 (SE 0.4015) |
| baseline | O | 6–10 | 0* | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) | 5* | 0.0313 ± 0.0021 (SE 0.0140) / 0.0462 ± 0.0075 (SE 0.0326) | 10.0000 ± 8.6603 (SE 10.0000) / 11.6667 ± 10.4083 (SE 7.2648) |
| baseline | entity | 6–10 | 1294 | 0.0200 ± 0.0009 (SE 0.0045) / 0.0201 ± 0.0013 (SE 0.0032) | 2.3238 ± 0.6033 (SE 0.2716) / 2.3721 ± 0.2093 (SE 0.2736) | 1825 | 0.0155 ± 0.0001 (SE 0.0013) / 0.0182 ± 0.0006 (SE 0.0013) | 3.9680 ± 0.4827 (SE 0.2878) / 4.2588 ± 0.1860 (SE 0.2944) |
| baseline | O | >10 | 0* | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) | 0* | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) |
| baseline | entity | >10 | 797 | 0.0150 ± 0.0005 (SE 0.0068) / 0.0099 ± 0.0001 (SE 0.0028) | 1.7496 ± 0.4202 (SE 0.2997) / 1.4865 ± 0.5124 (SE 0.2255) | 692 | 0.0086 ± 0.0004 (SE 0.0011) / 0.0091 ± 0.0004 (SE 0.0008) | 2.5811 ± 0.4102 (SE 0.3646) / 2.3202 ± 0.3842 (SE 0.3278) |
| scorer | O | 1 | 38276 | 0.0365 (SE 0.0009) / 0.0365 (SE 0.0009) | 0.7998 (SE 0.0255) / 0.8140 (SE 0.0255) | 35769 | 0.0647 (SE 0.0011) / 0.0617 (SE 0.0011) | 0.7626 (SE 0.0281) / 0.6771 (SE 0.0271) |
| scorer | entity | 1 | 1264 | 0.3411 (SE 0.0215) / 0.3837 (SE 0.0234) | 16.3008 (SE 0.8595) / 18.2918 (SE 0.9075) | 1891 | 0.3160 (SE 0.0143) / 0.2869 (SE 0.0123) | 23.5325 (SE 0.8201) / 23.1183 (SE 0.8066) |
| scorer | O | 2 | 73 | 0.0172 (SE 0.0106) / 0.0088 (SE 0.0032) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 2678 | 0.0272 (SE 0.0022) / 0.0298 (SE 0.0026) | 0.5570 (SE 0.1135) / 0.5072 (SE 0.1119) |
| scorer | entity | 2 | 917 | 0.0613 (SE 0.0068) / 0.0824 (SE 0.0096) | 5.9796 (SE 0.6608) / 6.6158 (SE 0.6839) | 1262 | 0.0637 (SE 0.0046) / 0.0695 (SE 0.0048) | 8.7559 (SE 0.6752) / 10.3969 (SE 0.7268) |
| scorer | O | 3 | 0* | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) | 261 | 0.0165 (SE 0.0035) / 0.0183 (SE 0.0039) | 0.7663 (SE 0.5408) / 0.1916 (SE 0.1352) |
| scorer | entity | 3 | 508 | 0.0458 (SE 0.0066) / 0.0451 (SE 0.0069) | 6.8077 (SE 0.9680) / 7.1112 (SE 1.0023) | 818 | 0.0379 (SE 0.0037) / 0.0397 (SE 0.0035) | 6.7237 (SE 0.7683) / 8.2213 (SE 0.8375) |
| scorer | O | 4–5 | 0* | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) | 86 | 0.0122 (SE 0.0043) / 0.0139 (SE 0.0051) | 2.3256 (SE 1.6347) / 2.3256 (SE 1.3481) |
| scorer | entity | 4–5 | 1077 | 0.0447 (SE 0.0057) / 0.0389 (SE 0.0046) | 4.6309 (SE 0.5418) / 4.4568 (SE 0.5146) | 1421 | 0.0202 (SE 0.0013) / 0.0314 (SE 0.0029) | 5.9905 (SE 0.5383) / 6.7793 (SE 0.5698) |
| scorer | O | 6–10 | 0* | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) | 5* | 0.0189 (SE 0.0117) / 0.0172 (SE 0.0111) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | entity | 6–10 | 1294 | 0.0193 (SE 0.0033) / 0.0191 (SE 0.0027) | 2.5213 (SE 0.3563) / 2.7660 (SE 0.3656) | 1825 | 0.0139 (SE 0.0010) / 0.0148 (SE 0.0008) | 4.1301 (SE 0.3905) / 4.7671 (SE 0.4224) |
| scorer | O | >10 | 0* | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) | 0* | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) |
| scorer | entity | >10 | 797 | 0.0151 (SE 0.0063) / 0.0101 (SE 0.0022) | 1.3540 (SE 0.3085) / 1.5109 (SE 0.3313) | 692 | 0.0081 (SE 0.0009) / 0.0074 (SE 0.0005) | 2.5711 (SE 0.4856) / 3.4140 (SE 0.5946) |

Distances count words from the focal position; null draws are excluded. Paired row counts are shown for every bin.

## 1b-C: seen gold entities

| Model | Group | Bin | CoNLL n | CoNLL S sel / rand | CoNLL flips % sel / rand | CoNLL original error % | Few n | Few S sel / rand | Few flips % sel / rand | Few original error % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | entity | unseen | 4282 | 0.1248 ± 0.0017 (SE 0.0067) / 0.1529 ± 0.0011 (SE 0.0070) | 5.7226 ± 0.0132 (SE 0.2197) / 6.7755 ± 0.0582 (SE 0.2317) | 12.4007 ± 0.6318 (SE 0.4742) | 6695 | 0.1094 ± 0.0011 (SE 0.0039) / 0.1393 ± 0.0014 (SE 0.0041) | 8.3290 ± 0.2462 (SE 0.2013) / 10.2751 ± 0.0635 (SE 0.2217) | 21.6679 ± 0.7813 (SE 0.4717) |
| baseline | entity | seen_different_type | 78 | 0.1281 ± 0.0063 (SE 0.0506) / 0.1804 ± 0.0108 (SE 0.0614) | 8.3333 ± 1.0509 (SE 1.9280) / 11.6987 ± 1.0509 (SE 2.3633) | 27.3504 ± 3.2264 (SE 4.7797) | 97 | 0.1081 ± 0.0027 (SE 0.0226) / 0.1234 ± 0.0120 (SE 0.0238) | 10.6529 ± 2.4189 (SE 1.6179) / 13.1443 ± 0.7839 (SE 1.9352) | 47.7663 ± 4.1664 (SE 4.7339) |
| baseline | entity | seen_same_type | 3413 | 0.0684 ± 0.0002 (SE 0.0049) / 0.0897 ± 0.0004 (SE 0.0058) | 3.3072 ± 0.0612 (SE 0.1848) / 4.0238 ± 0.1850 (SE 0.2064) | 5.1275 ± 0.2113 (SE 0.3426) | 1274 | 0.0872 ± 0.0003 (SE 0.0083) / 0.0780 ± 0.0011 (SE 0.0074) | 5.2983 ± 0.1132 (SE 0.3820) / 5.3637 ± 0.7116 (SE 0.3765) | 9.5500 ± 0.4532 (SE 0.7534) |
| baseline | entity | no_gold_entity | 359 | 0.1529 ± 0.0160 (SE 0.0146) / 0.1897 ± 0.0095 (SE 0.0181) | 11.3974 ± 0.5871 (SE 0.9650) / 12.4884 ± 0.2318 (SE 0.9572) | 62.3955 ± 1.8266 (SE 2.4141) | 664 | 0.1318 ± 0.0025 (SE 0.0112) / 0.1559 ± 0.0037 (SE 0.0125) | 13.0773 ± 0.6579 (SE 0.7450) / 14.2696 ± 0.4403 (SE 0.7910) | 87.2992 ± 1.7841 (SE 1.1218) |
| scorer | entity | unseen | 4282 | 0.1010 (SE 0.0054) / 0.1231 (SE 0.0057) | 6.1858 (SE 0.2580) / 7.2659 (SE 0.2684) | 12.9612 (SE 0.5133) | 6695 | 0.0901 (SE 0.0032) / 0.1113 (SE 0.0033) | 9.7535 (SE 0.2547) / 11.9268 (SE 0.2723) | 25.7655 (SE 0.5345) |
| scorer | entity | seen_different_type | 78 | 0.1058 (SE 0.0377) / 0.1434 (SE 0.0457) | 8.3333 (SE 2.1357) / 12.0192 (SE 2.4825) | 25.6410 (SE 4.9761) | 97 | 0.0908 (SE 0.0194) / 0.0966 (SE 0.0172) | 11.2113 (SE 2.0108) / 13.2732 (SE 2.4359) | 50.5155 (SE 5.1028) |
| scorer | entity | seen_same_type | 3413 | 0.0587 (SE 0.0039) / 0.0758 (SE 0.0047) | 3.6661 (SE 0.2184) / 4.4719 (SE 0.2419) | 6.8561 (SE 0.4326) | 1274 | 0.0786 (SE 0.0078) / 0.0736 (SE 0.0068) | 5.8379 (SE 0.4574) / 6.8681 (SE 0.4822) | 11.5385 (SE 0.8954) |
| scorer | entity | no_gold_entity | 359 | 0.1637 (SE 0.0171) / 0.1893 (SE 0.0192) | 19.1156 (SE 1.4856) / 18.2451 (SE 1.3803) | 100.0000 (SE 0.0000) | 664 | 0.1309 (SE 0.0129) / 0.1469 (SE 0.0118) | 19.2206 (SE 1.0725) / 20.5384 (SE 1.0898) | 100.0000 (SE 0.0000) |

Seen means an exact case-sensitive gold span string in classifier training. Same type additionally matches the entity type. Scorer-predicted entity rows with gold O appear as no_gold_entity.

## 1b-D: word-form ambiguity

| Model | Group | Bin | CoNLL n | CoNLL S sel / rand | CoNLL flips % sel / rand | CoNLL share % | Few n | Few S sel / rand | Few flips % sel / rand | Few share % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | unseen_word | 3438 | 0.1212 ± 0.0198 (SE 0.0067) / 0.1228 ± 0.0192 (SE 0.0068) | 4.0430 ± 0.6928 (SE 0.2336) / 3.9679 ± 0.6130 (SE 0.2346) | 8.9758 | 2469 | 0.2815 ± 0.0228 (SE 0.0108) / 0.2829 ± 0.0222 (SE 0.0112) | 13.1126 ± 1.6756 (SE 0.5525) / 12.7818 ± 1.4247 (SE 0.5499) | 6.5482 |
| baseline | O | one_type | 23566 | 0.0531 ± 0.0102 (SE 0.0017) / 0.0535 ± 0.0102 (SE 0.0017) | 1.5356 ± 0.3002 (SE 0.0461) / 1.5653 ± 0.2893 (SE 0.0460) | 61.5252 | 13703 | 0.0554 ± 0.0103 (SE 0.0020) / 0.0508 ± 0.0084 (SE 0.0019) | 1.3029 ± 0.2313 (SE 0.0751) / 1.2257 ± 0.1636 (SE 0.0749) | 36.3427 |
| baseline | O | two_or_more_types | 11299 | 0.0465 ± 0.0098 (SE 0.0027) / 0.0448 ± 0.0096 (SE 0.0027) | 1.2704 ± 0.3198 (SE 0.0742) / 1.2140 ± 0.3006 (SE 0.0731) | 29.4990 | 21533 | 0.1216 ± 0.0123 (SE 0.0028) / 0.1195 ± 0.0118 (SE 0.0028) | 3.2446 ± 0.4393 (SE 0.0994) / 3.1020 ± 0.3909 (SE 0.0977) | 57.1091 |
| baseline | entity | unseen_word | 2715 | 0.1232 ± 0.0008 (SE 0.0076) / 0.1592 ± 0.0013 (SE 0.0087) | 6.2400 ± 0.1866 (SE 0.2783) / 7.5123 ± 0.0645 (SE 0.3013) | 33.3866 | 2771 | 0.0985 ± 0.0003 (SE 0.0056) / 0.1312 ± 0.0017 (SE 0.0058) | 8.6656 ± 0.2856 (SE 0.3157) / 10.6294 ± 0.0807 (SE 0.3457) | 31.7411 |
| baseline | entity | one_type | 3159 | 0.0942 ± 0.0013 (SE 0.0071) / 0.0997 ± 0.0018 (SE 0.0068) | 4.0994 ± 0.0209 (SE 0.2291) / 4.3922 ± 0.1182 (SE 0.2295) | 38.8465 | 1433 | 0.1073 ± 0.0008 (SE 0.0084) / 0.1329 ± 0.0014 (SE 0.0093) | 7.3883 ± 0.2512 (SE 0.4037) / 9.4964 ± 0.3199 (SE 0.4776) | 16.4147 |
| baseline | entity | two_or_more_types | 2258 | 0.0888 ± 0.0007 (SE 0.0066) / 0.1309 ± 0.0011 (SE 0.0082) | 4.7129 ± 0.0477 (SE 0.2624) / 6.1430 ± 0.2472 (SE 0.2992) | 27.7668 | 4526 | 0.1138 ± 0.0014 (SE 0.0048) / 0.1311 ± 0.0011 (SE 0.0048) | 8.3140 ± 0.3462 (SE 0.2469) / 9.5697 ± 0.3849 (SE 0.2611) | 51.8442 |
| scorer | O | unseen_word | 3438 | 0.0868 (SE 0.0044) / 0.0883 (SE 0.0045) | 2.0070 (SE 0.1456) / 2.1197 (SE 0.1489) | 8.9758 | 2469 | 0.1557 (SE 0.0059) / 0.1553 (SE 0.0061) | 4.0198 (SE 0.2705) / 3.7566 (SE 0.2548) | 6.5482 |
| scorer | O | one_type | 23566 | 0.0333 (SE 0.0010) / 0.0331 (SE 0.0010) | 0.7633 (SE 0.0305) / 0.7712 (SE 0.0301) | 61.5252 | 13703 | 0.0370 (SE 0.0011) / 0.0323 (SE 0.0010) | 0.3667 (SE 0.0299) / 0.3092 (SE 0.0296) | 36.3427 |
| scorer | O | two_or_more_types | 11299 | 0.0277 (SE 0.0015) / 0.0277 (SE 0.0015) | 0.5056 (SE 0.0373) / 0.5034 (SE 0.0373) | 29.4990 | 21533 | 0.0663 (SE 0.0015) / 0.0650 (SE 0.0015) | 0.6542 (SE 0.0300) / 0.5735 (SE 0.0287) | 57.1091 |
| scorer | entity | unseen_word | 2715 | 0.0954 (SE 0.0058) / 0.1274 (SE 0.0070) | 7.2007 (SE 0.3496) / 8.2643 (SE 0.3544) | 33.3866 | 2771 | 0.0762 (SE 0.0045) / 0.0986 (SE 0.0042) | 9.6310 (SE 0.3905) / 12.1301 (SE 0.4228) | 31.7411 |
| scorer | entity | one_type | 3159 | 0.0840 (SE 0.0060) / 0.0834 (SE 0.0053) | 4.6890 (SE 0.2706) / 5.3300 (SE 0.2819) | 38.8465 | 1433 | 0.0889 (SE 0.0065) / 0.1099 (SE 0.0074) | 9.2725 (SE 0.5408) / 10.8252 (SE 0.5673) | 16.4147 |
| scorer | entity | two_or_more_types | 2258 | 0.0778 (SE 0.0055) / 0.1133 (SE 0.0072) | 5.3809 (SE 0.3213) / 6.4604 (SE 0.3492) | 27.7668 | 4526 | 0.1018 (SE 0.0044) / 0.1138 (SE 0.0042) | 10.2988 (SE 0.3200) / 12.0194 (SE 0.3371) | 51.8442 |

Ambiguity counts distinct training types per case-sensitive word, ignoring B/I. Few-NERD has O plus eight coarse types; CoNLL has O plus four.

## 1b-B: dropped-position distances

| Group | CoNLL mean sel / rand | CoNLL median sel / rand | CoNLL within 2 words % sel / rand | Few mean sel / rand | Few median sel / rand | Few within 2 words % sel / rand |
| --- | --- | --- | --- | --- | --- | --- |
| O | 11.0403 / 11.0403 | 8.0000 / 8.0000 | 15.7966 / 15.7966 | 13.7395 / 13.8304 | 10.0000 / 10.0000 | 12.4699 / 12.2912 |
| entity | 12.8788 / 12.9088 | 9.0000 / 9.0000 | 15.9139 / 16.1384 | 14.8957 / 14.8091 | 11.0000 / 11.0000 | 12.1692 / 11.8527 |

Distributions pool dropped word positions over both selector folds, weighting every dropped position equally.

## 1b-B: size- and distance-matched S ratio

| Model | Group | CoNLL ratio | CoNLL cells | CoNLL retained mass % | Few ratio | Few cells | Few retained mass % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | O | 0.9981 ± 0.0045 (SE 0.0137) | 6 | 99.9134 | 1.0161 ± 0.0074 (SE 0.0106) | 9 | 99.1379 |
| baseline | entity | 1.1367 ± 0.0111 (SE 0.1109) | 27 | 94.8212 | 1.0001 ± 0.0068 (SE 0.0379) | 34 | 98.1539 |
| scorer | O | 1.0042 (SE 0.0143) | 6 | 99.9134 | 1.0366 (SE 0.0106) | 9 | 99.1379 |
| scorer | entity | 1.0955 (SE 0.1011) | 27 | 94.8212 | 1.0339 (SE 0.0386) | 34 | 98.1539 |

Cell ratios use common rows, then selector cell-frequency weights. Cells below 50 rows or with zero random S are omitted; remaining weights are renormalized. Row SE holds weights fixed and accounts for rows appearing in multiple cells.

## 1b-D: baseline flip concentration

| Ranking | Top % | CoNLL n | CoNLL flip share % sel / rand | Few n | Few flip share % sel / rand |
| --- | --- | --- | --- | --- | --- |
| selector | 1 | 465 | 28.8214 ± 1.4382 (SE 1.0362) / 20.6392 ± 0.2772 (SE 0.9374) | 465 | 21.2181 ± 1.2659 (SE 0.7845) / 17.6038 ± 1.0766 (SE 0.7072) |
| selector | 5 | 2322 | 69.6175 ± 3.0231 (SE 0.7326) / 45.0337 ± 0.4776 (SE 0.9547) | 2322 | 61.7748 ± 1.3080 (SE 0.7402) / 50.1771 ± 0.9528 (SE 0.8217) |
| selector | 10 | 4644 | 92.8195 ± 1.9233 (SE 0.3743) / 60.5358 ± 1.2446 (SE 0.8456) | 4644 | 79.9026 ± 0.9378 (SE 0.5204) / 63.6486 ± 1.0108 (SE 0.7319) |
| random | 1 | 465 | 20.7870 ± 0.4668 (SE 0.9852) / 27.1217 ± 1.2761 (SE 0.9880) | 465 | 18.5986 ± 0.8077 (SE 0.7449) / 20.0001 ± 0.6914 (SE 0.7523) |
| random | 5 | 2322 | 48.0015 ± 0.7384 (SE 0.9729) / 69.1456 ± 2.9661 (SE 0.7029) | 2322 | 52.6196 ± 0.8050 (SE 0.8296) / 59.3490 ± 1.1077 (SE 0.7499) |
| random | 10 | 4644 | 64.4501 ± 1.3228 (SE 0.8237) / 92.7859 ± 1.8717 (SE 0.3419) | 4644 | 67.4164 ± 1.1424 (SE 0.7087) / 79.2956 ± 1.0820 (SE 0.5273) |

Each seed ranks rows by draw-averaged S. Both masks are evaluated on the same top rows for each ranking. Row SE conditions on the ranking.

## 1b-D: frequent forms in the baseline top 1%

| Rank | CoNLL selector | CoNLL random | Few selector | Few random |
| --- | --- | --- | --- | --- |
| 1 | CUP (19) | CUP (20) | of (40) | of (28) |
| 2 | L (18) | L (17) | the (18) | the (21) |
| 3 | AT (12) | Ago (6) | . (12) | and (13) |
| 4 | He (7) | GA (6) | ; (8) | ; (11) |
| 5 | Ago (6) | WESTERN (6) | The (7) | 's (8) |
| 6 | G (5) | of (5) | and (6) | `` (6) |
| 7 | ICE (5) | ICE (5) | A (6) | The (6) |
| 8 | W (5) | Internet (4) | II (5) | A (6) |
| 9 | GA (5) | Steers (4) | , (5) | amp (6) |
| 10 | WESTERN (5) | Harris (4) | 's (5) | II (5) |
| 11 | T (5) | G (4) | ) (5) | de (5) |
| 12 | Group (5) | GMT (4) | '' (5) | . (4) |
| 13 | Court (4) | He (4) | `` (4) | ) (4) |
| 14 | of (4) | AT (4) | Best (4) | 2nd (4) |
| 15 | Internet (4) | McLean (3) | Navy (4) | , (4) |
| 16 | Harris (4) | Court (3) | 2nd (3) | : (4) |
| 17 | Division (4) | . (3) | New (3) | '' (4) |
| 18 | Hooper (3) | New (3) | Year (3) | election (4) |
| 19 | . (3) | UNION (3) | de (3) | ' (3) |
| 20 | GMT (3) | Wednesday (3) | in (3) | & (3) |

Lists use seed-averaged S and count each focal-word row once. Seed-specific lists are retained in CSVs.

[Implementation](IMPLEMENTATION_11.md). [All tables](tables/fewnerd/). Cached fills and per-row arrays: `results/fewnerd_replication/generate/` and `measure/`. Native label inventories, training populations and calibration γ differ as specified; measurement definitions are unchanged.
