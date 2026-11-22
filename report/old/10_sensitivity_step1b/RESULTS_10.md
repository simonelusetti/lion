# Step 1b: diagnostics of in-domain sensitivity

Fills were **regenerated** with the original procedure, seeds, batch layout and four draws, because step 1 saved only focal vectors. Filled token IDs are now saved. Generation: **35.4 min**; measurement: **0.2 min**; analysis: **0.2 min**; wall-clock: **1.11 h**. Nothing was trained or selected. All analyses below use this single regenerated set.

Only CoNLL test, baseline (three seeds) and scorer appear in this report. Augmented arms and WNUT/WikiANN are in the CSVs; transfer rows are marked UNCERTIFIED. All tables put selector/random side by side. S is gamma=.01 KL over nine native tags, in nats. Flip/error/restoration/concentration percentages use a factor of 100. Baseline values are mean ± sample SD across seeds, followed by row SE. Scorer has no seed SD. Neither folds nor draws count as additional independent rows. Bins below 50 paired rows have an asterisk; matched ratios are not computed for those cells.

## Reproduction of step 1

| Mask | Rows | Step 1 saved S | Regenerated S | Difference |
|---|---:|---:|---:|---:|
| selector | 8132 | 0.10237256 | 0.10237256 | +1.155e-09 |
| random | 8132 | 0.12825142 | 0.12825142 | -7.253e-10 |

The supplied rounded baseline-entity references were selector 0.1024 and random 0.1283. The largest absolute difference from the saved full-precision step-1 values is 1.155e-09; differences from the rounded references are also saved in reproduction.csv.

## A. Actual changes and non-null sensitivity

| Model | Group | Full / paired non-null rows | Excluded sel / rand | Dropped words sel / rand | Changed words sel / rand | Restored % sel / rand | Null swaps % sel / rand | Non-null S sel / rand | Non-null flips % sel / rand |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| baseline | O | 38303 / 38278 | 25 / 23 | 22.782 / 22.782 | 22.250 / 22.249 | 2.336 / 2.339 | 0.082 / 0.079 | 0.0573 ± 0.0109 (SE 0.0015) / 0.0572 ± 0.0108 (SE 0.0014) | 1.6835 ± 0.3403 (SE 0.0417) / 1.6784 ± 0.3211 (SE 0.0416) |
| baseline | entity | 8132 / 7524 | 514 / 542 | 3.560 / 3.560 | 2.055 / 2.288 | 42.265 / 35.733 | 18.317 / 14.440 | 0.1255 ± 0.0011 (SE 0.0050) / 0.1546 ± 0.0009 (SE 0.0056) | 6.0999 ± 0.0615 (SE 0.1788) / 7.1030 ± 0.1157 (SE 0.1921) |
| scorer | O | 38303 / 38278 | 25 / 23 | 22.782 / 22.782 | 22.250 / 22.249 | 2.336 / 2.339 | 0.082 / 0.079 | 0.0365 (SE 0.0009) / 0.0365 (SE 0.0009) | 0.7994 (SE 0.0255) / 0.8138 (SE 0.0255) |
| scorer | entity | 8132 / 7524 | 514 / 542 | 3.560 / 3.560 | 2.055 / 2.288 | 42.265 / 35.733 | 18.317 / 14.440 | 0.1069 (SE 0.0041) / 0.1290 (SE 0.0046) | 7.1161 (SE 0.2234) / 7.9817 (SE 0.2286) |

Restored percentages count identical restored word occurrences among all dropped-word occurrences. Null swaps mean every dropped word was restored, including zero-drop draws. Non-null estimates average eligible draws within each fold, then available fold means within the word. The table uses words with non-null support for both masks; the two mask-specific all-null exclusions are shown before this additional pairing. No null draw contributes zero to the non-null average.

## B1. Changed-count and nearest-change bins

| Model | Group | Bin | Paired rows | S sel / rand | Flips % sel / rand |
|---|---|---|---:|---|---|
| baseline | O | changed_count: 0* | 37 | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) |
| baseline | O | changed_count: 1 | 338 | 0.1631 ± 0.0665 (SE 0.0249) / 0.1507 ± 0.0662 (SE 0.0221) | 7.9265 ± 2.9993 (SE 1.1961) / 7.4745 ± 2.8645 (SE 1.1371) |
| baseline | O | changed_count: 2 | 335 | 0.2090 ± 0.0425 (SE 0.0285) / 0.1962 ± 0.0456 (SE 0.0255) | 8.3085 ± 1.0988 (SE 1.0923) / 7.8151 ± 1.1275 (SE 1.0462) |
| baseline | O | changed_count: 3 | 592 | 0.1853 ± 0.0343 (SE 0.0244) / 0.1811 ± 0.0299 (SE 0.0245) | 5.5767 ± 1.0387 (SE 0.6600) / 5.4031 ± 0.9521 (SE 0.6448) |
| baseline | O | changed_count: 4 | 797 | 0.1928 ± 0.0285 (SE 0.0228) / 0.1890 ± 0.0272 (SE 0.0222) | 5.2523 ± 0.9934 (SE 0.5881) / 5.3273 ± 0.9758 (SE 0.5998) |
| baseline | O | changed_count: 5–9 | 7552 | 0.0799 ± 0.0140 (SE 0.0043) / 0.0804 ± 0.0137 (SE 0.0044) | 2.2884 ± 0.4081 (SE 0.1234) / 2.2546 ± 0.3824 (SE 0.1224) |
| baseline | O | changed_count: 10+ | 29084 | 0.0433 ± 0.0086 (SE 0.0013) / 0.0434 ± 0.0087 (SE 0.0013) | 1.2422 ± 0.2707 (SE 0.0362) / 1.2574 ± 0.2557 (SE 0.0367) |
| baseline | O | nearest_changed: 1 | 38276 | 0.0573 ± 0.0109 (SE 0.0015) / 0.0572 ± 0.0108 (SE 0.0014) | 1.6830 ± 0.3409 (SE 0.0417) / 1.6777 ± 0.3217 (SE 0.0415) |
| baseline | O | nearest_changed: 2 | 73 | 0.0393 ± 0.0069 (SE 0.0248) / 0.0432 ± 0.0188 (SE 0.0298) | 1.8265 ± 0.4943 (SE 1.4379) / 1.3699 ± 1.1416 (SE 1.0153) |
| baseline | O | nearest_changed: 3* | 0 | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) |
| baseline | O | nearest_changed: 4–5* | 0 | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) |
| baseline | O | nearest_changed: 6–10* | 0 | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) |
| baseline | O | nearest_changed: >10* | 0 | nan ± nan (SE nan) / nan ± nan (SE nan) | nan ± nan (SE nan) / nan ± nan (SE nan) |
| baseline | entity | changed_count: 0 | 1590 | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) | 0.0000 ± 0.0000 (SE 0.0000) / 0.0000 ± 0.0000 (SE 0.0000) |
| baseline | entity | changed_count: 1 | 3741 | 0.1410 ± 0.0002 (SE 0.0079) / 0.1789 ± 0.0013 (SE 0.0092) | 6.9222 ± 0.1240 (SE 0.2858) / 8.3170 ± 0.1807 (SE 0.3151) |
| baseline | entity | changed_count: 2 | 2711 | 0.1032 ± 0.0026 (SE 0.0085) / 0.1402 ± 0.0020 (SE 0.0108) | 5.5945 ± 0.0795 (SE 0.3251) / 6.3829 ± 0.2132 (SE 0.3552) |
| baseline | entity | changed_count: 3 | 2123 | 0.0929 ± 0.0037 (SE 0.0099) / 0.1081 ± 0.0038 (SE 0.0109) | 4.7869 ± 0.2160 (SE 0.3471) / 4.9831 ± 0.3135 (SE 0.3473) |
| baseline | entity | changed_count: 4 | 1569 | 0.1147 ± 0.0033 (SE 0.0149) / 0.1104 ± 0.0025 (SE 0.0116) | 5.3095 ± 0.0836 (SE 0.4283) / 5.8654 ± 0.4030 (SE 0.4654) |
| baseline | entity | changed_count: 5–9 | 1308 | 0.1105 ± 0.0023 (SE 0.0130) / 0.1067 ± 0.0020 (SE 0.0115) | 5.3538 ± 0.1991 (SE 0.4472) / 5.4133 ± 0.2942 (SE 0.4439) |
| baseline | entity | changed_count: 10+ | 116 | 0.0151 ± 0.0003 (SE 0.0051) / 0.0890 ± 0.0025 (SE 0.0535) | 0.9339 ± 0.6584 (SE 0.5930) / 2.6700 ± 0.2696 (SE 1.3300) |
| baseline | entity | nearest_changed: 1 | 1264 | 0.4247 ± 0.0084 (SE 0.0268) / 0.4834 ± 0.0067 (SE 0.0296) | 14.8635 ± 0.2989 (SE 0.7587) / 16.6513 ± 0.4485 (SE 0.8032) |
| baseline | entity | nearest_changed: 2 | 917 | 0.0655 ± 0.0018 (SE 0.0068) / 0.0790 ± 0.0025 (SE 0.0090) | 5.1194 ± 0.2263 (SE 0.4861) / 5.9297 ± 0.5934 (SE 0.5566) |
| baseline | entity | nearest_changed: 3 | 508 | 0.0435 ± 0.0010 (SE 0.0054) / 0.0445 ± 0.0002 (SE 0.0063) | 4.2186 ± 0.4735 (SE 0.6120) / 4.3772 ± 0.1799 (SE 0.6609) |
| baseline | entity | nearest_changed: 4–5 | 1077 | 0.0418 ± 0.0012 (SE 0.0052) / 0.0384 ± 0.0027 (SE 0.0051) | 3.5464 ± 0.4493 (SE 0.3616) / 3.4239 ± 0.7072 (SE 0.3667) |
| baseline | entity | nearest_changed: 6–10 | 1294 | 0.0200 ± 0.0009 (SE 0.0045) / 0.0201 ± 0.0013 (SE 0.0032) | 2.3238 ± 0.6033 (SE 0.2716) / 2.3721 ± 0.2093 (SE 0.2736) |
| baseline | entity | nearest_changed: >10 | 797 | 0.0150 ± 0.0005 (SE 0.0068) / 0.0099 ± 0.0001 (SE 0.0028) | 1.7496 ± 0.4202 (SE 0.2997) / 1.4865 ± 0.5124 (SE 0.2255) |
| scorer | O | changed_count: 0* | 37 | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | O | changed_count: 1 | 338 | 0.0469 (SE 0.0087) / 0.0394 (SE 0.0063) | 1.1095 (SE 0.4277) / 0.8876 (SE 0.3519) |
| scorer | O | changed_count: 2 | 335 | 0.1158 (SE 0.0151) / 0.1077 (SE 0.0132) | 3.5075 (SE 0.6108) / 3.6816 (SE 0.6100) |
| scorer | O | changed_count: 3 | 592 | 0.1084 (SE 0.0142) / 0.1106 (SE 0.0144) | 2.9842 (SE 0.4635) / 3.2517 (SE 0.4752) |
| scorer | O | changed_count: 4 | 797 | 0.1293 (SE 0.0152) / 0.1253 (SE 0.0145) | 3.7275 (SE 0.4828) / 3.5550 (SE 0.4496) |
| scorer | O | changed_count: 5–9 | 7552 | 0.0569 (SE 0.0025) / 0.0578 (SE 0.0026) | 1.2133 (SE 0.0807) / 1.2624 (SE 0.0804) |
| scorer | O | changed_count: 10+ | 29084 | 0.0272 (SE 0.0008) / 0.0271 (SE 0.0008) | 0.5708 (SE 0.0219) / 0.5649 (SE 0.0214) |
| scorer | O | nearest_changed: 1 | 38276 | 0.0365 (SE 0.0009) / 0.0365 (SE 0.0009) | 0.7998 (SE 0.0255) / 0.8140 (SE 0.0255) |
| scorer | O | nearest_changed: 2 | 73 | 0.0172 (SE 0.0106) / 0.0088 (SE 0.0032) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | O | nearest_changed: 3* | 0 | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) |
| scorer | O | nearest_changed: 4–5* | 0 | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) |
| scorer | O | nearest_changed: 6–10* | 0 | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) |
| scorer | O | nearest_changed: >10* | 0 | nan (SE nan) / nan (SE nan) | nan (SE nan) / nan (SE nan) |
| scorer | entity | changed_count: 0 | 1590 | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) | 0.0000 (SE 0.0000) / 0.0000 (SE 0.0000) |
| scorer | entity | changed_count: 1 | 3741 | 0.1236 (SE 0.0064) / 0.1536 (SE 0.0075) | 8.5104 (SE 0.3708) / 9.8369 (SE 0.3821) |
| scorer | entity | changed_count: 2 | 2711 | 0.0825 (SE 0.0064) / 0.1113 (SE 0.0085) | 5.7174 (SE 0.3734) / 6.6335 (SE 0.4009) |
| scorer | entity | changed_count: 3 | 2123 | 0.0752 (SE 0.0077) / 0.0940 (SE 0.0092) | 4.7751 (SE 0.3946) / 5.6210 (SE 0.4304) |
| scorer | entity | changed_count: 4 | 1569 | 0.0889 (SE 0.0118) / 0.0889 (SE 0.0098) | 5.3856 (SE 0.5053) / 5.2475 (SE 0.4823) |
| scorer | entity | changed_count: 5–9 | 1308 | 0.0932 (SE 0.0111) / 0.0790 (SE 0.0083) | 5.5619 (SE 0.5141) / 5.4632 (SE 0.5084) |
| scorer | entity | changed_count: 10+ | 116 | 0.0147 (SE 0.0065) / 0.0572 (SE 0.0350) | 0.5388 (SE 0.3870) / 2.3707 (SE 1.2979) |
| scorer | entity | nearest_changed: 1 | 1264 | 0.3411 (SE 0.0215) / 0.3837 (SE 0.0234) | 16.3008 (SE 0.8595) / 18.2918 (SE 0.9075) |
| scorer | entity | nearest_changed: 2 | 917 | 0.0613 (SE 0.0068) / 0.0824 (SE 0.0096) | 5.9796 (SE 0.6608) / 6.6158 (SE 0.6839) |
| scorer | entity | nearest_changed: 3 | 508 | 0.0458 (SE 0.0066) / 0.0451 (SE 0.0069) | 6.8077 (SE 0.9680) / 7.1112 (SE 1.0023) |
| scorer | entity | nearest_changed: 4–5 | 1077 | 0.0447 (SE 0.0057) / 0.0389 (SE 0.0046) | 4.6309 (SE 0.5418) / 4.4568 (SE 0.5146) |
| scorer | entity | nearest_changed: 6–10 | 1294 | 0.0193 (SE 0.0033) / 0.0191 (SE 0.0027) | 2.5213 (SE 0.3563) / 2.7660 (SE 0.3656) |
| scorer | entity | nearest_changed: >10 | 797 | 0.0151 (SE 0.0063) / 0.0101 (SE 0.0022) | 1.3540 (SE 0.3085) / 1.5109 (SE 0.3313) |

Each estimate uses the same words for both masks: a word needs at least one draw in that bin for each mask (after fold averaging). Nearest-change bins omit null swaps. Paired row counts vary by bin and are recorded above; no draw is counted as an independent observation.

## B2. Distances of dropped words

| Group | Dropped positions sel / rand | Mean distance sel / rand | Median sel / rand | Within 2 words % sel / rand |
|---|---:|---:|---:|---:|
| O | 1745258.000 / 1745258.000 | 11.040 / 11.040 | 8.000 / 8.000 | 15.797 / 15.797 |
| entity | 57899.000 / 57899.000 | 12.879 / 12.909 | 9.000 / 9.000 | 15.914 / 16.138 |

These distributions pool dropped word positions from both folds; every dropped position receives equal weight. They are mask properties and do not depend on classifier seed.

## B3. Size- and distance-matched S ratios

| Model | Group | Matched ratio | Seed SD | Row SE | Cells | Selector non-null mass retained % |
|---|---|---:|---:|---:|---:|---:|
| baseline | O | 0.9981 | 0.0045 | 0.0137 | 6 | 99.91 |
| baseline | entity | 1.1367 | 0.0111 | 0.1109 | 27 | 94.82 |
| scorer | O | 1.0042 | — | 0.0143 | 6 | 99.91 |
| scorer | entity | 1.0955 | — | 0.1011 | 27 | 94.82 |

Ratios are computed within changed-count × nearest-distance cells on common words, then averaged using selector cell frequencies. Cells below 50 paired words or with zero random S are omitted; weights are renormalized over retained cells. Row SE uses a delta method with cell weights held fixed, accounting for words shared across cells. Retained mass and cell counts describe the support used.

## C. Gold entity spans seen in training

| Model | Category | Rows | Original error % | S sel / rand | Flips % sel / rand |
|---|---|---:|---|---|---|
| baseline | unseen | 4282 | 12.4007 ± 0.6318 (SE 0.4742) | 0.1248 ± 0.0017 (SE 0.0067) / 0.1529 ± 0.0011 (SE 0.0070) | 5.7226 ± 0.0132 (SE 0.2197) / 6.7755 ± 0.0582 (SE 0.2317) |
| baseline | seen_different_type | 78 | 27.3504 ± 3.2264 (SE 4.7797) | 0.1281 ± 0.0063 (SE 0.0506) / 0.1804 ± 0.0108 (SE 0.0614) | 8.3333 ± 1.0509 (SE 1.9280) / 11.6987 ± 1.0509 (SE 2.3633) |
| baseline | seen_same_type | 3413 | 5.1275 ± 0.2113 (SE 0.3426) | 0.0684 ± 0.0002 (SE 0.0049) / 0.0897 ± 0.0004 (SE 0.0058) | 3.3072 ± 0.0612 (SE 0.1848) / 4.0238 ± 0.1850 (SE 0.2064) |
| baseline | no_gold_entity | 359 | 62.3955 ± 1.8266 (SE 2.4141) | 0.1529 ± 0.0160 (SE 0.0146) / 0.1897 ± 0.0095 (SE 0.0181) | 11.3974 ± 0.5871 (SE 0.9650) / 12.4884 ± 0.2318 (SE 0.9572) |
| scorer | unseen | 4282 | 12.9612 (SE 0.5133) | 0.1010 (SE 0.0054) / 0.1231 (SE 0.0057) | 6.1858 (SE 0.2580) / 7.2659 (SE 0.2684) |
| scorer | seen_different_type | 78 | 25.6410 (SE 4.9761) | 0.1058 (SE 0.0377) / 0.1434 (SE 0.0457) | 8.3333 (SE 2.1357) / 12.0192 (SE 2.4825) |
| scorer | seen_same_type | 3413 | 6.8561 (SE 0.4326) | 0.0587 (SE 0.0039) / 0.0758 (SE 0.0047) | 3.6661 (SE 0.2184) / 4.4719 (SE 0.2419) |
| scorer | no_gold_entity | 359 | 100.0000 (SE 0.0000) | 0.1637 (SE 0.0171) / 0.1893 (SE 0.0192) | 19.1156 (SE 1.4856) / 18.2451 (SE 1.3803) |

Groups remain the scorer-predicted groups from step 1. A scorer-predicted entity word whose gold tag is O has no gold entity span and is shown separately. Seen membership uses exact case-sensitive gold span word strings anywhere in T1 ∪ T2, excluding scorer split A. Original error compares the full native BIO tag, including B/I.

## D1. Training word-form ambiguity

| Model | Group | Word-form bin | Rows | Share of group % | S sel / rand | Flips % sel / rand |
|---|---|---|---:|---:|---|---|
| baseline | O | unseen_word | 3438 | 8.98 | 0.1212 ± 0.0198 (SE 0.0067) / 0.1228 ± 0.0192 (SE 0.0068) | 4.0430 ± 0.6928 (SE 0.2336) / 3.9679 ± 0.6130 (SE 0.2346) |
| baseline | O | one_type | 23566 | 61.53 | 0.0531 ± 0.0102 (SE 0.0017) / 0.0535 ± 0.0102 (SE 0.0017) | 1.5356 ± 0.3002 (SE 0.0461) / 1.5653 ± 0.2893 (SE 0.0460) |
| baseline | O | two_or_more_types | 11299 | 29.50 | 0.0465 ± 0.0098 (SE 0.0027) / 0.0448 ± 0.0096 (SE 0.0027) | 1.2704 ± 0.3198 (SE 0.0742) / 1.2140 ± 0.3006 (SE 0.0731) |
| baseline | entity | unseen_word | 2715 | 33.39 | 0.1232 ± 0.0008 (SE 0.0076) / 0.1592 ± 0.0013 (SE 0.0087) | 6.2400 ± 0.1866 (SE 0.2783) / 7.5123 ± 0.0645 (SE 0.3013) |
| baseline | entity | one_type | 3159 | 38.85 | 0.0942 ± 0.0013 (SE 0.0071) / 0.0997 ± 0.0018 (SE 0.0068) | 4.0994 ± 0.0209 (SE 0.2291) / 4.3922 ± 0.1182 (SE 0.2295) |
| baseline | entity | two_or_more_types | 2258 | 27.77 | 0.0888 ± 0.0007 (SE 0.0066) / 0.1309 ± 0.0011 (SE 0.0082) | 4.7129 ± 0.0477 (SE 0.2624) / 6.1430 ± 0.2472 (SE 0.2992) |
| scorer | O | unseen_word | 3438 | 8.98 | 0.0868 (SE 0.0044) / 0.0883 (SE 0.0045) | 2.0070 (SE 0.1456) / 2.1197 (SE 0.1489) |
| scorer | O | one_type | 23566 | 61.53 | 0.0333 (SE 0.0010) / 0.0331 (SE 0.0010) | 0.7633 (SE 0.0305) / 0.7712 (SE 0.0301) |
| scorer | O | two_or_more_types | 11299 | 29.50 | 0.0277 (SE 0.0015) / 0.0277 (SE 0.0015) | 0.5056 (SE 0.0373) / 0.5034 (SE 0.0373) |
| scorer | entity | unseen_word | 2715 | 33.39 | 0.0954 (SE 0.0058) / 0.1274 (SE 0.0070) | 7.2007 (SE 0.3496) / 8.2643 (SE 0.3544) |
| scorer | entity | one_type | 3159 | 38.85 | 0.0840 (SE 0.0060) / 0.0834 (SE 0.0053) | 4.6890 (SE 0.2706) / 5.3300 (SE 0.2819) |
| scorer | entity | two_or_more_types | 2258 | 27.77 | 0.0778 (SE 0.0055) / 0.1133 (SE 0.0072) | 5.3809 (SE 0.3213) / 6.4604 (SE 0.3492) |

Ambiguity counts distinct training types O/PER/LOC/ORG/MISC per case-sensitive focal word, ignoring B/I. The share denominator is the corresponding scorer-predicted group.

## D2. Baseline flip concentration

| Ranking by S | Top rows % | Top row count | Selector flip share % | Random flip share % |
|---|---:|---:|---|---|
| selector | 1 | 465 | 28.82 ± 1.44 (SE 1.04) | 20.64 ± 0.28 (SE 0.94) |
| selector | 5 | 2322 | 69.62 ± 3.02 (SE 0.73) | 45.03 ± 0.48 (SE 0.95) |
| selector | 10 | 4644 | 92.82 ± 1.92 (SE 0.37) | 60.54 ± 1.24 (SE 0.85) |
| random | 1 | 465 | 20.79 ± 0.47 (SE 0.99) | 27.12 ± 1.28 (SE 0.99) |
| random | 5 | 2322 | 48.00 ± 0.74 (SE 0.97) | 69.15 ± 2.97 (SE 0.70) |
| random | 10 | 4644 | 64.45 ± 1.32 (SE 0.82) | 92.79 ± 1.87 (SE 0.34) |

Rows are ranked separately by each mask’s S for each seed. For each ranking, both masks are evaluated on the same selected rows. These are shares of all flips, using draw-averaged flips as weights. Row SE conditions on the ranking; CSVs also contain O/entity-specific concentration.

## D3. Frequent forms in the baseline’s top 1%

| Rank | Selector-ranked word (count) | Random-ranked word (count) |
|---|---|---|
| 1 | CUP (19) | CUP (20) |
| 2 | L (18) | L (17) |
| 3 | AT (12) | Ago (6) |
| 4 | He (7) | GA (6) |
| 5 | Ago (6) | WESTERN (6) |
| 6 | G (5) | of (5) |
| 7 | ICE (5) | ICE (5) |
| 8 | W (5) | Internet (4) |
| 9 | GA (5) | Steers (4) |
| 10 | WESTERN (5) | Harris (4) |
| 11 | T (5) | G (4) |
| 12 | Group (5) | GMT (4) |
| 13 | Court (4) | He (4) |
| 14 | of (4) | AT (4) |
| 15 | Internet (4) | McLean (3) |
| 16 | Harris (4) | Court (3) |
| 17 | Division (4) | . (3) |
| 18 | Hooper (3) | New (3) |
| 19 | . (3) | UNION (3) |
| 20 | GMT (3) | Wednesday (3) |

These counts use the top 1% under seed-averaged S, once per word row. Seed-specific lists and the augmented-arm results are retained in the CSVs. No model or mask is selected from these rankings.

[Implementation](IMPLEMENTATION_10.md); [all CSVs](tables/sensitivity_1b/). Filled IDs and word-slot offsets are under `results/sensitivity_1b/generate/`; per-draw measurements under `results/sensitivity_1b/measure/`.
