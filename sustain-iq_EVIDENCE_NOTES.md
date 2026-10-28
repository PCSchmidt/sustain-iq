# Sustain-IQ: evidence notes

Companion to `sustain-iq.md`. Clone: fresh copy of GitHub PCSchmidt/sustain-iq at HEAD ef36a4f (23 Oct 2026), read-only. `git status` was empty before and after all runs. Labels: Confirmed, Ran it, Per the docs, My reading, Ask the author.

## E1 Fact sheet

| Fact | Value | Label |
| --- | --- | --- |
| Name, licence | Sustain-IQ; MIT; public; 0 stars, 0 forks (gh api) | Confirmed |
| Kind | Study plus reference implementation (Python host pipeline, three Rust no_std crates, a firmware bring-up crate); no app, no service | Confirmed |
| Status | README: "phase 1 (host side) complete and phase 2 prepared on the host; nothing has run on a microcontroller yet" | Per the docs; consistent with code |
| Size | Python library 611 lines; scripts 841; Python tests 326. Rust lib: tinyinfer 171, tinyfeat 129 (+272 generated tables), tinyadxl 193; Rust tests 97 + 205 + 313; pico2-bringup src 23 + feature_probe | Confirmed (wc -l) |
| Commits | 5: 1b84ce6 (17 Oct, phase 1), 5264db7 (17 Oct), 1b9edd4 (18 Oct), 34285c6 (18 Oct), ef36a4f (23 Oct). All authored "Chris Schmidt", all with trailer "Co-Authored-By: Claude ..." (name withheld from the guide by rule) | Confirmed |
| Tests | Python 32 pass; Rust 19 pass (tinyinfer 3 unit + 3 golden; tinyfeat 2; tinyadxl 11) = 51 | Ran it |
| Needs | numpy, scipy, scikit-learn (dev), pytest; Rust stable; CWRU download of 64 files, about 190 MB, from Case Western's site (for studies only) | Per the docs / pyproject |
| Deployed? | No. Hardware phase not started; design notes say an ADXL345 "is on order"; no board status stated | Per the docs; Ask the author |
| Related | physical-ai-portfolio (index), sentinel-node, triagebox (measurement habits) | Per the docs |
| Agent instruction files | None found (no CLAUDE.md, AGENTS.md, .cursor etc.) | Confirmed (git ls-files) |

## E2 Reading and running log

Read: README, all four docs (results, design, roadmap, bring-up checklist), pico2-bringup README, ci.yml, pyproject, signals.py, quant.py, tinyinfer lib and golden test, test names across Python and Rust, run_experiments.py (structure and lock study), all five results JSON files, git log. Not read line by line: data.py, cwru_files.py, mlp.py, export.py, tinyfeat/tinyadxl source, scripts other than run_experiments, the SVG figure beyond its description.

Commands run (venv in the session scratchpad; Rust target dir in the scratchpad):
- `python -m pytest -q -p no:cacheprovider` from the clone root (pyproject sets pythonpath=src, so no install into the clone): 32 passed in 16 s, Python 3.14.6 (CI uses 3.10 and 3.12; not tried here). Ran it.
- `cargo test --offline` in firmware/tinyinfer, tinyfeat, tinyadxl: all pass (6, 2, 11). Dependencies resolved from the local cargo cache. Ran it.
- `gh run list` and `gh api repos/...` for CI and stars. Ran it.
- Chart values recomputed from results/phase1_leak.json, phase1_lock.json, phase1_bandwidth.json; results.md tables for size ladder and confusion matrices compared with phase1_ladder.json and phase1_leak.json (all matched at the quoted rounding). Ran it (L3 recompute from saved files).

Not run: scripts/download_cwru.py (190 MB network download), run_experiments.py (needs that data), export_golden.py and export_feature_golden.py (they write test vectors into the repo and need the data), the thumbv8m cross-compile builds (target not installed; network), anything on hardware, the 2,004-window tinyfeat check (data not committed; 29 MB file not in repo). So the 52-window tinyfeat agreement and the 305-case tinyinfer agreement were run only against committed vectors; regenerating the vectors from the model was not.

## E3 Claims check

| ID | Claim (source) | Evidence | Level | Verdict |
| --- | --- | --- | --- | --- |
| CL-01 | 305 of 305 logits and classes match (README, results.md, commit) | golden.json has 305 inputs, 305 logits; Rust test passes | L2 against committed vectors; vectors not regenerated | Supported |
| CL-02 | Random split 99.9, leave-one-load-out 98.7, leave-one-size-out 51.9 (README) | phase1_leak.json: 0.9992, 0.9873, 0.5193 | L3 | Supported |
| CL-03 | Locked set 43.5% (absolute MLP); range 39-56% across variants | phase1_lock.json: 0.4345; min 0.3874 (shape MLP int8), max 0.5618 (envelope logistic) | L3 | Supported |
| CL-04 | Detection 100% in honest protocols, 97-100% locked | bandwidth json detect_float 1.0 at 12 kHz both protocols; lock fault_detected 0.973-1.0; at lower rates 0.9996-1.0 (README rounds to 100) | L3 | Supported, with rounding |
| CL-05 | RMS threshold 95-99.6% at full rate; 38-71% at lower rates | detect_rms_threshold: 0.996 load, 0.948 size at 12 kHz; unseen-load values 0.709, 0.549, 0.675, 0.381 | L3 | Supported |
| CL-06 | 172-parameter model 224 bytes flash, 97.6% across loads; 700 params 99.1% | ladder json: 224; 0.9765; 700 params 0.9907 | L3 | Supported |
| CL-07 | int8 costs at most 0.3 points; int4 0.1 to 3 | Largest int8 gap 0.31 (0.9765 to 0.9734); int4 gaps 0.1 to 3.0 | L3 | Supported |
| CL-08 | 1.8 KB flash, 96 bytes RAM, 1,600 MACs (deployed model) | ladder json flash 1824, RAM 96, MACs 1600; Rust golden test asserts the same arithmetic | L3 / Confirmed; accounting only | Supported as arithmetic |
| CL-09 | Rust features match Python: 52 committed windows, 2,004 local windows, 64,096 of 64,096 int8 identical | Committed 52-window test passes; 2,004-window run not reproducible (data not committed) | L2 for 52; docs only for 2,004 | Partly checked |
| CL-10 | Flash about 7.6 KB and RAM about 12.4 KB for the feature image (design.md) | Linker output not reproduced (target not installed) | Docs only | Not checked |
| CL-11 | Locked set evaluated only after other studies, never used to select; run twice with identical numbers (results.md) | Single commit contains everything; cannot verify order | Docs only | Cannot be verified from history |
| CL-12 | Feature sets designed after seeing the baseline's cross-size failure (results.md) | Disclosed by the project | Docs | Disclosed |
| CL-13 | Confusion recalls 0.29 / 0.58 / 0.30 across size | 812/2838, 1647/2840, 841/2844 from the saved matrix | L3 | Supported |
| CL-14 | Leak inflates by about a point | 0.9992 vs 0.9873 (1.2 points) | L3 | Supported |
| CL-15 | "Part of physical-ai-portfolio; applies habits from Sentinel Node and TriageBox" | README sentence | Docs | Per the docs |
| CL-16 | Rust crate is `no_std`, integer only, no allocation | `#![no_std]` in lib.rs; uses slices only | Confirmed | Supported |
| CL-17 | SPI mode 3 and 3.9 mg/LSB "confirmed in datasheet text" (ef36a4f, checklist) | Not independently checked (datasheet not fetched) | Docs only | See E14 |

## E4 Rule attacks

Not run beyond the existing tests. The rules are (a) splits keep whole recordings apart (tests test_leave_one_load_out..., test_leave_one_size_out..., test_random_split_is_leaky... read and spot-checked), (b) standardiser uses training statistics (test_standardizer_fits_train_statistics_only, name only), (c) Rust equals Python integers (golden test, read). No bad-input probes were made; the programs take no outside input beyond the downloaded files.

## E5 Code map

- src/sustain_iq: signals.py (decimate, windows, band_edges, band_features, shape_features, envelope features), data.py (loading, splits, balanced accuracy), cwru_files.py (file table), mlp.py (numpy MLP, Adam, balanced loss), quant.py (fixed_point, QModel, quantize, footprint), export.py (golden vectors).
- scripts: download_cwru.py, run_experiments.py (leak, ladder, bandwidth, features, lock), export_golden.py, export_feature_golden.py, gen_tinyfeat_tables.py, results_tables.py, make_figure.py.
- firmware: tinyinfer (Layer, Mlp, requantize, argmax), tinyfeat (Extractor, generated tables), tinyadxl (Adxl345 driver over embedded-hal 1 SpiDevice, simulated chip in tests/sim.rs), pico2-bringup (blink; feature_probe).

## E6 Model call sites

One learned model: numpy MLP 32-32-16-4 (mlp.py, trained in run_experiments.py and export_golden.py), ReLU, class-balanced cross-entropy, Adam, 60 epochs, seeds in SEEDS (3). Comparison model: scikit-learn logistic regression (from the research/dev extra). No pre-trained model, no outside call, no cost. CNN was not tried (design.md states the choice).

## E7 Tests and CI

- 51 tests (E1). Spot checks: (1) tinyinfer `every_logit_and_class_matches_the_python_reference` builds the network from exported integers, loops all 305 inputs and asserts exact equality of logits and class. It does depend on vectors produced by the same Python code, so it checks cross-language agreement, not correctness of the quantisation against float (that is covered by the Python test `test_int8_model_agrees_with_float_on_almost_every_sample`, name only). (2) `test_leave_one_size_out_holds_out_a_diameter_and_time_splits_normal_with_a_gap` asserts fault diameters on each side differ and that healthy train windows end before healthy test windows start; real assertions. (3) tinyinfer `requantize_rounds_half_up_and_clamps` asserts concrete values (1 to 1, 3 to 2, negative to 0, saturates 127); real arithmetic.
- tinyadxl tests run against a simulator written to the same reading of the datasheet (design.md says so): consistency only.
- CI (.github/workflows/ci.yml): python job on 3.10 and 3.12 (`pip install -e ".[dev]"`, `pytest -q`); rust job per crate (fmt check, `clippy -D warnings`, `cargo test`); firmware-build job (release builds for thumbv8m.main-none-eabihf, plus pico2-bringup). `gh run list`: 5 CI runs and 1 "Dependency Graph" dynamic run, all success; CI durations 25, 25, 29, 45, 56 s; dates 17 to 23 Oct 2026. CI runs no study and does not download CWRU. Coverage not measured.
- Python tests run without CWRU data (32 pass with no data folder).

## E8 Plan versus results and history

All of phase 1 (library, studies, results, Rust engine, docs) is in commit 1b84ce6 (17 Oct 2026, 09:07 -0400). No pre-registration or earlier plan commit exists, so the order of "locked set" versus choices cannot be verified from history; the claim rests on results.md. Commit credits: 5 of 5 carry an AI co-author trailer. No gates or approval records in the repo.

## E10 Environment

Python deps are lower-bounded (numpy>=1.26, scipy>=1.11), no lockfile; results are not stamped with library versions. Rust has Cargo.lock for each crate. This run used the latest numpy/scipy/scikit-learn in a fresh venv on Python 3.14.6 and all 32 tests passed. Study outputs were not regenerated, so whether a fresh install reproduces the JSON numbers is untested.

## E11 Security and privacy

Grep for secret-like words across tracked files returned no matches beyond ordinary text; no .env in the repo; no history scan beyond the five commits' messages. No outside services other than the CWRU download. No claims made about any live deployment (none exists).

## E12 Shortcomings register

| ID | Shortcoming | Evidence | Severity | Disclosed? | Options | Recommendation |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | Single lab rig, large seeded faults | results.md Limitations | High | Yes | (a) own-rig data per roadmap 0.7, gain real evidence, cost hardware and time; (b) add another public set, cheaper, still lab | Do (a); (b) as stopgap |
| S2 | Diagnosis fails across fault size/position | CL-02, CL-03, confusion matrix | High | Yes | (a) ship detection-only (design.md open decision); (b) add training diversity or physics-based features, uncertain payoff | (a), then measure (b) |
| S3 | Nothing measured on a device; flash/RAM/MACs are arithmetic, FFT cost unmeasured | design.md, README | High | Yes | (a) run bring-up checklist; (b) estimate with a cycle-accurate simulator, less convincing | (a) |
| S4 | Locked-set order unverifiable | single commit | Medium | Partly (results.md states the sequence) | (a) publish a dated plan before any future locked evaluation; (b) leave and note | (a) for future studies |
| S5 | Only 3 seeds and a few folds; no confidence intervals | results.md | Medium | Yes | More seeds or bootstrap over recordings | Bootstrap |
| S6 | Healthy side of unseen-size test is time-split inside recordings; locked set lacks healthy data | results.md | Medium | Yes | Add healthy recordings to the locked set or hold out a healthy load | Add |
| S7 | Sensor-bandwidth study uses an ideal filter | design.md Open decisions | Medium | Yes | Model the ADXL345's analog bandwidth | Do before trusting 3.2 kHz |
| S8 | No Python lockfile; no recorded versions | pyproject | Low | No | Add a lock or print versions into results JSON | Small |
| S9 | int4 simulated, Rust does int8 only | results.md | Low | Yes | Leave | Leave |

## E14 Discrepancies

- docs/design.md (Host-side preparation, ADXL345 paragraph) says the datasheet PDF "could not be fetched" and that the SPI mode and 3.9 mg scale "remain unverified", and the Phase 2 section says FPU suitability "is not verified". docs/bringup-checklist.md and the latest commit (ef36a4f) mark SPI mode 3, 3.9 mg/LSB and the RP2350 FPU as "confirmed" from datasheet text. The newer statement appears to supersede the older paragraph, which was not updated. Guide mentions this once.
- README item 5 says detection "100%" at 750 Hz; the JSON shows 0.9996. Rounding only.
- README says "100% under every honest protocol"; this holds for the full-rate, whole-recording protocols, and the unseen-size healthy side is optimistic (disclosed).

## E15 Production readiness (trait checklist)

| Trait | Rating | Evidence |
| --- | --- | --- |
| Run from clean start | Partly | README commands; 32 tests pass in a fresh venv; studies need a 190 MB download; no Python lockfile |
| Checks block a bad change | Partly | CI runs tests, lint, format, cross-build; no evaluation gate on results |
| Quality measured against baselines | Yes (on one dataset) | logistic regression and RMS threshold reported |
| Safe on failure | Not applicable / No | no runtime system exists |
| Human in the loop | Not applicable | no actions taken |
| Versioned prompts/models/settings | Partly | model integers exported; settings in code; no prompts |
| Logs and monitoring | No | none |
| Data and secrets | Partly | public data; no secrets seen |

Placement: Works locally (host only). Gap to deployable: a device run, measured cost, real data, packaging, monitoring, a security review; the roadmap lists most of these as planned.

## E17 Completeness

| Promised (roadmap/README) | Status | Evidence |
| --- | --- | --- |
| 0.1 host pipeline | Done and tested | src, tests |
| 0.2 studies | Done; results saved; not re-run | results JSON |
| 0.3 bit-exact Rust engine | Done and tested | golden test |
| 0.4 device bring-up | Planned; driver and checklist prepared | tinyadxl tests, checklist |
| 0.5 on-device features | Host side done; nothing on a board | tinyfeat tests |
| 0.6 on-device inference and energy | Planned | none |
| 0.7 own rig data | Planned | none |
| 0.8 detection-only and anomaly variants | Planned | none |
| Gateway / cloud path | Planned (optional) | none |

## Picture sources

images/sustain-iq-scores.png and images/sustain-iq-bandwidth.png: plotted by matplotlib from results/phase1_leak.json (mlp_float means for the first three bars), phase1_lock.json (abs, mlp_float balanced_accuracy) and phase1_bandwidth.json (unseen-load detect_float, float diagnosis, detect_rms_threshold). images/sustain-iq-pipeline.png is a drawn block diagram of the pipeline in docs/design.md; its bracket labels summarise tinyadxl, tinyfeat and tinyinfer tests. No planned feature is drawn as built.

## Picture sources (revised after the reader check)

All three images were regenerated with matplotlib from the saved JSON in results/ (script kept outside the repo, not committed).
- sustain-iq-scores.png: bars are mlp_float means from phase1_leak.json (99.92, 98.73, 51.93, spreads 0.04, 1.21, 3.02) and phase1_lock.json abs/mlp_float (43.45, no spread). Diamonds are the logistic regression (99.50, 98.29, 61.40 with spread 11.92 from phase1_leak.json; 45.32 on the locked set). Chance lines: 25% (four classes) for the first three bars and 33% (three fault classes; the locked set has no healthy recordings) for the locked bar. Colours explained in the legend.
- sustain-iq-bandwidth.png: unseen-motor-load rows of phase1_bandwidth.json (detect_float 1.0 to 0.9996; detect_rms_threshold 99.6, 70.9, 54.9, 67.5, 38.1; float diagnosis 98.7, 97.3, 93.3, 85.9, 91.7). The 750 Hz uptick is annotated as noise because docs/results.md says "with this few folds that is noise, not a trend" (README: "within the noise"). The 1,500 Hz uptick of the threshold line is not annotated; the docs do not comment on it.
- sustain-iq-pipeline.png: drawn block diagram; text now says FFT, with a legend for box and bracket colours; the sensor-driver bracket covers only the sensor/driver box.

Edit note: guide rewritten per the reader check to about 3,750 words. Facts unchanged. The 305-case figure appears once (results chapter); 51 tests once with a pointer. The stuck-sensor and driver-timing items were kept, the register/watermark and Clippy details were dropped from the body.


## Review addendum (independent accuracy review)

Re-run in a scratch copy of HEAD ef36a4f: Python 32 passed; Rust 3+3 (tinyinfer), 2 (tinyfeat), 11 (tinyadxl) = 19 passed (cargo --offline). gh run list: 5 CI runs plus 1 Dependency Graph run, all success. 0 stars, 0 forks, MIT, public. All chart values, spreads, ladder rows, locked-set variants (38.7 to 56.2 across nine; plain MLP 43.5, plain logistic regression 45.3) and confusion recalls recomputed from results/*.json and matched. Locked set 24 files confirmed (16 outer-race, 4 ball, 4 inner-race). Detection by RMS threshold on the unseen-load test is 70.9, 54.9, 67.5, 38.1% at 6,000 to 750 Hz (on the unseen-size test it is 48 to 80%). The plain logistic regression names 98.6% (not 100%) of larger inner-race locked windows; the guide now says 'feature networks'. Design.md vs latest commit: design.md (ADXL345 paragraph and Board bullet) still says the datasheet could not be fetched and SPI mode, 3.9 mg scale and FPU are unverified; bringup-checklist.md and ef36a4f mark them confirmed from datasheet text (commit names an ADXL345 Rev. 0 copy). The guide describes this correctly.
