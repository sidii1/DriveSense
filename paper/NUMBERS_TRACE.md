# NUMBERS_TRACE

Every result macro used in the abstract and Sections 1-6 of `main.tex`, with its value and origin. Macros are written by the scripts named below; nothing is typed by hand except the items under *Literal numbers*.

## Abstract

| macro | value | origin |
|---|---|---|
| `\LeakNuisAUC` | 1.00 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakAllAUC` | 0.96 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedAP` | 0.97 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\WFusedAUC` | 0.92 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAUCLo` | 0.91 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAUCHi` | 0.93 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WKinAUC` | 0.87 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleBoxAUC` | 0.45 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\TFlowInc` | 70 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |

## Explainable Anticipation Model

| macro | value | origin |
|---|---|---|
| `\EvalHorizon` | 2.0 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Data and Splits

| macro | value | origin |
|---|---|---|
| `\EvalNCrash` | 1500 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\EvalNNormal` | 3000 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\EvalNTrain` | 3600 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\EvalNTest` | 900 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\EvalNTestCrash` | 300 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Metrics

| macro | value | origin |
|---|---|---|
| `\EvalFAR` | 10 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Leakage Test and Within-Accident Protocol

| macro | value | origin |
|---|---|---|
| `\EvalFAR` | 10 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Control Comparison and Audit

| macro | value | origin |
|---|---|---|
| `\PilotN` | 1500 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotDay` | 1325 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotNight` | 175 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotNormalW` | 1141 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotSnow` | 235 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotRain` | 124 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotOnsetMin` | 3.0 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotOnsetMax` | 4.9 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |

## Audit: Which Cues Survive Accident Footage?

| macro | value | origin |
|---|---|---|
| `\LaneRate` | 9.3 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\LaneRateSnow` | 4.3 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\LaneClipsAny` | 37 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TDetEarly` | 1 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TDetLate` | 0.7 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TDetDec` | 48 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TDetP` | $<$0.001 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TVehEarly` | 2.4 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TVehLate` | 1 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TTCClipsAny` | 1195 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\PilotN` | 1500 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TInvTTCDiff` | +0 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TInvTTCInc` | 33 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TInvTTCDec` | 39 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\AlertEarly` | 11.2 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\AlertLate` | 10.9 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowEarly` | 1.14 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowLate` | 1.75 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowDiff` | +0.427 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowLo` | 0.347 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowHi` | 0.536 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowInc` | 70 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TFlowP` | $<$0.001 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TBrightEarly` | 0.446 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |
| `\TBrightLate` | 0.442 | paper/scripts/pilot.py <- experiments/exp3_ccd/results/{frame_features_all.csv,temporal_tests.csv} |

## Source Leakage

| macro | value | origin |
|---|---|---|
| `\LeakNuisAUC` | 1.00 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakNuisLo` | 1.00 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakNuisHi` | 1.00 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\BitrateCrash` | 105 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\BitrateNormal` | 369 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\BitrateRatio` | 3.5 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\ResOtherCrash` | 35 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\NoiseCrash` | 0.0 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\NoiseNormal` | 1.5 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\LeakSceneAUC` | 0.93 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakSceneLo` | 0.91 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakSceneHi` | 0.95 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakAllAUC` | 0.96 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakAllLo` | 0.95 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakAllHi` | 0.97 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakKinAUC` | 0.73 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\LeakMotionAUC` | 0.70 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\MatchPropAUC` | 1.000 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\MatchPairs` | 12 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |

## Crash-Versus-Normal Anticipation

| macro | value | origin |
|---|---|---|
| `\EvalFAR` | 10 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleTau` | 1.11 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleTunedAP` | 0.40 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleTunedRecall` | 16 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleFixedRecall` | 49 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleFixedFAR` | 34 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\RuleDivAP` | 0.39 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedAP` | 0.97 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedAPLo` | 0.96 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedAPHi` | 0.98 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedRecall` | 95 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\FusedmTTA` | 2.92 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DeltaAP` | 0.58 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DeltaAPLo` | 0.53 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DeltaAPHi` | 0.61 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\SceneAP` | 0.91 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\SceneRecall` | 88 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\KinAP` | 0.79 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\MotionAP` | 0.66 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapSceneShare` | 48 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapKinematicShare` | 36 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapMotionShare` | 16 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapTopOne` | dark\_channel | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapTopTwo` | sharpness | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\ShapTopThree` | n\_tracked | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CurveFusedRecallFarLo` | 92 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CurveFusedRecallFarHi` | 99 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CurveRuleTunedRecallFarLo` | 10 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CurveRuleTunedRecallFarHi` | 37 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Within-Accident Anticipation

| macro | value | origin |
|---|---|---|
| `\WNTest` | 300 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleBoxAUC` | 0.45 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleBoxAUCLo` | 0.43 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleBoxAUCHi` | 0.48 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleDivAUC` | 0.56 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleDivAUCLo` | 0.54 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WRuleDivAUCHi` | 0.59 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WMotionTTCAUC` | 0.82 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WKinAUC` | 0.87 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WMotionAUC` | 0.79 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WSceneAUC` | 0.80 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WKinMotionAUC` | 0.91 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAUC` | 0.92 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAUCLo` | 0.91 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAUCHi` | 0.93 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedAP` | 0.86 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedRecall` | 90 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedFAR` | 7 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\WFusedmTTA` | 1.39 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |
| `\EvalFAR` | 10 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |

## Are Pre-Accident Changes Accident-Specific?

| macro | value | origin |
|---|---|---|
| `\DynFlowMagCenterAccidentEarly` | 1.36 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynFlowMagCenterAccidentLate` | 2.19 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynFlowMagCenterNormalEarly` | 1.53 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynFlowMagCenterNormalLate` | 1.59 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CtrlFlowRrb` | 0.33 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\CtrlFlowP` | $<$0.001 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\CtrlDivPRrb` | 0.32 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\DynHasDetAccidentEarly` | 0.81 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynHasDetAccidentLate` | 0.59 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynHasDetNormalEarly` | 0.90 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynHasDetNormalLate` | 0.89 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CtrlDetRrb` | -0.37 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\CtrlVehRrb` | -0.41 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |
| `\DynMaxInvTtcPathAccidentEarly` | 0.17 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\DynMaxInvTtcPathAccidentLate` | 0.14 | experiments/exp3_ccd/04_anticipation_eval.py <- results/eval/eval_summary.json |
| `\CtrlInvTTCRrb` | -0.11 | experiments/exp3_ccd/09_extra_macros.py <- results/nuisance_*.csv, results/eval/{eval_summary.json,control_comparison.csv} |

## Cost and Camera-Observable Context

| macro | value | origin |
|---|---|---|
| `\RtDetect` | 44 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |
| `\RtScene` | 21 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |
| `\RtFlow` | 30 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |
| `\RtLane` | 5 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |
| `\RtTotal` | 100 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |
| `\RtFPS` | 10 | experiments/exp3_ccd/08_runtime.py <- results/eval/runtime.json |

## Conclusion

| macro | value | origin |
|---|---|---|
| `\WFusedAUC` | 0.92 | experiments/exp3_ccd/06_within_clip.py <- results/eval/within_clip.json |

## Literal numbers typed in the text

| number | meaning | source |
|---|---|---|
| 1.19 million | road deaths per year | WHO Global Status Report on Road Safety 2023 (ref. 1) |
| 1,500 / 3,000 / 50 frames / 10 fps / 5 s | CCD composition | CCD README (`data/CarCrash/README_CCD.md`), verified by `00b_build_crash_table.py` and `results/eval/clips_used.csv` |
| 0.25, 960, 0.22W, 0.6 s, 0.02 s^-1, 4 samples | detector / TTC settings | `experiments/exp3_ccd/config.py` |
| Farnebaeck parameters, central region | flow settings | `experiments/exp3_ccd/perception/flow.py` |
| 300 trees, depth 4, lr 0.05 | XGBoost settings | `04_anticipation_eval.py::fit_xgb` |
| 10 % FAR, 1,000 bootstrap, 2 s horizon, 20 % held-out | protocol | `04_anticipation_eval.py` defaults (`--far --boot --horizon`), `06_within_clip.py` constants |
| 14 cues | Holm family size of the audit | `results/temporal_tests.csv` (rows) |
| 3 % | largest relative shift of photometric cues in the audit (sharpness 593 -> 575) | `generated_pilot.tex` (TSharpEarly/TSharpLate) |
| 51,641; 11.2 %; 0.612 [0.595, 0.631]; 0.684 [0.668, 0.701]; +0.072 [0.058, 0.088]; 0.763 [0.749, 0.775]; ~70 % | CRSS 2024 injury model | `experiments/exp1_crss/results/evaluation_metrics.json` (`drivesense_tiers`, `delta_tier2_all_vs_tier1`), `crss_data_summary.json`; 70 % = (0.684-0.5)/(0.763-0.5) |
