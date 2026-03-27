# Bearing Condition Monitor — Version 2 Results

## Purpose

This document summarises the results from Version 2 of the Bearing Condition Monitor project.

Version 2 was developed to make the bearing fault diagnosis problem more realistic and more demanding than Version 1 by expanding the dataset to multiple fault sizes, adding richer engineered signal features, and using stronger model comparison and validation.

## Version 2 Scope

Version 2 retained the same four bearing condition classes used in Version 1:

- Normal
- Inner race fault
- Ball fault
- Outer race fault

However, the dataset scope was expanded substantially.

The analysis used 12 kHz drive-end vibration data from the CWRU bearing dataset with:

- normal data at loads 0, 1, 2, and 3
- inner race faults at 0.007, 0.014, and 0.021 inch
- ball faults at 0.007, 0.014, and 0.021 inch
- outer race faults at 0.007, 0.014, and 0.021 inch
- outer race faults fixed at the 6:00 position only

This gave a controlled 40-file Version 2 dataset:

- 4 normal files
- 12 inner race fault files
- 12 ball fault files
- 12 outer race fault files

So compared with Version 1, Version 2 moved from a single-fault-size benchmark to a broader multi-severity diagnosis problem while still keeping the dataset engineering-focused and well controlled.

## Dataset Construction

As in Version 1, the raw vibration signals were segmented into overlapping windows and converted into engineered feature vectors.

To keep the comparison fair across files, the dataset was balanced at the source-file level by sampling the same number of windows from each raw signal file. The shortest file contributed 117 windows, so Version 2 used:

- 117 windows per source file
- 40 source files in total
- 4680 samples overall

This produced:

- 468 normal samples
- 1404 inner race samples
- 1404 ball samples
- 1404 outer race samples

This is an important difference from Version 1. In Version 2, equal windows were taken from each source file, but because there are three fault sizes for each faulty condition and only one healthy file per load, the final class distribution is no longer perfectly class-balanced. For that reason, macro F1 score was reported alongside accuracy so that model performance was not judged only by overall classification rate.

## Engineered Features

Version 2 compared two feature sets.

### Baseline Feature Set

The baseline feature set retained the original Version 1-style engineered features:

- Mean
- Standard deviation
- RMS
- Peak-to-peak amplitude
- Crest factor
- Shape factor
- Impulse factor
- Clearance factor
- Skewness
- Kurtosis
- Dominant frequency

This gave 11 features per signal window.

### Enhanced Feature Set

The enhanced Version 2 feature set included all baseline features plus six additional descriptors:

- Variance
- Absolute mean
- Maximum absolute amplitude
- Dominant spectral amplitude
- Frequency centre
- Spectral entropy

This gave 17 features per signal window.

The purpose of this comparison was not just to make the feature vector larger, but to test whether a richer description of signal magnitude, waveform behaviour, and spectral distribution improved generalisation across fault severities and operating loads.

## Signal Characterisation

Representative raw signal, FFT, and spectrogram inspection was retained in Version 2 to confirm that the healthy and faulty cases still showed physically meaningful differences before machine learning was applied.

As in Version 1, the raw time histories showed differences in amplitude behaviour and waveform shape between normal and faulty bearings. The FFT comparisons showed that the fault cases also differed in frequency content, while the spectrograms provided a time-frequency view of how vibration energy was distributed over time.

These signal-level checks remained important in Version 2 because they confirmed that the expanded multi-severity dataset still contained physically interpretable differences rather than only statistical label separation.

## Models Evaluated

Three classifiers were evaluated in Version 2:

- Logistic Regression
- Random Forest
- XGBoost

Logistic Regression was retained as a linear baseline.

Random Forest was retained as a strong nonlinear ensemble model.

XGBoost was added in Version 2 as a stronger gradient-boosted tree model to test whether additional model capacity improved diagnosis performance on the expanded dataset.

## Held-Out Load Test Results

For the main held-out benchmark, the models were trained using loads 0, 1, and 2 and tested on load 3.

The held-out split contained:

- 3510 training samples
- 1170 test samples

Version 2 compared all three models on both feature sets.

### Held-Out Results Summary

#### Baseline Features

- Logistic Regression: accuracy = 0.752137, macro F1 = 0.738705
- Random Forest: accuracy = 0.943590, macro F1 = 0.953553
- XGBoost: accuracy = 0.977778, macro F1 = 0.980833

#### Enhanced Features

- Logistic Regression: accuracy = 0.985470, macro F1 = 0.987896
- Random Forest: accuracy = 0.965812, macro F1 = 0.971754
- XGBoost: accuracy = 0.989744, macro F1 = 0.991476

### Interpretation of Held-Out Results

The most important result from the held-out comparison is that the enhanced feature set produced a major improvement over the baseline feature set, especially for Logistic Regression.

Logistic Regression improved from 0.752137 accuracy with the baseline features to 0.985470 accuracy with the enhanced features. Its macro F1 score also increased from 0.738705 to 0.987896. This is a large improvement and shows that the richer Version 2 feature set made the class structure much more separable even for a simple linear classifier.

Random Forest was already strong on the baseline feature set and improved more modestly from 0.943590 to 0.965812 accuracy when using the enhanced features.

XGBoost gave the best held-out-load performance overall. It improved from 0.977778 accuracy on the baseline feature set to 0.989744 on the enhanced feature set, with the highest macro F1 score of 0.991476.

This suggests that the largest gain in Version 2 came from richer feature engineering, while the move from Random Forest to XGBoost provided a further but smaller improvement.

## Leave-One-Load-Out Validation

To make the validation stronger, Version 2 also used leave-one-load-out testing. In this setup, each operating load was held out in turn as the test condition, while the remaining three loads were used for training.

This produced a stronger measure of cross-load generalisation than a single held-out split.

### Mean Performance Across All Four Held-Out Loads

#### Baseline Features

- Logistic Regression: mean accuracy = 0.714957, mean macro F1 = 0.725370
- Random Forest: mean accuracy = 0.954701, mean macro F1 = 0.961462
- XGBoost: mean accuracy = 0.963889, mean macro F1 = 0.959068

#### Enhanced Features

- Logistic Regression: mean accuracy = 0.875641, mean macro F1 = 0.873702
- Random Forest: mean accuracy = 0.986325, mean macro F1 = 0.988131
- XGBoost: mean accuracy = 0.986538, mean macro F1 = 0.988772

### Worst-Case Performance Across Loads

The minimum performance across the four leave-one-load-out folds was also examined.

#### Baseline Features

- Logistic Regression: minimum accuracy = 0.335043, minimum macro F1 = 0.355376
- Random Forest: minimum accuracy = 0.885470, minimum macro F1 = 0.900841
- XGBoost: minimum accuracy = 0.885470, minimum macro F1 = 0.861848

#### Enhanced Features

- Logistic Regression: minimum accuracy = 0.536752, minimum macro F1 = 0.523370
- Random Forest: minimum accuracy = 0.965812, minimum macro F1 = 0.971754
- XGBoost: minimum accuracy = 0.962393, minimum macro F1 = 0.968607

### Interpretation of Leave-One-Load-Out Results

The leave-one-load-out results show that the enhanced feature set improved all three model families.

Logistic Regression improved substantially, with mean accuracy increasing from 0.714957 to 0.875641. However, it remained much less robust than the tree-based models, especially in its worst-case fold.

Random Forest remained very strong across all loads. Its mean accuracy increased from 0.954701 to 0.986325 when moving from baseline to enhanced features, and its minimum accuracy also increased from 0.885470 to 0.965812.

XGBoost gave the highest mean performance overall, with mean accuracy 0.986538 and mean macro F1 0.988772 on the enhanced feature set. However, the margin over enhanced Random Forest was very small.

In fact, enhanced Random Forest was slightly stronger in the worst-case sense, with marginally better minimum accuracy and minimum macro F1 than enhanced XGBoost.

So the leave-one-load-out results suggest that:

- enhanced XGBoost is the best model on average
- enhanced Random Forest is almost tied overall
- enhanced Random Forest may be slightly safer in the worst held-out operating condition
- Logistic Regression benefits strongly from better features but remains less robust under operating-condition shift

## Operating-Load Difficulty

The detailed leave-one-load-out results showed that not all held-out loads were equally difficult.

In particular, load 0 was clearly the hardest case for the simpler models. On the baseline feature set, Logistic Regression dropped to 0.335043 accuracy when load 0 was held out. Even with the enhanced feature set, its load-0 performance remained much weaker than its results on the other loads.

This is an important engineering result because it shows that operating-condition shift still matters even after expanding the feature set. The diagnosis problem did not become trivially easy simply because more data and more features were added.

The tree-based models were much more robust to this effect, which is one reason they remain the strongest practical choices in Version 2.

## Engineering Interpretation

Version 2 shows that the original Version 1 workflow was strong, but also that the Version 1 benchmark was relatively easy and highly controlled.

Once the dataset was expanded to multiple fault sizes, the diagnosis task became more realistic and more demanding. This exposed a much larger performance gap between simple and strong models when the baseline feature set was used.

The enhanced feature set then produced the most important improvement in the whole Version 2 study. This suggests that richer signal representation matters more than simply increasing classifier complexity.

That interpretation makes physical sense. The additional features improved the description of:

- amplitude magnitude
- signal spread
- spectral concentration
- spectral complexity

These are all properties that can change with defect type and defect severity, so it is reasonable that they improved classification across a broader multi-fault-size dataset.

The results also show that tree-based models remain the strongest choices for this application. Random Forest and XGBoost both handled the expanded dataset very well, and their leave-one-load-out performance indicates strong cross-load robustness.

The very small difference between enhanced Random Forest and enhanced XGBoost is also useful. It suggests that the project is not simply showing “the most advanced model wins,” but rather that most of the gain came from improving the engineering representation of the vibration signal.

## Overall Conclusion for Version 2

Version 2 successfully extended the Bearing Condition Monitor project from a narrow benchmark into a more realistic and technically stronger bearing fault diagnosis study.

The main outcomes of Version 2 are:

- successful expansion from a single-fault-size dataset to a controlled multi-fault-size dataset
- implementation of two feature-set tiers: baseline and enhanced
- inclusion of a stronger third classifier through XGBoost
- clear evidence that enhanced feature engineering materially improved diagnosis performance
- best mean cross-load performance from enhanced XGBoost
- near-identical and slightly more conservative worst-case robustness from enhanced Random Forest
- clear evidence that operating load still influences difficulty, especially for simpler models

Overall, Version 2 shows that the project has moved beyond a simple proof-of-concept benchmark. The workflow now demonstrates stronger engineering realism, more rigorous validation, and a clearer comparison between feature quality and model complexity.

These results still do not on their own prove industrial deployment readiness, but they do show that the Version 2 pipeline is technically credible, scalable, and well suited for an engineering portfolio project focused on vibration-based fault diagnosis.