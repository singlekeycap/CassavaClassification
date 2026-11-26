# Draft: deep-learning sections of the report

**For the teammate writing the paper.** This is draft text for the deep-learning parts, with every
number taken from the generated tables. Shorten, merge and reword it freely to fit the 6-page CVPR
limit; the full text here is about 3 pages, so expect to cut. Items marked **[TODO]** need team input.

**Figures and tables** are all in `reports/figures/` (vector PDF + 300-dpi PNG). LaTeX tables marked
`.tex` can be pasted with `\input{}`; they use `booktabs`.

**What not to claim** (each was checked and does not hold up):
- that any method is significantly better on **macro-F1** (it is not, after correction);
- that the weighted sampler significantly lowers accuracy (true on one fold only, not across 5);
- that Cassava labels are "noisy" as an established fact, unless the literature survey finds a source;
  our own evidence is phrased as "label ambiguity" below;
- that we beat or match the state of the art (rule R5).

**Deviations from the interim checkpoint (Oct 30) that the report must state.** Graders compare the two:

| Interim plan | What we did | Why / how to phrase it |
|---|---|---|
| Test set "touched once, for the final numbers" | Every run and every analysis was scored on it | No method, epoch or setting was chosen on test scores; all analyses were planned and are reported in full. See §3. |
| Inverse-frequency class weights (CBB 3.94 … CMD 0.33) | Square-root inverse frequency (CBB 1.51 … CMD 0.43), normalised to mean 1 | Full inverse frequency gives CBB ~12× the weight of CMD on 732 images; the softer version is more stable. Not tuned on test. |
| Only class-weighted loss | Four strategies compared (none, sampler, weights, oversampling) | An extension, the main experiment of the deep-learning part. |
| Crops 384–448 px, 90° rotations, almost no hue shift | 384 px crops, ±20° rotations, flips, colour jitter with hue 0.05 | State the final pipeline; hue jitter is small (±0.05). |
| Laptop RTX 3060 (6 GB) | Shared NVIDIA A100 80 GB GPUs | Server access became available. [TODO: confirm where the first four runs ran.] |

---

## 1. Split protocol (grouped split)

*Suggested place: Method, "Data and split protocol". The classical baseline should use the same split; confirm with its author. [TODO]*

The Cassava Leaf Disease dataset has 21,397 field photographs (800×600) in five classes: Cassava
Mosaic Disease (CMD, 61.5%), Healthy (12.0%), Cassava Green Mottle (CGM, 11.2%), Cassava Brown
Streak Disease (CBSD, 10.2%) and Cassava Bacterial Blight (CBB, 5.1%), an imbalance ratio of 12.1 : 1.
Predicting CMD for every image gives 61.5% accuracy but only 0.15 macro-F1.

The dataset provides no plant, field or farmer identifiers, so the unit of correlation (the photo
session or plant) has to be estimated. We embed every image with a self-supervised DINOv2 ViT-S/14 model,
link two images when they are mutual 3-nearest neighbours with cosine similarity ≥ 0.80 (on
mean-centred embeddings), and take connected components as groups. A 256-bit perceptual-hash check
found no exact or near-exact copies, so all links come from visual similarity: the groups are
**similar-image groups** (same plant or photo session), not copies. This yields 230 multi-image groups
covering 840 images (3.9%; largest group 27 images); all other images are singletons. *(Figure available: `reports/figures/group_examples.png`.)*

Using `StratifiedGroupKFold`, we hold out one seventh of the data as a fixed test set (3,078 images)
and split the rest into five stratified, grouped cross-validation folds (≈3,670 images each). No group
crosses the test set or two folds, and every split keeps the overall class proportions
(`reports/figures/imbalance/fig2_split_distribution.pdf`, `imbalance_table.tex`).
The test set is used only for final reporting: best epochs and early stopping use the validation fold,
and no model or setting was chosen on test scores.

## 2. Learned model and training

*Suggested place: Method, "Deep learning model".*

We fine-tune an ImageNet-pretrained ResNet-50 (torchvision `IMAGENET1K_V2` weights), replacing the
classifier with dropout (p = 0.3) and a 5-way linear layer. Images are trained at 384×384 with
random resized crops (scale 0.4–1), horizontal and vertical flips, ±20° rotation, colour jitter and
random erasing (p = 0.25); at test time we resize the short side to 384 and take the center crop.
We use AdamW (learning rate 10⁻³ for the head, 10⁻⁴ for the backbone, weight decay 10⁻⁴), one epoch of
linear warm-up followed by cosine decay, label smoothing 0.1, batch size 32 and mixed precision.
The backbone is frozen for the first epoch. Training runs for up to 20 epochs with early stopping
(patience 5) on validation macro-F1, and the best-validation checkpoint is evaluated on the test set.
All runs use seed 42.

**Class-imbalance strategies.** We compare four ways of handling the 12 : 1 imbalance, each applied to
the training split only (`reports/figures/imbalance/fig3_resampling_effect.pdf`, `fig4_class_weights.pdf`):

1. **None** (baseline): plain cross-entropy on the original distribution.
2. **Weighted sampler**: a `WeightedRandomSampler` draws images with probability ∝ (class frequency)^-0.5,
   so an epoch sees CMD 39% of the time instead of 62%.
3. **Class-weighted loss**: cross-entropy weighted by (class frequency)^-0.5, normalised to mean 1
   (CBB 1.51, CBSD 1.05, CGM 1.02, CMD 0.43, Healthy 0.98).
4. **Oversampling**: minority-class images are duplicated up to the median class count (1,619), so CBB
   and CBSD are oversampled and the other classes are unchanged.

We use the square root of inverse frequency rather than full inverse frequency to avoid over-weighting
the 732 CBB training images.

## 3. Evaluation protocol and metrics

*Suggested place: Method, "Evaluation". This also answers rule R6 (metrics beyond accuracy).*

Because accuracy is dominated by CMD, our primary metric is **macro-F1**, which weights the five classes
equally; we also report **balanced accuracy** (mean per-class recall), per-class F1 and recall, and
**expected calibration error** (ECE, 10 bins). Accuracy is reported for completeness only.

Each imbalance strategy is trained **five times, once with each cross-validation fold as validation**,
and every run is scored on the same fixed test set. We report mean ± standard deviation over the five runs
and compare each strategy with the baseline **fold by fold** (paired differences, a paired t-test and
the number of folds on which the strategy is better), with Holm correction for the three comparisons.
For a single run we additionally give 95% bootstrap confidence intervals, resampling the test set by
similar-image group (10,000 resamples, paired across methods), and McNemar's test.
The five training sets overlap by about 75%, so the runs are not fully independent and the paired
t-test is somewhat optimistic; we therefore read it together with the sign count.

**Use of the test set.** The interim plan said the test set would be touched once. In practice every
trained model was scored on it, and the analyses below (robustness, error analysis, learning curve,
random split) re-use those test predictions. We kept the rule's purpose: **model selection used the
validation folds only**, and no method, epoch, hyperparameter or analysis setting was chosen or changed
based on test scores; all planned comparisons are reported, including negative ones.

## 4. Results: class-imbalance strategies

*Table: `reports/figures/cross_validation/cv_summary.tex` (main results).
Figure: `reports/figures/cross_validation/cv_paired_diff.pdf` (differences vs baseline per fold).*

Table X summarises the 20 runs. All four strategies reach 87–88% accuracy and 77–78% macro-F1.
**Class-weighted loss is the only strategy that consistently beats the baseline:** it raises balanced
accuracy from 77.4 ± 1.0 to 79.3 ± 0.7 (+1.9 points, better on 5/5 folds, Holm-corrected p = 0.002).
The gain comes from minority-class recall (CBB +3.9, CBSD +3.4 on 5/5 folds, CGM +2.3 on 5/5 folds,
Healthy +1.8 points) paid for with a small loss in CMD recall (−1.7 points), which leaves CMD's F1
almost unchanged (−0.1). On macro-F1 its advantage is smaller and not significant after correction
(+0.6 points, better on 4/5 folds, p = 0.08). The same balanced-accuracy gain appears on a single run
with a test-set bootstrap (+2.3 points, 95% CI [+0.7, +4.0]), so it holds up against both training
noise and test-set sampling noise.

The weighted sampler and oversampling do not improve on the baseline on any metric (macro-F1 −0.7 and
+0.1 points; balanced accuracy +0.5 and +0.3 points, none significant). On fold 0 alone the sampler
appeared to lower accuracy significantly (McNemar p = 0.016), but across five folds the effect is
−0.7 points and not significant (p = 0.23): a single run would have led to a wrong conclusion.

**Interpretation.** With a strong pretrained backbone, imbalance handling changes *where* the decision
boundary sits more than *how well* the classes are separated: re-weighting the loss trades a little
majority-class recall for minority-class recall, which balanced accuracy rewards and macro-F1 only
partly. Resampling methods repeat the same few minority images, which with heavy augmentation adds
little new information.

**Calibration.** Class-weighted loss is markedly less well calibrated (ECE 0.107 vs 0.044 for the baseline):
it is *under*-confident, with accuracy above confidence in most bins
(`reports/figures/error_analysis/calibration.pdf`). If calibrated probabilities matter (e.g. to flag
uncertain plants for an expert), temperature scaling on the validation set would be needed.

## 5. Random vs grouped split (rule 5b)

*Tables: `reports/figures/split_and_learning_curve/random_vs_grouped_headline.csv`, `leakage_accuracy.csv`.
Figure: `leakage_accuracy.pdf`.*

We retrained the baseline and class-weighted loss on three naive stratified random splits with the same
sizes as the grouped split. Each model is scored on its own split's test set:

| Macro-F1 (%) | Grouped split (5 runs) | Random split (3 runs) |
|---|---|---|
| Baseline | 77.7 ± 0.7 | 79.2 ± 0.3 |
| Class-weighted loss | 78.3 ± 0.4 | 77.9 ± 0.4 |

The headline difference is inconsistent in sign and is confounded by the random runs having different
test sets, so we measure leakage directly. In a random split about 113 test images (3.7%) have a
same-group image in the training set; models classify these **84.3%** correctly, compared with
**82.0–83.1%** for grouped-split models on grouped images whose group was never seen in training.
Leakage therefore exists but is small (1–2 points on 4% of the test set, about 0.1 points overall).
This is expected: there are no exact copies, only similar images of the same session, and only 3.9% of
the data is grouped. It is consistent with the course's assessment that a random split is defensible for Cassava. The two image sets
also differ in class mix (the leaked set has more CBSD and fewer Healthy images), so even the 1–2 point
gap is approximate. We keep the grouped split: it costs nothing and removes the risk.

## 6. Generalisation study: performance vs training-set size (component 3)

*Figure: `reports/figures/split_and_learning_curve/learning_curve.pdf`. Table: `learning_curve.csv`, `gain_per_doubling.csv`.*

We trained the baseline and class-weighted loss on nested stratified subsets of 5, 10, 25 and 50% of
the training split (validation and test unchanged), on validation folds 0–2; 100% are the main runs.
Smaller subsets were given more epochs (60, 40, 30, 25) with proportionally longer early-stopping patience.

| Training images | 733 (5%) | 1,465 (10%) | 3,663 (25%) | 7,326 (50%) | 14,642 (100%) |
|---|---|---|---|---|---|
| Macro-F1, baseline | 69.9 ± 1.4 | 71.8 ± 1.4 | 74.8 ± 0.2 | 76.7 ± 0.9 | 78.0 ± 0.5 |
| Macro-F1, class-weighted | 69.0 ± 1.3 | 72.7 ± 1.0 | 74.3 ± 0.3 | 76.2 ± 1.0 | 78.5 ± 0.1 |
| CBB F1, baseline | 50.7 | 52.4 | 55.0 | 60.8 | 62.1 |

Performance grows roughly linearly in log(training size), by about **2 macro-F1 points per doubling**
of the data, and **has not saturated** at the full training set. Fitting the last three points and
extrapolating suggests roughly 80% macro-F1 with twice the data, an optimistic estimate since learning
curves usually flatten. The smallest class benefits most: CBB F1 rises by 11 points over the range.
Even 5% of the data (733 images, 37 of them CBB) reaches 70% macro-F1, consistent with a strong
pretrained backbone (we did not train without pretraining). The class-weighting advantage appears only at the full data size;
at smaller sizes the two methods are within each other's run-to-run variation.

*Limitations:* each fraction has a fixed epoch budget; three folds per fraction.

## 7. Robustness and distribution shift (component 4)

*Figures: `reports/figures/robustness/corruption_examples.png`, `degradation_curves.pdf`,
`relative_drop_heatmap.pdf`, `natural_conditions.pdf`. Tables: `robustness_macro_f1.csv`, `robustness_summary.csv`.*

Cassava has no site or device metadata, so we evaluate two kinds of shift on the 20 trained models,
without retraining.

**Synthetic capture degradations.** We apply six corruptions a field photo can suffer, each at three
severities, to the test images after preprocessing: Gaussian blur (σ = 1, 2, 4 px), darkening
(×0.6, 0.4, 0.25), brightening (×1.4, 1.7, 2.0), Gaussian noise (σ = 10, 20, 35 on 0–255), JPEG
compression (quality 30, 15, 7) and reduced resolution (192, 96, 64 px, upsampled back).
Mean macro-F1 over all 20 models (clean: 77.7%):

| Corruption | severity 1 | severity 2 | severity 3 |
|---|---|---|---|
| Darker | 77.4 | 76.4 | 74.3 |
| Brighter | 75.5 | 72.0 | 67.3 |
| JPEG | 73.5 | 67.5 | 52.9 |
| Blur | 74.2 | 63.2 | 37.6 |
| Low resolution | 71.8 | 58.4 | 43.0 |
| Noise | 72.1 | 51.7 | 29.4 |

The models are robust to exposure changes but fragile to anything that removes **fine texture**:
blur, noise, low resolution and strong JPEG compression. This is consistent with the symptoms
themselves being fine-scale (mosaic patterns, small lesions, vein yellowing). Even moderate JPEG
compression (quality 30) costs 4 points, which matters if photos are compressed when sent from the field.
The errors are not random: under heavy noise **61% of all images are predicted as CGM** (11% in truth),
suggesting the model associates high-frequency speckle with green mottle; heavy blur pushes predictions
towards CBB (19% vs 5%).

No imbalance strategy is more robust than the baseline: the mean relative macro-F1 drop over all
corruptions is 17.6% (baseline), 18.3% (class-weighted), 18.4% (oversampling) and 20.0% (sampler).
The sampler is the least robust on 5/5 folds, but this is not significant after correction (p = 0.33).

**Natural capture conditions.** Without any synthetic change, we split the clean test photos into thirds
by measured brightness (mean luminance) and sharpness (variance of the Laplacian). These results are
descriptive only, because the thirds differ in class mix. Darker photos score lower than brighter ones
(macro-F1 76.1% vs 79.4%), but the dark third also holds more CBB (6.6% vs 3.7%), the hardest class.
The sharpness buckets differ even more in class mix
(the sharpest third is 72% CMD and has only ~25 CBB images, the blurriest is enriched in CBB and CBSD),
so their scores cannot be compared directly. Notably, the blurriest third holds 7.9% CBB versus
2.5% in the sharpest, which may explain why synthetic blur shifts predictions towards CBB: the model
may partly use image blur as a cue for that class.

## 8. Error analysis (component 5)

*Figures: `reports/figures/error_analysis/error_consistency.pdf`, `failures_cbb.png`,
`failures_healthy.png`, `failures_most_confident.png`, `gradcam_failures.png`, `fixed_by_weights.png`.
Check the dataset licence before including the photo grids (rule R12). [TODO]*

Because all 20 models predict the same test set, we can separate **consistent failures** (images
misclassified by at least 18 of 20 models) from **unstable** ones (wrong for some models).
75% of test images are correct for every model, 18% are unstable and **7% (215 images) are consistent
failures**, a property of the image or its label rather than of a particular training run.
CBB has the highest share of consistent failures (19.5%), followed by CGM (14.2%), Healthy (13.8%),
CBSD (13.1%) and CMD (2.2%).

The most frequent consistent confusions are **CGM → CMD** (21 images), **CBB → Healthy** and
**Healthy → CBB** (17 each) and **CGM → Healthy** (15). Looking at the images, three patterns recur:

1. **Symptoms too small or absent in the frame.** Many misclassified CBB images are wide shots of a
   canopy where the angular leaf spots or wilting are small, at the edge or not visible; most models
   call them Healthy. Grad-CAM on these images often highlights green, symptom-free leaf areas.
2. **Visually overlapping symptoms.** CGM's leaf mottling and CMD's mosaic both produce yellow-green
   patterning, and CBSD's yellowing resembles CBB on some leaves; these pairs account for most confusions
   between disease classes.
3. **Possible label errors.** Some images are misclassified by all 20 models with near-zero probability
   for the given label, and several look as if they show symptoms of the predicted class; we treat these
   as *candidates* for label errors, not proven ones. The interim report noted that 96 of the 230
   similar-image groups contain more than one label; since the groups are same-session images rather than
   copies, part of this is likely different plants photographed together, so it is weak evidence on its own.
   [TODO: cite a source on label quality in this dataset if the literature survey finds one.]

**What class weighting changes.** Only 7 images are fixed consistently by class weighting (wrong in
≥ 4/5 baseline folds, right in ≥ 4/5 weighted folds) and 5 are broken (all CMD). Its gain therefore
comes from many small shifts of borderline minority-class images, not from solving hard cases.

**Center-crop hypothesis (negative result).** Test-time preprocessing discards the left and right 12.5%
of each 800×600 photo. Scoring the full 512×384 image instead changes macro-F1 by between −0.4 and
+0.5 points for the four fold-0 models, so the discarded border does not explain the errors (at least
at test time; we did not retrain on full images).

## 9. Discussion: what would improve performance

*Suggested place: Discussion. Includes the negative results required by rule R10.*

- **More data, especially CBB.** The learning curve has not saturated (≈2 points per doubling) and the
  smallest class gains most.
- **Higher resolution or multi-crop inference** for fine symptoms: the robustness study shows that
  removing fine texture is what hurts most, and many CBB failures have small symptoms in wide shots.
  Tiling the 800×600 photo instead of one 384 crop is a natural next step.
- **Texture-robust training**: adding blur, noise and JPEG augmentation during training would directly
  target the observed fragility and the noise → CGM shortcut.
- **Label cleaning or noise-robust losses** (e.g. reviewing the 215 consistent failures and the
  label-conflicting groups).
- **Stronger, more recent backbones and ensembles**, as used by the leading competition solutions
  [TODO: link to the literature survey].
- **Calibration** (temperature scaling) if class-weighted loss is deployed.

**Negative results.** The weighted sampler and oversampling did not help on any metric; scoring the
uncropped image did not help; no imbalance strategy improved robustness; a single-run comparison
suggested a significant sampler effect that disappeared across folds.

**Comparison with the state of the art (rule R5).** [TODO, with the literature survey] Leading Kaggle
solutions report around 90–91% accuracy on the competition's hidden test set using ensembles of larger
models [TODO: verify exact figures and cite]. Our 87.8% is on our own grouped test split and is not
directly comparable; we compare approaches, not absolute scores.

## 10. Reproducibility and compute (rule R9)

All code, configurations (`config.json` per run), seeds (42 for training, fixed split seeds) and
analysis notebooks are in the repository; `docs/workflow.md` lists every step and command.
Compute: shared NVIDIA A100 80 GB GPUs (up to four jobs in parallel), about **23 GPU-hours**
in total: 8.3 h for the 20 main runs, 10.3 h for the learning curve (24 runs), 3.8 h for the random-split
runs (6), and 0.7 h for the robustness evaluation. [TODO: the first four runs (fold 0) were trained
from a WSL machine according to their log; confirm which GPU and adjust the statement.]

**AI tool disclosure (rule R11).** [TODO: the team's statement, e.g. "We used Claude (Anthropic) as a
coding assistant to write analysis scripts, notebooks and experiment queues, and to draft parts of this
text; all results were produced by our code and checked by the team."]
