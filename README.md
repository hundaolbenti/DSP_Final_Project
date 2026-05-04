# DSP_Final_Project
By Hundaol Benti, Natnael Mulu, Robera Bushura
# Environmental Sound Classification Experiment Guide

This README reconstructs the full experiment workflow from the uploaded notebook exports and turns it into a single step-by-step guide.

The original uploaded files were PDFs exported from notebooks with names prefixed by `vertopal.com_`. In this guide, I refer to them using the notebook names **without** that prefix and with the `.ipynb` extension, as requested:

- `1_Data_Exploration.ipynb`
- `2_DSP_ Analysis.ipynb`
- `3_CNN_Models.ipynb`
- `4_Pretrained Model Options Comparison.ipynb`
- `5_Classification with EfficientNetB0 (2).ipynb`
- `6_DSP_FINAL_PROJECT.ipynb`

## Project Goal

The project builds an **environmental sound classification pipeline** on the **ESC-50** dataset. The overall goal is to move from raw audio to a well-validated classifier by following this progression:

1. Understand the dataset and confirm that it is suitable.
2. Study digital signal processing (DSP) representations of audio.
3. Train baseline CNN models.
4. Compare stronger pretrained model backbones.
5. Improve performance with augmentation and better training strategy.
6. Finalize the strongest configuration and report cross-validation results.

---

## End-to-End Experiment Flow

### Step 1: Explore the dataset with `1_Data_Exploration.ipynb`

This notebook is the starting point of the project. Its job is to verify that the data is clean, balanced, and meaningful before any modeling begins.

#### What this notebook does

1. **Downloads ESC-50 from GitHub** and extracts it locally.
2. **Loads the metadata** from `meta/esc50.csv`.
3. **Checks dataset structure**:
   - 2000 audio clips
   - 50 classes
   - 5 folds
   - 5-second mono WAV recordings at 44.1 kHz
4. **Verifies class balance** by counting samples per category.
5. **Verifies fold balance** by counting samples in each fold.
6. **Listens to random audio samples** to connect labels with real sounds.
7. **Plots waveforms and spectrograms** to inspect temporal and frequency behavior.
8. **Extracts simple descriptive statistics** such as durations, silence patterns, and category-level properties.

#### Why this notebook matters

This step prevents avoidable downstream mistakes. Before training any model, we need confidence that:

- the dataset is present and correctly loaded,
- the labels are balanced,
- the cross-validation folds are usable,
- the sounds have class-dependent patterns worth learning.

The reports show that the dataset is **perfectly balanced** at **40 samples per class** and **400 samples per fold**, which is important because it means later model comparisons are less likely to be distorted by class imbalance.

#### Useful code ideas from this notebook

- **Dataset download guard** using `if not os.path.exists(DIR_NAME)`:
  - useful because it makes the notebook rerunnable without re-downloading the dataset every time.
- **Metadata inspection with pandas**:
  - useful because it gives fast validation of class counts, fold counts, and column integrity.
- **Waveform and spectrogram visualization**:
  - useful because audio problems are often easier to catch visually than numerically.
- **Random sample playback**:
  - useful because hearing examples helps verify whether labels and classes make intuitive sense.

#### Key findings from this step

- ESC-50 is balanced by class and fold.
- Each audio clip is standardized to 5 seconds, which simplifies preprocessing.
- Different categories show visibly different waveform and spectrogram patterns.
- The dataset quality is high enough to justify feature engineering and modeling.

---

### Step 2: Build DSP intuition with `2_DSP_ Analysis.ipynb`

After confirming that the dataset is good, the next step is to understand how to represent sound signals numerically.

#### What this notebook does

1. Loads example audio clips.
2. Plots **raw waveforms** in the time domain.
3. Applies **DFT** and **FFT** to inspect frequency content.
4. Explains the limitation of global Fourier analysis: it shows **what frequencies exist**, but not **when** they happen.
5. Uses **STFT** to recover time-localized frequency information.
6. Converts STFT output to **mel spectrograms**.
7. Studies the effect of parameters such as:
   - `n_fft`
   - `hop_length`
   - number of mel bins
8. Extracts **MFCC**, **delta**, and **delta-delta** features.
9. Generates and saves large numbers of log-mel spectrograms for later model training.

#### Why this notebook matters

This notebook creates the bridge between raw audio and machine learning features.

The most important reasoning here is:

- **Raw waveforms are hard for simple image-style CNN pipelines to use directly.**
- **DFT/FFT lose timing information**, which is a serious problem for sound events because many classes differ not only by which frequencies exist, but by how they evolve over time.
- **STFT preserves both time and frequency structure**, making it a much better representation.
- **Mel spectrograms compress frequency into a perceptually meaningful scale**, which usually makes the representation more aligned with how humans distinguish sound events.

That is why the project repeatedly returns to **log-mel spectrograms** as the main representation.

#### Useful code ideas from this notebook

- **Waveform loading with `librosa.load(..., sr=None)`**:
  - useful because it preserves the original sampling rate when analysis accuracy matters.
- **DFT/FFT visualizations**:
  - useful because they clearly justify why a pure frequency-only representation is insufficient.
- **STFT and mel spectrogram generation**:
  - useful because these are the central features used by later models.
- **Parameter sweep for `n_fft` and `hop_length`**:
  - useful because time-frequency resolution is a tradeoff, and the best setting depends on the target task.
- **MFCC + delta + delta-delta extraction**:
  - useful because it creates a compact handcrafted feature set for traditional DSP baselines or multi-channel models.

#### Key findings from this step

- Mel spectrograms provide a strong feature space for environmental audio.
- Different parameter settings change time resolution and frequency resolution in important ways.
- The notebook explicitly motivates why later models use spectrogram images rather than only raw waveforms or plain FFT outputs.

---

### Step 3: Train baseline CNN systems with `3_CNN_Models.ipynb`

Once the spectrogram representation is established, this notebook starts model development with CNN-based architectures.

#### What this notebook does

1. Loads the ESC-50 dataset and confirms class balance.
2. Creates PyTorch dataset loaders for spectrogram-based inputs.
3. Defines and compares three CNN families:
   - **Model 1: Baseline CNN**
   - **Model 2: Multi-Channel CNN**
   - **Model 3: Augmented CNN**
4. Trains models fold by fold.
5. Tracks training and validation loss/accuracy.
6. Computes evaluation metrics such as:
   - accuracy
   - precision
   - recall
   - F1-score
7. Produces confusion matrices and learning curves.

#### Why this notebook matters

This notebook establishes the first real learning baseline. It answers:

- Can a custom CNN learn the task at all?
- Does feature stacking or augmentation help?
- How much performance can be obtained before switching to transfer learning?

The baseline CNN is useful because it is interpretable and lightweight. The multi-channel version is useful because it can combine different audio descriptors into one input tensor. The augmented version is useful because environmental sounds often vary in timing, intensity, and local structure, and augmentation improves robustness.

#### Useful code ideas from this notebook

- **Custom PyTorch `Dataset` classes**:
  - useful because they keep preprocessing reproducible and centralized.
- **Reusable train/validate functions**:
  - useful because later notebooks can keep the same evaluation structure and compare models fairly.
- **Confusion matrix plotting**:
  - useful because aggregate accuracy alone hides which classes are confused.
- **5-fold split logic using ESC-50 folds**:
  - useful because it respects the dataset’s official evaluation structure.

#### Key findings from this step

- Custom CNNs are strong enough to learn meaningful audio patterns.
- Architecture and input design matter.
- This stage creates the baseline that later pretrained methods must beat.

---

### Step 4: Compare stronger backbones with `4_Pretrained Model Options Comparison.ipynb`

After establishing CNN baselines, this notebook moves to transfer learning and broader architecture comparison.

#### What this notebook does

1. Reuses the ESC-50 data pipeline with **log-mel spectrogram** inputs.
2. Converts spectrograms to **3-channel tensors** so image pretrained networks can use them.
3. Builds and evaluates several stronger model families, including:
   - **ResNet34**
   - **EfficientNet-B0**
   - **PANNs-inspired CNN**
   - **AST / ViT-style transformer baseline**
4. Trains each model and compares validation accuracy.
5. Explores spectrogram parameter settings and saves winning configurations.

#### Why this notebook matters

This notebook is where the project asks a more strategic question:

> Is it better to keep scaling custom CNNs, or to leverage pretrained visual backbones on spectrogram images?

That is a good question because ESC-50 is relatively small. With only 2000 clips, pretrained models can often generalize better than models trained fully from scratch.

#### Useful code ideas from this notebook

- **3-channel duplication of spectrograms**:
  - useful because it lets ImageNet-pretrained backbones accept spectrogram inputs without changing the first layer too much.
- **Transfer learning with a replaced classifier head**:
  - useful because it reuses rich pretrained features while adapting the final decision layer to 50 audio classes.
- **Architecture benchmarking in one notebook**:
  - useful because model choice should be evidence-driven, not arbitrary.
- **Spectrogram parameter tuning tied to model performance**:
  - useful because representation quality and backbone choice interact strongly.

#### Reported results and interpretation

From the extracted report:

- **ResNet34** reached a final validation accuracy around **0.6950**, with a best value reported around **0.7375**.
- **EfficientNet-B0** reached a final validation accuracy around **0.7400**, with a best value reported around **0.7900**.
- **PANNs-inspired CNN** underperformed the strongest pretrained image backbones in this experiment.

This is an important turning point: it suggests that **pretrained backbones on log-mel spectrograms are strong candidates**, and it justifies the more focused optimization that happens afterward.

---

### Step 5: Optimize EfficientNet training with `5_Classification with EfficientNetB0 (2).ipynb`

This notebook is a focused optimization pass built around EfficientNetB0.

#### What this notebook does

1. Implements an `AudioDataset` that:
   - loads ESC-50 audio,
   - computes mel spectrograms,
   - pads or trims them to a fixed size,
   - converts them to **3-channel normalized images**.
2. Defines helper functions for:
   - audio loading,
   - mel spectrogram generation,
   - manual SpecAugment.
3. Builds an **EfficientNetB0-based classifier** by replacing the classifier head with a 50-class output layer.
4. Trains a baseline EfficientNet model.
5. Runs several hyperparameter configurations.
6. Tests two important augmentation strategies:
   - **SpecAugment**
   - **Mixup**
7. Runs **5-fold cross-validation** using the best configuration.
8. Performs **external audio evaluation with ensemble voting**.

#### Why this notebook matters

This notebook shifts from “which backbone is promising?” to “how do we get the best possible accuracy from this backbone?”

The reasoning behind the most important code choices is strong:

- **Fixed-size mel spectrogram tensors** are needed because CNN backbones require consistent input dimensions.
- **3-channel duplication** is used because EfficientNet expects image-style input.
- **Normalization to `[0, 1]`** stabilizes training.
- **SpecAugment** is used because masking time and frequency bands encourages the model to rely on broader patterns rather than memorizing narrow artifacts.
- **Mixup** is used because it regularizes decision boundaries and usually improves generalization on smaller datasets.
- **Fold-based training** is used because ESC-50 is designed for cross-validation, and a single train/validation split would be less reliable.

#### Useful code ideas from this notebook

- **`AudioDataset` class with mel generation inside `__getitem__`**:
  - useful because preprocessing stays synchronized with augmentation and training logic.
- **Manual SpecAugment implementation**:
  - useful because it gives direct control over mask lengths and where they are applied.
- **EfficientNet classifier replacement**:
  - useful because the pretrained feature extractor can be reused while adapting output to 50 classes.
- **Mixup in the training loop**:
  - useful because label interpolation belongs at batch construction time, not inside dataset loading.
- **Ensemble voting across folds**:
  - useful because averaging predictions from multiple fold-trained models often gives more stable external predictions.

#### Reported results from this notebook

- **SpecAugment best validation accuracy:** **82.25%**
- **Mixup best validation accuracy:** **83.00%**
- **5-fold best accuracies:**
  - Fold 1: **77.50%**
  - Fold 2: **77.25%**
  - Fold 3: **80.00%**
  - Fold 4: **82.75%**
  - Fold 5: **79.25%**
- **Overall 5-fold result:** about **79.35% ± 1.99%**

#### Interpretation

This notebook demonstrates that augmentation clearly helps, and among the tested EfficientNet-centered strategies, **Mixup** slightly outperformed **SpecAugment**. It also produces a robust ensemble-based evaluation path for unseen external audio.

---

### Step 6: Finalize the strongest pipeline with `6_DSP_FINAL_PROJECT.ipynb`

This is the final notebook in the sequence. It combines the earlier lessons into a stronger end-to-end system and reports the final project metrics.

#### What this notebook does

1. Revisits DSP configuration choices and compares parameter combinations.
2. Trains a stronger final candidate model.
3. Saves model checkpoints in `Experimental_Models/`.
4. Runs **full 5-fold cross-validation**.
5. Recovers saved models and recomputes evaluation metrics.
6. Produces a **per-class accuracy report** and exports raw CSV results.
7. Tests ensemble predictions on additional audio examples.

#### Why this notebook matters

This notebook is the project’s final scientific validation step.

Instead of stopping at one promising fold, it verifies that the pipeline remains strong across all five official ESC-50 folds. That is the correct way to make the final claim, because ESC-50 performance should be judged by fold-robustness rather than by one lucky split.

#### Useful code ideas from this notebook

- **Configuration comparison table for DSP settings**:
  - useful because preprocessing choices influence model performance as much as model architecture does.
- **Checkpoint recovery and re-evaluation**:
  - useful because it makes results reproducible without retraining everything.
- **Per-class accuracy analysis**:
  - useful because it reveals which categories remain difficult even when overall accuracy is strong.
- **CSV export of class-level results**:
  - useful because it supports report writing and further statistical analysis.

#### Reported final results

- Best single-run validation accuracy reported: **86.00%**
- 5-fold results reported in the final notebook:
  - Fold 1: **84.50%**
  - Fold 2: **86.75%**
  - Fold 3: **86.00%**
  - Fold 4: **90.00%**
  - Fold 5: **85.25%**
- **Overall mean accuracy:** **86.50% ± 2.13%**

#### Interpretation

This is the strongest result reported across the uploaded materials. The final notebook therefore represents the best version of the experiment and should be treated as the main reference for final conclusions.

---

## Recommended Order to Run the Experiment

If you are rerunning the project from the beginning, follow this exact order:

1. Run `1_Data_Exploration.ipynb`
   - confirm dataset download,
   - inspect metadata,
   - verify class/fold balance,
   - inspect waveforms and spectrograms.
2. Run `2_DSP_ Analysis.ipynb`
   - understand waveform, FFT, STFT, mel spectrograms, and MFCCs,
   - decide on DSP settings,
   - generate spectrogram artifacts if needed.
3. Run `3_CNN_Models.ipynb`
   - establish custom CNN baselines,
   - compare baseline, multi-channel, and augmentation-oriented CNN ideas.
4. Run `4_Pretrained Model Options Comparison.ipynb`
   - benchmark transfer learning backbones,
   - compare ResNet/EfficientNet/other candidates,
   - identify the most promising architecture direction.
5. Run `5_Classification with EfficientNetB0 (2).ipynb`
   - optimize EfficientNet training,
   - test SpecAugment and Mixup,
   - run cross-validation and ensemble prediction.
6. Run `6_DSP_FINAL_PROJECT.ipynb`
   - train the final best pipeline,
   - run complete 5-fold evaluation,
   - export final metrics and per-class analysis.

---

## Why the overall workflow makes sense

The progression of notebooks is methodologically sound:

- **Exploration first** avoids training on misunderstood data.
- **DSP second** ensures that feature design is motivated, not guessed.
- **Baselines before transfer learning** give a fair point of comparison.
- **Model comparison before optimization** prevents wasted effort on a weak architecture.
- **Augmentation and cross-validation near the end** improve generalization and produce trustworthy metrics.
- **Per-class reporting at the end** turns the project from a demo into a real experimental study.

In short, the project does not jump directly to a final model. It builds evidence step by step, which is exactly what a strong experiment should do.

---

## Main technical choices and the reasoning behind them

### 1. Why use ESC-50?

- It is a standard benchmark for environmental sound classification.
- It is balanced and fold-structured.
- It supports fair cross-validation.

### 2. Why use log-mel spectrograms?

- They preserve both time and frequency patterns.
- They are more perceptually meaningful than raw FFT bins.
- They work naturally with 2D convolutional and vision-style pretrained models.

### 3. Why keep 5-fold evaluation?

- ESC-50 is explicitly organized into 5 folds.
- Reported performance is more reliable when averaged across folds.
- It reduces the chance that the result is due to a favorable split.

### 4. Why use pretrained models?

- ESC-50 is small for deep learning.
- Transfer learning helps extract useful generic structure from limited data.
- Replacing only the final classifier is efficient and practical.

### 5. Why use SpecAugment and Mixup?

- Both reduce overfitting.
- SpecAugment improves robustness to missing local time/frequency cues.
- Mixup improves class boundary smoothness and generalization.

### 6. Why produce ensemble predictions?

- Fold-specific models learn slightly different decision boundaries.
- Combining them often reduces variance.
- Ensemble predictions are usually more stable on external audio.

---

## Final takeaway

The uploaded notebook sequence documents a clear research path:

- start with data understanding,
- build DSP intuition,
- establish CNN baselines,
- compare pretrained backbones,
- optimize augmentation and training,
- finish with robust cross-validated evaluation.

Among the uploaded results, the strongest final outcome is reported in `6_DSP_FINAL_PROJECT.ipynb`, with an overall mean accuracy of **86.50% ± 2.13%** across the 5 ESC-50 folds.

If you want to reproduce the project cleanly, use the six notebooks in the exact order listed above and treat the final notebook as the authoritative end-of-project evaluation stage.
