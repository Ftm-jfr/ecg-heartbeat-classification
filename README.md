# ECG Heartbeat Classification with Recurrent Neural Networks

Two deep-learning models that classify single ECG heartbeats, built with **PyTorch**:

| Task | Dataset | Model | Result |
|---|---|---|---|
| **5-class arrhythmia classification** (N, S, V, F, Q) | MIT-BIH | Bidirectional LSTM with packed sequences | 97.83% test accuracy, macro F1 0.887 |
| **Normal vs. abnormal** (myocardial infarction) detection | PTB Diagnostic ECG Database | Bidirectional LSTM with attention | 98.85% hold-out accuracy, macro F1 0.986 |

> Course project for **Deep Learning**, University of Isfahan.

## Repository structure

```
mitbih/
  mitbih-code.ipynb                5-class arrhythmia classifier
  best_ecg_model.pth               trained weights (state dict)
ptbdb/
  ptbdb.ipynb                      binary classifier with attention and 5-fold CV
  best_ecg_model_complete.pth      checkpoint (weights + normalization statistics + config)
```

## Dataset

The data was provided by the course instructor. It is the public **ECG Heartbeat Categorization Dataset** on Kaggle, which contains heartbeats segmented from two PhysioNet databases:

- Dataset: [kaggle.com/datasets/shayanfazeli/heartbeat](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Reference: M. Kachuee, S. Fazeli, M. Sarrafzadeh, *ECG Heartbeat Classification: A Deep Transferable Representation*, IEEE ICHI 2018, [arXiv:1805.00794](https://arxiv.org/abs/1805.00794)

Each row is one heartbeat of 187 samples (zero-padded at the end) followed by the label. The files used here are `mitbih_train.csv`, `mitbih_test.csv`, `ptbdb_normal.csv`, and `ptbdb_abnormal.csv`. The datasets are not included in this repository.


The notebooks were developed and run on **Kaggle** (GPU), so their data paths point to Kaggle input folders. The first cell of each notebook is the standard Kaggle starter cell; it only lists the files under `/kaggle/input` and does nothing harmful elsewhere.

## 1. MIT-BIH: 5-class arrhythmia classification

**Classes:** N (normal), S (supraventricular ectopic), V (ventricular ectopic), F (fusion), Q (unknown). The classes are highly imbalanced, with about 83% of beats being normal.

**Pipeline**
- Per-beat min-max normalization; the real length of each beat is found from its trailing zeros.
- **Packed sequences:** the zero padding is removed from the LSTM computation, so the model only sees the actual signal.
- **Class imbalance:** a `WeightedRandomSampler` draws rare classes more often.
- **Model:** 2-layer bidirectional LSTM (hidden size 128, dropout 0.3) on the concatenated final forward and backward states, followed by a linear classifier. Trained with Adam (lr 1e-3), gradient clipping, batch size 128, and 20 epochs.

**Test results** (21,892 beats)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| N (normal) | 0.9935 | 0.9833 | 0.9884 | 18,118 |
| S (supraventricular) | 0.7405 | 0.8723 | 0.8010 | 556 |
| V (ventricular) | 0.9475 | 0.9606 | 0.9540 | 1,448 |
| F (fusion) | 0.5892 | 0.8765 | 0.7047 | 162 |
| Q (unknown) | 0.9925 | 0.9845 | 0.9884 | 1,608 |
| **Accuracy** | | | **0.9783** | 21,892 |
| Macro average | 0.8526 | 0.9355 | 0.8873 | |

The rare classes S and F are the hardest: recall is high (the sampler works) but precision is lower because many normal beats are flagged as them.

## 2. PTB-DB: normal vs. abnormal

**Pipeline**
- Trailing zero padding is trimmed from each beat; signals are standardized with the mean and standard deviation of the training fold only (stored in the checkpoint).
- 15% of the data is held out as a final test set (stratified); the remaining 85% is used for **5-fold stratified cross-validation**.
- **Model:** 2-layer bidirectional LSTM (hidden size 128, dropout 0.3) with a masked **attention** layer over time steps, then LayerNorm, GELU, dropout, and a linear classifier.
- **Training:** AdamW (lr 1e-3, weight decay 1e-2), class-weighted cross-entropy, `ReduceLROnPlateau`, early stopping on validation macro F1, gradient clipping, up to 50 epochs.
- The best fold (by validation F1) is evaluated once on the hold-out set.

**Results**

- 5-fold cross-validation: mean macro F1 **0.9911** (± 0.0032)
- Hold-out test set (2,183 beats), best fold:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.9770 | 0.9819 | 0.9795 | 607 |
| Abnormal | 0.9930 | 0.9911 | 0.9921 | 1,576 |
| **Accuracy** | | | **0.9885** | 2,183 |
| Macro average | 0.9850 | 0.9865 | 0.9858 | |




