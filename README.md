# Token-Level Multi-Class Classification for Named Entity Recognition (NER)

## Overview

This project focuses on **token-level multi-class classification** for **Named Entity Recognition (NER)** using deep learning models. The goal is to classify each word in a sentence into predefined entity tags such as:

* `B-per` (Person)
* `B-geo` (Geographical entity)
* `B-org` (Organization)
* `O` (Non-entity)

I implement and compare multiple recurrent neural network architectures to evaluate their effectiveness on sequence labeling tasks.

---

## Dataset

* **Source:** Custom Dataset - [NER-1](https://drive.google.com/drive/folders/1rAkDEaz4-b414MuKTsp7eK5YkJo0SBZk)
* **Training samples:** 19,183 sentences
* **Testing samples:** 4,796 sentences
* **Total tags:** 17 entity classes + `PAD`
* **Average sentence length:** ~22 tokens

### Key Observation

* Significant **class imbalance**

  * `O` tag dominates (~355k occurrences)
  * Rare tags like `I-nat` appear very infrequently

---

## Exploratory Data Analysis (EDA)

I performed:

* Tag distribution visualization
* Sentence length distribution
* Token statistics

Insight:

* Highly imbalanced dataset impacts performance on rare classes

---

## Preprocessing

Steps applied:

1. **Data Cleaning**

   * Removed brackets, quotes, and formatting noise

2. **Tokenization**

   * Converted sentences into word lists

3. **Encoding**

   * Created:

     * `word2idx`
     * `tag2idx`

4. **Padding**

   * All sequences padded to max length (104)

5. **Class Weighting**

   * Applied to handle imbalance
   * Rare classes assigned higher weights

---

## Model Architectures

All models share:

* Embedding layer (128-dim)
* Hidden units: 64
* Dropout: 0.5
* Optimizer: Adam (lr = 0.0005)

### 1. Simple RNN

```
Embedding → SimpleRNN → TimeDistributed Dense
```

### 2. LSTM

```
Embedding → LSTM → TimeDistributed Dense
```

### 3. GRU

```
Embedding → GRU → TimeDistributed Dense
```

### 4. Bidirectional LSTM (BiLSTM)

```
Embedding → Bidirectional LSTM → TimeDistributed Dense
```

---

## Training Setup

* Batch size: 32
* Epochs: 15
* Early stopping enabled
* Loss: `sparse_categorical_crossentropy`
* Regularization:

  * L2 (λ = 0.01)
  * Recurrent dropout = 0.1

---

## Evaluation Metrics

We evaluated models using:

* Accuracy
* Macro F1-score
* Weighted F1-score
* Classification report

---

## Results

| Model   | Accuracy   | F1 (Macro) | F1 (Weighted) |
| ------- | ---------- | ---------- | ------------- |
| RNN     | 0.8374     | 0.3677     | 0.8701        |
| LSTM    | 0.8390     | 0.3759     | 0.8717        |
| **GRU** | **0.8856** | **0.4110** | **0.9035**    |
| BiLSTM  | 0.8276     | 0.3604     | 0.8583        |

---

## Key Findings

* **GRU performed best overall**
* Strong performance on frequent classes (`O`, `B-gpe`)
* Poor performance on rare classes (`B-art`, `I-nat`)
* Macro F1-score affected by class imbalance

---

## Limitations

* Severe class imbalance
* No pre-trained embeddings used
* No CRF layer for sequence optimization

---

## Future Work

* Integrate **GloVe / FastText embeddings**
* Use **BERT-based models**
* Add **CRF layer** for better sequence labeling
* Apply **data augmentation** for rare classes

---

## Tech Stack

* Python
* TensorFlow / Keras
* Scikit-learn
* NumPy, Pandas
* Matplotlib, Seaborn
