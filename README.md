# CampusLoop: Context-Aware Toxic Content Detection for Student Dialects

A machine learning and NLP project designed to detect and mitigate toxic content in **student-centric online communities**, with a focus on **Indian code-mixed and regional student dialects** such as Hinglish and Telugu-English (Tinglish).

The goal is to distinguish between **legitimate negative feedback** about products or services and **genuinely toxic or abusive content**, while reducing the impact of spelling variations, phonetic spellings, code-mixing, and intentional text manipulation commonly found in student conversations.

## Problem Statement

Traditional toxicity detection systems are often trained on standard English datasets and may struggle with the way students communicate on campus platforms.

Student marketplaces and community platforms can contain:

* Product reviews and complaints
* Buy/sell/rent discussions
* Social interactions
* Harassment and abusive messages
* Code-mixed language such as Hinglish and Telugu-English
* Phonetic spellings and informal abbreviations
* Intentionally modified or misspelled abusive words

A major challenge is distinguishing **product toxicity** from **user toxicity**.

For example, a statement such as:

> "This product is terrible and stopped working in two days."

may represent legitimate product criticism rather than an attack against another user.

CampusLoop therefore focuses on **context-aware toxicity detection** rather than simply flagging negative language.

## Dataset

The project can use a combination of publicly available toxicity datasets and **campus-specific code-mixed data**.

The dataset contains examples representing:

* Non-toxic conversations
* Toxic/abusive conversations
* Harassment
* Insults
* Threatening language
* Product complaints
* Negative reviews
* Hinglish text
* Telugu-English code-mixed text

Since publicly available datasets may not fully represent student dialects, an additional **campus-specific dataset can be created through annotation and controlled data augmentation**.

Each text sample can be assigned labels such as:

| Label            | Description                                        |
| ---------------- | -------------------------------------------------- |
| Non-Toxic        | Normal conversation or feedback                    |
| Product Toxicity | Negative comments about a product/service          |
| User Toxicity    | Harassment, abuse, or attacks against another user |

## Project Workflow

1. **Data Collection**

   * Collect publicly available toxicity and sentiment datasets.
   * Incorporate code-mixed Indian-language datasets.
   * Create additional campus-specific examples representing student conversations.
   * Annotate examples according to toxicity and contextual category.

2. **Text Preprocessing**

   * Remove unnecessary punctuation and noise.
   * Normalize repeated characters and informal spellings.
   * Handle code-mixed text.
   * Apply phonetic normalization for regional-language spellings.
   * Detect and normalize Unicode homoglyphs used to disguise toxic words.

3. **Phonetic Normalization**

   * An **IndicSoundex-based normalization layer** is used to reduce variations caused by phonetic spelling.
   * This helps identify words that sound similar but have different spellings.
   * It is particularly useful for informal Telugu-English and Hinglish communication.

4. **Semantic Branch**

   * Fine-tune a transformer-based language model such as **DistilBERT or mBERT**.
   * Capture contextual and semantic relationships between words.
   * Identify whether the toxicity is directed toward a user or represents legitimate product criticism.

5. **Statistical Branch**

   * Convert text into numerical features using **TF-IDF**.
   * Train a traditional classifier such as **Naive Bayes or SVM**.
   * Capture important lexical patterns and frequently occurring toxic terms.

6. **Hybrid Ensemble**

   * Combine predictions from the semantic and statistical branches.
   * Use **weighted soft voting** to produce the final prediction.
   * This allows contextual transformer features and lexical statistical features to complement each other.

7. **Semi-Supervised Learning**

   * Apply a **MixText-style approach** to leverage both labelled and unlabelled code-mixed data.
   * Generate additional training examples to improve model generalization.

8. **Adversarial Text Defense**

   * Normalize Unicode homoglyphs.
   * Apply phonetic normalization.
   * Handle deliberate spelling modifications.
   * Reduce the ability of users to bypass toxicity filters through simple character substitutions.

9. **Evaluation**

   * Accuracy
   * Precision
   * Recall
   * F1-score
   * Confusion matrix
   * ROC-AUC where applicable
   * Separate evaluation for standard English and code-mixed student dialects

## Proposed Architecture

The proposed CampusLoop framework consists of two complementary branches:

### Semantic Branch

**Input Text → Preprocessing → IndicSoundex → DistilBERT/mBERT → Contextual Representation → Toxicity Prediction**

This branch focuses on understanding the **meaning and context** of the message.

### Statistical Branch

**Input Text → Preprocessing → TF-IDF → Naive Bayes/SVM → Toxicity Prediction**

This branch focuses on **lexical patterns and word-level statistical information**.

### Ensemble Layer

The predictions from both branches are combined using:

**Weighted Soft Voting → Final Toxicity Classification**

This hybrid architecture aims to combine the contextual understanding of transformer models with the lightweight lexical detection capabilities of traditional machine-learning models.

## Context-Aware Classification

One of the key components of CampusLoop is distinguishing **negative product feedback** from **personal/user-directed toxicity**.

For example:

| Example                                            | Classification                   |
| -------------------------------------------------- | -------------------------------- |
| "This phone is terrible, battery died in one day." | Product criticism                |
| "The seller never delivered my order."             | Negative feedback                |
| "You are useless, get lost."                       | User toxicity                    |
| "Worst seller ever, don't buy from this person."   | Potential user-directed toxicity |

This distinction helps prevent legitimate customer complaints from being incorrectly removed by the moderation system.

## Code-Mixed Student Dialects

CampusLoop specifically considers informal communication patterns commonly found in Indian student communities.

Examples may include:

* Hinglish — Hindi + English
* Telugu-English / Tinglish — Telugu + English
* Informal transliteration
* Phonetic spelling
* Abbreviations and slang
* Repeated characters
* Intentional misspellings

For example, the same abusive or offensive expression may appear in multiple transliterated or phonetic forms.

A conventional keyword-based moderation system may miss these variations, whereas the proposed normalization and semantic layers are designed to improve robustness.

## Adversarial Text Handling

Users may intentionally modify toxic words to bypass moderation systems.

Examples include:

* Character substitution
* Unicode homoglyphs
* Repeated characters
* Spaces inserted between characters
* Phonetic spellings
* Regional-language transliteration

CampusLoop applies normalization techniques before classification to reduce these bypass strategies.

### Example

**Original:**
`toxicword`

**Modified:**
`t0xicw0rd`

**Phonetic variation:**
`regional transliteration`

The preprocessing layer attempts to normalize these variations before they reach the classification models.

## Model Evaluation

The system should be evaluated using multiple metrics rather than accuracy alone.

| Metric           | Purpose                                           |
| ---------------- | ------------------------------------------------- |
| Accuracy         | Overall classification correctness                |
| Precision        | How many flagged messages are actually toxic      |
| Recall           | How many toxic messages are successfully detected |
| F1-score         | Balance between precision and recall              |
| Confusion Matrix | Analyze false positives and false negatives       |
| ROC-AUC          | Evaluate classification discrimination            |

Special attention should be given to **false positives**, because incorrectly classifying legitimate product complaints as toxic can negatively affect users and suppress genuine feedback.

## Expected Results

The proposed hybrid architecture is expected to provide:

* Improved detection of code-mixed toxic content
* Better handling of phonetic spellings
* Greater robustness against simple adversarial modifications
* Improved contextual understanding compared with keyword-based filtering
* Better separation of product criticism and user-directed toxicity
* More reliable moderation for student-focused online communities

Actual performance metrics should be reported after training and evaluating the final model on a held-out test dataset.

## Key Insights

* **Context matters:** Negative language does not automatically mean user toxicity.
* **Code-mixed language is challenging:** Models trained only on standard English may struggle with Indian student communication.
* **Phonetic normalization helps:** The same word can appear in many transliterated forms.
* **Hybrid models can complement each other:** Transformer models capture context while TF-IDF-based models capture strong lexical patterns.
* **Adversarial normalization is important:** Toxic users may intentionally modify spellings to bypass moderation.
* **Moderation should minimize false positives:** Genuine product complaints should not be incorrectly removed.

## Tech Stack

* Python
* pandas
* NumPy
* scikit-learn
* PyTorch
* Hugging Face Transformers
* DistilBERT / mBERT
* TF-IDF
* Naive Bayes / SVM
* SHAP or other explainability techniques
* matplotlib
* seaborn
* NLP preprocessing libraries

## Proposed System Architecture

```text
                    ┌──────────────────────┐
                    │     Input Message    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Text Preprocessing   │
                    │ • Cleaning           │
                    │ • Code-mix Handling  │
                    │ • Unicode Normalization│
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │  IndicSoundex Layer  │
                    │ Phonetic Normalization│
                    └──────────┬───────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
       ┌──────────────────┐        ┌──────────────────┐
       │ Semantic Branch  │        │ Statistical Branch│
       │                  │        │                  │
       │ DistilBERT/mBERT │        │ TF-IDF           │
       │ Context Features │        │ Naive Bayes/SVM  │
       └────────┬─────────┘        └────────┬─────────┘
                │                           │
                └─────────────┬─────────────┘
                              ▼
                    ┌──────────────────────┐
                    │ Weighted Soft Voting │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Final Classification │
                    │                      │
                    │ Non-Toxic            │
                    │ Product Toxicity     │
                    │ User Toxicity        │
                    └──────────────────────┘
```

## How to Run

Install the required dependencies:

```bash
pip install pandas numpy scikit-learn torch transformers
```

Then run the preprocessing and training scripts:

```bash
python preprocessing.py
python train.py
python evaluate.py
```

The trained system will preprocess the input data, train the individual branches, combine their predictions, and generate evaluation metrics.

## Future Improvements

* Build a larger annotated **Telugu-English and Hinglish campus dataset**
* Explore Indic-language transformer models
* Compare DistilBERT, mBERT and other multilingual models
* Experiment with different ensemble weights
* Add explainability using SHAP or LIME
* Develop real-time moderation through an API
* Build a moderation dashboard for campus administrators
* Add active learning for continuously improving the model
* Detect emerging student slang and newly appearing toxic expressions
* Evaluate robustness against adversarial spelling variations
* Deploy the system as a scalable moderation service

## Applications

CampusLoop can potentially be integrated into:

* University discussion platforms
* Student marketplaces
* Campus social networks
* Buy/sell/rent platforms
* Student forums
* College community applications
* Online student support communities

The system is intended to support **human moderation**, rather than automatically making irreversible decisions about users.

## Project Objective

CampusLoop aims to create a more robust and context-aware moderation framework for student communities by combining **multilingual semantic understanding, statistical text classification, phonetic normalization, semi-supervised learning, and adversarial defenses**.

The overall objective is to improve toxic-content detection while reducing the risk of legitimate student feedback being incorrectly classified as abusive.
