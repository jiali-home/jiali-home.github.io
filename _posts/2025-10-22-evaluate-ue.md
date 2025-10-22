---
layout: post
title: Measuring Uncertainty Estimation - How to Evaluate Confidence in Language Models
summary: A blog post about PRR
---



In the world of **Large Language Models (LLMs)**, understanding *what a model knows* is just as important as understanding *what it doesn’t*.  
Uncertainty Estimation (UE) methods help us quantify a model’s confidence in its outputs — but how do we know if a UE method itself is **effective**?

This blog introduces key evaluation metrics for Uncertainty Estimation, focusing on the **Prediction Rejection Ratio (PRR)** — a principled way to measure how well a UE method distinguishes between reliable and unreliable model predictions.

---

## 1. Why We Need to Measure Uncertainty

LLMs often generate fluent but **incorrect** responses.  
Uncertainty Estimation aims to flag these cases automatically by assigning a numerical **uncertainty score** $U(x)$ to each response.

But different UE methods produce different scores — semantic entropy, Mahalanobis distance, self-judgment probability, etc.  
We therefore need a **quantitative way** to evaluate *how useful* these uncertainty estimates are in identifying bad answers.

An ideal uncertainty measure should satisfy:

- **Inverse correlation**: high uncertainty → low answer quality.  
- **Utility**: helps decide whether to accept or reject a generated answer.  
- **Consistency**: works across different datasets or prompt types.

---

## 2. Prediction Rejection Ratio (PRR): The Core Metric

The **Prediction Rejection Ratio (PRR)** is a unified metric designed to evaluate UE effectiveness.  
It measures how well a UE method can decide *when* to trust or reject an LLM’s answer.

---

### 2.1 Definition and Intuition

The idea behind PRR is simple:

> If the uncertainty estimate is reliable, then rejecting high-uncertainty predictions should quickly improve the average quality of the remaining (accepted) answers.

Thus, PRR quantifies **how efficiently model quality improves** as low-confidence answers are removed.

---

### 2.2 Step 1: Build the Prediction Rejection (PR) Curve

For a given dataset $\{x_i, y_i\}_{i=1}^N$ and their generated responses $f(x_i)$:

1. Compute the **uncertainty score** $U_i = U(f(x_i))$ for each response.  
2. Compute a **quality score** $Q(f(x_i), x_i, y_i)$, where  
   $Q = 1$ means a perfect match with the ground truth (e.g., by LLMScore or semantic similarity).  
3. Sort all responses by **descending uncertainty** and progressively reject them.  
4. At each rejection rate $r \in [0, 1]$, calculate the **average quality** of the remaining accepted predictions.

Plot this as the **Prediction Rejection (PR) Curve**, where:

- The $x$-axis represents the **rejection rate** (fraction of most uncertain samples removed).  
- The $y$-axis represents the **average answer quality** among the remaining predictions.

---

### 2.3 Step 2: Compute the Area Under the PR Curve (AUCPR)

Let $A_{\text{UE}}$ denote the area under the PR curve for a given UE method.

$$
A_{\text{UE}} = \int_0^1 Q_r \, dr,
$$

where $Q_r$ is the average quality at rejection rate $r$.

This area captures **how much improvement in quality** can be achieved through rejection guided by the UE method.

---

### 2.4 Step 3: Compare with Oracle and Random Baselines

To interpret this area meaningfully, two reference curves are defined:

- **Oracle Curve** ($A_{\text{oracle}}$): Rejects predictions in the *perfect* order (from worst to best quality).  
- **Random Curve** ($A_{\text{random}}$): Rejects predictions in a completely random order.

The *ideal* uncertainty estimator would behave like the oracle — always rejecting low-quality answers first.

---

### 2.5 Step 4: Compute the Prediction Rejection Ratio (PRR)

The **Prediction Rejection Ratio** measures how close a UE method is to the oracle.  
It normalizes the area between the UE curve and the random curve by the area between the oracle and random curves:

$$
\text{PRR} =
\frac{
A_{\text{UE}} - A_{\text{random}}
}{
A_{\text{oracle}} - A_{\text{random}}
}.
$$

Here:
- $A_{\text{UE}}$: area under the UE method’s PR curve  
- $A_{\text{oracle}}$: area under the oracle curve  
- $A_{\text{random}}$: area under the random curve (baseline)

---

### 2.6 Interpretation

- $\text{PRR} \approx 1$:  
  The UE method performs nearly as well as the oracle — it **perfectly identifies unreliable predictions**.  
- $\text{PRR} \approx 0$:  
  The UE method behaves no better than random guessing.  
- $\text{PRR} < 0$:  
  The method is **misleading** — high uncertainty corresponds to *better* answers, indicating an inverted correlation.

---

## 3. Why PRR is Superior to Simpler Metrics

While traditional metrics like **AUROC** or **AUPRC** measure general discriminative ability, PRR has several key advantages for LLMs:

| Metric | Focus | Limitation | Advantage of PRR |
|:-------|:------|:------------|:-----------------|
| **AUROC** | Distinguishes correct vs. incorrect | Does not measure *utility* of rejection | PRR reflects actual *quality improvement* after rejection |
| **ECE / Brier Score** | Calibration of probabilities | Requires probabilistic confidence, not always available | PRR uses any form of uncertainty score |
| **PRR** | Measures *utility* of UE for filtering outputs | — | Directly connects uncertainty to decision-making |

Thus, PRR is not just a diagnostic metric but a **deployment-oriented tool** — it tells us *how much better the system becomes* when we reject uncertain outputs.

---

## 4. Practical Example

Suppose we have 100 model outputs, each with:
- A quality score $Q_i$ (between 0 and 1)
- An uncertainty estimate $U_i$

We sort them by $U_i$ (highest to lowest) and compute $Q_r$ at each rejection rate $r$.  
Then, we calculate the areas:

$$
A_{\text{UE}} = 0.76, \quad
A_{\text{random}} = 0.50, \quad
A_{\text{oracle}} = 0.90.
$$

Finally,

$$
\text{PRR} = 
\frac{0.76 - 0.50}{0.90 - 0.50} = 0.65.
$$

This means the UE method achieves **65% of the optimal rejection performance** — a strong but imperfect estimator.

---

## 5. Interpreting PRR in Practice

| PRR Value | Interpretation | Reliability |
|:-----------|:---------------|:-------------|
| **0.9–1.0** | Nearly optimal | Excellent — highly reliable UE |
| **0.6–0.8** | Strong correlation | Good, practical for filtering |
| **0.3–0.5** | Weak discrimination | Needs refinement |
| **< 0.3** | Poor or misleading | Unreliable uncertainty scores |

---

## 6. Summary

The **Prediction Rejection Ratio (PRR)** provides a robust and interpretable way to evaluate Uncertainty Estimation methods for LLMs.

- It connects **uncertainty** to **answer quality**, not just correctness.  
- It quantifies **practical utility** — how much uncertainty helps in real decision-making.  
- It enables **fair comparison** among UE methods and models.

In short:

> A good uncertainty estimator doesn’t just say *“I’m unsure”* —  
> it says it at the **right time**, and PRR tells us how well it does that.