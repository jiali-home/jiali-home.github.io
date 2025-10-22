---
layout: post
title: Decoding the Unknown - A Look into Uncertainty Estimation Methods for Language Models
summary: A blog post about Uncertainty Estimation (UE) methods
---

In the rapidly evolving world of Large Language Models (LLMs), these powerful AI tools are becoming increasingly sophisticated. But how certain are we about their answers? Just like humans, LLMs can be unsure, and understanding this "uncertainty" is crucial for their reliable deployment. This is where Uncertainty Estimation (UE) methods come into play, offering ways to quantify how confident an LLM is in its responses.

---

## 0) Notation & Setup

Let an instruction/prompt be $x$. An LLM with parameters $\theta$ defines a next-token distribution
$$
p_\theta(y_t \mid x, y_{<t}) = \mathrm{softmax}\!\left(\frac{z_\theta(x,y_{<t})}{\tau}\right),
$$
where $z_\theta$ are logits and $\tau>0$ is (optional) temperature. A **sampled response** (completion) is a token sequence $y = (y_1,\dots,y_T)$.

We will often draw **multiple samples**
$$
\mathcal{S}=\{y^{(1)}, \ldots, y^{(K)}$$\sim p_\theta(\cdot\mid x)
$$
by sampling with nucleus/top-$p$ or top-$k$ decoding. For embedding-based methods, define an encoder $f(\cdot)$ (e.g., a sentence embedding model) that maps any text to a vector in $\mathbb{R}^d$.

---

## 1. Semantic Consistency Responses: When Meaning Matters

**Idea.** These methods tap into the meaning of an LLM's responses. Imagine asking an LLM a question multiple times – if it gives slightly different but semantically similar answers each time, it's likely more confident than if its answers widely diverge in meaning.


### 1.1 Semantic Entropy.

This method uses models to generate several responses and then measures how semantically similar they are. High similarity points to high certainty.


Compute pairwise semantic similarity 
$$
s_{ij}=\cos\big(f(y^{(i)}), f(y^{(j)})\big)\in[-1,1],
$$
where each $f(y)$ is a semantic embedding (e.g., from a model like DeBERTa or Sentence-BERT).  
Cosine similarity measures the **semantic closeness** between two responses because embedding models are trained so that semantically similar texts have vectors pointing in similar directions. Specifically:
$$
\cos(\theta) = \frac{f(y^{(i)}) \cdot f(y^{(j)})}{\|f(y^{(i)})\|\|f(y^{(j)})\|}.
$$
- $s_{ij}=1$: identical meaning  
- $s_{ij}=0$: unrelated meaning  
- $s_{ij}=-1$: opposite or contradictory meaning  

Convert these pairwise similarities into **paraphrase clusters** $$C_1,\dots,C_M$$ (e.g., by agglomerative clustering with a cosine threshold $\gamma$).  
Let $q_m = |C_m|/K$. Define the **semantic entropy**:
$$
H_{\text{sem}}(x) \;=\; -\sum_{m=1}^{M} q_m \log q_m.
$$
- **Low $H_{\text{sem}}$** (mass on 1–2 clusters) ⇒ consistent semantics ⇒ **low uncertainty**.  
- **High $H_{\text{sem}}$** ⇒ fragmented meanings ⇒ **high uncertainty**.


### 1.2 Graph Laplacian Consistency

This family of methods represents each response as a node in a **semantic similarity graph**, where edges connect semantically related answers.  
The intuition: if the graph is well connected (high overall similarity), the model is confident; if it breaks into weakly connected clusters, the model is uncertain.

#### (a) Standard Laplacian Consistency
Build a similarity graph $W\in\mathbb{R}^{K\times K}$ with cosine similarities:
$$
W_{ij} = \max(\cos(f(y^{(i)}), f(y^{(j)})), 0).
$$
Define the degree matrix $D=\mathrm{diag}(\sum_j W_{ij})$ and Laplacian $L=D-W$.  
The **algebraic connectivity** (second-smallest eigenvalue) $\lambda_2(L)$ measures how well-connected the graph is.

- Large $\lambda_2(L)$: one cohesive cluster → **low uncertainty**  
- Small $\lambda_2(L)$: several weakly connected clusters → **high uncertainty**

We define the uncertainty score:
$$
U_{\text{Lap}}(x) = \frac{1}{\lambda_2(L) + \epsilon}.
$$


#### (b) Eigenvalue Laplacian (NLI-based variant)
Instead of using cosine similarity from embeddings, this variant uses **Natural Language Inference (NLI)** models to compute semantic agreement between pairs of responses.  
For each pair $(y^{(i)}, y^{(j)})$, obtain an NLI “entailment probability”
$$
s_{ij} = p_\text{NLI}\big(y^{(i)} \Rightarrow y^{(j)}\big)
$$
and symmetrize it:
$$
W_{ij} = \tfrac{1}{2}\big[s_{ij} + s_{ji}\big].
$$
This produces a **semantic-agreement graph** $W$.  
Compute the normalized Laplacian $L_\text{norm} = I - D^{-1/2} W D^{-1/2}$ and examine its spectrum.

- A single dense semantic cluster → larger $\lambda_2(L_\text{norm})$  
- Multiple conflicting semantic groups → smaller $\lambda_2(L_\text{norm})$

Define the uncertainty:
$$
U_{\text{EigenLap}}(x) = \frac{1}{\lambda_2(L_\text{norm}) + \epsilon}.
$$

**Interpretation:**  
The NLI-based eigenvalue Laplacian captures *semantic consistency at the reasoning level*—if different responses logically entail each other (mutual entailment), the Laplacian’s second eigenvalue grows, signaling high confidence.  
If some responses contradict or fail to entail others, the graph splits and $\lambda_2$ drops, indicating uncertainty.


### 1.3 Eccentricity in Embedding Space

The **eccentricity** method quantifies how far the model’s generated responses are spread out in the embedding space.

After generating $K$ responses $y^{(1)}, \dots, y^{(K)}$, we obtain their embeddings $f(y^{(i)}) \in \mathbb{R}^d$.  
We first compute the **mean embedding** (the semantic “center” of all responses):
$$
\mu = \frac{1}{K}\sum_{i=1}^{K} f(y^{(i)}).
$$

Then, for each response, we measure how far its embedding is from the mean:
$$
d_i = \big\|f(y^{(i)}) - \mu\big\|_2.
$$
Finally, define the **eccentricity-based uncertainty** as the maximum distance:
$$
U_{\text{ecc}}(x) = \max_i d_i.
$$

### 1.4 Lexical Similarity

This baseline measures how **similar the wording** of multiple responses is — without using embeddings or semantic models.  
Instead of understanding meaning, it focuses on **surface-level overlap** of tokens or n-grams.

####  Definition

After generating $K$ responses $y^{(1)}, \dots, y^{(K)}$:

1. **Tokenize** each response into words or subwords.  
2. Compute a **pairwise lexical similarity** $s_{\text{lex}}(y^{(i)}, y^{(j)})$ for all pairs $(i,j)$ using a metric such as:
   - **Jaccard similarity** on word sets:
     $$
     s_{\text{lex}}(y^{(i)}, y^{(j)}) = 
     \frac{|W_i \cap W_j|}{|W_i \cup W_j|},
     $$
     where $W_i$ and $W_j$ are the sets of unique words in each response.
   - **BLEU-like n-gram overlap** for a more fine-grained comparison.

Then average all pairwise scores:
$$
\bar{s}_{\text{lex}} = \frac{2}{K(K-1)} \sum_{i<j} s_{\text{lex}}(y^{(i)}, y^{(j)}).
$$

Finally, define lexical uncertainty as:
$$
U_{\text{lex}} = 1 - \bar{s}_{\text{lex}}.
$$
Averaging ensures the value lies in $[0,1]$:  
- **High $\bar{s}_{\text{lex}}$** → wording is consistent → **low uncertainty**.  
- **Low $\bar{s}_{\text{lex}}$** → wording diverges → **high uncertainty**.


---

## 2. Information-Based Responses: The Probability Game

These methods focus on the raw mechanics of text generation, specifically the probabilities associated with each word or token.

- Perplexity: This widely used metric employs greedy log-likelihoods to measure how well a probability model predicts a sample. In simpler terms, it assesses the "surprise" of the model.

- Mean Token Entropy: This method calculates the average entropy of every token generated, giving an idea of how much "choice" the model had at each step.

- Maximum Sequence Probability: It sums the log probabilities of each token in a sequence to determine the overall likelihood of that specific response.

- Monte Carlo Sequence Entropy: Utilizing Monte Carlo estimates, this method provides a more robust way to determine uncertainty by sampling multiple possible continuations.


### 2.1 Per-Token Entropy and Mean Token Entropy
At step $t$, entropy is
$$
H_t \;=\; -\sum_v p_\theta(v\mid x,y_{<t}) \log p_\theta(v\mid x,y_{<t}).
$$
Define the **mean token entropy**
$$
\bar{H}(x,y) \;=\; \frac{1}{T}\sum_{t=1}^T H_t.
$$
Higher $\bar{H}$ = more branching choices at each step ⇒ **higher uncertainty**.

### 2.2 Sequence Log-Likelihood & Perplexity
The sequence log-probability:
$$
\log p_\theta(y\mid x) \;=\; \sum_{t=1}^T \log p_\theta\!\big(y_t \mid x,y_{<t}\big).
$$
Token-level **perplexity** is
$$
\mathrm{PP}(x,y) = \exp\!\Big(-\tfrac{1}{T}\log p_\theta(y\mid x)\Big).
$$
Higher perplexity ⇒ the model “finds its own output surprising” ⇒ **higher uncertainty**.

### 2.3 Monte-Carlo Sequence Entropy
We want $H\big(p_\theta(\cdot\mid x)\big) = -\sum_y p_\theta(y\mid x)\log p_\theta(y\mid x)$, but the space of $y$ is huge. Use samples $y^{(i)}\sim p_\theta(\cdot\mid x)$ and a self-normalizing estimator:
$$
\hat{H}_{\text{MC}} \;=\; - \sum_{i=1}^{K} w_i \log p_\theta(y^{(i)}\mid x), 
\quad 
w_i = \frac{p_\theta(y^{(i)}\mid x)}{\sum_{j} p_\theta(y^{(j)}\mid x)}.
$$
In practice, approximate $p_\theta(y^{(i)}\mid x)$ with token log-probs returned by the model. Higher $\hat{H}_{\text{MC}}$ ⇒ **higher uncertainty**.

**Note.** If you can’t get token log-probs, fall back to **semantic** methods (§1).

---



## 3. Density-Based Responses: How "Crowded" is the Answer Space?

These methods analyze how **dense** the model’s generated responses are in a shared embedding space.  
Intuitively, if all responses are close together (forming a tight cluster), the model is confident about its answer.  
If the responses scatter widely, it means the model is uncertain and is exploring multiple semantic possibilities.

Let each response embedding be $u_i = f(y^{(i)}) \in \mathbb{R}^d$,  
where $f(\cdot)$ is an embedding model such as DeBERTa, BERT, or OpenAI embeddings.

### 3.1 Mahalanobis Distance (Sequence Level)

We first estimate the **center** (mean vector) and **spread** (covariance) of all response embeddings:

$$
\hat{\mu} = \frac{1}{K} \sum_{i=1}^{K} u_i, \qquad 
\hat{\Sigma} = \frac{1}{K-1} \sum_{i=1}^{K} (u_i - \hat{\mu})(u_i - \hat{\mu})^\top.
$$

Then, for each response, we measure how far it lies from the center, **normalized by the shape of the cluster** (the covariance).  
This is the **Mahalanobis distance**:

$$
d_M(u_i) = \sqrt{(u_i - \hat{\mu})^\top \hat{\Sigma}^{-1} (u_i - \hat{\mu})}.
$$

Unlike the plain Euclidean distance, which treats all directions equally, the Mahalanobis distance **accounts for correlations** between embedding dimensions:
- In directions with high natural variance, a larger spread is tolerated.  
- In tightly constrained directions, even small deviations are penalized.

Hence, $d_M(u_i)$ measures how atypical a given embedding is **relative to the entire cluster**.

We can aggregate these distances in two common ways:

$$
U_{\text{Maha}} = \frac{1}{K} \sum_{i=1}^{K} d_M(u_i) 
\quad \text{or} \quad
U_{\text{Maha}}^{\max} = \max_i d_M(u_i).
$$

- **Average distance** → overall cluster spread.  
- **Maximum distance** → presence of strong outliers.  

Higher $U_{\text{Maha}}$ values indicate that responses are spread out in semantic space, implying **higher uncertainty**.


### 3.2 Robust Density Estimation

In high-dimensional settings (large $d$, small $K$), the sample covariance $\hat{\Sigma}$ may be ill-conditioned or even singular, making $\hat{\Sigma}^{-1}$ unstable.  
To address this, we use **robust covariance estimators** that stabilize the inverse.

#### (a) Shrinkage Estimator

A simple and effective approach is **shrinkage**, blending the sample covariance with the identity matrix:

$$
\hat{\Sigma}_\lambda = (1 - \lambda)\,\hat{\Sigma} + \lambda I,
$$

where $\lambda \in [0,1]$ controls how much regularization to apply.

- $\lambda = 0$: use full empirical covariance (high variance, possibly unstable).  
- $\lambda = 1$: assume all dimensions are independent with equal variance (too simple, but safe).  
- Intermediate $\lambda$ values (e.g., $0.1$–$0.3$) often yield the best trade-off.

Then we recompute Mahalanobis distances using $\hat{\Sigma}_\lambda$ instead of $\hat{\Sigma}$.

#### (b) Minimum Covariance Determinant (MCD)

Another robust approach is the **Minimum Covariance Determinant (MCD)**, which estimates the covariance matrix by focusing on the most central subset of data points (discarding outliers).  
This leads to a covariance estimate that is less sensitive to rare extreme responses that might otherwise distort $\hat{\Sigma}$.

#### (c) Dimensionality Reduction

When embeddings have very high dimensionality (e.g., $d = 1024$) and $K$ is small, we can first reduce dimension using **Principal Component Analysis (PCA)**:

$$
u_i' = P^\top (u_i - \hat{\mu}), \quad P \in \mathbb{R}^{d \times r}, \, r \ll d.
$$

We then compute the Mahalanobis distances in this reduced space $\{u_i'\}$, where the covariance is more stable.


#### Combined Uncertainty Score

The robust Mahalanobis uncertainty can be expressed as:

$$
U_{\text{RobustMaha}}(x)
= \frac{1}{K} \sum_{i=1}^K 
\sqrt{(u_i - \hat{\mu})^\top \hat{\Sigma}_\lambda^{-1} (u_i - \hat{\mu})}.
$$

This formulation balances **precision** (capturing true dispersion) with **stability** (avoiding numerical noise in small samples).

---

## 4. Reflexive Responses: Asking the Model Itself

#### 4.1 Self-Judgment / P(True)
After generating an answer $a$, prompt the LLM to **critique** or verify:
> “Given the question $x$ and answer $a$, how likely is $a$ to be correct? Respond with a probability in $[0,1]$.”

Let the returned scalar be $\hat{p}_{\text{true}}$. Define uncertainty
$$
U_{\text{self}} = 1 - \hat{p}_{\text{true}}.
$$



