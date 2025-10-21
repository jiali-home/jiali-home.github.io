---
layout: post
title: So, how attention actually works?
summary: Why attention was needed? what it is?
---

When I first hear the term attention in deep learning, it sounds almost human — as if the model is “focusing” on certain parts of the input more than others. And that’s actually not too far from the truth. The attention mechanism allows a model to decide which pieces of information are most relevant when processing a sequence, whether that’s a sentence, an image, or even a series of actions.



## The Mechanics: Query, Key, and Value

The brilliance of the Transformer paper is how it formalizes this intuition into a simple mathematical framework using three vectors: Query (Q), Key (K), and Value (V).

- Query (Q): What we’re looking for — the current word or token we want to understand in context.
- Key (K): What each word in the input offers — a representation of its “meaning” or identity.
- Value (V): The actual information that will be passed on once the attention weights are decided.

## The Attention Equation

Once we have these vectors, we can compute attention like this:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Let’s break this down:

1. **Similarity Score:**  
   $QK^T$ measures how much each **Query** aligns with each **Key** — basically, *how relevant is each input token to the one we’re currently looking at?*

2. **Scaling:**  
   The division by $ \sqrt{d_k} $ keeps the values stable (prevents very large dot products when the vector dimension is large).

3. **Softmax:**  
   Converts the similarity scores into probabilities that sum to 1 — these are our *attention weights*.

4. **Weighted Sum:**  
   Multiply these weights with the **Value (V)** vectors to get a weighted representation — this is what the model will actually use as contextual information.

## But where Do Q, K, and V Come From?


In the Transformer architecture, these vectors aren’t given magically — they’re *learned linear projections* derived from the **input embeddings**.

Let’s go step by step:

---

#### 1. **Start with Input Embeddings**

Each token (word, subword, or symbol) in your sequence — say, *“The”*, *“cat”*, *“sat”* — is first converted into an **embedding vector**.  
If your model’s hidden size is 512, then each word is represented as a 512-dimensional vector.

So, suppose we have an input sequence:

$$
X = [x_1, x_2, x_3, \dots, x_n]
$$

Each $ x_i \in \mathbb{R}^{d_{\text{model}}} $ (for example, $ d_{\text{model}} = 512 $).

---

#### 2. **Project Embeddings to Q, K, and V**

For each token embedding $ x_i $, the model learns three *different linear transformations*:

$$
Q = XW_Q, \quad K = XW_K, \quad V = XW_V
$$

where:

- $ W_Q, W_K, W_V \in \mathbb{R}^{d_{\text{model}} \times d_k} $ are **learned weight matrices**, trained end-to-end with the rest of the model.  
- $ d_k $ is typically smaller than $ d_{\text{model}} $ (e.g., 64 if $ d_{\text{model}} = 512 $ and there are 8 attention heads).

This means:
- Each **token embedding** gets mapped into **three distinct vector spaces** — one for queries, one for keys, and one for values.
- These projections let the model learn different ways to compare and extract information.

---

#### 3. **Where This Happens in the Transformer**

- In the **encoder**, Q, K, and V all come from the same source sequence → *self-attention*.  

Self-attention allows each token in a sequence to attend to *other tokens in the same sequence*.
If your input sentence is:
> “The cat sat on the mat.”

When encoding the word *“sat”*, the model can look at *“cat”* to understand that the subject doing the action is *“cat”*.

So every token (like *“sat”*) forms its own **Query**, compares it with **Keys** from *all* tokens (including itself), and blends their **Values** based on relevance.

$$
Q, K, V = \text{from the same input sequence (X)}
$$

This is why it’s called **self-attention** — the model is attending to *itself*.




- In the **decoder**, the first attention block is self-attention again, but the second uses:
  - Q from the decoder,  
  - K and V from the encoder outputs → *cross-attention*.

In the **decoder**, self-attention is *masked* — the model only attends to previous positions (not future words).  
This preserves the **auto-regressive** property (so it doesn’t “peek ahead” when generating text).

Cross-Attention Formula
$$
Q = \text{from decoder}, \quad K = \text{from encoder}, \quad V = \text{from encoder}
$$
The decoder sends out a **Query** like:  
> “I’m about to generate the next word — what parts of the input sentence should I focus on?”

And the encoder responds with its **Keys** and **Values** that represent the meaning of the entire input sentence.



## What Is Multi-Head Attention?

So far, we’ve talked about a **single attention mechanism** — one set of $ Q, K, V $ vectors producing one set of attention weights.  
But the Transformer paper (*“Attention Is All You Need”*) discovered that using **multiple attention “heads”** in parallel makes the model *much more powerful*.

### The Core Idea

Instead of learning **one** attention distribution, we learn **several different ones simultaneously** — each head focuses on a *different type of relationship* between tokens.

Mathematically:

$$
Q_i = XW_Q^{(i)}, \quad K_i = XW_K^{(i)}, \quad V_i = XW_V^{(i)}
$$

$$
\text{head}_i = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_k}}\right)V_i
$$

Then combine:

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W_O
$$

where $ W_O $ is a learned projection back to the model dimension.


```python
# attention_modules.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------
# Utilities
# ---------------------------

def make_causal_mask(seq_len: int, device=None):
    """
    Returns a [1, 1, seq_len, seq_len] upper-triangular causal mask with -inf above diagonal.
    Add this to attention scores BEFORE softmax.
    """
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)  # (L, L) upper triangle set to -inf
    return mask[None, None, :, :]        # (1, 1, L, L)


def apply_padding_mask(scores, key_padding_mask):
    """
    scores: (B, H, QL, KL)
    key_padding_mask: (B, KL) with True where positions are PAD/should be masked
    Returns scores with -inf where keys are padding.
    """
    if key_padding_mask is None:
        return scores
    # Expand to (B, 1, 1, KL) then broadcast
    mask = key_padding_mask[:, None, None, :].to(dtype=torch.bool)
    scores = scores.masked_fill(mask, float("-inf"))
    return scores


def scaled_dot_product_attention(Q, K, V, attn_mask=None, key_padding_mask=None, dropout_p=0.0):
    """
    Q: (B, H, QL, D)
    K: (B, H, KL, D)
    V: (B, H, KL, D)
    attn_mask: (1 or B, 1 or H, QL, KL) additive mask with 0 for keep and -inf for block
    key_padding_mask: (B, KL) boolean, True=mask (ignore)
    returns: (context, attn_weights)
      context: (B, H, QL, D)
      attn_weights: (B, H, QL, KL)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)  # (B, H, QL, KL)

    if attn_mask is not None:
        # Broadcast-compatible additive mask
        scores = scores + attn_mask

    scores = apply_padding_mask(scores, key_padding_mask)

    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0:
        attn = F.dropout(attn, p=dropout_p, training=Q.requires_grad)

    context = torch.matmul(attn, V)  # (B, H, QL, D)
    return context, attn


# ---------------------------
# (1) Single-Head Self-Attention
# ---------------------------

class SingleHeadSelfAttention(nn.Module):
    """
    Single-head self-attention.
    Inputs:
      x: (B, L, d_model)
      key_padding_mask: (B, L) boolean, True=pad (optional)
      causal: bool — apply causal mask (optional, default False)
    Returns:
      y: (B, L, d_model_head)  # equals head_dim
      attn: (B, 1, L, L)
    """
    def __init__(self, d_model: int, head_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, x, key_padding_mask=None, causal: bool = False):
        B, L, _ = x.shape
        Q = self.q(x).unsqueeze(1)  # (B, 1, L, D)
        K = self.k(x).unsqueeze(1)  # (B, 1, L, D)
        V = self.v(x).unsqueeze(1)  # (B, 1, L, D)

        attn_mask = make_causal_mask(L, device=x.device) if causal else None
        context, attn = scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, key_padding_mask=key_padding_mask, dropout_p=self.attn_dropout
        )
        y = context.squeeze(1)  # (B, L, D)
        return y, attn  # attn: (B, 1, L, L)


# ---------------------------
# (2) Single-Head Cross-Attention
# ---------------------------

class SingleHeadCrossAttention(nn.Module):
    """
    Single-head cross-attention.
    Inputs:
      q_inp: (B, Lq, d_model)      # e.g., decoder states
      kv_inp: (B, Lk, d_model)     # e.g., encoder outputs (memory)
      key_padding_mask: (B, Lk) boolean, True=pad (optional)
    Returns:
      y: (B, Lq, head_dim)
      attn: (B, 1, Lq, Lk)
    """
    def __init__(self, d_model: int, head_dim: int, attn_dropout: float = 0.0):
        super().__init__()
        self.q = nn.Linear(d_model, head_dim, bias=False)
        self.k = nn.Linear(d_model, head_dim, bias=False)
        self.v = nn.Linear(d_model, head_dim, bias=False)
        self.attn_dropout = attn_dropout

    def forward(self, q_inp, kv_inp, key_padding_mask=None):
        B, Lq, _ = q_inp.shape
        _, Lk, _ = kv_inp.shape
        Q = self.q(q_inp).unsqueeze(1)  # (B, 1, Lq, D)
        K = self.k(kv_inp).unsqueeze(1) # (B, 1, Lk, D)
        V = self.v(kv_inp).unsqueeze(1) # (B, 1, Lk, D)

        context, attn = scaled_dot_product_attention(
            Q, K, V, attn_mask=None, key_padding_mask=key_padding_mask, dropout_p=self.attn_dropout
        )
        y = context.squeeze(1)  # (B, Lq, D)
        return y, attn  # (B, 1, Lq, Lk)


# ---------------------------
# (3) Multi-Head Self-Attention
# ---------------------------

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with output projection.
    Inputs:
      x: (B, L, d_model)
      key_padding_mask: (B, L) boolean, True=pad (optional)
      causal: bool (optional)
    Returns:
      y: (B, L, d_model)
      attn: (B, H, L, L)
    """
    def __init__(self, d_model: int, num_heads: int, head_dim: int, attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.out = nn.Linear(inner_dim, d_model, bias=False)

        self.attn_dropout = attn_dropout
        self.out_dropout = nn.Dropout(out_dropout)

    def _reshape_heads(self, t):
        # (B, L, H*D) -> (B, H, L, D)
        B, L, _ = t.shape
        t = t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return t

    def forward(self, x, key_padding_mask=None, causal: bool = False):
        B, L, _ = x.shape
        Q = self._reshape_heads(self.q(x))
        K = self._reshape_heads(self.k(x))
        V = self._reshape_heads(self.v(x))
        # Q,K,V: (B, H, L, D)

        attn_mask = make_causal_mask(L, device=x.device) if causal else None
        context, attn = scaled_dot_product_attention(
            Q, K, V, attn_mask=attn_mask, key_padding_mask=key_padding_mask, dropout_p=self.attn_dropout
        )  # context: (B,H,L,D)

        # Merge heads: (B, L, H*D)
        context = context.transpose(1, 2).contiguous().view(B, L, self.num_heads * self.head_dim)
        y = self.out_dropout(self.out(context))  # (B, L, d_model)
        return y, attn  # (B,H,L,L)


# ---------------------------
# (4) Multi-Head Cross-Attention
# ---------------------------

class MultiHeadCrossAttention(nn.Module):
    """
    Multi-head cross-attention with output projection.
    Inputs:
      q_inp: (B, Lq, d_model)      # e.g., decoder states
      kv_inp: (B, Lk, d_model)     # e.g., encoder outputs (memory)
      key_padding_mask: (B, Lk) boolean, True=pad (optional)
    Returns:
      y: (B, Lq, d_model)
      attn: (B, H, Lq, Lk)
    """
    def __init__(self, d_model: int, num_heads: int, head_dim: int, attn_dropout: float = 0.0, out_dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim

        self.q = nn.Linear(d_model, inner_dim, bias=False)
        self.k = nn.Linear(d_model, inner_dim, bias=False)
        self.v = nn.Linear(d_model, inner_dim, bias=False)
        self.out = nn.Linear(inner_dim, d_model, bias=False)

        self.attn_dropout = attn_dropout
        self.out_dropout = nn.Dropout(out_dropout)

    def _reshape_heads(self, t):
        # (B, L, H*D) -> (B, H, L, D)
        B, L, _ = t.shape
        t = t.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        return t

    def forward(self, q_inp, kv_inp, key_padding_mask=None):
        B, Lq, _ = q_inp.shape
        _, Lk, _ = kv_inp.shape

        Q = self._reshape_heads(self.q(q_inp))     # (B,H,Lq,D)
        K = self._reshape_heads(self.k(kv_inp))    # (B,H,Lk,D)
        V = self._reshape_heads(self.v(kv_inp))    # (B,H,Lk,D)

        context, attn = scaled_dot_product_attention(
            Q, K, V, attn_mask=None, key_padding_mask=key_padding_mask, dropout_p=self.attn_dropout
        )  # context: (B,H,Lq,D)

        context = context.transpose(1, 2).contiguous().view(B, Lq, self.num_heads * self.head_dim)
        y = self.out_dropout(self.out(context))  # (B, Lq, d_model)
        return y, attn  # (B,H,Lq,Lk)

```
