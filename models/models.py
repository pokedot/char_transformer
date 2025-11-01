"""
models.py

This file is structured by sections so that all positional-encoding
helpers are grouped together (Sinusoidal, Learned, and Rotary/RoPE),
followed by attention, feed-forward, block and model definitions.

Sections:
  - Imports & small utilities
  - Positional encodings
      * Sinusoidal (absolute)
      * Learned (created as params in model.setup)
      * Rotary / RoPE helpers (sin/cos builder, rotate and apply)
  - Attention modules (RotarySelfAttention)
  - Feed-forward (MLP)
  - Decoder block
  - Decoder-only Transformer (entry point)

Functionality is preserved. Comments and docstrings added to make it
clear which pieces belong to each positional encoding approach.
"""
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen import attention as attn
import math
# ----------------------------
# Positional encodings
# ----------------------------
# All helpers related to positional encodings are grouped here so it's easy
# to find and reason about each approach.
# Supported approaches:
#  - "sinusoidal": absolute sinusoidal encodings added to token embeddings
#  - "rope"      : rotary positional embeddings applied inside attention
#  - "learned"   : learned positional embeddings (registered parameter)
# ----------------------------

# ---- Sinusoidal absolute positional embeddings ----
def sinusoidal_positions(seq_len, d_model):
    """Returns (seq_len, d_model) sinusoidal positional encodings."""
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

# ---- Rotary Position Embeddings (RoPE) helpers ----
# These helpers are used by the RoPE-aware attention implementation below.
def _build_rope_sin_cos(max_seq, head_dim):
    """Build sin/cos arrays for RoPE.

    Returns:
      sin: shape (max_seq, head_dim)
      cos: shape (max_seq, head_dim)
    We interleave sin/cos values so each pair of dims uses the same frequency.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, head_dim, 2) / head_dim))
    positions = jnp.arange(max_seq)
    sinusoid_inp = jnp.einsum("i,j->ij", positions, inv_freq)  # (max_seq, head_dim/2)
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)
    # repeat each value twice so shape becomes (max_seq, head_dim)
    sin = jnp.repeat(sin, 2, axis=-1)
    cos = jnp.repeat(cos, 2, axis=-1)
    return sin, cos

def _rotate_half(x):
    """Rotate pairs of dimensions: (x0, x1, x2, x3, ...) -> (-x1, x0, -x3, x2, ...)"""
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    # stack as pairs: (-x2, x1)
    x_rot = jnp.stack((-x2, x1), axis=-1)
    return x_rot.reshape(x.shape)
    
def apply_rope(q, k, sin, cos):
    """Apply RoPE to q and k.

    Args:
      q, k: arrays of shape (B, T, H, head_dim)
      sin, cos: arrays of shape (T, head_dim) or broadcastable shape

    Returns:
      q_rot, k_rot with same shapes as inputs
    """
    # ensure sin/cos have shape (1, T, 1, head_dim) for broadcasting
    # input sin/cos expected as (T, head_dim)
    sin = sin[None, :, None, :]
    cos = cos[None, :, None, :]
    q_rot = (q * cos) + (_rotate_half(q) * sin)
    k_rot = (k * cos) + (_rotate_half(k) * sin)
    return q_rot, k_rot

# ----------------------------
# Attention modules
# ----------------------------
# RotarySelfAttention groups the RoPE-specific logic so RoPE-related
# helpers are colocated with the module that consumes them.
# ----------------------------
class RotarySelfAttention(nn.Module):
    d_model: int
    n_heads: int
    head_dim: int
    dropout: float = 0.0
    deterministic: bool = True
    max_len: int = 512
    use_bias: bool = False

    @nn.compact
    def __call__(self, x, mask=None):
        """x: (B, T, d_model). mask: boolean causal mask (B, T, T) or (B,1,T,T)"""
        B, T, D = x.shape
        assert D == self.d_model, f"d_model mismatch: {D} vs {self.d_model}"
        # Project to q,k,v with DenseGeneral -> (B, T, n_heads, head_dim)
        q = nn.DenseGeneral(features=(self.n_heads, self.head_dim), use_bias=self.use_bias, name="q_proj")(x)
        k = nn.DenseGeneral(features=(self.n_heads, self.head_dim), use_bias=self.use_bias, name="k_proj")(x)
        v = nn.DenseGeneral(features=(self.n_heads, self.head_dim), use_bias=self.use_bias, name="v_proj")(x)

        # compute sin/cos up to T (RoPE uses absolute positions)
        sin, cos = _build_rope_sin_cos(self.max_len, self.head_dim)  # (max_len, head_dim)
        sin_t = sin[:T, :]
        cos_t = cos[:T, :]

        # apply RoPE to q,k
        q, k = apply_rope(q, k, sin_t, cos_t)

        # transpose to (B, heads, T, head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        # scaled dot-product attention
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_logits = jnp.einsum("bhqd,bhkd->bhqk", q, k) * scale  # (B, H, Q, K)

        if mask is not None:
            # mask is boolean where True means valid
            # accept masks of shapes (B, T, T) or (B, 1, T, T) or (B, T)
            if mask.ndim == 3:
                # (B, T, T) -> (B, 1, T, T)
                mask_b = mask[:, None, :, :]
            elif mask.ndim == 4:
                mask_b = mask
            elif mask.ndim == 2:
                # (B, T) -> causal mask from sequence lengths? fallback to causal
                mask_b = attn.make_causal_mask(mask)
            else:
                mask_b = mask
            # convert to float mask: 1.0 valid, 0.0 invalid
            mask_float = mask_b.astype(jnp.float32)
            # add large negative where invalid (1 - mask)
            attn_logits = attn_logits + (1.0 - mask_float) * -1e10

        attn_weights = nn.softmax(attn_logits, axis=-1)
        attn_weights = nn.Dropout(rate=self.dropout)(attn_weights, deterministic=self.deterministic)

        attn_out = jnp.einsum("bhqk,bhkd->bhqd", attn_weights, v)  # (B, H, Q, head_dim)
        # transpose back to (B, Q, H, head_dim) -> reshape to (B, Q, d_model)
        attn_out = jnp.transpose(attn_out, (0, 2, 1, 3)).reshape(B, T, -1)

        # final linear projection
        out = nn.Dense(self.d_model, use_bias=False, name="out_proj")(attn_out)
        out = nn.Dropout(rate=self.dropout)(out, deterministic=self.deterministic)
        return out

# ----------------------------
# Feed-forward (MLP)
# ----------------------------    
class MLP(nn.Module):
    d_model: int
    mlp_ratio: int = 4
    dropout: float = 0.0
    deterministic: bool = True

    @nn.compact
    def __call__(self, x):
        hidden = int(self.d_model * self.mlp_ratio)
        x = nn.Dense(hidden)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
        x = nn.Dense(self.d_model)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=self.deterministic)
        return x

# ----------------------------
# Decoder block
# ----------------------------
class DecoderBlock(nn.Module):
    """
    Single decoder (masked) transformer block.

    Picks attention implementation depending on pos_encoding:
      - "rope": uses RotarySelfAttention (RoPE applied inside attention)
      - otherwise: falls back to flax.linen.SelfAttention (assumes absolute pos are
        added to token embeddings prior to blocks for "sinusoidal" and "learned")
    """
    d_model: int
    n_heads: int
    mlp_ratio: int = 4
    dropout: float = 0.0
    deterministic: bool = True
    pos_encoding: str = "sinusoidal"
    max_len: int = 512

    @nn.compact
    def __call__(self, x, *, mask=None):
        h = nn.LayerNorm()(x)
        head_dim = self.d_model // self.n_heads

        if self.pos_encoding == "rope":
            # use RotarySelfAttention (compute head_dim here)
            h_attn = RotarySelfAttention(
                d_model=self.d_model,
                n_heads=self.n_heads,
                head_dim=head_dim,
                dropout=self.dropout,
                deterministic=self.deterministic,
                max_len=self.max_len
            )(h, mask=mask)
        else:
            # fallback to Flax's SelfAttention for sinusoidal/learned/no-pos
            h_attn = nn.SelfAttention(
                num_heads=self.n_heads,
                dropout_rate=self.dropout,
                deterministic=self.deterministic,
                use_bias=False,
            )(h, mask=mask)

        x = x + nn.Dropout(rate=self.dropout)(h_attn, deterministic=self.deterministic)
        h2 = nn.LayerNorm()(x)
        h2 = MLP(self.d_model, mlp_ratio=self.mlp_ratio, dropout=self.dropout, deterministic=self.deterministic)(h2)
        x = x + nn.Dropout(rate=self.dropout)(h2, deterministic=self.deterministic)
        return x

# ----------------------------
# Decoder-only Transformer
# ----------------------------
class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only Transformer supporting different positional encodings.

    pos_encoding: one of "learned", "sinusoidal", or "rope".
      - "learned": learned positional embeddings are created in setup()
      - "sinusoidal": absolute sinusoids are computed each call and added to token embeddings
      - "rope": no absolute embeddings are added; RoPE is applied inside attention

    tie_weights: if True, tie output projection to token embedding matrix
    """
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    max_len: int
    mlp_ratio: int = 4
    dropout: float = 0.0
    pos_encoding: str = "learned"  # "learned", "sinusoidal", or "rope"
    tie_weights: bool = True
    deterministic: bool = True

    def setup(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        self.tok_embed = nn.Embed(self.vocab_size, self.d_model)
        if self.pos_encoding == "learned":
            self.pos_embed = self.param("pos_embed", nn.initializers.normal(stddev=0.02), (self.max_len, self.d_model))
        # add blocks
        self.blocks = [
            DecoderBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                mlp_ratio=self.mlp_ratio,
                dropout=self.dropout,
                deterministic=self.deterministic,
                pos_encoding=self.pos_encoding,
                max_len=self.max_len,
            ) for _ in range(self.n_layers)
        ]
        self.layerNorm_final = nn.LayerNorm()
        if not self.tie_weights:
            self.project_to_vocab = nn.Dense(self.vocab_size, use_bias=False)

    def __call__(self, idx, *, deterministic=None):
        deterministic = self.deterministic if deterministic is None else deterministic
        B, T = idx.shape
        x = self.tok_embed(idx)
        # Positional encoding
        if self.pos_encoding == "learned":
            x = x + self.pos_embed[:T]
        elif self.pos_encoding == "sinusoidal":
            x = x + sinusoidal_positions(T, self.d_model)
        elif self.pos_encoding == "rope":
            # For RoPE we do not add absolute positional vectors to token embeddings.
            # The rotary transform is applied inside attention.
            pass

        mask = attn.make_causal_mask(jnp.ones((B, T), dtype=bool))
        for blk in self.blocks:
            x = blk(x, mask=mask)
        x = self.layerNorm_final(x)
        if self.tie_weights:
            logits = jnp.einsum('btd,vd->btv', x, self.tok_embed.embedding)
        else:
            logits = self.project_to_vocab(x)
        return logits
