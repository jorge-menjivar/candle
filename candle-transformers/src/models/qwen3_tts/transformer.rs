//! Decoder-only transformer shared by the Qwen3-TTS talker, its code predictor and the
//! speech-tokenizer decoder.
//!
//! The three stacks are all Qwen3-style (pre-norm, SwiGLU MLP, RoPE, grouped-query attention)
//! and only differ in a few knobs captured by [`TransformerConfig`]: whether queries/keys get a
//! per-head RMS normalization, whether the residual branches carry a learnt layer scale and
//! whether attention is restricted to a sliding window.
//!
//! The talker uses multimodal RoPE (`mrope`) in the reference implementation. Text-to-speech only
//! ever feeds it text/codec tokens, so the three position streams are identical and the result is
//! exactly the standard 1D rotary embedding implemented here.

use crate::models::with_tracing::{linear_b, linear_no_bias, Linear, RmsNorm};
use crate::utils::repeat_kv;
use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::{kv_cache::ConcatKvCache, Activation, VarBuilder};
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub hidden_act: Activation,
    pub attention_bias: bool,
    /// Qwen3 per-head RMS normalization of the queries and keys.
    pub qk_norm: bool,
    /// Learnt per-channel scaling of the residual branches.
    pub layer_scale: bool,
    /// Restrict every query to the `sliding_window` most recent keys (itself included).
    pub sliding_window: Option<usize>,
}

#[derive(Debug, Clone)]
pub(super) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub(super) fn new(
        dtype: DType,
        dim: usize,
        rope_theta: f64,
        max_seq_len: usize,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?.to_dtype(dtype)?,
            cos: freqs.cos()?.to_dtype(dtype)?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    pub(super) fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?;
        let sin = self.sin.narrow(0, offset, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))?,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

#[derive(Debug, Clone)]
struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: Option<RmsNorm>,
    k_norm: Option<RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: ConcatKvCache,
}

impl Attention {
    fn new(
        cfg: &TransformerConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = cfg.head_dim;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let b = cfg.attention_bias;
        let q_proj = linear_b(cfg.hidden_size, num_heads * head_dim, b, vb.pp("q_proj"))?;
        let k_proj = linear_b(cfg.hidden_size, num_kv_heads * head_dim, b, vb.pp("k_proj"))?;
        let v_proj = linear_b(cfg.hidden_size, num_kv_heads * head_dim, b, vb.pp("v_proj"))?;
        let o_proj = linear_b(num_heads * head_dim, cfg.hidden_size, b, vb.pp("o_proj"))?;
        let (q_norm, k_norm) = if cfg.qk_norm {
            let q_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
            let k_norm = RmsNorm::new(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            rotary_emb,
            kv_cache: ConcatKvCache::new(2),
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let (b, l, _) = xs.dims3()?;
        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // (B, L, H, D) -> (B, H, L, D)
        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(q_norm), Some(k_norm)) => {
                // Per-head RMS normalization, flatten the leading dims so the norm runs on a
                // contiguous 2D view.
                let q = q_norm.forward(&q.flatten(0, 2)?)?.reshape((
                    b,
                    self.num_heads,
                    l,
                    self.head_dim,
                ))?;
                let k = k_norm.forward(&k.flatten(0, 2)?)?.reshape((
                    b,
                    self.num_kv_heads,
                    l,
                    self.head_dim,
                ))?;
                (q, k)
            }
            _ => (q, k),
        };

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;
        let (k, v) = self.kv_cache.append(&k, &v)?;
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(mask) = attn_mask {
            scores = scores.broadcast_add(mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        probs
            .matmul(&v)?
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?
            .apply(&self.o_proj)
    }

    fn clear_kv_cache(&mut self) {
        self.kv_cache.reset()
    }
}

#[derive(Debug, Clone)]
struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            scale: vb.get(size, "scale")?,
        })
    }
}

impl Module for LayerScale {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.broadcast_mul(&self.scale)
    }
}

#[derive(Debug, Clone)]
struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    self_attn_layer_scale: Option<LayerScale>,
    mlp_layer_scale: Option<LayerScale>,
}

impl DecoderLayer {
    fn new(
        cfg: &TransformerConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(cfg, rotary_emb, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::new(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        let (self_attn_layer_scale, mlp_layer_scale) = if cfg.layer_scale {
            (
                Some(LayerScale::new(
                    cfg.hidden_size,
                    vb.pp("self_attn_layer_scale"),
                )?),
                Some(LayerScale::new(cfg.hidden_size, vb.pp("mlp_layer_scale"))?),
            )
        } else {
            (None, None)
        };
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            self_attn_layer_scale,
            mlp_layer_scale,
        })
    }

    fn forward(
        &mut self,
        xs: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
    ) -> Result<Tensor> {
        let h = self.input_layernorm.forward(xs)?;
        let h = self.self_attn.forward(&h, attn_mask, offset)?;
        let h = match &self.self_attn_layer_scale {
            Some(ls) => h.apply(ls)?,
            None => h,
        };
        let xs = (xs + h)?;
        let h = self
            .post_attention_layernorm
            .forward(&xs)?
            .apply(&self.mlp)?;
        let h = match &self.mlp_layer_scale {
            Some(ls) => h.apply(ls)?,
            None => h,
        };
        xs + h
    }

    fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache()
    }
}

/// A stack of decoder layers followed by the final RMS normalization.
///
/// The model works on input embeddings rather than token ids: the various Qwen3-TTS front-ends
/// sum text, codec and speaker embeddings before feeding the stack.
#[derive(Debug, Clone)]
pub struct Transformer {
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    sliding_window: Option<usize>,
    device: Device,
    dtype: DType,
}

impl Transformer {
    /// `vb` is expected to point at the module holding `layers` and `norm`.
    pub fn new(cfg: &TransformerConfig, vb: VarBuilder) -> Result<Self> {
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            vb.dtype(),
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_position_embeddings,
            vb.device(),
        )?);
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg,
                rotary_emb.clone(),
                vb_l.pp(layer_idx),
            )?);
        }
        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        Ok(Self {
            layers,
            norm,
            sliding_window: cfg.sliding_window,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.clear_kv_cache()
        }
    }

    /// Runs the stack on `xs` (B x L x hidden) whose first token sits at position `offset` and
    /// returns the normalized hidden states (B x L x hidden). Keys and values are appended to the
    /// per-layer cache, so consecutive calls must use increasing offsets.
    pub fn forward(&mut self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let (_b, seq_len, _) = xs.dims3()?;
        let attn_mask = if seq_len > 1 || self.sliding_window.is_some() {
            // The reference implementation lets a query attend to the keys `j` such that
            // `i - j < w`, i.e. the window covers `w` keys including the query itself, whereas
            // the candle helper covers `w + 1` keys.
            let sw = self.sliding_window.map(|w| w.saturating_sub(1));
            Some(crate::utils::build_additive_causal_mask(
                seq_len,
                offset,
                sw,
                &self.device,
                self.dtype,
            )?)
        } else {
            None
        };
        let mut xs = xs.clone();
        for layer in self.layers.iter_mut() {
            xs = layer.forward(&xs, attn_mask.as_ref(), offset)?;
        }
        self.norm.forward(&xs)
    }
}
