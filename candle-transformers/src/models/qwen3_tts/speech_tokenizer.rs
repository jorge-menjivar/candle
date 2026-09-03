//! Qwen3-TTS-Tokenizer-12Hz: the neural audio codec used by Qwen3-TTS.
//!
//! The [`Decoder`] turns 12.5 Hz frames of 16 codebook indices into 24 kHz audio in four stages:
//! a split residual vector quantizer producing latents, a sliding-window transformer, two
//! ConvNeXt upsampling stages and a SnakeBeta/transposed convolution vocoder (each frame yields
//! 1920 samples).
//!
//! The [`Encoder`] is the Mimi encoder from Kyutai as packaged by `transformers` (`MimiModel`):
//! a SEANet convolutional encoder, a transformer, a 2x downsampling convolution and a split
//! residual vector quantizer. It reuses the [`mimi`] implementation and is only needed to
//! compute the codes of a reference recording for voice cloning.

use super::transformer::{Transformer, TransformerConfig};
use crate::models::mimi;
use candle::{DType, IndexOp, Module, Result, StreamingModule, Tensor, D};
use candle_nn::{
    linear, Activation, Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig, LayerNorm,
    Linear, VarBuilder,
};

fn default_num_semantic_quantizers() -> usize {
    1
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct DecoderConfig {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_hidden_layers: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub sliding_window: Option<usize>,
    pub num_quantizers: usize,
    #[serde(default = "default_num_semantic_quantizers")]
    pub num_semantic_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

impl DecoderConfig {
    fn transformer_config(&self) -> TransformerConfig {
        TransformerConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            head_dim: self.head_dim,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: false,
            layer_scale: true,
            sliding_window: self.sliding_window,
        }
    }

    /// Number of audio samples produced per codec frame.
    pub fn total_upsample(&self) -> usize {
        self.upsample_rates.iter().product::<usize>()
            * self.upsampling_ratios.iter().product::<usize>()
    }
}

/// Configuration of the `speech_tokenizer/config.json` file shipped with the Qwen3-TTS models.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub decoder_config: DecoderConfig,
    pub encoder_config: EncoderConfig,
    pub input_sample_rate: usize,
    pub output_sample_rate: usize,
    pub decode_upsample_rate: usize,
    pub encode_downsample_rate: usize,
    pub encoder_valid_num_quantizers: usize,
}

/// Extra right padding so that the last frame of a causal convolution is complete.
fn extra_padding(len: usize, kernel_size: usize, padding_total: usize, stride: usize) -> usize {
    let n_frames = (len as f64 + padding_total as f64 - kernel_size as f64) / stride as f64 + 1.;
    let ideal_len =
        (n_frames.ceil() as i64 - 1) * stride as i64 + (kernel_size - padding_total) as i64;
    (ideal_len - len as i64).max(0) as usize
}

/// Conv1d with causal (left) zero padding, matching `Qwen3TTSTokenizerV2CausalConvNet`.
#[derive(Debug, Clone)]
struct CausalConv1d {
    conv: Conv1d,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl CausalConv1d {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        stride: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            padding: 0,
            stride,
            dilation,
            groups,
            cudnn_fwd_algo: None,
        };
        let conv = candle_nn::conv1d(in_c, out_c, kernel_size, cfg, vb.pp("conv"))?;
        let kernel_size = (kernel_size - 1) * dilation + 1;
        Ok(Self {
            conv,
            kernel_size,
            stride,
            padding: kernel_size - stride,
        })
    }
}

impl Module for CausalConv1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let len = xs.dim(D::Minus1)?;
        let extra = extra_padding(len, self.kernel_size, self.padding, self.stride);
        xs.pad_with_zeros(D::Minus1, self.padding, extra)?
            .apply(&self.conv)
    }
}

/// ConvTranspose1d whose trailing `kernel_size - stride` samples are trimmed to keep it causal.
#[derive(Debug, Clone)]
struct CausalConvTranspose1d {
    conv: ConvTranspose1d,
    right_pad: usize,
}

impl CausalConvTranspose1d {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = ConvTranspose1dConfig {
            stride,
            ..Default::default()
        };
        let conv = candle_nn::conv_transpose1d(in_c, out_c, kernel_size, cfg, vb.pp("conv"))?;
        Ok(Self {
            conv,
            right_pad: kernel_size - stride,
        })
    }
}

impl Module for CausalConvTranspose1d {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = xs.apply(&self.conv)?;
        let len = xs.dim(D::Minus1)?;
        xs.narrow(D::Minus1, 0, len - self.right_pad)
    }
}

#[derive(Debug, Clone)]
struct ConvNeXtBlock {
    dwconv: CausalConv1d,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let dwconv = CausalConv1d::new(dim, dim, 7, 1, 1, dim, vb.pp("dwconv"))?;
        let norm = candle_nn::layer_norm(dim, 1e-6, vb.pp("norm"))?;
        let pwconv1 = linear(dim, 4 * dim, vb.pp("pwconv1"))?;
        let pwconv2 = linear(4 * dim, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;
        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gamma,
        })
    }
}

impl Module for ConvNeXtBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.dwconv)?
            .transpose(1, 2)?
            .apply(&self.norm)?
            .apply(&self.pwconv1)?
            .gelu_erf()?
            .apply(&self.pwconv2)?
            .broadcast_mul(&self.gamma)?
            .transpose(1, 2)?;
        xs + h
    }
}

/// `x + sin(x * exp(alpha))^2 / (exp(beta) + eps)` with per-channel `alpha` and `beta`.
#[derive(Debug, Clone)]
struct SnakeBeta {
    alpha: Tensor,
    inv_beta: Tensor,
}

impl SnakeBeta {
    fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb
            .get(channels, "alpha")?
            .exp()?
            .reshape((1, channels, 1))?;
        let inv_beta = (vb.get(channels, "beta")?.exp()? + 1e-9)?
            .recip()?
            .reshape((1, channels, 1))?;
        Ok(Self { alpha, inv_beta })
    }
}

impl Module for SnakeBeta {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let s = xs.broadcast_mul(&self.alpha)?.sin()?.sqr()?;
        xs + s.broadcast_mul(&self.inv_beta)?
    }
}

#[derive(Debug, Clone)]
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn new(dim: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            act1: SnakeBeta::new(dim, vb.pp("act1"))?,
            conv1: CausalConv1d::new(dim, dim, 7, dilation, 1, 1, vb.pp("conv1"))?,
            act2: SnakeBeta::new(dim, vb.pp("act2"))?,
            conv2: CausalConv1d::new(dim, dim, 1, 1, 1, 1, vb.pp("conv2"))?,
        })
    }
}

impl Module for ResidualUnit {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.act1)?
            .apply(&self.conv1)?
            .apply(&self.act2)?
            .apply(&self.conv2)?;
        xs + h
    }
}

#[derive(Debug, Clone)]
struct DecoderBlock {
    act: SnakeBeta,
    upsample: CausalConvTranspose1d,
    residual_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    fn new(in_dim: usize, out_dim: usize, upsample_rate: usize, vb: VarBuilder) -> Result<Self> {
        let vb = vb.pp("block");
        let act = SnakeBeta::new(in_dim, vb.pp(0))?;
        let upsample = CausalConvTranspose1d::new(
            in_dim,
            out_dim,
            2 * upsample_rate,
            upsample_rate,
            vb.pp(1),
        )?;
        let mut residual_units = Vec::with_capacity(3);
        for (i, dilation) in [1, 3, 9].into_iter().enumerate() {
            residual_units.push(ResidualUnit::new(out_dim, dilation, vb.pp(i + 2))?);
        }
        Ok(Self {
            act,
            upsample,
            residual_units,
        })
    }
}

impl Module for DecoderBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut xs = xs.apply(&self.act)?.apply(&self.upsample)?;
        for unit in self.residual_units.iter() {
            xs = xs.apply(unit)?;
        }
        Ok(xs)
    }
}

/// Residual vector quantizer decoder: sums the codebook entries of every layer and projects the
/// result back to the latent dimension.
#[derive(Debug, Clone)]
struct ResidualVectorQuantizer {
    codebooks: Vec<Tensor>,
    output_proj: Conv1d,
}

impl ResidualVectorQuantizer {
    fn new(
        num_quantizers: usize,
        codebook_size: usize,
        dim: usize,
        output_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut codebooks = Vec::with_capacity(num_quantizers);
        let vb_l = vb.pp("vq.layers");
        for i in 0..num_quantizers {
            let vb_c = vb_l.pp(i).pp("_codebook");
            let cluster_usage = vb_c.get(codebook_size, "cluster_usage")?;
            let embedding_sum = vb_c.get((codebook_size, dim), "embedding_sum")?;
            let embedding =
                embedding_sum.broadcast_div(&cluster_usage.clamp(1e-5, f32::MAX)?.unsqueeze(1)?)?;
            codebooks.push(embedding);
        }
        let output_proj = candle_nn::conv1d_no_bias(
            dim,
            output_dim,
            1,
            Default::default(),
            vb.pp("output_proj"),
        )?;
        Ok(Self {
            codebooks,
            output_proj,
        })
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, output_dim, T).
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let (b, nq, t) = codes.dims3()?;
        if nq != self.codebooks.len() {
            candle::bail!(
                "expected {} quantizer layers, got {nq}",
                self.codebooks.len()
            )
        }
        let mut acc: Option<Tensor> = None;
        for (i, codebook) in self.codebooks.iter().enumerate() {
            let ids = codes.i((.., i))?.flatten_all()?;
            let q = codebook.index_select(&ids, 0)?.reshape((b, t, ()))?;
            acc = Some(match acc {
                None => q,
                Some(acc) => (acc + q)?,
            });
        }
        match acc {
            Some(acc) => acc.transpose(1, 2)?.apply(&self.output_proj),
            None => candle::bail!("no quantizer layers"),
        }
    }
}

#[derive(Debug, Clone)]
struct SplitResidualVectorQuantizer {
    rvq_first: ResidualVectorQuantizer,
    rvq_rest: ResidualVectorQuantizer,
    n_q_semantic: usize,
}

impl SplitResidualVectorQuantizer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let dim = cfg.codebook_dim / 2;
        let rvq_first = ResidualVectorQuantizer::new(
            cfg.num_semantic_quantizers,
            cfg.codebook_size,
            dim,
            cfg.codebook_dim,
            vb.pp("rvq_first"),
        )?;
        let rvq_rest = ResidualVectorQuantizer::new(
            cfg.num_quantizers - cfg.num_semantic_quantizers,
            cfg.codebook_size,
            dim,
            cfg.codebook_dim,
            vb.pp("rvq_rest"),
        )?;
        Ok(Self {
            rvq_first,
            rvq_rest,
            n_q_semantic: cfg.num_semantic_quantizers,
        })
    }

    /// `codes` has shape (B, num_quantizers, T); returns (B, codebook_dim, T).
    fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let nq = codes.dim(1)?;
        let first = self
            .rvq_first
            .decode(&codes.narrow(1, 0, self.n_q_semantic)?)?;
        if nq > self.n_q_semantic {
            let rest = self.rvq_rest.decode(&codes.narrow(
                1,
                self.n_q_semantic,
                nq - self.n_q_semantic,
            )?)?;
            first + rest
        } else {
            Ok(first)
        }
    }
}

#[derive(Debug, Clone)]
struct PreTransformer {
    input_proj: Linear,
    model: Transformer,
    output_proj: Linear,
}

impl PreTransformer {
    fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let input_proj = linear(cfg.latent_dim, cfg.hidden_size, vb.pp("input_proj"))?;
        let model = Transformer::new(&cfg.transformer_config(), vb.clone())?;
        let output_proj = linear(cfg.hidden_size, cfg.latent_dim, vb.pp("output_proj"))?;
        Ok(Self {
            input_proj,
            model,
            output_proj,
        })
    }

    fn forward(&mut self, xs: &Tensor) -> Result<Tensor> {
        self.model.clear_kv_cache();
        let xs = xs.apply(&self.input_proj)?;
        self.model.forward(&xs, 0)?.apply(&self.output_proj)
    }
}

/// The speech-tokenizer decoder, weights live under `decoder.` in
/// `speech_tokenizer/model.safetensors`.
#[derive(Debug, Clone)]
pub struct Decoder {
    quantizer: SplitResidualVectorQuantizer,
    pre_conv: CausalConv1d,
    pre_transformer: PreTransformer,
    upsample: Vec<(CausalConvTranspose1d, ConvNeXtBlock)>,
    decoder_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_act: SnakeBeta,
    final_conv: CausalConv1d,
    num_quantizers: usize,
    total_upsample: usize,
}

impl Decoder {
    pub fn new(cfg: &DecoderConfig, vb: VarBuilder) -> Result<Self> {
        let quantizer = SplitResidualVectorQuantizer::new(cfg, vb.pp("quantizer"))?;
        let pre_conv = CausalConv1d::new(
            cfg.codebook_dim,
            cfg.latent_dim,
            3,
            1,
            1,
            1,
            vb.pp("pre_conv"),
        )?;
        let pre_transformer = PreTransformer::new(cfg, vb.pp("pre_transformer"))?;
        let mut upsample = Vec::with_capacity(cfg.upsampling_ratios.len());
        let vb_u = vb.pp("upsample");
        for (i, &factor) in cfg.upsampling_ratios.iter().enumerate() {
            let vb_u = vb_u.pp(i);
            let conv = CausalConvTranspose1d::new(
                cfg.latent_dim,
                cfg.latent_dim,
                factor,
                factor,
                vb_u.pp(0),
            )?;
            let convnext = ConvNeXtBlock::new(cfg.latent_dim, vb_u.pp(1))?;
            upsample.push((conv, convnext));
        }
        let vb_d = vb.pp("decoder");
        let decoder_conv =
            CausalConv1d::new(cfg.latent_dim, cfg.decoder_dim, 7, 1, 1, 1, vb_d.pp(0))?;
        let mut decoder_blocks = Vec::with_capacity(cfg.upsample_rates.len());
        for (i, &rate) in cfg.upsample_rates.iter().enumerate() {
            let in_dim = cfg.decoder_dim / 2usize.pow(i as u32);
            let out_dim = cfg.decoder_dim / 2usize.pow(i as u32 + 1);
            decoder_blocks.push(DecoderBlock::new(in_dim, out_dim, rate, vb_d.pp(i + 1))?);
        }
        let output_dim = cfg.decoder_dim / 2usize.pow(cfg.upsample_rates.len() as u32);
        let n = cfg.upsample_rates.len();
        let final_act = SnakeBeta::new(output_dim, vb_d.pp(n + 1))?;
        let final_conv = CausalConv1d::new(output_dim, 1, 7, 1, 1, 1, vb_d.pp(n + 2))?;
        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample,
            decoder_conv,
            decoder_blocks,
            final_act,
            final_conv,
            num_quantizers: cfg.num_quantizers,
            total_upsample: cfg.total_upsample(),
        })
    }

    /// Number of audio samples produced per codec frame.
    pub fn total_upsample(&self) -> usize {
        self.total_upsample
    }

    /// Decodes a chunk of codes with shape (B, num_quantizers, T) into audio (B, 1, T * 1920).
    pub fn forward(&mut self, codes: &Tensor) -> Result<Tensor> {
        let nq = codes.dim(1)?;
        if nq != self.num_quantizers {
            candle::bail!("expected {} layers of codes, got {nq}", self.num_quantizers)
        }
        let hidden = self.quantizer.decode(codes)?;
        let hidden = hidden.apply(&self.pre_conv)?.transpose(1, 2)?;
        let hidden = self.pre_transformer.forward(&hidden)?;
        let mut hidden = hidden.transpose(1, 2)?;
        for (conv, convnext) in self.upsample.iter() {
            hidden = hidden.apply(conv)?.apply(convnext)?;
        }
        let mut wav = hidden.apply(&self.decoder_conv)?;
        for block in self.decoder_blocks.iter() {
            wav = wav.apply(block)?;
        }
        wav.apply(&self.final_act)?
            .apply(&self.final_conv)?
            .clamp(-1f32, 1f32)
    }

    /// Decodes codes with shape (B, T, num_quantizers) into audio samples (B, T * 1920).
    ///
    /// Long sequences are decoded in chunks of `chunk_size` frames with `left_context_size`
    /// frames of context, mirroring the reference `chunked_decode`.
    pub fn decode(
        &mut self,
        codes: &Tensor,
        chunk_size: usize,
        left_context_size: usize,
    ) -> Result<Tensor> {
        let codes = codes.to_dtype(DType::U32)?.transpose(1, 2)?.contiguous()?;
        let num_frames = codes.dim(2)?;
        let mut wavs = Vec::new();
        let mut start = 0;
        while start < num_frames {
            let end = usize::min(start + chunk_size, num_frames);
            let context = if start > left_context_size {
                left_context_size
            } else {
                start
            };
            let chunk = codes.narrow(2, start - context, end - start + context)?;
            let wav = self.forward(&chunk)?;
            let skip = context * self.total_upsample;
            let len = wav.dim(D::Minus1)?;
            wavs.push(wav.narrow(D::Minus1, skip, len - skip)?);
            start = end;
        }
        Tensor::cat(&wavs, D::Minus1)?.squeeze(1)
    }
}

fn default_frame_rate() -> f64 {
    12.5
}
fn default_true() -> bool {
    true
}
fn default_pad_mode() -> String {
    "constant".to_string()
}
fn default_layer_scale() -> f64 {
    0.01
}

/// Configuration of the encoder, the `encoder_config` section of `speech_tokenizer/config.json`
/// (a `transformers` `MimiConfig`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct EncoderConfig {
    pub audio_channels: usize,
    pub num_filters: usize,
    pub upsampling_ratios: Vec<usize>,
    pub kernel_size: usize,
    pub residual_kernel_size: usize,
    pub last_kernel_size: usize,
    pub dilation_growth_rate: usize,
    pub compress: usize,
    pub num_residual_layers: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: Option<usize>,
    pub intermediate_size: usize,
    pub hidden_act: Activation,
    pub norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub num_quantizers: usize,
    pub num_semantic_quantizers: usize,
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub vector_quantization_hidden_dimension: usize,
    pub sampling_rate: usize,
    #[serde(rename = "_frame_rate", default = "default_frame_rate")]
    pub frame_rate: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default = "default_layer_scale")]
    pub layer_scale_initial_scale: f64,
    #[serde(default = "default_true")]
    pub use_causal_conv: bool,
    #[serde(default)]
    pub use_conv_shortcut: bool,
    #[serde(default = "default_pad_mode")]
    pub pad_mode: String,
}

impl EncoderConfig {
    /// Frame rate of the convolutional encoder, before the downsampling convolution.
    fn encoder_frame_rate(&self) -> f64 {
        self.sampling_rate as f64 / self.upsampling_ratios.iter().product::<usize>() as f64
    }

    /// Maps the `transformers` configuration to the one of the [`mimi`] implementation.
    fn mimi_config(&self) -> Result<(mimi::seanet::Config, mimi::transformer::Config)> {
        if !self.use_causal_conv {
            candle::bail!("only causal convolutions are supported")
        }
        if self.hidden_act != Activation::Gelu {
            candle::bail!("unsupported activation {:?}", self.hidden_act)
        }
        if self.num_key_value_heads != self.num_attention_heads {
            candle::bail!("grouped query attention is not supported")
        }
        if self.codebook_dim != self.vector_quantization_hidden_dimension {
            candle::bail!(
                "the codebook dim {} and the quantization dim {} should match",
                self.codebook_dim,
                self.vector_quantization_hidden_dimension
            )
        }
        // The `mimi` implementation uses the default layer norm epsilon.
        if self.norm_eps != 1e-5 {
            candle::bail!("unsupported norm eps {}", self.norm_eps)
        }
        if self.head_dim() * self.num_attention_heads != self.hidden_size {
            candle::bail!("unsupported head dim {}", self.head_dim())
        }
        let pad_mode = match self.pad_mode.as_str() {
            "constant" => mimi::conv::PadMode::Constant,
            "replicate" => mimi::conv::PadMode::Replicate,
            m => candle::bail!("unsupported pad mode {m}"),
        };
        let seanet = mimi::seanet::Config {
            dimension: self.hidden_size,
            channels: self.audio_channels,
            causal: self.use_causal_conv,
            n_filters: self.num_filters,
            n_residual_layers: self.num_residual_layers,
            activation: Activation::Elu(1.),
            compress: self.compress,
            dilation_base: self.dilation_growth_rate,
            disable_norm_outer_blocks: 0,
            final_activation: None,
            kernel_size: self.kernel_size,
            residual_kernel_size: self.residual_kernel_size,
            last_kernel_size: self.last_kernel_size,
            lstm: 0,
            // The weights are stored with the norm already applied, which the `mimi`
            // implementation detects.
            norm: mimi::conv::Norm::WeightNorm,
            pad_mode,
            ratios: self.upsampling_ratios.clone(),
            true_skip: !self.use_conv_shortcut,
        };
        let transformer = mimi::transformer::Config {
            d_model: self.hidden_size,
            num_heads: self.num_attention_heads,
            num_layers: self.num_hidden_layers,
            causal: true,
            norm_first: true,
            bias_ff: false,
            bias_attn: self.attention_bias,
            layer_scale: Some(self.layer_scale_initial_scale),
            // `transformers` only applies `sliding_window` on the flash-attention path, so the
            // whole recording is attended to.
            context: self.max_position_embeddings,
            max_period: self.rope_theta as usize,
            max_seq_len: self.max_position_embeddings,
            positional_embedding: mimi::transformer::PositionalEmbedding::Rope,
            norm: mimi::NormType::LayerNorm,
            use_conv_block: false,
            conv_kernel_size: 3,
            use_conv_bias: true,
            cross_attention: false,
            gating: None,
            dim_feedforward: self.intermediate_size,
            kv_repeat: 1,
            conv_layout: true,
        };
        Ok((seanet, transformer))
    }

    fn head_dim(&self) -> usize {
        self.head_dim
            .unwrap_or(self.hidden_size / self.num_attention_heads)
    }
}

/// The Mimi encoder of the speech tokenizer, built from the shared [`mimi`] implementation.
///
/// The checkpoint stores a `transformers` `MimiModel` encoder under the `encoder.` prefix, which
/// is the layout the `mimi` module loads.
#[derive(Debug, Clone)]
pub struct Encoder {
    encoder: mimi::seanet::SeaNetEncoder,
    transformer: mimi::transformer::ProjectedTransformer,
    downsample: Option<mimi::conv::ConvDownsample1d>,
    quantizer: mimi::quantization::SplitResidualVectorQuantizer,
    downsample_rate: usize,
}

impl Encoder {
    /// `num_quantizers` codebooks are used out of the ones available and `downsample_rate` is the
    /// number of audio samples per frame.
    pub fn new(
        cfg: &EncoderConfig,
        num_quantizers: usize,
        downsample_rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        if num_quantizers < cfg.num_semantic_quantizers || num_quantizers > cfg.num_quantizers {
            candle::bail!(
                "the number of quantizers should be in {}..={}, got {num_quantizers}",
                cfg.num_semantic_quantizers,
                cfg.num_quantizers
            )
        }
        if cfg.num_semantic_quantizers != 1 {
            candle::bail!("only a single semantic quantizer is supported")
        }
        let (seanet_cfg, transformer_cfg) = cfg.mimi_config()?;
        let dim = cfg.hidden_size;
        let encoder = mimi::seanet::SeaNetEncoder::new(&seanet_cfg, vb.pp("encoder"))?;
        let transformer = mimi::transformer::ProjectedTransformer::new(
            dim,
            &[dim],
            &transformer_cfg,
            vb.pp("encoder_transformer"),
        )?;
        // The convolution is only there when the frame rate differs from the one of the
        // convolutional encoder.
        let stride = cfg.encoder_frame_rate() / cfg.frame_rate;
        let downsample = if stride > 1. {
            Some(mimi::conv::ConvDownsample1d::new(
                stride as usize,
                dim,
                cfg.use_causal_conv,
                true,
                vb.pp("downsample"),
            )?)
        } else {
            None
        };
        let quantizer = mimi::quantization::SplitResidualVectorQuantizer::new(
            cfg.codebook_dim,
            Some(dim),
            Some(dim),
            num_quantizers,
            cfg.codebook_size,
            vb.pp("quantizer"),
        )?;
        Ok(Self {
            encoder,
            transformer,
            downsample,
            quantizer,
            downsample_rate,
        })
    }

    /// Encodes audio of shape (B, samples) into codes of shape (B, frames, num_quantizers).
    pub fn encode(&mut self, pcm: &Tensor) -> Result<Tensor> {
        let (_b, len) = pcm.dims2()?;
        let xs = self.encoder.forward(&pcm.unsqueeze(1)?)?;
        self.transformer.reset_state();
        let xs = self.transformer.forward(&xs)?;
        let mut xs = xs[0].clone();
        if let Some(downsample) = &self.downsample {
            xs = xs.apply(downsample)?;
        }
        let codes = self.quantizer.encode(&xs)?;
        let num_frames = usize::min(len.div_ceil(self.downsample_rate), codes.dim(2)?);
        codes
            .narrow(2, 0, num_frames)?
            .transpose(1, 2)?
            .contiguous()
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    decoder: Decoder,
    encoder: Option<Encoder>,
    config: Config,
}

impl Model {
    /// Loads the decoder only.
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let decoder = Decoder::new(&cfg.decoder_config, vb.pp("decoder"))?;
        Ok(Self {
            decoder,
            encoder: None,
            config: cfg.clone(),
        })
    }

    /// Loads both the encoder and the decoder.
    pub fn new_with_encoder(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let mut model = Self::new(cfg, vb.clone())?;
        model.encoder = Some(Encoder::new(
            &cfg.encoder_config,
            cfg.encoder_valid_num_quantizers,
            cfg.encode_downsample_rate,
            vb.pp("encoder"),
        )?);
        Ok(model)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn input_sample_rate(&self) -> usize {
        self.config.input_sample_rate
    }

    pub fn output_sample_rate(&self) -> usize {
        self.config.output_sample_rate
    }

    /// Encodes audio of shape (B, samples) sampled at [`Model::input_sample_rate`] into codes
    /// of shape (B, frames, num_quantizers). Requires [`Model::new_with_encoder`].
    pub fn encode(&mut self, pcm: &Tensor) -> Result<Tensor> {
        match &mut self.encoder {
            Some(encoder) => encoder.encode(pcm),
            None => candle::bail!("the encoder was not loaded, use Model::new_with_encoder"),
        }
    }

    /// Number of audio samples produced for each codec frame.
    pub fn samples_per_frame(&self) -> usize {
        self.decoder.total_upsample()
    }

    /// Decodes codes with shape (B, T, num_quantizers) into audio samples (B, T * 1920).
    pub fn decode(&mut self, codes: &Tensor) -> Result<Tensor> {
        self.decode_with(codes, 300, 25)
    }

    /// Decodes codes with shape (B, T, num_quantizers) in chunks of `chunk_size` frames, each
    /// of them decoded with the `left_context` frames that precede it so that the result does
    /// not depend on where the chunk boundaries fall. Used for streaming, where the chunks are
    /// decoded as they are generated.
    pub fn decode_with(
        &mut self,
        codes: &Tensor,
        chunk_size: usize,
        left_context: usize,
    ) -> Result<Tensor> {
        self.decoder.decode(codes, chunk_size, left_context)
    }
}
