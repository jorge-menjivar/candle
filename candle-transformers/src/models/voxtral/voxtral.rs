use super::voxtral_llama::{Cache as LlamaCache, Config as LlamaConfig, VoxtralLlama};
use candle::{DType, Device, IndexOp, Module, Result, Tensor, D};
use candle_nn::{
    layer_norm, linear, linear_no_bias, Conv1d, Dropout, LayerNorm, Linear, VarBuilder,
};
use rand::Rng;
use tracing::debug;

/// Expand a single audio token position to accommodate multiple audio embeddings
fn expand_single_audio_token(
    inputs_embeds: &Tensor,
    audio_embeds: &Tensor,
    audio_position: (usize, usize),
    device: &Device,
) -> Result<Tensor> {
    let (batch_idx, seq_idx) = audio_position;
    let (batch_size, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let total_audio_embeds = audio_embeds.dim(0)?;

    debug!(
        "Expanding sequence from {} to {} positions for audio embeddings",
        seq_len,
        seq_len - 1 + total_audio_embeds
    );

    // Create new sequence: [prefix] + [audio_embeddings] + [suffix]
    let mut result_parts = Vec::new();

    // Add prefix (everything before audio token)
    if seq_idx > 0 {
        let prefix = inputs_embeds.i((.., 0..seq_idx, ..))?;
        result_parts.push(prefix);
    }

    // Add all audio embeddings (expand from 2D to 3D to match batch dimension)
    let audio_embeds_expanded =
        audio_embeds
            .unsqueeze(0)?
            .broadcast_as((batch_size, total_audio_embeds, hidden_size))?;
    result_parts.push(audio_embeds_expanded);

    // Add suffix (everything after audio token)
    if seq_idx + 1 < seq_len {
        let suffix = inputs_embeds.i((.., (seq_idx + 1).., ..))?;
        result_parts.push(suffix);
    }

    // Concatenate along sequence dimension
    if result_parts.len() == 1 {
        Ok(result_parts.into_iter().next().unwrap())
    } else {
        Tensor::cat(&result_parts, 1)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralEncoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub scale_embedding: bool,
    pub activation_function: String,
    pub num_mel_bins: usize,
    pub max_source_positions: usize,
    pub initializer_range: f64,
    pub attention_dropout: f64,
    // These are set to 0.0 for compatibility with Whisper modular architecture
    pub dropout: f64,
    pub layerdrop: f64,
    pub activation_dropout: f64,
}

#[derive(Debug, Clone)]
pub struct VoxtralConfig {
    pub audio_config: VoxtralEncoderConfig,
    pub text_config: LlamaConfig,
    pub audio_token_id: usize,
    pub projector_hidden_act: String,
}

impl Default for VoxtralConfig {
    fn default() -> Self {
        Self {
            audio_config: VoxtralEncoderConfig::default(),
            text_config: LlamaConfig::voxtral_3b(),
            audio_token_id: 24,
            projector_hidden_act: "gelu".to_string(),
        }
    }
}

impl Default for VoxtralEncoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 51866,
            hidden_size: 1280,
            intermediate_size: 5120,
            num_hidden_layers: 32,
            num_attention_heads: 20,
            num_key_value_heads: 20,
            head_dim: 64,
            scale_embedding: false,
            activation_function: "gelu".to_string(),
            num_mel_bins: 128,
            max_source_positions: 1500,
            initializer_range: 0.02,
            attention_dropout: 0.0,
            // Set for Whisper compatibility
            dropout: 0.0,
            layerdrop: 0.0,
            activation_dropout: 0.0,
        }
    }
}

impl VoxtralEncoderConfig {
    /// Ensures dropout values are properly set for Whisper compatibility
    pub fn with_whisper_compatibility(mut self) -> Self {
        self.dropout = 0.0;
        self.layerdrop = 0.0;
        self.activation_dropout = 0.0;
        self
    }
}

/// Custom cache for multimodal inputs
#[derive(Debug, Clone)]
pub struct VoxtralCache {
    llama_cache: LlamaCache,
    audio_processed: bool,
    cached_audio_embeds: Option<Tensor>,
    cached_audio_positions: Option<Vec<(usize, usize)>>,
    config: LlamaConfig,
}

impl VoxtralCache {
    pub fn new(
        use_kv_cache: bool,
        dtype: DType,
        config: &LlamaConfig,
        device: &Device,
    ) -> Result<Self> {
        Ok(Self {
            llama_cache: LlamaCache::new(use_kv_cache, dtype, config, device)?,
            audio_processed: false,
            cached_audio_embeds: None,
            cached_audio_positions: None,
            config: config.clone(),
        })
    }

    pub fn reset(&mut self) {
        // Reset the audio cache state
        self.audio_processed = false;
        self.cached_audio_embeds = None;
        self.cached_audio_positions = None;
        // Note: LlamaCache reset needs to be handled at a higher level
        // as it requires device access
    }
}

/// Generates sinusoidal position embeddings for audio sequences
fn sinusoids(num_positions: usize, embedding_dim: usize, device: &Device) -> Result<Tensor> {
    let half_dim = embedding_dim / 2;
    let emb = -(10000_f64.ln()) / (half_dim - 1) as f64;
    let emb = (0..half_dim)
        .map(|i| (i as f64 * emb).exp())
        .collect::<Vec<_>>();
    let emb = Tensor::new(emb.as_slice(), device)?;

    let pos = Tensor::arange(0u32, num_positions as u32, device)?
        .to_dtype(DType::F32)?
        .unsqueeze(1)?;

    let emb = emb.unsqueeze(0)?;
    let phase = pos.broadcast_mul(&emb)?;

    let sin = phase.sin()?;
    let cos = phase.cos()?;

    Tensor::cat(&[sin, cos], 1)
}

/// Safely clamp tensor values for different dtypes
fn safe_clamp(x: &Tensor) -> Result<Tensor> {
    match x.dtype() {
        DType::F16 => {
            let max_val = 65504.0; // f16::MAX with safety margin
            x.clamp(-max_val, max_val)
        }
        DType::BF16 => {
            // BF16 has larger range, typically doesn't need clamping
            Ok(x.clone())
        }
        _ => Ok(x.clone()),
    }
}

/// Replace audio tokens in embeddings with projected audio features
pub fn replace_audio_tokens(
    inputs_embeds: &Tensor,
    audio_embeds: &Tensor,
    audio_positions: &[(usize, usize)],
    device: &Device,
) -> Result<Tensor> {
    if audio_positions.is_empty() {
        return Ok(inputs_embeds.clone());
    }

    let (batch_size, seq_len, hidden_size) = inputs_embeds.dims3()?;
    let num_audio_tokens = audio_positions.len();

    // HF-style: audio_embeds shape is (total_audio_seq_len, hidden_size)
    let audio_embeds_dims = audio_embeds.dims2()?;
    let total_audio_embeds = audio_embeds_dims.0;

    debug!(
        "Audio replacement: {} audio tokens, {} audio embeddings",
        num_audio_tokens, total_audio_embeds
    );

    // HF-style: Use audio embeddings one-to-one with audio tokens
    // We should now have the right number of audio tokens in the input sequence
    let audio_embeds = if total_audio_embeds >= num_audio_tokens {
        // Take the first num_audio_tokens embeddings to match the audio tokens
        if num_audio_tokens == total_audio_embeds {
            audio_embeds.clone()
        } else {
            audio_embeds.i(0..num_audio_tokens)?
        }
    } else {
        candle::bail!(
            "Not enough audio embeddings: need {}, got {}. Input sequence should have {} audio tokens.",
            num_audio_tokens,
            total_audio_embeds,
            total_audio_embeds
        );
    };

    // Create result tensor starting with text embeddings
    let mut result = inputs_embeds.clone();

    // Replace audio tokens with audio embeddings
    // Since we don't have scatter operations, we'll do this manually
    for (idx, &(batch_idx, seq_idx)) in audio_positions.iter().enumerate() {
        if batch_idx >= batch_size || seq_idx >= seq_len {
            candle::bail!(
                "Invalid audio position: ({}, {}) for tensor shape ({}, {}, {})",
                batch_idx,
                seq_idx,
                batch_size,
                seq_len,
                hidden_size
            );
        }

        // Get the audio embedding for this position
        let audio_embed = audio_embeds.i(idx)?;

        // Create a mask for this specific position
        let mut position_mask = vec![0f32; batch_size * seq_len];
        position_mask[batch_idx * seq_len + seq_idx] = 1.0;
        let position_mask = Tensor::new(position_mask.as_slice(), device)?
            .reshape((batch_size, seq_len, 1))?
            .to_dtype(inputs_embeds.dtype())?;

        // Broadcast audio embedding to full tensor shape
        let audio_embed_broadcast = audio_embed.unsqueeze(0)?.unsqueeze(0)?.broadcast_as((
            batch_size,
            seq_len,
            hidden_size,
        ))?;

        // Update result: keep original where mask is 0, use audio where mask is 1
        let inverse_mask = (1.0 - &position_mask)?;
        result = (result.broadcast_mul(&inverse_mask)?
            + audio_embed_broadcast.broadcast_mul(&position_mask)?)?;
    }

    Ok(result)
}

/// Find positions of audio tokens in input sequences
pub fn find_audio_token_positions(
    input_ids: &Tensor,
    audio_token_id: usize,
) -> Result<Vec<(usize, usize)>> {
    // Handle both i64 and u32 token types by converting to i64 first if needed
    let input_ids = if input_ids.dtype() == candle::DType::U32 {
        input_ids.to_dtype(candle::DType::I64)?
    } else {
        input_ids.clone()
    };

    let input_ids = input_ids.to_vec2::<i64>()?;
    let mut positions = Vec::new();

    for (batch_idx, sequence) in input_ids.iter().enumerate() {
        for (seq_idx, &token_id) in sequence.iter().enumerate() {
            if token_id as usize == audio_token_id {
                positions.push((batch_idx, seq_idx));
            }
        }
    }

    Ok(positions)
}

#[derive(Debug, Clone)]
struct VoxtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    attention_dropout: Dropout,
}

impl VoxtralAttention {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = embed_dim / num_heads;

        if head_dim * num_heads != embed_dim {
            candle::bail!(
                "embed_dim must be divisible by num_heads ({} % {} != 0)",
                embed_dim,
                num_heads
            );
        }

        let scaling = (head_dim as f64).powf(-0.5);

        let q_proj = linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        let attention_dropout = Dropout::new(cfg.attention_dropout as f32);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
            attention_dropout,
        })
    }

    fn reshape_for_scores(&self, x: &Tensor, seq_len: usize, bsz: usize) -> Result<Tensor> {
        x.reshape((bsz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }
}

impl Module for VoxtralAttention {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (bsz, seq_len, _) = x.dims3()?;

        // Project and scale queries
        let q = (self.q_proj.forward(x)? * self.scaling)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape for multi-head attention
        let q = self.reshape_for_scores(&q, seq_len, bsz)?;
        let k = self.reshape_for_scores(&k, seq_len, bsz)?;
        let v = self.reshape_for_scores(&v, seq_len, bsz)?;

        // Compute attention scores
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;

        // Apply attention dropout (only during training)
        let attn_weights = self.attention_dropout.forward(&attn_weights, false)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            bsz,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

#[derive(Debug, Clone)]
struct VoxtralEncoderLayer {
    self_attn: VoxtralAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: candle_nn::Activation,
    dropout: Dropout,
    activation_dropout: Dropout,
}

impl VoxtralEncoderLayer {
    fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let embed_dim = cfg.hidden_size;

        let self_attn = VoxtralAttention::new(cfg, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(embed_dim, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = linear(cfg.intermediate_size, embed_dim, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("final_layer_norm"))?;

        let activation = match cfg.activation_function.as_str() {
            "gelu" => candle_nn::Activation::Gelu,
            "relu" => candle_nn::Activation::Relu,
            _ => candle::bail!(
                "Unsupported activation function: {}",
                cfg.activation_function
            ),
        };

        let dropout = Dropout::new(cfg.dropout as f32);
        let activation_dropout = Dropout::new(cfg.activation_dropout as f32);

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation,
            dropout,
            activation_dropout,
        })
    }

    pub fn get_fc1_out_dim(&self) -> usize {
        // Return the intermediate size from the config
        // Since Linear doesn't expose out_dim
        self.fc1.weight().dims()[0]
    }

    fn forward(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // Self-attention with residual connection
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = self.dropout.forward(&x, training)?;
        let x = (x + residual)?;

        // Feed-forward network with residual connection
        let residual = &x;
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?;
        let x = x.apply(&self.activation)?;
        let x = self.activation_dropout.forward(&x, training)?;
        let x = self.fc2.forward(&x)?;
        let x = self.dropout.forward(&x, training)?;
        let x = (x + residual)?;

        // Safe clamping for numerical stability
        safe_clamp(&x)
    }

    /// Forward and extract intermediate features after FC1 (for use in multimodal projector)
    fn forward_extract_intermediate(&self, x: &Tensor, training: bool) -> Result<Tensor> {
        // Self-attention with residual connection
        let residual = x;
        let x = self.self_attn_layer_norm.forward(x)?;
        let x = self.self_attn.forward(&x)?;
        let x = self.dropout.forward(&x, training)?;
        let x = (x + residual)?;

        // Feed-forward network - extract intermediate features after FC1
        let x = self.final_layer_norm.forward(&x)?;
        let x = self.fc1.forward(&x)?; // This outputs intermediate_size features
        let x = x.apply(&self.activation)?;

        // Return intermediate features without continuing through FC2
        Ok(x)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralEncoder {
    conv1: Conv1d,
    conv2: Conv1d,
    embed_positions: Tensor,
    layers: Vec<VoxtralEncoderLayer>,
    layer_norm: LayerNorm,
    embed_scale: f64,
    dropout: Dropout,
    layerdrop: f64,
    max_source_positions: usize,
}

impl VoxtralEncoder {
    pub fn new(cfg: &VoxtralEncoderConfig, vb: VarBuilder) -> Result<Self> {
        // Ensure Whisper compatibility
        let cfg = cfg.clone().with_whisper_compatibility();

        let embed_dim = cfg.hidden_size;

        let embed_scale = if cfg.scale_embedding {
            (embed_dim as f64).sqrt()
        } else {
            1.0
        };

        // Convolutional layers for processing mel features
        let conv1 = candle_nn::conv1d(
            cfg.num_mel_bins,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv1"),
        )?;

        let conv2 = candle_nn::conv1d(
            embed_dim,
            embed_dim,
            3,
            candle_nn::Conv1dConfig {
                stride: 2,
                padding: 1,
                ..Default::default()
            },
            vb.pp("conv2"),
        )?;

        // Position embeddings
        let embed_positions = vb.get(
            (cfg.max_source_positions, embed_dim),
            "embed_positions.weight",
        )?;

        // Transformer layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VoxtralEncoderLayer::new(
                &cfg,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let layer_norm = layer_norm(embed_dim, 1e-5, vb.pp("layer_norm"))?;
        let dropout = Dropout::new(cfg.dropout as f32);

        Ok(Self {
            conv1,
            conv2,
            embed_positions,
            layers,
            layer_norm,
            embed_scale,
            dropout,
            layerdrop: cfg.layerdrop,
            max_source_positions: cfg.max_source_positions,
        })
    }

    pub fn forward(&self, input_features: &Tensor) -> Result<Tensor> {
        self.forward_with_training(input_features, false)
    }

    pub fn forward_with_training(&self, input_features: &Tensor, training: bool) -> Result<Tensor> {
        // Get the expected dtype from the conv1 weights
        let expected_dtype = self.conv1.weight().dtype();

        // Convert input_features to match the model's dtype if needed
        let input_features = if input_features.dtype() != expected_dtype {
            input_features.to_dtype(expected_dtype)?
        } else {
            input_features.clone()
        };

        // Apply convolutional layers with GELU activation
        let x = self.conv1.forward(&input_features)?;
        let x = x.gelu()?;
        let x = self.conv2.forward(&x)?;
        let x = x.gelu()?;

        // Reshape: (batch, embed_dim, seq_len) -> (batch, seq_len, embed_dim)
        let x = x.transpose(1, 2)?;

        // Add position embeddings
        let seq_len = x.dim(1)?;
        let positions = self.embed_positions.i(..seq_len)?;
        let x = x.broadcast_add(&positions)?;

        // Apply dropout
        let mut x = self.dropout.forward(&x, training)?;

        // Apply transformer layers with optional layer dropout
        // All layers use normal forward pass to output hidden_size (1280)
        // The transformation to intermediate_size (5120) happens via reshape in get_audio_embeds
        for (idx, layer) in self.layers.iter().enumerate() {
            x = self.forward_layer_with_dropout(&x, layer, idx, training)?;
        }

        // Apply final layer normalization (critical for proper output values!)
        let x = self.layer_norm.forward(&x)?;

        Ok(x)
    }

    /// Forward the last layer and extract intermediate features from FC1
    fn forward_layer_extract_intermediate(
        &self,
        x: &Tensor,
        layer: &VoxtralEncoderLayer,
        training: bool,
    ) -> Result<Tensor> {
        // Use the layer's forward_extract_intermediate method
        layer.forward_extract_intermediate(x, training)
    }

    /// Forward a single layer with stochastic depth (layer dropout)
    fn forward_layer_with_dropout(
        &self,
        x: &Tensor,
        layer: &VoxtralEncoderLayer,
        _layer_idx: usize,
        training: bool,
    ) -> Result<Tensor> {
        if training && self.layerdrop > 0.0 {
            // Apply stochastic depth with proper randomization
            let mut rng = rand::rng();
            let keep_prob = 1.0 - self.layerdrop;
            let keep: bool = rng.random::<f64>() < keep_prob;

            if !keep {
                // Skip layer entirely (identity mapping)
                return Ok(x.clone());
            }
        }

        layer.forward(x, training)
    }

    /// Get the output dimension of the first FC layer (needed for projector)
    pub fn get_intermediate_size(&self) -> usize {
        if !self.layers.is_empty() {
            self.layers[0].get_fc1_out_dim()
        } else {
            // Fallback to config value
            5120 // Default intermediate size
        }
    }

    /// Process long audio sequences in chunks to save memory
    pub fn process_long_audio(
        &self,
        input_features: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let (_batch_size, _num_mel, seq_len) = input_features.dims3()?;

        if seq_len <= chunk_size {
            return self.forward(input_features);
        }

        let mut outputs = Vec::new();
        let step = chunk_size - overlap;

        for start in (0..seq_len).step_by(step) {
            let end = (start + chunk_size).min(seq_len);
            let chunk = input_features.i((.., .., start..end))?;

            // Process chunk
            let output = self.forward(&chunk)?;

            // Handle overlap by averaging
            if !outputs.is_empty() && overlap > 0 {
                let overlap_frames = overlap / 2; // Account for conv2 stride
                let last_output: &mut Tensor = outputs.last_mut().unwrap();
                let last_len = last_output.dim(1)?;

                // Average overlapping regions
                let overlap_start = last_len.saturating_sub(overlap_frames);
                let overlap_new = output.i((.., ..overlap_frames, ..))?;
                let overlap_old = last_output.i((.., overlap_start.., ..))?;
                let averaged = ((overlap_old + overlap_new)? * 0.5)?;

                // Update last output
                *last_output =
                    Tensor::cat(&[&last_output.i((.., ..overlap_start, ..))?, &averaged], 1)?;

                // Add non-overlapping part of current chunk
                outputs.push(output.i((.., overlap_frames.., ..))?);
            } else {
                outputs.push(output);
            }
        }

        // Concatenate all outputs
        let outputs_ref: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&outputs_ref, 1)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralMultiModalProjector {
    linear_1: Linear,
    linear_2: Linear,
    activation: candle_nn::Activation,
}

impl VoxtralMultiModalProjector {
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let linear_1 = linear_no_bias(
            cfg.audio_config.intermediate_size,
            cfg.text_config.hidden_size,
            vb.pp("linear_1"),
        )?;

        let linear_2 = linear_no_bias(
            cfg.text_config.hidden_size,
            cfg.text_config.hidden_size,
            vb.pp("linear_2"),
        )?;

        let activation = match cfg.projector_hidden_act.as_str() {
            "gelu" => candle_nn::Activation::Gelu,
            "relu" => candle_nn::Activation::Relu,
            _ => candle::bail!(
                "Unsupported projector activation: {}",
                cfg.projector_hidden_act
            ),
        };

        Ok(Self {
            linear_1,
            linear_2,
            activation,
        })
    }

    pub fn forward(&self, audio_features: &Tensor) -> Result<Tensor> {
        let x = self.linear_1.forward(audio_features)?;
        let x = x.apply(&self.activation)?;
        self.linear_2.forward(&x)
    }
}

#[derive(Debug, Clone)]
pub struct VoxtralForConditionalGeneration {
    audio_tower: VoxtralEncoder,
    language_model: VoxtralLlama,
    multi_modal_projector: VoxtralMultiModalProjector,
    audio_token_id: usize,
    audio_config: VoxtralEncoderConfig,
    text_config: LlamaConfig,
}

impl VoxtralForConditionalGeneration {
    pub fn new(cfg: &VoxtralConfig, vb: VarBuilder) -> Result<Self> {
        let audio_tower = VoxtralEncoder::new(&cfg.audio_config, vb.pp("audio_tower"))?;
        debug!("audio_tower created");
        let language_model = VoxtralLlama::load(vb.pp("language_model"), &cfg.text_config)?;
        debug!("language_model created");
        let multi_modal_projector =
            VoxtralMultiModalProjector::new(cfg, vb.pp("multi_modal_projector"))?;

        Ok(Self {
            audio_tower,
            language_model,
            multi_modal_projector,
            audio_token_id: cfg.audio_token_id,
            audio_config: cfg.audio_config.clone(),
            text_config: cfg.text_config.clone(),
        })
    }

    /// Get the audio token ID used for this model
    pub fn audio_token_id(&self) -> usize {
        self.audio_token_id
    }

    /// Get the text model configuration
    pub fn text_config(&self) -> &LlamaConfig {
        &self.text_config
    }

    /// Get the audio encoder configuration
    pub fn audio_config(&self) -> &VoxtralEncoderConfig {
        &self.audio_config
    }

    /// Process audio features through encoder and projector
    pub fn get_audio_embeds(&self, input_features: &Tensor) -> Result<Tensor> {
        debug!("=== RUST get_audio_embeds DEBUG ===");
        debug!("Input audio features shape: {:?}", input_features.dims());

        let audio_outputs = self.audio_tower.forward(input_features)?;
        debug!("Audio tower output shape: {:?}", audio_outputs.dims());

        // Compute statistics for audio tower output - handle F16
        let audio_stats = audio_outputs.mean_all()?;
        let audio_mean = match audio_outputs.dtype() {
            DType::F16 => audio_stats.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => audio_stats.to_scalar::<f32>()?,
        };
        let audio_outputs_sq = audio_outputs.powf(2.0)?;
        let audio_stats_sq = audio_outputs_sq.mean_all()?;
        let audio_mean_sq = match audio_outputs.dtype() {
            DType::F16 => audio_stats_sq.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => audio_stats_sq.to_scalar::<f32>()?,
        };
        let variance = audio_mean_sq - audio_mean * audio_mean;
        let audio_std = if variance >= 0.0 {
            variance.sqrt()
        } else {
            0.0
        };
        debug!(
            "Audio tower output stats: mean={:.6}, std={:.6}, variance={:.6}",
            audio_mean, audio_std, variance
        );

        // Show some sample values to debug
        if let Ok(flat) = audio_outputs.flatten(0, 2) {
            if let Ok(slice) = flat.narrow(0, 0, 10) {
                if let Ok(f32_slice) = slice.to_dtype(DType::F32) {
                    if let Ok(vals) = f32_slice.to_vec1::<f32>() {
                        debug!("First 10 audio tower values: {:?}", vals);
                    }
                }
            }
        }

        // Following HF implementation: reshape to (-1, config.intermediate_size) before projection
        // Python: audio_hidden_states.reshape(-1, self.config.audio_config.intermediate_size)
        // This transforms [1, 1500, 1280] -> [375, 5120] using intermediate_size from config
        let (batch_size, seq_len, hidden_size) = audio_outputs.dims3()?;
        debug!(
            "Audio encoder outputs hidden_size: {}, config intermediate_size: {}",
            hidden_size, self.audio_config.intermediate_size
        );

        // The key insight: Python reshapes from [1, 1500, 1280] to [375, 5120]
        // This means 1500 * 1280 = 375 * 5120 (1920000 elements)
        // So we need: new_batch_size = (batch_size * seq_len * hidden_size) / intermediate_size
        let total_elements = batch_size * seq_len * hidden_size;
        let new_batch_size = total_elements / self.audio_config.intermediate_size;

        debug!(
            "Reshaping from [{}, {}, {}] to [{}, {}]",
            batch_size, seq_len, hidden_size, new_batch_size, self.audio_config.intermediate_size
        );

        // Verify the division is exact
        if total_elements % self.audio_config.intermediate_size != 0 {
            return Err(candle::Error::DimOutOfRange {
                shape: candle::Shape::from_dims(&[batch_size, seq_len, hidden_size]),
                dim: 0,
                op: "reshape",
            }
            .into());
        }

        let audio_hidden =
            audio_outputs.reshape((new_batch_size, self.audio_config.intermediate_size))?;
        debug!(
            "Audio hidden reshaped to match HF: {:?}",
            audio_hidden.dims()
        );

        // Compute statistics after reshape - handle F16
        let reshape_stats = audio_hidden.mean_all()?;
        let reshape_mean = match audio_hidden.dtype() {
            DType::F16 => reshape_stats.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => reshape_stats.to_scalar::<f32>()?,
        };
        let audio_hidden_sq = audio_hidden.powf(2.0)?;
        let reshape_stats_sq = audio_hidden_sq.mean_all()?;
        let reshape_mean_sq = match audio_hidden.dtype() {
            DType::F16 => reshape_stats_sq.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => reshape_stats_sq.to_scalar::<f32>()?,
        };
        let reshape_std = (reshape_mean_sq - reshape_mean * reshape_mean).sqrt();
        debug!(
            "Reshape output stats: mean={:.6}, std={:.6}",
            reshape_mean, reshape_std
        );

        // Project to text space - this gives us embeddings for each audio position
        let projected = self.multi_modal_projector.forward(&audio_hidden)?;
        debug!("Projector output shape: {:?}", projected.dims());

        // Compute statistics for projector output - handle F16
        let proj_stats = projected.mean_all()?;
        let proj_mean = match projected.dtype() {
            DType::F16 => proj_stats.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => proj_stats.to_scalar::<f32>()?,
        };
        let projected_sq = projected.powf(2.0)?;
        let proj_stats_sq = projected_sq.mean_all()?;
        let proj_mean_sq = match projected.dtype() {
            DType::F16 => proj_stats_sq.to_dtype(DType::F32)?.to_scalar::<f32>()?,
            _ => proj_stats_sq.to_scalar::<f32>()?,
        };
        let proj_std = (proj_mean_sq - proj_mean * proj_mean).sqrt();
        debug!(
            "Projector output stats: mean={:.6}, std={:.6}",
            proj_mean, proj_std
        );

        // HF does NOT pool - it keeps all audio token embeddings
        // Return the projected embeddings directly for replacement
        // Shape: (batch_size * seq_len, text_hidden_size)

        // Check if we have reasonable values (not all zeros/NaN)
        match projected.flatten_all() {
            Ok(flattened) => {
                // Convert to f32 first if needed
                let flattened_f32 = if flattened.dtype() != candle::DType::F32 {
                    flattened.to_dtype(candle::DType::F32)?
                } else {
                    flattened
                };

                match flattened_f32.to_vec1::<f32>() {
                    Ok(values) => {
                        let non_zero_count = values.iter().filter(|&&x| x.abs() > 1e-6).count();
                        let has_nan = values.iter().any(|&x| x.is_nan());
                        let has_inf = values.iter().any(|&x| x.is_infinite());
                        let min_val = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
                        let max_val = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

                        debug!("Audio embeddings stats - non-zero: {}/{}, NaN: {}, Inf: {}, min: {:.6}, max: {:.6}",
                               non_zero_count, values.len(), has_nan, has_inf, min_val, max_val);

                        // Show first few values for debugging
                        if values.len() > 0 {
                            let sample_count = values.len().min(10);
                            debug!(
                                "First {} embedding values: {:?}",
                                sample_count,
                                &values[..sample_count]
                            );
                        }

                        if non_zero_count == 0 {
                            debug!("WARNING: All audio embeddings are zero!");
                        }
                        if has_nan || has_inf {
                            debug!("WARNING: Audio embeddings contain NaN or Inf values!");
                        }
                    }
                    Err(e) => debug!("Failed to convert audio embeddings to vec: {}", e),
                }
            }
            Err(e) => debug!("Failed to flatten audio embeddings: {}", e),
        }

        // Return shape: (batch_size * seq_len, text_hidden_size)
        // This matches HF implementation - no pooling, keep all audio token embeddings
        Ok(projected)
    }

    /// Process long audio sequences efficiently
    pub fn get_audio_embeds_chunked(
        &self,
        input_features: &Tensor,
        chunk_size: usize,
        overlap: usize,
    ) -> Result<Tensor> {
        let audio_outputs =
            self.audio_tower
                .process_long_audio(input_features, chunk_size, overlap)?;

        // Reshape and project (now outputs hidden_size, needs reshape to intermediate_size)
        let (batch_size, seq_len, hidden_size) = audio_outputs.dims3()?;
        // Apply same reshape logic as get_audio_embeds
        let total_elements = batch_size * seq_len * hidden_size;
        let new_batch_size = total_elements / self.audio_config.intermediate_size;
        let audio_hidden =
            audio_outputs.reshape((new_batch_size, self.audio_config.intermediate_size))?;

        let projected = self.multi_modal_projector.forward(&audio_hidden)?;

        // Reshape back to (batch_size, seq_len, text_hidden_size) for pooling
        let text_hidden_size = self.text_config.hidden_size;
        let projected = projected.reshape((batch_size, seq_len, text_hidden_size))?;

        // Apply mean pooling to reduce to single audio embedding per batch
        let pooled = projected.mean(1)?; // Mean across sequence dimension

        // Return shape: (batch_size, text_hidden_size)
        Ok(pooled)
    }

    /// Forward pass with audio features and text input
    pub fn forward(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        cache: &mut VoxtralCache,
        index_pos: usize,
    ) -> Result<Tensor> {
        // Get text embeddings
        let mut inputs_embeds = self.language_model.embed(input_ids)?;

        // If audio features are provided and not yet processed
        if let Some(features) = input_features {
            if !cache.audio_processed {
                let audio_embeds = self.get_audio_embeds(features)?;
                debug!("Audio embeds shape: {:?}", audio_embeds.dims());

                let audio_positions = find_audio_token_positions(input_ids, self.audio_token_id)?;
                debug!(
                    "Found {} audio token positions: {:?}",
                    audio_positions.len(),
                    audio_positions
                );

                // Cache for future use
                cache.cached_audio_embeds = Some(audio_embeds.clone());
                cache.cached_audio_positions = Some(audio_positions.clone());
                cache.audio_processed = true;

                // Replace audio tokens with audio embeddings
                debug!("Replacing audio tokens");
                inputs_embeds = replace_audio_tokens(
                    &inputs_embeds,
                    &audio_embeds,
                    &audio_positions,
                    input_ids.device(),
                )?;
            } else {
                debug!("Audio already processed, using cached embeddings");
            }
        }

        // Forward through language model using forward_input_embed
        self.language_model
            .forward_input_embed(&inputs_embeds, index_pos, &mut cache.llama_cache)
    }

    /// Generate text given audio input
    pub fn generate(
        &self,
        input_ids: &Tensor,
        input_features: Option<&Tensor>,
        max_new_tokens: usize,
        temperature: f64,
        top_p: Option<f64>,
        device: &Device,
        cache: Option<VoxtralCache>,
        ignore_eos: Option<bool>,
    ) -> Result<Vec<u32>> {
        debug!("Input_ids: {input_ids}");
        debug!("Input_features: {input_features:?}");
        debug!("Device: {device:?}");
        debug!("Cache: {cache:?}");
        debug!("Max_new_tokens: {max_new_tokens}");
        debug!("Temperature: {temperature}");
        debug!("Top_p: {top_p:?}");

        // Validate inputs
        if max_new_tokens == 0 {
            return Ok(input_ids.i(0)?.to_vec1::<u32>()?); // Get first batch
        }

        if temperature < 0.0 {
            candle::bail!("Temperature must be non-negative, got {}", temperature);
        }

        if let Some(p) = top_p {
            if !(0.0..=1.0).contains(&p) {
                candle::bail!("top_p must be between 0 and 1, got {}", p);
            }
        }

        let mut final_cache = if let Some(cache) = cache {
            cache
        } else {
            // Get the dtype from the language model by creating a small embedding
            let dummy_token = Tensor::new(&[1u32], device)?;
            let dummy_embed = self.language_model.embed(&dummy_token)?;
            let model_dtype = dummy_embed.dtype();
            VoxtralCache::new(true, model_dtype, &self.text_config, device)?
        };
        let mut tokens = input_ids.i(0)?.to_vec1::<u32>()?; // Get first batch
        let initial_len = tokens.len();

        for idx in 0..max_new_tokens {
            let (input, index_pos) = if idx == 0 {
                (input_ids.clone(), 0)
            } else {
                // For subsequent generation steps, use only the last token
                let last_token = tokens[tokens.len() - 1];
                (
                    Tensor::new(&[last_token], device)?.unsqueeze(0)?,
                    initial_len + idx - 1,
                )
            };

            let logits = if idx == 0 {
                // First pass - include audio features
                match self.forward(&input, input_features, &mut final_cache, index_pos) {
                    Ok(logits) => logits,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!(
                            "Failed to generate tokens: {}",
                            e
                        )));
                    }
                }
            } else {
                // Subsequent passes - text only
                match self.forward(&input, None, &mut final_cache, index_pos) {
                    Ok(logits) => logits,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!(
                            "Failed to generate tokens: {}",
                            e
                        )));
                    }
                }
            };

            // Handle both 2D [batch, vocab] and 3D [batch, seq_len, vocab] logits
            let logits = if logits.dims().len() == 3 {
                // 3D case: [batch, seq_len, vocab] -> get last token
                logits.i((.., logits.dim(1)? - 1, ..))?
            } else {
                // 2D case: [batch, vocab] -> already the right shape
                logits
            };

            let next_token = if temperature > 0.0 {
                // Sample with temperature
                let prs = (logits / temperature)?;
                let prs = candle_nn::ops::softmax_last_dim(&prs)?;

                if let Some(top_p_val) = top_p {
                    // Apply top-p sampling
                    sample_top_p(&prs.squeeze(0)?, top_p_val, device)?
                } else {
                    // Sample from full distribution
                    let probs_vec = prs.squeeze(0)?.to_vec1::<f32>()?;
                    let mut rng = rand::rng();
                    let mut cumsum = 0.0;
                    let rand_val: f32 = rng.random();
                    let mut sampled = 0u32;

                    for (idx, &prob) in probs_vec.iter().enumerate() {
                        cumsum += prob;
                        if cumsum > rand_val {
                            sampled = idx as u32;
                            break;
                        }
                    }
                    sampled
                }
            } else {
                // Debug logits for first token (compare with Python)
                if idx == 0 {
                    debug!("=== RUST FIRST TOKEN LOGITS DEBUG ===");
                    if let Ok(logits_flat) = logits.squeeze(0) {
                        // Check specific tokens that Python reported
                        let token_it = 1276u32;
                        let token_quote = 87125u32;

                        if let (Ok(logit_it), Ok(logit_quote)) = (
                            logits_flat.i(token_it as usize),
                            logits_flat.i(token_quote as usize),
                        ) {
                            if let (Ok(val_it), Ok(val_quote)) =
                                (logit_it.to_scalar::<f32>(), logit_quote.to_scalar::<f32>())
                            {
                                debug!("Token {} ('it'): {:.6}", token_it, val_it);
                                debug!("Token {} (\"'\"): {:.6}", token_quote, val_quote);
                                debug!("Difference: {:.6}", val_it - val_quote);
                            }
                        }

                        // Find top 10 tokens to compare with Python
                        if let Ok(logits_vec) = logits_flat.to_vec1::<f32>() {
                            let mut indexed_logits: Vec<(usize, f32)> = logits_vec
                                .iter()
                                .enumerate()
                                .map(|(i, &val)| (i, val))
                                .collect();
                            indexed_logits.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                            debug!("Top 10 Rust candidates:");
                            for (rank, (token_idx, logit_val)) in
                                indexed_logits.iter().take(10).enumerate()
                            {
                                debug!(
                                    "  {:2}: Token {:6} = logit {:.4}",
                                    rank + 1,
                                    token_idx,
                                    logit_val
                                );
                            }
                        }
                    }
                }

                // Greedy decoding - find the token with highest probability
                let argmax_result = match logits.argmax(D::Minus1) {
                    Ok(result) => result,
                    Err(e) => {
                        return Err(candle::Error::Msg(format!("Argmax failed: {}", e)));
                    }
                };

                // Handle the case where argmax returns [1] instead of scalar
                let token = if argmax_result.dims().len() == 0 {
                    // Already a scalar
                    match argmax_result.to_scalar::<u32>() {
                        Ok(token) => token,
                        Err(e) => {
                            return Err(candle::Error::Msg(format!("to_scalar failed: {}", e)));
                        }
                    }
                } else if argmax_result.dims() == &[1] {
                    // Shape [1] - extract the single element
                    match argmax_result.i(0) {
                        Ok(scalar_tensor) => match scalar_tensor.to_scalar::<u32>() {
                            Ok(token) => token,
                            Err(e) => {
                                return Err(candle::Error::Msg(format!(
                                    "to_scalar on extracted element failed: {}",
                                    e
                                )));
                            }
                        },
                        Err(e) => {
                            return Err(candle::Error::Msg(format!(
                                "indexing argmax result failed: {}",
                                e
                            )));
                        }
                    }
                } else {
                    return Err(candle::Error::Msg(format!(
                        "Unexpected argmax result shape: {:?}",
                        argmax_result.shape()
                    )));
                };
                token
            };

            tokens.push(next_token);

            // Debug logging for first few tokens
            if idx < 10 {
                debug!(
                    "Step {}: Generated token {} (total tokens: {})",
                    idx,
                    next_token,
                    tokens.len()
                );
            }

            // Check for EOS tokens - Voxtral uses different EOS tokens than hardcoded 2
            // Based on the Mistral/Voxtral tokenizer, common EOS tokens are:
            // 2 = </s>, 0 = <pad>, 128001, 128009 from various chat formats
            let eos_tokens = [2u32, 128001, 128009, 128256]; // Don't include 0 as it might be valid generation

            // Check for EOS tokens only if not ignoring them
            if !ignore_eos.unwrap_or(false) && eos_tokens.contains(&next_token) {
                debug!("Hit EOS token: {}", next_token);
                break;
            } else if ignore_eos.unwrap_or(false) && eos_tokens.contains(&next_token) {
                debug!("Ignoring EOS token: {} (ignore_eos=true)", next_token);
            }

            // Also break if we get repeated pad tokens (might indicate the model is stuck)
            if next_token == 0 && tokens.len() > 5 {
                let last_5_tokens = &tokens[tokens.len() - 5..];
                if last_5_tokens.iter().all(|&t| t == 0) {
                    debug!("Breaking due to repeated pad tokens");
                    break;
                }
            }
        }

        debug!("Generated tokens: {:?}", tokens);

        Ok(tokens)
    }
}

/// Sample from top-p probability distribution
fn sample_top_p(probs: &Tensor, top_p: f64, _device: &Device) -> Result<u32> {
    let (sorted_probs, sorted_indices) = probs.sort_last_dim(false)?;
    let cumsum = sorted_probs.cumsum(D::Minus1)?;
    let mask = cumsum.le(top_p)?;

    // Apply mask and renormalize
    let filtered_probs = sorted_probs.where_cond(&mask, &Tensor::zeros_like(&sorted_probs)?)?;
    let filtered_probs = (&filtered_probs / filtered_probs.sum_keepdim(D::Minus1)?)?;

    // Sample from filtered distribution
    // Since multinomial is not available, we'll use a simple sampling approach
    let probs_vec = filtered_probs.to_vec1::<f32>()?;
    let mut cumsum = 0.0;
    let mut rng = rand::rng();
    let rand_val: f32 = rng.random();
    let mut sample_idx = 0;

    for (idx, &prob) in probs_vec.iter().enumerate() {
        cumsum += prob;
        if cumsum > rand_val {
            sample_idx = idx;
            break;
        }
    }

    sorted_indices.i(sample_idx)?.to_scalar::<u32>()
}
