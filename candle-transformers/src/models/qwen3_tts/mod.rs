//! Qwen3-TTS: text-to-speech from the Qwen team.
//!
//! See [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) and the models on the hub, e.g.
//! `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`.
//!
//! The model generates 12.5 Hz frames of 16 codec tokens that the speech tokenizer decoder in
//! [`speech_tokenizer`] turns into 24 kHz audio. Each frame is produced in two steps:
//!
//! - the *talker*, a Qwen3 transformer fed with summed text and codec embeddings, predicts the
//!   first codebook entry of the next frame,
//! - the *code predictor*, a small Qwen3 transformer conditioned on the talker hidden state,
//!   autoregressively predicts the 15 remaining codebook entries of that frame.
//!
//! Three flavors of checkpoints exist, all handled by [`Model`]:
//!
//! - `CustomVoice`: a set of predefined speakers, see [`Model::supported_speakers`], optionally
//!   steered with a natural language instruction (1.7B only).
//! - `VoiceDesign`: the voice is described by a natural language instruction.
//! - `Base`: voice cloning from a speaker embedding and/or a reference transcript + codes.
//!
//! Text has to be tokenized with the Qwen2/Qwen3 tokenizer and wrapped in the chat template used
//! by the reference implementation, see [`Prompt`].

pub mod speaker_encoder;
pub mod speech_tokenizer;
pub mod transformer;

use crate::generation::{LogitsProcessor, Sampling};
use crate::models::with_tracing::{linear, linear_no_bias, Linear};
use candle::{DType, Device, IndexOp, Module, Result, Tensor};
use candle_nn::{embedding, Activation, Embedding, VarBuilder};
use std::collections::HashMap;
use transformer::{Transformer, TransformerConfig};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct CodePredictorConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
}

impl CodePredictorConfig {
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
            // The code predictor only ever sees `num_code_groups + 1` positions.
            max_position_embeddings: self.num_code_groups + 1,
            hidden_act: self.hidden_act,
            attention_bias: self.attention_bias,
            qk_norm: true,
            layer_scale: false,
            sliding_window: None,
        }
    }
}

/// Value of the `spk_is_dialect` entries: either `false` or the name of a dialect that
/// overrides the language tag when the language is Chinese (or auto).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub enum Dialect {
    Flag(bool),
    Name(String),
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TalkerConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub text_hidden_size: usize,
    pub text_vocab_size: usize,
    pub num_code_groups: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_bias: bool,
    pub code_predictor_config: CodePredictorConfig,
    pub codec_bos_id: u32,
    pub codec_eos_token_id: u32,
    pub codec_pad_id: u32,
    pub codec_think_id: u32,
    pub codec_nothink_id: u32,
    pub codec_think_bos_id: u32,
    pub codec_think_eos_id: u32,
    #[serde(default)]
    pub codec_language_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_id: HashMap<String, u32>,
    #[serde(default)]
    pub spk_is_dialect: HashMap<String, Dialect>,
}

impl TalkerConfig {
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
            qk_norm: true,
            layer_scale: false,
            sliding_window: None,
        }
    }
}

/// The `config.json` of a Qwen3-TTS checkpoint.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub talker_config: TalkerConfig,
    pub tts_bos_token_id: u32,
    pub tts_eos_token_id: u32,
    pub tts_pad_token_id: u32,
    pub im_start_token_id: u32,
    pub im_end_token_id: u32,
    pub assistant_token_id: u32,
    /// `base`, `custom_voice` or `voice_design`.
    #[serde(default)]
    pub tts_model_type: String,
    #[serde(default)]
    pub tts_model_size: String,
    /// Configuration of the speaker encoder of the Base models.
    #[serde(default)]
    pub speaker_encoder_config: Option<speaker_encoder::Config>,
}

/// Two-layer MLP projecting text embeddings to the talker dimension.
#[derive(Debug, Clone)]
struct ResizeMlp {
    linear_fc1: Linear,
    linear_fc2: Linear,
    act: Activation,
}

impl ResizeMlp {
    fn new(
        input_size: usize,
        intermediate_size: usize,
        output_size: usize,
        act: Activation,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            linear_fc1: linear(input_size, intermediate_size, vb.pp("linear_fc1"))?,
            linear_fc2: linear(intermediate_size, output_size, vb.pp("linear_fc2"))?,
            act,
        })
    }
}

impl Module for ResizeMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        xs.apply(&self.linear_fc1)?
            .apply(&self.act)?
            .apply(&self.linear_fc2)
    }
}

/// Predicts the codebooks `1..num_code_groups` of a frame from the talker hidden state and the
/// embedding of the first codebook entry.
#[derive(Debug, Clone)]
struct CodePredictor {
    model: Transformer,
    /// One embedding table per predicted codebook, in the talker dimension.
    codec_embeddings: Vec<Embedding>,
    lm_heads: Vec<Linear>,
    /// Maps talker-sized embeddings to the code predictor dimension when they differ.
    projection: Option<Linear>,
}

impl CodePredictor {
    fn new(cfg: &CodePredictorConfig, talker_hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let model = Transformer::new(&cfg.transformer_config(), vb.pp("model"))?;
        let n = cfg.num_code_groups - 1;
        let mut codec_embeddings = Vec::with_capacity(n);
        let mut lm_heads = Vec::with_capacity(n);
        let vb_e = vb.pp("model.codec_embedding");
        let vb_h = vb.pp("lm_head");
        for i in 0..n {
            codec_embeddings.push(embedding(cfg.vocab_size, talker_hidden_size, vb_e.pp(i))?);
            lm_heads.push(linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb_h.pp(i))?);
        }
        let projection = if cfg.hidden_size != talker_hidden_size {
            Some(linear(
                talker_hidden_size,
                cfg.hidden_size,
                vb.pp("small_to_mtp_projection"),
            )?)
        } else {
            None
        };
        Ok(Self {
            model,
            codec_embeddings,
            lm_heads,
            projection,
        })
    }

    fn project(&self, xs: &Tensor) -> Result<Tensor> {
        match &self.projection {
            Some(p) => xs.apply(p),
            None => Ok(xs.clone()),
        }
    }

    /// Embedding of `code` in codebook `group` (1-based, `group >= 1`).
    fn embed(&self, group: usize, code: u32) -> Result<Tensor> {
        let ids = Tensor::new(&[[code]], self.model.device())?;
        self.codec_embeddings[group - 1].forward(&ids)
    }

    /// `talker_hidden` and `code0_embed` both have shape (1, 1, talker_hidden_size).
    fn predict(
        &mut self,
        talker_hidden: &Tensor,
        code0_embed: &Tensor,
        lp: &mut LogitsProcessor,
    ) -> Result<Vec<u32>> {
        self.model.clear_kv_cache();
        let xs = Tensor::cat(&[talker_hidden, code0_embed], 1)?;
        let mut hidden = self.model.forward(&self.project(&xs)?, 0)?;
        let mut offset = xs.dim(1)?;
        let n = self.lm_heads.len();
        let mut codes = Vec::with_capacity(n);
        for i in 0..n {
            let last = hidden.narrow(1, hidden.dim(1)? - 1, 1)?;
            let logits = last.apply(&self.lm_heads[i])?.i((0, 0))?;
            let code = lp.sample(&logits)?;
            codes.push(code);
            if i + 1 < n {
                let emb = self.embed(i + 1, code)?;
                hidden = self.model.forward(&self.project(&emb)?, offset)?;
                offset += 1;
            }
        }
        Ok(codes)
    }
}

#[derive(Debug, Clone)]
struct Talker {
    text_embedding: Embedding,
    codec_embedding: Embedding,
    text_projection: ResizeMlp,
    model: Transformer,
    codec_head: Linear,
    code_predictor: CodePredictor,
}

impl Talker {
    fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = embedding(
            cfg.text_vocab_size,
            cfg.text_hidden_size,
            vb.pp("model.text_embedding"),
        )?;
        let codec_embedding = embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("model.codec_embedding"),
        )?;
        let text_projection = ResizeMlp::new(
            cfg.text_hidden_size,
            cfg.text_hidden_size,
            cfg.hidden_size,
            cfg.hidden_act,
            vb.pp("text_projection"),
        )?;
        let model = Transformer::new(&cfg.transformer_config(), vb.pp("model"))?;
        let codec_head = linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("codec_head"))?;
        let code_predictor = CodePredictor::new(
            &cfg.code_predictor_config,
            cfg.hidden_size,
            vb.pp("code_predictor"),
        )?;
        Ok(Self {
            text_embedding,
            codec_embedding,
            text_projection,
            model,
            codec_head,
            code_predictor,
        })
    }
}

/// Speaker conditioning of a prompt.
#[derive(Debug, Clone, Copy)]
pub enum Voice<'a> {
    /// No speaker conditioning: VoiceDesign models, or Base models without cloning.
    None,
    /// One of the predefined speakers of a CustomVoice model, see [`Model::supported_speakers`].
    Speaker(&'a str),
    /// A speaker embedding of shape `(hidden_size,)` for the Base models.
    Embedding(&'a Tensor),
}

/// In-context voice cloning reference for the Base models.
#[derive(Debug, Clone, Copy)]
pub struct IclReference<'a> {
    /// Tokens of `<|im_start|>assistant\n{reference transcript}<|im_end|>\n`.
    pub ref_ids: &'a [u32],
    /// Codes of the reference audio, shape `(frames, num_code_groups)`, as produced by the
    /// speech tokenizer encoder.
    pub ref_codes: &'a Tensor,
}

/// Inputs of one text-to-speech request.
///
/// The token sequences follow the chat template of the reference implementation and must be
/// produced by the Qwen tokenizer without any extra special tokens:
///
/// - `input_ids`: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`,
/// - `instruct_ids`: `<|im_start|>user\n{instruction}<|im_end|>\n`.
#[derive(Debug, Clone, Copy)]
pub struct Prompt<'a> {
    pub input_ids: &'a [u32],
    pub instruct_ids: Option<&'a [u32]>,
    /// One of [`Model::supported_languages`], `None` or `"auto"` for automatic detection.
    pub language: Option<&'a str>,
    pub voice: Voice<'a>,
    pub icl: Option<IclReference<'a>>,
    /// With `non_streaming_mode` the whole text is part of the prefix. Otherwise only its
    /// first token is and the rest is fed one token per generated frame, which is how the
    /// reference implementation runs voice cloning.
    pub non_streaming_mode: bool,
}

#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Maximum number of codec frames to generate, 12.5 of them per second of audio. The
    /// default matches the `generation_config.json` of the checkpoints.
    pub max_new_tokens: usize,
    /// Sampling of the first codebook by the talker.
    pub sampling: Sampling,
    /// Sampling of the other codebooks by the code predictor.
    pub subtalker_sampling: Sampling,
    /// Penalty applied to the first-codebook tokens already generated, 1 disables it.
    pub repetition_penalty: f32,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            sampling: Sampling::TopKThenTopP {
                k: 50,
                p: 1.0,
                temperature: 0.9,
            },
            subtalker_sampling: Sampling::TopKThenTopP {
                k: 50,
                p: 1.0,
                temperature: 0.9,
            },
            repetition_penalty: 1.05,
            seed: 299792458,
        }
    }
}

/// Number of frames the end-of-speech token is suppressed for, as in the reference
/// implementation (`min_new_tokens`).
const MIN_NEW_TOKENS: usize = 2;

/// Number of special (non acoustic) entries at the end of the talker codec vocabulary.
const NUM_SPECIAL_CODEC_TOKENS: usize = 1024;

#[derive(Debug, Clone)]
pub struct Model {
    talker: Talker,
    speaker_encoder: Option<speaker_encoder::Model>,
    /// Additive mask suppressing every special codec token but the end of speech one.
    codec_mask: Tensor,
    /// The same mask, also suppressing the end of speech token.
    codec_mask_no_eos: Tensor,
    config: Config,
    device: Device,
    dtype: DType,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let talker = Talker::new(&cfg.talker_config, vb.pp("talker"))?;
        // Only the Base checkpoints ship the speaker encoder.
        let speaker_encoder = match &cfg.speaker_encoder_config {
            Some(sc) if vb.contains_tensor("speaker_encoder.fc.weight") => {
                Some(speaker_encoder::Model::new(sc, vb.pp("speaker_encoder"))?)
            }
            _ => None,
        };
        let tc = &cfg.talker_config;
        let eos = tc.codec_eos_token_id as usize;
        if tc.vocab_size <= NUM_SPECIAL_CODEC_TOKENS || eos >= tc.vocab_size {
            candle::bail!(
                "unexpected talker vocab size {} for eos token {eos}",
                tc.vocab_size
            )
        }
        // Only the acoustic codes and the end of speech token can be sampled, the mask is
        // added to the logits rather than applied on the host at every step.
        let mut mask = vec![0f32; tc.vocab_size];
        for (i, m) in mask
            .iter_mut()
            .enumerate()
            .skip(tc.vocab_size - NUM_SPECIAL_CODEC_TOKENS)
        {
            if i != eos {
                *m = f32::NEG_INFINITY
            }
        }
        let mut mask_no_eos = mask.clone();
        mask_no_eos[eos] = f32::NEG_INFINITY;
        let device = vb.device().clone();
        let codec_mask = Tensor::from_vec(mask, tc.vocab_size, &device)?;
        let codec_mask_no_eos = Tensor::from_vec(mask_no_eos, tc.vocab_size, &device)?;
        Ok(Self {
            talker,
            speaker_encoder,
            codec_mask,
            codec_mask_no_eos,
            config: cfg.clone(),
            device,
            dtype: vb.dtype(),
        })
    }

    /// Whether the checkpoint supports voice cloning through [`Model::speaker_embedding`].
    pub fn has_speaker_encoder(&self) -> bool {
        self.speaker_encoder.is_some()
    }

    /// Speaker embedding of a mono recording sampled at 24 kHz, to use with
    /// [`Voice::Embedding`]. Returns a tensor of shape (hidden_size,).
    pub fn speaker_embedding(&self, samples: &[f32]) -> Result<Tensor> {
        let (encoder, cfg) = match (&self.speaker_encoder, &self.config.speaker_encoder_config) {
            (Some(e), Some(c)) => (e, c),
            _ => candle::bail!("this checkpoint has no speaker encoder, use a Base model"),
        };
        let mel_cfg = speaker_encoder::MelConfig {
            sample_rate: cfg.sample_rate,
            ..Default::default()
        };
        let mels = speaker_encoder::mel_spectrogram(samples, &mel_cfg, &self.device)?
            .to_dtype(self.dtype)?;
        encoder.forward(&mels)?.squeeze(0)
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Names of the predefined speakers, empty for models without any.
    pub fn supported_speakers(&self) -> Vec<&str> {
        let mut v: Vec<_> = self
            .config
            .talker_config
            .spk_id
            .keys()
            .map(|s| s.as_str())
            .collect();
        v.sort_unstable();
        v
    }

    /// Names of the languages accepted by [`Prompt::language`], `"auto"` included. The
    /// dialects are not listed, they are selected through the speaker as in the reference
    /// implementation.
    pub fn supported_languages(&self) -> Vec<&str> {
        let mut v: Vec<_> = self
            .config
            .talker_config
            .codec_language_id
            .keys()
            .filter(|s| !s.contains("dialect"))
            .map(|s| s.as_str())
            .collect();
        v.sort_unstable();
        v.insert(0, "auto");
        v
    }

    /// Text embeddings projected to the talker dimension, shape (1, len, hidden_size).
    fn text_embed(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        self.talker
            .text_embedding
            .forward(&ids)?
            .apply(&self.talker.text_projection)
    }

    /// Codec embeddings, shape (1, len, hidden_size).
    fn codec_embed(&self, ids: &[u32]) -> Result<Tensor> {
        let ids = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        self.talker.codec_embedding.forward(&ids)
    }

    /// Sum of the embeddings of all the codebooks of the given frames, shape (1, frames,
    /// hidden_size). Codebook 0 uses the talker table, the others the code predictor tables.
    fn frames_embed(&self, codes: &Tensor) -> Result<Tensor> {
        let codes = codes.to_dtype(DType::U32)?.to_device(&self.device)?;
        let (_frames, groups) = codes.dims2()?;
        let num_code_groups = self.config.talker_config.num_code_groups;
        if groups != num_code_groups {
            candle::bail!("expected {num_code_groups} codebooks per frame, got {groups}")
        }
        let mut acc = self
            .talker
            .codec_embedding
            .forward(&codes.i((.., 0))?.contiguous()?)?;
        for i in 1..groups {
            let ids = codes.i((.., i))?.contiguous()?;
            let e = self.talker.code_predictor.codec_embeddings[i - 1].forward(&ids)?;
            acc = (acc + e)?;
        }
        acc.unsqueeze(0)
    }

    fn speaker_embed(&self, voice: Voice) -> Result<Option<Tensor>> {
        let embed = match voice {
            Voice::None => None,
            Voice::Speaker(name) => {
                let name = name.to_lowercase();
                match self.config.talker_config.spk_id.get(&name) {
                    Some(&id) => Some(self.codec_embed(&[id])?),
                    None => candle::bail!(
                        "unknown speaker {name:?}, supported speakers: {:?}",
                        self.supported_speakers()
                    ),
                }
            }
            Voice::Embedding(t) => {
                Some(t.to_device(&self.device)?.to_dtype(self.dtype)?.reshape((
                    1,
                    1,
                    self.config.talker_config.hidden_size,
                ))?)
            }
        };
        Ok(embed)
    }

    fn language_id(&self, language: Option<&str>, voice: Voice) -> Result<Option<u32>> {
        let tc = &self.config.talker_config;
        let language = language.unwrap_or("auto").to_lowercase();
        let mut language_id = if language == "auto" {
            None
        } else {
            match tc.codec_language_id.get(&language) {
                Some(&id) => Some(id),
                None => candle::bail!(
                    "unknown language {language:?}, supported languages: {:?}",
                    self.supported_languages()
                ),
            }
        };
        // Dialect speakers switch the language tag to their dialect.
        if let (true, Voice::Speaker(name)) = (language == "chinese" || language == "auto", voice) {
            if let Some(Dialect::Name(dialect)) = tc.spk_is_dialect.get(&name.to_lowercase()) {
                match tc.codec_language_id.get(dialect) {
                    Some(&id) => language_id = Some(id),
                    None => candle::bail!("unknown dialect {dialect:?} for speaker {name:?}"),
                }
            }
        }
        Ok(language_id)
    }

    /// Builds the prefix embeddings fed to the talker, shape (1, len, hidden_size), together
    /// with the text embeddings to add to the generated frames (`trailing_text`) and the
    /// padding embedding used once they are exhausted.
    pub fn build_prompt(&self, prompt: &Prompt) -> Result<(Tensor, Tensor, Tensor)> {
        let cfg = &self.config;
        let tc = &cfg.talker_config;
        let n = prompt.input_ids.len();
        // <|im_start|>assistant\n {text} <|im_end|>\n<|im_start|>assistant\n
        if n < 8 {
            candle::bail!("input_ids is too short ({n}), it must follow the chat template")
        }
        let text_ids = &prompt.input_ids[3..n - 5];
        if text_ids.is_empty() {
            candle::bail!("the text to synthesize is empty")
        }

        let mut parts = Vec::new();
        if let Some(instruct_ids) = prompt.instruct_ids {
            if !instruct_ids.is_empty() {
                parts.push(self.text_embed(instruct_ids)?);
            }
        }

        let speaker_embed = self.speaker_embed(prompt.voice)?;
        let language_id = self.language_id(prompt.language, prompt.voice)?;

        let special = self.text_embed(&[
            cfg.tts_bos_token_id,
            cfg.tts_eos_token_id,
            cfg.tts_pad_token_id,
        ])?;
        let tts_bos = special.narrow(1, 0, 1)?;
        let tts_eos = special.narrow(1, 1, 1)?;
        let tts_pad = special.narrow(1, 2, 1)?;

        let codec_prefill = match language_id {
            None => vec![
                tc.codec_nothink_id,
                tc.codec_think_bos_id,
                tc.codec_think_eos_id,
            ],
            Some(id) => vec![
                tc.codec_think_id,
                tc.codec_think_bos_id,
                id,
                tc.codec_think_eos_id,
            ],
        };
        let mut codec_input = vec![self.codec_embed(&codec_prefill)?];
        if let Some(speaker_embed) = speaker_embed {
            codec_input.push(speaker_embed);
        }
        codec_input.push(self.codec_embed(&[tc.codec_pad_id, tc.codec_bos_id])?);
        let codec_input = Tensor::cat(&codec_input, 1)?;
        let k = codec_input.dim(1)?;

        // <|im_start|>assistant\n
        let role = self.text_embed(&prompt.input_ids[..3])?;
        // tts_pad * (k - 2) + tts_bos, summed with the codec prefix but its final codec_bos.
        let hidden_size = tc.hidden_size;
        let text_prefix = Tensor::cat(&[&tts_pad.expand((1, k - 2, hidden_size))?, &tts_bos], 1)?;
        let prefix = (text_prefix + codec_input.narrow(1, 0, k - 1)?)?;
        let mut embeds = vec![role, prefix];

        let codec_pad = self.codec_embed(&[tc.codec_pad_id])?;
        let codec_bos = codec_input.narrow(1, k - 1, 1)?;
        let trailing = match prompt.icl {
            Some(icl) => {
                let m = icl.ref_ids.len();
                if m < 5 {
                    candle::bail!("ref_ids is too short ({m}), it must follow the chat template")
                }
                let mut ids = icl.ref_ids[3..m - 2].to_vec();
                ids.extend_from_slice(text_ids);
                let text_embed = Tensor::cat(&[self.text_embed(&ids)?, tts_eos], 1)?;
                let codec_embed = Tensor::cat(&[codec_bos, self.frames_embed(icl.ref_codes)?], 1)?;
                let text_len = text_embed.dim(1)?;
                let codec_len = codec_embed.dim(1)?;
                if prompt.non_streaming_mode {
                    let text = text_embed.broadcast_add(&codec_pad)?;
                    let codec = codec_embed.broadcast_add(&tts_pad)?;
                    embeds.push(Tensor::cat(&[text, codec], 1)?);
                    tts_pad.clone()
                } else if text_len > codec_len {
                    embeds.push((text_embed.narrow(1, 0, codec_len)? + codec_embed)?);
                    text_embed.narrow(1, codec_len, text_len - codec_len)?
                } else {
                    let padding = tts_pad.expand((1, codec_len - text_len, hidden_size))?;
                    let text_embed = Tensor::cat(&[text_embed, padding], 1)?;
                    embeds.push((text_embed + codec_embed)?);
                    tts_pad.clone()
                }
            }
            None => {
                if prompt.non_streaming_mode {
                    let text = Tensor::cat(&[self.text_embed(text_ids)?, tts_eos], 1)?
                        .broadcast_add(&codec_pad)?;
                    embeds.push(text);
                    embeds.push((&tts_pad + codec_bos)?);
                    tts_pad.clone()
                } else {
                    embeds.push((self.text_embed(&text_ids[..1])? + codec_bos)?);
                    if text_ids.len() > 1 {
                        Tensor::cat(&[self.text_embed(&text_ids[1..])?, tts_eos], 1)?
                    } else {
                        tts_eos
                    }
                }
            }
        };
        parts.push(Tensor::cat(&embeds, 1)?);
        Ok((Tensor::cat(&parts, 1)?, trailing, tts_pad))
    }

    /// Samples the first codebook entry of the next frame from the talker hidden state.
    fn sample_code0(
        &self,
        hidden: &Tensor,
        generated: &[u32],
        step: usize,
        gen: &GenerationConfig,
        lp: &mut LogitsProcessor,
    ) -> Result<u32> {
        let logits = hidden
            .apply(&self.talker.codec_head)?
            .i((0, 0))?
            .to_dtype(DType::F32)?;
        let logits = if gen.repetition_penalty == 1. {
            logits
        } else {
            crate::utils::apply_repeat_penalty(&logits, gen.repetition_penalty, generated)?
        };
        // The end of speech token is suppressed for the first few frames, as the reference
        // implementation does through `min_new_tokens`.
        let mask = if step < MIN_NEW_TOKENS {
            &self.codec_mask_no_eos
        } else {
            &self.codec_mask
        };
        lp.sample(&(logits + mask)?)
    }

    /// Generates the codec frames for `prompt`, each frame holding `num_code_groups` codes.
    ///
    /// `on_frame` is called with every generated frame, see [`Model::generate`].
    pub fn generate_with_callback(
        &mut self,
        prompt: &Prompt,
        gen: &GenerationConfig,
        mut on_frame: impl FnMut(&[u32]) -> Result<()>,
    ) -> Result<Vec<Vec<u32>>> {
        let (embeds, trailing, tts_pad) = self.build_prompt(prompt)?;
        let mut lp = LogitsProcessor::from_sampling(gen.seed, gen.sampling.clone());
        let mut sub_lp = LogitsProcessor::from_sampling(gen.seed, gen.subtalker_sampling.clone());
        let eos = self.config.talker_config.codec_eos_token_id;
        let trailing_len = trailing.dim(1)?;

        self.talker.model.clear_kv_cache();
        let prefix_len = embeds.dim(1)?;
        let hidden = self.talker.model.forward(&embeds, 0)?;
        let mut last_hidden = hidden.narrow(1, prefix_len - 1, 1)?;
        let mut generated = Vec::new();
        let mut frames = Vec::new();
        for step in 0..gen.max_new_tokens {
            let code0 = self.sample_code0(&last_hidden, &generated, step, gen, &mut lp)?;
            generated.push(code0);
            if code0 == eos {
                break;
            }
            let code0_embed = self.codec_embed(&[code0])?;
            let rest =
                self.talker
                    .code_predictor
                    .predict(&last_hidden, &code0_embed, &mut sub_lp)?;
            let mut frame = Vec::with_capacity(rest.len() + 1);
            frame.push(code0);
            frame.extend_from_slice(&rest);
            on_frame(&frame)?;

            // The next talker input is the sum of the frame embeddings and of the next text
            // token if any, of the padding embedding otherwise.
            let mut input = code0_embed;
            for (i, &code) in rest.iter().enumerate() {
                input = (input + self.talker.code_predictor.embed(i + 1, code)?)?;
            }
            let text = if step < trailing_len {
                trailing.narrow(1, step, 1)?
            } else {
                tts_pad.clone()
            };
            let input = (input + text)?;
            frames.push(frame);
            if step + 1 == gen.max_new_tokens {
                break;
            }
            last_hidden = self.talker.model.forward(&input, prefix_len + step)?;
        }
        Ok(frames)
    }

    /// Generates the codec frames for `prompt`, each frame holding `num_code_groups` codes.
    /// Decode them with [`speech_tokenizer::Model::decode`] to get audio.
    pub fn generate(&mut self, prompt: &Prompt, gen: &GenerationConfig) -> Result<Vec<Vec<u32>>> {
        self.generate_with_callback(prompt, gen, |_| Ok(()))
    }

    /// Converts generated frames to a tensor of shape (1, frames, num_code_groups).
    pub fn frames_to_tensor(&self, frames: &[Vec<u32>]) -> Result<Tensor> {
        let num_code_groups = self.config.talker_config.num_code_groups;
        let flat: Vec<u32> = frames.iter().flatten().copied().collect();
        Tensor::from_vec(flat, (1, frames.len(), num_code_groups), &self.device)
    }
}
