//! ECAPA-TDNN speaker encoder of the Qwen3-TTS Base models.
//!
//! Turns a 24 kHz reference recording into a speaker embedding used to clone its voice. The
//! network consumes 128-bin log-mel spectrograms computed by [`mel_spectrogram`].
//!
//! See "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based
//! Speaker Verification" (https://huggingface.co/papers/2005.07143).

use candle::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, VarBuilder};

fn default_mel_dim() -> usize {
    128
}
fn default_enc_channels() -> Vec<usize> {
    vec![512, 512, 512, 512, 1536]
}
fn default_enc_kernel_sizes() -> Vec<usize> {
    vec![5, 3, 3, 3, 1]
}
fn default_enc_dilations() -> Vec<usize> {
    vec![1, 2, 3, 4, 1]
}
fn default_enc_attention_channels() -> usize {
    128
}
fn default_enc_res2net_scale() -> usize {
    8
}
fn default_enc_se_channels() -> usize {
    128
}
fn default_sample_rate() -> usize {
    24000
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    #[serde(default = "default_mel_dim")]
    pub mel_dim: usize,
    pub enc_dim: usize,
    #[serde(default = "default_enc_channels")]
    pub enc_channels: Vec<usize>,
    #[serde(default = "default_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,
    #[serde(default = "default_enc_dilations")]
    pub enc_dilations: Vec<usize>,
    #[serde(default = "default_enc_attention_channels")]
    pub enc_attention_channels: usize,
    #[serde(default = "default_enc_res2net_scale")]
    pub enc_res2net_scale: usize,
    #[serde(default = "default_enc_se_channels")]
    pub enc_se_channels: usize,
    #[serde(default = "default_sample_rate")]
    pub sample_rate: usize,
}

/// Reflect padding along the last dimension.
fn reflect_pad(xs: &Tensor, left: usize, right: usize) -> Result<Tensor> {
    if left == 0 && right == 0 {
        return Ok(xs.clone());
    }
    let len = xs.dim(D::Minus1)?;
    if left >= len || right >= len {
        candle::bail!("reflect padding ({left}, {right}) is too large for a length of {len}")
    }
    let mut indices = Vec::with_capacity(left + len + right);
    indices.extend((1..=left).rev().map(|i| i as u32));
    indices.extend(0..len as u32);
    indices.extend((1..=right).map(|i| (len - 1 - i) as u32));
    let indices = Tensor::new(indices, xs.device())?;
    xs.index_select(&indices, D::Minus1)
}

/// Conv1d with "same" reflect padding followed by a ReLU.
#[derive(Debug, Clone)]
struct TdnnBlock {
    conv: Conv1d,
    pad_left: usize,
    pad_right: usize,
}

impl TdnnBlock {
    fn new(
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let cfg = Conv1dConfig {
            dilation,
            ..Default::default()
        };
        let conv = candle_nn::conv1d(in_c, out_c, kernel_size, cfg, vb.pp("conv"))?;
        let total = dilation * (kernel_size - 1);
        let pad_left = total / 2;
        Ok(Self {
            conv,
            pad_left,
            pad_right: total - pad_left,
        })
    }
}

impl Module for TdnnBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        reflect_pad(xs, self.pad_left, self.pad_right)?
            .apply(&self.conv)?
            .relu()
    }
}

#[derive(Debug, Clone)]
struct Res2NetBlock {
    blocks: Vec<TdnnBlock>,
    scale: usize,
}

impl Res2NetBlock {
    fn new(
        in_channels: usize,
        out_channels: usize,
        scale: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let in_channel = in_channels / scale;
        let hidden_channel = out_channels / scale;
        let vb = vb.pp("blocks");
        let mut blocks = Vec::with_capacity(scale - 1);
        for i in 0..scale - 1 {
            blocks.push(TdnnBlock::new(
                in_channel,
                hidden_channel,
                kernel_size,
                dilation,
                vb.pp(i),
            )?);
        }
        Ok(Self { blocks, scale })
    }
}

impl Module for Res2NetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let chunks = xs.chunk(self.scale, 1)?;
        let mut outputs = Vec::with_capacity(self.scale);
        for (i, chunk) in chunks.iter().enumerate() {
            let output = if i == 0 {
                chunk.clone()
            } else if i == 1 {
                chunk.contiguous()?.apply(&self.blocks[0])?
            } else {
                let prev: &Tensor = &outputs[i - 1];
                (chunk + prev)?.apply(&self.blocks[i - 1])?
            };
            outputs.push(output);
        }
        Tensor::cat(&outputs, 1)
    }
}

#[derive(Debug, Clone)]
struct SqueezeExcitationBlock {
    conv1: Conv1d,
    conv2: Conv1d,
}

impl SqueezeExcitationBlock {
    fn new(
        in_channels: usize,
        se_channels: usize,
        out_channels: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv1 = candle_nn::conv1d(
            in_channels,
            se_channels,
            1,
            Default::default(),
            vb.pp("conv1"),
        )?;
        let conv2 = candle_nn::conv1d(
            se_channels,
            out_channels,
            1,
            Default::default(),
            vb.pp("conv2"),
        )?;
        Ok(Self { conv1, conv2 })
    }
}

impl Module for SqueezeExcitationBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let s = xs
            .mean_keepdim(D::Minus1)?
            .apply(&self.conv1)?
            .relu()?
            .apply(&self.conv2)?;
        xs.broadcast_mul(&candle_nn::ops::sigmoid(&s)?)
    }
}

#[derive(Debug, Clone)]
struct SeRes2NetBlock {
    tdnn1: TdnnBlock,
    res2net_block: Res2NetBlock,
    tdnn2: TdnnBlock,
    se_block: SqueezeExcitationBlock,
}

impl SeRes2NetBlock {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_channels: usize,
        out_channels: usize,
        res2net_scale: usize,
        se_channels: usize,
        kernel_size: usize,
        dilation: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            tdnn1: TdnnBlock::new(in_channels, out_channels, 1, 1, vb.pp("tdnn1"))?,
            res2net_block: Res2NetBlock::new(
                out_channels,
                out_channels,
                res2net_scale,
                kernel_size,
                dilation,
                vb.pp("res2net_block"),
            )?,
            tdnn2: TdnnBlock::new(out_channels, out_channels, 1, 1, vb.pp("tdnn2"))?,
            se_block: SqueezeExcitationBlock::new(
                out_channels,
                se_channels,
                out_channels,
                vb.pp("se_block"),
            )?,
        })
    }
}

impl Module for SeRes2NetBlock {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let h = xs
            .apply(&self.tdnn1)?
            .apply(&self.res2net_block)?
            .apply(&self.tdnn2)?
            .apply(&self.se_block)?;
        h + xs
    }
}

/// Attentive statistics pooling: returns the concatenated attention-weighted mean and standard
/// deviation of every channel.
#[derive(Debug, Clone)]
struct AttentiveStatisticsPooling {
    tdnn: TdnnBlock,
    conv: Conv1d,
}

impl AttentiveStatisticsPooling {
    const EPS: f64 = 1e-12;

    fn new(channels: usize, attention_channels: usize, vb: VarBuilder) -> Result<Self> {
        let tdnn = TdnnBlock::new(channels * 3, attention_channels, 1, 1, vb.pp("tdnn"))?;
        let conv = candle_nn::conv1d(
            attention_channels,
            channels,
            1,
            Default::default(),
            vb.pp("conv"),
        )?;
        Ok(Self { tdnn, conv })
    }

    /// Weighted mean and standard deviation over time, `weights` sums to one over time.
    fn statistics(xs: &Tensor, weights: &Tensor) -> Result<(Tensor, Tensor)> {
        let mean = xs.broadcast_mul(weights)?.sum_keepdim(D::Minus1)?;
        let var = xs
            .broadcast_sub(&mean)?
            .sqr()?
            .broadcast_mul(weights)?
            .sum_keepdim(D::Minus1)?;
        let std = var.clamp(Self::EPS, f64::MAX)?.sqrt()?;
        Ok((mean, std))
    }
}

impl Module for AttentiveStatisticsPooling {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (b, c, t) = xs.dims3()?;
        let uniform =
            Tensor::full(1f32 / t as f32, (b, 1, t), xs.device())?.to_dtype(xs.dtype())?;
        let (mean, std) = Self::statistics(xs, &uniform)?;
        let attention = Tensor::cat(&[xs, &mean.expand((b, c, t))?, &std.expand((b, c, t))?], 1)?;
        let attention = attention.apply(&self.tdnn)?.tanh()?.apply(&self.conv)?;
        let attention = candle_nn::ops::softmax_last_dim(&attention)?;
        let (mean, std) = Self::statistics(xs, &attention)?;
        Tensor::cat(&[mean, std], 1)
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    first: TdnnBlock,
    blocks: Vec<SeRes2NetBlock>,
    mfa: TdnnBlock,
    asp: AttentiveStatisticsPooling,
    fc: Conv1d,
}

impl Model {
    pub fn new(cfg: &Config, vb: VarBuilder) -> Result<Self> {
        let n = cfg.enc_channels.len();
        if n != cfg.enc_kernel_sizes.len() || n != cfg.enc_dilations.len() {
            candle::bail!(
                "enc_channels, enc_kernel_sizes and enc_dilations should have same length"
            )
        }
        // One TDNN block, at least one SE-Res2Net block and the multi-layer feature
        // aggregation.
        if n < 3 {
            candle::bail!("enc_channels should have at least three entries, got {n}")
        }
        let vb_b = vb.pp("blocks");
        let first = TdnnBlock::new(
            cfg.mel_dim,
            cfg.enc_channels[0],
            cfg.enc_kernel_sizes[0],
            cfg.enc_dilations[0],
            vb_b.pp(0),
        )?;
        let mut blocks = Vec::with_capacity(n - 2);
        for i in 1..n - 1 {
            blocks.push(SeRes2NetBlock::new(
                cfg.enc_channels[i - 1],
                cfg.enc_channels[i],
                cfg.enc_res2net_scale,
                cfg.enc_se_channels,
                cfg.enc_kernel_sizes[i],
                cfg.enc_dilations[i],
                vb_b.pp(i),
            )?);
        }
        let mfa = TdnnBlock::new(
            cfg.enc_channels[n - 1],
            cfg.enc_channels[n - 1],
            cfg.enc_kernel_sizes[n - 1],
            cfg.enc_dilations[n - 1],
            vb.pp("mfa"),
        )?;
        let asp = AttentiveStatisticsPooling::new(
            cfg.enc_channels[n - 1],
            cfg.enc_attention_channels,
            vb.pp("asp"),
        )?;
        let fc = candle_nn::conv1d(
            cfg.enc_channels[n - 1] * 2,
            cfg.enc_dim,
            1,
            Default::default(),
            vb.pp("fc"),
        )?;
        Ok(Self {
            first,
            blocks,
            mfa,
            asp,
            fc,
        })
    }

    /// `mels` has shape (B, frames, mel_dim), returns speaker embeddings of shape (B, enc_dim).
    pub fn forward(&self, mels: &Tensor) -> Result<Tensor> {
        let mut xs = mels.transpose(1, 2)?.contiguous()?.apply(&self.first)?;
        let mut outputs = Vec::with_capacity(self.blocks.len());
        for block in self.blocks.iter() {
            xs = xs.apply(block)?;
            outputs.push(xs.clone());
        }
        Tensor::cat(&outputs, 1)?
            .apply(&self.mfa)?
            .apply(&self.asp)?
            .apply(&self.fc)?
            .squeeze(D::Minus1)
    }
}

/// Parameters of the log-mel spectrogram expected by the speaker encoder.
#[derive(Debug, Clone)]
pub struct MelConfig {
    pub sample_rate: usize,
    pub n_fft: usize,
    pub hop_size: usize,
    pub win_size: usize,
    pub num_mels: usize,
    pub fmin: f64,
    pub fmax: f64,
}

impl Default for MelConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            n_fft: 1024,
            hop_size: 256,
            win_size: 1024,
            num_mels: 128,
            fmin: 0.,
            fmax: 12000.,
        }
    }
}

// Slaney-style mel scale, as in librosa with `htk=False`.
fn hz_to_mel(f: f64) -> f64 {
    let f_sp = 200. / 3.;
    let min_log_hz = 1000.;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.;
    if f >= min_log_hz {
        min_log_mel + (f / min_log_hz).ln() / logstep
    } else {
        f / f_sp
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let f_sp = 200. / 3.;
    let min_log_hz = 1000.;
    let min_log_mel = min_log_hz / f_sp;
    let logstep = 6.4f64.ln() / 27.;
    if m >= min_log_mel {
        min_log_hz * (logstep * (m - min_log_mel)).exp()
    } else {
        f_sp * m
    }
}

/// Mel filterbank of shape (num_mels, n_fft / 2 + 1) matching `librosa.filters.mel` with the
/// default Slaney normalization.
pub fn mel_filters(cfg: &MelConfig) -> Vec<f32> {
    let n_freqs = cfg.n_fft / 2 + 1;
    let fft_freqs: Vec<f64> = (0..n_freqs)
        .map(|k| k as f64 * cfg.sample_rate as f64 / cfg.n_fft as f64)
        .collect();
    let (min_mel, max_mel) = (hz_to_mel(cfg.fmin), hz_to_mel(cfg.fmax));
    let n_points = cfg.num_mels + 2;
    let mel_f: Vec<f64> = (0..n_points)
        .map(|i| mel_to_hz(min_mel + (max_mel - min_mel) * i as f64 / (n_points - 1) as f64))
        .collect();
    let mut weights = vec![0f32; cfg.num_mels * n_freqs];
    for i in 0..cfg.num_mels {
        let enorm = 2. / (mel_f[i + 2] - mel_f[i]);
        for (k, &f) in fft_freqs.iter().enumerate() {
            let lower = (f - mel_f[i]) / (mel_f[i + 1] - mel_f[i]);
            let upper = (mel_f[i + 2] - f) / (mel_f[i + 2] - mel_f[i + 1]);
            weights[i * n_freqs + k] = (lower.min(upper).max(0.) * enorm) as f32;
        }
    }
    weights
}

/// In-place iterative radix-2 FFT, `re` and `im` must have a power of two length.
fn fft(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let angle = -2. * std::f64::consts::PI / len as f64;
        let (w_re, w_im) = (angle.cos(), angle.sin());
        for start in (0..n).step_by(len) {
            let (mut cur_re, mut cur_im) = (1f64, 0f64);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (t_re, t_im) = (
                    re[b] * cur_re - im[b] * cur_im,
                    re[b] * cur_im + im[b] * cur_re,
                );
                re[b] = re[a] - t_re;
                im[b] = im[a] - t_im;
                re[a] += t_re;
                im[a] += t_im;
                let next_re = cur_re * w_re - cur_im * w_im;
                cur_im = cur_re * w_im + cur_im * w_re;
                cur_re = next_re;
            }
        }
        len <<= 1;
    }
}

/// Log-mel spectrogram of a mono recording sampled at `cfg.sample_rate`, returned as a tensor of
/// shape (1, frames, num_mels) on `device`.
///
/// This mirrors the reference `mel_spectrogram` function: reflect padding of `(n_fft - hop) / 2`
/// samples, periodic Hann window, magnitude spectrum, Slaney mel filterbank and a natural
/// logarithm with a `1e-5` floor.
pub fn mel_spectrogram(samples: &[f32], cfg: &MelConfig, device: &Device) -> Result<Tensor> {
    if !cfg.n_fft.is_power_of_two() {
        candle::bail!("n_fft must be a power of two, got {}", cfg.n_fft)
    }
    if cfg.hop_size == 0 || cfg.hop_size > cfg.n_fft {
        candle::bail!("hop_size {} should be in 1..={}", cfg.hop_size, cfg.n_fft)
    }
    if cfg.win_size > cfg.n_fft {
        candle::bail!(
            "win_size {} should be at most n_fft {}",
            cfg.win_size,
            cfg.n_fft
        )
    }
    let padding = (cfg.n_fft - cfg.hop_size) / 2;
    let len = samples.len();
    // The padding reflects around the first and last sample, and one full window has to fit in
    // the padded signal.
    if len <= padding || len + 2 * padding < cfg.n_fft {
        candle::bail!("the recording is too short ({len} samples)")
    }
    let mut padded = Vec::with_capacity(len + 2 * padding);
    padded.extend((1..=padding).rev().map(|i| samples[i] as f64));
    padded.extend(samples.iter().map(|&v| v as f64));
    padded.extend((1..=padding).map(|i| samples[len - 1 - i] as f64));
    let n_frames = (padded.len() - cfg.n_fft) / cfg.hop_size + 1;
    let n_freqs = cfg.n_fft / 2 + 1;
    let window: Vec<f64> = (0..cfg.win_size)
        .map(|i| 0.5 - 0.5 * (2. * std::f64::consts::PI * i as f64 / cfg.win_size as f64).cos())
        .collect();
    let filters = mel_filters(cfg);
    let mut mels = vec![0f32; n_frames * cfg.num_mels];
    let mut re = vec![0f64; cfg.n_fft];
    let mut im = vec![0f64; cfg.n_fft];
    let mut magnitudes = vec![0f64; n_freqs];
    for frame in 0..n_frames {
        let start = frame * cfg.hop_size;
        re.iter_mut().for_each(|v| *v = 0.);
        im.iter_mut().for_each(|v| *v = 0.);
        for i in 0..cfg.win_size {
            re[i] = padded[start + i] * window[i];
        }
        fft(&mut re, &mut im);
        for k in 0..n_freqs {
            magnitudes[k] = (re[k] * re[k] + im[k] * im[k] + 1e-9).sqrt();
        }
        for m in 0..cfg.num_mels {
            let filter = &filters[m * n_freqs..(m + 1) * n_freqs];
            let v: f64 = filter
                .iter()
                .zip(magnitudes.iter())
                .map(|(&w, &s)| w as f64 * s)
                .sum();
            mels[frame * cfg.num_mels + m] = v.max(1e-5).ln() as f32;
        }
    }
    Tensor::from_vec(mels, (1, n_frames, cfg.num_mels), device)?.to_dtype(DType::F32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> Config {
        Config {
            enc_dim: 32,
            mel_dim: 16,
            // The multi-layer feature aggregation takes the concatenation of the
            // SE-Res2Net outputs, so the last entry is (n - 2) times the block width.
            enc_channels: vec![24, 24, 24, 48],
            enc_kernel_sizes: vec![3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 1],
            enc_attention_channels: 8,
            enc_res2net_scale: 4,
            enc_se_channels: 8,
            sample_rate: 24000,
        }
    }

    /// The Res2Net chunks are not contiguous as soon as the batch holds more than one
    /// recording, which the convolutions do not accept.
    #[test]
    fn batched_forward() -> Result<()> {
        let device = Device::Cpu;
        let cfg = test_config();
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let model = Model::new(&cfg, vb)?;
        for batch_size in [1, 3] {
            let mels = Tensor::zeros((batch_size, 20, cfg.mel_dim), DType::F32, &device)?;
            let xs = model.forward(&mels)?;
            assert_eq!(xs.dims(), [batch_size, cfg.enc_dim]);
        }
        Ok(())
    }

    #[test]
    fn degenerate_configs_are_rejected() {
        let device = Device::Cpu;
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);
        let cfg = Config {
            enc_channels: vec![24],
            enc_kernel_sizes: vec![3],
            enc_dilations: vec![1],
            ..test_config()
        };
        assert!(Model::new(&cfg, vb).is_err());
        let samples = vec![0f32; 4096];
        let cfg = MelConfig {
            hop_size: 0,
            ..Default::default()
        };
        assert!(mel_spectrogram(&samples, &cfg, &device).is_err());
        let cfg = MelConfig {
            win_size: 4096,
            ..Default::default()
        };
        assert!(mel_spectrogram(&samples, &cfg, &device).is_err());
        let cfg = MelConfig::default();
        assert!(mel_spectrogram(&samples[..16], &cfg, &device).is_err());
    }
}
