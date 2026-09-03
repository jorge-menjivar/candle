//! Minimal wav reading and resampling for the reference audio of voice cloning.

use anyhow::{bail, Context, Result};

/// Reads a wav file and returns mono f32 samples in [-1, 1] together with the sample rate.
/// Supports 16/24/32-bit PCM and 32-bit float data, multi-channel files are averaged.
pub fn read_wav<P: AsRef<std::path::Path>>(path: P) -> Result<(Vec<f32>, u32)> {
    let data = std::fs::read(path.as_ref())
        .with_context(|| format!("reading {}", path.as_ref().display()))?;
    if data.len() < 12 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("{} is not a wav file", path.as_ref().display())
    }
    let u16_at = |i: usize| u16::from_le_bytes([data[i], data[i + 1]]);
    let u32_at = |i: usize| u32::from_le_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]);
    let mut pos = 12;
    let mut format = None;
    let mut samples = None;
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let size = u32_at(pos + 4) as usize;
        let body = pos + 8;
        // Chunks are truncated to what the file actually holds rather than trusting the
        // declared size.
        let end = usize::min(body.saturating_add(size), data.len());
        let size = end - body;
        match id {
            b"fmt " => {
                if size < 16 {
                    bail!("invalid fmt chunk")
                }
                let mut audio_format = u16_at(body);
                let channels = u16_at(body + 2) as usize;
                let sample_rate = u32_at(body + 4);
                let bits = u16_at(body + 14) as usize;
                // WAVE_FORMAT_EXTENSIBLE stores the actual format in the sub-format GUID.
                if audio_format == 0xFFFE && size >= 26 {
                    audio_format = u16_at(body + 24);
                }
                if sample_rate == 0 {
                    bail!("invalid sample rate")
                }
                format = Some((audio_format, channels, sample_rate, bits));
            }
            b"data" => {
                let (audio_format, channels, _, bits) =
                    format.context("data chunk before fmt chunk")?;
                let bytes = bits / 8;
                if bytes == 0 || channels == 0 {
                    bail!("invalid wav format")
                }
                let raw = &data[body..end];
                let n = raw.len() / (bytes * channels);
                let mut pcm = Vec::with_capacity(n);
                for i in 0..n {
                    let mut acc = 0f32;
                    for c in 0..channels {
                        let o = (i * channels + c) * bytes;
                        let s = &raw[o..o + bytes];
                        let v = match (audio_format, bytes) {
                            (1, 2) => i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.,
                            (1, 3) => {
                                i32::from_le_bytes([0, s[0], s[1], s[2]]) as f32 / 2147483648.
                            }
                            (1, 4) => {
                                i32::from_le_bytes([s[0], s[1], s[2], s[3]]) as f32 / 2147483648.
                            }
                            (3, 4) => f32::from_le_bytes([s[0], s[1], s[2], s[3]]),
                            _ => bail!("unsupported wav format {audio_format} with {bits} bits"),
                        };
                        acc += v;
                    }
                    pcm.push(acc / channels as f32);
                }
                samples = Some(pcm);
            }
            _ => {}
        }
        // Chunks are word aligned.
        pos = end + (size & 1);
    }
    let (_, _, sample_rate, _) = format.context("no fmt chunk")?;
    let samples = samples.context("no data chunk")?;
    Ok((samples, sample_rate))
}

/// Windowed-sinc resampling of a mono signal.
pub fn resample(input: &[f32], sr_in: u32, sr_out: u32) -> Vec<f32> {
    if sr_in == sr_out || sr_in == 0 || sr_out == 0 {
        return input.to_vec();
    }
    let ratio = sr_out as f64 / sr_in as f64;
    let out_len = (input.len() as f64 * ratio).round() as usize;
    // Low-pass at the smaller of the two Nyquist frequencies, relative to the input rate.
    let cutoff = ratio.min(1.) * 0.95;
    let taps = 32i64;
    let len = input.len() as i64;
    let mut output = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 / ratio;
        let center = pos.floor() as i64;
        let mut acc = 0f64;
        let mut norm = 0f64;
        for t in -taps..=taps {
            let idx = center + t;
            if idx < 0 || idx >= len {
                continue;
            }
            let x = idx as f64 - pos;
            let sinc = if x == 0. {
                1.
            } else {
                let a = std::f64::consts::PI * cutoff * x;
                a.sin() / a
            };
            let window = 0.5 * (1. + (std::f64::consts::PI * x / (taps as f64 + 1.)).cos());
            let w = sinc * window;
            acc += input[idx as usize] as f64 * w;
            norm += w;
        }
        output.push(if norm.abs() > 1e-12 {
            (acc / norm) as f32
        } else {
            0.
        });
    }
    output
}

/// A wav file that samples can be appended to, for writing audio while it is generated.
///
/// The header is written upfront with placeholder lengths which are patched in [`Self::finish`].
pub struct WavWriter {
    file: std::fs::File,
    samples: usize,
}

impl WavWriter {
    const HEADER_LEN: usize = 44;

    pub fn new<P: AsRef<std::path::Path>>(path: P, sample_rate: u32) -> Result<Self> {
        use std::io::Write;
        let mut file = std::fs::File::create(path.as_ref())
            .with_context(|| format!("creating {}", path.as_ref().display()))?;
        let n_channels = 1u16;
        file.write_all(b"RIFF")?;
        file.write_all(&0u32.to_le_bytes())?; // patched by finish
        file.write_all(b"WAVE")?;
        file.write_all(b"fmt ")?;
        file.write_all(&16u32.to_le_bytes())?;
        file.write_all(&1u16.to_le_bytes())?; // PCM
        file.write_all(&n_channels.to_le_bytes())?;
        file.write_all(&sample_rate.to_le_bytes())?;
        file.write_all(&(sample_rate * 2 * n_channels as u32).to_le_bytes())?;
        file.write_all(&2u16.to_le_bytes())?;
        file.write_all(&16u16.to_le_bytes())?;
        file.write_all(b"data")?;
        file.write_all(&0u32.to_le_bytes())?; // patched by finish
        Ok(Self { file, samples: 0 })
    }

    pub fn write(&mut self, samples: &[f32]) -> Result<()> {
        use std::io::Write;
        let mut bytes = Vec::with_capacity(samples.len() * 2);
        for sample in samples.iter() {
            let v = (sample.clamp(-1., 1.) * 32767.) as i16;
            bytes.extend(v.to_le_bytes())
        }
        self.file.write_all(&bytes)?;
        self.samples += samples.len();
        Ok(())
    }

    pub fn samples(&self) -> usize {
        self.samples
    }

    /// Patches the two length fields of the header and flushes the file.
    pub fn finish(mut self) -> Result<()> {
        use std::io::{Seek, SeekFrom, Write};
        let data_len = (self.samples * 2) as u32;
        self.file.seek(SeekFrom::Start(4))?;
        self.file
            .write_all(&(Self::HEADER_LEN as u32 - 8 + data_len).to_le_bytes())?;
        self.file
            .seek(SeekFrom::Start(Self::HEADER_LEN as u64 - 4))?;
        self.file.write_all(&data_len.to_le_bytes())?;
        self.file.flush()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wav_file(samples: &[i16], sample_rate: u32, channels: u16) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let mut out = Vec::new();
        out.extend(b"RIFF");
        out.extend((36 + data.len() as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(channels.to_le_bytes());
        out.extend(sample_rate.to_le_bytes());
        out.extend((sample_rate * channels as u32 * 2).to_le_bytes());
        out.extend((channels * 2).to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
        out
    }

    fn read_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32)> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static COUNTER: AtomicUsize = AtomicUsize::new(0);
        let dir = std::env::temp_dir().join(format!("qwen3-tts-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir)?;
        // The tests run in parallel so each file needs its own name.
        let path = dir.join(format!("{}.wav", COUNTER.fetch_add(1, Ordering::Relaxed)));
        std::fs::write(&path, bytes)?;
        let out = read_wav(&path);
        let _ = std::fs::remove_file(&path);
        out
    }

    #[test]
    fn read_mono_and_stereo() -> Result<()> {
        let (pcm, sample_rate) = read_bytes(&wav_file(&[0, 16384, -16384], 24000, 1))?;
        assert_eq!(sample_rate, 24000);
        assert_eq!(pcm, [0., 0.5, -0.5]);
        // The channels of a stereo file are averaged.
        let (pcm, sample_rate) = read_bytes(&wav_file(&[0, 16384, -16384, 16384], 16000, 2))?;
        assert_eq!(sample_rate, 16000);
        assert_eq!(pcm, [0.25, 0.]);
        Ok(())
    }

    /// Malformed files should be reported rather than panicking on an out of bounds index.
    #[test]
    fn malformed_files_are_rejected() {
        let full = wav_file(&[0, 1, 2, 3], 24000, 1);
        for len in [0, 8, 12, 20, 24, 36] {
            assert!(
                read_bytes(&full[..len]).is_err(),
                "{len} bytes were accepted"
            );
        }
        let mut zero_rate = full.clone();
        zero_rate[24..28].copy_from_slice(&0u32.to_le_bytes());
        assert!(read_bytes(&zero_rate).is_err());
        // A chunk size larger than the file.
        let mut bad_size = full.clone();
        bad_size[16..20].copy_from_slice(&4096u32.to_le_bytes());
        assert!(read_bytes(&bad_size).is_err());
    }

    #[test]
    fn resampling_keeps_a_constant_signal() {
        let input = vec![0.5f32; 480];
        let output = resample(&input, 48000, 24000);
        assert_eq!(output.len(), 240);
        for v in output {
            assert!((v - 0.5).abs() < 1e-4, "{v}")
        }
        // A degenerate sample rate is passed through rather than panicking.
        assert_eq!(resample(&input, 0, 24000).len(), input.len());
    }
}
