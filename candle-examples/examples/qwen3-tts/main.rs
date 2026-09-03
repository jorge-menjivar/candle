#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use anyhow::{Error as E, Result};
use clap::Parser;

use candle::{DType, Device, IndexOp, Tensor};
use candle_examples::hub::Api;
use candle_nn::VarBuilder;
use candle_transformers::generation::Sampling;
use candle_transformers::models::qwen3_tts::{
    speech_tokenizer, Config, GenerationConfig, IclReference, Model, Prompt, Voice,
};
use tokenizers::Tokenizer;

mod audio;

#[derive(Clone, Debug, Copy, PartialEq, Eq, clap::ValueEnum)]
enum Which {
    #[value(name = "0.6b-custom-voice")]
    CustomVoice0_6B,
    #[value(name = "1.7b-custom-voice")]
    CustomVoice1_7B,
    #[value(name = "1.7b-voice-design")]
    VoiceDesign1_7B,
    #[value(name = "0.6b-base")]
    Base0_6B,
    #[value(name = "1.7b-base")]
    Base1_7B,
}

impl Which {
    fn model_id(&self) -> &'static str {
        match self {
            Self::CustomVoice0_6B => "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
            Self::CustomVoice1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            Self::VoiceDesign1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            Self::Base0_6B => "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
            Self::Base1_7B => "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Run on CPU rather than on GPU.
    #[arg(long)]
    cpu: bool,

    /// Enable tracing (generates a trace-timestamp.json file).
    #[arg(long)]
    tracing: bool,

    /// The model variant to use.
    #[arg(long, default_value = "0.6b-custom-voice")]
    which: Which,

    /// The text to synthesize.
    #[arg(
        long,
        default_value = "Hello there, this is a test of text to speech with candle."
    )]
    text: String,

    /// The predefined speaker to use with the CustomVoice models, see --list-speakers.
    #[arg(long, conflicts_with = "ref_audio")]
    speaker: Option<String>,

    /// The language of the text, "auto" for automatic detection.
    #[arg(long, default_value = "auto")]
    language: String,

    /// A natural language description of the voice/style, VoiceDesign and 1.7B CustomVoice
    /// models only.
    #[arg(long)]
    instruct: Option<String>,

    /// Print the speakers and languages supported by the model and exit.
    #[arg(long)]
    list_speakers: bool,

    /// A wav file with a few seconds of speech to clone the voice from, Base models only.
    #[arg(long)]
    ref_audio: Option<String>,

    /// The transcript of the reference audio. When given, the reference audio is also used as
    /// an in-context example, otherwise only its speaker embedding is used.
    #[arg(long)]
    ref_text: Option<String>,

    /// Feed the text one token per generated frame instead of putting the whole text in the
    /// prefix. This is the default when cloning a voice.
    #[arg(long, conflicts_with = "non_streaming_text")]
    streaming_text: bool,

    /// Put the whole text in the prefix, this is the default when not cloning a voice.
    #[arg(long, conflicts_with = "streaming_text")]
    non_streaming_text: bool,

    /// The output file using the wav format.
    #[arg(long, default_value = "out.wav")]
    out_file: String,

    /// Decode and write the audio while it is being generated rather than at the end.
    #[arg(long)]
    stream: bool,

    /// The number of frames decoded at a time when streaming, 12.5 frames per second of audio.
    #[arg(long, default_value_t = 25)]
    stream_chunk: usize,

    /// The number of preceding frames each streamed chunk is decoded with, so that the audio
    /// matches what a single decode of everything would produce. Below 50 the chunk boundaries
    /// become audible, above it there is nothing left to gain.
    #[arg(long, default_value_t = 50)]
    stream_context: usize,

    /// Also save the generated codec tokens to this safetensors file.
    #[arg(long)]
    codes_file: Option<String>,

    /// Use greedy decoding for both the talker and the code predictor.
    #[arg(long)]
    greedy: bool,

    /// The temperature used to sample the first codebook.
    #[arg(long, default_value_t = 0.9)]
    temperature: f64,

    /// Only sample among the top K samples for the first codebook.
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Nucleus sampling probability cutoff for the first codebook.
    #[arg(long, default_value_t = 1.0)]
    top_p: f64,

    /// The temperature used to sample the other codebooks.
    #[arg(long, default_value_t = 0.9)]
    subtalker_temperature: f64,

    /// Only sample among the top K samples for the other codebooks.
    #[arg(long, default_value_t = 50)]
    subtalker_top_k: usize,

    /// Nucleus sampling probability cutoff for the other codebooks.
    #[arg(long, default_value_t = 1.0)]
    subtalker_top_p: f64,

    /// Penalty to be applied for repeating tokens, 1. means no penalty.
    #[arg(long, default_value_t = 1.05)]
    repetition_penalty: f32,

    /// The maximum number of codec frames to generate (12.5 frames per second).
    #[arg(long, default_value_t = 8192)]
    max_new_tokens: usize,

    /// The seed to use when generating random samples.
    #[arg(long, default_value_t = 299792458)]
    seed: u64,

    /// The dtype of the talker weights: f32, bf16 or f16. Defaults to bf16 on GPU and f32 on
    /// CPU. The speech tokenizer always runs in f32.
    #[arg(long)]
    dtype: Option<String>,

    #[arg(long)]
    model_id: Option<String>,

    #[arg(long, default_value = "main")]
    revision: String,

    /// The tokenizer.json file, defaults to the one of Qwen/Qwen3-0.6B which shares the
    /// vocabulary of the TTS models.
    #[arg(long)]
    tokenizer: Option<String>,

    #[arg(long)]
    config: Option<String>,

    #[arg(long)]
    weights: Option<String>,

    #[arg(long)]
    speech_tokenizer_config: Option<String>,

    #[arg(long)]
    speech_tokenizer_weights: Option<String>,
}

/// Decodes the generated frames as they come and appends the audio to the output file.
///
/// Each chunk is decoded together with the frames that precede it, whose audio is then
/// discarded, so that the result matches decoding everything at once.
struct Streamer<'a> {
    speech_tokenizer: &'a mut speech_tokenizer::Model,
    writer: audio::WavWriter,
    /// The reference frames of a voice clone followed by the generated ones, flattened.
    codes: Vec<u32>,
    num_code_groups: usize,
    /// Frames whose audio was written already, the reference ones included.
    written: usize,
    chunk: usize,
    left_context: usize,
    samples_per_frame: usize,
    device: Device,
    first_audio: Option<std::time::Duration>,
    start: std::time::Instant,
}

impl<'a> Streamer<'a> {
    fn new(
        speech_tokenizer: &'a mut speech_tokenizer::Model,
        out_file: &str,
        chunk: usize,
        left_context: usize,
        ref_codes: Option<&Tensor>,
        device: &Device,
    ) -> Result<Self> {
        let sample_rate = speech_tokenizer.output_sample_rate() as u32;
        let samples_per_frame = speech_tokenizer.samples_per_frame();
        let (codes, written, num_code_groups) = match ref_codes {
            Some(ref_codes) => {
                let (frames, groups) = ref_codes.dims2()?;
                (ref_codes.flatten_all()?.to_vec1::<u32>()?, frames, groups)
            }
            None => (vec![], 0, 0),
        };
        Ok(Self {
            speech_tokenizer,
            writer: audio::WavWriter::new(out_file, sample_rate)?,
            codes,
            num_code_groups,
            written,
            chunk,
            left_context,
            samples_per_frame,
            device: device.clone(),
            first_audio: None,
            start: std::time::Instant::now(),
        })
    }

    fn push(&mut self, frame: &[u32]) -> Result<()> {
        if self.num_code_groups == 0 {
            self.num_code_groups = frame.len()
        }
        self.codes.extend_from_slice(frame);
        let pending = self.codes.len() / self.num_code_groups - self.written;
        if pending >= self.chunk {
            self.decode_pending()?
        }
        Ok(())
    }

    /// Decodes everything that has been generated since the last chunk.
    fn decode_pending(&mut self) -> Result<()> {
        let frames = self.codes.len() / self.num_code_groups;
        if frames <= self.written {
            return Ok(());
        }
        let context = usize::min(self.left_context, self.written);
        let start = self.written - context;
        let len = frames - start;
        let codes = Tensor::from_slice(
            &self.codes[start * self.num_code_groups..frames * self.num_code_groups],
            (1, len, self.num_code_groups),
            &self.device,
        )?;
        let pcm = self.speech_tokenizer.decode(&codes)?.i(0)?;
        let skip = context * self.samples_per_frame;
        let pcm = pcm.narrow(0, skip, pcm.dim(0)? - skip)?.to_vec1::<f32>()?;
        self.writer.write(&pcm)?;
        self.written = frames;
        if self.first_audio.is_none() {
            self.first_audio = Some(self.start.elapsed())
        }
        Ok(())
    }

    fn finish(mut self) -> Result<(usize, Option<std::time::Duration>)> {
        self.decode_pending()?;
        let samples = self.writer.samples();
        let first_audio = self.first_audio;
        self.writer.finish()?;
        Ok((samples, first_audio))
    }
}

fn main() -> Result<()> {
    use tracing_chrome::ChromeLayerBuilder;
    use tracing_subscriber::prelude::*;

    let args = Args::parse();

    let _guard = if args.tracing {
        let (chrome_layer, guard) = ChromeLayerBuilder::new().build();
        tracing_subscriber::registry().with(chrome_layer).init();
        Some(guard)
    } else {
        None
    };
    println!(
        "avx: {}, neon: {}, simd128: {}, f16c: {}",
        candle::utils::with_avx(),
        candle::utils::with_neon(),
        candle::utils::with_simd128(),
        candle::utils::with_f16c()
    );

    let start = std::time::Instant::now();
    let api = Api::new()?;
    let model_id = args
        .model_id
        .clone()
        .unwrap_or_else(|| args.which.model_id().to_string());
    let repo = api.model(model_id).with_revision(&args.revision);
    let path = |arg: &Option<String>, name: &str| -> Result<std::path::PathBuf> {
        match arg {
            Some(p) => Ok(std::path::PathBuf::from(p)),
            None => Ok(repo.get(name)?),
        }
    };
    let config_file = path(&args.config, "config.json")?;
    let weights_file = path(&args.weights, "model.safetensors")?;
    let st_config_file = path(
        &args.speech_tokenizer_config,
        "speech_tokenizer/config.json",
    )?;
    let st_weights_file = path(
        &args.speech_tokenizer_weights,
        "speech_tokenizer/model.safetensors",
    )?;
    let tokenizer_file = match &args.tokenizer {
        Some(file) => std::path::PathBuf::from(file),
        None => api.model("Qwen/Qwen3-0.6B").get("tokenizer.json")?,
    };
    println!("retrieved the files in {:?}", start.elapsed());

    let start = std::time::Instant::now();
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(E::msg)?;
    let config: Config = serde_json::from_slice(&std::fs::read(config_file)?)?;
    let st_config: speech_tokenizer::Config =
        serde_json::from_slice(&std::fs::read(st_config_file)?)?;
    let device = candle_examples::device(args.cpu)?;
    let dtype = match args.dtype.as_deref() {
        Some("f32") => DType::F32,
        Some("bf16") => DType::BF16,
        Some("f16") => DType::F16,
        Some(dtype) => anyhow::bail!("unsupported dtype {dtype}"),
        None => device.bf16_default_to_f32(),
    };
    let mut model = {
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_file], dtype, &device)? };
        Model::new(&config, vb)?
    };
    let ref_text = args.ref_text.as_deref().filter(|s| !s.is_empty());
    let icl_mode = args.ref_audio.is_some() && ref_text.is_some();
    let mut speech_tokenizer = {
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[st_weights_file], DType::F32, &device)?
        };
        if icl_mode {
            speech_tokenizer::Model::new_with_encoder(&st_config, vb)?
        } else {
            speech_tokenizer::Model::new(&st_config, vb)?
        }
    };
    println!("loaded the models in {:?}", start.elapsed());

    let speakers = model.supported_speakers();
    if args.list_speakers {
        println!("model type: {}", config.tts_model_type);
        println!("speakers: {speakers:?}");
        println!("languages: {:?}", model.supported_languages());
        return Ok(());
    }

    // Voice cloning: speaker embedding and codec tokens of the reference audio.
    let (speaker_embedding, ref_codes) = match &args.ref_audio {
        Some(ref_audio) => {
            if !model.has_speaker_encoder() {
                anyhow::bail!(
                    "{} has no speaker encoder, voice cloning requires a Base model",
                    config.tts_model_type
                )
            }
            let start = std::time::Instant::now();
            let (pcm, sample_rate) = audio::read_wav(ref_audio)?;
            let target_rate = speech_tokenizer.input_sample_rate() as u32;
            let pcm = audio::resample(&pcm, sample_rate, target_rate);
            println!(
                "reference audio: {:.2}s at {sample_rate}Hz",
                pcm.len() as f64 / target_rate as f64
            );
            let speaker_embedding = model.speaker_embedding(&pcm)?;
            let ref_codes = if icl_mode {
                let pcm = Tensor::new(pcm.as_slice(), &device)?.unsqueeze(0)?;
                Some(speech_tokenizer.encode(&pcm)?.i(0)?)
            } else {
                None
            };
            println!("encoded the reference audio in {:?}", start.elapsed());
            (Some(speaker_embedding), ref_codes)
        }
        None => (None, None),
    };

    let speaker = match &args.speaker {
        Some(s) => Some(s.clone()),
        None if speaker_embedding.is_some() => None,
        None if config.tts_model_type == "custom_voice" => {
            let s = if speakers.contains(&"ryan") {
                "ryan"
            } else {
                speakers.first().copied().unwrap_or_default()
            };
            println!("no speaker specified, using {s:?} (see --list-speakers)");
            Some(s.to_string())
        }
        None => None,
    };
    let voice = match (&speaker, &speaker_embedding) {
        (Some(s), _) => Voice::Speaker(s),
        (None, Some(e)) => Voice::Embedding(e),
        (None, None) => Voice::None,
    };

    let encode = |text: &str| -> Result<Vec<u32>> {
        let encoding = tokenizer.encode(text, false).map_err(E::msg)?;
        Ok(encoding.get_ids().to_vec())
    };
    let input_ids = encode(&format!(
        "<|im_start|>assistant\n{}<|im_end|>\n<|im_start|>assistant\n",
        args.text
    ))?;
    let instruct = args.instruct.as_deref().filter(|s| !s.is_empty());
    let instruct = match instruct {
        Some(_) if config.tts_model_size == "0b6" => {
            println!("--instruct is not supported by the 0.6B models, ignoring it");
            None
        }
        Some(_) if args.ref_audio.is_some() => {
            println!("--instruct is not supported when cloning a voice, ignoring it");
            None
        }
        instruct => instruct,
    };
    let instruct_ids = match instruct {
        Some(instruct) => Some(encode(&format!(
            "<|im_start|>user\n{instruct}<|im_end|>\n"
        ))?),
        None => None,
    };
    let ref_ids = match ref_text {
        Some(ref_text) if ref_codes.is_some() => Some(encode(&format!(
            "<|im_start|>assistant\n{ref_text}<|im_end|>\n"
        ))?),
        _ => None,
    };
    let icl = match (&ref_ids, &ref_codes) {
        (Some(ref_ids), Some(ref_codes)) => Some(IclReference { ref_ids, ref_codes }),
        _ => None,
    };
    // The reference implementation streams the text when cloning a voice and uses the
    // non-streaming mode otherwise.
    let non_streaming_mode = if args.streaming_text {
        false
    } else if args.non_streaming_text {
        true
    } else {
        args.ref_audio.is_none()
    };
    let prompt = Prompt {
        input_ids: &input_ids,
        instruct_ids: instruct_ids.as_deref(),
        language: Some(&args.language),
        voice,
        icl,
        non_streaming_mode,
    };

    // Degenerate sampling parameters are turned into greedy decoding rather than being
    // rejected by the sampler in the middle of the generation.
    let sampling = |k: usize, p: f64, temperature: f64| {
        if args.greedy || temperature <= 0. {
            Sampling::ArgMax
        } else if k == 0 {
            Sampling::TopP { p, temperature }
        } else {
            Sampling::TopKThenTopP { k, p, temperature }
        }
    };
    let subtalker_sampling = sampling(
        args.subtalker_top_k,
        args.subtalker_top_p,
        args.subtalker_temperature,
    );
    let sampling = sampling(args.top_k, args.top_p, args.temperature);
    let gen = GenerationConfig {
        max_new_tokens: args.max_new_tokens,
        sampling,
        subtalker_sampling,
        repetition_penalty: args.repetition_penalty,
        seed: args.seed,
    };

    let sample_rate = speech_tokenizer.output_sample_rate() as u32;
    let mut streamer = if args.stream {
        if args.stream_chunk == 0 {
            anyhow::bail!("--stream-chunk should be at least one frame")
        }
        println!("streaming the audio to {}", args.out_file);
        Some(Streamer::new(
            &mut speech_tokenizer,
            &args.out_file,
            args.stream_chunk,
            args.stream_context,
            ref_codes.as_ref(),
            &device,
        )?)
    } else {
        None
    };

    let start = std::time::Instant::now();
    let mut num_frames = 0;
    let frames = model.generate_with_callback(&prompt, &gen, |frame| {
        num_frames += 1;
        print!(
            "\rgenerated {num_frames} frames ({:.1}s)",
            num_frames as f64 / 12.5
        );
        use std::io::Write;
        std::io::stdout().flush()?;
        if let Some(streamer) = streamer.as_mut() {
            streamer.push(frame).map_err(candle::Error::wrap)?
        }
        Ok(())
    })?;
    println!();
    let elapsed = start.elapsed();
    println!(
        "generated {} frames in {:.1}s ({:.1} frames/s)",
        frames.len(),
        elapsed.as_secs_f64(),
        frames.len() as f64 / elapsed.as_secs_f64()
    );
    if frames.is_empty() {
        anyhow::bail!("no audio frames were generated")
    }
    let codes = model.frames_to_tensor(&frames)?;
    if let Some(codes_file) = &args.codes_file {
        let mut tensors = vec![("codes", codes.i(0)?)];
        if let Some(ref_codes) = &ref_codes {
            tensors.push(("ref_codes", ref_codes.clone()));
        }
        candle::safetensors::save(&tensors.into_iter().collect(), codes_file)?;
    }

    if let Some(streamer) = streamer {
        let (samples, first_audio) = streamer.finish()?;
        if let Some(first_audio) = first_audio {
            println!("first audio chunk after {first_audio:?}");
        }
        println!(
            "wrote {:.2}s of audio to {}",
            samples as f64 / sample_rate as f64,
            args.out_file
        );
        return Ok(());
    }

    let start = std::time::Instant::now();
    // In ICL mode the reference codes are decoded together with the generated ones and the
    // corresponding audio is then discarded.
    let (codes, skip_frames) = match &ref_codes {
        Some(ref_codes) => {
            let codes = Tensor::cat(&[&ref_codes.unsqueeze(0)?, &codes], 1)?;
            (codes, ref_codes.dim(0)?)
        }
        None => (codes, 0),
    };
    let pcm = speech_tokenizer.decode(&codes)?.i(0)?;
    let skip = skip_frames * speech_tokenizer.samples_per_frame();
    let pcm = pcm.narrow(0, skip, pcm.dim(0)? - skip)?.to_vec1::<f32>()?;
    println!(
        "decoded {:.2}s of audio in {:?}",
        pcm.len() as f64 / sample_rate as f64,
        start.elapsed()
    );
    println!("writing output file {}", args.out_file);
    let mut output = std::fs::File::create(&args.out_file)?;
    candle_examples::wav::write_pcm_as_wav(&mut output, &pcm, sample_rate)?;
    Ok(())
}
