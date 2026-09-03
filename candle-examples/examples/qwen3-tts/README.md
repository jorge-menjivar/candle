# Qwen3-TTS

[Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) is a family of text-to-speech models from the
Qwen team. A Qwen3 based *talker* generates 12.5 Hz frames of 16 codec tokens, the first one
directly and the other 15 through a small *code predictor* transformer. The tokens are then
turned into 24 kHz audio by the decoder of the Qwen3-TTS-Tokenizer-12Hz codec.

The following checkpoints are supported through the `--which` flag:

| `--which`           | model                                   | voice control                              |
|---------------------|-----------------------------------------|--------------------------------------------|
| `0.6b-custom-voice` | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice`   | `--speaker`                                |
| `1.7b-custom-voice` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice`   | `--speaker`, optional `--instruct`         |
| `1.7b-voice-design` | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign`   | `--instruct`                               |
| `0.6b-base`         | `Qwen/Qwen3-TTS-12Hz-0.6B-Base`          | voice cloning with `--ref-audio`           |
| `1.7b-base`         | `Qwen/Qwen3-TTS-12Hz-1.7B-Base`          | voice cloning with `--ref-audio`           |

## Running the example

```bash
# Predefined speaker, CustomVoice model.
cargo run --example qwen3-tts --release --features cuda -- \
    --which 0.6b-custom-voice --speaker ryan --language english \
    --text "Hello there, this is a test of text to speech with candle."

# List the speakers and languages of a model.
cargo run --example qwen3-tts --release --features cuda -- --which 0.6b-custom-voice --list-speakers

# Describe the voice, VoiceDesign model.
cargo run --example qwen3-tts --release --features cuda -- \
    --which 1.7b-voice-design --language english \
    --instruct "A calm, deep male voice, speaking slowly with a warm tone." \
    --text "The quick brown fox jumps over the lazy dog."

# Stream the audio out as it is generated.
cargo run --example qwen3-tts --release --features cuda -- \
    --which 0.6b-custom-voice --speaker ryan --language english --stream \
    --text "Hello there, this is a test of text to speech with candle."

# Clone the voice of a recording, Base model. The transcript of the recording is optional,
# when given the recording is also used as an in-context example which gives better results.
cargo run --example qwen3-tts --release --features cuda -- \
    --which 0.6b-base --language english \
    --ref-audio reference.wav --ref-text "The transcript of the reference recording." \
    --text "The quick brown fox jumps over the lazy dog."
```

`--stream` decodes and writes the audio while it is generated rather than at the end, which
brings the time to the first audio down from the whole generation to a fraction of a second.
Each chunk is decoded together with the frames that precede it, so the result is the same as a
single decode, at the cost of decoding those frames again. `--stream-chunk` sets how many frames
are decoded at a time and `--stream-context` how many precede them.

The generated audio is written to `out.wav` (`--out-file`). Use `--cpu` to run on CPU, the
model then runs in f32. The text is tokenized with the `Qwen/Qwen3-0.6B` tokenizer which shares
its vocabulary with the TTS models, pass `--tokenizer` to use another `tokenizer.json`.

The reference audio for voice cloning can be any PCM wav file, it is converted to mono and
resampled to 24 kHz. The Base models run the speaker encoder on it and, when `--ref-text` is
given, the speech tokenizer encoder to get the codec tokens of the in-context example.

The whole text is put in the prefix by default whereas voice cloning feeds it one token per
generated frame like the reference implementation, `--streaming-text` and
`--non-streaming-text` override this.

`--instruct` is only used by the 1.7B CustomVoice and VoiceDesign models, the reference
implementation ignores it for the 0.6B models and when cloning a voice.

Sampling can be tuned with `--temperature`, `--top-k`, `--top-p`, `--repetition-penalty` for the
first codebook and `--subtalker-temperature`, `--subtalker-top-k`, `--subtalker-top-p` for the
other ones, `--greedy` disables sampling altogether (as does a temperature of 0).
`--max-new-tokens` caps the number of generated frames, 8192 of them being about 11 minutes of
audio. `--codes-file` additionally saves the generated codec tokens as a safetensors file.
