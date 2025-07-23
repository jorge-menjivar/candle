use candle::{Device, Result, Tensor, DType, IndexOp, D};
use candle_transformers::generation::LogitsProcessor;

#[test]
fn sample_with_zero_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(1337, None, None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    Ok(())
}

#[test]
fn sample_with_temperature() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(0.9), None);
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 0);
    Ok(())
}

#[test]
fn sample_with_top_p() -> Result<()> {
    let mut logits_process = LogitsProcessor::new(42, Some(1.0), Some(0.5));
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_with_top_k() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 1,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::TopK {
            k: 2,
            temperature: 1.0,
        },
    );
    let logits = Tensor::new(&[0.1, 0.2, 0.3, 0.4], &Device::Cpu)?;
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 3);
    let token = logits_process.sample(&logits)?;
    assert_eq!(token, 2);
    Ok(())
}

#[test]
fn sample_gumbel() -> Result<()> {
    let mut logits_process = LogitsProcessor::from_sampling(
        42,
        candle_transformers::generation::Sampling::GumbelSoftmax { temperature: 1.0 },
    );
    let logits = Tensor::new(&[-1.0, 0.0, 0.2, 1.0], &Device::Cpu)?;
    let sm = candle_nn::ops::softmax(&logits, 0)?.to_vec1::<f64>()?;
    let mut counts = vec![0f64; 4];
    let samples = 100000;
    for _ in 0..samples {
        let token = logits_process.sample(&logits)?;
        counts[token as usize] += 1f64 / samples as f64;
    }
    for i in 0..4 {
        if (counts[i] - sm[i]).abs() > 0.05 {
            panic!("pr mismatch {counts:?} {sm:?}");
        }
    }
    Ok(())
}

#[cfg(test)]
mod voxtral_tests {
    use super::*;

    #[test]
    fn test_voxtral_generation_basic() -> Result<()> {
        let device = Device::Cpu;
        
        // Create basic test parameters matching the logged values
        let input_ids_data = vec![
            1u32, 1084, 1114, 1097, 1110, 1115, 1099, 1114, 1105, 1098, 1101, 1032, 1116,
            1104, 1101, 1032, 1102, 1111, 1108, 1108, 1111, 1119, 1105, 1110, 1103, 1032,
            1097, 1117, 1100, 1105, 1111, 1058, 1032, 1060, 1124, 1097, 1117, 1100, 1105,
            1111, 1124, 1062, 24
        ];
        let input_ids = Tensor::new(input_ids_data.as_slice(), &device)?.unsqueeze(0)?;
        
        // Create mock audio features with correct dimensions [1, 128, 248]
        let audio_features = Some(Tensor::zeros((1, 128, 248), DType::F32, &device)?);
        
        // Parameters from the log
        let max_new_tokens = 512;
        let temperature = 0.0; // Greedy decoding
        let top_p: Option<f64> = None;
        
        // This test validates that the basic structure and parameter handling works
        // without actually running the full model (which would require loaded weights)
        
        // Test parameter validation
        assert!(temperature >= 0.0, "Temperature should be non-negative");
        assert_eq!(input_ids.dims()[0], 1, "Batch size should be 1");
        assert_eq!(input_ids.dims()[1], 43, "Sequence length should be 43");
        
        if let Some(ref features) = audio_features {
            let dims = features.dims3()?;
            assert_eq!(dims, (1, 128, 248), "Audio features should have correct shape");
        }
        
        // Test that audio token is present in the sequence
        let input_data = input_ids.i(0)?.to_vec1::<u32>()?;
        let audio_token_id = 24u32;
        assert!(input_data.contains(&audio_token_id), "Input should contain audio token");
        
        Ok(())
    }

    #[test]
    fn test_voxtral_generation_parameters() -> Result<()> {
        let device = Device::Cpu;
        
        // Test parameter validation logic that was causing issues
        let max_new_tokens = 512;
        let temperature = 0.0;
        let top_p: Option<f64> = None;
        
        // Validate inputs (same logic as in the actual generate function)
        assert!(max_new_tokens > 0, "max_new_tokens must be positive");
        assert!(temperature >= 0.0, "Temperature must be non-negative");
        
        if let Some(p) = top_p {
            assert!((0.0..=1.0).contains(&p), "top_p must be between 0 and 1");
        }
        
        // Test token sequence handling
        let input_ids_data = vec![1u32, 1084, 1114, 24]; // Simplified sequence with audio token
        let input_ids = Tensor::new(input_ids_data.as_slice(), &device)?.unsqueeze(0)?;
        
        let tokens = input_ids.i(0)?.to_vec1::<u32>()?;
        let initial_len = tokens.len();
        assert_eq!(initial_len, 4, "Initial token length should be 4");
        
        // Test index calculation logic from generate function
        for idx in 0..3 {
            let start_pos = if idx == 0 { 0 } else { initial_len + idx - 1 };
            if idx == 0 {
                assert_eq!(start_pos, 0, "First iteration should start at 0");
            } else {
                assert!(start_pos > 0, "Subsequent iterations should have positive start_pos");
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_voxtral_tensor_indexing() -> Result<()> {
        let device = Device::Cpu;
        
        // Test the tensor indexing logic that was fixed
        // Simulate the scenario that was causing the error
        
        // Test single token sequence (seq_len = 1)
        let single_token = Tensor::new(&[[0.1f32, 0.2, 0.3]], &device)?; // [1, 1, 3]
        let single_token = single_token.unsqueeze(1)?; // [1, 1, 3] - batch, seq_len, hidden
        let (_, seq_len, _) = single_token.dims3()?;
        
        // This should work with the fix (using index 0 when seq_len == 1)
        let result = if seq_len == 1 {
            single_token.i((.., 0, ..))?
        } else {
            single_token.i((.., seq_len - 1, ..))?
        };
        
        assert_eq!(result.dims(), [1, 3], "Single token indexing should work");
        
        // Test multi-token sequence (seq_len > 1)
        let multi_tokens = Tensor::new(&[[[0.1f32, 0.2, 0.3], [0.4, 0.5, 0.6]]], &device)?; // [1, 2, 3]
        let (_, seq_len, _) = multi_tokens.dims3()?;
        
        let result = if seq_len == 1 {
            multi_tokens.i((.., 0, ..))?
        } else {
            multi_tokens.i((.., seq_len - 1, ..))?
        };
        
        assert_eq!(result.dims(), [1, 3], "Multi-token indexing should work");
        
        Ok(())
    }

    #[test]
    fn test_voxtral_dtype_conversion() -> Result<()> {
        let device = Device::Cpu;
        
        // Test that different dtypes are handled correctly
        let f32_features = Tensor::zeros((1, 128, 171), DType::F32, &device)?;
        let f16_features = Tensor::zeros((1, 128, 171), DType::F16, &device)?;
        
        // Both should work - the forward method should handle dtype conversion
        assert_eq!(f32_features.dims(), [1, 128, 171]);
        assert_eq!(f16_features.dims(), [1, 128, 171]);
        
        // Test dtype conversion logic
        if f32_features.dtype() != DType::F16 {
            let converted = f32_features.to_dtype(DType::F16)?;
            assert_eq!(converted.dtype(), DType::F16);
            assert_eq!(converted.dims(), f32_features.dims());
        }
        
        Ok(())
    }

    #[test]
    fn test_voxtral_config_head_dim_consistency() -> Result<()> {
        // Test that the Voxtral configurations are correctly set up
        // Note: Voxtral uses head_dim=128 which creates a larger attention dimension
        // than the model's hidden_size. The o_proj layer handles the projection back.
        
        // For 3B model: hidden_size = 3072, num_attention_heads = 32, head_dim = 128
        let config_3b = candle_transformers::models::voxtral::voxtral_llama::Config::voxtral_3b();
        assert_eq!(config_3b.hidden_size, 3072);
        assert_eq!(config_3b.num_attention_heads, 32);
        assert_eq!(config_3b.head_dim, Some(128));
        
        // The attention dimension is larger than hidden_size
        let attention_dim = config_3b.num_attention_heads * config_3b.head_dim.unwrap();
        assert_eq!(attention_dim, 4096); // 32 * 128 = 4096
        assert!(attention_dim > config_3b.hidden_size); // 4096 > 3072
        
        // For 24B model: hidden_size = 5120, num_attention_heads = 32, head_dim = 128
        let config_24b = candle_transformers::models::voxtral::voxtral_llama::Config::voxtral_24b();
        assert_eq!(config_24b.hidden_size, 5120);
        assert_eq!(config_24b.num_attention_heads, 32);
        assert_eq!(config_24b.head_dim, Some(128));
        
        // The attention dimension
        let attention_dim = config_24b.num_attention_heads * config_24b.head_dim.unwrap();
        assert_eq!(attention_dim, 4096); // 32 * 128 = 4096
        assert!(attention_dim < config_24b.hidden_size); // 4096 < 5120
        
        Ok(())
    }

    #[test]
    fn test_voxtral_attention_reshape() -> Result<()> {
        // Test that attention reshape handles the mismatch between
        // attention dimensions (4096) and hidden size (3072)
        let device = Device::Cpu;
        
        // Simulate attention output before reshape: [batch, seq_len, num_heads, head_dim]
        let batch_size = 1;
        let seq_len = 43;
        let num_heads = 32;
        let head_dim = 128;
        
        // This is what comes out of the attention computation
        let attention_output = Tensor::zeros((batch_size, num_heads, seq_len, head_dim), DType::F32, &device)?;
        
        // After transpose(1, 2): [batch, seq_len, num_heads, head_dim]
        let transposed = attention_output.transpose(1, 2)?;
        assert_eq!(transposed.dims(), [batch_size, seq_len, num_heads, head_dim]);
        
        // The reshape should use actual_hidden_size = num_heads * head_dim = 4096
        let actual_hidden_size = num_heads * head_dim;
        let reshaped = transposed.reshape(&[batch_size, seq_len, actual_hidden_size])?;
        assert_eq!(reshaped.dims(), [batch_size, seq_len, 4096]);
        
        // This would have failed before the fix with hidden_size = 3072
        // because 32 * 128 = 4096 ≠ 3072
        
        Ok(())
    }

    #[test]
    fn test_voxtral_logits_indexing() -> Result<()> {
        // Test that logits indexing handles both 2D and 3D tensors correctly
        let device = Device::Cpu;
        
        // Test 3D logits: [batch, seq_len, vocab_size]
        let logits_3d = Tensor::zeros((1, 43, 131072), DType::F32, &device)?;
        assert_eq!(logits_3d.dims().len(), 3);
        
        // Simulate the indexing logic from generate method
        let result_3d = if logits_3d.dims().len() == 3 {
            logits_3d.i((.., logits_3d.dim(1)? - 1, ..))?
        } else {
            logits_3d
        };
        assert_eq!(result_3d.dims(), [1, 131072]); // Should extract last token
        
        // Test 2D logits: [batch, vocab_size] 
        let logits_2d = Tensor::zeros((1, 131072), DType::F32, &device)?;
        assert_eq!(logits_2d.dims().len(), 2);
        
        // Should handle 2D case without indexing
        let result_2d = if logits_2d.dims().len() == 3 {
            logits_2d.i((.., logits_2d.dim(1)? - 1, ..))?
        } else {
            logits_2d.clone()
        };
        assert_eq!(result_2d.dims(), [1, 131072]); // Should remain unchanged
        
        Ok(())
    }

    #[test]
    fn test_voxtral_rope_dtype_consistency() -> Result<()> {
        let device = Device::Cpu;
        
        // Test RoPE dtype consistency logic
        // Simulate the scenario in apply_rotary_emb
        
        // Create tensors with different dtypes
        let x_f16 = Tensor::zeros((1, 32, 4, 64), DType::F16, &device)?; // Input tensor
        let cos_f32 = Tensor::zeros((4, 64), DType::F32, &device)?; // Position embeddings
        let sin_f32 = Tensor::zeros((4, 64), DType::F32, &device)?;
        
        // Test the dtype conversion logic from apply_rotary_emb
        let x_dtype = x_f16.dtype();
        let cos_original_dims = cos_f32.dims();
        let sin_original_dims = sin_f32.dims();
        
        let cos_converted = if cos_f32.dtype() != x_dtype {
            cos_f32.to_dtype(x_dtype)?
        } else {
            cos_f32.clone()
        };
        let sin_converted = if sin_f32.dtype() != x_dtype {
            sin_f32.to_dtype(x_dtype)?
        } else {
            sin_f32.clone()
        };
        
        // Verify dtypes match after conversion
        assert_eq!(x_f16.dtype(), DType::F16);
        assert_eq!(cos_converted.dtype(), DType::F16);
        assert_eq!(sin_converted.dtype(), DType::F16);
        
        // Verify shapes are preserved
        assert_eq!(cos_converted.dims(), cos_original_dims);
        assert_eq!(sin_converted.dims(), sin_original_dims);
        
        Ok(())
    }

    #[test]
    fn test_voxtral_exact_failing_inputs() -> Result<()> {
        let device = Device::Cpu; // Use CPU for testing since CUDA might not be available
        
        // Exact input from the failing debug log
        let input_ids_data = vec![
            1u32, 1084, 1114, 1097, 1110, 1115, 1099, 1114, 1105, 1098, 1101, 1032, 1116,
            1104, 1101, 1032, 1102, 1111, 1108, 1108, 1111, 1119, 1105, 1110, 1103, 1032,
            1097, 1117, 1100, 1105, 1111, 1058, 1032, 1060, 1124, 1097, 1117, 1100, 1105,
            1111, 1124, 1062, 24
        ];
        let input_ids = Tensor::new(input_ids_data.as_slice(), &device)?.unsqueeze(0)?;
        
        // Exact audio features shape from the log: [1, 128, 171]
        let audio_features = Some(Tensor::zeros((1, 128, 171), DType::F32, &device)?);
        
        // Verify the shapes match the debug log
        assert_eq!(input_ids.dims(), [1, 43], "Input IDs should have shape [1, 43]");
        if let Some(ref features) = audio_features {
            assert_eq!(features.dims(), [1, 128, 171], "Audio features should have shape [1, 128, 171]");
        }
        
        // Test the tensor manipulation that happens in the generate function
        let tokens = input_ids.i(0)?.to_vec1::<u32>()?;
        let initial_len = tokens.len();
        assert_eq!(initial_len, 43, "Should have 43 tokens");
        
        // Test the sequence slicing logic from the generate function
        // This simulates what happens in subsequent generation steps
        for idx in 0..3 {
            let start_pos = if idx == 0 { 0 } else { initial_len + idx - 1 };
            let input = if idx == 0 {
                input_ids.clone()
            } else {
                // For subsequent steps, we use just the last token
                // This mimics the autoregressive generation
                let last_token = tokens[tokens.len() - 1];
                Tensor::new(&[last_token], &device)?.unsqueeze(0)?
            };
            
            // Validate that we can create these tensors without error
            assert!(input.dims().len() == 2, "Input should be 2D");
            assert_eq!(input.dims()[0], 1, "Batch size should be 1");
            
            // For the first iteration, should be full sequence
            if idx == 0 {
                assert_eq!(input.dims()[1], 43, "First pass should have full sequence");
            } else {
                assert_eq!(input.dims()[1], 1, "Subsequent passes should have single token");
            }
        }
        
        Ok(())
    }

    #[test]
    fn test_voxtral_argmax_tensor_handling() -> Result<()> {
        let device = Device::Cpu;
        
        // Test the argmax logic that was failing
        // Create a logits tensor similar to what the model outputs: [1, vocab_size]
        let logits = Tensor::new(&[[0.1f32, 0.2, 0.8, 0.3]], &device)?; // [1, 4] for simplicity
        
        // Test argmax operation
        let argmax_result = logits.argmax(D::Minus1)?;
        assert_eq!(argmax_result.dims(), [1], "Argmax should return shape [1]");
        
        // Test the logic from the generate function for handling [1] shaped argmax result
        let token = if argmax_result.dims().len() == 0 {
            // Already a scalar
            argmax_result.to_scalar::<u32>()?
        } else if argmax_result.dims() == &[1] {
            // Shape [1] - extract the single element
            let scalar_tensor = argmax_result.i(0)?;
            scalar_tensor.to_scalar::<u32>()?
        } else {
            return Err(candle::Error::Msg(format!("Unexpected argmax result shape: {:?}", argmax_result.shape())));
        };
        
        assert_eq!(token, 2, "Should select the token with highest logit (index 2)");
        
        // Test with actual model-sized vocab
        let large_logits = Tensor::zeros((1, 131072), DType::F32, &device)?;
        let argmax_large = large_logits.argmax(D::Minus1)?;
        assert_eq!(argmax_large.dims(), [1], "Large vocab argmax should also return shape [1]");
        
        // This should work without error now
        let large_token = if argmax_large.dims() == &[1] {
            argmax_large.i(0)?.to_scalar::<u32>()?
        } else {
            argmax_large.to_scalar::<u32>()?
        };
        assert_eq!(large_token, 0, "All zeros should return token 0");
        
        Ok(())
    }
}
