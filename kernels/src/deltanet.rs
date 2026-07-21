pub fn gated_delta_net_decode(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: &[f32],
    beta: &[f32],
    state: &mut [f32],
    output: &mut [f32],
    head_v_dim: usize,
    num_v_heads: usize,
    num_k_heads: usize,
) {
    let head_k_dim = if num_k_heads > 0 { q.len() / num_k_heads } else { return };
    let s_v = head_v_dim;
    let scale = 1.0 / (s_v as f32).sqrt();
    let repeat = if num_k_heads > 0 { num_v_heads / num_k_heads } else { 1 };

    let mut delta = vec![0.0f32; s_v];

    for h in 0..num_v_heads {
        let kh = h / repeat;
        let q_h = &q[kh * head_k_dim..(kh + 1) * head_k_dim];
        let k_h = &k[kh * head_k_dim..(kh + 1) * head_k_dim];
        let v_h = &v[h * s_v..(h + 1) * s_v];
        let s_h = &mut state[h * s_v * s_v..(h + 1) * s_v * s_v];
        let o_h = &mut output[h * s_v..(h + 1) * s_v];
        let decay = gate[h].exp();
        let beta_val = beta[h];

        // State stored transposed: s_h[j*s_v + i] = S[i][j]
        // Decay
        for val in s_h.iter_mut() {
            *val *= decay;
        }

        // Delta: delta[j] = (v[j] - dot(S_col_j, k)) * beta
        for j in 0..s_v {
            let col = &s_h[j * s_v..(j + 1) * s_v];
            let mut dot = 0.0f32;
            for i in 0..head_k_dim.min(s_v) {
                dot += col[i] * k_h[i];
            }
            delta[j] = (v_h[j] - dot) * beta_val;
        }

        // Outer product: S[i][j] += k[i] * delta[j]
        for j in 0..s_v {
            let col = &mut s_h[j * s_v..(j + 1) * s_v];
            for i in 0..head_k_dim.min(s_v) {
                col[i] += k_h[i] * delta[j];
            }
        }

        // Output: o[j] = dot(S_col_j, q) * scale
        for j in 0..s_v {
            let col = &s_h[j * s_v..(j + 1) * s_v];
            let mut dot = 0.0f32;
            for i in 0..head_k_dim.min(s_v) {
                dot += col[i] * q_h[i];
            }
            o_h[j] = dot * scale;
        }
    }
}

pub fn l2_normalize(x: &mut [f32], eps: f32) {
    let norm_sq: f32 = x.iter().map(|v| v * v).sum();
    let inv_norm = 1.0 / (norm_sq + eps).sqrt();
    for v in x.iter_mut() {
        *v *= inv_norm;
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softplus(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

pub fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

pub fn conv1d_step(
    conv_state: &mut [f32],
    input: &[f32],
    kernel: &[f32],
    output: &mut [f32],
    conv_dim: usize,
    kernel_size: usize,
) {
    let history_len = kernel_size - 1;
    // Shift history left by one step
    if history_len > 1 {
        for ch in 0..conv_dim {
            for t in 0..history_len - 1 {
                conv_state[t * conv_dim + ch] = conv_state[(t + 1) * conv_dim + ch];
            }
        }
    }
    // Write current input as the newest history entry
    if history_len > 0 {
        let last = history_len - 1;
        conv_state[last * conv_dim..last * conv_dim + conv_dim]
            .copy_from_slice(&input[..conv_dim]);
    }

    // Depthwise conv: for each channel, dot(kernel, history+current)
    for ch in 0..conv_dim {
        let mut sum = 0.0f32;
        for t in 0..history_len {
            sum += conv_state[t * conv_dim + ch] * kernel[t * conv_dim + ch];
        }
        sum += input[ch] * kernel[history_len * conv_dim + ch];
        output[ch] = sum;
    }
}

pub fn gated_rmsnorm_silu(output: &[f32], gate: &[f32], norm_weight: &[f32], eps: f32, result: &mut [f32], head_dim: usize, num_heads: usize) {
    for h in 0..num_heads {
        let o = &output[h * head_dim..(h + 1) * head_dim];
        let g = &gate[h * head_dim..(h + 1) * head_dim];
        let r = &mut result[h * head_dim..(h + 1) * head_dim];
        let rms: f32 = o.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
        let inv_rms = 1.0 / (rms + eps).sqrt();
        for i in 0..head_dim {
            r[i] = o[i] * inv_rms * norm_weight[i] * silu(g[i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gdn_decode_zero_state() {
        let s_v = 4;
        let num_heads = 2;
        let q = vec![1.0; s_v * num_heads];
        let k = vec![1.0; s_v * num_heads];
        let v = vec![1.0; s_v * num_heads];
        let gate = vec![0.0; num_heads]; // exp(0) = 1, no decay
        let beta = vec![1.0; num_heads]; // full update
        let mut state = vec![0.0f32; s_v * s_v * num_heads];
        let mut output = vec![0.0f32; s_v * num_heads];

        gated_delta_net_decode(&q, &k, &v, &gate, &beta, &mut state, &mut output, s_v, num_heads, num_heads);

        // With zero initial state: delta = v * beta = [1,1,1,1]
        // S = k outer delta = ones(4x4)
        // output = S^T q * scale = [4,4,4,4] * 0.5
        let scale = 1.0 / (s_v as f32).sqrt();
        for &o in &output[..s_v] {
            assert!((o - s_v as f32 * scale).abs() < 1e-5, "got {o}");
        }
    }

    #[test]
    fn test_l2_normalize() {
        let mut x = vec![3.0, 4.0];
        l2_normalize(&mut x, 1e-6);
        assert!((x[0] - 0.6).abs() < 1e-5);
        assert!((x[1] - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_sigmoid_softplus() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!((softplus(0.0) - 0.6931).abs() < 1e-3);
    }
}
