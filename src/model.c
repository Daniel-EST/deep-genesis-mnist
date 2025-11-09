#include "model.h"

#include <genesis.h>

#include "weights.h"

// Approximates e^x using (1 + x/256)^256 with bounds checking
inline fix32 fast_expf(fix32 x) {
  if (x < FIX32(-20.0f)) {
    return FIX32(0.0001f);
  }

  // Compute base term (1 + x / 256)
  fix32 base = FIX32(1.0f) + F32_div(x, FIX32(256.0f));

  // Prevent invalid base (negative or zero)
  if (base <= FIX32(0.0001f)) {
    return FIX32(0.0001f);
  }

  // Exponentiation by squaring: base^(256)
  // 8 squarings → 2^8 = 256
  fix32 result = base;
  for (int i = 0; i < 8; i++) {
    result = F32_mul(result, result);
  }

  return result;
}

void relu(fix32 *data, u32 size) {
  for (u32 i = 0; i < size; ++i) {
    if (data[i] < FIX32(0.0f)) {
      data[i] = FIX32(0.0f);
    }
  }
}

void softmax(fix32 *x, u32 n) {
  if (n == 0)
    return;

  // Find max value
  fix32 max_val = x[0];
  for (u32 i = 1; i < n; ++i) {
    if (x[i] > max_val)
      max_val = x[i];
  }

  // Exponentiate shifted values and accumulate sum
  fix32 sum = FIX32(0.0f);
  for (u32 i = 0; i < n; ++i) {
    x[i] = fast_expf(x[i] - max_val);
    // Ensure no negative values due to precision errors
    if (x[i] < FIX32(0.0f)) {
      x[i] = FIX32(0.0f);
    }
    sum += x[i];
  }

  // Normalization
  for (u32 i = 0; i < n; ++i) {
    x[i] = F32_div(x[i], sum);
  }
}

void linear_forward(const Linear *layer, const fix32 *input, fix32 *output) {
  for (u32 o = 0; o < layer->out_features; ++o) {
    fix32 sum = layer->bias[o];
    u32 base = o * layer->in_features;
    for (u32 i = 0; i < layer->in_features; ++i) {
      sum += F32_mul(layer->weights[base + i], input[i]);
    }
    output[o] = sum;
  }
}

void init_layer(Linear *layer, const fix32 *weights, const fix32 *bias,
                u32 in_features, u32 out_features) {
  layer->in_features = in_features;
  layer->out_features = out_features;
  layer->weights = weights;
  layer->bias = bias;
}

u8 mnist_forward(MNISTModel *model, const fix32 *input, fix32 *out) {
  fix32 *fc1_out = (fix32 *)MEM_alloc((sizeof(fix32) * FC1_OUT));
  fix32 *fc2_out = (fix32 *)MEM_alloc((sizeof(fix32) * FC2_OUT));

  // Layer 1
  init_layer(&model->fc1, fc1_weight_data, fc1_bias_data, FC1_IN, FC1_OUT);

  linear_forward(&model->fc1, input, fc1_out);
  relu(fc1_out, model->fc1.out_features);

  // Layer 2
  init_layer(&model->fc2, fc2_weight_data, fc2_bias_data, FC2_IN, FC2_OUT);

  linear_forward(&model->fc2, fc1_out, fc2_out);
  relu(fc2_out, model->fc2.out_features);

  MEM_free(fc1_out);

  // Output layer
  init_layer(&model->fc3, fc3_weight_data, fc3_bias_data, FC3_IN, FC3_OUT);
  linear_forward(&model->fc3, fc2_out, out);
  // softmax(out, model->fc3.out_features);

  MEM_free(fc2_out);

  // Argmax
  u8 best_idx = 0;
  fix32 best_val = out[0];
  for (u32 i = 1; i < model->fc3.out_features; ++i) {
    if (out[i] > best_val) {
      best_val = out[i];
      best_idx = (u8)i;
    }
  }

  return best_idx;
}
