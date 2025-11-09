#ifndef _MODEL_H_
#define _MODEL_H_

#include <genesis.h>

typedef struct {
  u32 in_features;
  u32 out_features;
  /* weights shape: [out_features, in_features] flattened */
  const fix32 *weights;
  /* bias shape: [out_features] */
  const fix32 *bias;
} Linear;

typedef struct {
  Linear fc1;
  Linear fc2;
  Linear fc3;
} MNISTModel;

inline fix32 fast_expf(fix32 x);

void relu(fix32 *data, u32 size);
void softmax(fix32 *x, u32 n);

void linear_forward(const Linear *layer, const fix32 *input, fix32 *output);

void init_layer(Linear *layer, const fix32 *weights, const fix32 *bias,
                u32 in_features, u32 out_features);

u8 mnist_forward(MNISTModel *model, const fix32 *input, fix32 *out);

#endif
