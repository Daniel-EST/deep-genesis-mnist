#ifndef _DRAW_H_
#define _DRAW_H_

#include <genesis.h>

#define BLACK 0
#define DARK_GRAY 3
#define LIGHT_GRAY 6
#define WHITE 9

void draw_input_image(const fix32 *input_image, u8 width, u8 height, u8 idx);
void draw_logits(const fix32 *logits, u8 size);
void draw_prediction(const u8 prediction);

#endif
