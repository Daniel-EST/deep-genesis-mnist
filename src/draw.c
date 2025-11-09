#include "draw.h"

#include <genesis.h>

void draw_input_image(const fix32 *input_image, u8 width, u8 height, u8 idx) {
  u8 padding = 10;
  u8 color_idx;

  u16 start = idx * width * height;
  for (u16 i = 0; i < height; ++i) {
    for (u16 j = 0; j < width; ++j) {
      u16 index = start + i * width + j;
      if (input_image[index] > FIX32(0.75)) {
        color_idx = WHITE;
      } else if (input_image[index] > FIX32(0.5)) {
        color_idx = LIGHT_GRAY;
      } else if (input_image[index] > FIX32(0.25)) {
        color_idx = DARK_GRAY;
      } else {
        color_idx = BLACK;
      }

      VDP_setTileMapXY(BG_B,
                       TILE_ATTR_FULL(PAL0, false, false, false, color_idx),
                       j + padding, i);
    }
  }
}

void draw_logits(const fix32 *logits, u8 size) {
  char buffer[32];

  VDP_drawText("Logits:", 0, 1);
  for (u16 i = 0; i < size; ++i) {
    sprintf(buffer, "%d: ", i);
    VDP_drawText(buffer, 1, i + 2);

    fix32ToStr(logits[i], buffer, 3);
    VDP_drawText(buffer, 4, i + 2);
  }
}

void draw_prediction(const u8 prediction) {
  char buffer[32];
  sprintf(buffer, "Prediction: %d", prediction);
  VDP_drawText(buffer, 0, 14);
}
