#include <genesis.h>

#include "debug.h"
#include "draw.h"
#include "engine.h"
#include "mnist.h"
#include "model.h"

MNISTModel model;
fix32 input[IMG_W * IMG_H];
u8 image_idx;

static void joy_cb(u16 joy, u16 changed, u16 state) {
  if (state & BUTTON_START) {
    fix32 logits[model.fc3.out_features];

    // Copy image data to RAM
    u16 start = image_idx * IMG_W * IMG_H;
    for (int i = 0; i < IMG_W * IMG_H; i++) {
      input[i] = images[start + i];
    }

    VDP_clearTextArea(0, 1, 24, 20);
    VDP_drawText("Loading...", 0, 24);
    u8 prediction = mnist_forward(&model, input, logits);
    VDP_clearText(0, 24, 10);

    draw_logits(logits, model.fc3.out_features);
    draw_prediction(prediction);

  } else if (state & BUTTON_RIGHT) {
    VDP_clearTextArea(0, 1, 24, 20);
    image_idx = clamp(image_idx + 1, 0, IMG_COUNT - 1);
    draw_input_image(images, IMG_W, IMG_H, image_idx);

  } else if (state & BUTTON_LEFT) {
    VDP_clearTextArea(0, 1, 24, 20);
    image_idx = clamp(image_idx - 1, 0, IMG_COUNT - 1);
    draw_input_image(images, IMG_W, IMG_H, image_idx);
  }
}

int main() {
  JOY_init();
  JOY_setEventHandler(joy_cb);

  image_idx = 0;
  draw_input_image(images, IMG_W, IMG_H, image_idx);

  while (true) {

    DEBUG_drawMenu();

    JOY_update();
    SYS_doVBlankProcess();
  }

  return 0;
}
