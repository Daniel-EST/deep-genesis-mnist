#ifndef _DEBUG_H_
#define _DEBUG_H_

#include <genesis.h>

#include "engine.h"

inline void DEBUG_drawText(char *text, Vect2D_u16 pos) {
  VDP_drawText(text, pos.x, pos.y);
}

inline void DEBUG_drawMemory(Vect2D_u16 pos) {
  char buffer[25];
  sprintf(buffer, "Memory: %d", MEM_getAllocated());
  VDP_drawText(buffer, pos.x, pos.y);
}

inline void DEBUG_drawTotalMemory(Vect2D_u16 pos) {
  char buffer[25];
  u16 total_memory = MEM_getAllocated() + MEM_getFree();
  sprintf(buffer, "Total Memory: %d", total_memory);
  VDP_drawText(buffer, pos.x, pos.y);
}

inline void DEBUG_drawFPS(Vect2D_u16 pos) { VDP_showFPS(false, pos.x, pos.y); }

inline void DEBUG_drawVersion(Vect2D_u16 pos) {
  char version[20];
  strcpy(version, VERSION);
  VDP_drawText(version, pos.x, pos.y);
}

inline void DEBUG_drawMenu() {
  DEBUG_drawMemory((Vect2D_u16){0, 26});
  DEBUG_drawTotalMemory((Vect2D_u16){0, 27});
  DEBUG_drawFPS((Vect2D_u16){38, 0});
  DEBUG_drawVersion((Vect2D_u16){36, 27});
}

#endif
