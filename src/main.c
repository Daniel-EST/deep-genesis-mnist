#include <genesis.h>

#include "debug.h"

int main() {

  while (true) {

    DEBUG_drawMenu();

    SYS_doVBlankProcess();
  }

  return 0;
}
