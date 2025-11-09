build *args:
    @echo "CompileFlags:\n  Add:\n    - -xc\n    - -I${SGDK_PATH}/inc\n    - -I../inc\n    - -I../res" > .clangd
    @make GDK=$SGDK_PATH -f $SGDK_PATH/makefile_wine.gen {{args}}


play:
    "$BLASTEM_PATH/blastem" ./out/rom.bin

train *args:
    @uv run python train.py {{args}}
    @uv run python convert_weights.py

predict *args:
    @uv run python predict.py {{args}}

convert-images *args:
    @uv run python convert_images.py {{args}}

convert-weights *args:
    @uv run python convert_weights.py {{args}}

clean:
    @make GDK=$SGDK_PATH -f $SGDK_PATH/makefile_wine.gen clean

help:
    @just --list
