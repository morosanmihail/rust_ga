build:
    cargo build --release

run-image:
    cargo run --release --bin image_approximation

gif output="evolution.gif":
    ffmpeg -framerate 10 -pattern_type glob -i 'image_output/gen_*.png' -vf scale=512:512 {{output}}
