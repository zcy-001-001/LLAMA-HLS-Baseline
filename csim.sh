export MODEL_BASE_PATH=$PWD
vitis-run --mode hls --csim --config ./llama2/hls_config.cfg --work_dir csim
