g++ -g -std=c++17 -Wall -O0 ./host/hw_emu_llama2.cpp \
    -o ./host/hw_emu_llama2.exe \
    -I$XILINX_XRT/include/ -L$XILINX_XRT/lib -lxrt_coreutil -pthread

emconfigutil --platform xilinx_u55c_gen3x16_xdma_3_202210_1

mkdir hw_emu
cd hw_emu

mkdir initial_embedding_lookup
mkdir transformer_layer_pipeline
mkdir final_norm_classifier
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config1.cfg --work_dir ./initial_embedding_lookup
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config2.cfg --work_dir ./transformer_layer_pipeline
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config3.cfg --work_dir ./final_norm_classifier

v++ -l -t hw_emu --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../src/u55C1.cfg h ./initial_embedding_lookup/initial_embedding_lookup.xo ./transformer_layer_pipeline/transformer_layer_pipeline.xo ./final_norm_classifier/final_norm_classifier.xo -o ./forward_hw_emu.xclbin

export XCL_EMULATION_MODE=hw_emu

cd ..
./host/hw_emu_llama2.exe \
    ./weights.bin \
    -z ./tokenizer.bin \
    -n 256 \
    -i "Once" \
    -k ./hw_emu/forward_hw_emu.xclbin