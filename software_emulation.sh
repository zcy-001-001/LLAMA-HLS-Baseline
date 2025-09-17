g++ -g -std=c++17 -Wall -O0 ./host/sw_emu_llama2.cpp \
    -o ./host/sw_emu_llama2.exe \
    -I$XILINX_XRT/include/ -L$XILINX_XRT/lib -lxrt_coreutil -pthread

emconfigutil --platform xilinx_u55c_gen3x16_xdma_3_202210_1

# ----------------------------------------------------
# 阶段一：创建工作目录 
# ----------------------------------------------------
mkdir sw_emu
cd sw_emu
mkdir initial_embedding_lookup_sw_emu
mkdir transformer_layer_pipeline_sw_emu
mkdir final_norm_classifier_sw_emu

# ----------------------------------------------------
# 阶段二：为软件仿真编译三个硬件核
# ----------------------------------------------------
v++ -c -t sw_emu --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k initial_embedding_lookup --config /home/CONNECT/czhang539/llama2-7B-Baseline/sw_emu/hls_config1.cfg -I../src ../src/independent_kernal/ini.cpp -o ./initial_embedding_lookup_sw_emu/initial_embedding_lookup.xo 

v++ -c -t sw_emu --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k transformer_layer_pipeline --config /home/CONNECT/czhang539/llama2-7B-Baseline/sw_emu/hls_config2.cfg -I../src ../src//independent_kernal/transformer.cpp -o ./transformer_layer_pipeline_sw_emu/transformer_layer_pipeline.xo 

v++ -c -t sw_emu --platform xilinx_u55c_gen3x16_xdma_3_202210_1 -k final_norm_classifier --config /home/CONNECT/czhang539/llama2-7B-Baseline/sw_emu/hls_config3.cfg -I../src ../src//independent_kernal/classifier.cpp -o ./final_norm_classifier_sw_emu/final_norm_classifier.xo 

# ----------------------------------------------------
# 阶段三：链接生成用于软件仿真的 xclbin 文件
# ----------------------------------------------------
v++ -l -t sw_emu --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../src/u55C1.cfg ./initial_embedding_lookup_sw_emu/initial_embedding_lookup.xo ./transformer_layer_pipeline_sw_emu/transformer_layer_pipeline.xo ./final_norm_classifier_sw_emu/final_norm_classifier.xo -o ./forward_sw_emu.xclbin

export XCL_EMULATION_MODE=sw_emu

cd ..
./host/sw_emu_llama2.exe \
    ./weights.bin \
    -z ./tokenizer.bin \
    -n 256 \
    -i "Once" \
    -k ./sw_emu/forward_sw_emu.xclbin 