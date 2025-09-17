g++ -g -std=c++17 -Wall -O0 ./host/llama2.cpp \
    -o ./host/llama2.exe \
    -I$XILINX_XRT/include/ -L$XILINX_XRT/lib -lxrt_coreutil -pthread

./host/llama2.exe \
    ./weights.bin \
    -z ./tokenizer.bin \
    -n 256 \
    -i "Once" \
    -k ./hw_end1/forward.xclbin