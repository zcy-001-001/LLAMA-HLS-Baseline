mkdir synth
cd synth

mkdir initial_embedding_lookup
mkdir transformer_layer_pipeline
mkdir final_norm_classifier

v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config1.cfg --work_dir ./initial_embedding_lookup
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config2.cfg --work_dir ./transformer_layer_pipeline
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config3.cfg --work_dir ./final_norm_classifier