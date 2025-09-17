mkdir hw_end1
cd hw_end1

mkdir initial_embedding_lookup
mkdir transformer_layer_pipeline
mkdir final_norm_classifier
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config1.cfg --work_dir ./initial_embedding_lookup
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config2.cfg --work_dir ./transformer_layer_pipeline
v++ -c --mode hls --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../llama2/hls_config3.cfg --work_dir ./final_norm_classifier
v++ -l -t hw --platform xilinx_u55c_gen3x16_xdma_3_202210_1 --config ../src/u55C1.cfg --vivado.impl.strategies Congestion_SpreadLogic_high --vivado.synth.jobs 256 --vivado.impl.jobs 256 ./initial_embedding_lookup/initial_embedding_lookup.xo ./transformer_layer_pipeline/transformer_layer_pipeline.xo ./final_norm_classifier/final_norm_classifier.xo -o ./forward.xclbin