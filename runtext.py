script = ""
script += "QUANT_CONFIG=./quantization_config/maxabs_quant.json \\\n"
# script += "QUANT_CONFIG=./quantization_config/maxabs_measure.json \\\n"
script += "python run_generation.py \\\n"
# script += "python run_lm_eval.py \\\n"
script += "--model_name_or_path /model_weights/meta-llama/Llama-3.1-8B-Instruct/ \\\n"

# for run_generation.py
# script += "--output_dir ./logs/run_generation/fp8_measure \\\n"
# for run_lm_eval.py
# script += "--output_file ./logs/eval/fp8_profile.json \\\n"
script += "--bf16 \\\n"

script += "--book_source \\\n"
script += "--max_input_tokens 100 \\\n"
script += "--max_new_tokens 100 \\\n"

script += "--batch_size 16 \\\n"
script += "--use_kv_cache \\\n"
# script += "--reuse_cache \\\n"
# script += "--n_iterations 10 \\\n"
script += "--use_hpu_graphs \\\n"
script += "--use_flash_attention \\\n"



script += "--device hpu"
# script += " >./logs/run_generation/fp8_measure.txt 2>&1"

print(script)