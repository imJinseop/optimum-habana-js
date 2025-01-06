import os
model_pair = {"gpt2": "gpt2", "bloom": "bigscience/bloom", "starcoder": "bigcode/starcoder", "gpt-j": "EleutherAI/gpt-j-6b", "stableLM": "stabilityai/stablelm-2-1_6b", "mistral": "mistralai/Mistral-7B-Instruct-v0.3", "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1"}
# model_pair = {"gpt2": "gpt2"}

def make_script(model, eval, type):

    script = ""
    if (type == "fp8"):
        script += "QUANT_CONFIG=./quantization_config/maxabs_quant.json \\\n"
    elif (type == "fp8_measure"):
    # script += "QUANT_CONFIG=./quantization_config/maxabs_quant_e5m2.json \\\n"
        script += "QUANT_CONFIG=./quantization_config/maxabs_measure.json \\\n"
    # script += "QUANT_CONFIG=./quantization_config/maxabs_measure_e5m2.json \\\n"
    elif (type == "bf16"):
        bf16 = True
    elif (type == "fp32"):
        bf16 = False
    else:
        assert(0)
    
    if (eval):
        script += "python run_lm_eval.py \\\n"
        # for run_lm_eval.py
        script += f"--output_file ./logs/{model}/eval/{type}.json \\\n"
    else:
        script += "python run_generation.py \\\n"
    # script += "--model_name_or_path /model_weights/meta-llama/Llama-3.1-8B-Instruct/ \\\n"
    script += f"--model_name_or_path {model_pair[model]} \\\n"

    # for run_generation.py
    # script += "--output_dir ./logs/run_generation/fp8_measure \\\n"

    if (type != "fp32"):
        script += "--bf16 \\\n"

    if (not eval):
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
    if (not eval):
        script += f" >./logs/{model}/run_generation/{type}.txt 2>&1"

    # print(script)
    return script


# types = ["fp8_measure", "fp8"]
types = ["fp32", "bf16", "fp8_measure", "fp8"]
for model in model_pair.keys():
    for eval in [False, True]:
        os.makedirs(f"./logs/{model}/run_geneartion")
        os.makedirs(f"./logs/{model}/eval")
        for type in types:
            script = make_script(model, eval, type)
            print(script)
            os.system(script)
        
