import os
import argparse

model_pair = {
    # "gpt2": "gpt2", 
    # "bloom-7b": "bigscience/bloom-7b1", 
    # "starcoder-16b": "bigcode/starcoder", 
    # "gpt-j-6b": "EleutherAI/gpt-j-6b", 
    # "stableLM-6b": "stabilityai/stablelm-2-1_6b", 
    # "mistral-7b": "mistralai/Mistral-7B-Instruct-v0.3", 
    "mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1", 
    # "codegen-6b": "Salesforce/codegen-6B-multi",
    "llama-3.1-8b": "/model_weights/meta-llama/Llama-3.1-8B-Instruct/",
    "phi-2": "microsoft/phi-2",
    # "gemma-2-9b": "google/gemma-2-9b",
    "falcon-7b": "tiiuae/falcon-7b",
    "gemma-7b": "google/gemma-7b"
}
# model_pair = {"gpt2": "gpt2"}

def make_script(model, runtype, dtype):

    script = ""
    if (dtype == "fp8"):
        quant_tail = "_mixtral" if "mixtral" in model else "_phi" if "phi" in model else "_gemma" if "gemma" in model else ""
        script += f"QUANT_CONFIG=./quantization_config/maxabs_quant{quant_tail}.json \\\n"
    elif (dtype == "fp8_measure"):
    # script += "QUANT_CONFIG=./quantization_config/maxabs_quant_e5m2.json \\\n"
        script += "QUANT_CONFIG=./quantization_config/maxabs_measure.json \\\n"
    # script += "QUANT_CONFIG=./quantization_config/maxabs_measure_e5m2.json \\\n"
    elif (dtype == "bf16"):
        bf16 = True
    elif (dtype == "fp32"):
        bf16 = False
    else:
        assert(0)
    
    if (runtype == "eval"):
        script += "python run_lm_eval.py \\\n"
        # for run_lm_eval.py
        script += f"--output_file ./logs/{model}/eval/{dtype}.json \\\n"
    else:
        script += "python run_generation.py \\\n"
    # script += "--model_name_or_path /model_weights/meta-llama/Llama-3.1-8B-Instruct/ \\\n"
    script += f"--model_name_or_path {model_pair[model]} \\\n"

    # for run_generation.py
    # script += "--output_dir ./logs/run_generation/fp8_measure \\\n"

    if (dtype != "fp32"):
        script += "--bf16 \\\n"

    if (runtype == "run"):
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
    if (runtype == "run"):
        script += f" >./logs/{model}/run_generation/{dtype}.txt 2>&1"

    # print(script)
    return script



def main(model, runtype, dtype):
    models = model_pair.keys() if model == None else model.split(',')
    runtypes = ["run", "eval"] if runtype == None else [runtype]
    dtypes = ["fp32", "bf16", "fp8_measure", "fp8"] if dtype == None else dtype.split(",")
    for model in models:
        for runtype in runtypes:
            if (runtype == "run"):
                os.system(f"mkdir -p ./logs/{model}/run_generation")
            else:
                os.system(f"mkdir -p ./logs/{model}/eval")
            for dtype in dtypes:
                script = make_script(model, runtype, dtype)
                print(script)
                os.system(script)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--runtype", type=str, default=None)
    parser.add_argument("--dtype", type=str, default=None)
    args = parser.parse_args()
    assert(args.model == None or set(args.model.split(',')).issubset(set(model_pair.keys())))
    assert(args.runtype == None or args.runtype in ["run", "eval"])
    assert(args.dtype == None or set(args.dtype.split(',')).issubset(set(["fp32", "bf16", "fp8_measure", "fp8"])))
    # return None
    main(args.model, args.runtype, args.dtype)
