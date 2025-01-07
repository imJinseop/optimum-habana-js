import os
import re
import json

def parse_txt(file_path):
    """Parse the .txt file to extract throughput and max memory."""
    with open(file_path, 'r') as f:
        content = f.read()

    # Regex patterns
    throughput_pattern = r"Throughput \(including tokenization\) = ([\d.]+) tokens/second"
    max_mem_pattern = r"Max memory allocated\s+= ([\d.]+) GB"

    throughput = re.search(throughput_pattern, content)
    max_mem = re.search(max_mem_pattern, content)

    return {
        "throughput": float(throughput.group(1)) if throughput else None,
        "max_mem": float(max_mem.group(1)) if max_mem else None,
    }

def parse_json(file_path):
    """Parse the .json file to extract benchmark results."""
    with open(file_path, 'r') as f:
        data = json.load(f)

    results = data.get("results", {})
    benchmarks = {}
    for task, metrics in results.items():
        if "acc" in metrics:
            benchmarks[f"{task}_acc"] = metrics["acc"]
        if "ppl" in metrics:
            benchmarks[f"{task}_ppl"] = metrics["ppl"]

    return benchmarks

def process_logs(output_csv, model_names):
    """Process all logs and save to a CSV file."""
    rows = []

    for model_name in model_names:
        row = [model_name]  # Start row with the model name
        for dtype in ["bf16", "fp8"]:
            generation_file = os.path.join(model_name, "run_generation", f"{dtype}.txt")
            eval_file = os.path.join(model_name, "eval", f"{dtype}.json")

            gen_data = parse_txt(generation_file) if os.path.exists(generation_file) else {}
            eval_data = parse_json(eval_file) if os.path.exists(eval_file) else {}

            row.extend([
                gen_data.get("throughput", ""),
                gen_data.get("max_mem", ""),
                eval_data.get("hellaswag_acc", ""),
                eval_data.get("lambada_openai_acc", ""),
                eval_data.get("piqa_acc", ""),
                eval_data.get("winogrande_acc", "")
            ])

        rows.append(row)

    # Static field order
    header = [
        "model",
        "throughput (bf16)", "max_mem (bf16)", "hellaswag_acc (bf16)", "lambada_openai_acc (bf16)", "piqa_acc (bf16)", "winogrande_acc (bf16)",
        "throughput (fp8)", "max_mem (fp8)", "hellaswag_acc (fp8)", "lambada_openai_acc (fp8)", "piqa_acc (fp8)", "winogrande_acc (fp8)"
    ]

    # Write to CSV manually
    with open(output_csv, "w", newline="") as csvfile:
        csvfile.write(",".join(header) + "\n")  # Write header
        for row in rows:
            row_values = [str(value) for value in row]
            csvfile.write(",".join(row_values) + "\n")

if __name__ == "__main__":
    model_names = ["gpt2", "bloom-7b", "codegen-6b", "gpt-j-6b", "mistral-7b", "mixtral-46b", "stableLM-6b", "starcoder-16b"]
    # Output CSV file
    output_csv_file = "model_stats.csv"

    # Process logs and save to CSV
    process_logs(output_csv_file, model_names)
    print(f"Parsed data saved to {output_csv_file}")
