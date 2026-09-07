"""
Project-4o GGUF Conversion Script
Converts the merged model to GGUF format for llama.cpp / Ollama / LM Studio.

Prerequisites:
    1. Clone llama.cpp: git clone https://github.com/ggerganov/llama.cpp
    2. Install requirements: pip install -r llama.cpp/requirements.txt
    3. Build llama.cpp quantizer: cd llama.cpp && make

Usage:
    python convert_gguf.py --config config.yaml
    python convert_gguf.py --config config.yaml --model ./merged_model --quantize q5_k_m
    python convert_gguf.py --config config.yaml --llama_cpp_path ./llama.cpp
"""

import argparse
import os
import subprocess
import sys

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def find_llama_cpp():
    """Try to find llama.cpp directory."""
    candidates = [
        "./llama.cpp",
        "../llama.cpp",
        os.path.expanduser("~/llama.cpp"),
        "/opt/llama.cpp",
    ]
    for path in candidates:
        convert_script = os.path.join(path, "convert_hf_to_gguf.py")
        if os.path.exists(convert_script):
            return os.path.abspath(path)
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert merged model to GGUF")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--model", type=str, default=None, help="Path to merged HF model")
    parser.add_argument("--quantize", type=str, default=None, help="Quantization type (q4_k_m, q5_k_m, q8_0, f16)")
    parser.add_argument("--llama_cpp_path", type=str, default=None, help="Path to llama.cpp directory")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = args.model or cfg["merge"]["output_dir"]
    output_dir = cfg["gguf"]["output_dir"]
    quantize_type = args.quantize or cfg["gguf"]["quantize"]

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("Run merge.py first, or specify --model <path>")
        return

    # Find llama.cpp
    llama_cpp = args.llama_cpp_path or find_llama_cpp()
    if llama_cpp is None:
        print("ERROR: llama.cpp not found.")
        print("Options:")
        print("  1. Clone it: git clone https://github.com/ggerganov/llama.cpp")
        print("  2. Specify path: python convert_gguf.py --llama_cpp_path /path/to/llama.cpp")
        sys.exit(1)

    convert_script = os.path.join(llama_cpp, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        convert_script = os.path.join(llama_cpp, "convert.py")

    if not os.path.exists(convert_script):
        print(f"ERROR: Conversion script not found in {llama_cpp}")
        print("Make sure llama.cpp is up to date.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert to unquantized GGUF (f16)
    model_name = os.path.basename(os.path.normpath(model_path))
    f16_gguf = os.path.join(output_dir, f"{model_name}-f16.gguf")

    print("=" * 60)
    print("Project-4o GGUF Conversion")
    print(f"Model:     {model_path}")
    print(f"llama.cpp: {llama_cpp}")
    print(f"Output:    {output_dir}")
    print(f"Quantize:  {quantize_type}")
    print("=" * 60)

    print(f"\n[1/2] Converting to F16 GGUF: {f16_gguf}")
    cmd = [
        sys.executable,
        convert_script,
        model_path,
        "--outfile", f16_gguf,
        "--outtype", "f16",
    ]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=llama_cpp)
    if result.returncode != 0:
        print("ERROR: Conversion to GGUF failed.")
        sys.exit(1)

    # Step 2: Quantize
    if quantize_type and quantize_type != "f16":
        quantized_gguf = os.path.join(output_dir, f"{model_name}-{quantize_type}.gguf")
        quantize_bin = os.path.join(llama_cpp, "llama-quantize")
        if not os.path.exists(quantize_bin):
            quantize_bin = os.path.join(llama_cpp, "build", "bin", "llama-quantize")
        if not os.path.exists(quantize_bin):
            print(f"WARNING: llama-quantize binary not found. F16 GGUF is at {f16_gguf}")
            print("Build llama.cpp to enable quantization.")
            return

        print(f"\n[2/2] Quantizing to {quantize_type}: {quantized_gguf}")
        cmd = [quantize_bin, f16_gguf, quantized_gguf, quantize_type]
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print("ERROR: Quantization failed.")
            print(f"F16 GGUF is still available at {f16_gguf}")
            sys.exit(1)

        if os.path.exists(quantized_gguf):
            os.remove(f16_gguf)
            print(f"\nDone! GGUF saved to {quantized_gguf}")
        else:
            print(f"\nDone! F16 GGUF saved to {f16_gguf}")
    else:
        print(f"\nDone! GGUF saved to {f16_gguf}")

    print("\nYou can now use the model with:")
    final_path = quantized_gguf if quantize_type != "f16" else f16_gguf
    print(f"  llama-cli -m {final_path} -p 'Hello!'")


if __name__ == "__main__":
    main()
