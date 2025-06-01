#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess

def get_available_models():
    return ['bge', 'e5', 'gte']

def run_pretrained_model(model_name):
    script_path = f"LM_pretrained/pretrained_{model_name}.py"
    if not os.path.exists(script_path):
        print(f"Error: {script_path} not found")
        return False
    
    print(f"Running pretrained {model_name.upper()} model...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, script_path], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running pretrained model: {e}")
        return False

def run_finetune_model(model_name):
    module_path = f"LM_finetune.finetune_{model_name}"
    
    print(f"Running fine-tuned {model_name.upper()} model...")
    print("=" * 60)
    
    try:
        result = subprocess.run([sys.executable, "-m", module_path], cwd=".")
        return result.returncode == 0
    except Exception as e:
        print(f"Error running finetune model: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run language model with optional fine-tuning")
    parser.add_argument("model", choices=get_available_models(), 
                       help="Language model to use: bge, e5, gte")
    parser.add_argument("--finetune", action="store_true", 
                       help="Use fine-tuned model instead of pretrained")
    
    args = parser.parse_args()
    
    model_name = args.model
    is_finetune = args.finetune
    
    mode_str = "fine-tuned" if is_finetune else "pretrained"
    print(f"Starting {model_name.upper()} model ({mode_str})...")
    
    if is_finetune:
        success = run_finetune_model(model_name)
    else:
        success = run_pretrained_model(model_name)
    
    print("=" * 60)
    if success:
        print(f"{model_name.upper()} model completed successfully!")
    else:
        print(f"{model_name.upper()} model failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 