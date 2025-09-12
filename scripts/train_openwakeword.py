import argparse, os, subprocess, json, tensorflow as tf, yaml

def estimate_arena(path):
    with open(path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    try:
        return interp._get_arena_used_bytes()
    except:
        return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake_word", required=True)
    args = parser.parse_args()

    wake_word = args.wake_word.replace(" ", "_")
    config_path = "my_model.yaml"

    # Load base config
    base_cfg = yaml.safe_load(open("openwakeword/examples/custom_model.yml"))
    base_cfg["target_phrase"] = [args.wake_word]
    base_cfg["model_name"] = wake_word
    base_cfg["n_samples"] = 1000
    base_cfg["n_samples_val"] = 500
    base_cfg["steps"] = 10000
    base_cfg["output_dir"] = "./my_custom_model"
    base_cfg["background_paths"] = ["./audioset_16k", "./fma"]
    base_cfg["false_positive_validation_data_path"] = "validation_set_features.npy"
    base_cfg["feature_data_files"] = {
        "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    }

    with open(config_path, "w") as f:
        yaml.dump(base_cfg, f)

    # Run openwakeword training pipeline
    subprocess.check_call(
        ["python", "openwakeword/openwakeword/train.py", "--training_config", config_path, "--generate_clips"]
    )
    subprocess.check_call(
        ["python", "openwakeword/openwakeword/train.py", "--training_config", config_path, "--augment_clips"]
    )
    subprocess.check_call(
        ["python", "openwakeword/openwakeword/train.py", "--training_config", config_path, "--train_model"]
    )

    onnx_model = f"my_custom_model/{wake_word}.onnx"
    tflite_model = f"my_custom_model/{wake_word}.tflite"

    # Convert with onnx2tf
    subprocess.check_call(["onnx2tf", "-i", onnx_model, "-o", "my_custom_model/"])
    float32_path = f"my_custom_model/{wake_word}_float32.tflite"
    if os.path.exists(float32_path):
        os.rename(float32_path, tflite_model)

    os.makedirs("output", exist_ok=True)
    arena = estimate_arena(tflite_model)
    with open("output/tensor_arena_size.txt", "w") as f:
        f.write(str(arena or "unknown"))

    meta = {"wake_word": args.wake_word, "model": tflite_model, "tensor_arena_size": arena}
    with open("output/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
