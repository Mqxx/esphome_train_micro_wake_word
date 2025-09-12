import argparse, os, subprocess, json, yaml, shutil, sys
import tensorflow as tf

def estimate_arena(path):
    with open(path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    try:
        return interp._get_arena_used_bytes()
    except:
        return None

def download_rirs():
    os.makedirs("./mit_rirs", exist_ok=True)
    subprocess.check_call(["git", "lfs", "install"])
    subprocess.check_call(["git", "clone", "--depth", "1",
                           "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses",
                           "temp_rir"])
    for f in os.listdir("temp_rir/16khz"):
        shutil.copy(os.path.join("temp_rir/16khz", f), "./mit_rirs/")
    shutil.rmtree("temp_rir")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake_word", required=True)
    args = parser.parse_args()
    wake_word = args.wake_word.replace(" ", "_")

    # Download RIRs
    download_rirs()

    # Load base config
    base_cfg = yaml.safe_load(open("openwakeword/examples/custom_model.yml"))
    base_cfg["target_phrase"] = [args.wake_word]
    base_cfg["model_name"] = wake_word
    base_cfg["n_samples"] = 1000
    base_cfg["n_samples_val"] = 500
    base_cfg["steps"] = 10000
    base_cfg["output_dir"] = "./my_custom_model"
    base_cfg["background_paths"] = ["./audioset_16k", "./fma"]
    base_cfg["rir_paths"] = ["./mit_rirs"]
    base_cfg["false_positive_validation_data_path"] = "validation_set_features.npy"
    base_cfg["feature_data_files"] = {
        "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    }

    os.makedirs(base_cfg["output_dir"], exist_ok=True)
    with open("my_model.yaml", "w") as f:
        yaml.dump(base_cfg, f)

    # Run openwakeword pipeline
    for step in ["--generate_clips", "--augment_clips", "--train_model"]:
        subprocess.check_call(["python", "openwakeword/openwakeword/train.py",
                               "--training_config", "my_model.yaml", step])

    onnx_model = f"my_custom_model/{wake_word}.onnx"
    tflite_model = f"my_custom_model/{wake_word}.tflite"

    # Convert ONNX → TFLite
    subprocess.check_call(["onnx2tf", "-i", onnx_model, "-o", "my_custom_model/"])
    float32_path = f"my_custom_model/{wake_word}_float32.tflite"
    if os.path.exists(float32_path):
        os.rename(float32_path, tflite_model)

    # Compute tensor arena size
    os.makedirs("output", exist_ok=True)
    arena = estimate_arena(tflite_model)
    with open("output/tensor_arena_size.txt", "w") as f:
        f.write(str(arena or "unknown"))

    # Save metadata
    meta = {"wake_word": args.wake_word, "model": tflite_model, "tensor_arena_size": arena}
    with open("output/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
