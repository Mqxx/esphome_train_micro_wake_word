import argparse, os, subprocess, json, yaml, shutil, sys
import tensorflow as tf

def estimate_arena(path):
    """Schätzt die Tensor Arena Größe des TFLite-Modells."""
    with open(path, "rb") as f:
        model_bytes = f.read()
    interp = tf.lite.Interpreter(model_content=model_bytes)
    interp.allocate_tensors()
    try:
        return interp._get_arena_used_bytes()
    except:
        return None

def download_file(url, path):
    """Hilfsfunktion: Datei herunterladen, falls fehlt."""
    if not os.path.exists(path):
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        subprocess.check_call(["wget", "-O", path, url])

def download_rirs():
    """Lädt RIRs für realistische Raumakustik herunter."""
    if not os.path.exists("./mit_rirs"):
        os.makedirs("./mit_rirs", exist_ok=True)
        subprocess.check_call(["git", "lfs", "install"])
        subprocess.check_call([
            "git", "clone", "--depth", "1",
            "https://huggingface.co/datasets/davidscripka/MIT_environmental_impulse_responses",
            "temp_rir"
        ])
        for f in os.listdir("temp_rir/16khz"):
            shutil.copy(os.path.join("temp_rir/16khz", f), "./mit_rirs/")
        shutil.rmtree("temp_rir")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wake_word", required=True)
    args = parser.parse_args()
    wake_word = args.wake_word.replace(" ", "_")

    # 1️⃣ Piper TTS Modell
    download_file(
        "https://github.com/rhasspy/piper-sample-generator/releases/download/v2.0.0/en_US-libritts_r-medium.pt",
        "piper-sample-generator/models/en_US-libritts_r-medium.pt"
    )

    # 2️⃣ RIRs
    download_rirs()

    # 3️⃣ ACAV100M Features
    download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/openwakeword_features_ACAV100M_2000_hrs_16bit.npy",
        "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    )

    # 4️⃣ Optional: Validation Features
    download_file(
        "https://huggingface.co/datasets/davidscripka/openwakeword_features/resolve/main/validation_set_features.npy",
        "validation_set_features.npy"
    )

    # 5️⃣ Load base config
    base_cfg = yaml.safe_load(open("openwakeword/examples/custom_model.yml"))

    # Wake Word konfigurieren
    base_cfg["target_phrase"] = [args.wake_word]
    base_cfg["model_name"] = wake_word
    base_cfg["n_samples"] = 1000
    base_cfg["n_samples_val"] = 500
    base_cfg["steps"] = 10000
    base_cfg["output_dir"] = "./my_custom_model"
    base_cfg["rir_paths"] = ["./mit_rirs"]

    # Background Pfade prüfen
    bg_paths = ["./audioset_16k", "./fma"]
    existing_bg_paths = [p for p in bg_paths if os.path.exists(p)]
    base_cfg["background_paths"] = existing_bg_paths

    # FP validation
    if os.path.exists("validation_set_features.npy"):
        base_cfg["false_positive_validation_data_path"] = "validation_set_features.npy"

    # Feature Daten
    base_cfg["feature_data_files"] = {
        "ACAV100M_sample": "openwakeword_features_ACAV100M_2000_hrs_16bit.npy"
    }

    os.makedirs(base_cfg["output_dir"], exist_ok=True)
    with open("my_model.yaml", "w") as f:
        yaml.dump(base_cfg, f)

    # OpenWakeWord Schritte
    for step in ["--generate_clips", "--augment_clips", "--train_model"]:
        subprocess.check_call([
            "python", "openwakeword/openwakeword/train.py",
            "--training_config", "my_model.yaml", step
        ])

    # ONNX → TFLite konvertieren
    onnx_model = f"my_custom_model/{wake_word}.onnx"
    tflite_model = f"my_custom_model/{wake_word}.tflite"
    subprocess.check_call(["onnx2tf", "-i", onnx_model, "-o", "my_custom_model/"])
    float32_path = f"my_custom_model/{wake_word}_float32.tflite"
    if os.path.exists(float32_path):
        os.rename(float32_path, tflite_model)

    # Tensor Arena Größe berechnen
    os.makedirs("output", exist_ok=True)
    arena = estimate_arena(tflite_model)
    with open("output/tensor_arena_size.txt", "w") as f:
        f.write(str(arena or "unknown"))

    # Metadata speichern
    meta = {"wake_word": args.wake_word, "model": tflite_model, "tensor_arena_size": arena}
    with open("output/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

if __name__ == "__main__":
    main()
