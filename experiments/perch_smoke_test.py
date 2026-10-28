"""Smoke test: load Perch 2.0 through perch-hoplite and embed some audio.

Usage:
    .venv/bin/python experiments/perch_smoke_test.py [AUDIO_FILE]

Without AUDIO_FILE it embeds a synthetic 12 s chirp, which only checks shapes.
With a real bird recording it also prints the model's top species guesses.
"""

import ctypes
import glob
import sys
import sysconfig
import time


def preload_cusolver():
    """Work around TensorFlow 2.21 wheels missing nvidia/cusolver in their runpath.

    Without this TF cannot dlopen libcusolver.so.11, skips the GPU, and
    perch-hoplite silently falls back to the perch_v2_cpu model.
    """
    libs = glob.glob(sysconfig.get_paths()["purelib"] + "/nvidia/cusolver/lib/libcusolver.so.*")
    if libs:
        ctypes.CDLL(libs[0])


preload_cusolver()

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from perch_hoplite.zoo import model_configs  # noqa: E402

print("GPUs visible to TensorFlow:", tf.config.list_physical_devices("GPU"))

t0 = time.perf_counter()
model = model_configs.load_model_by_name("perch_v2")
print(f"Model loaded in {time.perf_counter() - t0:.1f} s: {type(model).__name__}")
print(f"sample_rate={model.sample_rate} Hz, window_size_s={model.window_size_s}, hop_size_s={model.hop_size_s}")

if len(sys.argv) > 1:
    import librosa

    audio, _ = librosa.load(sys.argv[1], sr=model.sample_rate, mono=True)
    print(f"Loaded {sys.argv[1]}: {len(audio) / model.sample_rate:.1f} s")
else:
    duration_s = 12.0
    t = np.arange(int(duration_s * model.sample_rate)) / model.sample_rate
    # Chirp sweeping 2-6 kHz, roughly the range of many songbirds.
    audio = 0.5 * np.sin(2 * np.pi * (2000 * t + 4000 / (2 * duration_s) * t**2))
    audio = audio.astype(np.float32)
    print(f"Synthetic chirp: {duration_s} s")

for run in ("first (includes graph tracing)", "second"):
    t0 = time.perf_counter()
    outputs = model.embed(audio)
    print(f"embed() {run}: {time.perf_counter() - t0:.2f} s")

emb = outputs.embeddings
print("embeddings shape [windows, channels, dim]:", emb.shape, emb.dtype)

vectors = emb[:, 0, :]
norms = np.linalg.norm(vectors, axis=1)
print("raw L2 norms per window:", np.round(norms, 3))
unit = vectors / norms[:, None]
print("cosine similarity between windows:\n", np.round(unit @ unit.T, 3))

if outputs.logits:
    # Logit keys and class list keys differ ('label' vs. 'labels'), and the eBird
    # class list fails to load (duplicate 'no_ebird_code'), so match them by size.
    class_lists = {len(cl.classes): cl for cl in model.class_list.values()}
    for name, logits in outputs.logits.items():
        class_list = class_lists.get(logits.shape[-1])
        if class_list is None:
            continue
        mean_logits = logits.reshape(-1, logits.shape[-1]).mean(axis=0)
        top = np.argsort(mean_logits)[::-1][:5]
        print(f"top-5 '{name}' classes (of {len(class_list.classes)}):",
              [(class_list.classes[i], round(float(mean_logits[i]), 2)) for i in top])
