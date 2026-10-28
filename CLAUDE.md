# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Individual software project for the course NDBI034 Multimedia Retrieval (MFF UK, Prof. Tomáš Skopal), presented in person in January/February 2027. The result is a browser app that identifies a bird species from a sound recording by content-based similarity search: the user uploads or records audio and gets the most similar recordings from a reference collection, plus a species guess.

The course is graded on retrieval concepts, not on ML or front-end work. Prefer solutions that make those concepts explicit and measurable over the most convenient black box. Lecture references: 002 (relevance feedback, evaluation measures), 004–005 (similarity queries and distances), 010 (audio features, spectrograms, segmentation, MFCCs), 011–013 (metric indexing, approximate search).

## Status

Early stage: the environment is set up and Perch 2.0 runs; no app code, lint or tests yet. When you add them, record the exact commands here, including how to run a single test. Concept tests and throwaway scripts go in `experiments/`.

### Environment

```bash
sudo apt-get install libsndfile1 ffmpeg     # system packages (both installed)
python3 -m venv .venv                       # system Python 3.13; perch-hoplite needs >=3.10,<3.15
.venv/bin/pip install -r requirements.txt   # perch-hoplite[tf-cuda] + pinned cuDNN, ~7.5 GB
.venv/bin/python experiments/perch_smoke_test.py [AUDIO_FILE]   # load Perch 2.0, embed, print top species
.venv/bin/python experiments/xeno_canto_download.py --species-file experiments/species_central_europe_50.txt --max 50   # needs XC_API_KEY in .env
```

The xeno-canto download writes `datasets/xeno-canto/audio/XC<id>.<mp3|wav>` and appends to `datasets/xeno-canto/metadata.csv` (licence, recordist, upload date, background species `also`); it resumes by skipping known ids. The API lists newest uploads first, so the script fetches all matches (`area:europe len:5-60`, quality A then B), shuffles them with a fixed seed and takes at most `--per-recordist` (default 3) per recordist and species, relaxing the cap only when a species has too few recordists. `datasets/xeno-canto-blackbird-test/` is an earlier unbalanced test download (44 of 50 from one recordist); do not mix it in. The tag `since:YYYY-MM-DD` filters by upload date and is how to select post-model-release test recordings; `uploaded:` is rejected by API v3. API v3 has no strict rate limit (per its docs), but responses can take 25–35 s each, so a full 50×50 download runs for many hours; the script retries failed requests with backoff. `per_page` accepts 50–500. Restricted species have `file` redacted and are skipped. The API docs ask not to publish the key: it lives only in the gitignored `.env`.

GPU quirks on this machine (GTX 1050, 4 GB, compute capability 6.1, driver 550 / CUDA 12.4):
- TensorFlow 2.21's wheel runpath omits `nvidia/cusolver/lib`, so TF skips the GPU and `perch_v2` silently loads the CPU variant. Preload `libcusolver.so.*` with `ctypes.CDLL` before `import tensorflow` (see `preload_cusolver()` in `experiments/perch_smoke_test.py`).
- cuDNN 9.27 fails conv autotuning on Pascal ("No valid config found"); `requirements.txt` pins `nvidia-cudnn-cu12==9.3.0.75`.
- The first `embed()` call takes ~15–20 s (XLA compilation); later calls take ~0.08 s per 12 s clip.

## Planned pipeline

```
recordings ──▶ split into ~5 s windows ──▶ pretrained audio model ──▶ window embeddings (L2-normalized)
                                                                              │
query audio ──▶ same windowing + model ──▶ query vector(s) ──▶ index kNN ◀────┘
                                                                  │
                         ranked recordings + species vote ──▶ web GUI ──▶ 👍/👎 relevance feedback
```

- **Model**: one pretrained bioacoustics model, used frozen (no training). The first choice is Perch 2.0 (Google), the fallback is BirdNET. Do not add a multi-model comparison; the owner explicitly does not want one.
  - Do **not** depend on the `google-research/perch` repository: its own README says it is partly outdated and `pip install` is unlikely to work.
  - For inference use **`perch-hoplite`** (installed via `requirements.txt`; `perch-hoplite[tf]` is the CPU-only alternative). Its `zoo` module wraps Perch and BirdNET behind a common embedding interface; its `agile/01_embed_audio.ipynb` notebook shows the embedding workflow.
  - Load with `model_configs.load_model_by_name('perch_v2')`. It picks the Kaggle variation `google/bird-vocalization-classifier/tensorFlow2/perch_v2` (version 2) on GPU or `perch_v2_cpu` without one, and caches the 392 MB download in `~/.cache/kagglehub`. Verified: 32 kHz mono input, 5 s windows with 5 s hop, 1536-dim embeddings (raw L2 norm varies with the input, ≈2.7–4.8 seen so far, so normalize them). `model.embed(audio)` returns `embeddings` shaped `[windows, 1, 1536]`; a trailing partial window is dropped (12 s → 2 windows).
  - Logits: key `label`, 14,795 classes named in `assets/labels.csv` (scientific names plus some non-bird labels). The eBird-code class list fails to load in hoplite (duplicate `no_ebird_code`), which is harmless.
  - Licence of the weights is still unverified: check the Kaggle page for the `perch_v2` variation.
  - perch-hoplite's `db` module also has exact and approximate vector search. It is one candidate for the library-index option below, but our own index code must not depend on its storage format.
- **Data**: xeno-canto recordings of about 50–100 Central European species, plus the owner's own recordings as clean test queries. Each recording has its own Creative Commons licence (mostly BY-NC-ND), so keep the recording ids and licence metadata for attribution. Audio, embeddings and built indexes live under the gitignored `datasets/`.
- **Windows and recordings**: a recording is a set of window vectors, and its species label is weak (it applies to the whole recording, not each window). Similarity between recordings is either a set distance (e.g. the mean of nearest-window distances, Hausdorff-like) or a pooled single vector. Keep this aggregation separate from the index, which works on window vectors.
- **Distance**: Euclidean (L2) on L2-normalized vectors. It ranks results exactly like cosine similarity, but unlike cosine distance it is a metric, so the pivot lower bounds used by metric indexes stay valid. Normalize every vector, including query vectors after relevance feedback. All distance evaluations should go through one shared function that also counts distance computations, the main cost measure for comparing indexes.
- **Indexes**: whether they are implemented by hand (e.g. LAESA pivot table exact and approximate, permutation index, D-file/D-cache) or taken from a library (e.g. FAISS HNSW) is **not decided yet**. Either way, every index implements the same small interface: `build(vectors)` and `knn(query, k) -> (ids, stats)`, where `stats` includes the distance computation count and the time. The GUI selector, the evaluation and the tests then work with any index. Sequential scan is always present: it is the exact reference that index tests compare against.
- **Relevance feedback**: Rocchio query shifting, `q' = normalize(α·q + β·mean(liked) − γ·mean(disliked))`. The model is never involved; feedback only moves the query vector.
- **GUI**: planned in Streamlit. It reruns the whole script on every interaction, so load the model, embeddings and indexes in `@st.cache_resource`, and keep the query vector and feedback lists in `st.session_state`.

## Evaluation rules

- **Retrieval quality** uses species labels: P@k, mAP, precision-recall curves, top-k species accuracy, and mAP after feedback rounds (with a simulated user).
- **Index quality** uses the sequential-scan kNN as ground truth, not the labels: recall vs. exact kNN, distance computations per query, query time.
- **Split train/test by recording**, ideally by recordist or location. Windows from one recording must never be on both sides.
- **Training-data leakage**: Perch and BirdNET were trained largely on xeno-canto, so older xeno-canto recordings were probably seen by the model and give inflated results. Query/test recordings should be uploaded after the model's release (Perch 2.0: August 2025), or come from outside xeno-canto (e.g. the owner's own recordings).
- Tune parameters (feedback weights, number of pivots, candidate counts) on a validation set of queries, and report final numbers on separate test queries.
