# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Individual software project for the course NDBI034 Multimedia Retrieval (MFF UK, Prof. Tomáš Skopal), presented in person in January/February 2027. The result is a browser app that identifies a bird species from a sound recording by content-based similarity search: the user uploads or records audio and gets the most similar recordings from a reference collection, plus a species guess.

The course is graded on retrieval concepts, not on ML or front-end work. Prefer solutions that make those concepts explicit and measurable over the most convenient black box. Lecture references: 002 (relevance feedback, evaluation measures), 004–005 (similarity queries and distances), 010 (audio features, spectrograms, segmentation, MFCCs), 011–013 (metric indexing, approximate search).

## Status

The repository is new: only README, LICENSE (MIT) and a Python `.gitignore` exist. No dependencies, build, lint or test commands are set up yet. When you add them, record the exact commands here, including how to run a single test.

## Planned pipeline

```
recordings ──▶ split into ~5 s windows ──▶ pretrained audio model ──▶ window embeddings (L2-normalized)
                                                                              │
query audio ──▶ same windowing + model ──▶ query vector(s) ──▶ index kNN ◀────┘
                                                                  │
                         ranked recordings + species vote ──▶ web GUI ──▶ 👍/👎 relevance feedback
```

- **Model**: one pretrained bioacoustics model, used frozen (no training). The first choice is Perch 2.0 (Google), the fallback is BirdNET. Verify that the weights can be downloaded and check the licence before depending on either. Do not add a multi-model comparison; the owner explicitly does not want one.
- **Data**: xeno-canto recordings (or the BirdSet dataset on Hugging Face, which is built from them), about 50–100 Central European species. Each recording has its own Creative Commons licence, so keep the recording ids and licence metadata. Audio, embeddings and built indexes are large: keep them out of git.
- **Windows and recordings**: a recording is a set of window vectors, and its species label is weak (it applies to the whole recording, not each window). Similarity between recordings is either a set distance (e.g. the mean of nearest-window distances, Hausdorff-like) or a pooled single vector. Keep this aggregation separate from the index, which works on window vectors.
- **Distance**: Euclidean (L2) on L2-normalized vectors. It ranks results exactly like cosine similarity, but unlike cosine distance it is a metric, so the pivot lower bounds used by metric indexes stay valid. Normalize every vector, including query vectors after relevance feedback. All distance evaluations should go through one shared function that also counts distance computations, the main cost measure for comparing indexes.
- **Indexes**: whether they are implemented by hand (e.g. LAESA pivot table exact and approximate, permutation index, D-file/D-cache) or taken from a library (e.g. FAISS HNSW) is **not decided yet**. Either way, every index implements the same small interface: `build(vectors)` and `knn(query, k) -> (ids, stats)`, where `stats` includes the distance computation count and the time. The GUI selector, the evaluation and the tests then work with any index. Sequential scan is always present: it is the exact reference that index tests compare against.
- **Relevance feedback**: Rocchio query shifting, `q' = normalize(α·q + β·mean(liked) − γ·mean(disliked))`. The model is never involved; feedback only moves the query vector.
- **GUI**: planned in Streamlit. It reruns the whole script on every interaction, so load the model, embeddings and indexes in `@st.cache_resource`, and keep the query vector and feedback lists in `st.session_state`.

## Evaluation rules

- **Retrieval quality** uses species labels: P@k, mAP, precision-recall curves, top-k species accuracy, and mAP after feedback rounds (with a simulated user).
- **Index quality** uses the sequential-scan kNN as ground truth, not the labels: recall vs. exact kNN, distance computations per query, query time.
- **Split train/test by recording**, ideally by recordist or location. Windows from one recording must never be on both sides.
- Tune parameters (feedback weights, number of pivots, candidate counts) on a validation set of queries, and report final numbers on separate test queries.
