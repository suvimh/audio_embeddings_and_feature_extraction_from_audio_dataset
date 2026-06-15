# TUNI Emotion Dataset — Feature Extraction

Extract OpenSmile, CLAP, Whisper, and VGGish embeddings from the TUNI emotion singing dataset.

## Prerequisites

- Python 3.12+
- Repo dependencies: `pip install -r ../requirements.txt`
- Extractor extras: `pip install -r ../requirements-extractors.txt`
- ~900 `.wav` files under `wav/` at the repo root

See [Mac setup notes](#mac-setup-clap--vggish) if CLAP or VGGish fail.

## Data layout

Audio files live under `wav/` (repo root):

```
wav/
  <singer>/
    <genre>/           # pop | classical
      <emotion>/       # joy | sadness | anger | gentleness | neutral  (optional)
        *.wav
```

Some singers use a flat layout (`singer/genre/*.wav`). In that case `emotion` may be `None` in the output parquet; `genre` filters still work for plots.

## Configuration

Edit paths only if your layout differs:

| File | What to set |
|------|-------------|
| [`tuni_data_config.py`](tuni_data_config.py) | `DATA_INPUT_DIR` — defaults to `../wav` |
| [`tuni_extraction_config.py`](tuni_extraction_config.py) | `DATA_OUTPUT_DIR` — defaults to `./embeddings` |

## Run the notebook

1. Open a terminal in **this folder** (`TUNI_emotion_data_extraction/`).
2. Start Jupyter / open [`tuni_data_extraction.ipynb`](tuni_data_extraction.ipynb) from here (so relative config paths resolve).
3. **Restart kernel** → run **Section 1** (setup). Confirm the setup cell prints a non-empty `Root directory` ending in `wav`.
4. Run **one** extraction cell in Section 2 (prefer the cell with `console_output=True`). Do not run both extraction cells.
5. Expect **several hours** for a full run (900 files × 4 extractors × 2 window lengths). VGGish has a long silent startup (~10–20 min) on first load.
6. Run **Section 3** to load parquet files and plot LDA embeddings.

## Output files

Parquet naming: `{dataset_name}_{extractor_name}_{duration}s.parquet`

Examples (3.0 s windows):

| Extractor | Filename |
|---------|----------|
| OpenSmile | `tuni_emotion_dataset_opensmile-compare-2016_3.0s.parquet` |
| CLAP | `tuni_emotion_dataset_clap-2023_3.0s.parquet` |
| Whisper | `tuni_emotion_dataset_whisper_whisper-base_3.0s.parquet` |
| VGGish | `tuni_emotion_dataset_vggish_3.0s.parquet` |

Section 3 plots use OpenSmile and Whisper @ 3.0 s only.

## Mac setup (CLAP / VGGish)

**CLAP** uses `torchaudio.load` → torchcodec → FFmpeg shared libraries.

1. Install FFmpeg: `conda install -c conda-forge ffmpeg`
2. If torchcodec looks for Homebrew paths, symlink conda libs (one-time):
   ```bash
   sudo mkdir -p /opt/homebrew/opt/ffmpeg/lib
   for lib in avutil avcodec avformat avdevice avfilter swresample swscale; do
     sudo ln -sf /opt/miniconda3/lib/lib${lib}*.dylib /opt/homebrew/opt/ffmpeg/lib/
   done
   ```
3. Match torch versions: `pip install torch==2.11.0 torchaudio==2.11.0`

**VGGish** downloads from TensorFlow Hub on first run.

1. Run `/Applications/Python 3.12/Install Certificates.command` (python.org installs).
2. Test: `import tensorflow_hub as hub; hub.load("https://tfhub.dev/google/vggish/1")`

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `0 total files` | `DATA_INPUT_DIR` empty or wrong — save config, restart kernel |
| `900 already processed`, instant finish | Delete `*.checkpoint.jsonl` and broken `*.h5` for that extractor, re-run |
| CLAP `extracted=0` / torchcodec errors | See Mac setup above |
| VGGish SSL error | Install Certificates.command |
| `FileNotFoundError` in Section 3 | That parquet not extracted yet — check `embeddings/` listing cell |
