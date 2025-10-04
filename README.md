# TrueLens: Video Fake News Detection with Dual Level Evidence Gathering and Consolidation

This repository contains the source code for the paper “TrueLens: Video Fake News Detection with Dual Level Evidence Gathering and Consolidation”.

## Project Structure
- `data/` — dataset root
- `log/` — logs
- `preprocess/` — preprocessing scripts
- `src/` — training and model code

> Note: we do not redistribute datasets. Please obtain videos from the original sources:
> - FakeSV: https://github.com/ICTMCG/FakeSV
> - FakeTT: https://github.com/ICTMCG/FakingRecipe
> - FVC: https://github.com/MKLab-ITI/fake-video-corpus
> - After downloading, place files under the respective dataset folders and prepare the required metadata `.jsonl` files accordingly.

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```


## Run (from repo root)

### 1) Preprocess each dataset

```bash
# Extract frames (with timestamps)
python preprocess/extract_frame.py --datasets FakeSV FakeTT FVC --num_frames 16

# Build 2x2 quad mosaics
python preprocess/frames_to_quad_4.py --datasets FakeSV FakeTT FVC

# Visual features (CLIP/Chinese-CLIP)
python preprocess/make_video_feature.py --datasets FakeSV FakeTT FVC

# Convert videos to audio (WAV) and generate global transcripts (Whisper)
python preprocess/video_to_wav.py --datasets FakeSV FakeTT FVC
python preprocess/wav_to_transcript.py --datasets FakeSV FakeTT FVC

# Frame-aligned audio features (skip transcripts in this step)
python preprocess/audio_frame_processing.py --dataset FakeSV --skip_transcripts

#LLM-based descriptions (requires OPENAI_API_KEY in env)
python preprocess/LLM_extract.py --dataset FakeTT

# Build full-dataset retrieval artifacts
python preprocess/generate_full_retrieval.py --dataset FakeSV --use-pool
```

### 2) Train

```bash
python src/main.py --config-name TrueLens_FakeSV
python src/main.py --config-name TrueLens_FakeTT
python src/main.py --config-name TrueLens_FVC
```
