# TrueLens: Video Fake News Detection with Dual Level Evidence Gathering and Consolidation

This repository contains the source code for the paper “[TrueLens: Video Fake News Detection with Dual Level Evidence Gathering and Consolidation](./WWW_Soft_Copy.pdf)”.

## Project Structure
- `data/` — dataset root
- `log/` — logs
- `preprocess/` — preprocessing scripts
- `src/` — training and model code

> Note: we do not redistribute datasets. Please obtain videos from the original sources:
> - FakeSV: https://github.com/ICTMCG/FakeSV
> - FakeTT: https://github.com/ICTMCG/FakingRecipe
> - FVC: https://github.com/MKLab-ITI/fake-video-corpus
> - After downloading, place files under the respective dataset folders and prepare the required metadata `.jsonl` files accordingly. To facilitate reproduction, I have placed the LLM-generated content at the following link: https://drive.google.com/file/d/1hnXQTxbD-WZKwNIG75Zqw2K0o15lMzDD/view?usp=sharing.  
It contains LLM-generated descriptions for **FakeTT** and **FVC**.  
For **FakeSV**, the data was accidentally removed from our university server and is currently being recovered. Once the recovery is complete, I will update the Google Drive accordingly. (Update: the LLM generated descriptions for **FakeSV** can be accessed by https://drive.google.com/file/d/1LR_jX6_FnkjgOizsIGOfIZJa2ut8S7cm/view?usp=sharing)

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

#LLM-based descriptions (requires OPENAI_API_KEY in env; you can safely skip this step if you download our google drive data above)
python preprocess/LLM_extract.py --dataset FakeTT

# Build full-dataset retrieval artifacts
python preprocess/generate_full_retrieval.py --dataset FakeSV --use-pool --5-fold
```

### 2) Train

```bash
python src/main.py --config-name TrueLens_FakeSV
python src/main.py --config-name TrueLens_FakeTT
python src/main.py --config-name TrueLens_FVC
```
## Citation

If you find our research useful, please kindly consider citing the following work:

```bibtex
@inproceedings{chen2026truelens,
  author    = {Junyi Chen and Qian Liu and Jing Sun and Yi Zhang},
  title     = {TrueLens: Video Fake News Detection with Dual Level Evidence Gathering and Consolidation},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  address   = {Dubai, United Arab Emirates},
  month     = {April 13--17},
  publisher = {ACM},
  location  = {New York, NY, USA},
  pages     = {7178--7188},
  doi       = {10.1145/3774904.3792362},
  url       = {https://doi.org/10.1145/3774904.3792362}
}
```
## Acknowledgement
When developing this project, we referred to the following repository:

[ExMRD](https://github.com/ronpay/ExMRD)

Thanks for their contribution.
