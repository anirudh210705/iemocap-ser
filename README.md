# IEMOCAP Speech Emotion Recognition

Training-ready BTP code for speaker-independent, four-class speech emotion recognition on IEMOCAP. The four labels are **angry**, **happy** (`hap` + `exc`), **neutral**, and **sad**.

## What is implemented

- Official IEMOCAP annotation and transcription parser
- Audio integrity/duration verification
- Fixed session split and five-fold leave-one-session-out splits
- CNN log-Mel baseline
- Wav2Vec2/HuBERT encoder with attentive pooling
- Frozen BLOOMZ classifier driven by one projected acoustic token
- Noise, speed, pitch, combined waveform augmentation, and batch Mix-up
- Class-weighted loss, mixed precision, gradient accumulation, early stopping, and checkpoint resume
- Macro-F1 evaluation, per-class metrics, and confusion matrices
- Single-WAV inference
- Compact LLM checkpoints and five-fold experiment automation

No training starts automatically.

## Dataset layout

```text
data/
├── Session1/
│   ├── dialog/EmoEvaluation/*.txt
│   ├── dialog/transcriptions/*.txt
│   └── sentences/wav/**/*.wav
├── Session2/
├── Session3/
├── Session4/
└── Session5/
```

The dataset and generated artifacts are ignored by Git.

## Setup

Python 3.10–3.12 is recommended. On Colab or a local environment:

```bash
pip install -e .
```

Prepare metadata and splits (this reads audio headers but does not train anything):

```bash
python -m ser.prepare_iemocap --data-root data --output-dir artifacts
```

Generated files include:

```text
artifacts/metadata.csv
artifacts/splits/fixed/{train,val,test}.csv
artifacts/splits/loso/fold_1/{train,val,test}.csv
...
artifacts/splits/loso/fold_5/{train,val,test}.csv
```

The development split uses Sessions 1–3 for training, Session 4 for validation, and Session 5 for testing. Speakers never cross split boundaries.

## Training commands (run only when ready)

CNN baseline:

```bash
python -m ser.train --config configs/cnn_baseline.json
```

Wav2Vec2 with attentive pooling:

```bash
python -m ser.train --config configs/wav2vec2_base.json
```

Frozen BLOOMZ smoke test (560M parameters):

```bash
python -m ser.train --config configs/bloomz_560m_smoke.json
```

Main frozen-LLM experiment:

```bash
python -m ser.train --config configs/bloomz_1b7_main.json
```

The LLM architecture is `Wav2Vec2 -> attentive pooling -> projector -> one acoustic token -> frozen BLOOMZ -> four emotion-token logits`. The transcript is not supplied. BLOOMZ parameters remain frozen, but gradients pass through the LLM to train the projector, pooling layer, and selected audio-encoder layers. The optional `bloomz_7b1_optional.json` configuration follows the model family used in the reference paper and should only be attempted on a larger GPU.

Validate the four label tokens without downloading the LLM weights:

```bash
python -m ser.inspect_llm --config configs/bloomz_1b7_main.json
```

Run one forward/backward batch without an optimizer step:

```bash
python -m ser.smoke_test --config configs/bloomz_560m_smoke.json --max-seconds 4
```

Slurm templates for the smoke test and the three main models are provided under `slurm/`.

Resume after a Colab disconnect:

```bash
python -m ser.train --config configs/wav2vec2_base.json --resume artifacts/runs/wav2vec2_base/last.pt
```

To run an augmentation experiment, copy a configuration and set `augmentation` to `noise`, `speed`, `pitch`, `combined`, or `specaugment`. Set `mixup_alpha` to a positive value such as `0.2` for Mix-up.

## Evaluation and inference

```bash
python -m ser.evaluate \
  --checkpoint artifacts/runs/wav2vec2_base/best.pt \
  --csv artifacts/splits/fixed/test.csv \
  --output-dir artifacts/results/wav2vec2_base

python -m ser.infer \
  --checkpoint artifacts/runs/wav2vec2_base/best.pt \
  --audio path/to/sample.wav
```

On PowerShell, place each command on one line or use the PowerShell continuation character instead of `\`.

Inspect the planned core experiments without starting training:

```bash
python -m ser.run_experiments --dry-run
```

Run five leave-one-session-out folds after selecting a final configuration:

```bash
python -m ser.run_folds --config configs/bloomz_1b7_main.json
```

## Colab

Open `notebooks/colab_setup.ipynb` in VS Code, select a Colab GPU kernel, and adjust the project/data locations. For speed, keep active training data in `/content` and copy checkpoints back to Drive.
