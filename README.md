# Marine Vision Tracking

**Work in progress.** This repo is private until the deliverables in [TODO.md](TODO.md) are done.

Detection, segmentation and point tracking of boats, buoys and other small objects on water video. The pipeline:

```
video frames ─► YOLO detection ─► SAM mask per detection ─► point tracking inside each mask
            ─► per-object tracks (ID, box, mask, trajectory) ─► annotated video + metrics
```

Water is a hard setting for vision: glare, waves, wakes, tiny distant targets, and a moving camera. That's what this project is about.

## Setup

**Windows (PowerShell or cmd):**

```bash
git clone https://github.com/DbugDiver/marine-vision-tracking.git
cd marine-vision-tracking
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
git clone https://github.com/DbugDiver/marine-vision-tracking.git
cd marine-vision-tracking
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For a GPU, install the CUDA build of PyTorch first ([pytorch.org](https://pytorch.org/get-started/locally/)), then run `pip install -r requirements.txt`.

## Instructions for whoever works on this next (person or AI agent)

1. **Read [TODO.md](TODO.md) first.** It has the clean-room rules, the target structure, and the deliverables.
2. **Clean-room rule:** write everything from scratch. Use no code, footage, data, parameters or documents from any employer.
3. **Commit in small, meaningful steps.** Never squash or rewrite history.
4. **Report only real measurements** (FPS, track counts, failure cases). If something doesn't work well, say so.
5. **Don't make the repo public.** The owner reviews it first.
