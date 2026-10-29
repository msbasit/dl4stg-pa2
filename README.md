# AI651 Assignment 2

Q1 is a design proposal; no Q1 training is performed. Q2 uses its own notebook and experiment module. All learned Q2 training and inference require the Intel Arc XPU, with no silent CPU fallback. CPU preprocessing is used for data loading and the specified OpenCV optical-flow analysis.

Run `Q2_Motion_or_Memory.ipynb` in order with an XPU-enabled PyTorch kernel. Results and final-epoch checkpoints are cached so reopening the notebook does not repeat training. Training resumes only from complete epoch checkpoints. Models A-C follow the exact baseline; D changes both gate convolutions to dilation 2, with unchanged parameter count. No optional D+B experiment is included.

The assignment's required public GitHub link cannot be inferred from local files. A local Git repository records the required pre-experiment predictions; publish it to your own GitHub account before submission and insert its URL on the report's first page.

Environment used: existing `C:/Users/msbasit/Documents/Codex/2026-10-20/complete-this-assignment-end-to-end/work/venv-xpu/Scripts/python.exe`, PyTorch 2.14.1+xpu, torchvision 0.29.1+xpu. The ordinary system Python contains CPU-only PyTorch and must not be used for training.

Official XPU setup reference: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html
