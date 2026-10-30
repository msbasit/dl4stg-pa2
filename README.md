# AI651 Assignment 2

Q1 is a design proposal; no Q1 training is performed. Q2 uses its own notebook and experiment module. All learned Q2 training and inference require the Intel Arc XPU, with no silent CPU fallback. CPU preprocessing is used for data loading and the specified OpenCV optical-flow analysis.

Run `Q2_Motion_or_Memory.ipynb` in order with an XPU-enabled PyTorch kernel. Results and final-epoch checkpoints are cached so reopening the notebook does not repeat training. Training resumes only from complete epoch checkpoints. Models A-C follow the exact baseline; D keeps layer 1 at dilation 1 and changes layer 2 to dilation 3, with unchanged parameter count. The combined offsets cover -4 through +4 folded cells without holes. No optional D+B experiment is included.

Repository URL supplied for submission: https://github.com/msbasit/dl4stg-pa2. A local Git repository preserves the pre-experiment predictions and completed work. This update adds the supplied URL to the report; it does not publish files or verify repository visibility.

Environment used: existing `C:/Users/msbasit/Documents/Codex/2026-10-20/complete-this-assignment-end-to-end/work/venv-xpu/Scripts/python.exe`, PyTorch 2.14.1+xpu, torchvision 0.29.1+xpu. The ordinary system Python contains CPU-only PyTorch and must not be used for training.

The notebook kernel is registered locally as **Python (Intel Arc XPU - AI651)** (`ai651-assignment2-xpu`). Training uses XPU bfloat16 autocast with float32 weights, input intensities and loss. Windows `torch.compile` uses the already installed Visual Studio C++ Build Tools, located automatically by `compiler_environment()`. The initial ten completed epochs of A seed 0 used eager execution; its subsequent epochs and the other runs use compiled execution. Recorded epoch times include compilation encountered inside the timed epoch. Evaluation is eager for every learned model, and the group wrapper averages inverse-aligned outputs in float32.

Official XPU setup reference: https://docs.pytorch.org/docs/stable/notes/get_start_xpu.html

The report includes every required numbered table, figure and Q2 answer, including the negative result that D did not improve on C's forecast accuracy. `results/run_manifest.json` records the eight final checkpoints and their SHA-256 hashes. `make_notebooks_and_answers.py` regenerates the answers/notebooks from stored results and visual notes; `make_report.py` compiles the report with Pandoc and XeLaTeX (MiKTeX is installed on this machine). The ZIP excludes the 819 MB dataset download; the notebook reuses the local cached dataset or downloads it when needed.
