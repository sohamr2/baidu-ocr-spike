# ============================================================================
# Baidu Unlimited-OCR spike - ALL-IN-ONE smoke test (no runtime restart).
#
# How to use in Colab:
#   1. Runtime > Change runtime type > GPU (T4/L4/A100).
#   2. Runtime > Restart session  (ONE clean restart to clear any old state).
#   3. Run THIS single cell top to bottom. No further restarts.
#
# It keeps Colab's preinstalled torch (so there is no torch-downgrade restart),
# installs the remaining pinned deps, builds edlib from source (its shipped C++
# fails to compile on CPython 3.12+), loads baidu/Unlimited-OCR, and OCRs page 1
# of your PDF at /content/drive/MyDrive/ocr_spike/inputs/book1.pdf (no scoring).
# ============================================================================
import glob
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def sh(cmd: str) -> int:
    print("$", cmd)
    return subprocess.run(cmd, shell=True).returncode


# --- 1) Python deps (NO torch/torchvision/Pillow pin -> no restart) ---------
# Do NOT pin Pillow: Colab's torchvision (which the model's remote code imports)
# needs the Pillow it shipped with. Pillow 12 removed internals torchvision uses
# ("cannot import name '_Int' from 'PIL._typing'"), so keep Pillow on the 11.x line.
sh(
    f"{sys.executable} -m pip install -q "
    "transformers==4.57.1 huggingface-hub==0.35.3 safetensors==0.4.5 "
    "accelerate==1.1.1 einops==0.8.2 addict==2.4.0 easydict==1.13 "
    "pymupdf==1.27.2.2 jiwer==3.0.4 rapidfuzz==3.10.1 pandas==2.2.3"
)
# Keep/repair a torchvision-compatible Pillow (undo any earlier Pillow 12 install).
sh(f"{sys.executable} -m pip install -q 'Pillow>=11,<12'")
# Drop any half-imported PIL/torchvision so the corrected Pillow is picked up
# WITHOUT needing a runtime restart.
for _m in [
    k for k in list(sys.modules)
    if k == "PIL" or k.startswith("PIL.") or k == "torchvision" or k.startswith("torchvision.")
]:
    del sys.modules[_m]

# --- 2) edlib: build from source (shipped cpp includes longintrepr.h) -------
try:
    import edlib  # noqa: F401
    print("edlib already present")
except Exception:
    sh(f"{sys.executable} -m pip install -q -U cython setuptools wheel")
    _w = tempfile.mkdtemp()
    sh(f"{sys.executable} -m pip download edlib==1.3.9 --no-binary :all: --no-deps -d {_w}")
    _tar = glob.glob(f"{_w}/edlib-*.tar.gz")[0]
    sh(f"tar xzf {_tar} -C {_w}")
    _src = glob.glob(f"{_w}/edlib-*/")[0]
    _pyx = glob.glob(f"{_src}/*.pyx")[0]
    sh(f"cython --cplus -3 {_pyx} -o {_src}/edlib.bycython.cpp")
    sh(f"{sys.executable} -m pip install -q --no-build-isolation {_src}")
    import importlib
    importlib.invalidate_caches()
    import edlib  # noqa: F401
print("edlib OK:", edlib.align("acgt", "acxt")["editDistance"], "(expected 1)")

# --- 3) repo (the ocr_spike package) ----------------------------------------
REPO = "/content/baidu-ocr-spike"
if os.path.isdir(REPO):
    sh(f"cd {REPO} && git pull -q")
else:
    sh(f"git clone -q https://github.com/sohamr2/baidu-ocr-spike.git {REPO}")
if REPO not in sys.path:
    sys.path.insert(0, REPO)

# --- 4) GPU + dtype (T4 -> fp16 automatically) ------------------------------
import torch  # noqa: E402  (Colab's preinstalled build)
from ocr_spike import infer, pdf  # noqa: E402

runtime = infer.detect_runtime()
print("GPU:", runtime.gpu_name, "| dtype:", runtime.dtype_str, "| torch:", runtime.torch_version)

# --- 5) load the model (first run downloads a few GB) -----------------------
model, tokenizer, REV = infer.load_model("baidu/Unlimited-OCR", runtime.dtype)
print("model loaded @", REV)
print("API:")
for _k, _v in infer.introspect_api(model).items():
    print("  ", _k, ":", _v)

# --- 6) OCR smoke test on page 1 of your PDF (no scoring) -------------------
PDF = "/content/drive/MyDrive/ocr_spike/inputs/book1.pdf"
if not os.path.exists(PDF):
    from google.colab import drive
    drive.mount("/content/drive")
assert os.path.exists(PDF), f"PDF not found: {PDF}"

pages = pdf.pdf_to_images(PDF, "/content/smoke_pages", dpi=300, force=False)
print("rendered", len(pages), "pages")

GUNDAM = dict(
    base_size=1024, image_size=640, crop_mode=True, max_length=32768,
    no_repeat_ngram_size=35, ngram_window=128, save_results=True,
)
outcome = infer.run_inference(
    model, tokenizer, mode="single", image_files=[pages[0]],
    prompt="<image>document parsing.", output_dir=Path("/content/smoke_out/p0"),
    infer_kwargs=GUNDAM, capture_generate=False,
)
print("=" * 70)
print("OCR OUTPUT - page 1 (first 3000 chars):")
print("=" * 70)
print(outcome.cleaned_text[:3000])
