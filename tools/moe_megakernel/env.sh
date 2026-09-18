# Source this before anything here:  source tools/moe_megakernel/env.sh
#
# This box's python is PEP 668 externally-managed and python3-venv is not installed
# (installing it needs sudo apt = a system-wide change). So the extra packages live in a
# folder-local pip --target directory and go on PYTHONPATH. Nothing outside this folder
# is modified, and the system torch 2.9.1 / triton 3.5.1 are reused rather than
# duplicated -- an earlier attempt let pip resolve transformers' deps freely and it
# dragged in torch 2.14, which shadowed the system torch and broke torchvision's op
# registry. Hence --no-deps and the pins below.
#
#   transformers 4.57.1   (5.x demands torch >= 2.10)
#   tokenizers   0.22.2   (4.57.1 wants >=0.22.0,<=0.23.0)
#   hf_hub       0.35.3   (4.57.1 wants >=0.34.0,<1.0)
#
# `pip install --target` leaves the OLD .dist-info behind when upgrading in place, so
# importlib.metadata keeps reporting the old version and transformers' version check
# fails on a package that is actually correct. Delete the stale *.dist-info by hand.
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/pylibs${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
