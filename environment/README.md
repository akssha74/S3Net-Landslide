# Tested scientific environment

The dual-order HR-GLDD corrective runs and post-hoc CAS sensitivity were executed with
Homebrew Python 3.10 on Apple Silicon using the exact package versions in
`requirements-lock.txt`. PyTorch selected the MPS backend.

The LaTeX manuscript was built separately with Tectonic. The scientific Python
lock controls model training, inference, metric recomputation, and figure/table
generation.
