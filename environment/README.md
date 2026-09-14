# Tested scientific environment

R010 RGBN remediation and the CAS external confirmation were executed with
Homebrew Python 3.10 on Apple Silicon using the exact package versions in
`requirements-lock.txt`. PyTorch selected the MPS backend.

The LaTeX manuscript was built separately with Tectonic. The scientific Python
lock controls model training, inference, metric recomputation, and figure/table
generation.
