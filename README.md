# learning-eit
learn-eit contains the code used to reproduce the numerical experiments in the paper "[Extension and neural operator approximation of the electrical impedance tomography inverse map](https://arxiv.org/abs/2511.20361)." It includes:
- Training and evaluation scripts for Fourier Neural Operators (FNOs)
- MATLAB data generation routines for EIT forward problems
- Data processing and plotting utilities
- An implementation of the regularized D-bar EIT reconstruction method

The FNO code defaults to running on GPU, if one is available.

## Installation
The command
```
conda env create -f Project.yml
```
creates an environment called ``eit``. [PyTorch](https://pytorch.org/) will be installed in this step.

Activate the environment with
```
conda activate eit
```
and deactivate with
```
conda deactivate
```

## Data
The three datasets in the paper will be made publicly available soon. In the meantime, you may generate your own data in MATLAB (this will take over one day of wall-clock time). The requirements are:
* MATLAB R2019b
* Partial Differential Equation Toolbox
* Image Processing Toolbox

## Other References
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Linear and Nonlinear Inverse Problems with Practical Applications](https://epubs.siam.org/doi/book/10.1137/1.9781611972344?mobileUi=0)
- [Operator learning meets inverse problems: A probabilistic perspective](https://arxiv.org/abs/2508.20207)
- [An operator learning perspective on parameter-to-observable maps](https://arxiv.org/abs/2402.06031)

## Citing
If you use `learn-eit` in an academic paper, please cite the main reference "[Extension and neural operator approximation of the electrical impedance tomography inverse map](https://arxiv.org/abs/2511.20361)" as follows:
```
@article{de2025extension,
  title={Extension and neural operator approximation of the electrical impedance tomography inverse map},
  author={{de Hoop}, Maarten V and Kovachki, Nikola B and Lassas, Matti and Nelsen, Nicholas H},
  journal={preprint arXiv:2511.20361},
  year={2025}
}
```

## Contribute
You are welcome to submit an issue for any questions related to `learning-eit` or to contribute to the code by submitting pull requests.

## Acknowledgements
The FNO implementation in `learning-eit` is adapted from the [original implementation](https://github.com/neuraloperator/neuraloperator/tree/master) by Nikola Kovachki and Zongyi Li and its modifications in [fourier-neural-mappings](https://github.com/nickhnelsen/fourier-neural-mappings). The data generation code and regularized D-bar solver are adapted from the publicly available code for the book [Linear and Nonlinear Inverse Problems with Practical Applications](https://epubs.siam.org/doi/book/10.1137/1.9781611972344?mobileUi=0) by Jennifer L. Mueller and Samuli Siltanen. The `matplotlib` formatting used to produce figures is adapted from the [PyApprox package](https://github.com/sandialabs/pyapprox) by John Jakeman.
