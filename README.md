# learning-eit (OLD BELOW, TODO UPDATE README)
Forward solves divergence-form elliptic equation for Neumann/Dirichlet data pairs given a electrical conducitivity field on the unit disk; linear Gaussian Bayesian inversion for Neumann-to-Dirichlet (NtD) and Dirichlet-to-Neumann (DtN) maps; approximates the electrical impedance tomography (EIT) inversion operator with a Fourier Neural Operator-based Operator Recurrent Neural Network (OR-FNO), i.e., DtN input is operator-valued; learns an approximation from this class from noisy input-output data pairs. For comparison, also includes the regularized D-bar method as a direct inversion method for EIT

## Requirements
MATLAB (for solver and data generation)
* MATLAB 2019a
* Partial Differential Equation Toolbox
* Image Processing Toolbox

Python (for training and testing neural operator model)
* Python 3
* PyTorch 1.9.0 or later
* numpy
* scipy

## References
- [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895)
- [Linear and Nonlinear Inverse Problems with Practical Applications](https://epubs.siam.org/doi/book/10.1137/1.9781611972344?mobileUi=0)
- [Deep learning architectures for nonlinear operator functions and nonlinear inverse problems](https://arxiv.org/abs/1912.11090)
- [Convergence Rates for Learning Linear Operators from Noisy Data](https://arxiv.org/abs/2108.12515)
