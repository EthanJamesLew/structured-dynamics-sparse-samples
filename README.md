# Structured Dynamics Learning Using Sparse Optimization

This repo implements the sparse cyclic recovery example formulated in the paper

> Schaeffer, H., Tran, G., Ward, R., & Zhang, L. (2020). Extracting structured dynamical systems using sparse optimization with very few samples. *Multiscale Modeling & Simulation*, 18(4), 1435-1461.

This implementation seeks to generalize the implementation so it can be used across a range of system identification applications and simply so it can be run in native python without a Matlab license (also required a DL toolkit license).

## Testing

This implementation tests for a 1-1 output match against the Matlab implementation in the paper referenced [repo](https://github.com/linanzhang/SparseCyclicRecovery).

## Notebooks

Replicates the main matlab script in the [SparseCyclicRecovery repo](https://github.com/linanzhang/SparseCyclicRecovery).