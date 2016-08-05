# DECT
DECT derivative-free equality-constrained optimization

### Dependencies

Eigen matrix library: http://eigen.tuxfamily.org

Eigen is a header-only matrix library, and must be available in your default include path.

### Benchmark

Simply run `make` in your checkout directory to build the benchmark program.


### Usage

To use DECT in your own program, you only need the 'dect.h/cpp' files.

The DectModel class is the main entry point.

```c++
DectModel *dm = new DectModel(n_dim, n_constraints + 1);

dm->alloc();
dm->initBasis(xstart, radius);
dm->initModel(eval_fn);

DectModel::term_type res = dm->solve();
Eigen::VectorXd xopt = dm->xopt();
```
