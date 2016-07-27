//
//  rquad.h
//  eigen_rquad
//
//  Created by Christian Miller on 7/3/15.
//  Copyright (c) 2015 ckm. All rights reserved.
//

#ifndef __DECT__
#define __DECT__

#include <vector>
#include <functional>
#include <cassert>

#include <Eigen/Dense>

class DectModel
{
public:
    int n_dim;   // number of dimensions (n)
    int m_pts;   // number of sample points (m)
    int d_vals;  // number of values interpolated by the model (d)
    
    int kopt; // current selected point
    
    Eigen::VectorXd xbase; // (nx1) base point of basis
    Eigen::MatrixXd xpt;   // (nxm) point locations relative to xbase (each point is a column)
    
    // factorization of H (upsmat is symmetric)
    Eigen::MatrixXd zmat, ximat, upsmat;
    
    // run variables
    std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> func;
    double delta; // trust region radius
    double penalty; // penalty parameter in merit function
    
    // statistics
    int num_its; // number of iterations of the current solve
    int num_penalty_updates;
    int num_good_steps;
    int num_bad_steps;
    
    // temporary storage
    double beta;
    Eigen::VectorXd vlag_hi, vlag_lo;
    
    // model storage
    Eigen::MatrixXd fval;            // (mxd) evaluated function values (each column is the interpolation point evals for one output value)
    Eigen::MatrixXd gopt;            // (nxd) gradients of each output value at kopt (each column is a gradient for one output value)
    Eigen::MatrixXd lambda;          // (mxd) implicit model Hessian coeffs for each output value (column-wise as well)
    std::vector<Eigen::MatrixXd> mu; // (d x nxn) explicit model Hessian matrix for each output value (vector is indexed by output value), each matrix is symmetric
    
    // useful subtypes
    enum term_type {
        _None,
        _Solved,
        _TRHit,
        _MaxIters,
        _DegenerateModel,
        _MAX
    };
    static const char* termString(term_type tt);
    
    // constructor / destructor
    DectModel(int pn_dim, int pd_vals = 1);

    // sizes of things
    inline int nz() const { return m_pts - n_dim - 1; }
    
    // setup
    void alloc(); // allocates storage
    void initBasis(const Eigen::VectorXd &base, double r); // constructs initial basis
    void initModel(const std::function<Eigen::VectorXd(const Eigen::Ref<const Eigen::VectorXd>)> &pfunc); // constructs initial models and registers func
    
    // convenience
    inline Eigen::VectorXd xopt() const { return xbase + xpt.col(kopt); }
    inline Eigen::VectorXd fopt() const { return fval.row(kopt).transpose(); }
    
    // return sample location with xbase added
    inline Eigen::VectorXd getActualSample(int i) const { return xbase + xpt.col(i); }
    inline Eigen::MatrixXd getActualSamples() const { return xbase * Eigen::RowVectorXd::Ones(m_pts) + xpt; }
    
    // model operations
    void computeVlagBeta(const Eigen::VectorXd &d);
    bool updateModel(int knew, const Eigen::RowVectorXd &fnew, const Eigen::VectorXd &d); // update the model with a new interpolation point
    void changeKopt(int knew); // shifts the model optimum to knew
    void reselectOpt(double penalty); // re-evaluates for optimum point and moves to it (useful for penalty updates)
    void baseShift(); // shift so that xbase = xopt
    void makeExplicit(int knew); // shift all models' knew-th sample to explicit model
    void zeroExplicit(); // zero out all the explicit contributions
    int  findKfaropt(); // find farthest point from xopt
    int  findKfar(const Eigen::VectorXd &d); // find farthest point from new point xopt+d
    int  findKnewuoa(const Eigen::VectorXd &d); // find NEWUOA replacement point from new point xopt+d
    int  findKbobyqa(const Eigen::VectorXd &d); // find BOBYQA replacement point from new point xopt+d
    int  findKcustom(const Eigen::VectorXd &d); // find custom product from new point xopt+d
    int  findKmax(const Eigen::VectorXd &d); // find max distance-vlag product from new point xopt+d
    int  findKcircles(const Eigen::VectorXd &d, double eta); // find max distance-vlag product from new point xopt+d outside eta*delta, then inside if none are outside
    void updateRadius(double ratio, double dlen); // update the trust region radius according to ratio and step length
    
    // point movement
    Eigen::VectorXd solveTrustRegion(int i);  // integer is the model to optimize
    Eigen::VectorXd solvePoisedness(int knew, double radius); // integer is the interpolation point to replace
    
    // computes a primal-dual step (returns d, with multiplier step in dmults)
    Eigen::VectorXd solveCombinedTrustRegion(const Eigen::Ref<const Eigen::VectorXd> mults, Eigen::Ref<Eigen::VectorXd> dmults);
    bool clampRay(double &a, double rad, const Eigen::Ref<const Eigen::VectorXd> d, const Eigen::Ref<const Eigen::VectorXd> s);
    
    // experimental junk
    Eigen::VectorXd solveLsqrConstraints();
    Eigen::VectorXd solveLsqrLambdas(const Eigen::Ref<const Eigen::VectorXd> mults);
    
    // evaluation
    Eigen::RowVectorXd predictDifference(const Eigen::Ref<const Eigen::VectorXd> de);
    Eigen::VectorXd multiplyOneHessian(const Eigen::Ref<const Eigen::VectorXd> de, int i);
    Eigen::VectorXd multiplyFullHessian(const Eigen::Ref<const Eigen::VectorXd> de, const Eigen::Ref<const Eigen::VectorXd> weights);
    
    // the main solve function
    term_type solve(int maxits);
    
    // debug functions
    Eigen::MatrixXd getW();
    Eigen::MatrixXd getH();
    Eigen::MatrixXd checkConsistency(); // return diffs between interpolation points and predictions (mxd matrix)
    Eigen::MatrixXd getHess(const Eigen::Ref<const Eigen::VectorXd> mults); // return subproblem hessian (nxn matrix)
    double condW(); // compute condition number of W
    
    int numFar(); // computes number of sample points outside the trust region
};

#endif
