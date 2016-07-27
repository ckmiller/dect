//
//  rquad.cpp
//  eigen_rquad
//
//  Created by Christian Miller on 7/3/15.
//  Copyright (c) 2015 ckm. All rights reserved.
//

#include "dect.h"

#include <iostream>
#include <iomanip>

using namespace Eigen;
using namespace std;

// ========================== DectModel ==========================

const char* DectModel::termString(DectModel::term_type tt)
{
    static const char* tstrings[] = {
        "None",
        "Solved",
        "TRHit",
        "MaxIters",
        "DegenerateModel",
    };
    
    assert(tt < _MAX);
    
    return tstrings[tt];
}

DectModel::DectModel(int pn_dim, int pd_vals)
{
    n_dim   = pn_dim;
    m_pts   = 2 * n_dim + 1;
    d_vals  = pd_vals;
    
    kopt = -1;
}

void DectModel::alloc()
{
    xbase.resize(n_dim);
    xpt.resize(n_dim, m_pts);
    
    zmat.resize(m_pts, nz());
    ximat.resize(n_dim, m_pts);
    upsmat.resize(n_dim, n_dim);
    
    fval.resize(m_pts, d_vals);
    gopt.resize(n_dim, d_vals);
    lambda.resize(m_pts, d_vals);
    
    mu.resize(d_vals);
    for (auto &m : mu)
        m.resize(n_dim, n_dim);
}

void DectModel::initBasis(const VectorXd &base, double offset)
{
    assert(base.size() == n_dim && "base vector must be the same size as n_dim");
    
    xbase = base;
    kopt = 0;
    
    // initialize xpt and H matrices
    xpt.setZero();
    ximat.setZero();
    zmat.setZero();
    upsmat.setZero();

    // loop over every variable
    for (int i = 0; i < n_dim; i++)
    {
        // set up initial points in x matrix as +/- offset
        xpt(i, 2*i)     =  offset;
        xpt(i, 2*i+1)   = -offset;
        
        // set elements of xi matrix
        ximat(i, 2*i)   =  0.5 / offset;
        ximat(i, 2*i+1) = -0.5 / offset;
        
        // and z matrix
        zmat(0, i)      = -sqrt(2.0) / (offset * offset);
        zmat(2*i, i)    = 1.0 / (sqrt(2.0) * offset * offset);
        zmat(2*i+1, i)  = 1.0 / (sqrt(2.0) * offset * offset);
        
        // nothing happens to ups matrix, it stays zero
    }
    
    // initial trust-region radius is the offset
    delta = offset;
}

void DectModel::initModel(const std::function<VectorXd(const Ref<const VectorXd>)> &pfunc)
{
    // register the function
    func = pfunc;
    
    // evaluate all initial points
    for (int i = 0; i < m_pts; i++)
        fval.row(i) = func(getActualSample(i));
    
    // zero out explicit model contributions
    for (auto &m : mu)
        m.setZero();
    
    // multiply fval by H to get initial lambda and gopt
    lambda = zmat * (zmat.transpose() * fval);
    gopt = ximat * fval;
}

void DectModel::computeVlagBeta(const VectorXd &d)
{
    int i;
    double dx, dsq;
    VectorXd w(m_pts);
    VectorXd ytd, ytxopt, ztw;
    VectorXd xopt = xpt.col(kopt);
    
    // compute first m elements of (w - v), after some tricky algebra using d = x^+ - x_opt
    ytd = xpt.transpose() * d;
    ytxopt = xpt.transpose() * xopt;
    for (i = 0; i < m_pts; i++)
        w(i) = ytd(i) * (0.5 * ytd(i) + ytxopt(i));
    
    // compute vlag (that is, Hw = H(w-v) + e_opt)
    ztw = zmat.transpose() * w;
    vlag_hi = zmat * ztw + ximat.transpose() * d;
    vlag_hi(kopt) += 1.0;
    vlag_lo = ximat * w + upsmat * d;
    
    // compute beta, using more tricky algebra (WTF Powell)
    dx = d.dot(xopt);
    dsq = d.squaredNorm();
    beta = dx * dx + dsq * (xopt.squaredNorm() + 2.0 * dx + 0.5 * dsq); // 0.5 * mag(x^+)^4 term - (2 * w_opt - v_opt) as well
    beta -= ztw.squaredNorm() + 2.0 * d.dot(ximat * w) + d.dot(upsmat * d); // -w^T * H * w term
}

bool DectModel::updateModel(int knew, const RowVectorXd &fnew, const VectorXd &d)
{
    int i, j;
    double temp, tempa, tempb;
    double alpha, tau, denom;
    double ztest;
    VectorXd hcol_hi(m_pts), hcol_lo(n_dim); // _hi is the first m elts of H(:,knew), _lo is the last n
    
    assert(knew != kopt); // never replace existing optimum
    
    computeVlagBeta(d);
    
    // compute alpha and tau, then denominator
    alpha = zmat.row(knew).squaredNorm();
    tau = vlag_hi(knew);
    denom = alpha * beta + tau * tau;
    
    assert(isfinite(alpha));
    assert(isfinite(beta));
    assert(isfinite(tau));
    assert(isfinite(denom));
    
    if (denom <= 0.0)
    {
        /*cout << "Degenerate model!" << endl;
        cout << alpha << endl;
        cout << beta << endl;
        cout << tau << endl;
        cout << denom << endl;
        cout << "W = \n" << getW() << endl;
        cout << "H = \n" << getH() << endl;
        cout << "W*H = \n" << getW()*getH() << endl;*/
        return false;
    }
    
    // get the knew-th column of H
    hcol_hi = zmat * zmat.row(knew).transpose();
    hcol_lo = ximat.col(knew);
    
    // -------- update H --------
    
    // calculate zero tolerance
    ztest = max(fabs(zmat.minCoeff()), fabs(zmat.maxCoeff())) * 1.0e-20;
    
    // rotate zmat s.t. knew-th row has all zeroes except first column
    for (j = 1; j < nz(); j++)
    {
        tempb = zmat(knew, j);
        
        if (fabs(tempb) > ztest)
        {
            tempa = zmat(knew, 0);
            temp = sqrt(tempa * tempa + tempb * tempb);
            
            tempa /= temp;
            tempb /= temp;
            
            for (i = 0; i < m_pts; i++)
            {
                temp = tempa * zmat(i, 0) + tempb * zmat(i, j);
                zmat(i, j) = tempa * zmat(i, j) - tempb * zmat(i, 0);
                zmat(i, 0) = temp;
            }
        }
        
        zmat(knew, j) = 0.0;
    }
    
    // remove 1 from vlag to form -(e_knew - e_kopt - H(w - v))
    vlag_hi(knew) -= 1.0;
    
    // now we can update just the first column of zmat
    temp = sqrt(denom);
    tempb = zmat(knew, 0) / temp;
    tempa = tau / temp;
    for (i = 0; i < m_pts; i++)
        zmat(i, 0) = tempa * zmat(i, 0) - tempb * vlag_hi(i);
    
    // update ximat and upsmat (FIXME: could be made more efficient if necessary)
    ximat += ((alpha / denom) * vlag_lo) * vlag_hi.transpose() +
             ((-beta / denom) * hcol_lo) * hcol_hi.transpose() +
             (( -tau / denom) * hcol_lo) * vlag_hi.transpose() +
             (( -tau / denom) * vlag_lo) * hcol_hi.transpose();
    upsmat += ((alpha / denom) * vlag_lo) * vlag_lo.transpose() +
              ((-beta / denom) * hcol_lo) * hcol_lo.transpose() +
              (( -tau / denom) * hcol_lo) * vlag_lo.transpose() +
              (( -tau / denom) * vlag_lo) * hcol_lo.transpose();
    
    // -------- done update H --------
    
    // -------- now update models --------
    
    // kick old point out of implicit basis
    makeExplicit(knew);
    
    RowVectorXd prediction = predictDifference(d); // calculate predicted value at new point
    RowVectorXd df = fnew - fval.row(kopt); // calculate actual diff
    RowVectorXd ratio = df.cwiseQuotient(prediction); // and its ratio against prediction
    fval.row(knew) = fnew;
    
    df -= prediction; // subtract off prediction to get diff adjustment for model
    
    // replace interpolation point knew with the new point
    xpt.col(knew) = xpt.col(kopt) + d;
    
    // update model with new values
    MatrixXd hdf = zmat * (zmat.row(knew).transpose() * df);
    VectorXd ytd = xpt.transpose() * xpt.col(kopt);
    
    lambda += hdf;
    gopt += ximat.col(knew) * df;
    for (i = 0; i < d_vals; i++)
        gopt.col(i) += xpt * (hdf.col(i).asDiagonal() * ytd);
    
    // gopt.unaryExpr([](double v) { assert(isfinite(v)); return v; });
    
    // -------- done update models --------
    
    return true;
}

void DectModel::changeKopt(int knew)
{
    VectorXd d = xpt.col(knew) - xpt.col(kopt);
    
    // shift gradient to be at kopt
    VectorXd ytd = xpt.transpose() * d;
    for (int i = 0; i < d_vals; i++)
    {
        gopt.col(i) += xpt * (lambda.col(i).asDiagonal() * ytd);
        gopt.col(i) += mu[i].selfadjointView<Lower>() * d;
    }
    
    // change optimum index
    kopt = knew;
}

void DectModel::reselectOpt(double penalty)
{
    int kmin = -1;
    double vmin = 1e300;
    
    for (int i = 0; i < m_pts; i++)
    {
        double newmin = fval(i, 0) + penalty * fval.row(i).tail(d_vals-1).norm();
        
        if (newmin < vmin)
        {
            kmin = i;
            vmin = newmin;
        }
    }
    
    changeKopt(kmin);
}

void DectModel::baseShift()
{
    int i, j;
    MatrixXd gamma(n_dim, m_pts), gz;
    VectorXd xopt = xpt.col(kopt);
    double xoptsq, ytx;
    
    // zmat is unchanged by a base shift
    
    xoptsq = xopt.squaredNorm();
    
    // form the gamma matrix from the paper (after some algebra)
    for (i = 0; i < m_pts; i++)
    {
        ytx = xopt.dot(xpt.col(i));
        gamma.col(i) = xpt.col(i) * (ytx - 0.5 * xoptsq) + xopt * (0.5 * (xoptsq - ytx));
    }
    gz = gamma * zmat;
    
    // make update to upsmat first
    for (i = 0; i < n_dim; i++)
    {
        // TODO: make this symmetric and simplify
        for (j = 0; j <= i; j++)
        {
            upsmat(i, j) +=
                gamma.row(i).dot(ximat.row(j)) +
                gamma.row(j).dot(ximat.row(i)) +
                gz.row(i).dot(gz.row(j));
            
            upsmat(j, i) = upsmat(i, j); // maintain symmetry
        }
    }
    
    // then ximat
    ximat += gz * zmat.transpose();
    
    // do a rank-two update of the explicit parts of the models to account for the new base
    VectorXd ul(n_dim);
    for (i = 0; i < d_vals; i++)
    {
        ul = xpt * lambda.col(i) - 0.5 * lambda.col(i).sum() * xopt;
        mu[i].selfadjointView<Lower>().rankUpdate(ul, xopt, 1.0);
    }
    
    // update all points
    xbase += xopt;
    for (i = 0; i < m_pts; i++)
        xpt.col(i) -= xopt;
}

MatrixXd DectModel::getW()
{
    MatrixXd W(m_pts+n_dim+1, m_pts+n_dim+1);
    
    W.setZero();
    W.block(0, 0, m_pts, m_pts) = 0.5 * (xpt.transpose() * xpt).unaryExpr([](double v){return v*v;});
    W.block(m_pts, 0, n_dim, m_pts) = xpt;
    W.block(m_pts+n_dim, 0, 1, m_pts).setOnes();
    W.block(0, m_pts, m_pts, n_dim) = xpt.transpose();
    W.block(0, m_pts+n_dim, m_pts, 1).setOnes();
    
    return W;
}

MatrixXd DectModel::getH()
{
    MatrixXd H(m_pts+n_dim+1, m_pts+n_dim+1);
    
    H.setZero();
    H.block(0, 0, m_pts, m_pts) = zmat * zmat.transpose();
    H.block(m_pts, 0, n_dim, m_pts) = ximat;
    H.block(0, m_pts, m_pts, n_dim) = ximat.transpose();
    H.block(m_pts, m_pts, n_dim, n_dim) = upsmat;
    
    // manually compute last col, since we don't actually keep track of it
    // see BOBYQA paper for details
    VectorXd v(m_pts+n_dim+1);
    
    v.setZero();
    v.segment(0, m_pts) = 0.5 * (xpt.transpose() * xpt.col(kopt)).unaryExpr([](double v){return v*v;});
    v.segment(m_pts, n_dim) = xpt.col(kopt);
    
    v = -(H*v);
    v(kopt) += 1.0;
    
    H.row(m_pts+n_dim) = v;
    H.col(m_pts+n_dim) = v;
    
    return H;
}

MatrixXd DectModel::checkConsistency()
{
    MatrixXd res(m_pts, d_vals);
    
    for (int i = 0; i < m_pts; i++)
        res.row(i) = fval.row(i) - (predictDifference(xpt.col(i) - xpt.col(kopt)) + fval.row(kopt));
    
    return res;
}

MatrixXd DectModel::getHess(const Ref<const VectorXd> mults)
{
    // set amults to (1, mults), used for convenience
    VectorXd amults(d_vals);
    amults(0) = 1.0;
    amults.tail(d_vals-1) = mults;
    
    MatrixXd res(n_dim, n_dim);
    MatrixXd tmat(n_dim, n_dim);
    res.setZero();
    
    VectorXd weighted_lambda = lambda * amults;
    res = xpt * (weighted_lambda.asDiagonal() * xpt.transpose());
    
    // this is super expensive...
    VectorXd tmp;
    for (int i = 0; i < d_vals; i++)
    {
        double w = amults(i);
        tmat = mu[i].selfadjointView<Lower>();
        if (w != 0.0)
            res += w * tmat;
    }
    
    return res;
}

double DectModel::condW()
{
    JacobiSVD<MatrixXd> svd(getW());
    double cond = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size()-1);
    return cond;
}

int DectModel::numFar()
{
    int nfar = 0;
    
    for (int i = 0; i < m_pts; i++)
        if ((xpt.col(i) - xpt.col(kopt)).norm() > delta)
            nfar++;
    
    return nfar;
}

void DectModel::makeExplicit(int k)
{
    // adds a lambda term to the explicit matrix mu
    for (int i = 0; i < d_vals; i++)
        mu[i].selfadjointView<Lower>().rankUpdate(xpt.col(k), lambda(k, i));
    
    // zero out the corresponding lambda
    lambda.row(k).setZero();
}

void DectModel::zeroExplicit()
{
    for (int i = 0; i < d_vals; i++)
        mu[i].setZero();
}

int DectModel::findKfaropt()
{
    double maxdist = 0.0;
    int kfar = -1;
    
    for (int i = 0; i < m_pts; i++)
    {
        // skip current optimum
        if (i == kopt)
            continue;
        
        double dist = (xpt.col(kopt) - xpt.col(i)).squaredNorm();
        
        if (dist > maxdist)
        {
            maxdist = dist;
            kfar = i;
        }
    }
    
    assert(kfar > -1);
    
    return kfar;
}

int DectModel::findKfar(const VectorXd &d)
{
    double maxdist = 0.0;
    int kfar = -1;
    
    for (int i = 0; i < m_pts; i++)
    {
        // skip current optimum
        if (i == kopt)
            continue;
        
        double dist = (d + xpt.col(kopt) - xpt.col(i)).squaredNorm();
        
        if (dist > maxdist)
        {
            maxdist = dist;
            kfar = i;
        }
    }
    
    assert(kfar > -1);
    
    return kfar;
}

int DectModel::findKnewuoa(const VectorXd &d)
{
    int i;
    int ix = -1;
    double alpha, tau, tmp, sigma, id = -1.0e300;
    
    computeVlagBeta(d);
    
    // find worst point
    for (i = 0; i < m_pts; i++)
    {
        // never pick current optimum
        if (i == kopt)
            continue;
        
        // compute alpha and tau, then denominator
        alpha = zmat.row(i).squaredNorm();
        tau = vlag_hi(i);
        sigma = alpha * beta + tau * tau;
        
        // cout << "denom " << i << ": " << sigma << ", ";
        
        tmp = (xpt.col(kopt) + d - xpt.col(i)).norm() / (0.1 * delta);
        sigma *= std::fmax(1.0, tmp * tmp * tmp * tmp * tmp * tmp);
        
        // cout << tmp << " = " << sigma << endl;
        
        // biggest denominator
        if (sigma > id)
        {
            ix = i;
            id = sigma;
        }
    }
    
    assert(ix > -1);
    
    return ix;
}

int DectModel::findKbobyqa(const VectorXd &d)
{
    int i;
    int ix = -1;
    double alpha, tau, sigma, tmp, id = -1.0e300;
    
    computeVlagBeta(d);
    
    // find worst point
    for (i = 0; i < m_pts; i++)
    {
        // never pick current optimum
        if (i == kopt)
            continue;
        
        // compute alpha and tau, then denominator
        alpha = zmat.row(i).squaredNorm();
        tau = vlag_hi(i);
        sigma = alpha * beta + tau * tau;
        
        // cout << "denom " << i << ": " << sigma << ", ";
        
        tmp = std::fmax(0.1, (xpt.col(kopt) - xpt.col(i)).squaredNorm() / (delta * delta));
        sigma *= tmp;
        
        // cout << tmp << " = " << sigma << endl;
        
        // biggest denominator
        if (sigma > id)
        {
            ix = i;
            id = sigma;
        }
    }
    
    assert(ix > -1);
    
    return ix;
}

int DectModel::findKcustom(const VectorXd &d)
{
    int i;
    int ix = -1;
    double alpha, tau, sigma, tmp, id = -1.0e300;
    double dfloor = log(1e-12);
    
    computeVlagBeta(d);
    
    // find worst point
    for (i = 0; i < m_pts; i++)
    {
        // never pick current optimum
        if (i == kopt)
            continue;
        
        // compute alpha and tau, then denominator
        alpha = zmat.row(i).squaredNorm();
        tau = vlag_hi(i);
        sigma = alpha * beta + tau * tau;
        
        //cout << "denom " << i << ": " << sigma << ", ";
        
        tmp = (xpt.col(kopt) + d - xpt.col(i)).norm() / delta;
        
        if (sigma < 1e-12)
            sigma = 1e-12;
        
        //cout << (log(sigma) - dfloor) << ", ";
        
        sigma = (log(sigma) - dfloor) * log(tmp);
        
        //cout << tmp << " = " << sigma << endl;
        
        // biggest denominator
        if (sigma > id)
        {
            ix = i;
            id = sigma;
        }
    }
    
    assert(ix > -1);
    
    return ix;
}

int DectModel::findKmax(const Eigen::VectorXd &d)
{
    int i;
    int ix = -1;
    double tau, sigma, id = -1.0e300;
    
    computeVlagBeta(d);
    
    // find worst point
    for (i = 0; i < m_pts; i++)
    {
        // never pick current optimum
        if (i == kopt)
            continue;
        
        double dist = (xpt.col(kopt) + d - xpt.col(i)).norm() / delta;
        
        tau = vlag_hi(i);
        sigma = dist * dist * std::abs(tau);
        
        // cout << "denom " << i << ": " << dist * dist << " * " << std::abs(tau) << " = " << sigma << endl;
        
        // biggest denominator
        if (sigma > id)
        {
            ix = i;
            id = sigma;
        }
    }
    
    assert(ix > -1);
    
    return ix;
}

int DectModel::findKcircles(const Eigen::VectorXd &d, double eta)
{
    int i;
    int ix_near = -1, ix_far = -1;
    double tau, sigma, id_near = -1.0e300, id_far = -1.0e300;
    
    computeVlagBeta(d);
    
    // find worst point
    for (i = 0; i < m_pts; i++)
    {
        // never pick current optimum
        if (i == kopt)
            continue;
        
        double dist = (xpt.col(kopt) + d - xpt.col(i)).norm() / delta;
        
        tau = vlag_hi(i);
        sigma = dist * dist * std::abs(tau);
        
        // biggest denominator
        if (dist >= eta)
        {
            if (sigma > id_far)
            {
                ix_far = i;
                id_far = sigma;
            }
        }
        else
        {
            if (sigma > id_near)
            {
                ix_near = i;
                id_near = sigma;
            }
        }
    }
    
    assert(ix_near > -1 || ix_far > -1);
    
    if (ix_far > -1)
        return ix_far;
    else
        return ix_near;
}

void DectModel::updateRadius(double ratio, double dlen)
{
    // shrink / grow trust region based on ratio
    if (ratio <= 0.1)
        delta = 0.5 * delta; //std::min(0.5 * delta, dlen);
    else if (ratio <= 0.7)
        delta = std::max(0.5 * delta, dlen);
    else
        delta = std::max(0.5 * delta, 2.0 * dlen);
}

VectorXd DectModel::solveTrustRegion(int i)
{
    VectorXd d(n_dim), r(n_dim), p(n_dim), Ap(n_dim);
    double alpha, rr, rrold, curv, maxstep;
    
    d.setZero();
    r = -gopt.col(i);
    p = r;
    rrold = r.squaredNorm();
    
    for (int k = 0; k < n_dim; k++)
    {
        Ap = multiplyOneHessian(p, i);
        
        // maximum step in the search direction
        double pp = p.squaredNorm();
        double pd = p.dot(d);
        double dd = d.squaredNorm();
        
        // solve quadratic equation for step to trust region boundary
        double temp = delta * delta - dd;
        maxstep = temp / (pd + sqrt(pd * pd + pp * temp));
        
        curv = p.dot(Ap); // curvature in the search direction
        
        alpha = maxstep;
        if (curv > 0.0)
            alpha = std::fmin(alpha, rrold / curv);
        
        d += alpha * p;
        
        if (alpha == maxstep)
            break;
        
        r -= alpha * Ap;
        
        rr = r.squaredNorm();
        
        if (sqrt(rr) < 1e-10)
            break;
        
        p *= rr / rrold;
        p += r;
        
        rrold = rr;
    }
    
    return d;
}

VectorXd DectModel::solvePoisedness(int knew, double radius)
{
    VectorXd yx(n_dim), hcol(m_pts), glag(n_dim), xnew(n_dim);
    double minstep, maxstep, dgrad, ha;
    double bestval, beststep, bestdenom, temp;
    int i;
    
    bestdenom = 0.0;
    
    // compute the first m elements of the knew-th column of H
    hcol = zmat * zmat.row(knew).transpose();
    
    // compute 0.5 * alpha, used in denominator estimate
    ha = 0.5 * hcol(knew);
    
    // compute gradient of the knew-th lagrange function at xopt
    glag = ximat.col(knew) + xpt * (hcol.asDiagonal() * (xpt.transpose() * xpt.col(kopt)));
    
    // do line search along lines from xopt through each interpolation point
    for (i = 0; i < m_pts; i++)
    {
        if (i == kopt)
            continue;
        
        yx = xpt.col(i) - xpt.col(kopt); // line from xopt through y_i
        dgrad = yx.dot(glag); // directional derivative along that line
        
        // distance to the edges of the trust region
        maxstep = radius / yx.norm();
        minstep = -maxstep;
        
        // check relevant steps, pick the best for this line
        if (i == knew)
        {
            // if i == knew, then phi(1) = 1
            
            // trust region bounds first
            temp = fabs(maxstep * (dgrad - (dgrad - 1.0) * maxstep));
            bestval = temp;
            beststep = maxstep;
            
            temp = fabs(minstep * (dgrad - (dgrad - 1.0) * minstep));
            if (temp > bestval)
            {
                bestval = temp;
                beststep = minstep;
            }
            
            // then check critical point, if within bounds (checked without potential divide by zero)
            if ((2.0 * (dgrad - 1.0) * minstep) < dgrad && (2.0 * (dgrad - 1.0) * maxstep) > dgrad)
            {
                temp = fabs(0.25 * dgrad * dgrad / (dgrad - 1.0));
                if (temp > bestval)
                {
                    bestval = temp;
                    beststep = dgrad / (2.0 * (dgrad - 1.0));
                }
            }
        }
        else
        {
            // if i != knew, then phi(0) = 0, and things are a bit simpler
            
            // trust region bounds first
            temp = fabs(dgrad * maxstep * (1.0 - maxstep));
            bestval = temp;
            beststep = maxstep;
            
            temp = fabs(dgrad * minstep * (1.0 - minstep));
            if (temp > bestval)
            {
                bestval = temp;
                beststep = minstep;
            }
            
            // then critical point, which is guaranteed to be at phi(1/2)
            temp = fabs(0.25 * dgrad);
            if (temp > bestval && minstep < 0.5 && maxstep > 0.5)
            {
                bestval = temp;
                beststep = 0.5;
            }
        }
        
        // see if we've found a better denominator estimate with this point
        temp = yx.squaredNorm() * beststep * (1.0 - beststep);
        temp = bestval * bestval * (ha * temp * temp + bestval * bestval);
        
        if (temp > bestdenom)
        {
            bestdenom = temp;
            xnew = beststep * yx;
            
            assert(xnew.norm() <= radius + 1e-5);
        }
    }
    
    // now check constrained cauchy step, replace if it gives a better derivative estimate
    double dg, dhd;
    
    dg = glag.squaredNorm();
    dhd = glag.dot(xpt * (hcol.asDiagonal() * (xpt.transpose() * glag)));
    
    maxstep = radius / sqrt(dg);
    minstep = -maxstep;
    
    // trust region bounds first
    temp = fabs(maxstep * (dg + 0.5 * maxstep * dhd));
    bestval = temp;
    beststep = maxstep;
    
    temp = fabs(minstep * (dg + 0.5 * minstep * dhd));
    if (temp > bestval)
    {
        bestval = temp;
        beststep = minstep;
    }
    
    // then critical point along gradient
    if (dhd * minstep < -dg && dhd * maxstep > -dg)
    {
        temp = fabs(-0.5 * dg * dg / dhd);
        if (temp > bestval)
        {
            bestval = temp;
            beststep = -dg / dhd;
        }
    }
    
    // see if we've found a better denominator estimate with this point
    temp = glag.squaredNorm() * beststep * (1.0 - beststep);
    temp = bestval * bestval * (ha * temp * temp + bestval * bestval);
    
    if (temp > bestdenom)
    {
        bestdenom = temp;
        xnew = beststep * glag;
        assert(xnew.norm() <= radius + 1e-5);
    }

    return xnew;
}

bool DectModel::clampRay(double &a, double rad, const Ref<const VectorXd> d, const Ref<const VectorXd> s)
{
    double ss, ds, dd, temp, a1, a2, org_a = a;
    
    ss = s.squaredNorm();
    ds = d.dot(s);
    dd = d.squaredNorm();
    
    if (ss == 0.0)
        return false;
    
    temp = rad * rad - dd;
    temp = ds * ds + ss * temp;
    assert(temp >= 0.0);
    temp = sqrt(temp);
    
    a1 = (-ds + temp) / ss;
    a2 = (-ds - temp) / ss;
    
    if (a1 < a2)
        a = std::max(a1, std::min(a, a2));
    else
        a = std::max(a2, std::min(a, a1));
    
    return (a != org_a);
}

#define VERBOSE 0
#define LSQR_FIRST 1

VectorXd DectModel::solveCombinedTrustRegion(const Ref<const VectorXd> mults, Ref<VectorXd> dmults)
{
    // adapted from minresSOL.m, available here: http://web.stanford.edu/group/SOL/software/minres/
    // authors: Michael Saunders, SOL, Stanford University
    //          Sou Cheng Choi,  SCCM, Stanford University
    
    assert(mults.size() == d_vals-1);
    dmults.setZero();
    
    const Ref<const MatrixXd> con_mat(gopt.rightCols(d_vals-1).transpose());       // constraint matrix
    const Ref<const VectorXd> con_val(fval.row(kopt).tail(d_vals-1).transpose());  // constraint values
    // double con_mat_norm = con_mat.squaredNorm();                                   // constraint matrix norm
    
    VectorXd x(n_dim + d_vals-1);
    x.setZero();
    VectorXd r1(x.size()), y(x.size());
    
    // set amults to (1, mults), used for convenience
    VectorXd amults(d_vals);
    amults(0) = 1.0;
    amults.tail(d_vals-1) = mults;
    
    // do lsqr step first
    VectorXd ls_step(n_dim);
    ls_step.setZero();
    
#if LSQR_FIRST
    if (d_vals > 1)
        ls_step = solveLsqrConstraints();
#endif
    
    // if that hit bound, stop there
    if (ls_step.norm() > delta - 1e-5)
        return ls_step;
    
#if VERBOSE
    IOFormat mf(StreamPrecision, 0, ", ", "; ", "", "", "[", "]");
    cout << "ls = " << ls_step.format(mf) << ";\n";
    cout << "W = " << getHess(mults).format(mf) << ";\n";
    cout << "g = " << (gopt * amults + multiplyFullHessian(ls_step, amults)).format(mf) << ";\n";
    cout << "A = " << con_mat.format(mf) << ";\n";
#if LSQR_FIRST
    cout << "c = " << VectorXd::Zero(con_val.rows()).format(mf) << ";\n";
#else
    cout << "c = " << con_val.format(mf) << ";\n";
#endif
#endif
    
    double eps = 1e-5;
    double hmod = 0.0; // used to track Hessian modifications
    
    int istop(0), itn(0), max_iter(2 * (int)x.size());
    double Anorm(0.0), Acond(0.0), Arnorm(0.0);
    double rnorm(0.0), ynorm(0.0);
    bool show(false);
    double tol = 1e-5;
    
    // Step 1
    /*
     * Set up y and v for the first Lanczos vector v1.
     * y = beta1 P' v1, where P = C^(-1).
     * v is really P'v1
     */
    
    // calculate initial residual and step direction
    
    // full step version
    r1.head(n_dim) = gopt * amults;
    r1.tail(d_vals-1) = con_val;

#if LSQR_FIRST
    // precomputed normal step version
    r1.head(n_dim) += multiplyFullHessian(ls_step, amults);
    r1.tail(d_vals-1).setZero();
#endif
    
    r1 = -r1; // negate initial gradient
    y = r1;
    
    double beta1(0.0);
    beta1 = r1.dot(y);
    
    // If b = 0 exactly stop with x = x0.
    if (beta1 == 0.0)
        return x.head(n_dim) + ls_step;
    
    beta1 = sqrt(beta1); // Normalize y to get v1 later
    
    // STEP 2
    /* Initialize other quantities */
    double oldb(0.0), beta(beta1), dbar(0.0), epsln(0.0), oldeps(0.0);
    double qrnorm(beta1), phi(0.0), phibar(beta1), rhs1(beta1);
    double rhs2(0.0), tnorm2(0.0), ynorm2(0.0);
    double cs(-1.0), sn(0.0);
    double gmax(0.0), gmin(1e300);
    double alpha(0.0), gamma(0.0);
    double mdelta(0.0), gbar(0.0);
    double z(0.0);
    bool first = true;
    
    VectorXd w(x), w1(x), w2(x), r2(r1), tmpv;
    w.setZero();
    w1.setZero();
    w2.setZero();
    VectorXd v(x);
    
    if(show)
    {
        std::cout<<std::setw(6)<<"Itn"
        << std::setw(14) << "Compatible"
        << std::setw(14) << "LS"
        << std::setw(14) << "norm(A)"
        << std::setw(14) << "cond(A)"
        << std::setw(14) << "gbar/|A|"<<"\n";
    }
    
    /* Main Iteration */
    for (itn = 0; itn < max_iter; ++itn)
    {
        // STEP 3
        /*
         -----------------------------------------------------------------
         Obtain quantities for the next Lanczos vector vk+1, k = 1, 2,...
         The general iteration is similar to the case k = 1 with v0 = 0:
         
         p1      = Operator * v1  -  beta1 * v0,
         alpha1  = v1'p1,
         q2      = p2  -  alpha1 * v1,
         beta2^2 = q2'q2,
         v2      = (1/beta2) q2.
         
         Again, y = betak P vk,  where  P = C**(-1).
         .... more description needed.
         -----------------------------------------------------------------
         */
        v = y / beta; // Normalize previous vector (in y), v = vk if P = I
        
        // multiply hessian block
        y.head(n_dim)  = multiplyFullHessian(v.head(n_dim), amults); // multiply W * v
        y.head(n_dim) += hmod * v.head(n_dim); // add in hessian modification hmod * v
        
        // compute constraint jacobian parts
        y.head(n_dim)  += gopt.rightCols(d_vals-1) * v.tail(d_vals-1);
        y.tail(d_vals-1) = gopt.rightCols(d_vals-1).transpose() * v.head(n_dim);
        
        if (!first)
            y -= (beta / oldb) * r1;
        
        alpha = v.dot(y); // alphak
        y -= (alpha / beta) * r2;
        r1 = r2;
        r2 = y;
        
        oldb = beta; // oldb = betak
        beta = r2.dot(y); // beta = betak+1^2
        
        if (beta < 0)
        {
            istop = 9;
            break;
        }
        
        beta = sqrt(beta);
        tnorm2 += alpha * alpha + oldb * oldb + beta * beta;
        
        if (first) // Initialize a few things
        {
            if (beta / beta1 < 10.0 * eps) // beta2 = 0 or ~ 0.
                istop = 10;
            
            gmax = abs(alpha);
            gmin = gmax;
        }
        
        // Apply previous rotation Q_{k-1} to get
        // [mdelta_k epsln_{k+1}] = [cs sn]  [dbar_k 0]
        // [gbar_k   dbar_{k+1}]   [sn -cs] [alpha_k beta_{k+1}].
        oldeps = epsln;
        mdelta = cs * dbar + sn * alpha;
        gbar   = sn * dbar - cs * alpha;
        epsln  =             sn * beta;
        dbar   =           - cs * beta;
        double root = sqrt(gbar * gbar + dbar * dbar);
        Arnorm = phibar * root; // ||Ar_{k-1}||
        
        // Compute next plane rotation Q_k
        gamma  = sqrt(gbar * gbar + beta * beta); // gamma_k
        gamma  = std::max(gamma, eps);
        cs     = gbar / gamma;                    // c_k
        sn     = beta / gamma;                    // s_k
        phi    = cs * phibar;                     // phi_k
        phibar = sn * phibar;                     // phibar_{k+1}
        
        // Update x
        w1 = w2;
        w2 = w;
        w  = (v - oldeps * w1 - mdelta * w2) / gamma;
        
        double wnorm = w.norm();
        
        double curv = 0.0, ddir = 0.0;
        
        // TODO: any way to avoid calculating this twice?
        tmpv  = multiplyFullHessian(w.head(n_dim), amults);
        tmpv += hmod * w.head(n_dim);
        curv = tmpv.dot(w.head(n_dim));
        ddir = std::abs((gopt * amults).dot(w.head(n_dim)));
        
        // cout << "delta / ddir / curv: " << delta << " / " << ddir << " / " << curv << endl;
        
        // if we seem to be going to a saddle point or maximum, modify hessian and restart
        if (curv < 0.0)
        {
            //cout << "w: " << w.transpose() << ", " << wnorm << endl;
            //cout << "dw, ddw, uns, scale: " << (delta / wnorm) << ", " << ddir * wnorm / delta << ", " << (ddir * wnorm / delta - curv) << ", " << (ddir * wnorm / delta - curv) / (wnorm * wnorm) << endl;
            
            //hmod = std::max(hmod + 0.1, std::max(hmod * 10.0, (ddir * wnorm / delta - curv) / (wnorm * wnorm)));
            //hmod = std::max(hmod + 0.1, std::max(hmod * 1.1, (ddir * wnorm / delta - curv) / (wnorm * wnorm)));
            //hmod = std::max(hmod + 0.1, std::max(hmod * 1.1, -curv / (wnorm * wnorm) + 1e-4));
            hmod = std::max(hmod + 0.1, std::max(hmod * 1.1, std::max(-curv / (wnorm * wnorm) + 1e-4, (ddir * wnorm / delta - curv) / (wnorm * wnorm))));
            assert(hmod >= 0.0);
            
            x.setZero();
            
            // recompute residual
            r1.head(n_dim)   = gopt * amults; // + multiplyFullHessian(ls_step, amults);
            r1.tail(d_vals-1) = con_val;
            
#if LSQR_FIRST
            r1.head(n_dim) += multiplyFullHessian(ls_step, amults);
#endif
            
            // multiply hessian block
            r1.head(n_dim) += multiplyFullHessian(x.head(n_dim), amults); // multiply W * v
            r1.head(n_dim) += hmod * x.head(n_dim); // add in hessian modification hmod * v
            
            // compute constraint jacobian parts
            r1.head(n_dim)   += gopt.rightCols(d_vals-1) * x.tail(d_vals-1);
            r1.tail(d_vals-1) += gopt.rightCols(d_vals-1).transpose() * x.head(n_dim);
            
            r1 = -r1;
            
            beta1 = r1.norm();
            y = r1;
            r2 = r1;
            w.setZero();
            w1.setZero();
            w2.setZero();
            v = x;
            
            oldb = 0.0;
            beta = beta1;
            dbar = 0.0;
            epsln = 0.0;
            oldeps = 0.0;
            qrnorm = beta1;
            phi = 0.0;
            phibar = beta1;
            rhs1 = beta1;
            rhs2 = 0.0;
            tnorm2 = 0.0;
            ynorm2 = 0.0;
            cs = -1.0;
            sn = 0.0;
            gmax = 0.0;
            gmin = 1e300;
            alpha = 0.0;
            gamma = 0.0;
            mdelta = 0.0;
            gbar = 0.0;
            z = 0.0;
            first = true;
            itn--;
            
#if VERBOSE
            cout << "mu = " << hmod << ";\n";
#endif
            continue;
        }
        
        // clamp step to trust region in phase vars
        bool clamped = clampRay(phi, delta, x.head(n_dim), w.head(n_dim));
        
        //cout << "phi = " << phi << (clamped ? " clamp" : "") << endl;
        
        x += phi * w;
        
#if VERBOSE
        cout << "s" << itn << " = " << x.format(mf) << ";\n";
#endif
        
        assert(x.head(n_dim).norm() < delta * 1.1);
        
        //update_list.push_back(x);
        
        // Go round again.
        gmax = std::max(gmax, gamma);
        gmin = std::min(gmin, gamma);
        z    = rhs1 / gamma;
        rhs1 = rhs2 - mdelta * z;
        rhs2 =      - epsln * z;
        
        // Estimate various norms and test for convergence.
        Anorm  = sqrt(tnorm2);
        ynorm2 = x.squaredNorm();
        ynorm  = sqrt(ynorm2);
        double epsa = Anorm * eps;
        double epsx = epsa * ynorm;
        //double epsr = Anorm * ynorm * tol;
        double diag = gbar;
        if (diag == 0.0)
            diag = epsa;
        
        qrnorm = phibar;
        rnorm  = qrnorm;
        double test1(0.0), test2(0.0);
        test1  = rnorm / (Anorm * ynorm); //  ||r|| / (||A|| ||x||)
        test2  = root / Anorm;            // ||Ar|| / (||A|| ||r||)
        
        // Estimate cond(A)
        /*
         In this version we look at the diagonals of  R  in the
         factorization of the lower Hessenberg matrix,  Q * H = R,
         where H is the tridiagonal matrix from Lanczos with one
         extra row, beta(k+1) e_k^T.
         */
        Acond = gmax / gmin;
        
        //See if any of the stopping criteria is satisfied
        if (istop == 0)
        {
            double t1(1.0+test1), t2(1.0+test2); //This test work if tol < eps
            if (t2 <= 1.) istop = 2;
            if (t1 <= 1.) istop = 1;
            
            if (itn >= max_iter-1) istop = 6;
            if (Acond >= .1/eps)   istop = 4;
            if (epsx >= beta1)     istop = 3;
            if (test2 <= tol)      istop = 2;
            if (test1 <= tol)      istop = 1;
            if (clamped)           istop = 11;
        }
        
        if (show)
            std::cout<< std::setw(6) << itn
            << std::setw(14) << test1
            << std::setw(14) << test2
            << std::setw(14) << Anorm
            << std::setw(14) << Acond
            << std::setw(14) << gbar/Anorm << std::endl;
        
        if (0 != istop)
            break;
        
        first = false;
    }
    
    if (show)
    {
        std::vector<std::string> msg(12);
        msg[0]  = " beta1 = 0.  The exact solution is  x = 0 ";
        msg[1]  = " A solution to Ax = b was found, given tol ";
        msg[2]  = " A least-squares solution was found, given tol ";
        msg[3]  = " Reasonable accuracy achieved, given eps ";
        msg[4]  = " x has converged to an eigenvector ";
        msg[5]  = " acond has exceeded 0.1/eps ";
        msg[6]  = " The iteration limit was reached ";
        msg[7]  = " A  does not define a symmetric matrix ";
        msg[8]  = " M  does not define a symmetric matrix ";
        msg[9]  = " M  does not define a pos-def preconditioner ";
        msg[10] = " beta2 = 0.  If M = I, b and x are eigenvectors ";
        msg[11] = " trust region boundary hit ";
        
        std::cout << "Termination criteria " << istop << ":" << msg[istop] << std::endl;
    }
    
    //resid = rnorm;
    
    if (istop == 11)
        itn = -itn;
    
    dmults = x.tail(d_vals-1);
    
    return x.head(n_dim) + ls_step;
}

VectorXd DectModel::solveLsqrConstraints()
{
    // transcribed from lsqrSOL.m
    double conlim = 1e8;
    double ctol = 0.0;
    double damp = 0.0;
    
    double atol = 1e-5;
    double btol = 1e-5;
    
    int m = d_vals-1;
    int n = n_dim;
    
    int itnlim = 2 * n;
    bool disable = false;
    
    const Ref<const MatrixXd> con_mat(gopt.rightCols(m).transpose());       // constraint matrix
    const Ref<const VectorXd> con_val(fval.row(kopt).tail(m).transpose());  // constraint values
    
#if VERBOSE
    IOFormat mf(StreamPrecision, 0, ", ", "; ", "", "", "[", "]");
    cout << "A = " << con_mat.format(mf) << ";\n";
    cout << "c = " << con_val.format(mf) << ";\n";
#endif
    
    if (conlim > 0)
        ctol = 1.0 / conlim;
    
    int itn = 0, istop = 0;
    double anorm = 0, acond = 0, arnorm = 0, bnorm = 0;
    double dampsq = damp * damp, ddnorm = 0, res1 = 0, res2 = 0;
    double xnorm = 0, xxnorm = 0, Axnorm = 0, z = 0;
    double rhobar = 0, phibar = 0, rnorm = 0, r1norm = 0, r2norm = 0;
    double rhobar1 = 0, psi = 0, phi = 0, tau = 0;
    double delta = 0, gamma = 0, gambar = 0, rhs = 0, zbar = 0;
    double t1 = 0, t2 = 0;
    double rho = 0, cs = 0, sn = 0, theta = 0;
    double cs1 = -1.0, sn1 = 0;
    double cs2 = -1.0, sn2 = 0;
    double test1 = 0, test2 = 0, test3 = 0;
    double rtol = 0;
    bool show = false;
    
    if (show)
    {
        printf("\n");
        printf("LSQR            Least-squares solution of  Ax = b\n");
        printf("The matrix A has %8d rows  and %8d cols\n", m, n);
        printf("damp = %20.14f\n", damp);
        printf("atol = %8.2f                 conlim = %8.2f\n", atol, conlim);
        printf("btol = %8.2f                 itnlim = %8d\n"  , btol, itnlim);
    }
    
    // Set up the first vectors u and v for the bidiagonalization.
    // These satisfy  beta*u = b,  alpha*v = A'u.
    
    VectorXd u(m), x(n), v(n), w(n), dk(n);
    x.setZero();
    
    u = -con_val; // u = -c
    
    double alpha = 0, beta = u.norm();
    if (beta > 0)
    {
        u = (1.0 / beta) * u;
        v = con_mat.transpose() * u; // v = A' * u
        alpha = v.norm();
    }
    if (alpha > 0)
    {
        v = (1.0 / alpha) * v;
        w = v;
    }
    arnorm = alpha * beta;
    if (arnorm == 0)
    {
        //disp(msg(1,:));
        return x;
    }
    
    rhobar = alpha;		phibar = beta;		bnorm  = beta;
    rnorm  = beta;
    r1norm = rnorm;
    r2norm = rnorm;
    Axnorm = 0;
    
    if (show)
    {
        printf(" \n");
        printf("   Itn      x(1)       r1norm     r2norm  Compatible   LS      Norm A   Cond A\n");
        test1  = 1;		test2  = alpha / beta;
        
        printf("%6d %12.5f\n",        itn,   x(1) );
        printf("%10.3f %10.3f\n", r1norm, r2norm );
        printf("%8.1f %8.1f\n",   test1,  test2 );
    }
    
    //------------------------------------------------------------------
    //     Main iteration loop.
    //------------------------------------------------------------------
    while (itn < itnlim)
    {
        //     Perform the next step of the bidiagonalization to obtain the
        //     next  beta, u, alpha, v.  These satisfy the relations
        //                beta*u  =  A*v   -  alpha*u,
        //                alpha*v =  A'*u  -  beta*v.
        
        u    = con_mat * v  -  alpha*u;
        beta = u.norm();
        if (beta > 0.0)
        {
            u     = (1.0 / beta) * u;
            anorm = Vector4d(anorm, alpha, beta, damp).norm();
            v     = con_mat.transpose() * u  -  beta*v;
            alpha  = v.norm();
            if (alpha > 0)
                v = (1.0 / alpha) * v;
        }
        
        //     Use a plane rotation to eliminate the damping parameter.
        //     This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        
        rhobar1 = sqrt(rhobar * rhobar + damp * damp);
        cs1     = rhobar / rhobar1;
        sn1     = damp   / rhobar1;
        psi     = sn1 * phibar;
        phibar  = cs1 * phibar;
        
        //     Use a plane rotation to eliminate the subdiagonal element (beta)
        //     of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        
        rho     =  sqrt(rhobar1 * rhobar1 + beta * beta);
        cs      =  rhobar1/ rho;
        sn      =  beta   / rho;
        theta   =  sn * alpha;
        rhobar  = -cs * alpha;
        phi     =  cs * phibar;
        phibar  =  sn * phibar;
        tau     =  sn * phi;
        Axnorm  =  sqrt(Axnorm * Axnorm + phi * phi);
        
        //     Update x and w.
        
        t1      =  phi / rho;
        t2      = -theta / rho;
        dk      =  w / rho;
        
        bool clamped = clampRay(t1, this->delta, x, w);
        
        x       = x      +  t1*w;
        w       = v      +  t2*w;
        ddnorm  = ddnorm +  dk.squaredNorm();
        
#if VERBOSE
        cout << "s" << itn << " = " << x.format(mf) << ";\n";
#endif
        
        //     Use a plane rotation on the right to eliminate the
        //     super-diagonal element (theta) of the upper-bidiagonal matrix.
        //     Then use the result to estimate  norm(x).
        
        delta   =  sn2 * rho;
        gambar  = -cs2 * rho;
        rhs     =  phi  -  delta * z;
        zbar    =  rhs / gambar;
        xnorm   =  sqrt(xxnorm + zbar * zbar);
        gamma   =  sqrt(gambar * gambar + theta * theta);
        cs2     =  gambar / gamma;
        sn2     =  theta  / gamma;
        z       =  rhs    / gamma;
        xxnorm  =  xxnorm  +  z * z;
        
        //     Test for convergence.
        //     First, estimate the condition of the matrix  Abar,
        //     and the norms of  rbar  and  Abar'rbar.
        
        acond   =  anorm * sqrt( ddnorm );
        res1    =  phibar * phibar;
        res2    =  res2  +  psi * psi;
        rnorm   =  sqrt( res1 + res2 );
        arnorm  =  alpha * abs( tau );
        
        //     07 Aug 2002:
        //     Distinguish between
        //        r1norm = ||b - Ax|| and
        //        r2norm = rnorm in current code
        //               = sqrt(r1norm^2 + damp^2*||x||^2).
        //        Estimate r1norm from
        //        r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        //     Although there is cancellation, it might be accurate enough.
        
        double r1sq = rnorm * rnorm  -  dampsq * xxnorm;
        r1norm = sqrt( abs(r1sq) );
        if (r1sq < 0.0)
            r1norm = -r1norm;
        r2norm = rnorm;
        
        //     Now use these norms to estimate certain other quantities,
        //     some of which will be small near a solution.
        
        test1   =   rnorm / bnorm;
        test2   =   arnorm/( anorm * rnorm );
        test3   =     1.0 / acond;
        t1      =   test1 / (1.0  +  anorm * xnorm / bnorm);
        rtol    =   btol  +  atol *  anorm * xnorm / bnorm;
        
        //     The following tests guard against extremely small values of
        //     atol, btol  or  ctol.  (The user may have set any or all of
        //     the parameters  atol, btol, conlim  to 0.)
        //     The effect is equivalent to the normal tests using
        //     atol = eps,  btol = eps,  conlim = 1/eps.
        
        if (clamped)         istop = 10;
        if (itn >= itnlim)   istop = 7;
        if (1.0 + test3  <= 1.0) istop = 6;
        if (1.0 + test2  <= 1.0) istop = 5;
        if (1.0 + t1     <= 1.0) istop = 4;
        
        //     Allow for tolerances set by the user.
        
        if  (test3 <= ctol)  istop = 3;
        if  (test2 <= atol)  istop = 2;
        if  (test1 <= rtol)  istop = 1;
        
        //     See if it is time to print something.
        
        int prnt = 0;
        if (n     <= 40       ) prnt = 1;
        if (itn   <= 10       ) prnt = 1;
        if (itn   >= itnlim-10) prnt = 1;
        if (itn % 10 == 0  )    prnt = 1;
        if (test3 <=  2*ctol  ) prnt = 1;
        if (test2 <= 10*atol  ) prnt = 1;
        if (test1 <= 10*rtol  ) prnt = 1;
        if (istop !=  0       ) prnt = 1;
        
        if (prnt == 1)
        {
            if (show)
            {
                printf("%6d %12.5f\n",        itn,   x(1) );
                printf(" %10.3f %10.3f\n", r1norm, r2norm );
                printf("  %8.1f %8.1f\n",   test1,  test2 );
                printf(" %8.1f %8.1f\n",    anorm,  acond );
            }
        }
        if (!disable)
        {
            if (istop > 0) break;
            else if (disable && (itn <= itnlim))
            {
                Axnorm = (con_mat * x).norm();
                if (Axnorm < atol * anorm * xnorm)
                {
                    istop = 9;
                    break;
                }
            }
        }
        
        itn = itn + 1;
    }
    
    //     End of iteration loop.
    //     Print the stopping condition.
    
    if (show)
    {
        std::vector<std::string> msg(11);
        msg[0]  = "The exact solution is  x = 0. ";
        msg[1]  = "Ax - b is small enough, given atol, btol ";
        msg[2]  = "The least-squares solution is good enough, given atol. ";
        msg[3]  = "The estimate of cond(Abar) has exceeded conlim. ";
        msg[4]  = "Ax - b is small enough for this machine. ";
        msg[5]  = "The least-squares solution is good enough for this machine.";
        msg[6]  = "Cond(Abar) seems to be too large for this machine. ";
        msg[7]  = "The iteration limit has been reached. ";
        msg[8]  = "xnorm too large. ";
        msg[9]  = "A null vector obtained, given atol.  ";
        msg[10] = "trust region boundary hit ";
        
        printf(" LSQR finished %s\n", msg[istop].c_str());
        printf("istop =%8d   r1norm =%8.1f   ",   istop, r1norm );
        printf("anorm =%8.1f   arnorm =%8.1f\n", anorm, arnorm );
        printf("itn   =%8d   r2norm =%8.1f   ",     itn, r2norm );
        printf("acond =%8.1f   xnorm  =%8.1f\n", acond, xnorm  );
    }
    
    return x;
}

VectorXd DectModel::solveLsqrLambdas(const Ref<const VectorXd> mults)
{
    // transcribed from lsqrSOL.m
    double conlim = 1e8;
    double ctol = 0.0;
    double damp = 0.0;
    
    double atol = 1e-5;
    double btol = 1e-5;
    
    int m = n_dim;
    int n = d_vals-1;
    
    int itnlim = 2 * n;
    bool disable = false;
    
    const Ref<const MatrixXd> con_mat_transpose(gopt.rightCols(n));
    
    if (conlim > 0)
        ctol = 1.0 / conlim;
    
    int itn = 0, istop = 0;
    double anorm = 0, acond = 0, arnorm = 0, bnorm = 0;
    double dampsq = damp * damp, ddnorm = 0, res1 = 0, res2 = 0;
    double xnorm = 0, xxnorm = 0, Axnorm = 0, z = 0;
    double rhobar = 0, phibar = 0, rnorm = 0, r1norm = 0, r2norm = 0;
    double rhobar1 = 0, psi = 0, phi = 0, tau = 0;
    double delta = 0, gamma = 0, gambar = 0, rhs = 0, zbar = 0;
    double t1 = 0, t2 = 0;
    double rho = 0, cs = 0, sn = 0, theta = 0;
    double cs1 = -1.0, sn1 = 0;
    double cs2 = -1.0, sn2 = 0;
    double test1 = 0, test2 = 0, test3 = 0;
    double rtol = 0;
    bool show = false;
    
    if (show)
    {
        printf(" ");
        printf("LSQR            Least-squares solution of  Ax = b");
        printf("The matrix A has %8d rows  and %8d cols", m, n);
        printf("damp = %20.14f", damp);
        printf("atol = %8.2f                 conlim = %8.2f", atol, conlim);
        printf("btol = %8.2f                 itnlim = %8d"  , btol, itnlim);
    }
    
    // Set up the first vectors u and v for the bidiagonalization.
    // These satisfy  beta*u = b,  alpha*v = A'u.
    
    VectorXd u(m), x(n), v(n), w(n), dk(n);
    x.setZero();
    
    // set amults to (1, mults), used for convenience
    VectorXd amults(d_vals);
    amults(0) = 1.0;
    amults.tail(d_vals-1) = mults;
    
    u = -(gopt * amults); // u = -(g + A'*lambda)
    
    IOFormat mf(StreamPrecision, 0, ", ", "; ", "", "", "[", "]");
    cout << "A = " << con_mat_transpose.format(mf) << ";\n";
    cout << "u = " << u.format(mf) << ";\n";
    
    double alpha = 0, beta = u.norm();
    if (beta > 0)
    {
        u = (1.0 / beta) * u;
        v = con_mat_transpose.transpose() * u; // v = A' * u
        alpha = v.norm();
    }
    if (alpha > 0)
    {
        v = (1.0 / alpha) * v;
        w = v;
    }
    arnorm = alpha * beta;
    if (arnorm == 0)
    {
        //disp(msg(1,:));
        return x;
    }
    
    rhobar = alpha;		phibar = beta;		bnorm  = beta;
    rnorm  = beta;
    r1norm = rnorm;
    r2norm = rnorm;
    Axnorm = 0;
    
    if (show)
    {
        printf(" ");
        printf("   Itn      x(1)       r1norm     r2norm  Compatible   LS      Norm A   Cond A");
        test1  = 1;		test2  = alpha / beta;
        
        printf("%6d %12.5f",        itn,   x(1) );
        printf("%10.3f %10.3f", r1norm, r2norm );
        printf("%8.1f %8.1f",   test1,  test2 );
    }
    
    //------------------------------------------------------------------
    //     Main iteration loop.
    //------------------------------------------------------------------
    while (itn < itnlim)
    {
        //     Perform the next step of the bidiagonalization to obtain the
        //     next  beta, u, alpha, v.  These satisfy the relations
        //                beta*u  =  A*v   -  alpha*u,
        //                alpha*v =  A'*u  -  beta*v.
        
        u    = con_mat_transpose * v  -  alpha*u;
        beta = u.norm();
        if (beta > 0.0)
        {
            u     = (1.0 / beta) * u;
            anorm = Vector4d(anorm, alpha, beta, damp).norm();
            v     = con_mat_transpose.transpose() * u  -  beta*v;
            alpha  = v.norm();
            if (alpha > 0)
                v = (1.0 / alpha) * v;
        }
        
        //     Use a plane rotation to eliminate the damping parameter.
        //     This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        
        rhobar1 = sqrt(rhobar * rhobar + damp * damp);
        cs1     = rhobar / rhobar1;
        sn1     = damp   / rhobar1;
        psi     = sn1 * phibar;
        phibar  = cs1 * phibar;
        
        //     Use a plane rotation to eliminate the subdiagonal element (beta)
        //     of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        
        rho     =  sqrt(rhobar1 * rhobar1 + beta * beta);
        cs      =  rhobar1/ rho;
        sn      =  beta   / rho;
        theta   =  sn * alpha;
        rhobar  = -cs * alpha;
        phi     =  cs * phibar;
        phibar  =  sn * phibar;
        tau     =  sn * phi;
        Axnorm  =  sqrt(Axnorm * Axnorm + phi * phi);
        
        //     Update x and w.
        
        t1      =  phi / rho;
        t2      = -theta / rho;
        dk      =  w / rho;
        
        x       = x      +  t1*w;
        w       = v      +  t2*w;
        ddnorm  = ddnorm +  dk.squaredNorm();
        
        cout << "s" << itn << " = " << x.format(mf) << ";\n";
        
        //     Use a plane rotation on the right to eliminate the
        //     super-diagonal element (theta) of the upper-bidiagonal matrix.
        //     Then use the result to estimate  norm(x).
        
        delta   =  sn2 * rho;
        gambar  = -cs2 * rho;
        rhs     =  phi  -  delta * z;
        zbar    =  rhs / gambar;
        xnorm   =  sqrt(xxnorm + zbar * zbar);
        gamma   =  sqrt(gambar * gambar + theta * theta);
        cs2     =  gambar / gamma;
        sn2     =  theta  / gamma;
        z       =  rhs    / gamma;
        xxnorm  =  xxnorm  +  z * z;
        
        //     Test for convergence.
        //     First, estimate the condition of the matrix  Abar,
        //     and the norms of  rbar  and  Abar'rbar.
        
        acond   =  anorm * sqrt( ddnorm );
        res1    =  phibar * phibar;
        res2    =  res2  +  psi * psi;
        rnorm   =  sqrt( res1 + res2 );
        arnorm  =  alpha * abs( tau );
        
        //     07 Aug 2002:
        //     Distinguish between
        //        r1norm = ||b - Ax|| and
        //        r2norm = rnorm in current code
        //               = sqrt(r1norm^2 + damp^2*||x||^2).
        //        Estimate r1norm from
        //        r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        //     Although there is cancellation, it might be accurate enough.
        
        double r1sq = rnorm * rnorm  -  dampsq * xxnorm;
        r1norm = sqrt( abs(r1sq) );
        if (r1sq < 0.0)
            r1norm = -r1norm;
        r2norm = rnorm;
        
        //     Now use these norms to estimate certain other quantities,
        //     some of which will be small near a solution.
        
        test1   =   rnorm / bnorm;
        test2   =   arnorm/( anorm * rnorm );
        test3   =     1.0 / acond;
        t1      =   test1 / (1.0  +  anorm * xnorm / bnorm);
        rtol    =   btol  +  atol *  anorm * xnorm / bnorm;
        
        //     The following tests guard against extremely small values of
        //     atol, btol  or  ctol.  (The user may have set any or all of
        //     the parameters  atol, btol, conlim  to 0.)
        //     The effect is equivalent to the normal tests using
        //     atol = eps,  btol = eps,  conlim = 1/eps.
        
        if (itn >= itnlim)   istop = 7;
        if (1.0 + test3  <= 1.0) istop = 6;
        if (1.0 + test2  <= 1.0) istop = 5;
        if (1.0 + t1     <= 1.0) istop = 4;
        
        //     Allow for tolerances set by the user.
        
        if  (test3 <= ctol)  istop = 3;
        if  (test2 <= atol)  istop = 2;
        if  (test1 <= rtol)  istop = 1;
        
        //     See if it is time to print something.
        
        int prnt = 0;
        if (n     <= 40       ) prnt = 1;
        if (itn   <= 10       ) prnt = 1;
        if (itn   >= itnlim-10) prnt = 1;
        if (itn % 10 == 0  )    prnt = 1;
        if (test3 <=  2*ctol  ) prnt = 1;
        if (test2 <= 10*atol  ) prnt = 1;
        if (test1 <= 10*rtol  ) prnt = 1;
        if (istop !=  0       ) prnt = 1;
        
        if (prnt == 1)
        {
            if (show)
            {
                printf("%6d %12.5f",        itn,   x(1) );
                printf(" %10.3f %10.3f", r1norm, r2norm );
                printf("  %8.1f %8.1f",   test1,  test2 );
                printf(" %8.1f %8.1f",    anorm,  acond );
            }
        }
        if (!disable)
        {
            if (istop > 0) break;
            else if (disable && (itn <= itnlim))
            {
                Axnorm = (con_mat_transpose * x).norm();
                if (Axnorm < atol * anorm * xnorm)
                {
                    istop = 9;
                    break;
                }
            }
        }
        
        itn = itn + 1;
    }
    
    //     End of iteration loop.
    //     Print the stopping condition.
    
    if (show)
    {
        std::vector<std::string> msg(11);
        msg[0]  = "The exact solution is  x = 0. ";
        msg[1]  = "Ax - b is small enough, given atol, btol ";
        msg[2]  = "The least-squares solution is good enough, given atol. ";
        msg[3]  = "The estimate of cond(Abar) has exceeded conlim. ";
        msg[4]  = "Ax - b is small enough for this machine. ";
        msg[5]  = "The least-squares solution is good enough for this machine.";
        msg[6]  = "Cond(Abar) seems to be too large for this machine. ";
        msg[7]  = "The iteration limit has been reached. ";
        msg[8]  = "xnorm too large. ";
        msg[9]  = "A null vector obtained, given atol.  ";
        msg[10] = "trust region boundary hit ";
        
        printf(" LSQR finished %s", msg[istop].c_str());
        printf("istop =%8d   r1norm =%8.1f   ",   istop, r1norm );
        printf("anorm =%8.1f   arnorm =%8.1f\n", anorm, arnorm );
        printf("itn   =%8d   r2norm =%8.1f   ",     itn, r2norm );
        printf("acond =%8.1f   xnorm  =%8.1f\n", acond, xnorm  );
    }
    
    return x;
}

RowVectorXd DectModel::predictDifference(const Eigen::Ref<const Eigen::VectorXd> de)
{
    VectorXd ytd = xpt.transpose() * de;
    RowVectorXd res;
    
    res = de.transpose() * gopt;
    
    for (int i = 0; i < d_vals; i++)
        res(i) += 0.5 * (ytd.dot(lambda.col(i).asDiagonal() * ytd) + de.dot(mu[i].selfadjointView<Lower>() * de));
    
    return res;
}

VectorXd DectModel::multiplyOneHessian(const Eigen::Ref<const Eigen::VectorXd> de, int i)
{
    VectorXd ytd = xpt.transpose() * de;
    VectorXd res;
    
    res = xpt * (lambda.col(i).asDiagonal() * ytd) + (mu[i].selfadjointView<Lower>() * de);
    
    return res;
}

VectorXd DectModel::multiplyFullHessian(const Eigen::Ref<const Eigen::VectorXd> de, const Eigen::Ref<const Eigen::VectorXd> weights)
{
    VectorXd ytd = xpt.transpose() * de;
    VectorXd weighted_lambda = lambda * weights;
    VectorXd res;
    
    res = xpt * (weighted_lambda.asDiagonal() * ytd);
    
    // this is super expensive...
    VectorXd tmp;
    for (int i = 0; i < d_vals; i++)
    {
        double w = weights(i);
        
        if (w != 0.0)
        {
            tmp = mu[i].selfadjointView<Lower>() * de;
            res += w * tmp;
        }
    }
    
    return res;
}

DectModel::term_type DectModel::solve(int maxits)
{
    int ncon = d_vals - 1;
    DectModel::term_type tt = _MaxIters;
    
    VectorXd mults(ncon), dmults(ncon);
    mults.setZero();
    
    // initial guess for penalty
    double init_gnorm = std::max(gopt.col(0).norm(), 1e-3);
    double init_cnorm = std::max(fval.row(kopt).tail(ncon).norm(), 1e-3);
    //double init_cnorm = std::max(((gopt.col(0).transpose() / init_gnorm) * gopt.rightCols(ncon)).norm(), 1e-3);
    
    penalty = init_gnorm / init_cnorm;
    
    //penalty = 10.0; // TODO: better than this for initial penalty
    
#if VERBOSE
    cout << "Initial penalty = " << penalty << endl;
#endif
    
    reselectOpt(penalty);
    
    num_penalty_updates = 0;
    num_good_steps = 0;
    num_bad_steps = 0;
    
    double last_rho = 0.0;
    
    for (num_its = 0; num_its < maxits; num_its++)
    {
        // check for convergence
        VectorXd amults(d_vals);
        amults(0) = 1.0;
        amults.tail(d_vals-1) = mults;
        
        int knew = -1;
        RowVectorXd feval(d_vals), old_fopt = fval.row(kopt);
        VectorXd d(n_dim);
        
        double gnorm = (gopt * amults).norm();
        double cnorm = fopt().tail(ncon).norm();
        
        if (gnorm < 1e-4 && cnorm < 1e-4) // && std::abs(last_rho - 1.0) < 1e-3)
        {
            /*knew = findKfaropt();
            
            if ((xpt.col(knew) - xpt.col(kopt)).norm() > delta)
            {
                d = solvePoisedness(knew, delta);
                feval = func(xopt() + d).transpose();
                
                if (!updateModel(knew, feval, d))
                {
                    IOFormat mff(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
                    cout << "X = " << xpt.format(mff) << ";\n";
                    cout << "kopt = " << kopt << endl;
                    cout << "knew = " << knew << endl;
                    cout << "step: " << (xpt.col(kopt) + d).format(mff) << endl;
#if VERBOSE
                    cout << "degenerate model!\n";
#endif
                    tt = _DegenerateModel;
                    break;
                }
                
                if (d.norm() < 1.0e-3 * xpt.col(kopt).norm())
                    baseShift();
                reselectOpt(penalty);
                continue;
            }
            
            if (delta <= 1e-5)
            {*/
#if VERBOSE
                cout << "Successful termination!\n";
#endif
                tt = _Solved;
                break;
            /*}
            
            delta /= 10.0;
            continue;*/
        }
        
        // normal iteration
        d = solveCombinedTrustRegion(mults, dmults);
        
        knew = findKnewuoa(d);
        bool altmov = false;
        
        // evaluate prediction and actual value at the new point
        RowVectorXd predicted;
        predicted = predictDifference(d);
        feval = func(xopt() + d).transpose();
        
        RowVectorXd fdiff = feval - old_fopt;
        double dnorm = d.norm();
        
        // merit function values
        double pred = predicted(0);// + predicted.tail(d_vals-1).dot(dmults);
        double ared = fdiff(0);// + fdiff.tail(d_vals-1).dot(dmults);
        //double pcon = (old_fopt.tail(ncon) + predicted.tail(ncon)).norm() - old_fopt.tail(ncon).norm(); // full model constraint prediction
        double pcon = (old_fopt.tail(ncon) + d.transpose() * gopt.rightCols(ncon)).norm() - old_fopt.tail(ncon).norm(); // linear-only constraint prediction
        double acon = feval.tail(ncon).norm() - old_fopt.tail(ncon).norm();
        
        // cout << "pred: " << setprecision(12) << (old_fopt.tail(ncon) + d.transpose() * gopt.rightCols(ncon)).norm() << " / " << setprecision(12) << old_fopt.tail(ncon).norm() << endl;
        
        double pmerit = pred + penalty * pcon;
        double amerit = ared + penalty * acon;
        
        double ratio = 0.0;
        
        if (pmerit != 0.0)
            ratio = amerit / pmerit;
        
#if VERBOSE
        IOFormat mf(StreamPrecision, 0, ", ", "; ", "", "", "[", "]");
        cout << "value = " << old_fopt << endl;
        cout << "lambdas = " << mults.format(mf) << endl;
        cout << "delta = " << delta << endl;
        cout << "start: " << xopt().format(mf) << " :: " << mults.format(mf) << endl;
        cout << "step: " << d.format(mf) << " :: " << dmults.format(mf) << endl;
        cout << "predicted = " << predicted.format(mf) << endl;
        cout << "actual = " << fdiff.format(mf) << endl;
        cout << "pred, ared = " << pred << ", " << ared << endl;
        cout << "pcon, acon = " << pcon << ", " << acon << endl;
        cout << "pmerit, amerit, ratio = " << pmerit << ", " << amerit << ", " << ratio << endl;
        
        
        IOFormat mff(StreamPrecision, 0, ", ", ";\n", "", "", "[", "]");
        cout << "X = " << xpt.format(mff) << ";\n";
        cout << "kopt = " << kopt << endl;
        cout << "knew = " << knew << endl;
#endif
        
        if (!updateModel(knew, feval, d))
        {
#if VERBOSE
            cout << "degenerate model!\n";
#endif
            tt = _DegenerateModel;
            break;
        }
        
        if (ratio > 0.1)
        {
            // base shift if we didn't step very far
            if (dnorm < 1.0e-3 * xpt.col(kopt).norm())
                baseShift();
            
            changeKopt(knew);
            mults += dmults;
            
            // bump up penalty if it would make merit pass
            if (acon < 0.0 && ared > 0.0)
            {
                penalty = std::max(penalty * 1.1, -ared / acon + 1e-4);
                num_penalty_updates++;
#if VERBOSE
                cout << "Increase penalty: " << penalty << endl;
#endif
                // reselectOpt(penalty);
            }
            
            num_good_steps++;
            last_rho = ratio;
            
            /*if (
                std::abs(ratio - 1.0) < 1e-4 &&
                dnorm < 1e-4 * delta
                )
            {
                tt = _Solved;
                break;
            }*/
            
#if VERBOSE
            cout << "ACCEPT\n\n";
#endif
        }
        else if (!altmov)
        {
            num_bad_steps++;
            
#if VERBOSE
            cout << "REJECT\n\n";
#endif
        }
        
        updateRadius(ratio, dnorm);
        
        if (delta < 1e-5)
        {
#if VERBOSE
            cout << "TR radius hit bound\n";
#endif
            tt = _TRHit;
            break;
        }
    }
    
    return tt;
}
