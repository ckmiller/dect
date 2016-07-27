//
//  benchmark.cpp
//  eigen_rquad
//
//  Created by Christian Miller on 3/26/16.
//  Copyright (c) 2016 ckm. All rights reserved.
//

#include "benchmark.h"

#include "dect.h"

#include <iostream>

namespace benchmark {
    
    using namespace std;
    using namespace Eigen;
    
    inline double sqr(double v) { return v * v; }
    
    void init_bt1(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 0.08, 0.06;
        xend   << 1.0, 0.0;
    }
    
    VectorXd eval_bt1(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = 100.0 * sqr(x(0)) + 100.0*sqr(x(1))-x(0)-100.0;
        
        res(1) = sqr(x(0)) + sqr(x(1)) - 1.0;
        
        return res;
    }
    
    void init_bt2(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 10.0, 10.0, 10.0;
        xend   << 1.1049, 1.1967, 1.5353;
    }
    
    VectorXd eval_bt2(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sqr(x(0)-1.0) + sqr(x(0)-x(1)) + pow(x(1)-x(2), 4);
        
        res(1) = x(0) * (1.0 + sqr(x(1))) + pow(x(2), 4) - 8.2426407;
        
        return res;
    }
    
    void init_bt3(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 20.0, 20.0, 20.0, 20.0, 20.0;
        xend   << -0.76744, 0.25581, 0.62791, -0.11628, 0.25581;
    }
    
    VectorXd eval_bt3(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)-x(1)) + sqr(x(1)+x(2)-2.0) + sqr(x(3)-1.0) + sqr(x(4)-1.0);
        
        res(1) = x(0)+3.0*x(1);
        res(2) = x(2)+x(3)-2.0*x(4);
        res(3) = x(1)-x(4);
        
        return res;
    }
    
    void init_bt4(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 3.1494, 1.4523, -3.6017;
        xend   << 4.0382, -2.9470, -0.09115;
    }
    
    VectorXd eval_bt4(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = x(0) - x(1) + pow(x(1), 3);
        
        res(1) = sqr(x(0)) + sqr(x(1)) + pow(x(2),2) - 25; // power on x(2) reduced to 2 to prevent unbounded problem
        res(2) = x(0) + x(1) + x(2) - 1;
        
        return res;
    }
    
    void init_bt5(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0;
        xend   << 3.5121, 0.21699, 3.5522;
    }
    
    VectorXd eval_bt5(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = 1000.0 - sqr(x(0)) - sqr(x(2)) - 2.0*sqr(x(1)) - x(0)*x(1) - x(0)*x(2);
        
        res(1) = -25.0 + sqr(x(0)) + sqr(x(1)) + sqr(x(2));
        res(2) = 8.0*x(0)+14.0*x(1)+7.0*x(2) - 56.0;
        
        return res;
    }
    
    void init_bt6(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0, 2.0, 2.0;
        xend   << 1.1662, 1.1821, 1.3803, 1.5060, 0.61092;
    }
    
    VectorXd eval_bt6(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-1.0) + sqr(x(0)-x(1)) + sqr(x(2)-1.0) + pow(x(3)-1.0, 4) + pow(x(4)-1.0, 6);
        
        res(1) = x(3)*sqr(x(0)) + sin(x(3)-x(4)) - 2*sqrt(2.0);
        res(2) = pow(x(2), 4)*sqr(x(1)) + x(1) - 8+sqrt(2.0);
        
        return res;
    }
    
    void init_bt7(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -2.0, 1.0, 1.0, 1.0, 1.0;
        xend   << -0.79212, -1.2624, 0.0, -0.89532, 1.1367;
    }
    
    VectorXd eval_bt7(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = 100*sqr(x(1)-sqr(x(0))) + sqr(x(0)-1.0);
        
        res(1) = x(0)*x(1) - sqr(x(2)) - 1.0;
        res(2) = sqr(x(1)) - sqr(x(3)) + x(0);
        res(3) = sqr(x(4)) + x(0) - 0.5;
        
        return res;
    }
    
    void init_bt8(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 1.0, 1.0, 1.0, 0.0, 0.0;
        xend   << 1.0, 0.0, 0.0, 0.0, 0.0;
    }
    
    VectorXd eval_bt8(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0))+sqr(x(1))+sqr(x(2));
        
        res(1) = x(0)-1-sqr(x(3))+sqr(x(1));
        res(2) = -1+sqr(x(0))+sqr(x(1))-sqr(x(4));
        
        return res;
    }
    
    void init_bt9(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0, 2.0;
        xend   << 1.0, 1.0, 0.0, 0.0;
    }
    
    VectorXd eval_bt9(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = -x(0);
        
        res(1) = x(1)-pow(x(0), 3)-sqr(x(2));
        res(2) = -x(1)+sqr(x(0))-sqr(x(3));
        
        return res;
    }
    
    void init_bt10(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0;
        xend   << 1.0, 1.0;
    }
    
    VectorXd eval_bt10(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = -x(0);
        
        res(1) = x(1)-pow(x(0),3);
        res(2) = -x(1)+sqr(x(0));
        
        return res;
    }
    
    void init_bt11(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0, 2.0, 2.0;
        xend   << 1.1912, 1.3626, 1.4728, 1.6349, 1.6790;
    }
    
    VectorXd eval_bt11(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)-1.0) + sqr(x(0)-x(1)) + sqr(x(1)-x(2)) + pow(x(2)-x(3),4) + pow(x(3)-x(4),4);
        
        res(1) = x(0)+sqr(x(1))+pow(x(2),3) - -2+sqrt(18.0);
        res(2) = x(1)+x(3)-sqr(x(2)) - -2+sqrt(8.0);
        res(3) = x(0)-x(4) - 2.0;
        
        return res;
    }
    
    void init_bt12(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0, 2.0, 2.0;
        xend   << 15.811, 1.5811, 0, 15.083, 3.7164;
    }
    
    VectorXd eval_bt12(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = 0.01*sqr(x(0))+sqr(x(1));
        
        res(1) = x(0)+x(1)-sqr(x(2)) - 25.0;
        res(2) = sqr(x(0))+sqr(x(1))-sqr(x(3)) - 25.0;
        res(3) = x(0)-sqr(x(4)) - 2.0;
        
        return res;
    }
    
    void init_byrdsphr(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 5.0, 0.0001, -0.0001;
        xend   << 0, 0, 0;
    }
    
    VectorXd eval_byrdsphr(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = -x(0)-x(1)-x(2);
        
        res(1) = -9.0+sqr(x(0))+sqr(x(1))+sqr(x(2));
        res(2) = -9.0+sqr(x(0)-1.0)+sqr(x(1))+sqr(x(2));
        
        return res;
    }
    
    void init_hs6(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -1.2, 1.0;
        xend   << 1.0, 1.0;
    }
    
    VectorXd eval_hs6(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sqr((1-x(0)));
        
        res(1) = 10*(x(1) - sqr(x(0)));
        
        return res;
    }
    
    void init_hs7(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0;
        xend   << 0.0, 1.732050807568877;
    }
    
    VectorXd eval_hs7(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = log(1+sqr(x(0))) - x(1);
        
        res(1) = sqr(1+sqr(x(0))) + sqr(x(1)) - 4;
        
        return res;
    }
    
    void init_hs8(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 1.0;
        xend   << 4.60159, 1.95584;
    }
    
    VectorXd eval_hs8(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = -1.0;
        
        res(1) = sqr(x(0)) + sqr(x(1)) - 25;
        res(2) = x(0)*x(1) - 9;
        
        return res;
    }
    
    void init_hs9(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 0.000001, 0.000001;
        xend   << -3.0, -4.0;
    }
    
    VectorXd eval_hs9(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sin(M_PI * x(0) / 12) * cos(M_PI * x(1) / 16);
        
        res(1) = 4*x(0) - 3*x(1);
        
        return res;
    }
    
    void init_hs26(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -2.6, 2.0, 2.0;
        xend   << 1.0, 1.0, 1.0;
    }
    
    VectorXd eval_hs26(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sqr(x(0) - x(1)) + pow(x(1) - x(2), 4);
        
        res(1) = (1 + sqr(x(1)))*x(0) + pow(x(2), 4) - 3;
        
        return res;
    }
    
    void init_hs27(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0;
        xend   << -1.0, 1.0, 0.0;
    }
    
    VectorXd eval_hs27(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sqr(x(0) - 1)/100 + sqr(x(1) - sqr(x(0)));
        
        res(1) = x(0) + sqr(x(2)) - -1;
        
        return res;
    }
    
    void init_hs28(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -4.0, 1.0, 1.0;
        xend   << 0.5, -0.5, 0.5;
    }
    
    VectorXd eval_hs28(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        
        res(0) = sqr(x(0) + x(1)) + sqr(x(1) + x(2));
        
        res(1) = x(0) + 2*x(1) + 3*x(2) - 1;
        
        return res;
    }
    
    void init_hs39(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.0, 2.0, 2.0, 2.0;
        xend   << 1.0, 1.0, 0.0, 0.0;
    }
    
    VectorXd eval_hs39(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = -x(0);
        
        res(1) = x(1) - pow(x(0),3) - sqr(x(2));
        res(2) = sqr(x(0)) - x(1) - sqr(x(3));
        
        return res;
    }
    
    void init_hs40(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 0.8, 0.8, 0.8, 0.8;
        xend   << 0.793701, 0.707107, 0.529732, 0.840896;
    }
    
    VectorXd eval_hs40(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = -x(0)*x(1)*x(2)*x(3);
        
        res(1) = pow(x(0),3) + sqr(x(1)) - 1;
        res(2) = sqr(x(0))*x(3) - x(2);
        res(3) = sqr(x(3)) - x(1);
        
        return res;
    }
    
    void init_hs42(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 1.0, 1.0, 1.0, 1.0;
        xend   << 2.0, 2.0, 0.848529, 1.13137;
    }
    
    VectorXd eval_hs42(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-1) + sqr(x(1)-2) + sqr(x(2)-3) + sqr(x(3)-4);
        
        res(1) = x(0) - 2;
        res(2) = sqr(x(2)) + sqr(x(3)) - 2;
        
        return res;
    }
    
    void init_hs46(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << sqrt(2.0)/2.0, 1.75, 0.5, 2.0, 2.0;
        xend   << 0.0, 0.0, 0.0, 0.0, 0.0;
    }
    
    VectorXd eval_hs46(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-x(1)) + sqr(x(2)-1) + pow(x(3)-1,4) + pow(x(4)-1,6);
        
        res(1) = sqr(x(0))*x(3) + sin(x(3) - x(4)) - 1;
        res(2) = x(1) + pow(x(2),4)*sqr(x(3)) - 2;
        
        return res;
    }
    
    void init_hs48(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 3.0, 5.0, -3.0, 2.0, -2.0;
        xend   << 1.0, 1.0, 1.0, 1.0, 1.0;
    }
    
    VectorXd eval_hs48(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-1) + sqr(x(1)-x(2)) + sqr(x(3)-x(4));
        
        res(1) = x.sum() - 5.0;
        res(2) = x(2) - 2*(x(3)+x(4)) - -3.0;
        
        return res;
    }
    
    void init_hs49(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 10.0, 7.0, 2.0, -3.0, 0.8;
        xend   << 1.0, 1.0, 1.0, 1.0, 1.0;
    }
    
    VectorXd eval_hs49(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-x(1)) + sqr(x(2)-1) + pow(x(3)-1,4) + pow(x(4)-1,6);
        
        res(1) = x(0) + x(1) + x(2) + x(3) + 3.0 * x(3) - 7;
        res(2) = x(2) + 5*x(4) - 6;
        
        return res;
    }
    
    void init_hs50(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 35.0, -31.0, 11.0, 5.0, -5.0;
        xend   << 1.0, 1.0, 1.0, 1.0, 1.0;
    }
    
    VectorXd eval_hs50(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)-x(1)) + sqr(x(1)-x(2)) + pow(x(2)-x(3), 4) + sqr(x(3)-x(4));
        
        res(1) = x(0) + 2*x(1) + 3*x(2) - 6;
        res(2) = x(1) + 2*x(2) + 3*x(3) - 6;
        res(3) = x(2) + 2*x(3) + 3*x(4) - 6;
        
        return res;
    }
    
    void init_hs51(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2.5, 0.5, 2.0, -1.0, 0.5;
        xend   << 1.0, 1.0, 1.0, 1.0, 1.0;
    }
    
    VectorXd eval_hs51(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)-x(1)) + sqr(x(1)+x(2)-2) + sqr(x(3)-1) + sqr(x(4)-1);
        
        res(1) = x(0) + 3*x(1) - 4;
        res(2) = x(2) +   x(3) - 2*x(4);
        res(3) = x(1) -   x(4);
        
        return res;
    }
    
    void init_hs61(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 0.0, 0.0, 0.0;
        xend   << 5.326770157, -2.118998639, 3.210464239;
    }
    
    VectorXd eval_hs61(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = 4*sqr(x(0)) + 2*sqr(x(1)) + 2*sqr(x(2)) - 33*x(0) + 16*x(1) - 24*x(2);
        
        res(1) = 3*x(0) - 2*sqr(x(1)) - 7;
        res(2) = 4*x(0) -   sqr(x(2)) - 11;
        
        return res;
    }
    
    void init_hs77(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2, 2, 2, 2, 2;
        xend   << 1.166172, 1.182111, 1.380257, 1.506036, .6109203;
    }
    
    VectorXd eval_hs77(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-1) + sqr(x(0)-x(1)) + sqr(x(2)-1) + pow(x(3)-1, 4) + pow(x(4)-1, 6);
        
        res(1) = sqr(x(0)) * x(3) + sin(x(3) - x(4)) - 2.0 * sqrt(2.0);
        res(2) = x(1) + pow(x(2),4) * sqr(x(3)) - 8.0 - sqrt(2.0);
        
        return res;
    }
    
    void init_hs78(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -2, 1.5, 2, -1, -1;
        xend   << -1.717142, 1.595708, 1.827248, -.7636429, -.7636435;
    }
    
    VectorXd eval_hs78(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = x(0)*x(1)*x(2)*x(3)*x(4);
        
        res(1) = sqr(x(0))+sqr(x(1))+sqr(x(2))+sqr(x(3))+sqr(x(4)) - 10.0;
        res(2) = x(1)*x(2) - 5.0*x(3)*x(4);
        res(3) = pow(x(0),3) + pow(x(1),3) + 1;
        
        return res;
    }
    
    void init_hs79(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 2, 2, 2, 2, 2;
        xend   << 1.191127, 1.362603, 1.472818, 1.635017, 1.679081;
    }
    
    VectorXd eval_hs79(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)-1) + sqr(x(0)-x(1)) + sqr(x(1)-x(2)) + pow(x(2)-x(3), 4) + pow(x(3)-x(4), 4);
        
        res(1) = x(0) + sqr(x(1)) * pow(x(2),3) - 2.0 - 3.0*sqrt(2.0);
        res(2) = x(1) - sqr(x(2)) * x(3) + 2.0 - 2.0*sqrt(2.0);
        res(3) = x(0)*x(4) - 2;
        
        return res;
    }
    
    void init_maratos(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 1.1, 0.1;
        xend   << 1.0, 0.0;
    }
    
    VectorXd eval_maratos(const Ref<const VectorXd> x)
    {
        VectorXd res(2);
        double tau = 0.000001;
        
        res(0) = -x(0)-tau+tau*(sqr(x(0))+sqr(x(1)));
        
        res(1) = -1.0+sqr(x(0))+sqr(x(1));
        
        return res;
    }
    
    void init_mwright(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -1, 2, 1, -2, -2;
        xend   << -1.2946, 2.46308, 1.21263, -0.164164, -1.54488;
    }
    
    VectorXd eval_mwright(const Ref<const VectorXd> x)
    {
        VectorXd res(4);
        
        res(0) = sqr(x(0)) + pow(x(0)-x(1), 2) + pow(x(1)-x(2), 3) + pow(x(2)-x(3), 4) + pow(x(3)-x(4), 4);
        
        res(1) = sqr(x(1))+sqr(x(2))+x(0)-3*sqrt(2)-2;
        res(2) = -sqr(x(2))+x(1)+x(3)-2*sqrt(2)+2;
        res(3) = x(0)*x(4)-2;
        
        return res;
    }
    
    void init_hs100lnp(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 1, 2, 0, 4, 0, 1, 1;
        xend   << 2.330499, 1.951372, -0.4775414, 4.365726, -0.6244870, 1.038131, 1.594227;
    }
    
    VectorXd eval_hs100lnp(const Ref<const VectorXd> x)
    {
        VectorXd res(3);
        
        res(0) = sqr(x(0)-10) + 5*sqr(x(1)-12) + pow(x(2),4) + 3*sqr(x(3)-11) + 10*pow(x(4),6) + 7*sqr(x(5)) + pow(x(6),4) - 4*x(5)*x(6) - 10*x(5) - 8*x(6);
        
        res(1) = 2*sqr(x(0)) + 3*pow(x(1),4) + x(2) + 4*sqr(x(3)) + 5*x(4) - 127;
        res(2) = -4*sqr(x(0)) - sqr(x(1)) + 3*x(0)*x(1) -2*sqr(x(2)) - 5*x(5) + 11*x(6);
        
        return res;
    }
    
    void init_hs111lnp(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart.setConstant(-3.2);
        xend   << -3.201212, -1.912060, -.2444413, -6.537489, -.7231524, -7.267738, -3.596711, -4.017769, -3.287462, -2.335582;
    }
    
    VectorXd eval_hs111lnp(const Ref<const VectorXd> x)
    {
        VectorXd c(10);
        c << -6.089, -17.164, -34.054, -5.914, -24.721, -14.986, -24.100, -10.708, -26.662, -22.179;
        
        VectorXd res(4);
        
        double sumk = log(x.unaryExpr([](double v){return exp(v);}).sum());
        
        res(0) = 0.0;
        for (int j = 0; j < 10; j++)
            res(0) += exp(x(j)) * (c(j) + x(j) - sumk);
        
        res(1) = exp(x(0)) + 2*exp(x(1)) + 2*exp(x(2)) + exp(x(5)) + exp(x(9)) - 2;
        res(2) = exp(x(3)) + 2*exp(x(4)) + exp(x(5)) + exp(x(6)) - 1;
        res(3) = exp(x(2)) + exp(x(6)) + exp(x(7)) + 2*exp(x(8)) + exp(x(9)) - 1;
        
        return res;
    }
    
    void init_robot(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart.setZero();
        xend.setZero();
    }
    
    VectorXd eval_robot(const Ref<const VectorXd> x)
    {
        VectorXd res(10);
        
        double XPOS = 4;
        double YPOS = 4;
        //double HIGH = 2.356194;
        //double DOWN = -2.356194;
        
        const Ref<const VectorXd> TH = x.head(7);
        const Ref<const VectorXd> THI = x.tail(7);
        
        res(0) = (TH - THI).squaredNorm();
        
        res(1) = TH.unaryExpr([](double v){return cos(v);}).sum()-0.5*cos(TH[6]) -XPOS;
        res(2) = TH.unaryExpr([](double v){return sin(v);}).sum()-0.5*sin(TH[6]) -YPOS;
        res.tail(7) = THI;
        
        return res;
    }
    
    void init_orthregb(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << 1, 0, 0, 1, 0, 1, 0, 0, 0, 9.5, 9.5, 0.5, 6.5, -5.5, 0.5, -8.5, -8.5, 0.5, -5.5, 6.5, 0.5, 0.5, 0.5, 7.5, 0.5, 0.5, -6.5;
        xend.setZero();
    }
    
    VectorXd eval_orthregb(const Ref<const VectorXd> x)
    {
        VectorXd res(7);
        
        int idx = 0;
        const double &h11 = x(idx++);
        const double &h12 = x(idx++);
        const double &h13 = x(idx++);
        const double &h22 = x(idx++);
        const double &h23 = x(idx++);
        const double &h33 = x(idx++);
        const double &g1 = x(idx++);
        const double &g2 = x(idx++);
        const double &g3 = x(idx++);
        const double &x1 = x(idx++);
        const double &y1 = x(idx++);
        const double &z1 = x(idx++);
        const double &x2 = x(idx++);
        const double &y2 = x(idx++);
        const double &z2 = x(idx++);
        const double &x3 = x(idx++);
        const double &y3 = x(idx++);
        const double &z3 = x(idx++);
        const double &x4 = x(idx++);
        const double &y4 = x(idx++);
        const double &z4 = x(idx++);
        const double &x5 = x(idx++);
        const double &y5 = x(idx++);
        const double &z5 = x(idx++);
        const double &x6 = x(idx++);
        const double &y6 = x(idx++);
        const double &z6 = x(idx++);
        
        res(0) = (x1 - 9.5)*(x1 - 9.5) + (y1 - 9.5)*(y1 - 9.5) + (z1 - 0.5)*(z1 - 0.5) +
                 (x2 -  6.5)*(x2 - 6.5) + (y2 + 5.5)*(y2 + 5.5) + (z2 - 0.5)*(z2 - 0.5) +
                 (x3 + 8.5)*(x3 + 8.5) + (y3 + 8.5)*(y3 + 8.5) + (z3 - 0.5)*(z3 - 0.5) +
                 (x4 + 5.5)*(x4 + 5.5) + (y4 - 6.5)*(y4 - 6.5) + (z4 - 0.5)*(z4 - 0.5) +
                 (x5 -0.5)*(x5 - 0.5) + (y5 - 0.5)*(y5 - 0.5) + (z5 - 7.5)*(z5 - 7.5) +
                 (x6 -0.5)*(x6 - 0.5) + (y6 - 0.5)*(y6 - 0.5) + (z6 + 6.5)*(z6 + 6.5);
        
        res(1) = h11 * x1 * x1 + 2.0*h12 * x1 * y1 + h22 * y1 * y1 - 2.0*g1 * x1 - 2.0*g2 * y1 +
                 2.0*h13 * x1 * z1 + 2.0*h23 * y1 * z1 + h33 * z1 * z1 - 2.0*g3 * z1 - 1.0;
        res(2) = h11 * x2 * x2 + 2.0*h12 * x2 * y2 + h22 * y2 * y2 - 2.0*g1 * x2 - 2.0*g2 * y2 +
                 2.0*h13 * x2 * z2 + 2.0*h23 * y2 * z2 + h33 * z2 * z2 - 2.0*g3 * z2 - 1.0;
        res(3) = h11 * x3 * x3 + 2.0*h12 * x3 * y3 + h22 * y3 * y3 - 2.0*g1 * x3 - 2.0*g2 * y3 +
                 2.0*h13 * x3 * z3 + 2.0*h23 * y3 * z3 + h33 * z3 * z3 - 2.0*g3 * z3 - 1.0;
        res(4) = h11 * x4 * x4 + 2.0*h12 * x4 * y4 + h22 * y4 * y4 - 2.0*g1 * x4 - 2.0*g2 * y4 +
                 2.0*h13 * x4 * z4 + 2.0*h23 * y4 * z4 + h33 * z4 * z4 - 2.0*g3 * z4 - 1.0;
        res(5) = h11 * x5 * x5 + 2.0*h12 * x5 * y5 + h22 * y5 * y5 - 2.0*g1 * x5 - 2.0*g2 * y5 +
                 2.0*h13 * x5 * z5 + 2.0*h23 * y5 * z5 + h33 * z5 * z5 - 2.0*g3 * z5 - 1.0;
        res(6) = h11 * x6 * x6 + 2.0*h12 * x6 * y6 + h22 * y6 * y6 - 2.0*g1 * x6 - 2.0*g2 * y6 +
                 2.0*h13 * x6 * z6 + 2.0*h23 * y6 * z6 + h33 * z6 * z6 - 2.0*g3 * z6 - 1.0;
        
        return res;
    }
    
    void init_catena(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        int N = 10;
        double bl = 1.0;
        double fract = 0.6;
        double length = bl*(N+1)*fract;
        
        xstart.setZero();
        
        // set N variables x[1..N]
        for (int i = 1; i <= N; i++)
            xstart(i - 1) = i*length/(N+1);
        
        // set N+1 variables y[1..N+1]
        for (int i = 1; i <= N+1; i++)
            xstart(N + i - 1) = -i*length/(N+1);
        
        // all the rest are z[0..N+1] = 0
        
        xend.setZero();
    }
    
    VectorXd eval_catena(const Ref<const VectorXd> vx)
    {
        int N = 10;
        
        double gamma = 9.81;
        double tmass = 500.0;
        double bl = 1.0;
        double fract = 0.6;
        
        double length = bl*(N+1)*fract;
        double mass = tmass/(N+1);
        double mg = mass*gamma;
        
        VectorXd x(N+2), y(N+2), z(N+2);
        
        x(0) = 0.0;
        x.segment(1, N) = vx.segment(0, N);
        x(N+1) = length;
        
        y(0) = 0.0;
        y.tail(N+1) = vx.segment(N, N+1);
        
        z(0) = 0.0;
        z.tail(N+1) = vx.tail(N+1);
        
        VectorXd res(12);
        
        res(0) = mg*y(0)/2;
        for (int i = 1; i <= N; i++)
            res(0) += mg*y(i) + mg*y(N+1)/2;
        
        for (int i = 1; i <= N+1; i++)
            res(i) = sqr(x(i)-x(i-1)) + sqr(y(i)-y(i-1)) + sqr(z(i)-z(i-1)) - sqr(bl);
        
        return res;
    }
    
    void init_dixchlng(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        xstart << -2.0, -1.0/2.0, 3.0, 1.0/3.0, -4.0, -1.0/4.0, 5.0, 1.0/5.0, -6, -1.0/6.0;
        xend.setZero();
    }
    
    VectorXd eval_dixchlng(const Ref<const VectorXd> x)
    {
        VectorXd res(6);
        
        res(0) = 0.0;
        
        for (int i = 0; i < 7; i++)
        {
            res(0) += 100*sqr(x(i+1)-sqr(x(i)));
            res(0) += sqr(x(i)-1);
            res(0) += 90*sqr(x(i+3)-sqr(x(i+2)));
            res(0) += sqr(x(i+2)-1);
            res(0) += 10.1*sqr(x(i+1)-1);
            res(0) += 10.1*sqr(x(i+3)-1);
            res(0) += 19.8*(x(i+1)-1)*(x(i+3)-1);
        }
        
        for (int i = 0; i < 5; i++)
        {
            double prod = 1.0;
            for (int j = 0; j < 2*(i+1); j++)
                prod *= x(j);
            res(i+1) = prod - 1.0;
        }
        
        return res;
    }
    
    void init_eigena2(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        int n = 10;
        
        Map<VectorXd> dm(xstart.data(), n, 1);
        Map<MatrixXd> qm(xstart.tail(n*n).data(), n, n);
        
        // d first, then q
        dm.setConstant(1.0);
        qm.setIdentity();
        
        xend.setZero();
    }
    
    VectorXd eval_eigena2(const Ref<const VectorXd> x)
    {
        int n = 10;
        int i, j;
        
        VectorXd res(56);
        
        const Map<const VectorXd> dm(x.data(), n, 1);
        const Map<const MatrixXd> qm(x.tail(n*n).data(), n, n);
        
        // set up the matrix to be factored
        MatrixXd A(n,n);
        A.setZero();
        for (j = 1; j <= n; j++)
            for (i = 1; i <= n; i++)
                A(j-1, i-1) = ((i == j) ? j : 0.0);
        
        // cout << qm.transpose() * dm.asDiagonal() * qm << endl;
        // cout << qm.transpose() * qm << endl << endl;
        
        res(0) = (dm.asDiagonal() * qm - qm * A).squaredNorm();
        
        int idx = 1;
        MatrixXd qtq = qm.transpose() * qm - MatrixXd::Identity(n,n);
        for (j = 1; j <= n; j++)
            for (i = 1; i <= j; i++)
                res(idx++) = qtq(j-1, i-1);
        
        return res;
    }
    
    void init_eigencco(Ref<VectorXd> xstart, Ref<VectorXd> xend)
    {
        int m = 2;
        int n = 2*m+1;
        
        Map<VectorXd> dm(xstart.data(), n, 1);
        Map<MatrixXd> qm(xstart.tail(n*n).data(), n, n);
        
        // d first, then q
        dm.setConstant(1.0);
        qm.setIdentity();
        
        xend.setZero();
    }
    
    VectorXd eval_eigencco(const Ref<const VectorXd> x)
    {
        int m = 2;
        int n = 2*m+1;
        int i, j;
        
        VectorXd res(16);
        
        const Map<const VectorXd> dm(x.data(), n, 1);
        const Map<const MatrixXd> qm(x.tail(n*n).data(), n, n);
        
        // set up the matrix to be factored
        MatrixXd A(n,n);
        A.setZero();
        for (j = 1; j <= n; j++)
        {
            for (i = 1; i <= j; i++)
            {
                double val = 0.0;
                
                if (i == j)
                    val = m+1-j;
                else if (i == j-1)
                    val = 1.0;
                
                A(j-1, i-1) = val;
            }
        }
        
        // cout << qm.transpose() * dm.asDiagonal() * qm << endl;
        // cout << qm.transpose() * qm << endl << endl;
        
        res(0) = (qm.transpose() * dm.asDiagonal() * qm - A).squaredNorm();
        
        int idx = 1;
        MatrixXd qtq = qm.transpose() * qm - MatrixXd::Identity(n,n);
        for (j = 1; j <= n; j++)
            for (i = 1; i <= j; i++)
                res(idx++) = qtq(j-1, i-1);
        
        return res;
    }
    
#define DECLARE_TEST(NAME, N, M) { #NAME, (N), (M), init_##NAME, eval_##NAME }
    vector<test> testlist = {
        DECLARE_TEST(bt1, 2, 1),
        DECLARE_TEST(bt2, 3, 1),
        DECLARE_TEST(bt3, 5, 3),
        DECLARE_TEST(bt4, 3, 2),
        DECLARE_TEST(bt5, 3, 2),
        DECLARE_TEST(bt6, 5, 2),
        DECLARE_TEST(bt7, 5, 3),
        DECLARE_TEST(bt8, 5, 2),
        DECLARE_TEST(bt9, 4, 2),
        DECLARE_TEST(bt10, 2, 2),
        DECLARE_TEST(bt11, 5, 3),
        DECLARE_TEST(bt12, 5, 3),
        DECLARE_TEST(hs6, 2, 1),
        DECLARE_TEST(hs7, 2, 1),
        DECLARE_TEST(hs8, 2, 2),
        DECLARE_TEST(hs9, 2, 1),
        DECLARE_TEST(hs26, 3, 1),
        DECLARE_TEST(hs27, 3, 1),
        DECLARE_TEST(hs28, 3, 1),
        DECLARE_TEST(hs39, 4, 2),
        DECLARE_TEST(hs40, 4, 3),
        DECLARE_TEST(hs42, 4, 2),
        DECLARE_TEST(hs46, 5, 2),
        DECLARE_TEST(hs48, 5, 2),
        DECLARE_TEST(hs49, 5, 2),
        DECLARE_TEST(hs50, 5, 3),
        DECLARE_TEST(hs51, 5, 3),
        DECLARE_TEST(hs61, 3, 2),
        DECLARE_TEST(hs77, 5, 2),
        DECLARE_TEST(hs78, 5, 3),
        DECLARE_TEST(hs79, 5, 3),
        DECLARE_TEST(hs100lnp, 7, 2),
        DECLARE_TEST(hs111lnp, 10, 3),
        DECLARE_TEST(byrdsphr, 3, 2),
        DECLARE_TEST(maratos, 2, 1),
        DECLARE_TEST(mwright, 5, 3),
        DECLARE_TEST(robot, 14, 9),
        DECLARE_TEST(orthregb, 27, 6),
        /*DECLARE_TEST(catena, 32, 11),
        DECLARE_TEST(dixchlng, 10, 5),
        DECLARE_TEST(eigena2, 110, 55),
        DECLARE_TEST(eigencco, 30, 15),*/
    };
    
    void run_benchmark(double rad, int max_its)
    {
        printf("  # ||   name   ||   n, m  ||  its  ||       g, b, %%     ||     result      ||     xerr     ||     fopt     ||     cnorm    \n");
        printf("----||----------||---------||-------||-------------------||-----------------||--------------||--------------||--------------\n");
        for (int i = 0; i < testlist.size(); i++)
        {
            const test &t = testlist[i];
            
            DectModel *rq = new DectModel(t.n, t.m + 1);
            DectModel::term_type res = DectModel::_None;
            
            VectorXd xstart(t.n), xend(t.n);
            
            t.init_fn(xstart, xend);
            
            rq->alloc();
            rq->initBasis(xstart, rad);
            rq->initModel(t.eval_fn);
            
            res = rq->solve(max_its);
            
            VectorXd xres = rq->xopt();
            double xerr = (xres - xend).norm();
            
            printf("% 3d || %8s || %3d, %2d || % 5d || %4d, %4d, %4.0f%% || %15s || %e || %12f || %14e\n",
                   i, t.name, t.n, t.m,
                   rq->num_its + rq->m_pts, rq->num_good_steps, rq->num_bad_steps, 100.0f * (float)rq->num_good_steps / (float)std::max(rq->num_its, 1),
                   DectModel::termString(res), xerr,
                   rq->fopt()(0), rq->fopt().tail(t.m).norm());
            
            delete rq;
        }
    }
}
