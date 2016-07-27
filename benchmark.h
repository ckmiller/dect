//
//  benchmark.h
//  eigen_rquad
//
//  Created by Christian Miller on 3/26/16.
//  Copyright (c) 2016 ckm. All rights reserved.
//

#ifndef __DECT_BENCHMARK__
#define __DECT_BENCHMARK__

#include <Eigen/Core>

#include <vector>
#include <string>

namespace benchmark {
    
    struct test
    {
        const char *name;
        int n, m;
        void (*init_fn)(Eigen::Ref<Eigen::VectorXd> xstart, Eigen::Ref<Eigen::VectorXd> xend);
        Eigen::VectorXd (*eval_fn)(const Eigen::Ref<const Eigen::VectorXd> x);
    };
    
    extern std::vector<test> testlist;
    
    void run_benchmark(double rad, int max_its);
};

#endif
