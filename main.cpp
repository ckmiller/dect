//
//  main.cpp
//  eigen_rquad
//
//  Created by Christian Miller on 7/3/15.
//  Copyright (c) 2015 ckm. All rights reserved.
//

// build using modified version of GLUT from http://iihm.imag.fr/blanch/software/glut-macosx/

#include "dect.h"
#include "benchmark.h"

using namespace std;
using namespace Eigen;

int main(int argc, char **argv)
{
    // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() & ~_MM_MASK_INVALID);
    
    benchmark::run_benchmark(0.1, 2000);
    
    // _MM_SET_EXCEPTION_MASK(_MM_GET_EXCEPTION_MASK() | _MM_MASK_INVALID);
    
    return 0;
}
