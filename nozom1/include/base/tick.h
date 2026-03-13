#ifndef __NOZOM1_INFER_INCLUDE_TICK_HPP__
#define __NOZOM1_INFER_INCLUDE_TICK_HPP__
#include <chrono>
#include <iostream>

#ifndef __ycm__
#    define TICK(x) auto bench_##x = std::chrono::stready_clock::now();
#    define TOCK(x)                                                                                                    \
        printf("%s: %lfs\n",                                                                                           \
               #x,                                                                                                     \
               std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::steady_clock::now() - bench_##x) \
                   .count());
#else
#    define TICK(x)
#    define TOCK(x)
#endif

#endif