/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once

#include <ctime>
#include <cstdlib>
#include <chrono>

class TicToc
{
  public:
    TicToc()
    {
        tic();
    }

    void tic()
    {
        start = std::chrono::steady_clock::now();
    }

    double toc()
    {
        end = std::chrono::steady_clock::now();
        auto elapsed_microseconds = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        double millisecond = (double)elapsed_microseconds.count()/1000;
        return millisecond;
    }

  private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};
