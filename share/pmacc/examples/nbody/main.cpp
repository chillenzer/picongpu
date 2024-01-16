/* Copyright 2024 Julian Lenz
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>

using ::pmacc::DataSpace;

namespace nbody
{
    auto basicSetup()
    {
        DataSpace<DIM3> devices{1, 1, 1};
        DataSpace<DIM3> grid{1, 1, 1};
        DataSpace<DIM3> periodic{1, 1, 1};
        uint32_t steps = 10;
        return std::make_tuple(steps, devices, grid, periodic);
    }

    struct Simulation
    {
        uint32_t steps;
        DataSpace<DIM3> devices{1, 1, 1};
        DataSpace<DIM3> grid{1, 1, 1};
        DataSpace<DIM3> periodic{1, 1, 1};

        void init()
        {
        }
        void start()
        {
        }
        void finalize()
        {
        }
    };

} // namespace nbody
/*! main function of nbody problem
 *
 * this program computes the gravitational nbody problem basically
 * reimplementing in PMAcc the approach from here:
 * https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda
 *
 * @param argc count of arguments in argv
 * @param argv arguments of program start
 */
int main(int argc, char** argv)
{
    // TODO: Should later be read from boost program_options.
    // Also potentially generalise to 2D use as well.

    auto [steps, devices, grid, periodic] = nbody::basicSetup();

    /* start game of life simulation */
    nbody::Simulation sim{steps, grid, devices, periodic};
    sim.init();
    sim.start();
    sim.finalize();

    /* finalize the pmacc context */
    pmacc::Environment<>::get().finalize();

    return 0;
}
