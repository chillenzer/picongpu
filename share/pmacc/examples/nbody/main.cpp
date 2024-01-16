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
#include <pmacc/memory/buffers/GridBuffer.hpp>

using Space = pmacc::DataSpace<DIM3>;
using Buffer = pmacc::GridBuffer<float, DIM3>;

namespace nbody
{
    /*! basic setup returning number of devices, steps and grid sites as well as periodicity
     *
     * this is currently hardcoded but shall later be the place to read cmdline
     * args, etc.
     */
    auto basicSetup()
    {
        // TODO: Should later be read from boost program_options.
        // Also potentially generalise to 2D use as well.
        Space devices{1, 1, 1};
        Space gridSize{1, 1, 1};
        Space periodic{1, 1, 1};
        uint32_t steps = 10;
        return std::make_tuple(steps, devices, gridSize, periodic);
    }

    struct Simulation
    {
        Space gridSize{1, 1, 1};
        uint32_t steps;

        Simulation(uint32_t steps, Space gridSize, Space devices, Space periodic) : gridSize(gridSize), steps(steps)
        {
            /* -First this initializes the GridController with number of 'devices'*
             *  and 'periodic'ity. The init-routine will then create and manage   *
             *  the MPI processes and communication group and topology.           *
             * -Second the cudaDevices will be allocated to the corresponding     *
             *  Host MPI processes where hostRank == deviceNumber, if the device  *
             *  is not marked to be used exclusively by another process. This     *
             *  affects: cudaMalloc,cudaKernelLaunch,                             *
             * -Then the CUDA Stream Controller is activated and one stream is    *
             *  added. It's basically a List of cudaStreams. Used to parallelize  *
             *  Memory transfers and calculations.                                */
            pmacc::Environment<DIM3>::get().initDevices(devices, periodic);

            /* Now we have allocated every node to a grid position in the GC. We  *
             * use that grid position to allocate every node to a position in the *
             * physic grid. Using the localGridSize = the number of cells per     *
             * node = number of cells / nodes, we can get the position of the     *
             * current node as an offset in numbers of cells                      */
            pmacc::GridController<DIM3>& gc = pmacc::Environment<DIM3>::get().GridController();
            Space localGridSize(gridSize / devices);

            /* - This forwards arguments to SubGrid.init()                        *
             * - Create Singletons: EnvironmentController, DataConnector,         *
             *                      PluginConnector, device::MemoryInfo           */
            pmacc::Environment<DIM3>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);
        }

        Simulation& init()
        {
            return *this;
        }
        Simulation& start()
        {
            return *this;
        }
        Simulation& finalize()
        {
            return *this;
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
    auto [steps, devices, grid, periodic] = nbody::basicSetup();
    nbody::Simulation{steps, grid, devices, periodic}.init().start().finalize();
    pmacc::Environment<>::get().finalize();

    return 0;
}
