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

#include "Particles.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/mpi/GatherSlice.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <memory>


using Space = pmacc::DataSpace<DIM3>;

namespace nbody
{
    /*! read basic setup returning number of devices, steps and grid sites as well as periodicity
     *
     * this is currently hardcoded but shall later be the place to read cmdline
     * args, etc.
     */
    auto readArgs([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
    {
        // TODO: Should later be read from boost program_options.
        // Also potentially generalise to 2D use as well.
        Space devices{1, 1, 1};
        Space gridSize{8, 8, 4};
        Space localGridSize(gridSize / devices);
        Space periodic{1, 1, 1};
        uint32_t steps = 10;
        return std::make_tuple(steps, devices, gridSize, localGridSize, periodic);
    }

    pmacc::GridLayout<DIM3> initGrids(Space const& gridSize, Space const& localGridSize, Space const& periodic)
    {
        pmacc::GridController<DIM3>& gc = pmacc::Environment<DIM3>::get().GridController();
        pmacc::Environment<DIM3>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);
        const pmacc::SubGrid<DIM3>& subGrid = pmacc::Environment<DIM3>::get().SubGrid();
        return {subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT()};
    }

    auto runSimulation(){};

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
    using namespace nbody;
    auto [steps, devices, gridSize, localGridSize, periodic] = readArgs(argc, argv);
    pmacc::Environment<DIM3>::get().initDevices(devices, periodic);
    auto layout = initGrids(gridSize, localGridSize, periodic);
    Particles{std::make_shared<DeviceHeap>(), MappingDesc{layout.getDataSpaceWithoutGuarding()}};
    runSimulation();
    pmacc::Environment<>::get().finalize();

    return 0;
}
