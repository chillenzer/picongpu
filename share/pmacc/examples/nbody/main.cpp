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

#include "DeviceHeap.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/DataSpace.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/memory/dataTypes/Mask.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/mpi/GatherSlice.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>
#include <pmacc/traits/NumberOfExchanges.hpp>

#include <memory>


using Space = pmacc::DataSpace<DIM3>;
using float3 = pmacc::math::Vector<float, 3u>;

namespace nbody
{
    value_identifier(float3, position, float3::create(0.));
    value_identifier(float, mass, 1.);

    constexpr const uint32_t numSlots = 256;
    using MappingDesc = pmacc::MappingDescription<DIM3, pmacc::math::CT::Int<16, 16, 16>>;
    using TrivialParticleDescription = pmacc::ParticleDescription<
        PMACC_CSTRING("particle"),
        std::integral_constant<uint32_t, numSlots>,
        MappingDesc::SuperCellSize,
        pmacc::MakeSeq_t<position, mass>>;

    using Particles = pmacc::ParticlesBase<TrivialParticleDescription, MappingDesc, DeviceHeap>;

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

    struct Evolution
    {
        template<class... T>
        void init(T... args){};
        template<class... T>
        void initEvolution(T... args){};
    };

    struct Simulation
    {
    private:
        Space gridSize{1, 1, 1};
        uint32_t steps;
        std::unique_ptr<pmacc::mpi::GatherSlice> gather;
        bool isMaster;
        Evolution evo;

        using MappingDesc = pmacc::MappingDescription<DIM3, pmacc::math::CT::Int<16, 16, 16>>;

    public:
        Simulation(uint32_t const steps, Space const& gridSize, Space const& devices, Space const& periodic)
            : gridSize(gridSize)
            , steps(steps)
        {
            pmacc::Environment<DIM3>::get().initDevices(devices, periodic);
            auto layout = initGrids(devices, periodic);
            initEvolution(layout);
            initCommunication();
        }

    private:
        pmacc::GridLayout<DIM3> initGrids(Space const& devices, Space const& periodic)
        {
            Space localGridSize(gridSize / devices);
            pmacc::GridController<DIM3>& gc = pmacc::Environment<DIM3>::get().GridController();
            pmacc::Environment<DIM3>::get().initGrids(gridSize, localGridSize, gc.getPosition() * localGridSize);
            const pmacc::SubGrid<DIM3>& subGrid = pmacc::Environment<DIM3>::get().SubGrid();
            return {subGrid.getLocalDomain().size, MappingDesc::SuperCellSize::toRT()};
        }

        void initEvolution(pmacc::GridLayout<DIM3> const& layout)
        {
            evo.init(layout.getDataSpace(), Space::create(1));
        }

        void initCommunication()
        {
            gather = std::make_unique<pmacc::mpi::GatherSlice>();
            isMaster = gather->participate(true);
        }

    public:
        Simulation& start()
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
    nbody::Simulation{steps, grid, devices, periodic}.start();
    pmacc::Environment<>::get().finalize();

    return 0;
}
