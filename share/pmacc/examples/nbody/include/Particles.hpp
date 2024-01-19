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

#pragma once

#include "DeviceHeap.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>

namespace nbody
{
    using MappingDesc = pmacc::MappingDescription<DIM3, pmacc::math::CT::Int<8, 8, 4>>;
    using Space = pmacc::DataSpace<DIM3>;
    namespace detail
    {
        using float3 = pmacc::math::Vector<float, 3u>;
        value_identifier(float3, position, float3::create(0.));
        value_identifier(float, mass, 1.);

        constexpr const uint32_t numSlots = 256;
        using TrivialParticleDescription = pmacc::ParticleDescription<
            PMACC_CSTRING("particle"),
            std::integral_constant<uint32_t, numSlots>,
            MappingDesc::SuperCellSize,
            pmacc::MakeSeq_t<position, mass>>;

        using SpecialisedParticlesBase = pmacc::ParticlesBase<TrivialParticleDescription, MappingDesc, DeviceHeap>;

        struct KernelFillGridWithParticles
        {
            template<typename T_Worker, typename... T>
            void operator()(T_Worker const& worker, T... args) const
            {
                constexpr uint32_t cellsPerSupercell
                    = pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value;
                auto forEachCellInSuperCell = pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
            }
        };
    } // namespace detail

    struct Particles : public detail::SpecialisedParticlesBase
    {
        // TODO: Actually write this, currently it just tries to pass everything
        // to the base
        template<typename... T>
        Particles(T... args) : detail::SpecialisedParticlesBase(args...)
        {
            initPositions();
        };

        void syncToDevice() override
        {
            // well-established recipe from picongpu/particles/Particles.tpp:
            // do nothing
        }

    private:
        void initPositions()
        {
            auto const totalGpuCellOffset = computeTotalGpuCellOffset();
            auto const mapper = pmacc::makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(this->cellDescription);
            auto workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(detail::KernelFillGridWithParticles{}, workerCfg)
            (mapper.getGridDim())(totalGpuCellOffset, this->particlesBuffer->getDeviceParticleBox(), mapper);

            this->fillAllGaps();
        }

        Space computeTotalGpuCellOffset()
        {
            uint32_t const numSlides = 0u;
            pmacc::SubGrid<DIM3> const& subGrid = pmacc::Environment<DIM3>::get().SubGrid();
            Space localCells = subGrid.getLocalDomain().size;
            Space totalGpuCellOffset = subGrid.getLocalDomain().offset;
            totalGpuCellOffset.y() += numSlides * localCells.y();
            return totalGpuCellOffset;
        }
    };
} // namespace nbody
