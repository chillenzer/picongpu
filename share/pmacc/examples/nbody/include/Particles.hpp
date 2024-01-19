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
            template<typename T_Worker, typename T_ParBox, typename T_Mapping>
            void operator()(T_Worker const& worker, T_ParBox pb, T_Mapping mapper) const
            {
                // CAUTION: This currently only works for a single super cell and
                // a single frame.
                // TODO: Generalise!
                Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                auto frame = pb.getEmptyFrame(worker);
                pb.setAsLastFrame(worker, frame, superCellIdx);
                constexpr uint32_t cellsPerSupercell
                    = pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value;
                auto forEachCellInSuperCell = pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
                forEachCellInSuperCell(
                    [&frame](uint32_t const idx)
                    {
                        frame[idx][pmacc::multiMask_] = 1;
                        frame[idx][pmacc::localCellIdx_] = idx;
                    });
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
            auto const mapper = pmacc::makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(this->cellDescription);
            auto workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(detail::KernelFillGridWithParticles{}, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
            this->fillAllGaps();
        }
    };
} // namespace nbody
