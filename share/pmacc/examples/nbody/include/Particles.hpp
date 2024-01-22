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
#include "pmacc/lockstep/WorkerCfg.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>
#include <pmacc/math/vector/Vector.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/particles/Identifier.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>

#include <cupla/device/Atomic.hpp>

namespace nbody
{
    using MappingDesc = pmacc::MappingDescription<DIM3, pmacc::math::CT::Int<8, 8, 4>>;
    using Space = pmacc::DataSpace<DIM3>;
    namespace detail
    {
        using float3 = pmacc::math::Vector<float, 3u>;
        value_identifier(float3, position, float3::create(0.));
        value_identifier(float3, velocity, float3::create(0.));
        value_identifier(float, mass, 1.);

        constexpr float epsilon = 1.e-4;
        constexpr float timestep = 0.1;

        constexpr const uint32_t numSlots = 256;
        using TrivialParticleDescription = pmacc::ParticleDescription<
            PMACC_CSTRING("particle"),
            std::integral_constant<uint32_t, numSlots>,
            MappingDesc::SuperCellSize,
            pmacc::MakeSeq_t<position, velocity, mass>>;

        using SpecialisedParticlesBase = pmacc::ParticlesBase<TrivialParticleDescription, MappingDesc, DeviceHeap>;


        template<typename T_Worker, typename T_ParBox, typename T_Mapping>
        auto createEmptyLastFrame(T_Worker const& worker, T_ParBox& pb, T_Mapping mapper)
        {
            Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
            auto frame = pb.getEmptyFrame(worker);
            pb.setAsLastFrame(worker, frame, superCellIdx);
            return frame;
        }
        template<typename T_Worker, typename T_Mapping>
        auto makeForEachInSuperCell(T_Worker const& worker, T_Mapping mapper)
        {
            constexpr uint32_t cellsPerSupercell
                = pmacc::math::CT::volume<typename T_Mapping::SuperCellSize>::type::value;
            return pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
        }

        template<typename T_Worker, typename T_ParBox, typename T_Mapping>
        auto kernelSetup(T_Worker const& worker, T_ParBox const& pb, T_Mapping mapper)
        {
            Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
            auto frame = pb.getLastFrame(superCellIdx);
            auto forEach = makeForEachInSuperCell(worker, mapper);
            return std::make_tuple(frame, forEach);
        }

        struct KernelFillGridWithParticles
        {
            template<typename T_Worker, typename T_ParBox, typename T_Mapping>
            void operator()(T_Worker const& worker, T_ParBox pb, T_Mapping mapper) const
            {
                // CAUTION: This currently only works for a single super cell and
                // a single frame.
                // TODO: Generalise!
                auto frame = createEmptyLastFrame(worker, pb, mapper);
                auto forEachCellInSuperCell = makeForEachInSuperCell(worker, mapper);
                forEachCellInSuperCell(
                    [&frame](uint32_t const idx)
                    {
                        frame[idx][pmacc::multiMask_] = 1;
                        frame[idx][pmacc::localCellIdx_] = idx;
                        frame[idx][detail::mass_] = 1.;
                        // TODO: This is a hack.
                        frame[idx][detail::position_].x()
                            = static_cast<float>(pmacc::math::mapToND(Space{8, 8, 4}, static_cast<int>(idx))[0]);
                        frame[idx][detail::position_].y()
                            = static_cast<float>(pmacc::math::mapToND(Space{8, 8, 4}, static_cast<int>(idx))[1]);
                        frame[idx][detail::position_].z()
                            = static_cast<float>(pmacc::math::mapToND(Space{8, 8, 4}, static_cast<int>(idx))[2]);
                    });
                worker.sync();
            }
        };

        template<typename T_Particle, typename T_Frame>
        float3 computeVelocity(T_Particle const& particle, T_Frame const& frame)
        {
            auto acceleration = float3::create(0.);
            // NOTE: Frames don't have (c)begin and (c)end, so we can't use a
            // range-based for loop here.
            for(uint32_t i = 0; i < numSlots; ++i)
            {
                auto const& other = frame[i]; // NOTE: Should be obtained directly from (auto const& other : frame)
                auto difference = other[position_] - particle[position_];
                auto denominator = sqrt(l2norm2(difference) + epsilon);
                acceleration += other[mass_] * difference / (denominator * denominator * denominator);
            }
            return particle[velocity_] + acceleration * timestep;
        }

        struct KernelUpdateVelocities
        {
            template<typename T_Worker, typename T_ParBox, typename T_Mapping>
            void operator()(T_Worker const& worker, T_ParBox pb, T_Mapping mapper) const
            {
                // CAUTION: This currently only works for a single super cell and
                // a single frame.
                // TODO: Generalise!
                auto [frame, forEachCellInSuperCell] = kernelSetup(worker, pb, mapper);
                forEachCellInSuperCell([&frame](uint32_t const idx)
                                       { frame[idx][detail::velocity_] = computeVelocity(frame[idx], frame); });
            }
        };
        struct KernelUpdatePositions
        {
            template<typename T_Worker, typename T_ParBox, typename T_Mapping>
            void operator()(T_Worker const& worker, T_ParBox pb, T_Mapping mapper) const
            {
                // CAUTION: This currently only works for a single super cell and
                // a single frame.
                // TODO: Generalise!
                auto [frame, forEachCellInSuperCell] = kernelSetup(worker, pb, mapper);
                forEachCellInSuperCell([&frame](uint32_t const idx)
                                       { frame[idx][detail::position_] += timestep * frame[idx][detail::velocity_]; });
            }
        };
    } // namespace detail

    // NOTE: This is only a class because ParticleBase has a protected constructor.
    struct Particles : public detail::SpecialisedParticlesBase
    {
        pmacc::AreaMapping<pmacc::type::CORE + pmacc::type::BORDER, MappingDesc> const mapper;
        pmacc::lockstep::MakeWorkerCfg_t<MappingDesc::SuperCellSize> const workerCfg{};

        // TODO: Actually write this, currently it just tries to pass everything
        // to the base
        template<typename... T>
        Particles(T... args) : detail::SpecialisedParticlesBase(args...)
                             , mapper(this->cellDescription)
        {
            initPositions();
        };

        void syncToDevice() override
        {
            // well-established recipe from picongpu/particles/Particles.tpp:
            // do nothing
        }

        void updateVelocities()
        {
            apply(detail::KernelUpdateVelocities{});
        }
        void updatePositions()
        {
            apply(detail::KernelUpdatePositions{});
        }

    private:
        void initPositions()
        {
            apply(detail::KernelFillGridWithParticles{});
        }

        template<typename T_Kernel>
        void apply(T_Kernel kernel)
        {
            PMACC_LOCKSTEP_KERNEL(kernel, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
        }
    };
} // namespace nbody
