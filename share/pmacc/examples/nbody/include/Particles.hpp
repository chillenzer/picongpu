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
#include "cupla/device/Atomic.hpp"
#include "pmacc/math/vector/Vector.hpp"
#include "pmacc/particles/Identifier.hpp"

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
        struct CheckInit
        {
            template<typename T_Worker, typename T_ParBox, typename T_Mapping>
            void operator()(T_Worker const& worker, T_ParBox pb, T_Mapping mapper) const
            {
                // CAUTION: This currently only works for a single super cell and
                // a single frame.
                // TODO: Generalise!
                Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                auto frame = pb.getLastFrame(superCellIdx);
                constexpr uint32_t cellsPerSupercell
                    = pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value;
                auto forEachCellInSuperCell = pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
                PMACC_SMEM(worker, correct, bool);
                correct = true;
                forEachCellInSuperCell(
                    [&frame, &worker, &correct](uint32_t const idx)
                    {
                        cupla::atomicAnd(
                            worker.getAcc(),
                            &correct,
                            frame[idx][pmacc::multiMask_] == 1 && frame[idx][pmacc::localCellIdx_] == idx
                                && frame[idx][detail::position_] == float3::create(0.)
                                && frame[idx][detail::velocity_] == float3::create(0.)
                                && frame[idx][detail::mass_] == 1.,
                            ::alpaka::hierarchy::Threads{});
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
                const auto& other = frame[i];
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
                Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                auto frame = pb.getLastFrame(superCellIdx);
                constexpr uint32_t cellsPerSupercell
                    = pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value;
                auto forEachCellInSuperCell = pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
                forEachCellInSuperCell([&frame](uint32_t const idx)
                                       { frame[idx][detail::velocity_] = computeVelocity(frame[idx], frame); });
                worker.sync();
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
                Space const superCellIdx(mapper.getSuperCellIndex(Space(cupla::blockIdx(worker.getAcc()))));
                auto frame = pb.getLastFrame(superCellIdx);
                constexpr uint32_t cellsPerSupercell
                    = pmacc::math::CT::volume<MappingDesc::SuperCellSize>::type::value;
                auto forEachCellInSuperCell = pmacc::lockstep::makeForEach<cellsPerSupercell>(worker);
                forEachCellInSuperCell([&frame](uint32_t const idx)
                                       { frame[idx][detail::position_] += timestep * frame[idx][detail::velocity_]; });
                worker.sync();
            }
        };
    } // namespace detail

    // NOTE: This is only a class because ParticleBase has a protected constructor.
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

        void updateVelocities()
        {
            auto const mapper = pmacc::makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(this->cellDescription);
            auto workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(detail::KernelUpdateVelocities{}, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
        }
        void updatePositions()
        {
            auto const mapper = pmacc::makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(this->cellDescription);
            auto workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(detail::KernelUpdatePositions{}, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
        }

    private:
        void initPositions()
        {
            auto const mapper = pmacc::makeAreaMapper<pmacc::type::CORE + pmacc::type::BORDER>(this->cellDescription);
            auto workerCfg = pmacc::lockstep::makeWorkerCfg(MappingDesc::SuperCellSize{});
            PMACC_LOCKSTEP_KERNEL(detail::KernelFillGridWithParticles{}, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
            this->fillAllGaps();
            PMACC_LOCKSTEP_KERNEL(detail::CheckInit{}, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
        }
    };
} // namespace nbody
