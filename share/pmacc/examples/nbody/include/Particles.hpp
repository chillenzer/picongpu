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
#include "Particles.kernel"
#include "infrastructure.hpp"

#include <pmacc/Environment.hpp>
#include <pmacc/dimensions/GridLayout.hpp>
#include <pmacc/mappings/kernel/AreaMapping.hpp>
#include <pmacc/math/vector/Vector.hpp>
#include <pmacc/meta/String.hpp>
#include <pmacc/particles/ParticleDescription.hpp>
#include <pmacc/particles/ParticlesBase.hpp>

#include <cupla/device/Atomic.hpp>


namespace nbody::particles
{
    using nbody::infrastructure::MappingDesc;
    namespace detail
    {
        using nbody::infrastructure::mass;
        using nbody::infrastructure::numSlots;
        using nbody::infrastructure::position;
        using nbody::infrastructure::velocity;

        using TrivialParticleDescription = pmacc::ParticleDescription<
            PMACC_CSTRING("particle"),
            std::integral_constant<uint32_t, numSlots>,
            MappingDesc::SuperCellSize,
            pmacc::MakeSeq_t<position, velocity, mass>>;

        using SpecialisedParticlesBase = pmacc::ParticlesBase<TrivialParticleDescription, MappingDesc, DeviceHeap>;
    } // namespace detail

    // NOTE: This is only a class because ParticleBase has a protected constructor.
    struct Particles : public detail::SpecialisedParticlesBase
    {
        pmacc::AreaMapping<pmacc::type::CORE + pmacc::type::BORDER, MappingDesc> const mapper;
        pmacc::lockstep::MakeWorkerCfg_t<MappingDesc::SuperCellSize> const workerCfg{};

        // TODO: Actually write this, currently it just tries to pass everything
        // to the base
        Particles(pmacc::GridLayout<DIM3> const& layout)
            : detail::SpecialisedParticlesBase(
                std::make_shared<DeviceHeap>(),
                MappingDesc{layout.getDataSpaceWithoutGuarding()})
            , mapper(this->cellDescription)
        {
            initPositions();
        };

        void syncToDevice() override
        {
            // well-established recipe from picongpu/particles/Particles.tpp:
            // do nothing
        }

        // NOTE: Couldn't make these free functions because `cellDescription` is
        // protected.
        void updateVelocities()
        {
            apply(kernels::KernelUpdateVelocities{});
        }
        void updatePositions()
        {
            apply(kernels::KernelUpdatePositions{});
        }

    private:
        void initPositions()
        {
            apply(kernels::KernelFillGridWithParticles{});
        }

        template<typename T_Kernel>
        void apply(T_Kernel kernel)
        {
            PMACC_LOCKSTEP_KERNEL(kernel, workerCfg)
            (mapper.getGridDim())(this->particlesBuffer->getDeviceParticleBox(), mapper);
        }
    };
} // namespace nbody::particles
