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

#include <pmacc/identifier/value_identifier.hpp>
#include <pmacc/mappings/kernel/MappingDescription.hpp>

namespace nbody::infrastructure
{
    using MappingDesc = pmacc::MappingDescription<DIM3, pmacc::math::CT::Int<8, 8, 4>>;
    using Space = pmacc::DataSpace<DIM3>;

    using float3 = pmacc::math::Vector<float, 3u>;
    value_identifier(float3, position, float3::create(0.));
    value_identifier(float3, velocity, float3::create(0.));
    value_identifier(float, mass, 1.);

    constexpr float epsilon = 1.e-4;
    constexpr float timestep = 0.1;
    constexpr const uint32_t numSlots = 256;
} // namespace nbody::infrastructure
