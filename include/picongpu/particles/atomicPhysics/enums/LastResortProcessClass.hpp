/* Copyright 2023 Brian Marre
 *
 * This file is part of PIConGPU.
 *
 * PIConGPU is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PIConGPU is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

//! @file get last resort processClass from transition direction and active processes

#include "picongpu/simulation_defines.hpp"

#include "picongpu/particles/atomicPhysics/enums/ProcessClass.hpp"
#include "picongpu/particles/atomicPhysics/enums/TransitionDataSet.hpp"

#include <cstdint>

namespace picongpu::particles::atomicPhysics::enums
{
    template<TransitionDataSet T_TransitionDataSet>
    struct LastResort;

    template<>
    struct LastResort<TransitionDataSet::boundBoundUpward>
    {
        template<bool T_spontaneousDeexcitation>
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::electronicExcitation);
        }
    };

    template<>
    struct LastResort<TransitionDataSet::boundBoundDownward>
    {
        template<bool T_spontaneousDeexcitation>
        static constexpr uint8_t processClass()
        {
            if constexpr(T_spontaneousDeexcitation)
                return u8(ProcessClass::spontaneousDeexcitation);
            else
                return u8(ProcessClass::electronicDeexcitation);
        }
    };

    template<>
    struct LastResort<TransitionDataSet::boundFreeUpward>
    {
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::electronicIonization);
        }
    };

    template<>
    struct LastResort<TransitionDataSet::autonomousDownward>
    {
        static constexpr uint8_t processClass()
        {
            return u8(ProcessClass::autonomousIonization);
        }
    };
} // namespace picongpu::particles::atomicPhysics::enums
