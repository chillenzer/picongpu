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

#pragma once


namespace picongpu::particles::atomicPhysics
{
    //! linear approximation acceptance probability functor
    struct LinearApproximationProbability
    {
        /** probability for transition with initial and final atomic state the same
         *
         * @param rate rate R_ji of transition, with convention R_ji > 0, [1/UNIT_TIME]
         * @param timeStep time step length of the current atomic physics step(may be != PIC-time step),
         *  [UNIT_TIME]
         */
        HDINLINE static float_X probabilityChange(float_X const rate, float_X const timeStep)
        {
            // 1/UNIT_TIME * UNIT_TIME = unitless
            return rate * timeStep; // unitless
        }

        /** probability for transition with initial and final atomic state the same
         *
         * @param rate rate R_ii of transition, with convention R_ii < 0, [1/UNIT_TIME]
         * @param timeStep time step length of the current atomic physics step(may be != PIC-time step),
         *  [UNIT_TIME]
         */
        HDINLINE static float_X probabilityNoChange(float_X const rate, float_X const timeStep)
        {
            // remember (R_ii <= 0)
            // unitless + 1/UNIT_TIME * UNIT_TIME = unitless
            return 1._X + rate * timeStep; // unitless
        }
    };
} // namespace picongpu::particles::atomicPhysics
