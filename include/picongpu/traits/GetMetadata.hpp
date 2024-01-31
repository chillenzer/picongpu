/* Copyright 2024 Julian Lenz
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with PIConGPU.
 * If not, see <http://www.gnu.org/licenses/>.
 */

// TODO: This is a nasty hack because I somehow couldn't get the include to work. Revise later!
#include <../../../thirdParty/nlohmann_json/single_include/nlohmann/json.hpp>

namespace picongpu::traits
{
    using json = nlohmann::json;

    template<typename T>
    json getMetadata(T const& obj)
    {
        return {{"info", 42}};
    }
} // namespace picongpu::traits
