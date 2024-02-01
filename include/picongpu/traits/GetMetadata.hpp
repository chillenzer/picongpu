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
    using Json = nlohmann::json;
    using std::vector;

    template<typename T>
    struct GetMetadata
    {
        // `obj` is not used in the default implementation
        // but we want the compiler to auto-generate the constructor
        //
        //    GetMetadata(T const& obj) = default;
        //
        // for us while abiding by the rule of zero.
        T const obj;

        Json json() const
        {
            return {};
        }
    };

    template<typename T>
    Json getMetadata(T const& obj)
    {
        return GetMetadata<T>{obj}.json();
    }

    Json mergeMetadata(vector<Json> const& metadata)
    {
        Json result = {};
        for(auto const& entry : metadata)
        {
            result.merge_patch(entry);
        }
        return result;
    }
} // namespace picongpu::traits
