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
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <picongpu/traits/GetMetadata.hpp>

using Json = nlohmann::json;
using picongpu::traits::getMetadata;

struct EmptyStruct
{
};

struct SomethingWithRTInfo
{
    int info = 0;
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithRTInfo>
{
    SomethingWithRTInfo const obj;
    Json json() const
    {
        return {{"info", obj.info}};
    }
};

struct SomethingWithMoreRTInfo
{
    int info = 0;
    char character = 'j';
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithMoreRTInfo>
{
    SomethingWithMoreRTInfo const obj;
    Json json() const
    {
        return {{"info", obj.info}, {"character", obj.character}};
    }
};

struct SomethingWithUnusedRTInfo
{
    int info = 0;
    int not_into_json = -1;
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithUnusedRTInfo>
{
    SomethingWithUnusedRTInfo const obj;
    Json json() const
    {
        // does not use the `not_into_json` attribute at all
        return {{"info", obj.info}};
    }
};

TEST_CASE("unit::GetMetadata", "[GetMetadata test]")
{
    SECTION("RT")
    {
        SECTION("EmptyStruct")
        {
            EmptyStruct obj{};
            CHECK(getMetadata(obj).size() == 0u);
        }

        SECTION("Single info")
        {
            auto i = GENERATE(range(0, 3));
            SomethingWithRTInfo obj{i};
            CHECK(getMetadata(obj)["info"] == obj.info);
        }

        SECTION("Multiple info")
        {
            SomethingWithMoreRTInfo obj{42, 'x'};
            CHECK(getMetadata(obj)["info"] == obj.info);
            CHECK(getMetadata(obj)["character"] == obj.character);
        }

        SECTION("Unused information")
        {
            SomethingWithUnusedRTInfo obj{42, -42};
            CHECK(getMetadata(obj)["info"] == obj.info);
            CHECK(getMetadata(obj)["not_into_json"] == nullptr);
        }
    }
}
