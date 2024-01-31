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

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <picongpu/traits/GetMetadata.hpp>

using picongpu::traits::getMetadata;

struct SomethingWithRTInfo
{
    int info = 0;
};

struct SomethingWithMoreRTInfo
{
    int info = 0;
    char character = 'j';
};

TEST_CASE("unit::GetMetadata", "[GetMetadata test]")
{
    SECTION("RT")
    {
        auto i = GENERATE(range(0, 3));
        SECTION("Single info")
        {
            SomethingWithRTInfo obj{i};
            CHECK(getMetadata(obj)["info"] == obj.info);
        }
        SECTION("Multiple info")
        {
            SomethingWithMoreRTInfo obj{i, 'x'};
            CHECK(getMetadata(obj)["info"] == obj.info);
            CHECK(getMetadata(obj)["character"] == obj.character);
        }
    }
}
