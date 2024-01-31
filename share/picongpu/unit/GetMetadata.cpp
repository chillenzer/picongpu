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
#include <picongpu/traits/GetMetadata.hpp>

using picongpu::traits::getMetadata;

struct SomethingWithRTInfo
{
    int info = 0;
};

TEST_CASE("unit::GetMetadata", "[GetMetadata test]")
{
    SECTION("RT")
    {
        SomethingWithRTInfo obj{42};
        CHECK(getMetadata(obj)["info"] == obj.info);
    }
}
