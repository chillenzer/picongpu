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
#include <vector>

#include <../../../thirdParty/nlohmann_json/single_include/nlohmann/json.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_range.hpp>
#include <picongpu/traits/GetMetadata.hpp>

using Json = nlohmann::json;
using picongpu::traits::getMetadata;
using picongpu::traits::mergeMetadata;
using std::vector;

// The following are all different examples of what classes the `GetMetadata` trait can work with.

struct EmptyStruct
{
};

// First examples of runtime information
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

struct SomethingWithRTInfoFromFunction
{
    int info = 0;

    int infoForJson() const
    {
        return info * 42;
    }
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithRTInfoFromFunction>
{
    SomethingWithRTInfoFromFunction const obj;
    Json json() const
    {
        return {{"infoForJson", obj.infoForJson()}};
    }
};

struct SomeParameters
{
    static constexpr int info = 0;
};
// Examples of compile time information
struct SomethingWithCTInfo
{
    using Parameters = SomeParameters;
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithCTInfo>
{
    // isn't used but has to comply with the interface
    SomethingWithCTInfo obj;

    Json json() const
    {
        return {{"info", SomethingWithCTInfo::Parameters::info}};
    }
};

struct SomethingWithCTAndRTInfo
{
    using Parameters = SomeParameters;
    int infoRT = -42;
};

template<>
struct picongpu::traits::GetMetadata<SomethingWithCTAndRTInfo>
{
    // isn't used but has to comply with the interface
    SomethingWithCTAndRTInfo obj;

    Json json() const
    {
        return {{"infoCT", SomethingWithCTAndRTInfo::Parameters::info}, {"infoRT", obj.infoRT}};
    }
};

TEST_CASE("unit::GetMetadata", "[GetMetadata test]")
{
    SECTION("RT")
    {
        SECTION("EmptyStruct")
        {
            EmptyStruct const obj{};
            CHECK(getMetadata(obj).size() == 0u);
        }

        SECTION("Single info")
        {
            auto const i = GENERATE(range(0, 3));
            SomethingWithRTInfo const obj{i};
            CHECK(getMetadata(obj)["info"] == obj.info);
        }

        SECTION("Multiple info")
        {
            SomethingWithMoreRTInfo const obj{42, 'x'};
            CHECK(getMetadata(obj)["info"] == obj.info);
            CHECK(getMetadata(obj)["character"] == obj.character);
        }

        SECTION("Unused information")
        {
            SomethingWithUnusedRTInfo const obj{42, -42};
            CHECK(getMetadata(obj)["info"] == obj.info);
            CHECK(getMetadata(obj)["not_into_json"] == nullptr);
        }

        SECTION("Info from function")
        {
            auto const i = GENERATE(range(0, 3));
            SomethingWithRTInfoFromFunction const obj{i};
            CHECK(getMetadata(obj)["infoForJson"] == obj.info * 42);
        }
    }

    SECTION("CT")
    {
        SomethingWithCTInfo const obj{};
        CHECK(getMetadata(obj)["info"] == decltype(obj)::Parameters::info);
    }

    SECTION("Mixed CT and RT")
    {
        SomethingWithCTAndRTInfo const obj{};
        CHECK(getMetadata(obj)["infoCT"] == decltype(obj)::Parameters::info);
        CHECK(getMetadata(obj)["infoRT"] == obj.infoRT);
    }
}

TEST_CASE("unit::mergeMetadata", "[mergeMetadata test]")
{
    SECTION("Empty list")
    {
        vector<Json> const empty = {};
        CHECK(mergeMetadata(empty) == Json{});
    }

    SECTION("Copies single element")
    {
        // apparently braced construction is flawed:
        // https://json.nlohmann.me/home/faq/#brace-initialization-yields-arrays
        Json const content = {{"a", 1}};
        vector<Json> const singleElement{content};
        CHECK(mergeMetadata(singleElement) == content);
    }

    SECTION("Handles two elements")
    {
        Json const content1 = {{"a", 1}};
        Json const content2 = {{"b", 2}};
        Json const expected = {{"a", 1}, {"b", 2}};
        vector<Json> const twoElements({content1, content2});
        CHECK(mergeMetadata(twoElements) == expected);
    }
}
