/* Copyright 2025
 *
 * This file is part of PMacc.
 *
 * PMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * PMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with PMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */
#include "catch2/generators/catch_generators.hpp"
#include "catch2/matchers/catch_matchers.hpp"
#include "catch2/matchers/catch_matchers_range_equals.hpp"
#include "picongpu/plugins/binning/BinningData.hpp"
#include "picongpu/plugins/binning/DomainInfo.hpp"
#include "picongpu/plugins/binning/FilteredSpecies.hpp"
#include "picongpu/plugins/binning/FunctorDescription.hpp"
#include "picongpu/plugins/binning/axis/Axis.hpp"
#include "picongpu/plugins/binning/axis/LinearAxis.hpp"
#include "picongpu/plugins/binning/binners/FieldBinner.hpp"
#include "picongpu/plugins/binning/binners/ParticleBinner.hpp"
#include "pmacc/Environment.hpp"
#include "pmacc/mappings/kernel/MappingDescription.hpp"
#include "pmacc/meta/String.hpp"
#include "pmacc/particles/Identifier.hpp"

#include <pmacc/test/PMaccFixture.hpp>

#include <algorithm>
#include <tuple>
#include <type_traits>
#include <vector>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators_all.hpp>
#include <catch2/matchers/catch_matchers_all.hpp>
#include <openPMD/openPMD.hpp>


constexpr auto simDim = 3;

//! Helper to setup the PMacc environment
using TestFixture = pmacc::test::PMaccFixture<simDim>;
static TestFixture fixture;

using namespace picongpu::plugins::binning;
using SuperCellSize = typename picongpu::mCT::shrinkTo<picongpu::mCT::Int<8, 8, 4>, simDim>::type;

auto getAxisTuple()
{
    auto getPositionY = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo, auto const& particle) -> int
    {
        auto posBin = getParticlePosition<DomainOrigin::TOTAL>(domainInfo, particle);
        return posBin[1];
    };

    // Create Functor Description
    auto cellPositionYDescription = createFunctorDescription<int>(getPositionY, "position_axisY");

    // Create Axis Splitting
    auto rangeY = axis::Range{0, 1};
    auto cellY_splitting = axis::AxisSplitting(rangeY, 1);

    // Create Axis
    auto ax_y = axis::createLinear(cellY_splitting, cellPositionYDescription);
    return std::make_tuple(ax_y);
}

auto getAxisTupleField()
{
    auto getPositionY = [] ALPAKA_FN_ACC(auto const& worker, auto const& domainInfo) -> int { return 1; };

    // Create Functor Description
    auto cellPositionYDescription = createFunctorDescription<int>(getPositionY, "position_axisY");

    // Create Axis Splitting
    auto rangeY = axis::Range{0, 1};
    auto cellY_splitting = axis::AxisSplitting(rangeY, 1);

    // Create Axis
    auto ax_y = axis::createLinear(cellY_splitting, cellPositionYDescription);
    return std::make_tuple(ax_y);
}

template<typename T, size_t dim>
struct DummyBuffer
{
    std::vector<T> my_data{};

    DummyBuffer(size_t size) : my_data(size)
    {
    }

    DummyBuffer(DummyBuffer<T, dim> const& other) = default;
    DummyBuffer(DummyBuffer<T, dim>&& other) = default;

    decltype(auto) getHostBuffer()
    {
        return *this;
    }

    decltype(auto) getDeviceBuffer()
    {
        return *this;
    }

    void deviceToHost()
    {
    }

    void hostToDevice()
    {
    }

    auto data()
    {
        return my_data.data();
    }

    void setValue(T const& val)
    {
        std::fill(std::begin(my_data), std::end(my_data), val);
    }

    auto capacityND()
    {
        return pmacc::MemSpace<1>{my_data.size()};
    }

    decltype(auto) getDataBox()
    {
        return *this;
    }

    decltype(auto) operator()(pmacc::DataSpace<1> index)
    {
        return my_data[index.x()];
    }
};

namespace alpaka
{

    template<typename T, size_t d>
    struct IsKernelArgumentTriviallyCopyable<DummyBuffer<T, d>> : std::true_type
    {
    };

} // namespace alpaka

std::string outputName{"binnerOutputName"};
std::string functorName{"testFunctor"};
int iterationNumber{42};

template<typename T>
auto read_value(auto iterationNumber) -> T
{
    auto series
        = openPMD::Series(std::string("binningOpenPMD/") + outputName + "_%06T.bp4", openPMD::Access::READ_ONLY);
    auto i = series.iterations[iterationNumber];
    ::openPMD::MeshRecordComponent dataset
        = series.iterations[iterationNumber].meshes["Binning"][::openPMD::RecordComponent::SCALAR];
    ::openPMD::Extent extent = dataset.getExtent();
    ::openPMD::Offset offset(extent.size(), 0);
    std::vector<T> loadedVal(100, 5);
    dataset.loadChunk(std::shared_ptr<T>(loadedVal.data(), [](auto const*) {}), offset, extent);
    series.flush();
    series.iterations[iterationNumber].close();
    return loadedVal[2];
}

TEST_CASE("Binner")
{
    pmacc::Environment<simDim>::get().initGrids(
        pmacc::DataSpace<simDim>(8, 8, 4),
        pmacc::DataSpace<simDim>(8, 8, 4),
        pmacc::DataSpace<simDim>(0, 0, 0));
    auto GRID_VOLUME = pmacc::Environment<simDim>::get().SubGrid().getGlobalDomain().size.productOfComponents();
    SECTION("TRIVIAL Particle")
    {
        auto depData = createFunctorDescription<double>([]() -> double { return 0.; }, functorName);
        auto bd = ParticleBinningData(outputName, getAxisTuple(), std::tuple<>{}, depData, std::tuple<>{});

        auto cellDescription = pmacc::MappingDescription<simDim, SuperCellSize>(pmacc::DataSpace<simDim>(8, 8, 4));
        auto binner = ParticleBinner<std::remove_cvref_t<decltype(bd)>, DummyBuffer>(bd, &cellDescription);

        binner.notify(iterationNumber);
    }

    SECTION("TRIVIAL field")
    {
        auto perform_test = [](auto&& depData, auto expectedValue)
        {
            auto bd = FieldBinningData(outputName, getAxisTupleField(), std::tuple<>{}, depData, std::tuple<>{});
            auto cellDescription = pmacc::MappingDescription<simDim, SuperCellSize>(pmacc::DataSpace<simDim>(8, 8, 4));
            auto binner = FieldBinner(bd, &cellDescription);
            binner.notify(iterationNumber);

            CHECK(read_value<int>(iterationNumber) == expectedValue);
        };

        SECTION("constant values")
        {
            auto func = [](auto const worker, auto const& domainInfo) -> int { return GENERATE(1, 10, -41); };
            auto return_value = func(nullptr, nullptr);
            auto depData = createFunctorDescription<int>(func, functorName);
            perform_test(depData, return_value * GRID_VOLUME);
        }

        SECTION("linear values")
        {
            auto func = [](auto const worker, auto const& domainInfo) -> int { return domainInfo.localCellIdx[0]; };
            auto depData = createFunctorDescription<int>(func, functorName);
            auto gaussFormula = [](int n) { return n * (n - 1) / 2; };
            perform_test(depData, 8 * 4 * gaussFormula(8));
        }
    }
}
