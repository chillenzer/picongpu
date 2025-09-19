"""
This file is part of PIConGPU.
Copyright 2025 PIConGPU contributors
Authors: Julian Lenz
License: GPLv3+
"""

from datetime import datetime, timezone
from matplotlib.colors import LogNorm
import numpy as np
from matplotlib import pyplot as plt
from openpmd_api import Mesh_Record_Component, Series, Access_Type
from picongpu.piccom.schema import LogEntry
from scipy.constants import elementary_charge
from sympy import integrate, symbols

from picongpu.piccom.adaptor import LocalFolderAdaptor, Result
from picongpu.pypicongpu.simulation_logger.simulation_logger import _generate_action_id

from .parameters import BOX_SIZE, DIRECTORIES, MAX_STEPS, foil


def total_particle_mass(density, width, mass, box_size):
    return float(mass * integrate(foil(density, width)(*symbols("x,y,z"))).subs(zip("xyz", box_size)))


def store_total_particle_mass(communicator):
    # Not sure why you'd want to do that but here we compute
    # the total electron mass in our box
    # and store it as an additional parameter.
    batches = LocalFolderAdaptor(communicator).get(
        {
            "id": "ids",
            "physics": {
                "density": "additional_parameters/density",
                "width": "additional_parameters/width",
                "mass": "species/electron/constants/mass/mass_si",
            },
            "full": "full_metadata",
        },
        # using batches here is just for fun,
        # you could leave this out and you'd get a single iterable
        # with all the information
        batch_size=5,
    )
    for batch in batches:
        for info in batch:
            payload = {"total_particle_mass": total_particle_mass(**info["physics"], box_size=BOX_SIZE)}
            parent_action = next(
                filter(
                    lambda obj: obj[1]["action_name"] == "generate_input_files",
                    info["full"]["log"].items(),
                )
            )[0]
            communicator.update_one(
                {"_id": info["id"]},
                {
                    "$set": {
                        f"log.{_generate_action_id(payload)}": LogEntry(
                            action_name="additional_parameters",
                            update_of=parent_action,
                            timestamp=datetime.now(timezone.utc),
                            content=payload,
                        ).model_dump(mode="json")
                    }
                },
            )


def _extract_spectrum(series):
    spectrum_info = series.iterations[MAX_STEPS].meshes["Binning"][Mesh_Record_Component.SCALAR]
    spectrum = np.array(spectrum_info[:, :]) * 1.0 / 1e-12 * elementary_charge / 1e6 * 1 / 1e3
    E_bins = np.array(spectrum_info.get_attribute("Energy_bin_edges")) / elementary_charge / 1.0e6
    theta_bins = np.array(spectrum_info.get_attribute("pointingXY_bin_edges"))
    series.flush()
    return E_bins, theta_bins, spectrum[1:-1, 1:-1]


def plot_electron_spectrum(communicator):
    results = LocalFolderAdaptor(communicator).get(
        [
            ["additional_parameters/width", "laser/data/pulse_duration_si"],
            Result("binning/electron_spectrum/path"),
        ],
        status={"run": "success"},
    )
    widths = sorted(set(map(lambda r: r[0][0], results)))
    durations = sorted(set(map(lambda r: r[0][1], results)))

    fig, axes = plt.subplots(len(widths), len(durations), layout="constrained")
    # We need to do this explicitly in the case where one of widths or durations only has a single entry.
    axes = axes.reshape(len(widths), len(durations))
    results = {
        (width, duration): (
            axes[widths.index(width), durations.index(duration)],
            _extract_spectrum(Series(path, Access_Type.read_only)),
        )
        for (width, duration), path in results
    }

    spectra = [s for a, (e, t, s) in results.values()]
    norm = LogNorm(vmin=np.min(spectra), vmax=np.max(spectra))

    for ax, (e, theta, spectrum) in results.values():
        im = ax.pcolormesh(e, theta, spectrum, norm=norm)

    cb = fig.colorbar(im, ax=axes)
    cb.set_label(r"$\frac{\mathrm{d}^2 Q}{\mathrm{d} E \mathrm{d}\theta} \, \mathrm{[pC/MeV/mrad]}$")

    fig.suptitle("Electron Spectra")
    fig.supxlabel(r"$E \, \mathrm{[MeV]}$", fontsize=18)
    fig.supylabel(r"$\theta \, \mathrm{[mrad]}$", fontsize=18)

    filename = DIRECTORIES["plot"]() / "electron_spectra.svg"
    filename.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filename)


def postprocessing(communicator):
    store_total_particle_mass(communicator)
    plot_electron_spectrum(communicator)
