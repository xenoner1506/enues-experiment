from collections.abc import Mapping, Sequence
from itertools import product
from os.path import relpath
from pathlib import Path
from typing import Literal

from numpy import ndarray
from numpy.random import Generator


from collections.abc import Mapping, Sequence
from itertools import product
from os.path import relpath
from pathlib import Path

from dag_modelling.core import Graph, NodeStorage
from nested_mapping import NestedMapping
from numpy import ndarray
from numpy.random import Generator

from dag_modelling.bundles.file_reader import FileReader
from dag_modelling.bundles.load_graph import load_graph
from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.bundles.load_record import load_record_data
from dag_modelling.lib.hist import AxisDistortionMatrixPointwise, Rebin
from dag_modelling.lib.arithmetic import (
    Division,
    Product,
    Sum,
)

from dag_modelling.lib.common import Array, Proxy, View
from dag_modelling.lib.integration import Integrator
from dag_modelling.lib.interpolation import Interpolator
from dag_modelling.lib.linalg import Cholesky, VectorMatrixProduct
from dag_modelling.lib.physics import EnergyResolution
from dag_modelling.lib.statistics import (
    Chi2,
    CovarianceMatrixGroup,
    MonteCarlo,
)
from dag_modelling.lib.summation import ArraySum, SumMatOrDiag
from dag_modelling.tools.schema import LoadYaml

from dgm_reactor_neutrino import (
    IBDXsecVBO1Group,
    InverseSquareLaw,
)



class model_experiment_v0:
    __slots__ = (
        "storage",
        "graph",
        "index",
        "combinations",
        "_override_indices",
        "_path_data",
        "_source_type",
        "_strict",
        "_close",
        "_generator",
    )

    storage: NodeStorage
    graph: Graph | None
    index: dict[str, tuple[str, ...]]
    combinations: dict[str, tuple[tuple[str, ...], ...]]
    _path_data: Path
    _override_indices: Mapping[str, Sequence[str]]
    _source_type: Literal["tsv"]
    _strict: bool
    _close: bool
    _generator: Generator

    def __init__(
        self,
        *,
        source_type: Literal["tsv"] = "tsv",
        strict: bool = True,
        close: bool = True,
        override_indices: Mapping[str, Sequence[str]] = {},
        seed: int = 0,
        parameter_values: dict[str, float | str] = {},
    ):
        self._strict = strict
        self._close = close

        self.graph = None
        self.storage = NodeStorage()
        self._path_data = Path("data-1ad-point")
        self._source_type = source_type
        self._override_indices = override_indices
        self._generator = self._create_generator(seed)

        self.index = {}
        self.combinations = {}

        self.build()

        if parameter_values:
            self.set_parameters(parameter_values)

    def build(self):
        storage = self.storage
        path_data = self._path_data

        path_parameters = path_data / "parameters"
        path_arrays = path_data / self._source_type

        from dag_modelling.tools.schema import LoadPy

        antineutrino_model_edges = LoadPy(
            path_parameters / "reactor_antineutrino_spectrum_edges.py",
            variable="edges",
            type=ndarray,
        )

        index_names = {
            "U235": "²³⁵U",
            "U238": "²³⁸U",
            "Pu239": "²³⁹Pu",
            "Pu241": "²⁴¹Pu",
        }

        # Provide a list of indices and their values. Values should be globally unique
        index = self.index
        index["isotope"] = ("U235", "U238", "Pu239", "Pu241")
        index["isotope_lower"] = tuple(i.lower() for i in index["isotope"])
        index["detector"] = ("AD11",)
        # index["detector"] = ("AD11", "AD12")
        index["subdetector"] = ("sub1",)
        # index["subdetector"] = ("sub1", "sub2", "sub3", "sub4", "sub5", "sub6")
        index["site"] = ("EH1",)
        index["reactor"] = ("R1",)
        index["anue_source"] = ("main", "offeq")
        index["anue_unc"] = ("uncorr", "corr")
        index["lsnl"] = ("nominal", "pull0", "pull1", "pull2", "pull3")
        index["lsnl_nuisance"] = ("pull0", "pull1", "pull2", "pull3")
        index["spec"] = tuple(
            f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges))
        )

        index.update(self._override_indices)

        index_all = index["isotope"] + index["detector"] + index["reactor"]
        set_all = set(index_all)
        if len(index_all) != len(set_all):
            raise RuntimeError("Repeated indices")

        required_combinations = tuple(index.keys()) + (
            "reactor.detector",
            "reactor.isotope",
            "reactor.isotope.detector",
            "anue_unc.isotope",
            "reactor.detector.subdetector",
        )
        # Provide the combinations of indices
        combinations = self.combinations
        for combname in required_combinations:
            combitems = combname.split(".")
            items = []
            for it in product(*(index[item] for item in combitems)):
                items.append(it)
            combinations[combname] = tuple(items)

        combinations["anue_source.reactor.isotope.detector"] = tuple(
            ("main",) + cmb for cmb in combinations["reactor.isotope.detector"]
        )

        with (
            Graph(close_on_exit=self._close, strict=self._strict) as graph,
            storage,
            FileReader,
        ):
            # fmt: off
            self.graph = graph
            #
            # Load parameters
            #
            load_parameters(path="oscprob",    load=path_parameters/"oscprob.yaml")
            load_parameters(path="oscprob",    load=path_parameters/"oscprob_solar.yaml", joint_nuisance=True)
            load_parameters(path="oscprob",    load=path_parameters/"oscprob_constants.yaml")

            load_parameters(path="ibd",        load=path_parameters/"pdg2012.yaml")
            load_parameters(path="ibd.csc",    load=path_parameters/"ibd_constants.yaml")
            load_parameters(path="conversion", load=path_parameters/"conversion_thermal_power.yaml")
            load_parameters(path="conversion", load=path_parameters/"conversion_oscprob_argument.yaml")

            load_parameters(                   load=path_parameters/"baselines.yaml")
            load_parameters(                   load=path_parameters/"baselines-weighted.yaml")

            load_parameters(path="detector",   load=path_parameters/"detector_efficiency.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_normalization.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_nprotons_correction.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_eres.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_relative.yaml",)

            load_parameters(path="reactor",    load=path_parameters/"reactor_energy_per_fission.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_thermal_power_nominal.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_fission_fraction_scale.yaml")

            # TODO: Add backgrounds
            # load_parameters(path="bkg.rate",   load=path_parameters/"bkg_rates.yaml")
            # fmt: on

            # Normalization constants
            load_parameters(
                format="value",
                state="fixed",
                parameters={
                    "conversion": {
                        "seconds_in_day_inverse": 1 / (60 * 60 * 24),
                    },
                    "oscprob": {
                        "gamma": 0.816191,
                        "delta": 300.875,
                    },
                },
                labels={
                    "conversion": {
                        "seconds_in_day_inverse": "One divided by seconds in day",
                    },
                    "oscprob": {
                        "gamma": "gamma",
                        "delta": "delta",
                    },
                },
            )

            nodes = storage.create_child("nodes")
            inputs = storage.create_child("inputs")
            outputs = storage.create_child("outputs")
            data = storage.create_child("data")
            parameters = storage("parameters")
            parameters_nuisance_normalized = storage("parameters.normalized")

            # fmt: off
            #
            # Create nodes
            #
            labels = LoadYaml(relpath(__file__.replace(".py", "_labels.yaml")))

            from numpy import arange, concatenate, linspace

            #
            # Define binning
            #
            in_edges_fine = linspace(0, 12, 481)
            # in_edges_final = concatenate(([0.7], arange(1.2, 8.01, 0.20), [12.0]))
            in_edges_final = arange(.95, 12.01, .05)

            edges_costheta, _ = Array.replicate(name="edges.costheta", array=[-1, 1])
            # edges_costheta, _ = Array.replicate(name="edges.costheta", array=[0])
            edges_energy_common, _ = Array.replicate(
                name="edges.energy_common", array=in_edges_fine
            )
            edges_energy_final, _ = Array.replicate(
                name="edges.energy_final", array=in_edges_final
            )
            View.replicate(name="edges.energy_enu", output=edges_energy_common)
            edges_energy_edep, _ = View.replicate(name="edges.energy_edep", output=edges_energy_common)
            edges_energy_evis, _ = View.replicate(name="edges.energy_evis", output=edges_energy_common)
            edges_energy_evis, _ = Array.replicate(name="edges.energy_evis_fine", array=linspace(0., 12., 2401))
            edges_energy_erec, _ = View.replicate(name="edges.energy_erec", output=edges_energy_common)

            Array.replicate(name="reactor_anue.spec_model_edges", array=antineutrino_model_edges)

            #
            # Integration, kinematics
            #
            integration_orders_edep = Array.from_value("kinematics_sampler.ordersx", 5, edges=edges_energy_edep)
            integration_orders_costheta = Array.from_value("kinematics_sampler.ordersy", 3, edges=edges_costheta)
            # integration_orders_costheta = Array.from_value("kinematics_sampler.ordersy", 1, edges=edges_costheta)

            Integrator.replicate(
                "gl2d",
                path="kinematics",
                names={
                    "sampler": "sampler",
                    "integrator": "integral",
                    "mesh_x": "sampler.mesh_edep",
                    "mesh_y": "sampler.mesh_costheta",
                    "orders_x": "sampler.orders_edep",
                    "orders_y": "sampler.orders_costheta",
                },
                replicate_outputs=combinations["anue_source.reactor.isotope.detector"]
            )
            integration_orders_edep >> inputs.get_value("kinematics.sampler.orders_edep")
            integration_orders_costheta >> inputs.get_value("kinematics.sampler.orders_costheta")

            ibd, _ = IBDXsecVBO1Group.replicate(path="kinematics.ibd", input_energy="edep")
            ibd << storage("parameters.constant.ibd")
            ibd << storage("parameters.constant.ibd.csc")
            outputs.get_value("kinematics.sampler.mesh_edep") >> ibd.inputs["edep"]
            outputs.get_value("kinematics.sampler.mesh_costheta") >> ibd.inputs["costheta"]
            kinematic_integrator_enu = ibd.outputs["enu"]

            # load_record_data(
            #     name="distributions",
            #     filenames=path_arrays/f"distributions.tsv",
            #     # replicate_outputs=index["detector"],
            #     # objects={"distributions": "AD11"},
            #     columns=("x", "y"),
            # )

            #
            # Oscillations
            #
            from dgm_reactor_neutrino import NueSurvivalProbability
            NueSurvivalProbability.replicate(
                name="oscprob",
                distance_unit="m",
                replicate_outputs=combinations["reactor.detector"],
                leading_mass_splitting_3l_name="DeltaMSq32",
                surprobArgConversion=True,
            )
            kinematic_integrator_enu >> inputs("oscprob.enu")
            parameters("constant.baseline_subdetector") >> inputs("oscprob.L")
            parameters.get_value("all.conversion.surprobArgConversion") >> inputs("oscprob.surprobArgConversion")
            nodes("oscprob") << parameters("free.oscprob")
            nodes("oscprob") << parameters("constant.oscprob")
            nodes("oscprob") << parameters("constrained.oscprob")

            # from models.nodes.NueSurvivalProbability4Weighted import NueSurvivalProbability4Weighted
            # NueSurvivalProbability4Weighted.replicate(
            #     name="oscprob_subdetectors",
            #     distance_unit="m",
            #     replicate_outputs=combinations["reactor.detector.subdetector"],
            #     surprobArgConversion=True,
            # )
            # kinematic_integrator_enu >> inputs("oscprob_subdetectors.enu")
            # parameters("constant.baseline_subdetector") >> inputs("oscprob_subdetectors.L")
            # parameters.get_value("all.conversion.surprobArgConversion") >> inputs("oscprob_subdetectors.surprobArgConversion")
            # nodes("oscprob_subdetectors") << parameters("free.oscprob")
            # nodes("oscprob_subdetectors") << parameters("constant.oscprob")
            # nodes("oscprob_subdetectors") << parameters("constrained.oscprob")

            # Product.replicate(
            #     parameters("all.baseline_subdetector_weights"),
            #     outputs("oscprob_subdetectors"),
            #     name="oscprob_weighted",
            #     replicate_outputs=combinations["reactor.detector.subdetector"],
            # )
            #
            # Sum.replicate(
            #     outputs("oscprob_weighted"),
            #     name="oscprob",
            #     replicate_outputs=combinations["reactor.detector"],
            # )

            #
            # Nominal antineutrino spectrum
            #
            load_graph(
                name = "reactor_anue.neutrino_per_fission_per_MeV_input",
                filenames = path_arrays / f"reactor_anue_spectra_50kev.tsv",
                x = "enu",
                y = "spec",
                merge_x = True,
                replicate_outputs = index["isotope"],
            )

            #
            # Pre-interpolate input spectrum on coarser grid
            # NOTE:
            #     - not needed with the current scheme:
            #         - spectrum correction applied by multiplication
            #     - introduced for the consistency with GNA
            #     - to be removed in v1 TODO
            #
            Interpolator.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer_pre",
                    "interpolator": "reactor_anue.neutrino_per_fission_per_MeV_nominal_pre",
                    },
                replicate_outputs = index["isotope"],
            )
            outputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_input.enu") >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre.xcoarse")
            outputs("reactor_anue.neutrino_per_fission_per_MeV_input.spec") >> inputs("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre.ycoarse")
            outputs.get_value("reactor_anue.spec_model_edges") >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre.xfine")

            #
            # Interpolate for the integration mesh
            #
            Interpolator.replicate(
                method = "exp",
                names = {
                    "indexer": "reactor_anue.spec_indexer",
                    "interpolator": "reactor_anue.neutrino_per_fission_per_MeV_nominal",
                    },
                replicate_outputs = index["isotope"],
            )
            outputs.get_value("reactor_anue.spec_model_edges") >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal.xcoarse")
            outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre") >> inputs("reactor_anue.neutrino_per_fission_per_MeV_nominal.ycoarse")
            kinematic_integrator_enu >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal.xfine")

            #
            # Antineutrino spectrum
            #
            Product.replicate(
                    outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal"),
                    name = "reactor_anue.part.neutrino_per_fission_per_MeV_main",
                    replicate_outputs=index["isotope"],
                    )

            #
            # Livetime
            #
            load_record_data(  # TODO: Change data
                name="daily_data.detector_all",
                filenames=path_arrays/f"livetimes.tsv",
                replicate_outputs=index["detector"],
                # objects = {"livetimes": "AD11"},
                columns=("day", "livetime", "eff", "efflivetime"),
            )
            from models.bundles.refine_detector_data import \
                refine_detector_data2
            refine_detector_data2(  # FIXME
                data("daily_data.detector_all"),
                data.create_child("daily_data.detector"),
                detectors = index["detector"]
            )

            load_record_data(
                name = "daily_data.reactor_all",
                filenames = path_arrays/f"weekly_power.tsv",
                replicate_outputs = index["reactor"],
                columns = ("week", "day", "core", "power") + index["isotope_lower"],
            )

            from models.bundles.refine_reactor_data import refine_reactor_data2
            refine_reactor_data2(
                data("daily_data.reactor_all"),
                data.create_child("daily_data.reactor"),
                reactors = index["reactor"],
                isotopes = index["isotope"],
            )

            Array.from_storage(
                "daily_data.detector.livetime",
                storage("data"),
                remove_processed_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.detector.eff",
                storage("data"),
                remove_processed_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.detector.efflivetime",
                storage("data"),
                remove_processed_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.reactor.power",
                storage("data"),
                remove_processed_arrays = True,
                dtype = "d"
            )

            Array.from_storage(
                "daily_data.reactor.fission_fraction",
                storage("data"),
                remove_processed_arrays = True,
                dtype = "d"
            )
            del storage["data.daily_data"]

            #
            # Neutrino rate
            #
            Product.replicate(
                    parameters("all.reactor.nominal_thermal_power"),
                    parameters.get_value("all.conversion.reactorPowerConversion"),
                    name = "reactor.thermal_power_nominal_MeVs",
                    replicate_outputs = index["reactor"]
                    )

            Product.replicate(
                    parameters("central.reactor.nominal_thermal_power"),
                    parameters.get_value("all.conversion.reactorPowerConversion"),
                    name = "reactor.thermal_power_nominal_MeVs_central",
                    replicate_outputs = index["reactor"]
                    )

            # Time dependent, fit dependent (non-nominal) for reactor core
            Product.replicate(
                    parameters("all.reactor.fission_fraction_scale"),
                    outputs("daily_data.reactor.fission_fraction"),
                    name = "daily_data.reactor.fission_fraction_scaled",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            #
            # Fission fraction normalized
            #
            Product.replicate(
                    parameters("all.reactor.energy_per_fission"),
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    name = "reactor.energy_per_fission_weighted_MeV",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Sum.replicate(
                    outputs("reactor.energy_per_fission_weighted_MeV"),
                    name = "reactor.energy_per_fission_average_MeV",
                    replicate_outputs=index["reactor"],
                    )

            Product.replicate(
                    outputs("daily_data.reactor.power"),
                    outputs("daily_data.reactor.fission_fraction_scaled"),
                    outputs("reactor.thermal_power_nominal_MeVs"),
                    name = "reactor.thermal_power_isotope_MeV_per_second",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            Division.replicate(
                    outputs("reactor.thermal_power_isotope_MeV_per_second"),
                    outputs("reactor.energy_per_fission_average_MeV"),
                    name = "reactor.fissions_per_second",
                    replicate_outputs=combinations["reactor.isotope"],
                    )

            # Effective number of fissions seen in Detector from Reactor from Isotope during Period
            Product.replicate(
                    outputs("reactor.fissions_per_second"),
                    outputs("daily_data.detector.efflivetime"),
                    name = "reactor_detector.number_of_fissions_daily",
                    replicate_outputs=combinations["reactor.isotope.detector"],
                    allow_skip_inputs = True,
                    )

            # Total effective number of fissions from a Reactor seen in the Detector during Period
            ArraySum.replicate(
                    outputs("reactor_detector.number_of_fissions_daily"),
                    name = "reactor_detector.number_of_fissions",
                    )

            # Baseline factor from Reactor to Detector: 1/(4πL²)
            InverseSquareLaw.replicate(
                name="baseline_factor_per_cm2",
                scale="m_to_cm",
                replicate_outputs=combinations["reactor.detector"]
            )
            parameters("constant.baseline") >> inputs("baseline_factor_per_cm2")

            # Number of protons per detector
            Product.replicate(
                    parameters.get_value("all.detector.nprotons_nominal_ad"),
                    parameters("all.detector.nprotons_correction"),
                    name = "detector.nprotons",
                    replicate_outputs = index["detector"]
            )

            # Number of fissions × N protons × ε / (4πL²)  (main)
            Product.replicate(
                    outputs("reactor_detector.number_of_fissions"),
                    outputs("detector.nprotons"),
                    outputs("baseline_factor_per_cm2"),
                    parameters.get_value("all.detector.efficiency"),
                    name = "reactor_detector.number_of_fissions_nprotons_per_cm2",
                    replicate_outputs=combinations["reactor.isotope.detector"],
                    )

            # Detector live time
            ArraySum.replicate(
                    outputs("daily_data.detector.livetime"),
                    name = "detector.livetime",
                    )

            ArraySum.replicate(
                    outputs("daily_data.detector.efflivetime"),
                    name = "detector.efflivetime",
                    )

            Product.replicate(
                    outputs("detector.efflivetime"),
                    parameters.get_value("all.conversion.seconds_in_day_inverse"),
                    name="detector.efflivetime_days",
                    allow_skip_inputs=True,
                    )

            #
            # Integrand: flux × oscillation probability × cross section
            # [Nν·cm²/fission/proton]
            #
            Product.replicate(
                    outputs.get_value("kinematics.ibd.crosssection"),
                    outputs.get_value("kinematics.ibd.jacobian"),
                    name="ibd.crosssection_jacobian",
            )

            Product.replicate(
                    outputs.get_value("ibd.crosssection_jacobian"),
                    outputs("oscprob"),
                    name="ibd.crosssection_jacobian_oscillations",
                    replicate_outputs=combinations["reactor.detector"]
            )

            Product.replicate(
                    outputs("ibd.crosssection_jacobian_oscillations"),
                    outputs("reactor_anue.part.neutrino_per_fission_per_MeV_main"),
                    name="neutrino_cm2_per_MeV_per_fission_per_proton.part.main",
                    replicate_outputs=combinations["reactor.isotope.detector"]
            )

            outputs("neutrino_cm2_per_MeV_per_fission_per_proton.part.main") >> inputs("kinematics.integral.main")

            #
            # Multiply by the scaling factors:
            #  - main:  fissions_per_second[p,r,i] × effective live time[p,d] × N protons[d] × efficiency[d]
            #
            Product.replicate(
                    outputs("kinematics.integral.main"),
                    outputs("reactor_detector.number_of_fissions_nprotons_per_cm2"),
                    name = "eventscount.parts.main",
                    replicate_outputs = combinations["reactor.isotope.detector"]
                    )

            # Debug node: eventscount.reactor_active_periods
            Sum.replicate(
                outputs("eventscount.parts.main"),
                name="eventscount.raw",
                replicate_outputs=index["detector"]
            )

            Product.replicate(
                parameters.get_value("all.detector.global_normalization"),
                parameters.get_value("constrained.detector.detector_relative.efficiency_factor"),
                name = "detector.normalization",
                replicate_outputs=index["detector"],
            )

            EnergyResolution.replicate(path="detector.eres")

            nodes.get_value("detector.eres.sigma_rel") << parameters("constrained.detector.eres")
            outputs.get_value("edges.energy_evis") >> inputs.get_value(
                "detector.eres.matrix.e_edges"
            )
            outputs.get_value("edges.energy_evis") >> inputs.get_value("detector.eres.e_edges")
            outputs.get_value("edges.energy_erec") >> inputs.get_value(
                "detector.eres.matrix.e_edges_out"
            )

            # EnergyResolution.replicate(path="detector.eres_fine")
            # nodes.get_value("detector.eres_fine.sigma_rel") << parameters("constrained.detector.eres")
            # outputs.get_value("edges.energy_evis_fine") >> inputs.get_value("detector.eres_fine.matrix.e_edges")
            # outputs.get_value("edges.energy_evis_fine") >> inputs.get_value("detector.eres_fine.e_edges")

            VectorMatrixProduct.replicate(name="eventscount.erec", mode="column", replicate_outputs=combinations["detector"])
            outputs.get_value("detector.eres.matrix") >> inputs("eventscount.erec.matrix")
            outputs("eventscount.raw") >> inputs("eventscount.erec.vector")

            Product.replicate(
                outputs("detector.normalization"),
                outputs("eventscount.erec"),
                name="eventscount.fine.ibd_normalized",
                replicate_outputs=index["detector"],
            )

            Rebin.replicate(
                names={"matrix": "detector.rebin_matrix_ibd", "product": "eventscount.final.ibd"},
                replicate_outputs=index["detector"],
            )
            edges_energy_erec >> inputs.get_value("detector.rebin_matrix_ibd.edges_old")
            edges_energy_final >> inputs.get_value("detector.rebin_matrix_ibd.edges_new")
            outputs("eventscount.fine.ibd_normalized") >> inputs("eventscount.final.ibd")


            Sum.replicate(
                outputs("eventscount.final.ibd"),
                name="eventscount.final.detector_period",
                replicate_outputs=index["detector"],
            )

            Sum.replicate(
                outputs("eventscount.final.detector_period"),
                name="eventscount.final.concatenated",
            )

            #
            # Covariance matrices
            #
            covariance = CovarianceMatrixGroup(store_to="covariance")

            for name, parameters_source in (
                    ("eres", "detector.eres"),
                    ("detector_relative", "detector.detector_relative"),
                    ("energy_per_fission", "reactor.energy_per_fission"),
                    # ("nominal_thermal_power", "reactor.nominal_thermal_power"),
                    ("fission_fraction", "reactor.fission_fraction_scale"),
            ):
                covariance.add_covariance_for(name, parameters_nuisance_normalized[parameters_source])
            covariance.add_covariance_sum()

            outputs.get_value("eventscount.final.concatenated") >> covariance

            #
            # Statistic
            #
            # Create Nuisance parameters
            Sum.replicate(outputs("statistic.nuisance.parts"), name="statistic.nuisance.all")

            MonteCarlo.replicate(
                name="data.pseudo.self",
                mode="asimov",
            )
            outputs.get_value("eventscount.final.concatenated") >> inputs.get_value("data.pseudo.self.data")

            Proxy.replicate(
                name="data.pseudo.proxy",
            )
            outputs.get_value("data.pseudo.self") >> inputs.get_value("data.pseudo.proxy.input")

            Cholesky.replicate(
                name="covariance.cholesky.proxy",
            )
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("covariance.cholesky.proxy")

            SumMatOrDiag.replicate(name="covariance.covmat_full_n")
            outputs.get_value("data.pseudo.proxy") >> nodes.get_value("covariance.covmat_full_n")
            outputs.get_value("covariance.covmat_syst.sum") >> nodes.get_value("covariance.covmat_full_n")

            Cholesky.replicate(name="cholesky.covmat_full_n")
            outputs.get_value("covariance.covmat_full_n") >> inputs.get_value("cholesky.covmat_full_n")

            # list_parameters_nuisance_normalized = list(parameters_nuisance_normalized.walkvalues())
            # npars_nuisance = len(list_parameters_nuisance_normalized)

            # from dagflow.lib.ParArrayInput import ParArrayInput
            # parinp_mc = ParArrayInput(
            #     name="mc.parameters.inputs",
            #     parameters=list_parameters_nuisance_normalized,
            # )

            # MonteCarlo.replicate(
            #     name="mc.parameters.toymc",
            #     mode="normal-unit",
            #     shape=(npars_nuisance,),
            #     generator=self._random_generator,
            # )
            # outputs.get_value("mc.parameters.toymc") >> parinp_mc
            # nodes["mc.parameters.inputs"] = parinp_mc

            Chi2.replicate(
                name="statistics.chi2_n",
            )
            outputs.get_value("eventscount.final.concatenated") >> inputs.get_value("statistics.chi2_n.theory")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistics.chi2_n.data")
            outputs.get_value("covariance.cholesky.proxy") >> inputs.get_value("statistics.chi2_n.errors")

            Sum.replicate(
                outputs.get_value("statistic.nuisance.all"),
                outputs.get_value("statistics.chi2_n"),
                name="statistics.chi2_n_full",
            )

            Chi2.replicate(name="statistics.chi2n_covmat")
            outputs.get_value("data.pseudo.proxy") >> inputs.get_value("statistics.chi2n_covmat.data")
            outputs.get_value("eventscount.final.concatenated") >> inputs.get_value("statistics.chi2n_covmat.theory")
            outputs.get_value("cholesky.covmat_full_n") >> inputs.get_value("statistics.chi2n_covmat.errors")

            # fmt: on

        processed_keys_set = set()
        # storage("nodes").read_labels(labels, processed_keys_set=processed_keys_set)
        # storage("outputs").read_labels(labels, processed_keys_set=processed_keys_set)
        storage("inputs").remove_connected_inputs()
        storage.read_paths(index=index)
        graph.build_index_dict(index)

        labels_mk = NestedMapping(labels, sep=".")
        if self._strict:
            for key in processed_keys_set:
                labels_mk.delete_with_parents(key)
            # if labels_mk:
            #     raise RuntimeError(
            #         f"The following label groups were not used: {tuple(labels_mk.walkkeys())}"
            #     )

    @staticmethod
    def _create_generator(seed: int) -> Generator:
        from numpy.random import MT19937, SeedSequence

        (sequence,) = SeedSequence(seed).spawn(1)
        algo = MT19937(seed=sequence.spawn(1)[0])
        return Generator(algo)

    def touch(self) -> None:
        frozen_nodes = (
            "pseudo.data",
            "cholesky.stat.frozen",
            "cholesky.covmat_full_p.stat_frozen",
            "cholesky.covmat_full_p.stat_unfrozen",
            "cholesky.covmat_full_n",
            "covariance.data.frozen",
        )
        for node in frozen_nodes:
            self.storage.get_value(f"nodes.{node}").touch()

    def set_parameters(
        self,
        parameter_values: (
            Mapping[str, float | str] | Sequence[tuple[str, float | int]]
        ) = (),
    ):
        parameters_storage = self.storage("parameters.all")
        if isinstance(parameter_values, Mapping):
            iterable = parameter_values.items()
        else:
            iterable = parameter_values

        for parname, svalue in iterable:
            value = float(svalue)
            par = parameters_storage[parname]
            par.push(value)
            print(f"Set {parname}={svalue}")

    def next_sample(self) -> None:
        self.storage.get_value("nodes.pseudo.parameters.toymc").next_sample()
        self.storage.get_value("nodes.pseudo.parameters.inputs.toymc").touch()
        self.storage.get_value("nodes.pseudo.data").next_sample()
        self.storage.get_value("nodes.pseudo.parameters.inputs.initial").touch()
