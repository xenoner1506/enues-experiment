from collections.abc import Mapping, Sequence
from itertools import product
from os.path import relpath
from pathlib import Path
from typing import Literal

from dag_modelling.bundles.file_reader import FileReader
from dag_modelling.bundles.load_graph import load_graph
from dag_modelling.bundles.load_parameters import load_parameters
from dag_modelling.bundles.load_record import load_record_data
from dag_modelling.core import Graph, NodeStorage
from dag_modelling.lib.arithmetic import Division, Product, Sum
from dag_modelling.lib.common import Array, Proxy, View
from dag_modelling.lib.hist import AxisDistortionMatrixPointwise, Rebin
from dag_modelling.lib.integration import Integrator
from dag_modelling.lib.interpolation import Interpolator
from dag_modelling.lib.linalg import Cholesky, VectorMatrixProduct
from dag_modelling.lib.physics import EnergyResolution
from dag_modelling.lib.statistics import (Chi2, CovarianceMatrixGroup,
                                          MonteCarlo)
from dag_modelling.lib.summation import ArraySum, SumMatOrDiag
from dag_modelling.tools.schema import LoadYaml
from dgm_reactor_neutrino import IBDXsecVBO1Group, InverseSquareLaw
from nested_mapping import NestedMapping
from numpy import ndarray
from numpy.random import Generator


class model_experiment_v1:
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
        index["spec"] = tuple(f"spec_scale_{i:02d}" for i in range(len(antineutrino_model_edges)))

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
            load_parameters(path="detector",   load=path_parameters/"detector_nelectrons_correction.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_eres.yaml")
            load_parameters(path="detector",   load=path_parameters/"detector_relative.yaml",)

            load_parameters(path="reactor",    load=path_parameters/"reactor_energy_per_fission.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_thermal_power_nominal.yaml")
            load_parameters(path="reactor",    load=path_parameters/"reactor_fission_fraction_scale.yaml")

            load_parameters(path="enues",      load=path_parameters/"enues.yaml")

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
            in_edges_final = arange(.05, 12.01, .05)

            edges_costheta, _ = Array.replicate(name="edges.costheta", array=[-1, 1])
            edges_energy_common, _ = Array.replicate(
                name="edges.energy_common", array=in_edges_fine
            )
            edges_energy_final, _ = Array.replicate(
                name="edges.energy_final", array=in_edges_final
            )
            View.replicate(name="edges.energy_enu", output=edges_energy_common)
            edges_energy_edep, _ = View.replicate(name="edges.energy_edep", output=edges_energy_common)

            edges_energy_t_e, _ = Array.replicate(name="edges.energy_t_e", array=linspace(0.0, 12.0, 1201))
            edges_energy_anue, _ = Array.replicate(name="edges.energy_anue", array=linspace(0.0, 12.0, 1201))
            edges_energy_evis, _ = View.replicate(name="edges.energy_evis", output=edges_energy_t_e)
            edges_energy_erec, _ = View.replicate(name="edges.energy_erec", output=edges_energy_t_e)

            Array.replicate(name="reactor_anue.spec_model_edges", array=antineutrino_model_edges)

            # ENuES
            integration_orders_t_e = Array.from_value("kinematics_enues_sampler.ordersx", 3, edges=edges_energy_t_e)
            integration_orders_anue = Array.from_value("kinematics_enues_sampler.ordersy", 1, edges=edges_energy_anue)

            Integrator.replicate(
                "gl2d",
                path="kinematics_enues",
                names={
                    "sampler": "sampler",
                    "integrator": "integral",
                    "mesh_x": "sampler.mesh_t_e",
                    "mesh_y": "sampler.mesh_anue",
                    "orders_x": "sampler.orders_t_e",
                    "orders_y": "sampler.orders_anue",
                },
                dropdim=True,
                replicate_outputs=combinations["anue_source.reactor.isotope.detector"]
            )
            integration_orders_t_e >> inputs.get_value("kinematics_enues.sampler.orders_t_e")
            integration_orders_anue >> inputs.get_value("kinematics_enues.sampler.orders_anue")

            from models.nodes.enues_xsec_o1 import ENuESXsecO1

            enues, _ = ENuESXsecO1.replicate(name="kinematics.enues")
            enues << storage("parameters.all.enues")
            enues << storage("parameters.constant.ibd")

            outputs.get_value("kinematics_enues.sampler.mesh_anue") >> inputs["kinematics.enues.enu"]
            outputs.get_value("kinematics_enues.sampler.mesh_t_e") >> inputs["kinematics.enues.t_e"]

            kinematic_integrator_enu = enues.outputs["result"]


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
            kinematic_integrator_enu >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre.xfine")

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
            # kinematic_integrator_enu >> inputs.get_value("reactor_anue.neutrino_per_fission_per_MeV_nominal.xfine")

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
                columns=("day", "livetime", "eff", "eff_livetime"),
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
                "daily_data.detector.eff_livetime",
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
                    outputs("daily_data.detector.eff_livetime"),
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
                    outputs("daily_data.detector.eff_livetime"),
                    name = "detector.eff_livetime",
                    )

            Product.replicate(
                    outputs("detector.eff_livetime"),
                    parameters.get_value("all.conversion.seconds_in_day_inverse"),
                    name="detector.eff_livetime_days",
                    allow_skip_inputs=True,
                    )

            Product.replicate(
                parameters.get_value("all.detector.n_electrons_nominal_ad"),
                parameters.get_dict("all.detector.n_electrons_correction"),
                name="detector.n_electrons",
                replicate_outputs=index["detector"],
            )

            Product.replicate(
                outputs.get_dict("reactor.fissions_per_second"),
                outputs.get_dict("daily_data.detector.eff_livetime"),
                name="reactor_detector.n_fissions_daily",
                replicate_outputs=combinations["reactor.isotope.detector"],
            )

            ArraySum.replicate(
                outputs.get_dict("reactor_detector.n_fissions_daily"),
                name="reactor_detector.n_fissions",
            )

            InverseSquareLaw.replicate(
                name="reactor_detector.baseline_factor_per_cm2",
                scale="m_to_cm",
                replicate_outputs=combinations["reactor.detector"],
            )
            parameters.get_dict("constant.baseline") >> inputs.get_dict(
                "reactor_detector.baseline_factor_per_cm2"
            )

            Product.replicate(
                outputs.get_dict("reactor_detector.n_fissions"),
                outputs.get_dict("detector.n_electrons"),
                outputs.get_dict("reactor_detector.baseline_factor_per_cm2"),
                parameters.get_value("all.detector.efficiency"),
                name="reactor_detector.n_fissions_nelectrons_per_cm2",
                replicate_outputs=combinations["reactor.isotope.detector"],
            )

            Product.replicate(
                outputs("reactor_anue.neutrino_per_fission_per_MeV_nominal_pre"),
                outputs["reactor_detector.n_fissions_nelectrons_per_cm2"],
                outputs["kinematics.enues"],
                name="kinematics.enues_anue",
                replicate_outputs=combinations["reactor.isotope.detector"],
            )

            outputs.get_dict(
                "kinematics.enues_anue"
            ) >> inputs.get_dict("kinematics_enues.integral")

            from models.nodes.integral_2d_1d import Integral2d1d

            integral_2d_1d = Integral2d1d.replicate(
                keepdim=0,
                name="kinematics_enues.integral1d",
                replicate_outputs=combinations["reactor.isotope.detector"],
            )

            outputs["kinematics_enues.integral.main"] >> inputs["kinematics_enues.integral1d"]

            Sum.replicate(
                outputs["kinematics_enues.integral1d"],
                name="eventscount.stages.evis",
                replicate_outputs=combinations["detector"],
            )

            EnergyResolution.replicate(path="detector.eres")

            nodes.get_value("detector.eres.sigma_rel") << parameters.get_dict(
                "constrained.detector.eres"
            )
            outputs.get_value("edges.energy_evis") >> inputs.get_value("detector.eres.e_edges")
            outputs.get_value("edges.energy_evis") >> inputs.get_value(
                "detector.eres.matrix.e_edges"
            )
            outputs.get_value("edges.energy_erec") >> inputs.get_value(
                "detector.eres.matrix.e_edges_out"
            )

            VectorMatrixProduct.replicate(
                name="eventscount.stages.erec",
                mode="column",
                replicate_outputs=combinations["detector"],
            )
            outputs.get_value("detector.eres.matrix") >> inputs.get_dict(
                "eventscount.stages.erec.matrix"
            )
            outputs.get_dict("eventscount.stages.evis") >> inputs.get_dict(
                "eventscount.stages.erec.vector"
            )

            Rebin.replicate(
                names={
                    "matrix": "detector.rebin.matrix_enues",
                    "product": "eventscount.final.enues",
                },
                replicate_outputs=combinations["detector"],
            )

            # Connect old (Erec) and new (final) energy edges.
            edges_energy_erec >> inputs.get_value("detector.rebin.matrix_enues.edges_old")
            edges_energy_final >> inputs.get_value("detector.rebin.matrix_enues.edges_new")
            # Pass the fine-bin spectra into inputs.
            outputs.get_dict("eventscount.stages.evis") >> inputs.get_dict(
                "eventscount.final.enues"
            )

            # Compute a product of global normalization and per-detector efficiency
            # factor, to be used to scale the IBD spectrum.
            # Product.replicate(
            #     parameters.get_value("all.detector.global_normalization"),
            #     parameters.get_dict("selected.detector.parameters_relative.efficiency_factor"),
            #     name="detector.normalization",
            #     replicate_outputs=index["detector"],
            # )

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
        parameter_values: Mapping[str, float | str] | Sequence[tuple[str, float | int]] = (),
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
