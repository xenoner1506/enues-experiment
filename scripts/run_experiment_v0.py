#!/usr/bin/env python
from __future__ import annotations
from argparse import Namespace
import matplotlib.pyplot as plt

from dag_modelling.core.graph import Graph
from dag_modelling.core.storage import NodeStorage
from dag_modelling.tools.logger import DEBUG as INFO4
from dag_modelling.tools.logger import INFO1, INFO2, INFO3, set_level

from dag_modelling.tools import disable_implicit_numpy_multithreading

# isort: on

from argparse import Namespace

from dag_modelling.tools.logger import set_verbosity
from models import load_model

set_level(INFO1)


def main(opts: Namespace) -> None:
    if opts.verbose:
        opts.verbose = min(opts.verbose, 3)
        set_level(globals()[f"INFO{opts.verbose}"])

    override_indices = {idxdef[0]: tuple(idxdef[1:]) for idxdef in opts.index}
    model = load_model(
            opts.version,
            model_options=opts.model_options,
            close=opts.close,
            strict=opts.strict,
            override_indices=override_indices,
            parameter_values=opts.par
            )

    graph = model.graph
    storage = model.storage

    if not graph.closed:
        print("Nodes")
        print(storage("nodes").to_table(truncate="auto"))
        print("Outputs")
        print(storage("outputs").to_table(truncate="auto"))
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate="auto"))

        if opts.graph_auto:
            plot_graph(graph, storage)
        return

    if opts.print_all:
        print(storage.to_table(truncate="auto"))
    for sources in opts.print:
        for source in sources:
            print(storage(source).to_table(truncate="auto"))
    if len(storage("inputs")) > 0:
        print("Not connected inputs")
        print(storage("inputs").to_table(truncate="auto"))

    if opts.plot_all:
        storage("outputs").plot(folder=opts.plot_all)

    if opts.plot:
        folder, sources = opts.plot[0], opts.plot[1:]
        for source in sources:
            storage(source).plot(folder=f"{folder}/{source.replace('.', '/')}")
    if opts.plot_observables:
        plt.scatter(model.storage["outputs.edges.energy_final"].data[:-1], model.storage["outputs.eventscount.final.concatenated"].data)
        plt.scatter(model.storage["outputs.edges.energy_erec"].data[:-1], model.storage["outputs.eventscount.stages.erec.AD11"].data)
        plt.savefig("observables.pdf", dpi=150, bbox_inches='tight')
        plt.close()

    if opts.latex:
        storage.to_datax("output/model_v0_data.tex")

    if opts.graph_auto:
        plot_graph(graph, storage)

    if opts.graph_from_node:
        from dag_modelling.graphviz import GraphDot

        nodepath, filepath = opts.graph_from_node
        node = storage("nodes")[nodepath]
        GraphDot.from_node(
            node,
            show="all",
            mindepth=opts.mindepth,
            maxdepth=opts.maxdepth,
            keep_direction=True,
        ).savegraph(filepath)

    if opts.interactive:
        from IPython import embed

        embed(colors="neutral")



def plot_graph(graph: Graph, storage: NodeStorage) -> None:
    from dag_modelling.graphviz import GraphDot

    GraphDot.from_graph(graph, show="all").savegraph("output/model_v0.dot")
    GraphDot.from_graph(
        graph,
        show=["type", "mark", "label", "path"],
        filter={
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
    ).savegraph("output/model_v0_reduced.dot")
    GraphDot.from_graph(
        graph,
        show="all",
        filter={
            "reactor": [0],
            "detector": [0, 1],
            "isotope": [0],
            "period": [0],
            "background": [0],
        },
    ).savegraph("output/model_v0_reduced_full.dot")
    GraphDot.from_node(
        storage["nodes.statistic.nuisance.all"],
        show="all",
        mindepth=-1,
        maxdepth=0,
    ).savegraph("output/model_v0_nuisance.dot")
    # GraphDot.from_output(
    #     storage["outputs.statistic.stat.chi2p"],
    #     show="all",
    #     mindepth=-1,
    #     filter={
    #         "reactor": [0],
    #         "detector": [0],
    #         "isotope": [0],
    #         "period": [0],
    #         "background": [0],
    #     },
    #     maxdepth=0,
    # ).savegraph("output/model_v0_stat.dot")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "-v", "--verbose", default=0, action="count", help="verbosity level"
    )
    parser.add_argument(
        "-s",
        "--source-type",
        "--source",
        choices=("tsv", "hdf5", "root", "npz"),
        default="tsv",
        help="Data source type",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start IPython session",
    )

    plot = parser.add_argument_group("plot", "plotting related options")
    plot.add_argument(
        "--plot-all", help="plot all the nodes to the folder", metavar="folder"
    )
    plot.add_argument(
        "--plot",
        nargs="+",
        help="plot the nodes in storages",
        metavar=("folder", "storage"),
    )
    plot.add_argument(
        "--plot-observables",
        action="store_true",
        help="Plot observables"
    )

    storage = parser.add_argument_group("storage", "storage related options")
    storage.add_argument("-P", "--print-all", action="store_true", help="print all")
    storage.add_argument(
        "-p", "--print", action="append", nargs="+", default=[], help="print all"
    )
    storage.add_argument(
        "-l", "--latex", action="store_true", help="print latex tables with parameters"
    )

    graph = parser.add_argument_group("graph", "graph related options")
    graph.add_argument(
        "--no-close", action="store_false", dest="close", help="Do not close the graph"
    )
    graph.add_argument(
        "--no-strict", action="store_false", dest="strict", help="Disable strict mode"
    )
    graph.add_argument(
        "-i",
        "--index",
        nargs="+",
        action="append",
        default=[],
        help="override index",
        metavar=("index", "value1"),
    )

    dot = parser.add_argument_group("graphviz", "plotting graphs")
    dot.add_argument(
        "-g",
        "--graph-from-node",
        nargs=2,
        help="plot the graph starting from the node",
        metavar=("node", "file"),
    )
    dot.add_argument("--mindepth", "--md", type=int, help="minimal depth")
    dot.add_argument("--maxdepth", "--Md", type=int, help="maximaldepth depth")
    dot.add_argument(
        "--graph-auto", "--ga", action="store_true", help="plot graphs auto"
    )

    model = parser.add_argument_group("model", "model related options")
    model.add_argument("--seed", default=0, type=int, help="seed of randomization")
    model.add_argument(
        "--version",
        default="v0",
        help="model version",
    )
    model.add_argument("--model-options", "--mo", default={}, help="Model options as yaml dict")

    pars = parser.add_argument_group("pars", "setup pars")
    pars.add_argument(
        "--par", nargs=2, action="append", default=[], help="set parameter value"
    )

    main(parser.parse_args())
