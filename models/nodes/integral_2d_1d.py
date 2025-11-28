from numpy import sum

from dag_modelling.core.input_strategy import AddNewInputAddNewOutput
from dag_modelling.core.node import Node
from dag_modelling.core.type_functions import check_node_has_inputs, check_inputs_are_matrices_or_diagonals, copy_from_inputs_to_outputs
from dag_modelling.core.input import Input


class Integral2d1d(Node):

    __slots__ = (
        "_keepdim",
        "_dropdim",
        "_step",
        "_orders_x_input",
        "_orders_y_input",
    )
    _orders_x_input: Input
    _orders_y_input: Input

    def __init__(self, *args, keepdim, step=1, **kwargs):
        kwargs.setdefault(
            "input_strategy",
            AddNewInputAddNewOutput(input_fmt="matrix", output_fmt="S"),
        )
        super().__init__(*args, **kwargs)
        self._labels.setdefault("mark", "V→S")
        self._step = step
        self._keepdim = keepdim
        self._dropdim = (keepdim + 1) % 2
        self._orders_x_input = self._add_input("orders_x", positional=False)
        self._orders_y_input = self._add_input("orders_y", positional=False)

    def fcn(self):
        self.inputs.touch()

        for indata, outdata in zip(self.inputs.iter_data(), self.outputs.iter_data_unsafe()):
            sum(indata * self._step, axis=self._dropdim, out=outdata)

    def _type_function(self) -> None:
        check_node_has_inputs(self)
        ndim = check_inputs_are_matrices_or_diagonals(self, slice(None), check_square=True)
        copy_from_inputs_to_outputs(self, slice(None), slice(None))

        self.function = self.fcn

        input0 = self.inputs[0]
        dtype = input0.dd.dtype
        shape = input0.dd.shape[self._keepdim]
        edges = input0.dd.axes_edges[self._keepdim]

        for output in self.outputs:
            output.dd.dtype = dtype
            output.dd.shape = (shape,)
            output.dd.axes_edges = edges
