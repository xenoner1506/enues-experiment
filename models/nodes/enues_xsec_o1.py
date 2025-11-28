from __future__ import annotations

from typing import TYPE_CHECKING

from dag_modelling.core.input_strategy import AddNewInputAddNewOutput
from dag_modelling.core.node import Node
from dag_modelling.core.type_functions import (
    assign_axes_from_inputs_to_outputs,
    check_dimension_of_inputs,
    check_dtype_of_inputs,
    check_inputs_equivalence,
    copy_from_inputs_to_outputs,
)
from numba import njit
from numpy import pi, power, empty
from scipy.constants import value as constant

if TYPE_CHECKING:
    from dag_modelling.core.input import Input
    from dag_modelling.core.output import Output
    from numpy import double
    from numpy.typing import NDArray


class ENuESXsecO1(Node):
    #    """Inverse beta decay cross section by Vogel and Beacom."""

    __slots__ = (
        "_enu",
        "_t_e",
        "_sin_sq_weinberg",
        "_const_fermi",
        "_result",
        "_const_me",
        # "_const_fps",
    )

    _enu: Input
    _t_e: Input
    _sin_sq_weinberg: Input
    _const_fermi: Input
    _result: Output
    _const_me: Input
    # _const_fps: Input

    def __init__(self, name, *args, **kwargs):  # Вот здесь не понятно, что писать
        super().__init__(name, *args, **kwargs, input_strategy=AddNewInputAddNewOutput())
        self.labels.setdefaults(
            {
                "text": r"IBD cross section σ(Eν,cosθ), cm⁻²",
                "plot_title": r"IBD cross section $\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
                "latex": r"IBD cross section $\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
                "axis": r"$\sigma(E_{\nu}, \cos\theta)$, cm$^{-2}$",
            }
        )

        self._enu = self._add_input("enu", positional=True, keyword=True)
        self._t_e = self._add_input("t_e", positional=True, keyword=True)
        self._result = self._add_output(
            "result", positional=True, keyword=True
        )  # результат - сечение
        self._sin_sq_weinberg = self._add_input(
            "sin_sq_weinberg", positional=False, keyword=True
        )  # positional = False, тк считает квадрат синуса константой
        self._const_fermi = self._add_input("const_fermi", positional=False, keyword=True)
        self._const_me = self._add_input("ElectronMass", positional=False, keyword=True)
        # self._const_fps = self._add_input("PhaseSpaceFactor", positional=False, keyword=True)

    def _function(self):
        print(
            self._const_me.data[0],
            self._const_fermi.data[0],
            self._sin_sq_weinberg.data[0],)
        _enues_xsec(
            self._enu.data,
            self._t_e.data,
            self._result._data,
            self._const_me.data[0],
            self._sin_sq_weinberg.data[0],
            self._const_fermi.data[0],
            # self._const_fps.data[0],
        )

    # def _post_allocate(self) -> None:
    #     super()._post_allocate()
    #     self._result = empty(shape=(self._enu.dd.shape[0], self._t_e.dd.shape[0]), dtype=self._enu.dd.dtype)

    def _type_function(self) -> None:
        """A output takes this function to determine the dtype and shape."""
        check_dtype_of_inputs(
            self, slice(None), dtype="d"
        )  # проверяет входные данные на тип float64
        check_dimension_of_inputs(self, slice(0, 1), 2)  # проверяет раазмерность входных данных
        check_inputs_equivalence(self, slice(0, 1))  # проверяет эквивалентность 2х входов
        copy_from_inputs_to_outputs(self, "enu", "result", edges=False, meshes=False)  # ????
        assign_axes_from_inputs_to_outputs(
            self,
            ("enu", "t_e"),
            "result",
            assign_meshes=True,
            merge_input_axes=True,
        )


@njit(cache=True)
def _enues_xsec(  # Вроде функция ниже переписана
    EnuIn: NDArray[double],
    T_eIn: NDArray[double],
    Result: NDArray[double],
    ElectronMass: float,
    SinSqWeinberg: float,
    ConstFermi: float,
    # const_fps: float,
):

    g_L = 1 + 2 * SinSqWeinberg
    g_R = 2 * SinSqWeinberg
    sq_g_L = g_L**2
    sq_g_R = g_R**2
    composition_gL_gR_Emass = g_L * g_R * ElectronMass
    composition_sqCF_Emass_reverse2pi = ConstFermi**2 * ElectronMass / (2 * pi)

    result = Result.ravel()
    for i, (e_i, te_i) in enumerate(zip(EnuIn.ravel(), T_eIn.ravel())):
        if te_i < 2 * e_i * e_i / (ElectronMass + 2 * e_i):
            t = te_i / e_i
            a = sq_g_R * (1 - t) ** 2
            b = composition_gL_gR_Emass * t / e_i
            result[i] = composition_sqCF_Emass_reverse2pi * (sq_g_L + a - b)
        else:
            result[i] = 0.0
