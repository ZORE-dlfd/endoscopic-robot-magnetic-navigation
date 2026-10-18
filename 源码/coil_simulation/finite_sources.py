from dataclasses import asdict, dataclass

import numpy as np
from numba import njit
from numpy.polynomial.legendre import leggauss


MU0 = 4e-7 * np.pi


@dataclass(frozen=True)
class CoilDesign:
    core_length_m: float = 0.052
    core_radius_m: float = 0.016
    winding_length_m: float = 0.044
    winding_inner_radius_m: float = 0.017
    winding_outer_radius_m: float = 0.028
    turns: int = 1000
    copper_diameter_m: float = 0.00055
    insulated_diameter_m: float = 0.00060
    relative_permeability: float = 2000.0
    saturation_polarization_t: float = 1.8
    copper_resistivity_ohm_m: float = 1.724e-8
    current_limit_a: float = 10.0
    ideal_driver_voltage_limit_v: float = 150.0

    @property
    def saturation_magnetization_a_m(self):
        return self.saturation_polarization_t / MU0

    @property
    def packing_fraction(self):
        wire_area = np.pi * (self.insulated_diameter_m / 2)**2
        pack_area = self.winding_length_m * (
            self.winding_outer_radius_m - self.winding_inner_radius_m
        )
        return self.turns * wire_area / pack_area

    @property
    def resistance_ohm(self):
        wire_length = self.turns * np.pi * (
            self.winding_inner_radius_m + self.winding_outer_radius_m
        )
        copper_area = np.pi * (self.copper_diameter_m / 2)**2
        return self.copper_resistivity_ohm_m * wire_length / copper_area

    @property
    def demagnetizing_factor(self):
        semi_major = self.core_length_m / 2
        semi_minor = self.core_radius_m
        if semi_major <= semi_minor:
            raise ValueError("The chosen prolate-spheroid approximation requires length > diameter")
        eccentricity = np.sqrt(1 - (semi_minor / semi_major)**2)
        return (1 - eccentricity**2) / (2 * eccentricity**3) * (
            np.log((1 + eccentricity) / (1 - eccentricity)) - 2 * eccentricity
        )

    def dictionary(self):
        return {
            **asdict(self),
            "packing_fraction": self.packing_fraction,
            "resistance_ohm_at_assumed_temperature": self.resistance_ohm,
            "demagnetizing_factor_prolate_approximation": self.demagnetizing_factor,
            "saturation_magnetization_a_m": self.saturation_magnetization_a_m,
            "parameter_provenance": "core dimensions selected by user; remaining parameters hypothetical, not measured",
        }


def _gauss_interval(count, lower, upper):
    nodes, weights = leggauss(count)
    return (lower + upper) / 2 + nodes * (upper - lower) / 2, weights * (upper - lower) / 2


def _core_nodes(design, angular_count=32, radial_count=5):
    radii, radial_weights = _gauss_interval(radial_count, 0, design.core_radius_m)
    angles = np.arange(angular_count) * 2 * np.pi / angular_count
    coordinates = []
    weights = []
    for sign in (-1, 1):
        for radius, radial_weight in zip(radii, radial_weights):
            for angle in angles:
                coordinates.append([
                    radius * np.cos(angle), radius * np.sin(angle),
                    sign * design.core_length_m / 2,
                ])
                weights.append(sign * radius * radial_weight * 2 * np.pi / angular_count)
    return np.array(coordinates), np.array(weights)


def _winding_nodes(design, angular_count=32, radial_count=3, axial_count=8):
    radii, radial_weights = _gauss_interval(
        radial_count, design.winding_inner_radius_m, design.winding_outer_radius_m
    )
    heights, axial_weights = _gauss_interval(
        axial_count, -design.winding_length_m / 2, design.winding_length_m / 2
    )
    angles = np.arange(angular_count) * 2 * np.pi / angular_count
    coordinates = []
    elements = []
    denominator = design.winding_length_m * (
        design.winding_outer_radius_m - design.winding_inner_radius_m
    )
    for radius, radial_weight in zip(radii, radial_weights):
        for height, axial_weight in zip(heights, axial_weights):
            turn_weight = design.turns * radial_weight * axial_weight / denominator
            for angle in angles:
                coordinates.append([radius * np.cos(angle), radius * np.sin(angle), height])
                elements.append(
                    np.array([-np.sin(angle), np.cos(angle), 0.0])
                    * radius * 2 * np.pi / angular_count * turn_weight
                )
    return np.array(coordinates), np.array(elements)


@njit(cache=True)
def _integrate_core(point, nodes, weights):
    field = np.zeros(3)
    gradient = np.zeros((3, 3))
    for element_index in range(len(weights)):
        displacement = point - nodes[element_index]
        distance_squared = np.dot(displacement, displacement)
        if distance_squared < 1e-20:
            raise ValueError("A field query intersects a magnetic-charge quadrature node")
        inverse_cubed = distance_squared**-1.5
        factor = 1e-7 * weights[element_index]
        for component in range(3):
            field[component] += factor * displacement[component] * inverse_cubed
            for coordinate in range(3):
                gradient[component, coordinate] += factor * inverse_cubed * (
                    (1.0 if component == coordinate else 0.0)
                    - 3 * displacement[component] * displacement[coordinate] / distance_squared
                )
    return field, gradient


@njit(cache=True)
def _integrate_winding(point, nodes, elements):
    field = np.zeros(3)
    gradient = np.zeros((3, 3))
    for element_index in range(len(elements)):
        displacement = point - nodes[element_index]
        element = elements[element_index]
        distance_squared = np.dot(displacement, displacement)
        if distance_squared < 1e-20:
            raise ValueError("A field query intersects a current quadrature node")
        inverse_cubed = distance_squared**-1.5
        cross_product = np.array([
            element[1] * displacement[2] - element[2] * displacement[1],
            element[2] * displacement[0] - element[0] * displacement[2],
            element[0] * displacement[1] - element[1] * displacement[0],
        ])
        cross_derivative = np.array([
            [0.0, -element[2], element[1]],
            [element[2], 0.0, -element[0]],
            [-element[1], element[0], 0.0],
        ])
        for component in range(3):
            field[component] += 1e-7 * cross_product[component] * inverse_cubed
            for coordinate in range(3):
                gradient[component, coordinate] += 1e-7 * inverse_cubed * (
                    cross_derivative[component, coordinate]
                    - 3 * cross_product[component] * displacement[coordinate] / distance_squared
                )
    return field, gradient


def cylinder_field_gradient(point_m, design):
    nodes, weights = _core_nodes(design)
    return _integrate_core(np.asarray(point_m, dtype=float), nodes, weights)


def winding_field_gradient(point_m, design):
    nodes, elements = _winding_nodes(design)
    return _integrate_winding(np.asarray(point_m, dtype=float), nodes, elements)


def layout(scale):
    indices = np.arange(8)
    azimuth = indices * np.pi / 4
    polar = np.deg2rad(np.where(indices % 2, 60.0, 35.0))
    radii = np.where(indices % 2, 0.13, 0.16) * scale
    radial = np.column_stack((
        np.sin(polar) * np.cos(azimuth), np.sin(polar) * np.sin(azimuth), np.cos(polar)
    ))
    return radii[:, None] * radial, -radial


def _basis(axis):
    helper = np.array([0.0, 0.0, 1.0])
    if abs(axis @ helper) > 0.95:
        helper = np.array([1.0, 0.0, 0.0])
    first = np.cross(helper, axis)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    return np.column_stack((first, second, axis))


@njit(cache=True)
def _array_components(point, core_nodes, core_weights, winding_nodes, winding_elements):
    core_fields = np.empty((3, 8))
    winding_fields = np.empty((3, 8))
    core_gradients = np.empty((8, 3, 3))
    winding_gradients = np.empty((8, 3, 3))
    for source_index in range(8):
        core_fields[:, source_index], core_gradients[source_index] = _integrate_core(
            point, core_nodes[source_index], core_weights
        )
        winding_fields[:, source_index], winding_gradients[source_index] = _integrate_winding(
            point, winding_nodes[source_index], winding_elements[source_index]
        )
    return core_fields, core_gradients, winding_fields, winding_gradients


class FiniteArray:
    def __init__(self, design=None, layout_scale=0.8, quadrature_multiplier=1):
        self.design = design or CoilDesign()
        self.layout_scale = float(layout_scale)
        self.offsets, self.axes = layout(self.layout_scale)
        core_nodes, self.core_weights = _core_nodes(
            self.design, angular_count=32 * quadrature_multiplier,
            radial_count=5 * quadrature_multiplier,
        )
        winding_nodes, elements = _winding_nodes(
            self.design, angular_count=32 * quadrature_multiplier,
            radial_count=3 * quadrature_multiplier,
            axial_count=8 * quadrature_multiplier,
        )
        bases = np.array([_basis(axis) for axis in self.axes])
        self.core_nodes = np.array([
            core_nodes @ basis.T + offset for basis, offset in zip(bases, self.offsets)
        ])
        self.winding_nodes = np.array([
            winding_nodes @ basis.T + offset for basis, offset in zip(bases, self.offsets)
        ])
        self.winding_elements = np.array([elements @ basis.T for basis in bases])
        self.coupling_matrix, winding_h = self._coupling()
        inverse_susceptibility = 1 / (self.design.relative_permeability - 1)
        self.magnetic_system_matrix = (
            np.eye(8) * (self.design.demagnetizing_factor + inverse_susceptibility)
            - self.coupling_matrix
        )
        self.magnetization_matrix = np.linalg.solve(self.magnetic_system_matrix, winding_h)
        self.susceptibility_condition_number = float(np.linalg.cond(self.magnetic_system_matrix))

    def _coupling(self):
        coupling = np.zeros((8, 8))
        winding_h = np.empty((8, 8))
        for receiver_index, point in enumerate(self.offsets):
            for source_index in range(8):
                winding_field, _ = _integrate_winding(
                    point, self.winding_nodes[source_index], self.winding_elements[source_index]
                )
                winding_h[receiver_index, source_index] = (
                    winding_field @ self.axes[receiver_index] / MU0
                )
                if receiver_index != source_index:
                    core_field, _ = _integrate_core(
                        point, self.core_nodes[source_index], self.core_weights
                    )
                    coupling[receiver_index, source_index] = (
                        core_field @ self.axes[receiver_index] / MU0
                    )
        return coupling, winding_h

    def responses(self, point_from_anchor_m):
        core_fields, core_gradients, winding_fields, winding_gradients = _array_components(
            np.asarray(point_from_anchor_m, dtype=float), self.core_nodes, self.core_weights,
            self.winding_nodes, self.winding_elements,
        )
        fields = winding_fields + core_fields @ self.magnetization_matrix
        gradients = winding_gradients + np.einsum(
            "kij,kl->lij", core_gradients, self.magnetization_matrix
        )
        return fields, gradients

    def magnetization(self, currents_a):
        return self.magnetization_matrix @ np.asarray(currents_a, dtype=float)

    def validate_currents(self, currents_a):
        currents = np.asarray(currents_a, dtype=float)
        if np.max(np.abs(currents)) > self.design.current_limit_a + 1e-7:
            raise ValueError("Current bounds exceeded")
        saturation_fraction = np.max(np.abs(self.magnetization(currents))) / (
            self.design.saturation_magnetization_a_m
        )
        if saturation_fraction > 1:
            raise ValueError("The linear core approximation exceeds the assumed saturation limit")
        return float(saturation_fraction)
