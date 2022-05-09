import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, bmat, csc_matrix, vstack
from scipy.sparse.linalg import spsolve, inv
from time import perf_counter


class Grid:
    def __init__(self,
                 node_file_path: str,
                 lines_file_path: str,
                 *,
                 s_base: int = 1000,  # kVA - 1 phase
                 v_base: float = 11  # kV - 1 phase
                 ):
        self.node_file_path = node_file_path
        self.lines_file_path = lines_file_path
        self.s_base = s_base
        self.v_base = v_base
        self.z_base = (self.v_base ** 2 * 1000) / self.s_base
        self.i_base = self.s_base / (np.sqrt(3) * self.v_base)

        self._load_system_data()
        self._make_y_bus()
        self._compute_alphas()

        self.v_0 = np.ones((self.nb - 1)) + 1j * np.zeros((self.nb - 1))  # Flat start

    def _load_system_data(self) -> None:
        """
        Load .csv files which has the node information e.g., type of load, PV installed capacity, etc. and Lines
        data e.g., resistance, reactance, capacitance and buses connected.
        """

        self.branch_info = pd.read_csv(self.lines_file_path)
        self.bus_info = pd.read_csv(self.node_file_path)

        # TODO: Assert that the columns exists and everything hast the proper dimensions

        self.P_file = self.bus_info[self.bus_info.Tb != 1].PD.values  # Vector with all active power except slack
        self.Q_file = self.bus_info[self.bus_info.Tb != 1].QD.values   # Vector with all reactive power except slack

    def _make_y_bus(self) -> None:
        """
        Compute Y_bus submatrices

        For each branch, compute the elements of the branch admittance matrix where
              | Is |   | Yss  Ysd |   | Vs |
              |    | = |          | * |    |
              |-Id |   | Yds  Ydd |   | Vd |


        """

        self.nb = self.bus_info.shape[0]  # number of buses
        self.nl = self.branch_info.shape[0]  # number of lines

        sl = self.bus_info[self.bus_info['Tb'] == 1]['NODES'].tolist()  # Slack node(s)

        stat = self.branch_info.iloc[:, 5]  # ones at in-service branches
        Ys = stat / ((self.branch_info.iloc[:, 2] + 1j * self.branch_info.iloc[:, 3]) / (
                    self.v_base ** 2 * 1000 / self.s_base))  # series admittance
        Bc = stat * self.branch_info.iloc[:, 4] * (self.v_base ** 2 * 1000 / self.s_base)  # line charging susceptance
        tap = stat * self.branch_info.iloc[:, 6]  # default tap ratio = 1

        Ytt = Ys + 1j * Bc / 2
        Yff = Ytt / tap
        Yft = - Ys / tap
        Ytf = Yft

        # build connection matrices
        f = self.branch_info.iloc[:, 0] - 1  # list of "from" buses
        t = self.branch_info.iloc[:, 1] - 1  # list of "to" buses

        # connection matrix for line & from buses
        Cf = csr_matrix((np.ones(self.nl), (range(self.nl), f)), (self.nl, self.nb))

        # connection matrix for line & to buses
        Ct = csr_matrix((np.ones(self.nl), (range(self.nl), t)), (self.nl, self.nb))

        # build Yf and Yt such that Yf * V is the vector of complex branch currents injected
        # at each branch's "from" bus, and Yt is the same for the "to" bus end
        i = np.r_[range(self.nl), range(self.nl)]  ## double set of row indices

        Yf = csr_matrix((np.r_[Yff, Yft], (i, np.r_[f, t])))
        Yt = csr_matrix((np.r_[Ytf, Ytt], (i, np.r_[f, t])))

        # build Ybus
        Ybus = Cf.T * Yf + Ct.T * Yt  # Full Ybus

        self.Yss = csr_matrix(Ybus[sl[0] - 1, sl[0] - 1], shape=(len(sl), len(sl)))
        self.Ysd = Ybus[0, 1:]
        self.Yds = self.Ysd.T
        self.Ydd = Ybus[1:, 1:]

    def _compute_alphas(self):
        self.alpha_P = self.bus_info[self.bus_info.Tb != 1].Pct.values.reshape(-1, ).tolist()
        self.alpha_I = self.bus_info[self.bus_info.Tb != 1].Ict.values.reshape(-1, ).tolist()
        self.alpha_Z = self.bus_info[self.bus_info.Tb != 1].Zct.values.reshape(-1, ).tolist()

    def run_pf(self,
               active_power: np.ndarray = None,
               reactive_power: np.ndarray = None,
               *,
               iterations: int = 100,
               tolerance: float = 1e-6,
               flat_start: bool = True) -> np.ndarray:
        """Run power flow for the current consumption of nodes"""

        if flat_start:
            self.v_0 = np.ones((self.nb - 1)) + 1j * np.zeros((self.nb - 1))  # Flat start

        if (active_power is not None) and (reactive_power is not None):
            assert len(active_power.shape) == 1, "Array should be one dimensional."
            assert len(reactive_power.shape) == 1, "Array should be one dimensional."
            assert len(active_power) == len(reactive_power) == self.nb - 1, "All load nodes must have power values."
        else:
            active_power = self.P_file
            reactive_power = self.Q_file

        active_power_pu = active_power / self.s_base  # Vector with all active power except slack
        reactive_power_pu = reactive_power / self.s_base  # Vector with all reactive power except slack

        self.S_nom = (active_power_pu + 1j * reactive_power_pu).reshape(-1, ).tolist()
        self.B = csc_matrix(np.diag(np.multiply(self.alpha_Z, np.conj(self.S_nom))) + self.Ydd)  # Constant
        self.C = csr_matrix(self.Yds +
                            (np.multiply(self.alpha_I, np.conj(self.S_nom))).reshape(self.nb - 1, 1))  # Constant
        self.B_inv = inv(self.B)

        iteration = 0
        tol = np.inf
        start = perf_counter()
        while (iteration < iterations) & (tol >= tolerance):
            v = self._solve_sam()
            tol = max(abs(abs(v) - abs(self.v_0)))
            self.v_0 = v  # Voltage at load buses
            iteration += 1
        # print(f"Time: {perf_counter() - start} seconds.")
        # print(f"Iterations: {iteration - 1}")

        return self.v_0  # Solution of voltage in complex numbers

    def _solve_sam(self) -> np.ndarray:
        A = csr_matrix(np.diag(np.multiply(self.alpha_P, 1. / np.conj(self.v_0) ** 2) * np.conj(self.S_nom)))
        D = csr_matrix(
            (np.multiply(np.multiply(2, self.alpha_P), 1. / np.conj(self.v_0)) * np.conj(self.S_nom)).reshape(-1, 1)
        )
        V = self.B_inv @ (A @ np.conj(self.v_0[:, np.newaxis]) - self.C - D)

        return np.array(V.real + 1j * V.imag).flatten()

    def _solve_ds(self) -> np.ndarray:
        A = csr_matrix(np.diag(np.multiply(self.alpha_P, 1. / np.conj(self.v_0) ** 2) * np.conj(self.S_nom)))
        D = csr_matrix(
            (np.multiply(np.multiply(2, self.alpha_P), 1. / np.conj(self.v_0)) * np.conj(self.S_nom)).reshape(-1, 1))

        M11 = np.real(A) - np.real(self.B)
        M12 = np.imag(A) + np.imag(self.B)
        M21 = np.imag(A) - np.imag(self.B)
        M22 = -np.real(A) - np.real(self.B)
        N1 = np.real(self.C) + np.real(D)
        N2 = np.imag(self.C) + np.imag(D)
        M = csr_matrix(bmat([[M11, M12], [M21, M22]]))
        N = vstack([N1, N2])
        V = spsolve(M, N)

        return np.add(V[0:self.nb - 1], 1j * V[self.nb - 1:])

    def line_currents(self, volt_solutions):
        raise NotImplementedError
