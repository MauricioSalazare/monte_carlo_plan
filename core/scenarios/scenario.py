import numpy as np
from itertools import product


class ScenarioGenerator:
    """
    Assemble the copulas from load consumption and PV generation in one place.

    Growth functions:
    ----------------
    The load and PV growth are curves (functions) that has support from 0 to 1, representing an horizon of medium term
    planning, for instance, in a period of 11 years, 0 corresponds to 0 years and 1 correspond to 11 years.

    f() -> Growth function and support: supp(f) -> [0,1]
    y = f() -> where y belongs to [0,1], meaning range or image is: image(f) -> [0,1]

    The default assumption of the model is a linear growth.

    Copula functions:
    ----------------
    The


    Arguments:
    ----------
        copula_models:
        mixture_copulas_irradiance:

        n_levels_load_growth: int: Resolution of load growth i.e., growth_res = 1.0 / n_levels_load_growth
        n_levels_pv_growth: int: Resolution of pv growth i.e., pv_res = 1.0 / n_levels_load_growth
        n_levels_mixtures: int: Define the grid resolution i.e., grid_res = 1.0 / n_levels_mixtures, which is required
            for the mixture model. The combination of the mixtures must sum to 1.0.
        grid_info: pd.DataFrame: Grid parameters (lines)


    Methods:
    --------
        create_scenarios_case(): Returns a dictionary with active and reactive power for the grid

    """
    def __init__(self,
                 copula_load: dict,
                 mixture_model_irradiance,
                 grid_info,
                 n_levels_load_growth: int = 5,
                 n_levels_pv_growth: int = 5,
                 n_levels_mixtures: int = 5,
                 lineal_growth=True,
                 load_growth=None,
                 pv_growth=None):
        """
        Parameters:
        -----------
            copula_load: dict of dict: with the form copula_load[cluster_number] = {"copula": copula model,
                                                                                    "original_data": pd.DataFrame,
                                                                                    "idx_irradiance": np.array(bool)}
            mixture_model_irradiance: is an instance of MixtureCopulaIrradiance class.
                    Basically, is a group of copula models grouped together, but the class has the versatility to
                    sample the mixture model easily.
            grid_info: pd.DataFrame: pandas data frame with the data of the grid, without the slack node. Normally,
                    this dataframe is used to get the base PV installed capacity and the cluster number of the node.
                    It is not used to to PF simulations in this class.

        """


        self.copula_load = copula_load
        self.mixture_model_irradiance = mixture_model_irradiance
        self.grid_info = grid_info

        self.n_levels_load_growth = n_levels_load_growth
        self.n_levels_pv_growth = n_levels_pv_growth
        self.n_levels_mixtures = n_levels_mixtures
        self.mixture_combinations = None
        self.n_cases = None

        self.n_clusters = len(self.copula_load.keys())  #TODO: This should be the number of copula irradiance models not load!!
        self.cluster_labels = list(range(len(self.copula_load.keys())))

        self.percentages_load_growth = np.linspace(0, 1.0, self.n_levels_load_growth + 1).round(2)
        self.percentages_pv_growth = np.linspace(0, 1.0, self.n_levels_pv_growth + 1).round(2)

        self.mapper_load_growth = self.create_mapper_load_growth()
        self.cases_combinations = self.create_cases()

    def create_cases(self):
        """
        Compute all possible combinations of cases

        A case consist of: (mixture components of irradiance, % load growth, % pv_growth)

        mixture components of irradiance = (cloudy, sunny, dark)

        Where mixture components is a tuple of "n" number that sum up to 1. Each number correspond to the probability
        for each type of irradiance to sample. e.g., (0.2, 0.7, 0.1)

        % load growth is a number between [0, 1], that corresponds how the load will grow in the grid.

        % load growth is a number between [0, 1], that corresponds how the PV installed capacity will grow in the grid.

        """

        percentages_mixtures = np.linspace(0, 1.0, self.n_levels_mixtures + 1).round(1)
        mixture_combinations = [mixture for mixture in product(percentages_mixtures,
                                                               repeat=self.n_clusters) if np.isclose(1.0, sum(mixture))]

        self.mixture_combinations = mixture_combinations
        cases = list(product(mixture_combinations, self.percentages_load_growth, self.percentages_pv_growth))

        self.n_cases = len(cases)

        print(f"Total cases: {len(cases)}")

        return cases

    def create_mapper_load_growth(self) -> dict:
        """
        Computes a dictionary that has a discrete load growth to avoid numerical instability
        e.g., mapper_cluster_load_growth[LOAD_GROWTH_STEP] -> Annual energy value in GWh/year
        where LOAD_GROWTH_STEP is a float between [0, 1.0]
        """

        cluster_labels = list(range(len(self.copula_load.keys())))
        mapper_cluster_load_growth = {}

        for k_cluster in cluster_labels:
            energy_values = self.copula_load[k_cluster]["original_data"]["avg_gwh"].\
                                                                    value_counts().sort_index().index.to_numpy()
            lower_bound_energy = np.nanquantile(energy_values, q=0.1)  # Min. possible annual energy
            upper_bound_energy = np.nanquantile(energy_values, q=0.9)  # Max. possible annual energy

            lower_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - lower_bound_energy))]
            upper_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - upper_bound_energy))]

            energy_levels = np.linspace(lower_bound_energy_discrete, upper_bound_energy_discrete,
                                        self.n_levels_load_growth + 1)

            energy_level_discrete = []

            # Find the closest discrete energy level in the dataset, that corresponds to the lineal energy growth.
            for energy_level in energy_levels:
                energy_level_discrete.append(energy_values[np.argmin(np.abs(energy_values - energy_level))])
            energy_level_discrete = np.array(energy_level_discrete)

            mapper_load_growth = dict(zip(self.percentages_load_growth, energy_level_discrete))

            mapper_cluster_load_growth[k_cluster] = mapper_load_growth

        return mapper_cluster_load_growth

    def create_load_scenarios_cluster(self,
                                      k_cluster,
                                      n_scenarios,
                                      load_growth):
        """
        Sample the copula model of the active power load profile, for one cluster.

        Parameter:
        =========
            k_cluster: int: cluster number to select the copula
            n_scenarios: int: number of profiles that should be sampled from the copula model
            load_growth: float: Value of annual energy in GWh/year that the copula must be conditioned.

        Returns:
        =======
            ap_dict_matrix_cluster: dict: sampled active load profiles organized by node.
                Key of the dictionary is the node number
            rp_dict_matrix_cluster: dict: sampled reactive load profiles organized by node.
                Key of the dictionary is the node number
        """

        # Mapper to get the variable name right from the copula.
        variable_names = self.copula_load[k_cluster]["original_data"].columns.to_list()
        n_variables = len(variable_names)
        mapper_variable_names = dict(zip(variable_names, ["x" + str(ii) for ii in range(1, n_variables + 1)]))
        variable_names.remove("avg_gwh")

        # TODO: Resolve if the source of reactive power is the copula

        idx_column_active_power = [variable_.endswith("_ap") for variable_ in variable_names]
        # idx_column_reactive_power = [variable_.endswith("_rp") for variable_ in variable_names]

        nodes_per_cluster = self.grid_info["cluster"].value_counts().sort_index()
        nodes_in_cluster = nodes_per_cluster[k_cluster]

        n_samples = nodes_in_cluster * n_scenarios

        cond_variable_dict = {mapper_variable_names["avg_gwh"]: self.mapper_load_growth[k_cluster][load_growth]}
        sampled_copula_ = self.copula_load[k_cluster]["copula"].sample(n_samples=n_samples,
                                                                       conditional=True,
                                                                       variables=cond_variable_dict,
                                                                       drop_inf=True)
        sampled_copula = sampled_copula_.T
        active_power_sampled_copula = sampled_copula[:, idx_column_active_power]

        assert active_power_sampled_copula.shape[1] == 48, "It is not 30 min resolution"

        ap_sampled_per_node = active_power_sampled_copula.reshape(nodes_in_cluster,
                                                                  n_scenarios,
                                                                  active_power_sampled_copula.shape[1])

        assert np.allclose(ap_sampled_per_node[1, 0, :],
                           active_power_sampled_copula[n_scenarios, :]), "Check the reshaping"

        nodes_cluster = self.grid_info["NODES"][self.grid_info["cluster"] == k_cluster].values

        ap_dict_matrix_cluster = {}
        rp_dict_matrix_cluster = {}

        for ii, key_ in enumerate(nodes_cluster):
            ap_dict_matrix_cluster[key_] = ap_sampled_per_node[ii, ...]

            # tan(\phi) = 0.1 -> \phi approx 5.7105° cos(\phi) = 0.995 : Here P * tan(\phi) = Q
            rp_dict_matrix_cluster[key_] = ap_sampled_per_node[ii, ...] * 0.1

        return ap_dict_matrix_cluster, rp_dict_matrix_cluster

    def create_load_scenarios(self,
                              n_scenarios,
                              load_growth):

        active_power = {}
        reactive_power = {}

        for k_cluster_ in self.cluster_labels:
            ap_dict_sampled, rp_dict_sampled = self.create_load_scenarios_cluster(k_cluster=k_cluster_,
                                                                                  n_scenarios=n_scenarios,
                                                                                  load_growth=load_growth)
            active_power.update(ap_dict_sampled)
            reactive_power.update(rp_dict_sampled)

        # Build Tensor
        active_power_stack = np.array([active_power[node_grid] for node_grid in self.grid_info["NODES"].values])
        reactive_power_stack = np.array([reactive_power[node_grid] for node_grid in self.grid_info["NODES"].values])

        return active_power_stack, reactive_power_stack

    def create_case_scenarios(self,
                              case,
                              n_scenarios):
        """
        Create a dictionary with the specific required case:

        Parameters:
        -----------
            case: tuple(tuple(float, float, float), float, float), describes:
                       ((pi_0, pi_1, pi_2), load_growth, pv_growth)
                where:
                    pi_j: j = 1,..,3, Correspond to the mixture of (cloudy, sunny, dark) days and all pi sums to 1.
                    load_growth: Number between 0,..1, to describe how much the percentage of load increased in the nodes
                    pv_growth:  Number between 0,..1, to describe how much the percentage of the PV increased in the nodes

            n_scenarios: int: number of scenarios to be simulated PER CASE.


        Returns:
        --------
            case_dictionary: dict: Output of the sample of the probabilistic distribution from the copula models.

                case_dictionary[(case_number, time_step)] -> has another dictionary with the following structure:
                        data = {"P": np.array(with active power consumption per node),
                                "Q": np.array(with reactive power consumption per node)}


                Example:
                    To access the reactive power value of the node 13, case number 10, time step 23, you use:
                        active_power_value = case_dictionary[(10, 23)]["Q"][13]
                    Meaning:
                        case_dictionary[(case_number, time_step)][type_power][node]

                It must be noted that the nodes power where sample with respect to the cluster assigned to the node
                from the file of the grid, which can be find in the parameter self.grid_info, "cluster" column.

            TODO: The class is hardcoded to have 48 time steps, or 30 min resolution of a day profile.


        """



        mixture_prob, load_growth, pv_growth = case

        active_power_stack, reactive_power_stack = self.create_load_scenarios(n_scenarios=n_scenarios,
                                                                              load_growth=load_growth)

        # Adjust for PV active power generation
        irradiance_scenarios = self.mixture_model_irradiance.sample_mixture(n_samples=n_scenarios, prob=mixture_prob)

        assert irradiance_scenarios.shape == active_power_stack[0, ...].shape
        assert irradiance_scenarios.shape == reactive_power_stack[0, ...].shape

        irr_matrix = irradiance_scenarios.divide(1000).values  # Convert irradiance to kW/m^2
        kwp_nodes = self.grid_info["kwp"].multiply(1 + pv_growth).values

        pv_generation_kw = np.array([irr_matrix * kwp_node for kwp_node in kwp_nodes])
        active_power_stack_net = active_power_stack - pv_generation_kw

        case_dictionary = {}
        for scenario in range(n_scenarios):
            for time_step in range(48):
                case_dictionary[(scenario, time_step)] = {"P": active_power_stack_net[:, scenario, time_step],
                                                          "Q": reactive_power_stack[:, scenario, time_step]}

        self._last_active_power_stack = active_power_stack
        self._last_reactive_power_stack = reactive_power_stack
        self._last_pv_generation_kw = pv_generation_kw

        return case_dictionary