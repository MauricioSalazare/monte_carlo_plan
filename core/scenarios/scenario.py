import numpy as np
from itertools import product


class ScenarioGenerator:
    """
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
                 copula_load,
                 mixture_model_irradiance,
                 grid_info,
                 n_levels_load_growth: int = 5,
                 n_levels_pv_growth: int = 5,
                 n_levels_mixtures: int = 5):

        self.copula_load = copula_load
        self.mixture_model_irradiance = mixture_model_irradiance
        self.grid_info = grid_info

        self.n_levels_load_growth = n_levels_load_growth
        self.n_levels_pv_growth = n_levels_pv_growth
        self.n_levels_mixtures = n_levels_mixtures
        self.mixture_combinations = None
        self.n_cases = None

        self.n_clusters = len(self.copula_load.keys())
        self.cluster_labels = list(range(len(self.copula_load.keys())))

        self.percentages_load_growth = np.linspace(0, 1.0, self.n_levels_load_growth + 1).round(2)
        self.percentages_pv_growth = np.linspace(0, 1.0, self.n_levels_pv_growth + 1).round(2)

        self.mapper_load_growth = self.create_mapper_load_growth()
        self.cases_combinations = self.create_cases()

    def create_cases(self):
        """
        Compute all possible combinations of cases

        A case consist of: (mixture components of irradiance, % load growth, % pv_growth)

        Where mixture components is a tuple of "n" number that sum up to 1. Each number correspond to the probability
        for each type of irradiance to sample. e.g., (0.2, 0.7, 0.1)

        % load growth is a number between [0, 1], that corresponds how the load will grow in the grid.

        % load growth is a number between [0, 1], that corresponds how the PV installed capacity will grow in the grid.

        """


        percentages_mixtures = np.linspace(0, 1.0, self.n_levels_mixtures + 1).round(1)
        mixture_combinations = [mixture for mixture in product(percentages_mixtures,
                                                               repeat=self.n_clusters) if sum(mixture) == 1]

        self.mixture_combinations = mixture_combinations
        cases = list(product(mixture_combinations, self.percentages_load_growth, self.percentages_pv_growth))

        self.n_cases = len(cases)

        print(f"Total cases: {len(cases)}")

        return cases

    def create_mapper_load_growth(self):
        # Discrete load growth to avoid numerical instability
        cluster_labels = list(range(len(self.copula_load.keys())))
        mapper_cluster_load_growth = {}

        for k_cluster in cluster_labels:
            energy_values = self.copula_load[k_cluster]["original_data"]["avg_gwh"].\
                                                                    value_counts().sort_index().index.to_numpy()
            lower_bound_energy = np.nanquantile(energy_values, q=0.1)
            upper_bound_energy = np.nanquantile(energy_values, q=0.9)

            lower_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - lower_bound_energy))]
            upper_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - upper_bound_energy))]

            energy_levels = np.linspace(lower_bound_energy_discrete, upper_bound_energy_discrete,
                                        self.n_levels_load_growth + 1)

            energy_level_discrete = []
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

        return case_dictionary