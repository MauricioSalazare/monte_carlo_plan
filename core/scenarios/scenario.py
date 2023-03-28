import numpy as np
from itertools import product

def nearest_energy_level(energy_values_function,  energy_values_dataset):
    """Find the closest energy value of the real dataset, to each element of energy_values_function"""

    nearest_energy_level = []
    # Find the closest discrete energy level in the dataset, that corresponds to the lineal energy growth.
    for energy_level in energy_values_function:
        nearest_energy_level.append(energy_values_dataset[np.argmin(np.abs(energy_values_dataset - energy_level))])
    nearest_energy_level = np.array(nearest_energy_level)

    return nearest_energy_level


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
                 load_growth_function=None,
                 pv_growth_function=None):
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
            load_growth_function: any: It is the load growth function, possibly non lineal. This function must have domain
                    and image of [0,1]. Also, it must comply f(0) = 0 and f(1) = 1.
                    The  function represents the increase in percentage and its normalized output in energy growth.
                    This function could be a lambda function or a spline from scipy.
                    e.g. f = lambda x: x **2
                         f = lambda x: np.sqrt(x)
                         f = UnivariateSpline(x, y)

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
        self.nodes_per_cluster =  self.grid_info["cluster"].value_counts().sort_index()

        self.percentages_load_growth = np.linspace(0, 1.0, self.n_levels_load_growth + 1).round(2)
        self.percentages_pv_growth = np.linspace(0, 1.0, self.n_levels_pv_growth + 1).round(2)
        self.min_max_annual_energy_per_cluster = None

        self.mapper_load_growth = self.create_mapper_load_growth(lineal_growth=lineal_growth,
                                                                 f=load_growth_function)
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

        # Only get the valid mixture combination (all values must sum to one)
        mixture_combinations = [mixture for mixture in product(percentages_mixtures,
                                                               repeat=self.n_clusters) if np.isclose(1.0, sum(mixture))]

        self.mixture_combinations = mixture_combinations
        cases = list(product(mixture_combinations, self.percentages_load_growth, self.percentages_pv_growth))

        self.n_cases = len(cases)

        print(f"Total cases: {len(cases)}")

        return cases

    def _get_min_max_energy_per_cluster(self, lower_quantile=0.1, upper_quantile=0.9):
        """
        Compute the specified lower and upper quantiles over the distribution of the annual energy consumption
        per cluster.

        The energy values are bounded, because the copula models have problems to simulate data in the extreme
        cases. Copula models do not extrapolate data to unseen values.
        The bounding is between the 10%-90% percentiles.
        """


        min_max_energy_per_cluster = {}

        for k_cluster in self.cluster_labels:
            energy_values = self.copula_load[k_cluster]["original_data"]["avg_gwh"]. \
                value_counts().sort_index().index.to_numpy()

            """
            The energy values are bounded, because the copula models have problems to simulate data in the extreme
            cases. Copula models do not extrapolate data to unseen values.
            The bounding is between the 10%-90% percentiles.
            """

            lower_bound_energy = np.nanquantile(energy_values, q=lower_quantile)  # Min. possible annual energy
            upper_bound_energy = np.nanquantile(energy_values, q=upper_quantile)  # Max. possible annual energy

            # Find the closes point in the dataset to the computed lower and upper values.
            lower_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - lower_bound_energy))]
            upper_bound_energy_discrete = energy_values[np.argmin(np.abs(energy_values - upper_bound_energy))]

            cluster_energy_data = {k_cluster: {"min": lower_bound_energy_discrete,
                                               "max": upper_bound_energy_discrete,
                                               "annual_energy_gwh": energy_values}}

            min_max_energy_per_cluster.update(cluster_energy_data)

        self.min_max_annual_energy_per_cluster = min_max_energy_per_cluster


        return min_max_energy_per_cluster



    def create_mapper_load_growth(self, lineal_growth: bool=True, f=None) -> dict:
        """
        Computes a dictionary that has the mapping of discrete load growth vs energy value that follows a lineal or
        non-lineal function.
        The  output energy value of the dictionary is the closes energy value of the dataset to the function, this is
        done to avoid numerical instability
        e.g., mapper_cluster_load_growth[LOAD_GROWTH_STEP] -> Annual energy value in GWh/year
        where LOAD_GROWTH_STEP is a float between [0, 1.0]

        Parameters:
        -----------
            lineal_growth: bool: Flag, indicates if the function is lineal or not. Default is True.
            f: any: It is the load growth function, possibly non lineal. This function must have domain and
                    image of [0,1]. Also, it must comply f(0) = 0 and f(1) = 1.
                    The  function represents the increase in percentage and its normalized output in energy growth.
                    This function could be a lambda function or a spline from scipy.
                    e.g. f = lambda x: x **2
                         f = lambda x: np.sqrt(x)
                         f = UnivariateSpline(x, y)

        """

        if not lineal_growth and f is not None:
            assert np.allclose(f(0), 0, atol=0.01), "The non-lineal function should map closed to f(0) == 0.0"
            assert np.allclose(f(1), 1, rtol=0.01), "The non-lineal function should map closed to f(1) == 1.0"


        perc_load_growth = self.percentages_load_growth

        mapper_cluster_load_growth = {}

        """
        The energy values are bounded, because the copula models have problems to simulate data in the extreme
        cases. Copula models do not extrapolate data to unseen values.
        The bounding is between the 10%-90% percentiles.
        """

        min_max_energy_per_cluster = self._get_min_max_energy_per_cluster(lower_quantile=0.1,
                                                                          upper_quantile=0.9)

        for k_cluster in self.cluster_labels:
            lower_bound_energy_discrete = min_max_energy_per_cluster[k_cluster]["min"]
            upper_bound_energy_discrete = min_max_energy_per_cluster[k_cluster]["max"]
            energy_values = min_max_energy_per_cluster[k_cluster]["annual_energy_gwh"]

            if lineal_growth:
                # Linear function:
                energy_values_function = np.linspace(lower_bound_energy_discrete, upper_bound_energy_discrete,
                                                     self.n_levels_load_growth + 1)
            else:
                # Non-linear function:
                energy_values_function = (
                            f(perc_load_growth) * (upper_bound_energy_discrete - lower_bound_energy_discrete) +
                            lower_bound_energy_discrete
                                          )

            energy_level_discrete = nearest_energy_level(energy_values_function, energy_values)

            ## Helper figure to see the result of the non-linear function
            # fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            # plt.subplots_adjust(left=0.15, right=0.95)
            # ax.plot(perc_load_growth, energy_values_function, "-.", label="Ideal curve growth", color="C1")
            # ax.plot(perc_load_growth, energy_level_discrete, "--x", label="Nearest energy value to the dataset",
            #         color="C2")
            # ax.scatter(np.zeros(len(energy_values)), energy_values, s=2, label="Energy values (Real dataset)",
            #            color="C0")
            # ax.legend(fontsize="x-small")
            # ax.set_title(f"Cluster: {k_cluster}")
            # ax.set_xlabel("Growth percentage")
            # ax.set_ylabel("Annual Energy consumption [GWh/year]")

            mapper_load_growth = dict(zip(perc_load_growth, energy_level_discrete))
            mapper_cluster_load_growth[k_cluster] = mapper_load_growth

        return mapper_cluster_load_growth

    def create_load_scenarios_cluster(self,
                                      k_cluster,
                                      n_scenarios,
                                      load_growth):
        """
        Sample the copula model of the active power load profile, for one cluster.

        Right now the reactive power is hard coded to have a a power factor of 0.995 (cos(\phi))
        Meaning: tan(\phi) = 0.1 -> \phi approx 5.7105° cos(\phi) = 0.995 : Here P * tan(\phi) = Q

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


        # Assign the generated scenarios to the correct node according to the cluster number.
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


    @staticmethod
    def flip_matrix(power_matrix):
        """
        Flip the 3D matrix, that must change the dimensions
         (nodes, scenarios, time_steps) into (scenarios, time_steps, nodes)
        This makes the matrix compatible with tensorpowerflow.
        """

        (nodes_, scenarios_, time_steps_) = power_matrix.shape

        power_stack_flip = []
        for scenario_ in range(scenarios_):
            power_stack_flip.append(power_matrix[:, scenario_, :].T)

        return np.stack(power_stack_flip)


    def sample_scenarios(self,
                         case,
                         n_scenarios):
        """
        Create the net active power and reactive power matrices for a required case:

        Parameters:
        -----------
            case: tuple(tuple(float, float, float), float, float), describes:
                       ((pi_0, pi_1, pi_2), load_growth, pv_growth)
                       ((cloudy, sunny, dark), ..., ...)
                where:
                    pi_j: j = 1,..,3, Correspond to the mixture of (cloudy, sunny, dark) days and all pi sums to 1.
                    load_growth: Number between 0,..1, to describe how much the percentage of load increased in the nodes
                    pv_growth:  Number between 0,..1, to describe how much the percentage of the PV increased in the nodes

            n_scenarios: int: number of scenarios to be simulated PER CASE.

        Returns:
        --------
            active_power_stack_net: np.ndarray: array with dimension (nodes, n_scenarios, time_steps).
                Be aware that the script is hard coded to have 48 time steps. i.e., 30-min resolution consumption
            reactive_power_stack: np.ndarray: array with the same dimensions as active_power_stack_net.
                Be aware that the script is currently hard coded to have a constant power factor.
        """

        mixture_prob, load_growth, pv_growth = case

        active_power_stack, reactive_power_stack = self.create_load_scenarios(n_scenarios=n_scenarios,
                                                                              load_growth=load_growth)

        """
        The shape of the active power matrix.
        active_power_stack => (nodes, scenarios, time_steps)
        """

        # Adjust for PV active power generation
        irradiance_scenarios = self.mixture_model_irradiance.sample_mixture(n_samples=n_scenarios, prob=mixture_prob)

        assert irradiance_scenarios.shape == active_power_stack[0, ...].shape
        assert irradiance_scenarios.shape == reactive_power_stack[0, ...].shape


        # TODO: Change the PV load growth to be a function per node and not linealy proportional.
        irr_matrix = irradiance_scenarios.divide(1000).values  # Convert irradiance from W/m^2 to to kW/m^2
        kwp_nodes = self.grid_info["kwp"].multiply(1 + pv_growth).values

        pv_generation_kw = np.array([irr_matrix * kwp_node for kwp_node in kwp_nodes])
        active_power_stack_net = active_power_stack - pv_generation_kw

        self._last_active_power_stack = active_power_stack
        self._last_reactive_power_stack = reactive_power_stack
        self._last_pv_generation_kw = pv_generation_kw

        return active_power_stack_net, reactive_power_stack

    def create_case_scenarios(self,
                              case,
                              n_scenarios):
        """
        Create a dictionary with the specific required case:

        An overview of the nested methods called by this function:

        create_case_scenarios ->     samples_scenarios      -> create_load_scenarios -> create_load_scenarios_cluster
           (PQ, PV matrices)  ->  (PQ matrix all clusters   ->     (PQ all clusters) ->      (PQ one cluster)
                                    and Irradiance profile,
                                    which transformed to PV)
        Parameters:
        -----------
            case: tuple(tuple(float, float, float), float, float), describes:
                       ((pi_0, pi_1, pi_2), load_growth, pv_growth)
                       ((cloudy, sunny, dark), ..., ...)
                where:
                    pi_j: j = 1,..,3, Correspond to the mixture of (cloudy, sunny, dark) days and all pi sums to 1.
                    load_growth: Number between 0,..1, to describe how much the percentage of load increased in the nodes
                    pv_growth:  Number between 0,..1, to describe how much the percentage of the PV increased in the nodes

            n_scenarios: int: number of scenarios to be simulated PER CASE.

        Returns:
        --------
            case_dictionary: dict: Output of the sample of the probabilistic distribution from the copula models.

                case_dictionary[(scenario_#, time_step)] -> has another dictionary with the following structure:
                        data = {"P": np.array(with active power consumption per node),
                                "Q": np.array(with reactive power consumption per node)}


                Example:
                    To access the reactive power value of the node 13, scenario 10, time step 23, you use:
                        active_power_value = case_dictionary[(10, 23)]["Q"][13]
                    Meaning:
                        case_dictionary[(scenario_number, time_step)][type_power][node]

                It must be noted that the nodes power where sample with respect to the cluster assigned to the node
                from the file of the grid, which can be find in the parameter self.grid_info, "cluster" column.

            TODO: The class is hardcoded to have 48 time steps, or 30 min resolution of a day profile.
        """
        active_power_stack_net, reactive_power_stack = self.sample_scenarios(case=case, n_scenarios=n_scenarios)

        case_dictionary = {}
        for scenario in range(n_scenarios):
            for time_step in range(48):  # Hard coded to 30 min resolution.
                case_dictionary[(scenario, time_step)] = {"P": active_power_stack_net[:, scenario, time_step],
                                                          "Q": reactive_power_stack[:, scenario, time_step]}

        return case_dictionary

    def create_case_scenarios_tensorpoweflow(self,
                                             case,
                                             n_scenarios):
        """
        Same method as create_case_scenarios, but instead of returning a dictionary, it returns the active and
        reactive power arrays, flipped with the dimensions necessary for the tensorpowerflow algorithm to run.

        Return:
        -------
            active_power_stack_net_flipped: np.ndarray: Net active power with dimensions (scenarios, time_steps, nodes)
            reactive_power_stack_flipped: np.ndarray: Reactive power with dimensions (scenarios, time_steps, nodes)

        """
        active_power_stack_net, reactive_power_stack = self.sample_scenarios(case=case, n_scenarios=n_scenarios)
        active_power_stack_net_flipped = self.flip_matrix(active_power_stack_net)
        reactive_power_stack_flipped = self.flip_matrix(reactive_power_stack)

        return active_power_stack_net_flipped, reactive_power_stack_flipped