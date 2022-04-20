from core.copula import (create_rlp_and_annual_energy_for_all_clusters,
                         create_copula_models_for_all_clusters,
                         check_copula_model,
                         check_energy_levels)
from core.copula import EllipticalCopula
import numpy as np
import pickle


np.random.seed(123456)
file_time_series=r"data/processed_data/consumption_weather/time_series_data.csv"
file_transformer_outliers = r"data/processed_data/consumption_weather/outlier_transformers.csv"
file_load_clusters = r"data/processed_data/consumption_weather/clustering_load.csv"
file_annual_energy = r"data/processed_data/consumption_weather/year_consumption_gwh.csv"

rlp_clusters = create_rlp_and_annual_energy_for_all_clusters(file_time_series,
                                                             file_transformer_outliers,
                                                             file_load_clusters,
                                                             file_annual_energy)

copula_models = create_copula_models_for_all_clusters(file_time_series,
                                                      file_transformer_outliers,
                                                      file_load_clusters,
                                                      file_annual_energy,
                                                      add_reactive_power=True)

check_copula_model(copula_models, k_cluster=0)
check_copula_model(copula_models, k_cluster=1)
check_copula_model(copula_models, k_cluster=2)

check_energy_levels(copula_models, k_cluster=2, drop_inf=False)

# file_name_load_models = "models/copula_model_load_with_reactive_power.pkl"
# with open(file_name_load_models, "wb") as pickle_file:
#     pickle.dump(copula_models, pickle_file)
