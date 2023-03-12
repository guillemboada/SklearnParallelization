''' Script to analyse the benefit of training with multi-core processing'''

import os
import time
import json
import datetime
import logging
import pathlib
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import joblib
import matplotlib.pyplot as plt


start = time.time()

# Set up logging configuration
current_datetime_string = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
logfile_name = f"experiment_{current_datetime_string}.log"
logging.basicConfig(
    level=logging.INFO,
    format="{asctime} {levelname:<8} {message}",
    style='{',
    filename=logfile_name,
    filemode='w'
)

# Read experiment configuration
with open('config.json', 'r') as f:
    config = json.load(f)

n_jobs_to_test = config["n_jobs_to_test"]
models = config["models"]
parallelization_backends = config["parallelization_backends"]
n_samples = config["n_samples"]
n_features = config["n_features"]
n_trials = config["n_trials"]
results_directory = config["results_directory"]
image_name_base = config["image_name_base"]
logging.info(f"Read experiment configuration: {config}")

# Generate dummy data
data_generation_start = time.time()
X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=2, cluster_std=100, random_state=0)
data_generation_time = time.time() - data_generation_start
logging.info(f"Created dummy dataset ({n_samples} samples, {n_features} dimensions) in {data_generation_time:.2f} seconds")

# Train with different number of logical processors (n_jobs)
n_cpu = os.cpu_count()
logging.info(f"Found {n_cpu} logical processors")
if n_jobs_to_test != -1:
    n_jobs_to_test = n_jobs_to_test
    if max(n_jobs_to_test) > n_cpu:
        raise ValueError(f"Found {n_cpu} logical processors, but intended to use {max(n_jobs_to_test)}")
else:
    n_jobs_to_test = list(range(1, n_cpu))
    logging.info(f"N_JOBS_TO_TEST = -1. All n_jobs up to one minus the number of available logical processors will be tested: {n_jobs_to_test}")

all_training_times_means = []
all_training_times_variances = []
overall_plot_labels = []
for selected_model in models:
    logging.info(f"Selected model: {selected_model}")
    for selected_parallelization_backend in parallelization_backends:
        logging.info(f"Selected parallelization backend: {selected_parallelization_backend}")
        training_times = np.zeros((len(n_jobs_to_test), n_trials))
        for i, iteration_n_jobs in enumerate(n_jobs_to_test):
            for n in range(n_trials):
                
                match selected_model:
                    case "RandomForestClassifier":
                        model = RandomForestClassifier()
                    case "KNeighborsClassifier":
                        model = KNeighborsClassifier()
                    case "LogisticRegression":
                        model = LogisticRegression()
                    case "MLPClassifier":
                        model = MLPClassifier()
                    case "SVC":
                        model = SVC() 
                    case "GaussianNB":
                        model = GaussianNB()
                    case "DecisionTreeClassifier":
                        model = DecisionTreeClassifier()
                    case "SGDClassifier":
                        model = SGDClassifier()
                    case _:
                        raise NotImplementedError(f"Model {selected_model} is not available.")
                logging.info(f"Initialized {selected_model}")

                training_start = time.time()
                with joblib.parallel_backend(selected_parallelization_backend, n_jobs=iteration_n_jobs):
                    model.fit(X, y)
                training_time = time.time() - training_start
                training_times[i, n] = training_time
                logging.info(f"Trained with n_jobs={iteration_n_jobs}: {training_time:.2f} seconds")

        # Compute mean and variance
        training_times_means = np.mean(training_times, axis=-1)
        training_times_variances = np.var(training_times, axis=-1)
        all_training_times_means.append(list(training_times_means))
        all_training_times_variances.append(list(training_times_variances))
        logging.info(f"Mean training durations with {n_trials} trials: {training_times_means}")
        logging.info(f"Variance of the training durations with {n_trials} trials: {training_times_means}")

        # Plot and save iteration results 
        plt.errorbar(n_jobs_to_test, training_times_means, training_times_variances)
        plt.xlabel("n_jobs (number of logical processors)")
        plt.ylabel("Training time (seconds)")
        plt.xticks(n_jobs_to_test)
        plt.grid()
        iteration_parameters_string = f"{selected_model}_{selected_parallelization_backend}"
        overall_plot_labels.append(iteration_parameters_string)
        image_name = f"{image_name_base}_{n_samples}x{n_features}_{iteration_parameters_string}.png"
        script_path = pathlib.Path(__file__).parent.resolve()
        results_path = os.path.join(script_path, results_directory)
        if not os.path.exists(results_directory):
            os.makedirs(results_directory)
        plt.savefig(os.path.join(results_path, image_name))
        plt.clf()
        logging.info(f"Saved iteration results into {image_name}")

# Plot and save overall results
logging.info(f"All mean training durations: {all_training_times_means}")
logging.info(f"All training durations variances: {all_training_times_variances}")
logging.info(f"All iteration labels: {overall_plot_labels}")
plt.plot(n_jobs_to_test, np.transpose(np.array(all_training_times_means)))
plt.xlabel("n_jobs (number of logical processors)")
plt.ylabel("Training time (seconds)")
plt.legend(overall_plot_labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
plt.xticks(n_jobs_to_test)
plt.grid()
image_name = f"{image_name_base}_{n_samples}x{n_features}_OverallResults.png"
script_path = pathlib.Path(__file__).parent.resolve()
plt.savefig(os.path.join(results_path, image_name), bbox_inches='tight')
plt.clf()
logging.info(f"Saved overall results into {image_name}")

total_time = time.time() - start
logging.info(f"Total wall time: {total_time:.2f} seconds")
