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

class Config():
    def __init__(self, n_jobs_to_test, models, parallelization_backends, n_samples, n_features, n_trials, results_directory, image_name_base):
        self.n_jobs_to_test = n_jobs_to_test
        self.models = models
        self.parallelization_backends = parallelization_backends
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_trials = n_trials
        self.results_directory = results_directory
        self.image_name_base = image_name_base


def create_results_directory(results_directory):
    script_path = pathlib.Path(__file__).parent.resolve()
    results_path = os.path.join(script_path, results_directory)
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    return results_path


def create_experiment_results_directory(results_path):
    current_datetime_string = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    experiment_results_directory = f"{current_datetime_string}_{config.n_samples}x{config.n_features}"
    experiment_results_path = os.path.join(results_path, experiment_results_directory)
    os.makedirs(experiment_results_path)

    return experiment_results_path


def generate_dummy_data(n_samples, n_features):
    data_generation_start = time.time()
    X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=20, cluster_std=100, random_state=0)
    data_generation_time = time.time() - data_generation_start
    logging.info(f"Created dummy dataset ({n_samples} samples, {n_features} dimensions) in {data_generation_time:.2f} seconds")

    return X, y


def validate_n_jobs_to_test(n_jobs_to_test):
    n_cpu = os.cpu_count()
    logging.info(f"Found {n_cpu} logical processors")
    if n_jobs_to_test != -1:
        if max(n_jobs_to_test) > n_cpu:
            raise ValueError(f"Found {n_cpu} logical processors, but intended to use {max(n_jobs_to_test)}")
    else:
        n_jobs_to_test = list(range(1, n_cpu))
        logging.info(f"N_JOBS_TO_TEST = -1. All n_jobs up to one minus the number of available logical processors will be tested: {n_jobs_to_test}")

    return n_jobs_to_test


def initialize_model(selected_model):
    match selected_model:
        case "RandomForestClassifier":
            model = RandomForestClassifier()
        case "KNeighborsClassifier":
            model = KNeighborsClassifier()
        case "DefaultLogisticRegression":
            model = LogisticRegression()                    
        case "SagaLogisticRegression":
            model = LogisticRegression(multi_class="ovr", solver="saga")
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

    return model


def train_model(model):
    training_start = time.time()
    with joblib.parallel_backend(selected_parallelization_backend, n_jobs=iteration_n_jobs):
        model.fit(X, y)
    training_time = time.time() - training_start
    logging.info(f"Trained with n_jobs={iteration_n_jobs}: {training_time:.2f} seconds")

    return training_time


def compute_duration_statistics(training_times):
    training_times_means = np.mean(training_times, axis=-1)
    training_times_variances = np.var(training_times, axis=-1)
    logging.info(f"Mean training durations with {config.n_trials} trials: {training_times_means}")
    logging.info(f"Variance of the training durations with {config.n_trials} trials: {training_times_means}")

    return training_times_means, training_times_variances


if __name__ == "__main__": 

    start = time.time()

    # Read configuration
    with open('config.json', 'r') as f:
        config = json.load(f, object_hook=lambda d: Config(**d))

    # Create results directory structure
    results_path = create_results_directory(config.results_directory)
    experiment_results_path = create_experiment_results_directory(results_path)

    # Set up logging configuration
    experiment_results_directory = os.path.basename(experiment_results_path)
    logfile_name = f"{experiment_results_directory}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="{asctime} {levelname:<8} {message}",
        style='{',
        filename=os.path.join(experiment_results_path, logfile_name),
        filemode='w'
    )
    logging.info(f"Read experiment configuration: {config}")

    # Generate dummy data
    X, y = generate_dummy_data(config.n_samples, config.n_features)

    # Train with different number of logical processors (n_jobs)  
    n_jobs_to_test = validate_n_jobs_to_test(config.n_jobs_to_test)

    all_training_times_means = []
    all_training_times_variances = []
    overall_plot_labels = []
    for selected_model in config.models:
        logging.info(f"Selected model: {selected_model}")
        for selected_parallelization_backend in config.parallelization_backends:
            logging.info(f"Selected parallelization backend: {selected_parallelization_backend}")
            training_times = np.zeros((len(n_jobs_to_test), config.n_trials))
            for i, iteration_n_jobs in enumerate(n_jobs_to_test):
                for n in range(config.n_trials):
                    model = initialize_model(selected_model)                   
                    training_time = train_model(model)
                    training_times[i, n] = training_time

            # Compute duration statistics (mean and variance)
            training_times_means, training_times_variances = compute_duration_statistics(training_times)
            all_training_times_means.append(list(training_times_means))
            all_training_times_variances.append(list(training_times_variances))

            # Plot and save iteration results in absolute duration
            iteration_parameters_string = f"{selected_model}_{selected_parallelization_backend}"
            overall_plot_labels.append(iteration_parameters_string)
            plt.figure(1)    
            plt.errorbar(n_jobs_to_test, training_times_means, training_times_variances)
            plt.xlabel("n_jobs (number of logical processors)")
            plt.ylabel("Training time (seconds)")
            plt.title(iteration_parameters_string)
            plt.xticks(n_jobs_to_test)
            plt.ylim([0, 1.1 * max(training_times_means)])
            plt.grid()
            image_name = f"{config.image_name_base}_{config.n_samples}x{config.n_features}_{iteration_parameters_string}.png"
            plt.savefig(os.path.join(experiment_results_path, image_name))
            plt.clf()

            # Plot and save iteration results in percentual duration normalized to n_jobs=1
            normalized_training_times_means = (np.array(training_times_means) / training_times_means[0]) * 100
            plt.figure(2)
            plt.plot(n_jobs_to_test, normalized_training_times_means)
            plt.xlabel("n_jobs (number of logical processors)")
            plt.ylabel("Normalized training time (%)")
            plt.title(iteration_parameters_string)
            plt.xticks(n_jobs_to_test)
            plt.ylim([0, 1.1 * max(normalized_training_times_means)])
            plt.grid()
            percentual_image_name = f"Percentual{config.image_name_base}_{config.n_samples}x{config.n_features}_{iteration_parameters_string}.png"
            plt.savefig(os.path.join(experiment_results_path, percentual_image_name))
            plt.clf()
            
            logging.info(f"Saved iteration results into {image_name} and {percentual_image_name}")

    logging.info(f"All mean training durations: {all_training_times_means}")
    logging.info(f"All training durations variances: {all_training_times_variances}")
    logging.info(f"All iteration labels: {overall_plot_labels}")

    # Plot and save overall results
    plt.figure(1)
    plt.plot(n_jobs_to_test, np.transpose(np.array(all_training_times_means)))
    plt.xlabel("n_jobs (number of logical processors)")
    plt.ylabel("Training time (seconds)")
    plt.legend(overall_plot_labels, bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.xticks(n_jobs_to_test)
    plt.grid()
    image_name = f"{config.image_name_base}_{config.n_samples}x{config.n_features}_OverallResults.png"
    plt.savefig(os.path.join(experiment_results_path, image_name), bbox_inches='tight')
    plt.clf()
    logging.info(f"Saved overall results into {image_name}")

    total_time = time.time() - start
    logging.info(f"Total wall time: {total_time:.2f} seconds")
