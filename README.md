# Performance of Scikit-learn parallelization
This repository contains a script to evaluate the speed up of training Scikit-learn algorithms leveraging parallelization, mainly using the flag `n_jobs` from `joblib.parallel_backend`. The script `experiment.py` times the training of a given set of models with a dummy dataset using different numbers of logical processors and parallelization backends. The measured times are logged as well as plotted for a convenient analysis.

To run it in your machine, follow the next steps:

1. Install dependencies using `pip install -r requirements.txt`.
2. Select your experiment configuration in `config.json`.
    * List of number of logical processors to be tested, `N_JOBS_TO_TEST = [1, 3, 5, 10]`
    * Set of models, e.g., `MODELS = ["RandomForestClassifier", "KNeighborsClassifier", "LogisticRegression", "MLPClassifier", "SVC", "GaussianNB", "DecisionTreeClassifier", "SGDClassifier"]`.
    * Parallelization backends, e.g., `PARALLELIZATION_BACKENDS = ["threading", "loky"]`.
    * Dummy dataset size, e.g., `N_SAMPLES = 100` and `N_FEATURES = 100`.
    * Number of trials per combination, e.g., `N_TRIALS = 3`. Useful to take into account the variance, included in the plots with error bars.
    * Base name for the plot images, e.g., `IMAGE_NAME_BASE = "TrainingDurations"`
3. Execute `experiment.py`. A log file (e.g. `experiment_2023_03_12_19_20_48.log`) can be used to track the execution.
4. Find your results as plots the automatically generated directory `Results`.

Please write me if you have any question understanding the code (guillemboada@hotmail.com).