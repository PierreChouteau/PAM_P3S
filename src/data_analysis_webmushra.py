import pandas as pd
import numpy as np


def read_data(path: str):
    data = pd.read_csv(path)
    experiment_names = pd.value_counts(data["trial_id"]).keys().values
    stimuli = pd.value_counts(data["rating_stimulus"]).keys().values
    return data, data.columns.values, experiment_names, stimuli


def get_data_for_subject(data, subject):
    sub_data = data[subject]
    return sub_data


def stats(data, type):
    if type is str:
        return pd.value_counts(data)
    return data.mean(), data.std(), len(data)


def scale(data, min, max, total_data):
    scaled_data = (data - min) / (max - min)
    if (scaled_data[scaled_data>1] is not []):
        print("Some scores are beyond maximum")
        print(total_data[scaled_data>1])
    if (scaled_data[scaled_data<0] is not []):
        print("Some scores are below minimum")
        print(total_data[scaled_data<0])
    return scaled_data

def retrieve_experiment_data(data, experiment):
    return data[data["trial_id"] == experiment]

def retrieve_stimulus_data(data, stimulus):
    return data[data["rating_stimulus"] == stimulus]


def plot_stats(data, experiments, stimuli):
    data_experiments = []
    for e in experiments:
        data_experiments.append(retrieve_experiment_data(data, e))
    for s in stimuli:
        for i, data in enumerate(data_experiments):
            experiment = experiments[i]
            rating_stats = stats(data["rating_score"], float)
            gender_stats = stats(data["gender"], str)
            print(f"Stimulus: {s}\nExperiment: {experiment}\nRating Stats: {rating_stats}\nGender Stats: {gender_stats}")
            
    


# Stats for ages
# Stats for reference for overall
# Stats for reference for distorion
# ...
# Stats for sto for overall
# Stats for anchor for overall

# Stats for every mix in every configuration

# Scale with min as anchor and max as reference individual-wise for the same experiment
# Beware of values beyond the borders

# Box plot / Violin plot with mean and standard deviation


def main():
    filename = "mushra.csv"
    data, keys, experiment_names, stimuli = read_data(filename)
    print(f"Columns: {keys}\nExperiments: {experiment_names}\nStimuli: {stimuli}\n\n\n")
    str_data = get_data_for_subject(data, "gender")
    num_data = get_data_for_subject(data, "rating_score")
    # print(stats(str_data, str))
    # print(scale(num_data, 20, 100, data))

    data_experiment_0 = retrieve_experiment_data(data, experiment_names[0])
    # print(data_experiment_0)

    data_stimulus_0 = retrieve_stimulus_data(data_experiment_0, stimuli[0])
    # print(data_stimulus_0[['gender', 'rating_score']])

    plot_stats(data, experiment_names, stimuli)


if __name__ == "__main__":
    main()
