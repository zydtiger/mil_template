import numpy as np
import os
import json
from defs import PatientInstances

# Set random seed for reproducibility
np.random.seed(42)

# Default parameters
params = {
    "n_patients": 100,  # Number of patients
    "n_samples_range": (6, 20),  # Range of instances per patient
    "feature_dim": 192,  # Feature dimension
    "positive_patient_ratio": 0.5,  # Ratio of positive patients
    "distinct_sample_ratio": 0.1,  # Ratio of distinct samples in positive bags
    "control_mean": 0.0,  # Mean for control features
    "control_std": 1.0,  # Standard deviation for control features
    "distinct_mean_shift": 2.0,  # Mean shift for distinct features
    "distinct_std_scale": 1.5,  # Standard deviation scale for distinct features
}


def generate_patient_id(idx: int, is_positive: bool) -> str:
    """Generate patient ID in the required format."""
    if is_positive:
        return f"S{idx:03d}"
    else:
        return f"C{idx:03d}"


def generate_instance_features(is_distinct: bool) -> np.ndarray:
    """Generate feature vector for a single instance."""
    if is_distinct:
        # Generate distinct features with shifted distribution
        return np.random.normal(
            params["control_mean"] + params["distinct_mean_shift"],
            params["control_std"] * params["distinct_std_scale"],
            params["feature_dim"],
        )
    else:
        # Generate control features
        return np.random.normal(
            params["control_mean"],
            params["control_std"],
            params["feature_dim"],
        )


def generate_patient_data(
    patient_idx: int, patient_label: int
) -> PatientInstances:
    """Generate all instances for a single patient."""

    # Generate patient ID
    patient_id = generate_patient_id(patient_idx, patient_label == 1)

    # Determine number of instances for this patient
    n_instances = np.random.randint(
        params["n_samples_range"][0], params["n_samples_range"][1] + 1
    )

    # Initialize list to store instance features
    instance_features = []

    # For positive patients, determine which instances will be distinct
    if patient_label == 1:
        n_distinct = max(1, int(n_instances * params["distinct_sample_ratio"]))
        distinct_indices = np.random.choice(
            n_instances, n_distinct, replace=False
        )
    else:
        distinct_indices = []

    # Generate instances
    for instance_idx in range(n_instances):
        is_distinct = instance_idx in distinct_indices

        # Generate features and convert to list
        features = generate_instance_features(is_distinct).tolist()
        instance_features.append(features)

    # Create and return PatientInstances object
    return PatientInstances(
        name=patient_id, label=patient_label, features=instance_features
    )


def generate_mil_dataset() -> list[PatientInstances]:
    """Generate complete MIL dataset."""

    # Determine patient labels
    n_positive = int(params["n_patients"] * params["positive_patient_ratio"])
    patient_labels = np.zeros(params["n_patients"])
    patient_labels[:n_positive] = 1
    np.random.shuffle(patient_labels)

    # Keep track of positive and negative counts for ID generation
    pos_count = 0
    neg_count = 0

    # Initialize list to store all patient data
    all_patients = []

    # Generate data for each patient
    for patient_idx in range(params["n_patients"]):
        patient_label = int(patient_labels[patient_idx])

        # Use the appropriate counter for ID generation
        if patient_label == 1:
            actual_idx = pos_count
            pos_count += 1
        else:
            actual_idx = neg_count
            neg_count += 1

        # Generate patient data
        patient_data = generate_patient_data(actual_idx, patient_label)
        all_patients.append(patient_data)

    return all_patients


def save_dataset(patients: list[PatientInstances], output_file: str):
    """Save the generated dataset to a JSON file."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Convert to JSON-serializable format
    patients_json = [patient.model_dump() for patient in patients]

    with open(output_file, "w") as f:
        json.dump(patients_json, f)


if __name__ == "__main__":
    train_dataset = generate_mil_dataset()
    val_dataset = generate_mil_dataset()
    save_dataset(train_dataset, "./data/train.json")
    save_dataset(val_dataset, "./data/validate.json")

    # Function to calculate and display statistics
    def print_dataset_stats(dataset: list[PatientInstances], name: str):
        n_patients = len(dataset)
        n_positive = sum(1 for patient in dataset if patient.label == 1)
        n_negative = n_patients - n_positive
        total_instances = sum(len(patient.features) for patient in dataset)
        avg_instances = total_instances / n_patients if n_patients > 0 else 0

        print(f"\n{'-' * 20} {name} DATASET {'-' * 20}")
        print(f"{'Metric':<25} {'Value':<10}")
        print("-" * 35)
        print(f"{'Total patients':<25} {n_patients:<10}")
        print(f"{'Positive patients':<25} {n_positive:<10}")
        print(f"{'Negative patients':<25} {n_negative:<10}")
        print(f"{'Total instances':<25} {total_instances:<10}")
        print(f"{'Avg instances/patient':<25} {avg_instances:.2f}")

    # Print statistics for both datasets
    print_dataset_stats(train_dataset, "TRAIN")
    print_dataset_stats(val_dataset, "VALIDATION")
