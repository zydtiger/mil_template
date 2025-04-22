import json
import torch
from torch.utils.data import Dataset, DataLoader
from defs import PatientInstances


class MILDataset(Dataset):
    def __init__(self, json_path: str, max_instances: int):
        """
        Initialize the MIL dataset from a JSON file

        Args:
            json_path (str): Path to the JSON file containing the MIL dataset
            max_instances (int): Maximum number of instances to pad/truncate to
        """
        # Load the JSON file
        with open(json_path, "r") as f:
            data = json.load(f)

        # Convert JSON data to list of PatientInstances
        self.patients = [
            PatientInstances(**patient_data) for patient_data in data
        ]
        self.max_instances = max_instances

    def __len__(self) -> int:
        """Return the total number of patients in the dataset"""
        return len(self.patients)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor, torch.Tensor]:
        """
        Get a single patient's data with padding

        Args:
            idx (int): Index of the patient

        Returns:
            tuple: (label, mask, features) where:
                - label (int): Patient's label (0 or 1)
                - mask (torch.Tensor): Binary mask indicating valid instances (1) vs padding (0)
                - features (torch.Tensor): Padded feature vectors for each instance
        """
        patient = self.patients[idx]
        features = patient.features
        n_instances = len(features)
        feature_dim = len(features[0])

        # Create padded features tensor
        padded_features = torch.zeros((self.max_instances, feature_dim))
        # Create mask tensor (1 for valid instances, 0 for padding)
        mask = torch.zeros(self.max_instances)

        # Fill in actual instances and mask
        n_instances = min(
            n_instances, self.max_instances
        )  # Truncate if needed
        padded_features[:n_instances] = torch.tensor(features[:n_instances])
        mask[:n_instances] = 1.0

        return patient.label, mask, padded_features


def get_train_loader(batch_size: int, max_instances: int) -> DataLoader:
    dataset = MILDataset("./data/train.json", max_instances)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_val_loader(batch_size: int, max_instances: int) -> DataLoader:
    dataset = MILDataset("./data/validate.json", max_instances)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# Example usage
if __name__ == "__main__":
    # Create dataset instance
    dataset = MILDataset("./data/train.json", max_instances=20)

    # Print some basic information
    print(f"Dataset size: {len(dataset)} patients")

    # Get first patient's data
    label, mask, features = dataset[0]
    print(f"First patient:")
    print(f"- Label: {label}")
    print(f"- Mask shape: {mask.shape}")
    print(f"- Number of valid instances: {mask.sum().item()}")
    print(f"- Features shape: {features.shape}")

    # Test the dataloader
    train_loader = get_train_loader(batch_size=4, max_instances=20)
    batch_labels, batch_masks, batch_features = next(iter(train_loader))
    print("\nBatch information:")
    print(f"- Labels shape: {batch_labels.shape}")
    print(f"- Masks shape: {batch_masks.shape}")
    print(f"- Features shape: {batch_features.shape}")
