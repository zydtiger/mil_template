import torch
import torch.nn as nn


class AttentionMIL(nn.Module):
    def __init__(self, input_dim):
        super(AttentionMIL, self).__init__()

        # Instance-level feature extractor
        self.instance_encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(16, 8),
            nn.Tanh(),
            nn.Linear(8, 1),
        )
        # Note: Removed Softmax from Sequential as we'll apply it after masking

        # Final classification layer
        self.classifier = nn.Sequential(nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            x: Input tensor of shape (batch_size, n_instances, input_dim)
                where n_instances is padded to max_instances
            mask: Binary mask of shape (batch_size, n_instances) where:
                 1 indicates valid instance
                 0 indicates padding
        Returns:
            patient_prob: Probability of patient being positive
            instance_attn: Attention weights for each instance (0 for padded instances)
        """
        # Get batch size and number of instances
        batch_size, n_instances, _ = x.size()

        # Reshape to process all instances
        x = x.view(-1, x.size(-1))  # (batch_size * n_instances, input_dim)

        # Extract instance-level features
        instance_features = self.instance_encoder(
            x
        )  # (batch_size * n_instances, 16)

        # Reshape back to separate bags
        instance_features = instance_features.view(
            batch_size, n_instances, -1
        )  # (batch_size, n_instances, 16)

        # Calculate attention scores (before softmax)
        attention_scores = self.attention(
            instance_features
        )  # (batch_size, n_instances, 1)
        attention_scores = attention_scores.squeeze(
            -1
        )  # (batch_size, n_instances)

        # Apply mask to attention scores
        # Set attention scores for padded instances to -inf before softmax
        attention_scores = attention_scores.masked_fill(
            mask == 0, float("-inf")
        )

        # Apply softmax to get attention weights (masked instances will get ~0 weight)
        attention_weights = torch.softmax(
            attention_scores, dim=1
        )  # (batch_size, n_instances)

        # Expand attention weights for broadcasting
        attention_weights = attention_weights.unsqueeze(
            -1
        )  # (batch_size, n_instances, 1)

        # Apply attention pooling
        bag_features = torch.sum(
            attention_weights * instance_features, dim=1
        )  # (batch_size, 16)

        # Final classification
        patient_prob = self.classifier(bag_features)  # (batch_size, 1)

        return patient_prob, attention_weights

    def get_instance_attention(self, x: torch.Tensor, mask: torch.Tensor):
        """
        Helper method to get attention weights for interpretability

        Args:
            x: Input tensor of shape (batch_size, n_instances, input_dim)
            mask: Binary mask of shape (batch_size, n_instances)

        Returns:
            attention: Attention weights (0 for padded instances)
        """
        with torch.no_grad():
            _, attention = self.forward(x, mask)
        return attention


# Example usage
if __name__ == "__main__":
    # Test the model
    batch_size = 4
    n_instances = 20
    input_dim = 192

    # Create dummy data
    x = torch.randn(batch_size, n_instances, input_dim)
    mask = torch.ones(batch_size, n_instances)
    # Simulate some padding in the last few instances
    mask[:, -5:] = 0

    # Initialize model
    model = AttentionMIL(input_dim)

    # Forward pass
    probs, attention = model(x, mask)

    # Print shapes
    print(f"Input shape: {x.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Output probabilities shape: {probs.shape}")
    print(f"Attention weights shape: {attention.shape}")

    # Verify that attention weights are 0 for masked instances
    print("\nAttention for masked instances:")
    print(attention[0, -5:])  # Should be ~0
    print("\nAttention for valid instances (should sum to ~1):")
    print(attention[0, :-5].sum())
