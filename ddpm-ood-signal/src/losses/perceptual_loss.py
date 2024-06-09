import torch
import torch.nn as nn

from typing import Dict, Tuple


class Simple1DPerceptualNet(nn.Module):
    def __init__(self, in_channels):
        super(Simple1DPerceptualNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Global Average Pooling
        )

    def forward(self, x):
        return self.layers(x)


class Simple2DPerceptualNet(nn.Module):
    def __init__(self, in_channels):
        super(Simple2DPerceptualNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        )

    def forward(self, x):
        return self.layers(x)


class Simple3DPerceptualNet(nn.Module):
    def __init__(self, in_channels):
        super(Simple3DPerceptualNet, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)  # Global Average Pooling
        )

    def forward(self, x):
        return self.layers(x)


class PerceptualLoss(nn.Module):
    def __init__(
            self,
            dimensions: int,
            in_channels: int,
            include_pixel_loss: bool = True,
            is_fake_3d: bool = True,
            drop_ratio: float = 0.0,
            fake_3d_axis: Tuple[int, ...] = (2, 3, 4),
            lpips_normalize: bool = True,
            spatial: bool = False,
    ):
        super(PerceptualLoss, self).__init__()

        if not (dimensions in [1, 2, 3]):
            raise NotImplementedError("Perceptual loss is implemented only in 1D, 2D, and 3D.")

        if dimensions == 3 and is_fake_3d is False:
            raise NotImplementedError("True 3D perceptual loss is not implemented yet.")

        self.dimensions = dimensions
        self.in_channels = in_channels
        self.include_pixel_loss = include_pixel_loss
        self.lpips_normalize = lpips_normalize

        # Define the perceptual function based on dimensions
        if self.dimensions == 1:
            self.perceptual_function = Simple1DPerceptualNet(in_channels)
        elif self.dimensions == 2:
            self.perceptual_function = Simple2DPerceptualNet(in_channels)
        else:  # self.dimensions == 3
            self.perceptual_function = Simple3DPerceptualNet(in_channels)

        self.perceptual_factor = 1

    def forward(self, y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y = y.float()
        y_pred = y_pred.float()

        if self.lpips_normalize:
            y = (y - 0.5) * 2  # Assuming input is in [0, 1]
            y_pred = (y_pred - 0.5) * 2  # Assuming input is in [0, 1]

        batch_size = y.shape[0]

        # Calculate perceptual loss for each item in the batch
        losses = torch.zeros(batch_size, device=y.device)

        for i in range(batch_size):
            y_i = y[i].unsqueeze(0)  # Shape: (1, C, L) for 1D, (1, C, H, W) for 2D, (1, C, D, H, W) for 3D
            y_pred_i = y_pred[i].unsqueeze(0)

            y_features = self.perceptual_function(y_i)
            y_pred_features = self.perceptual_function(y_pred_i)

            losses[i] = nn.functional.mse_loss(y_features, y_pred_features)

        return losses * self.perceptual_factor

    def _calculate_fake_3d_loss(
            self,
            y: torch.Tensor,
            y_pred: torch.Tensor,
            permute_dims: Tuple[int, int, int, int, int],
            view_dims: Tuple[int, int, int],
    ):
        y_slices = (
            y.permute(*permute_dims)
            .contiguous()
            .view(-1, y.shape[view_dims[0]], y.shape[view_dims[1]], y.shape[view_dims[2]])
        )

        y_pred_slices = (
            y_pred.permute(*permute_dims)
            .contiguous()
            .view(
                -1,
                y_pred.shape[view_dims[0]],
                y_pred.shape[view_dims[1]],
                y_pred.shape[view_dims[2]],
            )
        )

        indices = torch.randperm(y_pred_slices.shape[0], device=y_pred_slices.device)[
                  : int(y_pred_slices.shape[0] * self.keep_ratio)
                  ]

        y_pred_slices = y_pred_slices[indices]
        y_slices = y_slices[indices]

        y_features = self.perceptual_function(y_slices.unsqueeze(1))
        y_pred_features = self.perceptual_function(y_pred_slices.unsqueeze(1))
        p_loss = nn.functional.mse_loss(y_features, y_pred_features)

        return p_loss

    def get_summaries(self) -> Dict[str, torch.Tensor]:
        return self.summaries

    def get_perceptual_factor(self) -> float:
        return self.perceptual_factor

    def set_perceptual_factor(self, perceptual_factor: float) -> float:
        self.perceptual_factor = perceptual_factor
        return self.get_perceptual_factor()
