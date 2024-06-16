import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, latent_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / self.num_embeddings, 1 / self.num_embeddings)

    def forward(self, x):
        # Calculate distances
        flat_x = x.view(-1, self.embedding_dim)
        distances = torch.cdist(flat_x, self.embeddings.weight, p=2)

        # Get the closest embedding index for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embeddings.weight).view(x.shape)

        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, loss


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(latent_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_dim, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        quantized, vq_loss = self.vq_layer(z)
        x_recon = self.decoder(quantized)
        recon_loss = F.mse_loss(x_recon, x)
        return x_recon, recon_loss + vq_loss


if __name__ == "__main__":
    # Parámetros del modelo
    in_channels = 2  # Dos canales de entrada
    hidden_dim = 64
    latent_dim = 32
    num_embeddings = 512
    commitment_cost = 0.25

    # Inicialización del modelo y forward
    model = VQVAE(in_channels, hidden_dim, latent_dim, num_embeddings, commitment_cost)
    x = torch.randn(1, in_channels, 100)  # Ejemplo de entrada con tamaño variable
    x_recon, loss = model(x)
    print(f'Reconstructed: {x_recon.shape}, Loss: {loss.item()}')
