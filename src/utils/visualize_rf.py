import numpy as np
import matplotlib.pyplot as plt


def visualize_radio_signal(tensor, N=3):
    """
    Visualizes N samples from the given tensor of radio signals.

    Parameters:
    tensor (numpy.ndarray): The input tensor of shape (batch size, 2 channels, length).
    N (int): The number of samples to visualize. Default is 3.

    """
    batch_size, channels, length = tensor.shape
    assert channels == 2, "The tensor should have 2 channels (real and imaginary parts)."

    # Randomly select N samples from the batch
    indices = np.random.choice(batch_size, N, replace=False)
    samples = tensor[indices]

    fig, axs = plt.subplots(N, 3, figsize=(18, 4 * N))

    for i, sample in enumerate(samples):
        real_part = sample[0]
        imaginary_part = sample[1]
        complex_signal = real_part + 1j * imaginary_part

        # Time-Domain Plot
        axs[i, 0].plot(real_part, label='Real Part')
        axs[i, 0].plot(imaginary_part, label='Imaginary Part')
        axs[i, 0].set_title(f'Sample {indices[i]} - Time Domain')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].legend()

        # Frequency-Domain Plot
        fft_signal = np.fft.fft(complex_signal)
        freqs = np.fft.fftfreq(length)
        axs[i, 1].plot(freqs, np.abs(fft_signal))
        axs[i, 1].set_title(f'Sample {indices[i]} - Frequency Domain')
        axs[i, 1].set_xlabel('Frequency')
        axs[i, 1].set_ylabel('Magnitude')

        # Constellation Diagram
        axs[i, 2].scatter(real_part, imaginary_part)
        axs[i, 2].set_title(f'Sample {indices[i]} - Constellation Diagram')
        axs[i, 2].set_xlabel('Real Part')
        axs[i, 2].set_ylabel('Imaginary Part')

    plt.tight_layout()
    plt.show()


# Example usage
# Assuming tensor is your input tensor of shape (batch size, 2 channels, length)
# tensor = np.random.randn(10, 2, 100)  # Example tensor
# visualize_radio_signal(tensor, N=3)

import numpy as np
import matplotlib.pyplot as plt


def visualize_original_vs_reconstructed(original_tensor, reconstructed_tensor, N=3):
    """
    Visualizes N samples of original and reconstructed radio signals.

    Parameters:
    original_tensor (numpy.ndarray): The original tensor of shape (batch size, 2 channels, length).
    reconstructed_tensor (numpy.ndarray): The reconstructed tensor of shape (batch size, 2 channels, length).
    N (int): The number of samples to visualize. Default is 3.

    """
    assert original_tensor.shape == reconstructed_tensor.shape, "Original and reconstructed tensors must have the same shape."

    batch_size, channels, length = original_tensor.shape
    assert channels == 2, "The tensors should have 2 channels (real and imaginary parts)."

    # Randomly select N samples from the batch
    indices = np.random.choice(batch_size, N, replace=False)
    original_samples = original_tensor[indices]
    reconstructed_samples = reconstructed_tensor[indices]

    fig, axs = plt.subplots(N, 6, figsize=(24, 4 * N))

    for i in range(N):
        original_sample = original_samples[i]
        reconstructed_sample = reconstructed_samples[i]

        # Original Signal
        original_real_part = original_sample[0]
        original_imaginary_part = original_sample[1]
        original_complex_signal = original_real_part + 1j * original_imaginary_part

        # Reconstructed Signal
        reconstructed_real_part = reconstructed_sample[0]
        reconstructed_imaginary_part = reconstructed_sample[1]
        reconstructed_complex_signal = reconstructed_real_part + 1j * reconstructed_imaginary_part

        # Time-Domain Plot
        axs[i, 0].plot(original_real_part, label='Real Part')
        axs[i, 0].plot(original_imaginary_part, label='Imaginary Part')
        axs[i, 0].set_title(f'Original Sample {indices[i]} - Time Domain')
        axs[i, 0].set_xlabel('Time')
        axs[i, 0].set_ylabel('Amplitude')
        axs[i, 0].legend()

        axs[i, 1].plot(reconstructed_real_part, label='Real Part')
        axs[i, 1].plot(reconstructed_imaginary_part, label='Imaginary Part')
        axs[i, 1].set_title(f'Reconstructed Sample {indices[i]} - Time Domain')
        axs[i, 1].set_xlabel('Time')
        axs[i, 1].set_ylabel('Amplitude')
        axs[i, 1].legend()

        # Frequency-Domain Plot
        original_fft_signal = np.fft.fft(original_complex_signal)
        original_freqs = np.fft.fftfreq(length)
        axs[i, 2].plot(original_freqs, np.abs(original_fft_signal))
        axs[i, 2].set_title(f'Original Sample {indices[i]} - Frequency Domain')
        axs[i, 2].set_xlabel('Frequency')
        axs[i, 2].set_ylabel('Magnitude')

        reconstructed_fft_signal = np.fft.fft(reconstructed_complex_signal)
        reconstructed_freqs = np.fft.fftfreq(length)
        axs[i, 3].plot(reconstructed_freqs, np.abs(reconstructed_fft_signal))
        axs[i, 3].set_title(f'Reconstructed Sample {indices[i]} - Frequency Domain')
        axs[i, 3].set_xlabel('Frequency')
        axs[i, 3].set_ylabel('Magnitude')

        # Constellation Diagram
        axs[i, 4].scatter(original_real_part, original_imaginary_part)
        axs[i, 4].set_title(f'Original Sample {indices[i]} - Constellation Diagram')
        axs[i, 4].set_xlabel('Real Part')
        axs[i, 4].set_ylabel('Imaginary Part')

        axs[i, 5].scatter(reconstructed_real_part, reconstructed_imaginary_part)
        axs[i, 5].set_title(f'Reconstructed Sample {indices[i]} - Constellation Diagram')
        axs[i, 5].set_xlabel('Real Part')
        axs[i, 5].set_ylabel('Imaginary Part')

    plt.tight_layout()
    plt.show()

# Example usage
# Assuming original_tensor and reconstructed_tensor are your input tensors of shape (batch size, 2 channels, length)
# original_tensor = np.random.randn(10, 2, 100)  # Example tensor
# reconstructed_tensor = np.random.randn(10, 2, 100)  # Example tensor
# visualize_original_vs_reconstructed(original_tensor, reconstructed_tensor, N=3)
