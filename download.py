# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from faster_whisper import WhisperModel
import torch


def download_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    model_size = "large-v2"
    compute_type = "float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)


if __name__ == "__main__":
    download_model()
