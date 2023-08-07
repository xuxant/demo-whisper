# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from faster_whisper import WhisperModel

def download_model():
    device = 0 if torch.cuda.is_available() else -1
    model_size = "large-v2"
    compute_type = "int8_float16"
    model = WhisperModel(
        model_size=model_size,
        device=device,
        compute_type=compute_type
    )

if __name__ == "__main__":
    download_model()