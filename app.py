import os
import base64
from io import BytesIO
from faster_whisper import WhisperModel

from potassium import Potassium, Request, Response
import torch

app = Potassium("echilly-io_serverless-whisper-large")


# @app.init runs at startup, and initializes the app's context
@app.init
def init():
    device = 0 if torch.cuda.is_available() else -1
    model_size = "large-v2"
    compute_type = "int8_float16"
    model = WhisperModel(model_size, device, compute_type)
    context = {
        "model": model,
    }

    return context


@app.handler()
def inference(context: dict, request: Request) -> Response:
    model = context.get("model")
    prompt = request.json.get("prompt")

    mp3BytesString = prompt.get("mp3BytesString")
    if mp3BytesString == None:
        return Response(
            json={
                "output": "Mp3Bytes String not provided",
            },
            status=500,
        )
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    with open("input.mp3", "wb") as file:
        file.write(mp3Bytes.getbuffer())

    result = model.transcribe("input.mp3")

    return Response(json=result, status=200)


if __name__ == "__main__":
    app.serve()
