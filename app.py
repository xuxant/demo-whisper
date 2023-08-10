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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_size = "large-v2"
    compute_type = "int8_float16"
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    context = {
        "model": model,
    }

    return context


@app.handler()
def inference(context: dict, request: Request) -> Response:
    model = context.get("model")
    prompt = request.json.get("prompt")
    file_format = "mp3"
    kwargs = {}
    try:
        file_format = prompt.get("format")
        kwargs = prompt.get("model_inputs")
    except:
        pass
    mp3BytesString = prompt.get("mp3BytesString")
    if mp3BytesString == None:
        return Response(
            json={
                "output": "Mp3Bytes String not provided",
            },
            status=500,
        )
    mp3Bytes = BytesIO(base64.b64decode(mp3BytesString.encode("ISO-8859-1")))
    tmp_file = 'input.'+format
    with open(tmp_file, "wb") as file:
        file.write(mp3Bytes.getbuffer())

    segments, info = model.transcribe("input.mp3", **kwargs)
    text = ""
    real_segments = []
    for segment in segments:
        text += segment.text + " "
        current_segment = {
            "text": segment.text,
            "start": segment.start,
            "end": segment.end,
        }
        if kwargs.get("word_timestamps", False) and segment.words:
            current_segment["word_timestamps"] = segment.words

        real_segments.append(current_segment)
    
    result = {
        "text": text,
        "segments": segments,
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }
    
    return Response(json=result, status=200)


if __name__ == "__main__":
    app.serve()
