from flask import Flask, request, jsonify, send_file
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import tempfile
import os
import torch
from sox import Transformer

# Set environment variables for GPU usage
os.environ["TORCH_CUDA_ARCH_LIST"] = "All"
os.environ["FORCE_CUDA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use the first GPU

app = Flask(__name__)

# Download and load all models
preload_models(
    text_use_small=True,
    coarse_use_small=True,
    fine_use_gpu=True,
    fine_use_small=True)

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']

    # Determine if running on CPU or GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")


    # Generate audio from text
    audio_array = generate_audio(text)

    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_mp3_file:
            transformer = Transformer()
            transformer.build(
                input_array=audio_array,
                sample_rate_in=SAMPLE_RATE,
                output_filepath=temp_mp3_file.name)

            return send_file(temp_mp3_file.name, mimetype="audio/mp3", as_attachment=True, download_name="bark_generation.mp3")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8090)
