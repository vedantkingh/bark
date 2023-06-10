from flask import Flask, request, jsonify, send_file
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
import tempfile

app = Flask(__name__)

# Download and load all models
preload_models()

@app.route('/synthesize', methods=['POST'])
def synthesize():
    text = request.json['text']
    
    # Generate audio from text
    audio_array = generate_audio(text)
    
    # Save audio to a temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as temp_file:
        write_wav(temp_file.name, SAMPLE_RATE, audio_array)
        temp_file.seek(0)
        
        return send_file(temp_file.name, mimetype="audio/wav", as_attachment=True, download_name="bark_generation.wav")

if __name__ == '__main__':
    app.run()
