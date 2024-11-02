from flask import Flask, request, jsonify, Response
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
from phonemizer import phonemize
import librosa
import json

app = Flask(__name__)

# 音素解析用のWav2Vec 2.0と音素変換関数の設定
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

def text_to_phonemes(text):
    phonemes = phonemize(text, language="en-us", backend="espeak")
    return phonemes

def evaluate_pronunciation(audio_file, expected_text):
    audio_data, sample_rate = sf.read(audio_file)
    
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)

    input_values = processor(audio_data, sampling_rate=16000, return_tensors="pt").input_values

    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    predicted_phonemes = text_to_phonemes(transcription)
    expected_phonemes = text_to_phonemes(expected_text)

    feedback = []
    for p_pred, p_exp in zip(predicted_phonemes.split(), expected_phonemes.split()):
        if p_pred != p_exp:
            feedback.append(f"Expected {p_exp} but got {p_pred}")

    result = {
        "recognized_text": transcription,
        "phoneme_transcription": predicted_phonemes,
        "expected_phonemes": expected_phonemes,
        "feedback": feedback or ["Good pronunciation!"]
    }

    # JSONレスポンスをUTF-8で返す
    return Response(json.dumps(result, ensure_ascii=False), content_type="application/json; charset=utf-8")

@app.route('/analyze', methods=['POST'])
def analyze():
    audio_file = request.files['audio']
    text_file = request.files['expected_text']

    expected_text = text_file.read().decode("utf-8")

    result = evaluate_pronunciation(audio_file, expected_text)
    return result

if __name__ == "__main__":
    app.run(debug=True)
