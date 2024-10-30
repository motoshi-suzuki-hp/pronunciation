# app.py
from flask import Flask, request, jsonify
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import soundfile as sf
import librosa

app = Flask(__name__)

# Wav2Vec 2.0のモデルとプロセッサの読み込み
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

@app.route("/analyze", methods=["POST"])
def analyze_audio():
    # 音声ファイルの取得と読み込み
    audio_file = request.files["audio"]
    audio_data, sample_rate = sf.read(audio_file)

    # サンプルレートが16,000でない場合はリサンプリング
    if sample_rate != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
        sample_rate = 16000

    # 音声データの前処理
    input_values = processor(audio_data, sampling_rate=sample_rate, return_tensors="pt").input_values

    # モデルでテキスト化
    with torch.no_grad():
        logits = model(input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.decode(predicted_ids[0])

    # シンプルなフィードバック（サンプル）
    feedback = {
        "recognized_text": transcription,
        "feedback": "Overall pronunciation clarity could be improved."  # 改善ポイントの例
    }

    return jsonify(feedback)

if __name__ == "__main__":
    app.run(debug=True)
