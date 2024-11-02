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

    # 音素のリストを位置ごとに保持する
    predicted_phonemes = text_to_phonemes(transcription).split()
    expected_phonemes = text_to_phonemes(expected_text).split()

    # 音素に合わせたアドバイスメッセージを追加する辞書
    phoneme_advice = {
        "tə": "t音の後に、口を閉じずに短く「ə」を発音しましょう。",
        "tuː": "t音の後に「uː」を伸ばして発音します。",
        # 他の音素のアドバイスも同様に追加
    }

    # フィードバックの生成部分を更新
    feedback = []
    for idx, (p_pred, p_exp) in enumerate(zip(predicted_phonemes, expected_phonemes)):
        if p_pred != p_exp:
            message = f"位置 {idx+1} の音素 '{p_pred}' を '{p_exp}' に修正してください。"
            # アドバイスを追加
            advice = phoneme_advice.get(p_exp, "")
            if advice:
                message += f" {advice}"
            feedback.append({
                "position": idx,
                "expected_phoneme": p_exp,
                "predicted_phoneme": p_pred,
                "message": message
            })


    result = {
        "recognized_text": transcription,
        "phoneme_transcription": predicted_phonemes,
        "expected_phonemes": expected_phonemes,
        "feedback": feedback or ["発音が良好です！"]
    }

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
