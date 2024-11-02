// frontend/src/App.tsx
import React, { useState } from 'react';
import { analyzePronunciation } from './services/api';

function App() {
    const [audioFile, setAudioFile] = useState<File | null>(null);
    const [expectedText, setExpectedText] = useState<string>('');
    const [feedback, setFeedback] = useState<string | null>(null);

    const handleAudioChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        if (e.target.files) {
            setAudioFile(e.target.files[0]);
        }
    };

    const handleTextChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setExpectedText(e.target.value);
    };

    const handleSubmit = async () => {
        if (audioFile && expectedText) {
            try {
                const result = await analyzePronunciation(audioFile, expectedText);
                setFeedback(JSON.stringify(result.feedback, null, 2)); // フィードバックを表示
            } catch (error) {
                console.error('Error analyzing pronunciation:', error);
                setFeedback('エラーが発生しました。再試行してください。');
            }
        } else {
            alert('音声ファイルとテキストを入力してください');
        }
    };

    return (
        <div className="App">
            <h1>発音分析アプリ</h1>
            <input type="file" accept="audio/*" onChange={handleAudioChange} />
            <input 
                type="text" 
                value={expectedText} 
                onChange={handleTextChange} 
                placeholder="期待されるテキストを入力"
            />
            <button onClick={handleSubmit}>解析を送信</button>
            <div>
                <h2>フィードバック</h2>
                <pre>{feedback}</pre>
            </div>
        </div>
    );
}

export default App;
