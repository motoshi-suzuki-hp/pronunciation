// frontend/src/services/api.ts
export async function analyzePronunciation(audioFile: File, expectedText: string) {
    const formData = new FormData();
    formData.append('audio', audioFile);
    formData.append('expected_text', new Blob([expectedText], { type: 'text/plain' }));

    const response = await fetch('http://localhost:5000/analyze', {
        method: 'POST',
        body: formData,
    });
    return response.json();
}
