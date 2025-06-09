async function analyzeText() {
    const text = document.getElementById('newsText').value.trim();
    const loading = document.querySelector('.loading');
    const result = document.getElementById('result');
    const prediction = document.getElementById('prediction');
    const confidence = document.getElementById('confidence');

    if (!text) {
        alert('Please enter some text to analyze');
        return;
    }

    // Show loading spinner
    loading.style.display = 'block';
    result.style.display = 'none';

    try {
        const response = await fetch('/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text }),
        });

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const data = await response.json();

        // Hide loading spinner
        loading.style.display = 'none';
        
        // Display results
        result.style.display = 'block';
        result.className = `result ${data.is_fake ? 'fake' : 'real'}`;
        prediction.textContent = data.is_fake ? 'This news appears to be FAKE' : 'This news appears to be REAL';
        confidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    } catch (error) {
        loading.style.display = 'none';
        alert('An error occurred while analyzing the text. Please try again.');
        console.error('Error:', error);
    }
}

// Add event listener for Enter key
document.getElementById('newsText').addEventListener('keypress', function(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        analyzeText();
    }
}); 