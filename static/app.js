const canvas = document.querySelector('#digit-canvas');
const context = canvas.getContext('2d');
const clearButton = document.querySelector('#clear-button');
const recognizeButton = document.querySelector('#recognize-button');
const brushSize = document.querySelector('#brush-size');
const resultEmpty = document.querySelector('#result-empty');
const resultContent = document.querySelector('#result-content');
const resultError = document.querySelector('#result-error');
const predictionDigit = document.querySelector('#prediction-digit');
const confidenceValue = document.querySelector('#confidence-value');
const confidenceBar = document.querySelector('#confidence-bar');
const probabilityList = document.querySelector('#probability-list');

context.fillStyle = '#171717';
context.fillRect(0, 0, canvas.width, canvas.height);
context.lineCap = 'round';
context.lineJoin = 'round';

let drawing = false;

function pointFromEvent(event) {
  const bounds = canvas.getBoundingClientRect();
  return {
    x: (event.clientX - bounds.left) * canvas.width / bounds.width,
    y: (event.clientY - bounds.top) * canvas.height / bounds.height,
  };
}

canvas.addEventListener('pointerdown', (event) => {
  drawing = true;
  canvas.setPointerCapture(event.pointerId);
  const point = pointFromEvent(event);
  context.beginPath();
  context.moveTo(point.x, point.y);
  context.lineTo(point.x + 0.1, point.y + 0.1);
  context.strokeStyle = '#ffffff';
  context.lineWidth = Number(brushSize.value);
  context.stroke();
});

canvas.addEventListener('pointermove', (event) => {
  if (!drawing) return;
  const point = pointFromEvent(event);
  context.lineWidth = Number(brushSize.value);
  context.lineTo(point.x, point.y);
  context.stroke();
});

function stopDrawing() { drawing = false; }
canvas.addEventListener('pointerup', stopDrawing);
canvas.addEventListener('pointercancel', stopDrawing);

function clearCanvas() {
  context.fillStyle = '#171717';
  context.fillRect(0, 0, canvas.width, canvas.height);
  resultEmpty.classList.remove('hidden');
  resultContent.classList.add('hidden');
  resultError.classList.add('hidden');
}

clearButton.addEventListener('click', clearCanvas);

recognizeButton.addEventListener('click', async () => {
  recognizeButton.disabled = true;
  recognizeButton.innerHTML = 'Reading…';
  resultError.classList.add('hidden');
  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: canvas.toDataURL('image/png') }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Prediction failed.');

    resultEmpty.classList.add('hidden');
    resultContent.classList.remove('hidden');
    predictionDigit.textContent = data.digit;
    const confidence = data.confidence * 100;
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceBar.style.width = `${confidence}%`;
    probabilityList.innerHTML = data.probabilities.map((value, digit) =>
      `<div class="probability-item"><strong>${(value * 100).toFixed(0)}%</strong>${digit}</div>`
    ).join('');
  } catch (error) {
    resultError.textContent = error.message;
    resultError.classList.remove('hidden');
  } finally {
    recognizeButton.disabled = false;
    recognizeButton.innerHTML = 'Recognize digit <span>→</span>';
  }
});
