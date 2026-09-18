# Roadmap

## Phase 1: Baseline Project

- Create a repeatable Python project structure.
- Load MNIST from `torchvision`.
- Train a small CNN.
- Save the best model checkpoint.
- Report loss and accuracy.

Success criteria:

- Training script runs end to end.
- Evaluation accuracy reaches at least `97%` on full MNIST.

## Phase 2: Training Quality

- Add validation split from the training set.
- Track best validation accuracy.
- Save metrics as JSON.
- Add configurable hyperparameters.

Success criteria:

- Training can be repeated with different batch sizes, learning rates, and epochs.
- Best checkpoint is selected by validation accuracy.

## Phase 3: Evaluation and Inference

- Evaluate the saved checkpoint on the test set.
- Predict from a local image file.
- Add confusion matrix reporting.

Success criteria:

- Test accuracy is stored in `reports/eval_metrics.json`.
- Single-image prediction works from the command line.

## Phase 4: Documentation and Packaging

- Document setup, data requirements, training, and evaluation.
- Add a short model card.
- Add tests for the model output shape and configuration loading.

Success criteria:

- A new user can clone/open the folder and train the model from the README alone.

## Phase 4.5: Browser Demo

- Serve the trained checkpoint through a local Flask app.
- Draw a digit with mouse or touch input.
- Preprocess, classify, and display confidence in the browser.

Status: complete for the baseline demo. A browser smoke test recognized a hand-drawn `7` as `7` at `99.7%` confidence.

## Phase 5: Improvements

Potential upgrades:

- Data augmentation with small rotations/translations.
- Learning rate scheduler.
- Early stopping.
- Experiment tracking with TensorBoard or Weights & Biases.
- Export to ONNX or TorchScript.
- Improve the web demo with example digits, prediction history, and optional model confidence warnings.
