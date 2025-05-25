# DistilBERT Document Classification Model

This document explains how to set up, train, and use the DistilBERT transformer model for document classification in the Document Classifier system.

## Overview

The document classification pipeline now supports a DistilBERT-based model for improved classification accuracy over the previous Naive Bayes approach. DistilBERT is a smaller, faster transformer model that retains most of the performance of BERT while requiring fewer computational resources.

Benefits of the transformer-based approach:

- Better understanding of context and semantics in documents
- More accurate classification of complex documents
- Improved handling of nuanced language patterns
- Higher confidence scores on correctly classified documents

## Prerequisites

The DistilBERT model requires the following dependencies (already added to `requirements.txt`):

- `transformers==4.42.3`
- `torch==2.2.2`

## Training the Model

A new training script is provided to create and train the DistilBERT model on synthetic document data.

### Basic Training Command

```bash
python scripts/train_distilbert_model.py
```

This will:

- Generate 1000 synthetic document samples
- Train a DistilBERT model for 3 epochs
- Save the model to `datasets/distilbert_model/`

### Advanced Training Options

```bash
python scripts/train_distilbert_model.py --samples 3000 --epochs 5 --batch-size 8 --learning-rate 2e-5
```

| Parameter         | Description                             | Default   |
| ----------------- | --------------------------------------- | --------- |
| `--samples`       | Number of synthetic samples to generate | 1000      |
| `--output`        | Directory where model will be saved     | datasets/ |
| `--epochs`        | Number of training epochs               | 3         |
| `--batch-size`    | Training batch size                     | 16        |
| `--learning-rate` | Learning rate                           | 5e-5      |

## Model Directory Structure

After training, the model will be saved to `datasets/distilbert_model/` with the following structure:

```
distilbert_model/
├── config.json         # Model configuration with label mappings
├── pytorch_model.bin   # Model weights
├── special_tokens_map.json
├── tokenizer_config.json
└── vocab.txt          # Tokenizer vocabulary
```

## Classification Pipeline Integration

The model is integrated into the classification pipeline through the updated `src/classification/model.py` module. The public API remains unchanged, ensuring compatibility with the existing classification stages:

```python
from src.classification.model import predict

# Get document label and confidence
label, confidence = predict(text)
```

## Performance Considerations

- The DistilBERT model is significantly larger than the previous Naive Bayes model (typically 250-300MB vs <1MB)
- It is lazily loaded on first use to minimize startup latency
- The first classification will be slower as the model is loaded into memory
- Consider GPU acceleration for production deployments with high throughput requirements

## Troubleshooting

If you encounter errors related to the model:

1. **Model Not Found**: Ensure you've run the training script and the model directory exists at `datasets/distilbert_model/`

2. **CUDA Out of Memory**: Reduce batch size during training with `--batch-size 4` or use CPU-only training by setting the environment variable:

   ```
   export CUDA_VISIBLE_DEVICES=""
   ```

3. **Slow Inference**: Consider quantizing the model for production or using a CPU-optimized inference setup if GPU is not available

## Reverting to Naive Bayes

To revert to the previous Naive Bayes model, you can run the original training script:

```bash
python scripts/train_model.py
```

This will generate the `model.pkl` file expected by the previous implementation.
