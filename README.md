# 🖼️ CIFAR-10 Image Classification with Deep Learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 32×32 colour images, 10 classes), comparing three approaches of increasing complexity:

| Model | Idea |
|---|---|
| `mlp` | Baseline — flattened pixels → one dense layer |
| `cnn` | Custom VGG-style CNN with data augmentation, batch-norm and dropout |
| `resnet50` | Transfer learning — frozen ImageNet ResNet50 (images up-scaled 8× to 256×256) + a small classification head |

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

## Project structure

```
cifar10-image-classification/
├── src/cifar10_classifier.py   # training + evaluation for all three models
├── requirements.txt
└── LICENSE
```

## Getting started

```bash
git clone https://github.com/<your-username>/cifar10-image-classification.git
cd cifar10-image-classification
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

CIFAR-10 is downloaded automatically by Keras on first run.

## Usage

```bash
python src/cifar10_classifier.py --model all          # train and compare all three
python src/cifar10_classifier.py --model cnn --epochs 30
python src/cifar10_classifier.py --model resnet50 --batch-size 32
python src/cifar10_classifier.py --smoke-test         # 1 epoch on tiny random data (checks the setup, no download)
```

Options: `--model {mlp,cnn,resnet50,all}`, `--epochs N`, `--batch-size N`, `--output-dir DIR`, `--smoke-test`.

Outputs (in `outputs/`): a loss/accuracy curve per model (`*_history.png`), the trained models (`*.keras`) and `results.json` with test loss/accuracy.

> **Tip:** the ResNet50 model works on 256×256 inputs and is slow on CPU — use a GPU (or Google Colab) for the `resnet50` and `cnn` runs. Early stopping (patience 5) is enabled.

## Results

Run the script to produce your own numbers — they are written to `outputs/results.json`. As a rough guide, the MLP baseline typically lands roughly 40–50 % test accuracy, while the CNN and transfer-learning models score substantially higher.

## Ideas for improvement

- Fine-tune the top ResNet50 blocks after the head has converged.
- Add a confusion matrix and per-class report.
- Try learning-rate schedules or mixed precision.

## License

[MIT](LICENSE)
