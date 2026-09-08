# Dataset sources and cache

All download URLs, exact byte counts and SHA-256 digests are in
[datasets.json](datasets.json). Downloads use HTTPS, verify both size and digest,
and atomically replace the destination only after verification. Cached files are
verified on each run. Interrupted downloads leave no accepted partial file; rerun
the command to retry. A failed download reports the error without substituting
synthetic data. No accounts, API keys or dataset frameworks are required.

## MNIST

The four compressed IDX files come from the OSSCI mirror used by the
[torchvision MNIST loader](https://github.com/pytorch/vision/blob/master/torchvision/datasets/mnist.py).
The example decodes them directly using Python gzip/struct and NumPy; torchvision
is not installed or imported. See the [MNIST dataset documentation](https://docs.pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html)
for the source dataset description.

The first run downloads 11,594,722 bytes in total. The cache defaults to
`artifacts/datasets/mnist`. There are 60,000 training and 10,000 test images.
The demo defaults to the first 10,000 training and 512 test images. Its held-out
metric is therefore a subset check, not full-test-set benchmark accuracy.
Use `--limit 60000 --eval-samples 10000` to include both complete splits.

## Tiny Shakespeare

The corpus is Andrej Karpathy's
[Tiny Shakespeare text](https://github.com/karpathy/char-rnn/blob/master/data/tinyshakespeare/input.txt),
also used in the [nanoGPT Shakespeare example](https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare/prepare.py).
The pinned file contains 1,115,394 bytes and is cached as
`artifacts/datasets/language-model/tinyshakespeare.txt`. Although the URL uses the
upstream branch, the checksum pins the accepted content. Upstream changes require
an explicit manifest update; they cannot silently change an experiment.

The first 90% of characters form the training split; the last 10% form validation.
The vocabulary is learned from the training split only. Unseen validation/prompt
characters map to an explicit unknown token. Windows and next-character targets
stay inside their respective split. `--text path/to/file.txt` selects your own
UTF-8 corpus, up to 32 MiB, with the same splitting rules.

## Generated data

Segmentation produces seeded circles, ellipses and rectangles on noisy RGB
backgrounds with exact binary foreground masks. Independent seeds produce
training and validation sets; no image files or network access are needed to
construct the data.

MNIST's `--synthetic` mode renders noisy seven-segment digits. The language
model's `--synthetic` mode uses repeated generated sentences. These modes are
explicit offline execution checks and are recorded as synthetic in metrics and
checkpoints.

## Offline reuse

Pass `--data-dir /path/to/cache` to either downloading example. To provision an
offline machine, copy the verified cache from an online run, preserving filenames.
The manifest also allows downloading the files manually. Keep dataset source
attribution with any redistributed data; this repository contains loaders and
generators, not copies of the downloaded corpora.
