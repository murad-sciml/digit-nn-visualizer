"""
Retrain the network and save its weights to models/digits_mlp.npz.

    python train.py              # with data augmentation (recommended for drawing)
    python train.py --no-augment # plain dataset only (higher test score, worse on drawings)
"""
import argparse

from nn_viz.model import train

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-augment", action="store_true", help="train without data augmentation")
    ap.add_argument("--copies", type=int, default=4, help="augmented copies per training image")
    args = ap.parse_args()
    train(save=True, augment_copies=0 if args.no_augment else args.copies)
