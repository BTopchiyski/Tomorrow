import argparse
from tomorrow.train_models import main as train_models
from tomorrow.transform_raw_data import main as transform_data

def main():
    parser = argparse.ArgumentParser(description="Tomorrow Project")
    parser.add_argument("operation", choices=["train", "transform"], help="Choose an operation to perform.")
    args = parser.parse_args()

    if args.operation == "train":
        train_models()
    elif args.operation == "transform":
        transform_data()

if __name__ == "__main__":
    main()