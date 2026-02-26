from train import train
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    print("Sequence Initiated..")
    model = train()

    print("Training complete.")


if __name__ == "__main__":

    main()