import torch

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.randn(3, 3, device=device)
    print("torch:", torch.__version__)
    print("device:", device)
    print(x @ x)

if __name__ == "__main__":
    main()
