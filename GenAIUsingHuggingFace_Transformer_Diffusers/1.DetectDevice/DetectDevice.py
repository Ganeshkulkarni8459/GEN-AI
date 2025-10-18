from genaibook.core import get_device

def main():
    device =  get_device()
    print(f"Detected device: {device}")

if __name__ == "__main__":
    main()