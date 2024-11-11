import os

def setup_environment():
    print("Setting up the environment...")
    os.system("pip install -r requirements.txt")
    print("Environment setup complete.")

if __name__ == "__main__":
    setup_environment()
