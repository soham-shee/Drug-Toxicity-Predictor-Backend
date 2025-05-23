
## Installation

Install the dependencies.

```bash
    python -m venv venv
    source venv/bin/activate  # Windows: venv\Scripts\activate

    # Install requirements
    pip install -r requirements.txt

    # If you have git-lfs skip this part
    # For Ubuntu/Debian
    sudo apt install git-lfs

    # For macOS (with Homebrew)
    brew install git-lfs

    # Then initialize
    git lfs install

    git lfs pull # to pull large files(.h5)-model
```

## Development

Starting the server (http://localhost:8000/predict)

```bash
    # Run FastAPI
    uvicorn main:app --reload
```
    