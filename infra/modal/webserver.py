"""Michael Feil, MIT License, 2024-06-17

This is a reference implementation for infinity server via CLI. 

"""
import subprocess
import os
import pathlib
from modal import Image, Secret, App, gpu, web_server

# CONFIG.
# Writing the configuration to a .env file to have the example in one file.
PORT = 7997
INFINITY_PIPY_VERSION = "0.0.45"
pathlib.Path(".env").write_text(
    f"""
# Auto-generated by webserver.py
# Per model args, padded by `;`
INFINITY_MODEL_ID="jinaai/jina-clip-v1;michaelfeil/bge-small-en-v1.5;mixedbread-ai/mxbai-rerank-xsmall-v1;philschmid/tiny-bert-sst2-distilled;"
INFINITY_REVISION="1cbe5e8b11ea3728df0b610d5453dfe739804aa9;ab7b31bd10f9bfbb915a28662ec4726b06c6552a;1d1adfbd0fde63df646402cf33e157e5852ead3;
INFINITY_MODEL_WARMUP="false;false;false;false;"
INFINITY_BATCH_SIZE="8;8;8;8;"
# One-off args
INFINITY_QUEUE_SIZE="1024"
INFINITY_PORT={PORT}
INFINITY_API_KEY=""
"""
)
CMD = f"infinity_emb v2"


def download_models():
    """downloads the models into the docker container at build time.
    Ensures no downtime when huggingface is down.
    """
    print(f"downloading models {os.environ.get('INFINITY_MODEL_ID')}")
    process = subprocess.Popen(CMD + " " + "--preload-only", shell=True)
    exit_code = process.wait()
    print(f"downloading models done.")
    assert exit_code == 0, f"Failed to download models. Exit code: {exit_code}"


# ### Image definition
# We'll start from a recommended Docker Hub image and install `infinity`.
image = (
    Image.from_registry("nvidia/cuda:12.1.1-devel-ubuntu22.04", add_python="3.10")
    .pip_install(
        f"infinity_emb[torch,vision,optimum,einops,cache,logging,server]=={INFINITY_PIPY_VERSION}"
    )
    .run_function(
        download_models,
        secrets=[
            Secret.from_dotenv(),
        ],
        timeout=60 * 20,
    )
)

app = App("infinity", image=image)
GPU_CONFIG = gpu.T4(count=1)


# Run a web server on port 7997 and expose the Infinity embedding server
@app.function(
    allow_concurrent_inputs=500,
    container_idle_timeout=30,
    gpu=GPU_CONFIG,
    secrets=[
        Secret.from_dotenv(),
    ],
    name="serve",
)
@web_server(PORT, startup_timeout=300, custom_domains=["infinity.modal.michaelfeil.eu"])
def serve():
    subprocess.Popen(CMD, shell=True)