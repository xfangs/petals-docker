from petals.constants import PUBLIC_INITIAL_PEERS

from data_structures import ModelInfo

INITIAL_PEERS = ['/ip4/172.31.10.237/tcp/31337/p2p/QmS9fCbG8shJJsGMUEKzE8spkwEhJLQdQd1RQp4qFsWoii']

MODELS = [
    ModelInfo(
        dht_prefix="Llama-2-13b-chat-hf",
        repository="https://huggingface.co/meta-llama/Llama-2-13b-chat-hf",
        num_blocks=40
    )
]

UPDATE_PERIOD = 60
