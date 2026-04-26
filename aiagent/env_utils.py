import os


def get_eth_rpc_url() -> str:
    rpc_url = os.getenv("ETH_RPC_URL")
    if not rpc_url or not rpc_url.strip():
        raise RuntimeError(
            "Environment variable ETH_RPC_URL not found. "
            "Please set it to an Ethereum JSON-RPC endpoint."
        )
    return rpc_url.strip()


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Environment variable OPENAI_API_KEY not found. "
            "Please set it in your system or current shell session."
        )

    for idx, ch in enumerate(api_key):
        if ord(ch) > 127:
            raise RuntimeError(
                "OPENAI_API_KEY contains a non-ASCII character at position "
                f"{idx}. Please re-copy your API key from OpenAI dashboard."
            )

    return api_key
