import pickle
import argparse
from backend import xp, to_cpu
from architectures.transformer import Transformer
from tokenization.bpetokenizer import BPETokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="examples/transformer_3.16M/tokenizer.json")
    parser.add_argument("--model", default="examples/transformer_3.16M/transformer_shakespeare.pkl")
    parser.add_argument("--prompt", default="ROMEO:")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--top_p", type=float, default=0.8)
    return parser.parse_args()


def filter_logits(logits, top_k, top_p):
    sorted_idx = xp.argsort(-logits)
    sorted_logits = logits[sorted_idx].copy()

    # Top-K
    if top_k > 0 and top_k < sorted_logits.shape[0]:
        sorted_logits[top_k:] = -xp.inf

    # Top-P
    if top_p < 1.0:
        shifted = sorted_logits - xp.max(sorted_logits)
        probs = xp.exp(shifted)
        probs /= xp.sum(probs)
        cumsum = xp.cumsum(probs)
        mask = cumsum > top_p
        mask[0] = False
        sorted_logits[mask] = -xp.inf

    filtered = xp.full_like(logits, -xp.inf)
    filtered[sorted_idx] = sorted_logits
    return filtered


def sample_from_logits(logits, temp):
    # Temperature
    if temp <= 0:
        return int(xp.argmax(logits))

    logits = logits / temp
    logits = logits - xp.max(logits)
    exp = xp.exp(logits)
    probs = exp / xp.sum(exp)
    cdf = xp.cumsum(probs)
    r = xp.random.random()
    return int(xp.searchsorted(cdf, r))


def main():
    args = parse_args()

    tokenizer = BPETokenizer.load(args.tokenizer)

    with open(args.model, "rb") as f:
        state = pickle.load(f)
    model = Transformer.from_state_dict(state)
    model.eval()

    tokens = xp.array([tokenizer.encode(args.prompt)], dtype=xp.int64)
    printed_str = tokenizer.decode(to_cpu(tokens[0]).tolist())
    print(printed_str, end="", flush=True)

    max_ctx = len(model.freqs_cis)

    for _ in range(args.steps):
        inp = tokens if tokens.shape[1] <= max_ctx else tokens[:, -max_ctx:]
        logits = model.forward(inp)[0, -1]
        logits = filter_logits(logits, args.top_k, args.top_p)
        next_tok = sample_from_logits(logits, args.temp)

        tokens = xp.concatenate(
            [tokens, xp.array([[next_tok]], dtype=xp.int64)],
            axis=1
        )

        full_text = tokenizer.decode(to_cpu(tokens[0]).tolist())
        new_text = full_text[len(printed_str):]

        if new_text:
            print(new_text, end="", flush=True)
            printed_str = full_text



if __name__ == "__main__":
    main()
