import os
import argparse
import urllib.request
import pickle
import json
import backend as xp
from backend import to_cpu
from architectures.transformer import Transformer
from modules.optimizers import AdamW
from modules.schedulers import CosineAnnealingLR
from tokenization.bpetokenizer import BPETokenizer
from modules.losses import CrossEntropyLoss
from modules.gradient_clipping import clip_grad_norm
from data_utils.dataloader import DataLoader
from data_utils.dataset import ArrayDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--query_heads", type=int, default=8)
    parser.add_argument("--key_value_heads", type=int, default=4)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--max_seq_len", type=int, default=512) # Max sequence length for inference, that's the length that we precompute freqs_cis for
    parser.add_argument("--training_seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--eta_min", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--data_path", type=str, default="shakespeare.txt")
    parser.add_argument("--tokenizer_path", type=str, default="tokenizer.json")
    parser.add_argument("--data_url", type=str, default="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt")
    parser.add_argument("--logs_path", type=str, default="logs.json")
    parser.add_argument("--save_path", type=str, default="transformer_shakespeare.pkl")
    return parser.parse_args()

def get_data_and_tokenizer(args):
    if not os.path.exists(args.data_path):
        print("Downloading data from url")
        urllib.request.urlretrieve(args.data_url, args.data_path)
    
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = BPETokenizer(vocab_size=args.vocab_size)
    if os.path.exists(args.tokenizer_path):
        print("Loading tokenizer")
        tokenizer = BPETokenizer.load(args.tokenizer_path)
        if tokenizer.vocab_size != args.vocab_size:
            raise ValueError(f"Tokenizer vocab size {tokenizer.vocab_size} doesn't match requested {args.vocab_size}.")

    else:
        print("Training BPE Tokenizer")
        tokenizer.train(text)
        tokenizer.save(args.tokenizer_path)

    data = xp.array(tokenizer.encode(text), dtype=xp.int32)

    num_sequences = len(data) // (args.training_seq_len + 1)
    data = data[:num_sequences * (args.training_seq_len + 1)] 

    data = data.reshape(num_sequences, args.training_seq_len + 1)
    X = data[:, :-1] 
    y = data[:, 1:]   
    return X, y, tokenizer

def main():
    args = parse_args()
    X, y, tokenizer = get_data_and_tokenizer(args)
    
    model = Transformer(args.vocab_size, args.embed_dim, args.query_heads, args.key_value_heads, 
                        hidden_dim=args.hidden_dim, layers=args.layers, max_seq_len=args.max_seq_len, dropout_rate=args.dropout_rate)
    
    def count_params(model):
        total = 0
        for param in model.params:
            total += param.size
        return total

    print(f"Total parameters: {count_params(model):,}")

    optimizer = AdamW(model.params, lr=args.lr, weight_decay=args.weight_decay)
    
    dataset = ArrayDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    
    total_steps = args.epochs * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.eta_min)
    criterion = CrossEntropyLoss()
    
    print(f"Epoch count: {args.epochs} | Vocab: {args.vocab_size} | Batches per epoch: {len(dataloader)} | Total steps: {total_steps}")

    training_history = {
        'steps': [],
        'losses': [],
        'epochs': [],
        'learning_rates': []
    }

    step = 0
    model.train() # It's in training mode by default so this line is redundant, but let's keep it for clarity
    for epoch in range(args.epochs):
        for batch_idx, batch in enumerate(dataloader):
            xb, yb = batch
            
            # Forward Pass
            logits = model.forward(xb)
            loss = criterion(logits, yb)
            
            # Backward Pass
            d_logits = criterion.backward()
            model.backward(d_logits)
            
            # Optimization
            clip_grad_norm(model.grads, max_norm=args.max_grad_norm)
            optimizer.step(model.grads)
            optimizer.zero_grad(model.grads) # Grads are reset every backward pass in current implementation so this is not needed for now
            scheduler.step()
            step += 1

            if step % args.log_interval == 0 or step == 1:
                epoch_progress = epoch + (batch_idx + 1) / len(dataloader)
                print(f"Step {step}/{total_steps}, Loss: {loss:.4f}, LR: {scheduler.lr:.6f}, Epoch: {epoch_progress:.2f}")

                if args.logs_path is not None:
                    training_history['steps'].append(step)
                    training_history['losses'].append(loss.item())
                    training_history['epochs'].append(epoch_progress)
                    training_history['learning_rates'].append(scheduler.lr)

    if step % args.log_interval != 0: # Save last step if it's not a multiple of log_interval
        epoch_progress = args.epochs
        if args.logs_path is not None:
            training_history['steps'].append(step)
            training_history['losses'].append(loss.item())
            training_history['epochs'].append(epoch_progress)
            training_history['learning_rates'].append(scheduler.lr)

    print("\nTraining complete. Saving model...")
    # Save model
    with open(args.save_path, "wb") as f:
        pickle.dump(model.state_dict(complete=True), f) # We use complete=True to save the config as well
    
    # Save training history
    if args.logs_path is not None:
        with open(args.logs_path, "w") as f:
            json.dump(training_history, f, indent=2)

if __name__ == "__main__":
    main()