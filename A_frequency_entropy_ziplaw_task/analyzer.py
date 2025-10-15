"""
analyzer.py:
Advanced text analyzer — clean collected texts, compute character- and word-level entropy,
and verify Zipf's law on English and Chinese corpora with scalable sampling.
"""

import os
import re
import json
import math
import jieba
import argparse
import collections
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator
import regex as re

# =============================
# 1. Text Loading & Cleaning
# =============================
def clean_text(text, lang="english"):
    """
    Clean text: remove punctuation, digits, and garbled characters.
    Normalize spaces and strip whitespace.
    """
    if not text:
        return ""
    text = text.strip()

    if lang == "chinese":
        # Remove punctuation, Latin letters, digits, and control chars
        text = re.sub(r"[a-zA-Z0-9\s\p{P}]+", "", text)
        # Remove any other non-Chinese character (keep Han)
        text = re.sub(r"[^\u4e00-\u9fa5]", "", text)
    else:  # English
        # Keep only a–z and space
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text)  # Normalize spaces
        text = text.lower()

    return text.strip()


def load_corpus(folder, text_type="english"):
    """Load, clean, and concatenate .txt files."""
    texts = []
    for fn in sorted(os.listdir(folder)):
        if not fn.endswith(".txt") or not fn.startswith(text_type):
            continue
        path = os.path.join(folder, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
                clean = clean_text(content, text_type)
                if clean:
                    texts.append(clean)
                else:
                    print(f"Warning: {fn} cleaned to empty text, skipped.")
        except Exception as e:
            print(f"Warning: failed to read {fn} ({str(e)}).")

    joined = "\n".join(texts)
    if not joined:
        print(f"Error: No valid texts found for '{text_type}'.")
    return joined


# =============================
# 2. Entropy Calculation
# =============================
def calc_entropy(tokens):
    """Compute Shannon entropy from a list of tokens."""
    if not tokens:
        return None, 0
    counter = collections.Counter(tokens)
    total = sum(counter.values())
    probs = [v / total for v in counter.values()]
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy, counter


def segment_text(text, lang="english"):
    """Tokenize text for word-level analysis."""
    if lang == "chinese":
        words = list(jieba.cut(text))
        words = [w for w in words if w.strip()]
    else:
        words = text.split()
    return words


# =============================
# 3. Zipf’s Law Analysis
# =============================
def plot_zipf(counter, lang, scale_label, out_dir="results"):
    """Plot rank-frequency curve in log-log scale."""
    os.makedirs(out_dir, exist_ok=True)

    sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    freqs = [v for _, v in sorted_items]
    ranks = range(1, len(freqs) + 1)

    plt.figure(figsize=(5, 4))
    plt.loglog(ranks, freqs, marker=".", linestyle="none")
    plt.title(f"Zipf's Law ({lang}, {scale_label})")
    plt.xlabel("Rank (log)")
    plt.ylabel("Frequency (log)")
    plt.grid(True, which="both", linestyle="--", alpha=0.4)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"zipf_{lang}_{scale_label}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Zipf plot saved to: {out_path}")

    # Estimate slope (α)
    if len(freqs) > 10:
        import numpy as np
        log_rank = np.log(ranks[:100])
        log_freq = np.log(freqs[:100])
        slope, _ = np.polyfit(log_rank, log_freq, 1)
        print(f"[{lang}] Zipf slope ≈ {slope:.2f} (expected ≈ -1.0)")


# =============================
# 4. Analysis Across Scales
# =============================
def expand_text(text, size):
    """Repeat text to reach desired size."""
    if not text:
        return ""
    text = text.strip()
    n = len(text)
    repeat = size // n
    rem = size % n
    return (text * repeat) + text[:rem]


def analyze(folder, scales, text_type="english", out_dir="results"):
    """Perform full entropy and Zipf analysis across scales."""
    base_text = load_corpus(folder, text_type)
    if not base_text:
        return []

    os.makedirs(out_dir, exist_ok=True)
    results = []

    for s in scales:
        expanded = expand_text(base_text, s)
        scale_label = f"{s//1_000_000}M"

        # Character-level entropy
        chars = list(expanded)
        char_entropy, _ = calc_entropy(chars)

        # Word-level entropy
        words = segment_text(expanded, text_type)
        word_entropy, word_counter = calc_entropy(words)

        # Save Zipf plot (word-level)
        plot_zipf(word_counter, text_type, scale_label, out_dir)

        result = {
            "scale": s,
            "char_entropy": round(char_entropy, 4),
            "word_entropy": round(word_entropy, 4),
            "unique_chars": len(set(chars)),
            "unique_words": len(word_counter),
        }
        results.append(result)

        print(f"[{text_type}] Scale={s:,} chars "
              f"=> CharEntropy={char_entropy:.4f}, WordEntropy={word_entropy:.4f}, "
              f"UniqueWords={len(word_counter)}")

    # Plot entropy vs scale
    plt.figure(figsize=(6, 4))
    x = [r["scale"] / 1e6 for r in results]
    plt.plot(x, [r["char_entropy"] for r in results], "o-", label="Char Entropy")
    plt.plot(x, [r["word_entropy"] for r in results], "s-", label="Word Entropy")
    plt.xlabel("Text Scale (Million chars)")
    plt.ylabel("Entropy (bits)")
    plt.title(f"{text_type.capitalize()} Text Entropy vs Scale")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"entropy_curve_{text_type}.png"), dpi=200)
    plt.close()

    return results


# =============================
# 5. CLI Entry Point
# =============================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entropy and Zipf analysis tool.")
    parser.add_argument("--input_dir", default="collected_samples",
                        help="Folder containing input .txt files")
    parser.add_argument("--type", default="english", choices=["english", "chinese"],
                        help="Text type (english/chinese)")
    parser.add_argument("--scales", nargs="+", type=int,
                        default=[1_000_000, 2_000_000, 5_000_000],
                        help="List of target scales (in chars)")
    parser.add_argument("--out", default="results.json", help="Output JSON path")

    args = parser.parse_args()

    res = analyze(args.input_dir, args.scales, args.type)
    if res:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {args.out}")
