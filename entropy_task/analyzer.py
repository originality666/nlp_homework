"""
analyzer.py:
Calculate character probability and Shannon entropy for collected texts, and simulate different scales (e.g., 2M/5M characters).
"""

import os          # For file system operations
import json        # For handling JSON data
import math        # For mathematical operations (logarithm)
import collections # For counting character frequencies
import argparse    # For parsing command-line arguments
import matplotlib.pyplot as plt  # For plotting the entropy curve

def load_corpus(folder, text_type="english"):
    """
    Load and concatenate ONLY .txt files in a folder (matching text_type prefix).
    Filter empty text and print warnings for invalid files.
    
    Args:
        folder (str): Path to the folder containing text files.
        text_type (str): Prefix of target .txt files (e.g., "english" for "english_1.txt").
        
    Returns:
        str: Concatenated non-empty text (empty string if no valid files).
    """
    texts = []
    # Get all files in folder (skip directories)
    for fn in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, fn)
        # Skip directories, only process .txt files starting with text_type
        if os.path.isdir(file_path) or not (fn.endswith(".txt") and fn.startswith(text_type)):
            continue
        
        # Read file with error handling
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()  # 去除首尾空白（避免全是空格的文本）
                if content:  # 仅保留非空文本
                    texts.append(content)
                else:
                    print(f"Warning: Empty text in {fn}, skipped.")
        except Exception as e:
            print(f"Warning: Failed to read {fn} (error: {str(e)}), skipped.")
    
    # Join non-empty texts with newlines
    base_text = "\n".join(texts)
    # Warn if no valid text loaded
    if not base_text:
        print(f"Warning: No valid .txt files (starting with '{text_type}') found in {folder}.")
    return base_text

def char_entropy(text):
    """
    Calculate Shannon entropy with exception handling for empty text.
    
    Args:
        text (str): Input text to analyze.
        
    Returns:
        tuple: (entropy value, character frequency counter, total character count)
               Returns (None, None, 0) if text is empty.
    """
    # Remove pure whitespace (avoid counting spaces as valid characters)
    clean_text = text.strip()
    if not clean_text:
        print("Error: Empty text input for entropy calculation.")
        return None, None, 0  # 返回None标识异常
    
    # Count character frequencies (exclude whitespace if needed; adjust based on demand)
    counts = collections.Counter(clean_text)
    total = sum(counts.values())
    # Calculate probability (avoid division by zero, though clean_text is non-empty)
    probs = [cnt / total for cnt in counts.values()]
    # Compute Shannon entropy
    entropy = -sum(p * math.log2(p) for p in probs if p > 0)
    return entropy, counts, total

def expand_text(text, target_size):
    """
    Expand a text to target size (skip if text is empty).
    
    Args:
        text (str): Original non-empty text to expand.
        target_size (int): Desired total length of the expanded text.
        
    Returns:
        str: Expanded text (empty string if input is empty).
    """
    clean_text = text.strip()
    if not clean_text or target_size <= 0:
        print("Error: Invalid input for text expansion (empty text or negative target size).")
        return ""
    
    # Calculate repeat times (avoid redundant copies)
    text_len = len(clean_text)
    repeat = target_size // text_len
    remainder = target_size % text_len
    # Expand and truncate to exact target size
    expanded_text = (clean_text * repeat) + clean_text[:remainder]
    return expanded_text

def analyze(folder, scales, text_type="english"):
    """
    Analyze entropy across scales with validation for base text.
    
    Args:
        folder (str): Folder containing source .txt files.
        scales (list[int]): List of target text lengths (in characters).
        text_type (str): Prefix of target .txt files.
        
    Returns:
        list[dict]: Analysis results (empty list if no valid base text).
    """
    # Load valid base text
    base_text = load_corpus(folder, text_type)
    if not base_text:
        print("Error: No valid base text to analyze.")
        return []
    
    results = []
    for s in scales:
        # Skip invalid scale (negative/zero)
        if s <= 0:
            print(f"Warning: Invalid scale {s} (must be positive), skipped.")
            continue
        
        # Expand text to target scale
        expanded_text = expand_text(base_text, s)
        if len(expanded_text) != s:
            print(f"Warning: Failed to expand to {s} chars (actual: {len(expanded_text)}), skipped.")
            continue
        
        # Calculate entropy (skip if text is empty)
        entropy, counts, total = char_entropy(expanded_text)
        if entropy is None:
            continue
        
        # Store results
        results.append({
            "scale": s,
            "entropy": round(entropy, 4),  # 保留4位小数，避免冗余
            "unique_chars": len(counts),
            "total_chars": total
        })
        # Print clear progress
        print(f"[{text_type}] Scale: {s:,} chars -> Entropy={entropy:.4f}, Unique chars={len(counts)}")
    return results

def plot_curve(results, text_type, out_path="entropy_curve.png"):
    """
    Plot entropy curve with correct path and validation.
    
    Args:
        results (list[dict]): Analysis results from `analyze` function.
        text_type (str): Text type (for filename and title).
        out_path (str): Base path to save the output image.
    """
    if not results:
        print("Error: No results to plot (empty results list).")
        return
    
    # Prepare data (scale in million chars)
    x = [r["scale"] / 1e6 for r in results]
    y = [r["entropy"] for r in results]
    
    # Create plot
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, marker="o", color="#2E86AB", linewidth=2)  # 美化颜色和线条
    plt.xlabel("Text Scale (Million Characters)")
    plt.ylabel("Shannon Entropy (bits/char)")
    plt.title(f"{text_type.capitalize()} Text: Scale vs Shannon Entropy")  # 标题含文本类型
    plt.grid(True, alpha=0.3)  # 透明网格，不遮挡线条
    plt.tight_layout()  # 自动调整布局，避免标签截断
    
    # Save plot with correct path
    full_out_path = f"results/{text_type}_{out_path}"
    plt.savefig(full_out_path, dpi=200)
    print(f"Entropy curve saved to: {full_out_path}")  # 打印实际路径

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser(description="Calculate text entropy across different scales.")
    parser.add_argument(
        "--input_dir", 
        default="collected_samples", 
        help="Directory containing input .txt files (default: 'collected_samples')"
    )
    parser.add_argument(
        "--out", 
        default="results.json", 
        help="Path to save analysis results (default: 'results.json')"
    )
    parser.add_argument(
        "--scales", 
        nargs="+", 
        type=int, 
        default=[2000000, 5000000], 
        help="List of target text scales (in chars, default: 2000000 5000000)"
    )
    parser.add_argument(
        "--type", 
        default="english",
        type=str,
        help="Type of text (prefix of .txt files, e.g., 'english' for 'english_1.txt'; default: 'english')"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist.")
        exit(1)  # 终止程序，避免后续错误
    
    # Run analysis
    results = analyze(args.input_dir, args.scales, args.type)
    
    # Save results (only if results are valid)
    if results:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Analysis results saved to: {args.out}")
    else:
        print("Warning: No valid results to save.")
    
    # Generate plot
    plot_curve(results, args.type)