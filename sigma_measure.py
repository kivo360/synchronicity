#!/usr/bin/env python3
"""
σ Measurement Tool — measure mechanism efficiency for any LLM.

σ = H(p) / H(p,q)

Where:
  H(p)   = intrinsic entropy of the text (compression lower bound in bits)
  H(p,q) = model's cross-entropy on that text (from logprobs)

This tool sends text from different domains to an LLM API, gets the
logprobs, and computes σ for each domain.

A model with σ=0.95 on "code" is highly efficient at predicting code.
A model with σ=0.60 on "legal" wastes 40% of its cognitive energy on legal text.

The σ profile across domains reveals what a model is specialized for.

API: OpenAI-compatible (Z.ai, OpenAI, Ollama, vLLM, etc.)
Requires: logprobs support in the API.
"""

import os
import sys
import math
import json
import time
import argparse
from collections import Counter
from typing import Optional

import requests

# ═══════════════════════════════════════════════════════════════════════
#  Domain text samples — short, representative samples from different
#  domains. These are used to measure σ per domain.
# ═══════════════════════════════════════════════════════════════════════

DOMAIN_SAMPLES = {
    "code_python": """
def fibonacci(n: int) -> list[int]:
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return fib

class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert(self, value):
        if value < self.value:
            if self.left is None:
                self.left = BinaryTree(value)
            else:
                self.left.insert(value)
        else:
            if self.right is None:
                self.right = BinaryTree(value)
            else:
                self.right.insert(value)
""".strip(),

    "code_javascript": """
const debounce = (fn, delay) => {
    let timeoutId;
    return function(...args) {
        clearTimeout(timeoutId);
        timeoutId = setTimeout(() => fn.apply(this, args), delay);
    };
};

class Event Emitter {
    constructor() {
        this.events = {};
    }

    on(event, listener) {
        (this.events[event] || (this.events[event] = [])).push(listener);
    }

    emit(event, ...args) {
        (this.events[event] || []).forEach(listener => listener(...args));
    }
}
""".strip(),

    "legal": """
WHEREAS, the Party of the First Part (hereinafter "Licensor") is the sole
and exclusive owner of certain intellectual property rights, including but
not limited to copyrights, trademarks, and trade secrets associated with
the software application known as "Platform X"; and WHEREAS, the Party of
the Second Part (hereinafter "Licensee") desires to obtain a non-exclusive,
non-transferable, non-sublicensable license to use said software in
accordance with the terms and conditions hereinafter set forth; NOW,
THEREFORE, in consideration of the mutual covenants and agreements
contained herein, the parties agree as follows.
""".strip(),

    "medical": """
The patient presented with acute onset of pleuritic chest pain, dyspnea,
and hemoptysis. Physical examination revealed tachycardia (HR 112),
tachypnea (RR 24), and decreased breath sounds in the right lower lobe.
D-dimer was elevated at 4.8 mg/L. CT pulmonary angiography demonstrated
a segmental pulmonary embolism in the right lower lobe pulmonary artery.
Anticoagulation therapy was initiated with apixaban 10 mg twice daily
for the first seven days, followed by 5 mg twice daily thereafter.
""".strip(),

    "business": """
Q3 revenue grew 18% year-over-year to $4.2 billion, driven primarily by
strong performance in the cloud infrastructure segment, which saw a 34%
increase in ARR. Gross margin expanded 220 basis points to 68.5%,
reflecting operational efficiencies and favorable product mix. Operating
income was $840 million, with an operating margin of 20%. The company
raised its full-year guidance, projecting revenue of $16.8-$17.0 billion
and adjusted EPS of $7.20-$7.40.
""".strip(),

    "general_prose": """
The old lighthouse stood at the edge of the cliff, its beam sweeping
across the dark water like a searching eye. Maria pulled her jacket
tighter against the wind and climbed the spiral staircase for the last
time. Forty years she had kept the light, through storms and calm nights
alike. Now the automated system would take over, and the lighthouse
would become a museum. She paused at the top, looking out at the horizon
where the sea met the sky, and let the moment settle.
""".strip(),

    "mathematics": """
Let f: R^n → R be a twice continuously differentiable function. A point
x* is a local minimum if and only if ∇f(x*) = 0 and the Hessian matrix
H(x*) is positive semidefinite. The method of Lagrange multipliers
states that at a constrained local minimum of f(x) subject to g(x) = 0,
there exists a scalar λ such that ∇f(x*) = λ∇g(x*). The Karush-Kuhn-
Tucker conditions generalize this to inequality constraints.
""".strip(),

    "scientific": """
CRISPR-Cas9 is a genome editing tool derived from the adaptive immune
system of bacteria. The system consists of a guide RNA (gRNA) that
complements the target DNA sequence and the Cas9 nuclease, which creates
double-strand breaks at the targeted location. The cell's natural DNA
repair mechanisms then introduce insertions, deletions, or homology-
directed repair, enabling precise genetic modifications with high
efficiency and specificity across diverse organisms.
""".strip(),
}


# ═══════════════════════════════════════════════════════════════════════
#  σ Computation
# ═══════════════════════════════════════════════════════════════════════

def compute_entropy_huffman(text: str) -> float:
    """Estimate H(p) — the intrinsic entropy of the text.

    Uses compression-based estimation: the entropy of the text is
    approximately the Shannon entropy of its character distribution.

    This is a lower bound on H(p) — the actual intrinsic difficulty
    could be higher because structure beyond character frequency
    (grammar, semantics) also carries information.

    For σ computation, what matters is the RATIO H(p)/H(p,q), so
    as long as H(p) is estimated consistently across models, the
    comparison is valid.
    """
    if not text:
        return 0.0

    # Shannon entropy of character distribution (bits per character)
    freq = Counter(text)
    total = len(text)
    entropy = 0.0
    for count in freq.values():
        p = count / total
        entropy -= p * math.log2(p)

    # Convert to bits per token (approximate: average token ≈ 4 chars)
    # This is rough but consistent across models
    bits_per_token = entropy * 4.0

    return bits_per_token


def compute_cross_entropy_from_logprobs(logprobs_data: list) -> float:
    """Compute H(p,q) from API logprobs.

    logprobs_data: list of {token, logprob} dicts from the API.
    Returns average cross-entropy in bits per token.
    """
    if not logprobs_data:
        return 0.0

    total_neg_logprob = 0.0
    count = 0
    for entry in logprobs_data:
        if entry.get("logprob") is not None:
            # logprob is natural log → convert to bits
            total_neg_logprob += -entry["logprob"] / math.log(2)
            count += 1

    return total_neg_logprob / max(count, 1)


def compute_sigma(h_p: float, h_pq: float) -> float:
    """σ = H(p) / H(p,q)."""
    if h_pq <= 0:
        return 1.0
    sigma = h_p / h_pq
    return max(0.01, min(sigma, 0.999))


# ═══════════════════════════════════════════════════════════════════════
#  API Client (OpenAI-compatible)
# ═══════════════════════════════════════════════════════════════════════

def get_logprobs(
    text: str,
    model: str,
    base_url: str,
    api_key: str,
    max_tokens_to_measure: int = 200,
) -> Optional[list]:
    """Send text to the API and get logprobs for each token.

    We send the text as the prompt and ask for continuation with logprobs.
    The logprobs of the CONTINUATION tokens tell us how well the model
    predicts the text — that IS the cross-entropy.

    Strategy: use the "echo" approach — send a prefix and have the model
    complete the rest, measuring logprobs on the completion tokens.
    """
    # Split text: use first half as context, second half as target
    tokens = text.split()
    if len(tokens) < 10:
        return None

    mid = len(tokens) // 3  # 1/3 context, 2/3 target
    context = " ".join(tokens[:mid])
    target = " ".join(tokens[mid:])

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Continue the following text exactly. Do not add commentary."},
            {"role": "user", "content": f"Continue this text:\n\n{context}\n\nTarget continuation:\n{target[:500]}"}
        ],
        "max_tokens": 5,
        "temperature": 0,
        "logprobs": True,
        "top_logprobs": 1,
    }

    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        # Extract logprobs from response
        choices = data.get("choices", [])
        if not choices:
            return None

        logprobs_content = choices[0].get("logprobs", {})
        if isinstance(logprobs_content, dict):
            token_logprobs = logprobs_content.get("content", [])
        elif isinstance(logprobs_content, list):
            token_logprobs = logprobs_content
        else:
            # Some APIs return logprobs in a different structure
            return None

        result = []
        for entry in token_logprobs:
            if isinstance(entry, dict):
                result.append({
                    "token": entry.get("token", ""),
                    "logprob": entry.get("logprob"),
                })

        return result if result else None

    except Exception as e:
        print(f"  API error: {e}")
        return None


def measure_sigma_behavioral(
    model: str,
    base_url: str,
    api_key: str,
    domains: list = None,
) -> dict:
    """Measure σ using behavioral prediction accuracy.

    Since many APIs (including Z.ai) don't return logprobs, we use a
    behavioral proxy: give the model the first N tokens of domain text
    and ask it to predict the next token. Score how close its prediction
    is to the actual text using BLEU/character overlap.

    Higher accuracy → lower effective cross-entropy → higher σ.

    This is a RANK-PRESERVING proxy: if model A beats model B on this
    metric, model A has lower H(p,q) on this domain, hence higher σ.
    """
    if domains is None:
        domains = list(DOMAIN_SAMPLES.keys())

    profile = {}

    for domain in domains:
        text = DOMAIN_SAMPLES.get(domain, "")
        if not text:
            continue

        tokens = text.split()
        if len(tokens) < 15:
            continue

        # Give model 40% of tokens as context, ask for next 20 tokens
        context_end = int(len(tokens) * 0.4)
        target_start = context_end
        target_end = min(target_start + 20, len(tokens))
        target = " ".join(tokens[target_start:target_end])
        context = " ".join(tokens[:context_end])

        # H(p) — intrinsic entropy
        h_p = compute_entropy_huffman(text)

        # Ask model to predict the continuation
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a text continuation engine. Given text, output ONLY the exact next words that would follow. No commentary, no explanations, just the continuation."},
                {"role": "user", "content": f"Continue this text exactly (output the next ~20 words):\n\n{context}\n"},
            ],
            "max_tokens": 60,
            "temperature": 0,
        }

        try:
            resp = requests.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            completion = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not completion:
                completion = data.get("choices", [{}])[0].get("message", {}).get("reasoning_content", "")

            # Measure prediction accuracy: character-level overlap
            accuracy = char_overlap(completion.lower(), target.lower())

            # Convert accuracy to σ proxy:
            # accuracy=1.0 → σ=0.99 (perfect prediction)
            # accuracy=0.3 → σ=0.50 (poor prediction)
            # accuracy=0.0 → σ=0.10 (no overlap)
            sigma = 0.10 + accuracy * 0.89

            T = 1.0 / (1.0 - sigma) if sigma < 0.999 else 1000.0

            # Estimate H(p,q) from σ
            h_pq = h_p / sigma if sigma > 0.01 else h_p * 10

            profile[domain] = {
                "h_p": round(h_p, 4),
                "h_pq": round(h_pq, 4),
                "sigma": round(sigma, 4),
                "temperature": round(T, 2),
                "accuracy": round(accuracy, 4),
                "prediction_sample": completion[:80],
                "target_sample": target[:80],
            }

        except Exception as e:
            print(f"  Error measuring {domain}: {e}")
            continue

    return profile


def char_overlap(text1: str, text2: str) -> float:
    """Compute character-level overlap between two texts.

    Uses a normalized Levenshtein-like ratio: the fraction of characters
    that match in order, accounting for insertions/deletions.

    Returns: 0.0 (no overlap) to 1.0 (identical)
    """
    if not text1 or not text2:
        return 0.0

    # Simple approach: normalize whitespace, compute sequence matching
    import difflib
    ratio = difflib.SequenceMatcher(None, text1, text2).ratio()
    return ratio


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Measure σ for LLMs")
    parser.add_argument("--model", default="glm-5.2",
                       help="Model name (default: glm-5.2)")
    parser.add_argument("--base-url", default="https://api.z.ai/api/coding/paas/v4",
                       help="API base URL")
    parser.add_argument("--api-key", default=None,
                       help="API key (or set ZAI_API_KEY)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple models")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ZAI_API_KEY") or os.environ.get("GLM_API_KEY")
    if not api_key:
        print("Error: no API key. Set ZAI_API_KEY or use --api-key")
        sys.exit(1)

    if args.compare:
        models = [
            ("glm-5.2", "GLM-5.2 (frontier)"),
            ("glm-5.1", "GLM-5.1 (compact)"),
            ("glm-5-turbo", "GLM-5-turbo (fast)"),
        ]

        print(f"\n{'=' * 80}")
        print(f"  σ PROFILE COMPARISON ACROSS MODELS")
        print(f"{'=' * 80}")

        all_profiles = {}
        for model_id, label in models:
            print(f"\n  Measuring {label}...")
            profile = measure_sigma_behavioral(model_id, args.base_url, api_key)
            all_profiles[model_id] = profile

        # Print comparison table
        domains = list(DOMAIN_SAMPLES.keys())
        print(f"\n  σ VALUES:")
        print(f"\n  {'Domain':<18}", end="")
        for _, label in models:
            print(f" {label.split()[0]:>12}", end="")
        print()
        print(f"  {'─' * (18 + 13 * len(models))}")

        for domain in domains:
            print(f"  {domain:<18}", end="")
            for model_id, _ in models:
                s = all_profiles.get(model_id, {}).get(domain, {}).get("sigma", "?")
                print(f" {s:>12.4f}" if isinstance(s, float) else f" {'?':>12}", end="")
            print()

        # Composition rule demonstration
        print(f"\n  COMPOSITION RULE (σ_total = σ₁·σ₂·...·σₙ for 5-hop chain):")
        print(f"  {'─' * 60}")
        for model_id, label in models:
            profile = all_profiles[model_id]
            avg_sigma = statistics.mean([d["sigma"] for d in profile.values()])
            chain5 = avg_sigma ** 5
            print(f"  {label:<25} avg σ={avg_sigma:.4f}  σ_total(5 hops)={chain5:.4f}  "
                  f"→ waste={1-chain5:.1%}")

    else:
        print(f"\n{'=' * 80}")
        print(f"  σ PROFILE: {args.model}")
        print(f"{'=' * 80}")

        profile = measure_sigma_behavioral(args.model, args.base_url, api_key)

        print(f"\n  {'Domain':<18} {'H(p)':>8} {'H(p,q)':>8} {'σ':>8} {'T':>8}")
        print(f"  {'─' * 52}")

        for domain, data in sorted(profile.items(), key=lambda x: -x[1]["sigma"]):
            print(f"  {domain:<18} {data['h_p']:>8.3f} {data['h_pq']:>8.3f} "
                  f"{data['sigma']:>8.4f} {data['temperature']:>8.2f}")

        sigmas = [d["sigma"] for d in profile.values()]
        print(f"\n  Average σ: {sum(sigmas)/len(sigmas):.4f}")
        print(f"  Best domain: {max(profile, key=lambda d: profile[d]['sigma'])}")
        print(f"  Worst domain: {min(profile, key=lambda d: profile[d]['sigma'])}")
        print(f"  T (overall) = {1/(1 - sum(sigmas)/len(sigmas)):.2f}")


if __name__ == "__main__":
    import statistics
    main()
