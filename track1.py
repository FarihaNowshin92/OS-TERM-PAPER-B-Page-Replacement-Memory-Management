"""
Track 1
Learned Page Replacement


    FIFO, LRU, Optimal (Belady)

"""

import csv
import math
import os
import random
import statistics
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAVE_PLOT = True
except Exception:
    HAVE_PLOT = False

FRAMES = 4
FIRST_HALF = 300
SECOND_HALF = 300
TOTAL = FIRST_HALF + SECOND_HALF
TRAIN_SEEDS = range(20)
TEST_SEED = 2026
OUT = Path("results")


def make_trace(seed):
    """First half: locality/sequential. Second half: random/bursty."""
    rng = random.Random(seed)

    first = []
    for i in range(FIRST_HALF):
        # Mostly sequential/local, with occasional wider accesses.
        if i % 20 < 14:
            first.append(i % 5)
        else:
            first.append(rng.randrange(8))

    second = []
    for i in range(SECOND_HALF):
        # Alternating bursts of a small working set and random references.
        if i % 30 < 10:
            second.append(rng.randrange(4))
        else:
            second.append(rng.randrange(12))

    return first + second


def fifo(trace, frames=FRAMES):
    memory = []
    queue = []
    faults = 0

    for page in trace:
        if page in memory:
            continue

        faults += 1

        if len(memory) < frames:
            memory.append(page)
            queue.append(page)
        else:
            old = queue.pop(0)
            memory[memory.index(old)] = page
            queue.append(page)

    return faults


def lru(trace, frames=FRAMES):
    memory = []
    last_used = {}
    faults = 0

    for i, page in enumerate(trace):
        if page not in memory:
            faults += 1

            if len(memory) < frames:
                memory.append(page)
            else:
                old = min(memory, key=lambda x: last_used.get(x, -1))
                memory[memory.index(old)] = page

        last_used[page] = i

    return faults


def next_use(trace, index, page):
    for j in range(index + 1, len(trace)):
        if trace[j] == page:
            return j
    return 10**9


def optimal_eviction(trace, index, memory):
    return max(memory, key=lambda page: next_use(trace, index, page))


def optimal(trace, frames=FRAMES):
    memory = []
    faults = 0

    for i, page in enumerate(trace):
        if page in memory:
            continue

        faults += 1

        if len(memory) < frames:
            memory.append(page)
        else:
            old = optimal_eviction(trace, i, memory)
            memory[memory.index(old)] = page

    return faults


def candidate_features(history, candidate):
    """Features available without looking into the future."""
    recent_30 = history[-30:]
    recent_100 = history[-100:]

    recency = 100
    for distance, page in enumerate(reversed(history), start=1):
        if page == candidate:
            recency = distance
            break

    f30 = recent_30.count(candidate) / 30.0
    f100 = recent_100.count(candidate) / 100.0
    not_recent = 1.0 if recency >= 100 else 0.0

    return [
        min(recency, 100) / 100.0,
        f30,
        f100,
        not_recent
    ]


def sigmoid(z):
    z = max(-30.0, min(30.0, z))
    return 1.0 / (1.0 + math.exp(-z))


def train_logistic(samples, epochs=800, learning_rate=0.15):
    """Tiny binary logistic regression trained with gradient descent."""
    dimension = len(samples[0][0])
    weights = [0.0] * (dimension + 1)

    for _ in range(epochs):
        gradient = [0.0] * (dimension + 1)

        for x, y in samples:
            z = weights[0]
            for j in range(dimension):
                z += weights[j + 1] * x[j]

            prediction = sigmoid(z)
            error = prediction - y

            gradient[0] += error
            for j in range(dimension):
                gradient[j + 1] += error * x[j]

        n = len(samples)
        for j in range(dimension + 1):
            weights[j] -= learning_rate * gradient[j] / n

    return weights


def predict_probability(weights, x):
    z = weights[0]
    for j, value in enumerate(x):
        z += weights[j + 1] * value
    return sigmoid(z)


def build_training_set():
    samples = []

    for seed in TRAIN_SEEDS:
        trace = make_trace(seed)
        memory = []
        history = []
        last_used = {}

        for i, page in enumerate(trace):
            if page in memory:
                last_used[page] = i
                history.append(page)
                continue

            if len(memory) == FRAMES:
                correct = optimal_eviction(trace, i, memory)

                for candidate in memory:
                    x = candidate_features(history, candidate)
                    y = 1 if candidate == correct else 0
                    samples.append((x, y))

                # Use LRU to create the next training state.
                old = min(memory, key=lambda x: last_used.get(x, -1))
                memory[memory.index(old)] = page
            else:
                memory.append(page)

            last_used[page] = i
            history.append(page)

    return samples


def learned_policy(trace, weights):
    """Run the learned eviction policy and collect detailed metrics."""
    memory = []
    history = []
    last_used = {}
    faults = 0
    decisions = []

    for i, page in enumerate(trace):
        if page in memory:
            last_used[page] = i
            history.append(page)
            continue

        faults += 1

        if len(memory) < FRAMES:
            memory.append(page)
            last_used[page] = i
            history.append(page)
            continue

        probabilities = []
        for candidate in memory:
            x = candidate_features(history, candidate)
            probabilities.append(predict_probability(weights, x))

        best_index = max(range(len(memory)),
                         key=lambda j: probabilities[j])

        predicted = memory[best_index]
        actual = optimal_eviction(trace, i, memory)

        sorted_probs = sorted(probabilities, reverse=True)
        confidence = sorted_probs[0] - sorted_probs[1]

        decisions.append({
            "index": i,
            "predicted": predicted,
            "actual": actual,
            "correct": predicted == actual,
            "confidence": confidence
        })

        memory[memory.index(predicted)] = page
        last_used.pop(predicted, None)
        last_used[page] = i
        history.append(page)

    return faults, decisions


def section_faults(trace, start, end):
    part = trace[start:end]
    return {
        "FIFO": fifo(part),
        "LRU": lru(part),
        "Optimal": optimal(part),
    }


def hit_ratio(reference_count, faults):
    """Calculate hit ratio from the number of references and page faults."""
    if reference_count == 0:
        return 0.0
    return (reference_count - faults) / reference_count


def save_csv(filename, rows, fieldnames):
    with open(OUT / filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows, decisions):
    if not HAVE_PLOT:
        return

    names = ["FIFO", "LRU", "Optimal", "Learned"]

    plt.figure(figsize=(8, 4.5))
    totals = [rows[0][n] for n in names]
    plt.bar(names, totals)
    plt.ylabel("Page faults")
    plt.title("Page faults on the shifted workload")
    plt.tight_layout()
    plt.savefig(OUT / "total_page_faults.png", dpi=150)
    plt.close()

    before = [rows[0][n + "_before"] for n in names[:3]]
    after = [rows[0][n + "_after"] for n in names[:3]]

    plt.figure(figsize=(8, 4.5))
    x = range(3)
    width = 0.35
    plt.bar([i - width / 2 for i in x], before, width, label="Before shift")
    plt.bar([i + width / 2 for i in x], after, width, label="After shift")
    plt.xticks(list(x), names[:3])
    plt.ylabel("Page faults")
    plt.title("Classical policies before and after workload shift")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "before_after_shift.png", dpi=150)
    plt.close()

    if decisions:
        correct_conf = [d["confidence"] for d in decisions if d["correct"]]
        wrong_conf = [d["confidence"] for d in decisions if not d["correct"]]

        plt.figure(figsize=(8, 4.5))
        plt.boxplot([correct_conf, wrong_conf], tick_labels=["Correct", "Wrong"])
        plt.ylabel("Confidence margin")
        plt.title("Learned policy confidence")
        plt.tight_layout()
        plt.savefig(OUT / "confidence.png", dpi=150)
        plt.close()


def main():
    OUT.mkdir(exist_ok=True)

    training = build_training_set()
    weights = train_logistic(training)

    trace = make_trace(TEST_SEED)

    fifo_total = fifo(trace)
    lru_total = lru(trace)
    optimal_total = optimal(trace)

    learned_total, decisions = learned_policy(trace, weights)

    learned_before = sum(
        1 for i in range(FIRST_HALF)
        if False
    )
    # The learned policy is evaluated by replaying the complete trace,
    # then counting its decisions in each half.
    # Faults before the first full frame are not learned decisions.
    learned_before = 0
    learned_after = 0

    # Re-run to get exact learned fault counts by half.
    memory = []
    history = []
    last_used = {}
    for i, page in enumerate(trace):
        if page in memory:
            last_used[page] = i
            history.append(page)
            continue

        if len(memory) < FRAMES:
            memory.append(page)
            last_used[page] = i
            history.append(page)
            if i < FIRST_HALF:
                learned_before += 1
            else:
                learned_after += 1
            continue

        probs = [
            predict_probability(weights, candidate_features(history, c))
            for c in memory
        ]
        old = memory[max(range(len(memory)), key=lambda j: probs[j])]
        memory[memory.index(old)] = page
        last_used.pop(old, None)
        last_used[page] = i
        history.append(page)

        if i < FIRST_HALF:
            learned_before += 1
        else:
            learned_after += 1

    classical_before = section_faults(trace, 0, FIRST_HALF)
    classical_after = section_faults(trace, FIRST_HALF, TOTAL)

    # Each half contains FIRST_HALF references, so hit ratio is:
    # (references - page faults) / references.
    before_count = FIRST_HALF
    after_count = TOTAL - FIRST_HALF

    faults_before = {
        "FIFO": classical_before["FIFO"],
        "LRU": classical_before["LRU"],
        "Optimal": classical_before["Optimal"],
        "Learned": learned_before,
    }
    faults_after = {
        "FIFO": classical_after["FIFO"],
        "LRU": classical_after["LRU"],
        "Optimal": classical_after["Optimal"],
        "Learned": learned_after,
    }

    hit_ratios_before = {
        n: hit_ratio(before_count, faults_before[n])
        for n in ["FIFO", "LRU", "Optimal", "Learned"]
    }
    hit_ratios_after = {
        n: hit_ratio(after_count, faults_after[n])
        for n in ["FIFO", "LRU", "Optimal", "Learned"]
    }

    correct = sum(d["correct"] for d in decisions)
    accuracy = correct / len(decisions)

    correct_conf = [d["confidence"] for d in decisions if d["correct"]]
    wrong_conf = [d["confidence"] for d in decisions if not d["correct"]]

    row = {
        "seed": TEST_SEED,
        "frames": FRAMES,
        "FIFO": fifo_total,
        "LRU": lru_total,
        "Optimal": optimal_total,
        "Learned": learned_total,
        "FIFO_before": classical_before["FIFO"],
        "LRU_before": classical_before["LRU"],
        "Optimal_before": classical_before["Optimal"],
        "Learned_before": learned_before,
        "FIFO_after": classical_after["FIFO"],
        "LRU_after": classical_after["LRU"],
        "Optimal_after": classical_after["Optimal"],
        "Learned_after": learned_after,
        "FIFO_hit_ratio_before": hit_ratios_before["FIFO"],
        "LRU_hit_ratio_before": hit_ratios_before["LRU"],
        "Optimal_hit_ratio_before": hit_ratios_before["Optimal"],
        "Learned_hit_ratio_before": hit_ratios_before["Learned"],
        "FIFO_hit_ratio_after": hit_ratios_after["FIFO"],
        "LRU_hit_ratio_after": hit_ratios_after["LRU"],
        "Optimal_hit_ratio_after": hit_ratios_after["Optimal"],
        "Learned_hit_ratio_after": hit_ratios_after["Learned"],
        "learned_accuracy": accuracy,
        "mean_confidence_correct": statistics.mean(correct_conf) if correct_conf else 0,
        "mean_confidence_wrong": statistics.mean(wrong_conf) if wrong_conf else 0,
    }

    save_csv(
        "results.csv",
        [row],
        list(row.keys())
    )

    decision_rows = []
    for d in decisions:
        decision_rows.append(d)
    save_csv(
        "learned_decisions.csv",
        decision_rows,
        ["index", "predicted", "actual", "correct", "confidence"]
    )

    make_plots([row], decisions)

    with open(OUT / "summary.txt", "w", encoding="utf-8") as f:
        f.write("CSE-307 Track 1 - Learned Page Replacement\n")
        f.write("=" * 50 + "\n")
        f.write(f"Frames: {FRAMES}\n")
        f.write(f"Trace length: {TOTAL}\n")
        f.write("Shift: locality/sequential -> random/bursty\n\n")
        for n in ["FIFO", "LRU", "Optimal", "Learned"]:
            f.write(f"{n}: {row[n]} page faults\n")
        f.write("\nBefore shift:\n")
        for n in ["FIFO", "LRU", "Optimal", "Learned"]:
            f.write(f"{n}: {row[n + '_before']}\n")
        f.write("\nAfter shift:\n")
        for n in ["FIFO", "LRU", "Optimal", "Learned"]:
            f.write(f"{n}: {row[n + '_after']}\n")
        f.write("\nHit ratio before shift:\n")
        for n in ["FIFO", "LRU", "Optimal", "Learned"]:
            f.write(f"{n}: {row[n + '_hit_ratio_before']:.3f} ({row[n + '_hit_ratio_before'] * 100:.1f}%)\n")
        f.write("\nHit ratio after shift:\n")
        for n in ["FIFO", "LRU", "Optimal", "Learned"]:
            f.write(f"{n}: {row[n + '_hit_ratio_after']:.3f} ({row[n + '_hit_ratio_after'] * 100:.1f}%)\n")
        f.write(f"\nLearned eviction accuracy: {accuracy:.3f}\n")
        f.write(f"Mean confidence, correct: {row['mean_confidence_correct']:.3f}\n")
        f.write(f"Mean confidence, wrong: {row['mean_confidence_wrong']:.3f}\n")

    print("Experiment complete.")
    print(f"Results saved in: {OUT.resolve()}")
    print(f"FIFO    : {fifo_total}")
    print(f"LRU     : {lru_total}")
    print(f"Optimal : {optimal_total}")
    print(f"Learned : {learned_total}")
    print("\nHit ratios before shift:")
    for n in ["FIFO", "LRU", "Optimal", "Learned"]:
        print(f"{n:8}: {row[n + '_hit_ratio_before']:.3f} ({row[n + '_hit_ratio_before'] * 100:.1f}%)")
    print("Hit ratios after shift:")
    for n in ["FIFO", "LRU", "Optimal", "Learned"]:
        print(f"{n:8}: {row[n + '_hit_ratio_after']:.3f} ({row[n + '_hit_ratio_after'] * 100:.1f}%)")
    print(f"Learned eviction accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    main()
