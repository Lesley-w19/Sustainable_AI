import numpy as np
import pandas as pd
from prompt_engine import compute_features

def generate_dataset(n=2000):
    rows = []
    for _ in range(n):
        # random fake prompt
        text = " ".join(np.random.choice(
            ["data", "model", "network", "training", "learning", "classification", "prompt",
             "token", "embedding", "context", "analysis", "dataset", "neural", "energy"], 
            size=np.random.randint(5, 40))
        )

        features = compute_features(text)

        layers = np.random.randint(2, 60)
        train_h = np.random.uniform(0.5, 10)
        flops = np.random.uniform(1e19, 5e20)

        # use your formula as ground truth for synthetic prediction
        base = layers * train_h * flops * 1e-23
        complexity = (
            0.4*features["type_token_ratio"]
            + 0.3*(features["avg_sentence_len"]/40)
            + 0.2*features["sections"]
            + 0.1*(1-features["stopword_ratio"])
        )
        complexity = np.clip(complexity, 0, 2)
        energy = base * (1 + complexity)

        rows.append({
            **features,
            "layers": layers,
            "training_hours": train_h,
            "flops_per_hour": flops,
            "energy_kwh": energy,
        })

    df = pd.DataFrame(rows)
    df.to_csv("energy_training_data.csv", index=False)
    return df

if __name__ == "__main__":
    generate_dataset()
