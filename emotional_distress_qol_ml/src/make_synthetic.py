
import numpy as np, pandas as pd, argparse, os
from numpy.random import default_rng

def make(n=1500, seed=42):
    rng = default_rng(seed)
    # Cohort role
    cohort = rng.choice(["patient", "family"], size=n, p=[0.65, 0.35])
    age = rng.normal(55, 12, size=n).clip(18, 90).round(0)
    sex = rng.choice(["male","female"], size=n, p=[0.48,0.52])
    stage = rng.choice(["I","II","III","IV"], size=n, p=[0.15,0.30,0.35,0.20])
    treatment = rng.choice(["chemo","radio","surgery","immuno","none"], size=n, p=[0.35,0.15,0.30,0.10,0.10])

    # Simulate QoL subscales (0-100). Worse QoL -> higher distress probability.
    qlq_phys = rng.normal(70, 15, size=n).clip(0,100)
    qlq_emot = rng.normal(65, 18, size=n).clip(0,100)
    qlq_cogn = rng.normal(72, 12, size=n).clip(0,100)
    fact_wb  = rng.normal(68, 14, size=n).clip(0,100)
    promis_dep = rng.normal(40, 10, size=n).clip(0,100)  # higher worse
    promis_anx = rng.normal(45, 10, size=n).clip(0,100)

    # Interaction: family members' distress depends more on promis_anx; patients on qlq_emot.
    base = -1.0
    logit = (
        base
        + 0.03*(50 - qlq_emot)
        + 0.02*(50 - qlq_phys)
        + 0.02*(promis_dep - 50)
        + 0.025*(promis_anx - 50)
        + 0.15*(stage == "IV").astype(float)
        + 0.10*(treatment == "chemo").astype(float)
        + 0.10*(cohort == "family").astype(float)
    )
    # subtle interaction
    logit += 0.015*(cohort == "patient").astype(float)*(50 - qlq_emot)
    logit += 0.015*(cohort == "family").astype(float)*(promis_anx - 50)

    p = 1/(1+np.exp(-logit))
    distress = (rng.uniform(size=n) < p).astype(int)

    df = pd.DataFrame({
        "subject_id": np.arange(1, n+1),
        "cohort_role": cohort,
        "age": age, "sex": sex, "stage": stage, "treatment": treatment,
        "qlq_physical": qlq_phys,
        "qlq_emotional": qlq_emot,
        "qlq_cognitive": qlq_cogn,
        "fact_wellbeing": fact_wb,
        "promis_depression": promis_dep,
        "promis_anxiety": promis_anx,
        "distress": distress
    })
    # introduce some missingness
    for col in ["qlq_emotional", "promis_anxiety", "age"]:
        mask = rng.uniform(size=n) < 0.05
        df.loc[mask, col] = np.nan
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="data/qol_dataset.csv")
    args = ap.parse_args()
    df = make(args.n, args.seed)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} with shape {df.shape}")

if __name__ == "__main__":
    main()
