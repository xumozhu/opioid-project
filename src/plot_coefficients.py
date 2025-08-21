# src/plot_coefficients.py
import os
import pandas as pd
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "datasets", "mixed_effects_summary.csv")
OUT  = os.path.join(HERE, "..", "outputs", "policy_coef_plot.png")

def load_and_normalize(path):
    df = pd.read_csv(path)

    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    colmap = {}

    if   "variable" in df.columns: colmap["variable"] = "variable"
    elif "term"     in df.columns: colmap["variable"] = "term"
    elif "param"    in df.columns: colmap["variable"] = "param"
    else:
        raise ValueError("can't find variable/term/param 之一。")

    if   "coef"        in df.columns: colmap["coef"] = "coef"
    elif "coefficient" in df.columns: colmap["coef"] = "coefficient"
    elif "estimate"    in df.columns: colmap["coef"] = "estimate"
    else:
        raise ValueError("error")

    if   "pvalue"   in df.columns: colmap["pvalue"] = "pvalue"
    elif "p_value"  in df.columns: colmap["pvalue"] = "p_value"
    elif "p"        in df.columns: colmap["pvalue"] = "p"
    elif "pr"       in df.columns: colmap["pvalue"] = "pr"
    elif "pr_>|z|"  in df.columns: colmap["pvalue"] = "pr_>|z|"
    elif "pr_>|t|"  in df.columns: colmap["pvalue"] = "pr_>|t|"
    else:
        raise ValueError("error")
    
    df = df[[colmap["variable"], colmap["coef"], colmap["pvalue"]]].rename(
        columns={colmap["variable"]: "variable", colmap["coef"]: "coef", colmap["pvalue"]: "pvalue"}
    )

    df["variable"] = df["variable"].astype(str).str.strip()

    df["coef"] = pd.to_numeric(df["coef"], errors="coerce")
    df["pvalue"] = pd.to_numeric(df["pvalue"], errors="coerce")

    if df["coef"].isna().any() or df["pvalue"].isna().any():
        bad_rows = df[df["coef"].isna() | df["pvalue"].isna()]
        print("⚠️ ")
        print(bad_rows)
        df = df.dropna(subset=["coef", "pvalue"])

    return df

def stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return ""

def main():
    df = load_and_normalize(DATA)

    df = df[~df["variable"].str.lower().isin(["intercept", "(intercept)"])]

    df["stars"] = df["pvalue"].apply(stars)

    df = df.sort_values("coef")

    plt.figure(figsize=(9, 5.5))
    y = range(len(df))
    colors = df["coef"].apply(lambda x: "steelblue" if x > 0 else "tomato")
    plt.barh(list(y), df["coef"], color=colors)

    for i, (c, s) in enumerate(zip(df["coef"], df["stars"])):
        xtext = c + (0.05 if c >= 0 else -0.05)
        ha = "left" if c >= 0 else "right"
        plt.text(xtext, i, f"{c:.2f} {s}", va="center", ha=ha)

    plt.yticks(list(y), df["variable"])
    plt.axvline(0, linestyle="--", linewidth=1, color="gray")
    plt.xlabel("Coefficient")
    plt.title("Mixed Effects Model – Policy Coefficients")
    plt.tight_layout()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, dpi=300)
    print(f"✅ Saved: {OUT}")

if __name__ == "__main__":
    main()
