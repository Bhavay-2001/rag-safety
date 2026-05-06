import ir_datasets
import random
import csv
from pathlib import Path


def main():
    ds = ir_datasets.load("natural-questions/train")

    Path("data/qa").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    queries = list(ds.queries_iter())
    sample = random.sample(queries, 2000)

    out_path = Path("data/qa/nq_queries_2000.csv")
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["query_id", "query"])
        for q in sample:
            w.writerow([q.query_id, q.text])

    print(f"Wrote {out_path} with {len(sample)} queries.")


if __name__ == "__main__":
    main()
