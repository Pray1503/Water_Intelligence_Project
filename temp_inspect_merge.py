from pathlib import Path
from src.loaders import load_cleaned_dataset, standardize_dataset

files = sorted(Path("data/cleaned").glob("*.csv"))
for path in files:
    df = load_cleaned_dataset(path)
    standardized = standardize_dataset(
        df, dataset_name=path.stem.replace("_cleaned", ""), source_file=path.name
    )
    keys = ["timestamp", "station"]
    dup = standardized.duplicated(subset=keys, keep=False)
    if dup.sum() > 0:
        print(
            path.name,
            "dup key count",
            int(dup.sum()),
            "unique keys",
            standardized.groupby(keys).ngroups,
        )
        print(
            standardized.loc[dup, ["timestamp", "station"]]
            .head(10)
            .to_string(index=False)
        )
        print()
