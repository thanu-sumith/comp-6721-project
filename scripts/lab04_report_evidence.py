"""Reproduce the Lab 04 report audit without fitting or scoring a test model.

Run from the repository root: python scripts/lab04_report_evidence.py
Existing saved split assignments are required and never regenerated here.
"""
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.baseline import load_data, saved_split, prior_baseline


def main():
    for name in ['manifest.json', 'split_ids.csv']:
        if not (ROOT / 'data/splits' / name).exists():
            raise FileNotFoundError('Existing saved splits required; run the project notebook first.')
    raw = pd.read_csv(ROOT / 'data/raw/train.csv')
    df, fingerprint = load_data()
    split = saved_split(df, fingerprint)
    # Treat A/B-swapped pairs as the same pair; use parsed, joined text.
    pairs = [json.dumps(sorted(pair), ensure_ascii=False) for pair in
             zip(df.response_a_text, df.response_b_text)]
    audit = pd.DataFrame({'pair': pairs, 'split': split})
    crossing = audit.groupby('pair')['split'].nunique().gt(1)
    lengths = df[['prompt_text', 'response_a_text', 'response_b_text']].apply(lambda col: col.str.len())
    evidence = {
        'verified_on': '2026-10-21',
        'dataset_sha256': fingerprint,
        'raw_shape': list(raw.shape),
        'joined_text_character_length_summary': lengths.describe().round(2).to_dict(),
        'empty_joined_text_counts': lengths.eq(0).sum().to_dict(),
        'preprocessed_shape': list(df.shape),
        'raw_missing_values': raw.isna().sum().to_dict(),
        'class_counts': df.label.value_counts().sort_index().to_dict(),
        'class_percentages': (100 * df.label.value_counts(normalize=True).sort_index()).to_dict(),
        'repeated_prompt_rows_after_first': int(df.prompt_group.duplicated().sum()),
        'duplicate_joined_prompt_response_triplets_after_first': int(df.duplicated(['prompt_text', 'response_a_text', 'response_b_text']).sum()),
        'partition_class_counts': pd.crosstab(split, df.label).to_dict(),
        'split_counts': split.value_counts().to_dict(),
        'prompt_groups_crossing_partitions': int(pd.DataFrame({'g': df.prompt_group, 's': split}).groupby('g').s.nunique().gt(1).sum()),
        'unordered_joined_response_pair_groups_crossing_partitions': int(crossing.sum()),
        'rows_in_crossing_response_pair_groups': int(audit.pair.isin(crossing[crossing].index).sum()),
        'validation_baseline': prior_baseline(df, split, fingerprint),
        'versions': {p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'matplotlib']},
    }
    evidence['versions']['python'] = platform.python_version()
    output = ROOT / 'results/metrics/lab04_report_evidence.json'
    output.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps(evidence, indent=2))


if __name__ == '__main__':
    main()
