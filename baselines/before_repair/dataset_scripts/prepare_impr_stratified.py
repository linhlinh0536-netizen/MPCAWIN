import pandas as pd
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

FILE_2017 = BASE / "cicids2017-imp" / "Reformatted_CIC-IDS2017-Imp.csv"
FILE_2018 = BASE / "cicids2018-imp" / "Reformatted_CSE-CIC-IDS2018-Imp.csv"

OUT_2017 = BASE / "Impr2017_stratified.csv"
OUT_2018 = BASE / "Impr2018_stratified.csv"

SUMMARY_2017 = BASE / "Impr2017_attack_family_summary.csv"
SUMMARY_2018 = BASE / "Impr2018_attack_family_summary.csv"

TARGET_SIZE = 160_000
ATTACK_RATIO = 0.03
N_ATTACK = int(TARGET_SIZE * ATTACK_RATIO)   # 4,800
N_BENIGN = TARGET_SIZE - N_ATTACK            # 155,200

MIN_PER_ATTACK_FAMILY = 5
RANDOM_SEED = 42
CHUNK_SIZE = 500_000


def count_dataset(input_file):
    benign_count = 0
    attack_family_counts = {}

    for i, chunk in enumerate(
        pd.read_csv(
            input_file,
            usecols=["attack_flag", "attack_name"],
            chunksize=CHUNK_SIZE
        )
    ):
        benign_count += int((chunk["attack_flag"] == 0).sum())

        attack_chunk = chunk.loc[
            chunk["attack_flag"] == 1, "attack_name"
        ].fillna("UNKNOWN_ATTACK").astype(str)

        vc = attack_chunk.value_counts()
        for family, count in vc.items():
            attack_family_counts[family] = (
                attack_family_counts.get(family, 0) + int(count)
            )

        if (i + 1) % 10 == 0:
            print(f"Count pass: processed {(i + 1) * CHUNK_SIZE:,} rows...")

    return benign_count, dict(sorted(attack_family_counts.items()))


def allocate_attack_quotas(family_counts, target_attack, min_per_family):
    families = list(family_counts.keys())

    if not families:
        raise ValueError("No attack families found.")

    total_available = sum(family_counts.values())
    if target_attack > total_available:
        raise ValueError(
            f"Requested {target_attack} attacks but only "
            f"{total_available} are available."
        )

    quotas = {
        family: min(min_per_family, family_counts[family])
        for family in families
    }

    if sum(quotas.values()) > target_attack:
        quotas = {family: 1 for family in families}
        if sum(quotas.values()) > target_attack:
            raise ValueError(
                "target_attack is smaller than the number of attack families."
            )

    remaining = target_attack - sum(quotas.values())

    capacities = {
        family: family_counts[family] - quotas[family]
        for family in families
    }
    total_capacity = sum(capacities.values())

    if remaining > total_capacity:
        raise ValueError(
            "Not enough remaining attack records to fill target quota."
        )

    if remaining > 0:
        raw_add = {
            family: remaining * capacities[family] / total_capacity
            for family in families
        }

        floor_add = {
            family: min(
                capacities[family],
                int(np.floor(raw_add[family]))
            )
            for family in families
        }

        for family in families:
            quotas[family] += floor_add[family]

        leftover = target_attack - sum(quotas.values())

        ranked = sorted(
            families,
            key=lambda f: raw_add[f] - floor_add[f],
            reverse=True
        )

        for family in ranked:
            if leftover == 0:
                break
            if quotas[family] < family_counts[family]:
                quotas[family] += 1
                leftover -= 1

    if sum(quotas.values()) != target_attack:
        raise RuntimeError(
            f"Quota allocation failed: {sum(quotas.values())} != {target_attack}"
        )

    return quotas


def choose_uniform_from_remaining(
    rng,
    positions,
    remaining_population,
    remaining_sample
):
    current_count = len(positions)

    if current_count == 0 or remaining_sample == 0:
        return np.array([], dtype=int), 0

    future_count = remaining_population - current_count

    selected_count = int(
        rng.hypergeometric(
            ngood=current_count,
            nbad=future_count,
            nsample=remaining_sample
        )
    )

    if selected_count == 0:
        return np.array([], dtype=int), 0

    chosen = rng.choice(
        positions,
        size=selected_count,
        replace=False
    )

    return np.asarray(chosen, dtype=int), selected_count


def create_stratified_impr(
    input_file,
    output_file,
    summary_file,
    dataset_name
):
    print("\n" + "=" * 72)
    print(f"CREATING {dataset_name}")
    print("=" * 72)

    print("\n[PASS 1] Counting benign records and attack families...")
    benign_total, family_counts = count_dataset(input_file)

    attack_total = sum(family_counts.values())
    total_rows = benign_total + attack_total

    print("\nOriginal dataset:")
    print(f"Total rows : {total_rows:,}")
    print(f"Benign     : {benign_total:,}")
    print(f"Attack     : {attack_total:,}")
    print(f"Families   : {len(family_counts)}")

    print("\nOriginal attack families:")
    for family, count in sorted(
        family_counts.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"{family:45s} {count:,}")

    if benign_total < N_BENIGN:
        raise ValueError(
            f"Not enough benign records: {benign_total:,} < {N_BENIGN:,}"
        )

    quotas = allocate_attack_quotas(
        family_counts,
        target_attack=N_ATTACK,
        min_per_family=MIN_PER_ATTACK_FAMILY
    )

    print("\nTarget attack quotas:")
    for family, quota in sorted(
        quotas.items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(
            f"{family:45s} "
            f"original={family_counts[family]:8,d} "
            f"quota={quota:5,d}"
        )

    print(
        f"\nTarget: {N_BENIGN:,} benign + "
        f"{N_ATTACK:,} attack = {TARGET_SIZE:,}"
    )

    print("\n[PASS 2] Sampling records...")

    rng = np.random.default_rng(RANDOM_SEED)

    remaining_benign_population = benign_total
    remaining_benign_sample = N_BENIGN

    remaining_family_population = family_counts.copy()
    remaining_family_sample = quotas.copy()

    selected_parts = []
    total_processed = 0

    for i, chunk in enumerate(
        pd.read_csv(input_file, chunksize=CHUNK_SIZE)
    ):
        total_processed += len(chunk)

        labels = chunk["attack_flag"].to_numpy()

        benign_positions = np.flatnonzero(labels == 0)

        chosen_benign, k_benign = choose_uniform_from_remaining(
            rng,
            benign_positions,
            remaining_benign_population,
            remaining_benign_sample
        )

        if k_benign > 0:
            selected_parts.append(chunk.iloc[chosen_benign].copy())

        remaining_benign_population -= len(benign_positions)
        remaining_benign_sample -= k_benign

        attack_positions = np.flatnonzero(labels == 1)

        if len(attack_positions) > 0:
            names = (
                chunk["attack_name"]
                .fillna("UNKNOWN_ATTACK")
                .astype(str)
                .to_numpy()
            )

            attack_names = names[attack_positions]

            temp = pd.DataFrame({
                "pos": attack_positions,
                "family": attack_names
            })

            for family, group in temp.groupby("family", sort=False):
                family = str(family)
                positions = group["pos"].to_numpy(dtype=int)

                if family not in remaining_family_population:
                    raise RuntimeError(
                        f"Unexpected attack family in pass 2: {family}"
                    )

                chosen_attack, k_attack = choose_uniform_from_remaining(
                    rng,
                    positions,
                    remaining_family_population[family],
                    remaining_family_sample[family]
                )

                if k_attack > 0:
                    selected_parts.append(
                        chunk.iloc[chosen_attack].copy()
                    )

                remaining_family_population[family] -= len(positions)
                remaining_family_sample[family] -= k_attack

        if (i + 1) % 10 == 0:
            selected_so_far = sum(len(x) for x in selected_parts)
            print(
                f"Processed {total_processed:,} rows "
                f"| selected so far: {selected_so_far:,}"
            )

    if remaining_benign_sample != 0:
        raise RuntimeError(
            f"Benign sample incomplete: {remaining_benign_sample} missing."
        )

    missing_attack_quota = {
        family: q
        for family, q in remaining_family_sample.items()
        if q != 0
    }

    if missing_attack_quota:
        raise RuntimeError(
            f"Attack sample incomplete: {missing_attack_quota}"
        )

    result = pd.concat(selected_parts, ignore_index=True)

    result = result.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    print("\n--- FINAL DATASET ---")
    print("Shape:", result.shape)

    label_counts = result["attack_flag"].value_counts().sort_index()
    print("\nAttack distribution:")
    print(label_counts)

    attack_ratio = result["attack_flag"].mean()
    print(f"\nAttack ratio: {attack_ratio * 100:.2f}%")

    sampled_family_counts = (
        result.loc[result["attack_flag"] == 1, "attack_name"]
        .fillna("UNKNOWN_ATTACK")
        .astype(str)
        .value_counts()
    )

    print("\nSampled attack families:")
    print(sampled_family_counts)

    original_families = set(family_counts.keys())
    sampled_families = set(sampled_family_counts.index)
    missing_families = sorted(original_families - sampled_families)

    print("\nMissing attack families:", missing_families)

    if missing_families:
        raise RuntimeError(
            "Rare attack family was lost. "
            f"Missing: {missing_families}"
        )

    if len(result) != TARGET_SIZE:
        raise RuntimeError(
            f"Wrong final size: {len(result)} != {TARGET_SIZE}"
        )

    if int((result["attack_flag"] == 1).sum()) != N_ATTACK:
        raise RuntimeError("Wrong attack sample size.")

    if int((result["attack_flag"] == 0).sum()) != N_BENIGN:
        raise RuntimeError("Wrong benign sample size.")

    result.to_csv(output_file, index=False)

    summary_rows = []
    for family in sorted(family_counts.keys()):
        summary_rows.append({
            "attack_family": family,
            "original_count": family_counts[family],
            "target_quota": quotas[family],
            "sampled_count": int(
                sampled_family_counts.get(family, 0)
            )
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(summary_file, index=False)

    print("\nSaved dataset:")
    print(output_file)

    print("\nSaved family summary:")
    print(summary_file)

    print(f"\n{dataset_name} DONE.")


create_stratified_impr(
    FILE_2017,
    OUT_2017,
    SUMMARY_2017,
    "Impr2017_stratified"
)

create_stratified_impr(
    FILE_2018,
    OUT_2018,
    SUMMARY_2018,
    "Impr2018_stratified"
)

print("\nALL DONE.")
