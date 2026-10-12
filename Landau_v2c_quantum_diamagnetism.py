#!/usr/bin/env python3

"""LANDAU DIAMAGNETISM V2c: learned structural-response susceptibility."""

import argparse
import csv
import itertools
from pathlib import Path
import numpy as np



DEFAULT = "/data/loto7_4698_k80.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_2971.csv"
# DEFAULT = "/data/loto7_4698_k80_loto_plus_1727.csv"




N, K, WINDOW = 39, 7, 768
LAGS = (1, 2, 8, 32)
START = max(LAGS) + 1
PAIRS = tuple(
    (a, b) for a in range(K) for b in range(a, K)
)


def load(path):
    rows = []
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue
            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem sedam celih brojeva."
                ) from error

            if (len(values) != K or values != sorted(values)
                    or len(set(values)) != K
                    or not all(1 <= v <= N for v in values)):
                raise ValueError(
                    f"Red {line}: neispravna sortirana 7/39 kombinacija."
                )
            rows.append(values)

    return np.asarray(rows, dtype=float).reshape(-1, K)


def change_fields(rows):
    q = rows / (N + 1)
    return np.asarray([
        [
            (q[t-1] - q[t-1-lag]) / lag
            for lag in LAGS
        ]
        for t in range(START, len(rows) + 1)
    ])


def response_design(fields):
    count = len(fields)
    width = K + len(LAGS) * len(PAIRS)
    a = np.zeros((count, K, width))

    for j in range(K):
        a[:, j, j] = 1.

    # Svaka matrica odziva parametrizovana je simetricno.
    for group in range(len(LAGS)):
        for index, (j, k) in enumerate(PAIRS):
            column = K + group * len(PAIRS) + index
            a[:, j, column] = fields[:, group, k]
            if j != k:
                a[:, k, column] = fields[:, group, j]

    return a.reshape(count * K, width)


def prepare(rows, all_fields, t):
    begin = max(START, t-WINDOW)
    fields = all_fields[begin-START:t-START]
    q = rows / (N + 1)

    # Ciljni red t iskljucen je iz podesavanja.
    y = (q[begin:t] - q[begin-1:t-1]).ravel()
    a = response_design(fields)
    scale = np.maximum(
        np.sqrt((a*a).mean(axis=0)), 1e-8
    )

    return (
        a / scale,
        scale,
        y,
        all_fields[t-START],
        begin,
    )


def response(fields, bias, susceptibility):
    return bias + np.einsum(
        "gij,gj->i", susceptibility, fields
    )


def free_energy(fields, bias, susceptibility):
    # Negativni izvod po poljima daje nauceni odziv.
    return float(
        -bias @ fields[0]
        - .5 * np.einsum(
            "gi,gij,gj",
            fields,
            susceptibility,
            fields,
        )
    )


def fit(prepared, rows, t, alpha):
    a, scale, y, query, begin = prepared
    penalty = np.diag(np.r_[
        np.zeros(K),
        np.full(a.shape[1]-K, alpha),
    ])

    coefficient = np.linalg.solve(
        a.T @ a + penalty,
        a.T @ y,
    )
    physical = coefficient / scale
    bias = physical[:K]

    susceptibility = np.zeros((len(LAGS), K, K))
    for group in range(len(LAGS)):
        for index, (j, k) in enumerate(PAIRS):
            value = physical[
                K + group*len(PAIRS) + index
            ]
            susceptibility[group, j, k] = value
            susceptibility[group, k, j] = value

    target = rows[t-1] + (N+1) * response(
        query, bias, susceptibility
    )

    residual = (
        (y - a @ coefficient).reshape(-1, K)
    ) * (N+1)

    position_scale = np.maximum(
        1., np.median(np.abs(residual), axis=0)
    )
    gap_residual = np.diff(
        np.column_stack((
            np.zeros(len(residual)),
            residual,
            np.zeros(len(residual)),
        )),
        axis=1,
    )
    gap_scale = np.maximum(
        1., np.median(np.abs(gap_residual), axis=0)
    )

    return (
        target,
        position_scale,
        gap_scale,
        bias,
        susceptibility,
    )


def gaps(x, n=N):
    return np.diff(
        np.r_[0., x, float(n+1)]
    ) - 1.


def cost(combo, target, position_scale, gap_scale, n=N):
    return float(
        (
            np.abs(np.asarray(combo)-target)
            / position_scale
        ).sum()
        + (
            np.abs(gaps(combo, n)-gaps(target, n))
            / gap_scale
        ).sum()
    )


def decode(target, position_scale, gap_scale, n=N):
    # Tacan globalni minimum nad svim rastucim kombinacijama.
    k = len(target)
    target_gaps = gaps(target, n)
    table = np.full((k, n+1), np.inf)
    parent = np.full((k, n+1), -1, dtype=int)

    for v in range(1, n-k+2):
        table[0, v] = (
            abs(v-target[0]) / position_scale[0]
            + abs(v-1-target_gaps[0]) / gap_scale[0]
        )

    for j in range(1, k):
        for v in range(j+1, n-k+j+2):
            previous = np.arange(j, v)
            choices = (
                table[j-1, previous]
                + np.abs(
                    v-previous-1-target_gaps[j]
                ) / gap_scale[j]
            )
            best = int(np.argmin(choices))
            parent[j, v] = previous[best]
            table[j, v] = (
                choices[best]
                + abs(v-target[j]) / position_scale[j]
            )

    ends = np.arange(k, n+1)
    scores = (
        table[k-1, ends]
        + np.abs(n-ends-target_gaps[k]) / gap_scale[k]
    )
    best = int(np.argmin(scores))
    v = int(ends[best])
    combo = [v]

    for j in range(k-1, 0, -1):
        v = int(parent[j, v])
        combo.append(v)

    return tuple(reversed(combo)), float(scores[best])


def hits(prediction, actual):
    return len(
        set(map(int, prediction)) & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}"
    )


def self_test():
    ps = np.array([1., 1.4, 1.1])
    gs = np.array([1., 1.3, 1.2, 1.])

    for target in (
        [1.8, 4.6, 7.1],
        [8., 2., 5.],
        [-2., 12., 3.],
    ):
        target = np.asarray(target)
        combo, score = decode(target, ps, gs, n=9)
        exact = min(
            cost(c, target, ps, gs, n=9)
            for c in itertools.combinations(range(1, 10), 3)
        )
        assert np.isclose(
            score, exact, rtol=1e-12, atol=1e-12
        )
        assert np.isclose(
            score, cost(combo, target, ps, gs, n=9)
        )

    rows = np.tile(np.array([
        [1, 2, 3, 4, 5, 6, 7],
        [4, 12, 18, 28, 35, 36, 37],
        [33, 34, 35, 36, 37, 38, 39],
        [2, 7, 12, 18, 24, 30, 39],
    ], dtype=float), (50, 1))

    fields = change_fields(rows)
    t = 170
    result = fit(
        prepare(rows, fields, t), rows, t, 10.
    )

    changed = rows.copy()
    changed[t:] = [3, 8, 13, 19, 25, 31, 38]
    altered = fit(
        prepare(changed, change_fields(changed), t),
        changed,
        t,
        10.,
    )

    for a, b in zip(result, altered):
        assert np.array_equal(a, b)

    target, ps, gs, bias, chi = result
    assert np.allclose(
        chi, chi.transpose(0, 2, 1)
    )

    h = fields[t-START]
    derivative = np.zeros(K)
    eps = 1e-6

    for group in range(len(LAGS)):
        for j in range(K):
            plus, minus = h.copy(), h.copy()
            plus[group, j] += eps
            minus[group, j] -= eps
            derivative[j] -= (
                free_energy(plus, bias, chi)
                - free_energy(minus, bias, chi)
            ) / (2*eps)

    assert np.allclose(
        derivative,
        response(h, bias, chi),
        atol=1e-8,
    )
    assert (
        decode(target, ps, gs)
        == decode(target, ps, gs)
    )

    print(
        "Self-test OK: odziv iz energije, simetrija, "
        "hronologija i globalni minimum."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", default=DEFAULT)
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    rows = load(args.csv)
    if args.validation < 16 or args.test < 16:
        parser.error(
            "Validation i test moraju imati bar 16 kola."
        )

    split = len(rows)-args.test
    begin = split-args.validation
    if begin < START+128:
        parser.error(
            "Premalo istorije za zadatu validaciju i test."
        )

    fields = change_fields(rows)
    alphas = (1., 10., 100.)
    validation = [[] for _ in alphas]

    print(
        f"Ulaz: {Path(args.csv).resolve()}; "
        f"redova={len(rows)}",
        flush=True,
    )

    for t in range(begin, split):
        prepared = prepare(rows, fields, t)

        for j, alpha in enumerate(alphas):
            target, ps, gs, _, _ = fit(
                prepared, rows, t, alpha
            )
            combo, _ = decode(target, ps, gs)
            validation[j].append(hits(combo, rows[t]))

    means = [
        np.mean(values) for values in validation
    ]
    best = int(np.argmax(means))
    alpha = alphas[best]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; prosek={means[best]:.4f}",
        flush=True,
    )

    model_scores = []
    previous_scores = []

    for t in range(split, len(rows)):
        target, ps, gs, _, _ = fit(
            prepare(rows, fields, t),
            rows,
            t,
            alpha,
        )
        combo, _ = decode(target, ps, gs)
        model_scores.append(hits(combo, rows[t]))
        previous_scores.append(hits(rows[t-1], rows[t]))

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU DIAMAGNETISM V2c", model_scores)
    summarize("Prethodna kombinacija", previous_scores)

    target, ps, gs, bias, chi = fit(
        prepare(rows, fields, len(rows)),
        rows,
        len(rows),
        alpha,
    )
    combo, score = decode(target, ps, gs)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1].astype(int)))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Trosak projekcije={score:.10f}; "
        "globalni minimum potvrden."
    )
    print(
        "Svojstvene vrednosti odziva lag=1: "
        + ",".join(
            f"{v:.6f}" for v in np.linalg.eigvalsh(chi[0])
        )
    )
    print(
        "Smer odziva naucen je iz podataka; "
        "trosak nije verovatnoca."
    )


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: redovi 4587..4634; alpha=10; prosek=1.2083
Test: redovi 4635..4698
LANDAU DIAMAGNETISM V2c: prosek=1.5156; pogodaka 0..7=[9, 24, 22, 7, 2, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 8,x,18,y,26,z,35
Trosak projekcije=1.3394883044; globalni minimum potvrden.
Svojstvene vrednosti odziva lag=1: -0.288183,-0.239515,-0.234712,-0.183808,-0.157909,-0.110986,-0.087813
Smer odziva naucen je iz podataka; trosak nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: redovi 2860..2907; alpha=1; prosek=1.3750
Test: redovi 2908..2971
LANDAU DIAMAGNETISM V2c: prosek=1.4375; pogodaka 0..7=[7, 31, 18, 7, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 7,x,15,y,28,z,38
Trosak projekcije=1.2824856940; globalni minimum potvrden.
Svojstvene vrednosti odziva lag=1: -0.290855,-0.252436,-0.209174,-0.194844,-0.150940,-0.145237,-0.103530
Smer odziva naucen je iz podataka; trosak nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: redovi 1616..1663; alpha=1; prosek=1.1458
Test: redovi 1664..1727
LANDAU DIAMAGNETISM V2c: prosek=1.1719; pogodaka 0..7=[13, 33, 13, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 6,x,17,y,28,z,37
Trosak projekcije=1.1864537873; globalni minimum potvrden.
Svojstvene vrednosti odziva lag=1: -0.312325,-0.236433,-0.224236,-0.193152,-0.160529,-0.141809,-0.105810
Smer odziva naucen je iz podataka; trosak nije verovatnoca.
"""



"""
V2c — LANDAU DIAMAGNETISM: model koji uči smer i jačinu odziva rasporeda na prethodne promene.

Provera je potvrdila da odziv odgovara izvodu definisane energije i da promena budućih redova ne utiče na raniju predikciju. Izbor cele kombinacije rešava se tačno dinamičkim programiranjem. 

Odziv distribucije na promenu uslova. U fizici se odziv dobija iz promene energije pod magnetnim poljem. Za moj model trebalo bi definisati merljivu promenu prethodnih konfiguracija i učiti kako se naredna struktura menja u odnosu na nju. Ne bih unapred zadavao da odgovor mora biti suprotan prethodnom pomeranju — smer i jačinu treba izvesti iz podataka. Inspiracija je Landauov rad o dijamagnetizmu.
V2c zahteva preciznije određenje šta u modelu predstavlja pobudu, a šta odziv.
"""
 
