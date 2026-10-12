#!/usr/bin/env python3

"""Landau V2c: odziv energije CELE kombinacije na prethodne promene."""

import argparse
import csv
from pathlib import Path
import numpy as np

DEFAULTS = (
    "/data/loto7_4698_k80.csv",
    "/data/loto7_4698_k80_loto_2971.csv",
    "/data/loto7_4698_k80_loto_plus_1727.csv",
)

N, K, D, WINDOW = 39, 7, 8, 64
LAGS = (1, 2, 8, 32)
START = max(LAGS) + 1
PAIRS = tuple(
    (i, j) for i in range(D) for j in range(i, D)
)
WIDTH = 2*D + len(LAGS)*(D + len(PAIRS))
ALPHAS = (.01, .1, 1.)


def load(path):
    rows = []

    with Path(path).open(
        newline="", encoding="utf-8-sig"
    ) as stream:
        for line, row in enumerate(csv.reader(stream), 1):
            if not row or all(not v.strip() for v in row):
                continue

            try:
                values = [int(v.strip()) for v in row]
            except ValueError as error:
                raise ValueError(
                    f"Red {line}: ocekujem cele brojeve."
                ) from error

            if (
                len(values) != K
                or values != sorted(values)
                or len(set(values)) != K
                or not all(1 <= v <= N for v in values)
            ):
                raise ValueError(
                    f"Red {line}: neispravna sortirana "
                    "7/39 kombinacija."
                )

            rows.append(values)

    return np.asarray(rows, dtype=int).reshape(-1, K)


def structure(rows):
    rows = np.atleast_2d(rows)

    edges = np.column_stack((
        np.zeros(len(rows)),
        rows,
        np.full(len(rows), 40)
    ))

    # Osam razmaka odredjuje celu kombinaciju.
    # Faktor 8 samo menja jedinice: 8/40 = 1/5.
    return np.diff(edges, axis=1) / 5.


def fields(g, t):
    return np.asarray([
        (g[t-1] - g[t-1-lag]) / lag
        for lag in LAGS
    ])


def replacements(row):
    unused = np.setdiff1d(np.arange(1, 40), row)
    result = np.tile(row, (K*len(unused), 1))

    for j in range(K):
        result[j*len(unused):(j+1)*len(unused), j] = unused

    return np.sort(result, axis=1)


def design(g, h):
    # Skor S=-F.
    # Linearni i kvadratni odziv zavise od cele kombinacije.
    # Nema cilja u obliku prosecnih polozaja ili razmaka.
    pieces = [g, g*g]

    for field in h:
        z = g*field
        pieces.append(z)

        pieces.append(np.column_stack([
            (.5 if i == j else 1.) * z[:, i] * z[:, j]
            for i, j in PAIRS
        ]))

    return np.column_stack(pieces)


def free_energy(combo, h, coefficient):
    return -float(
        design(structure(combo), h)[0] @ coefficient
    )


def response(combo, h, coefficient):
    # R = -dF/dH, zasebno za svako istorijsko polje.
    g = structure(combo)[0]
    result = np.zeros_like(h, dtype=float)
    offset = 2*D

    for group in range(len(LAGS)):
        bias = coefficient[offset:offset+D] * g
        offset += D

        chi = np.zeros((D, D))

        for value, (i, j) in zip(
            coefficient[offset:offset+len(PAIRS)],
            PAIRS
        ):
            chi[i, j] = chi[j, i] = value*g[i]*g[j]

        offset += len(PAIRS)
        result[group] = bias + chi @ h[group]

    return result


def prepare(history):
    if len(history) < WINDOW+START:
        raise ValueError(
            f"Potrebno je najmanje "
            f"{WINDOW+START} prethodnih redova."
        )

    history = history[-(WINDOW+START):]
    g = structure(history)

    gram = np.zeros((WIDTH, WIDTH))
    target = np.zeros(WIDTH)

    # Stvarni naslednik treba da ima veci skor
    # od svake kombinacije dobijene zamenom jednog broja.
    for t in range(START, len(history)):
        h = fields(g, t)

        actual = design(g[t:t+1], h)
        alternatives = design(
            structure(replacements(history[t])), h
        )
        contrasts = actual - alternatives

        gram += (
            contrasts.T @ contrasts / len(contrasts)
        )
        target += contrasts.sum(axis=0) / len(contrasts)

    gram /= WINDOW
    target /= WINDOW

    candidates = np.unique(
        np.vstack(
            [history[START:]]
            + [
                replacements(row)
                for row in history[START:]
            ]
        ),
        axis=0
    )

    h = fields(g, len(history))
    features = design(structure(candidates), h)

    return gram, target, candidates, features, h


def fit(prepared, alpha):
    gram, target = prepared[:2]

    return np.linalg.solve(
        gram + alpha*np.eye(WIDTH),
        target
    )


def choose(prepared, coefficient):
    candidates, features = prepared[2:4]
    scores = features @ coefficient

    if not np.all(np.isfinite(scores)):
        raise ArithmeticError("Neispravan skor energije.")

    best = int(np.argmax(scores))

    return (
        tuple(map(int, candidates[best])),
        float(scores[best])
    )


def predict(prepared, alpha):
    return choose(prepared, fit(prepared, alpha))


def hits(prediction, actual):
    return len(
        set(map(int, prediction))
        & set(map(int, actual))
    )


def summarize(label, scores):
    print(
        f"{label}: prosek={np.mean(scores):.4f}; "
        f"pogodaka 0..7="
        f"{np.bincount(scores, minlength=8).tolist()}",
        flush=True
    )


def run_file(path, validation_count=48, test_count=64):
    rows = load(path)
    split = len(rows) - test_count
    begin = split - validation_count

    if (
        validation_count < 16
        or test_count < 16
        or begin < WINDOW+START
    ):
        raise ValueError(
            "Premalo istorije ili manje od "
            "16 validacionih/test kola."
        )

    print(
        f"\nUlaz: {Path(path).resolve()}; "
        f"redova={len(rows)}",
        flush=True
    )

    validation = [[] for _ in ALPHAS]

    for t in range(begin, split):
        prepared = prepare(rows[:t])

        for j, alpha in enumerate(ALPHAS):
            combo, _ = predict(prepared, alpha)
            validation[j].append(hits(combo, rows[t]))

        if (t-begin+1) % 16 == 0:
            print(
                f"Validacija: "
                f"{t-begin+1}/{validation_count}",
                flush=True
            )

    means = [np.mean(s) for s in validation]
    chosen = int(np.argmax(means))
    alpha = ALPHAS[chosen]

    print(
        f"Validacija: redovi {begin+1}..{split}; "
        f"alpha={alpha:g}; "
        f"prosek={means[chosen]:.4f}",
        flush=True
    )

    scores, previous = [], []

    # Alpha se bira pre testa i zatim ostaje fiksna.
    # Svako test kolo koristi samo prethodne redove.
    for t in range(split, len(rows)):
        combo, _ = predict(prepare(rows[:t]), alpha)

        scores.append(hits(combo, rows[t]))
        previous.append(hits(rows[t-1], rows[t]))

        if (t-split+1) % 16 == 0:
            print(
                f"Test: {t-split+1}/{test_count}",
                flush=True
            )

    print(f"Test: redovi {split+1}..{len(rows)}")
    summarize("LANDAU V2c DIAMAGNETISM", scores)
    summarize("Prethodna kombinacija", previous)

    prepared = prepare(rows)
    coefficient = fit(prepared, alpha)
    combo, score = choose(prepared, coefficient)

    print(
        f"Poslednji red: "
        f"{','.join(map(str, rows[-1]))}"
    )
    print(
        f"SLEDECI RED {len(rows)+1}: "
        f"{','.join(map(str, combo))}"
    )
    print(
        f"Skor S=-F: {score:.10f}; "
        f"kandidata={len(prepared[2])}"
    )
    print(
        "Izbor: poslednje 64 kombinacije "
        "i sve njihove zamene jednog broja."
    )
    print(
        "Smer i jacina odziva uce se iz podataka; "
        "skor nije verovatnoca."
    )


def self_test():
    base = np.array([
        [1,2,3,4,5,6,7],
        [4,12,18,28,35,36,37],
        [33,34,35,36,37,38,39],
        [2,7,12,18,24,30,39]
    ])
    history = np.tile(base, (30, 1))
    p = prepare(history)

    assert np.allclose(p[0], p[0].T)
    assert np.linalg.eigvalsh(p[0]).min() >= -1e-8

    for row in base:
        neighbors = replacements(row)

        assert (
            len(neighbors)
            == len(np.unique(neighbors, axis=0))
            == 224
        )
        assert all(hits(row, x) == 6 for x in neighbors)

    for alpha in ALPHAS:
        coefficient = fit(p, alpha)
        result = choose(p, coefficient)

        assert result == predict(prepare(history), alpha)
        assert any(
            np.array_equal(result[0], x)
            for x in p[2]
        )
        assert np.isclose(
            -free_energy(result[0], p[4], coefficient),
            result[1]
        )

        # Identitet R=-dF/dH za svaku komponentu polja.
        analytic = response(
            result[0], p[4], coefficient
        )
        numeric = np.zeros_like(analytic)
        eps = 1e-6

        for group in range(len(LAGS)):
            for j in range(D):
                plus = p[4].copy()
                minus = p[4].copy()
                plus[group, j] += eps
                minus[group, j] -= eps

                numeric[group, j] = -(
                    free_energy(
                        result[0], plus, coefficient
                    )
                    - free_energy(
                        result[0], minus, coefficient
                    )
                ) / (2*eps)

        assert np.allclose(
            analytic, numeric,
            atol=1e-7, rtol=1e-6
        )

    altered = history.copy()
    altered[-1] = base[0]

    assert (
        predict(prepare(history[:-1]), .1)
        == predict(prepare(altered[:-1]), .1)
    )

    print(
        "Self-test OK: odziv iz energije, rangiranje, "
        "zamene, ponovljivost i hronologija."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="*")
    parser.add_argument("--validation", type=int, default=48)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    for path in args.csv or DEFAULTS:
        run_file(path, args.validation, args.test)


if __name__ == "__main__":
    main()



"""
Ulaz: /data/loto7_4698_k80.csv; redova=4698
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 4587..4634; alpha=0.01; prosek=1.1458
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 4635..4698
LANDAU V2c DIAMAGNETISM: prosek=1.2656; pogodaka 0..7=[15, 24, 19, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2188; pogodaka 0..7=[13, 30, 16, 4, 1, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 4699: 1,x,22,y,24,z,36
Skor S=-F: 0.3376087270; kandidata=14396
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Smer i jacina odziva uce se iz podataka; skor nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_2971.csv; redova=2971
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 2860..2907; alpha=1; prosek=1.3125
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 2908..2971
LANDAU V2c DIAMAGNETISM: prosek=1.2344; pogodaka 0..7=[13, 29, 17, 4, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[11, 30, 16, 7, 0, 0, 0, 0]
Poslednji red: 18,20,21,24,29,34,38
SLEDECI RED 2972: 27,x,30,y,32,z,37
Skor S=-F: 2.5151327692; kandidata=14392
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Smer i jacina odziva uce se iz podataka; skor nije verovatnoca.



Ulaz: /data/loto7_4698_k80_loto_plus_1727.csv; redova=1727
Validacija: 16/48
Validacija: 32/48
Validacija: 48/48
Validacija: redovi 1616..1663; alpha=0.1; prosek=1.2708
Test: 16/64
Test: 32/64
Test: 48/64
Test: 64/64
Test: redovi 1664..1727
LANDAU V2c DIAMAGNETISM: prosek=1.3281; pogodaka 0..7=[11, 28, 19, 5, 1, 0, 0, 0]
Prethodna kombinacija: prosek=1.2969; pogodaka 0..7=[15, 23, 18, 8, 0, 0, 0, 0]
Poslednji red: 4,12,18,28,35,36,37
SLEDECI RED 1728: 1,x,4,y,11,z,17
Skor S=-F: 0.0041645150; kandidata=14388
Izbor: poslednje 64 kombinacije i sve njihove zamene jednog broja.
Smer i jacina odziva uce se iz podataka; skor nije verovatnoca.
"""



"""
popravljen način rangiranja celih kombinacija
Uči energiju cele kombinacije i njen odziv na prethodne promene. 
Nema ciljnih prosečnih položaja, nametnute normalne ili uniformne raspodele, niti random postupka.

Fajl	                Sledeća kombinacija	    Prosek pogodaka na 64 kola	Prethodna kombinacija
Objedinjeni, red 4699	1,x,22,y,24,z,36	    1,2656	                    1,2188
Loto, red 2972	        27,x,30,y,32,z,37	    1,2344	                    1,2969
Loto Plus, red 1728	    1,x,4,y,11,z,17	        1,3281	                    1,2969

Pretraga bira između poslednje 64 kombinacije i svih njihovih zamena jednog broja. Provereni su odziv iz energije, ponovljivost i korišćenje isključivo prethodnih redova.
"""
