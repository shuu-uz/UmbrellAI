"""Train with python rain_score.py; score with python rain_score.py --score 60 1 3 4."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUTPUT = Path(__file__).parent / "results" / "final_20261123" / "rain_score.json"
FEATURES = ["travel_rain_forecast", "carry_discomfort", "wet_discomfort"]


def score(model, daily_pop, rain, carry, wet):
    if daily_pop != 60 or rain not in (0, 1) or carry not in range(1, 6) or wet not in range(1, 6):
        raise ValueError("Supported: daily POP 60, travel rain forecast 0/1, discomfort ratings 1..5")
    z = model["intercept"] + np.dot(model["weights"], [rain, carry, wet])
    return round(100 / (1 + math.exp(-z)), 2)


def train(source):
    raw = pd.read_csv(source)
    def question(number):
        col, = [c for c in raw if c.startswith(f"{number}. ")]
        return raw[col]
    frames = []
    for rain, number in enumerate([3, 4]):
        choice = question(number).str.strip()
        assert choice.isin(["챙긴다", "챙기지 않는다", "고민된다"]).all()
        frames.append(pd.DataFrame(dict(person=raw.index, travel_rain_forecast=rain,
                                       carry_discomfort=question(5), wet_discomfort=question(6),
                                       target=choice.eq("챙긴다").astype(int))))
    df = pd.concat(frames, ignore_index=True)
    assert df[FEATURES[1:]].isin(range(1, 6)).all().all()
    y = df.target
    folds = list(GroupKFold(5).split(df, y, groups=df.person))
    metrics = {}
    for name, features in [("weather_only", FEATURES[:1]), ("combined", FEATURES)]:
        x = df[features]
        prob = np.full(len(df), np.nan)
        estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        for tr, te in folds:
            assert set(df.person.iloc[tr]).isdisjoint(df.person.iloc[te])
            estimator.fit(x.iloc[tr], y.iloc[tr])
            prob[te] = estimator.predict_proba(x.iloc[te])[:, 1]
        metrics[name] = dict(brier=brier_score_loss(y, prob), auc=roc_auc_score(y, prob),
                             accuracy=accuracy_score(y, prob >= .5))
    estimator.fit(x, y)
    scale, regression = estimator.steps[0][1], estimator.steps[1][1]
    weights = regression.coef_[0] / scale.scale_
    model = dict(features=FEATURES, weights=weights.tolist(),
                 intercept=float(regression.intercept_[0] - weights @ scale.mean_),
                 fixed_daily_pop=60, outcome="survey carry intention", respondents=len(raw),
                 rows=len(df), validation=metrics)
    assert np.allclose([score(model, 60, *row)/100 for row in x.to_numpy()],
                       estimator.predict_proba(x)[:,1], atol=.000051)
    OUTPUT.write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf8")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--survey", type=Path)
    parser.add_argument("--score", nargs=4, type=int, metavar=("DAILY_POP","TRAVEL_RAIN","CARRY","WET"))
    args = parser.parse_args()
    if args.score:
        print(score(json.loads(OUTPUT.read_text(encoding="utf8")), *args.score))
    else:
        source = args.survey or next((Path.home() / "Downloads").glob("*설문지 응답 시트1.csv"))
        model = train(source)
        print(model["validation"])
        for rain in (0, 1):
            print(rain, score(model, 60, rain, 3, 4))
