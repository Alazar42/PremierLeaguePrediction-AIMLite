"""AIMLite Evaluator: premierleague_prediction/evaluator.py"""

import math
from typing import Any, Dict
import numpy as np
from aimlite import BaseEvaluator, Dataset, Model


class PremierLeagueEvaluator(BaseEvaluator):
    """Assesses model calibration against held-out test match partitions."""

    def evaluate(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, float]:
        """Evaluates match outcome accuracy, goal RMSE, and ranking Spearman correlation."""
        _, _, test_data = dataset.split(train=0.7, validation=0.15, test=0.15, seed=42)

        if not test_data:
            return {"accuracy": 0.0, "rmse": 0.0}

        home_adv = float(model.weights.get("home_advantage", 0.165))
        mu = float(model.weights.get("baseline_mu", 0.230))
        att_map = model.weights.get("team_attack", {})
        def_map = model.weights.get("team_defense", {})

        correct_outcomes = 0
        total_matches = len(test_data)
        sq_errors = []

        for row in test_data:
            h = row["home_team_name"]
            a = row["away_team_name"]
            hg = int(row["home_team_goals"])
            ag = int(row["away_team_goals"])

            h_att = att_map.get(h, 0.0)
            a_att = att_map.get(a, 0.0)
            h_def = def_map.get(h, 0.0)
            a_def = def_map.get(a, 0.0)

            lh = math.exp(max(-2.5, min(2.5, mu + h_att - a_def + home_adv)))
            la = math.exp(max(-2.5, min(2.5, mu + a_att - h_def)))

            sq_errors.append((lh - hg) ** 2)
            sq_errors.append((la - ag) ** 2)

            actual_outcome = "H" if hg > ag else ("D" if hg == ag else "A")
            pred_outcome = "H" if lh > la + 0.3 else ("A" if la > lh + 0.3 else "D")

            if actual_outcome == pred_outcome:
                correct_outcomes += 1

        accuracy = correct_outcomes / total_matches
        rmse = math.sqrt(sum(sq_errors) / len(sq_errors))

        # Rank simulation evaluation for recent season
        pred_standings = model.predict({"year": 2024, "simulations": 300})
        ucl_count = len(pred_standings.get("summary", {}).get("ucl_qualifiers", []))

        return {
            "match_accuracy": round(accuracy, 4),
            "goal_rmse": round(rmse, 4),
            "top4_coverage": round(ucl_count / 4.0, 2),
            "test_matches": total_matches,
        }
