"""AIMLite Trainer: premierleague_prediction/trainer.py"""

import math
from typing import Any, Dict, List
import numpy as np
from scipy.optimize import minimize
from aimlite import BaseTrainer, Dataset, Model


class PremierLeagueTrainer(BaseTrainer):
    """Calibrates team attack and defense ratings via Dixon-Coles Poisson maximum likelihood."""

    def fit(self, model: Model, dataset: Dataset, **kwargs: Any) -> Dict[str, Any]:
        """Fits model parameters on historical Premier League fixture records."""
        train_data, val_data, _ = dataset.split(train=0.8, validation=0.2, test=0.0, seed=42)

        # Collect unique teams
        all_teams = sorted(list(set(
            [r["home_team_name"] for r in train_data] + [r["away_team_name"] for r in train_data]
        )))
        team_idx = {t: i for i, t in enumerate(all_teams)}
        n_teams = len(all_teams)

        # Prepare vectorized training matrices
        h_idx = np.array([team_idx[r["home_team_name"]] for r in train_data])
        a_idx = np.array([team_idx[r["away_team_name"]] for r in train_data])
        h_goals = np.array([int(r["home_team_goals"]) for r in train_data])
        a_goals = np.array([int(r["away_team_goals"]) for r in train_data])

        seasons = np.array([int(r.get("season", 2024)) for r in train_data])
        max_s = seasons.max() if len(seasons) > 0 else 2025
        # Exponential recency weighting
        weights = np.exp(0.25 * (seasons - max_s))

        def loss_function(params: np.ndarray) -> float:
            home_adv = params[0]
            mu = params[1]
            att = params[2 : 2 + n_teams]
            defn = params[2 + n_teams : 2 + 2 * n_teams]

            lh = np.exp(np.clip(mu + att[h_idx] - defn[a_idx] + home_adv, -3.0, 3.0))
            la = np.exp(np.clip(mu + att[a_idx] - defn[h_idx], -3.0, 3.0))

            # Poisson negative log-likelihood: sum(lambda - y * log(lambda))
            nll_h = weights * (lh - h_goals * np.log(lh + 1e-9))
            nll_a = weights * (la - a_goals * np.log(la + 1e-9))

            # L2 regularization to anchor zero-sum centering
            reg = 0.05 * (np.sum(att**2) + np.sum(defn**2))
            return float(np.sum(nll_h) + np.sum(nll_a) + reg)

        init_params = np.zeros(2 + 2 * n_teams)
        init_params[0] = 0.165  # home advantage
        init_params[1] = 0.230  # baseline mu

        opt_result = minimize(
            loss_function,
            init_params,
            method="L-BFGS-B",
            options={"maxiter": 120, "ftol": 1e-6},
        )

        opt_params = opt_result.x
        home_adv = float(opt_params[0])
        mu = float(opt_params[1])
        att = opt_params[2 : 2 + n_teams]
        defn = opt_params[2 + n_teams : 2 + 2 * n_teams]

        # Validation loss evaluation
        val_h_idx = np.array([team_idx.get(r["home_team_name"], 0) for r in val_data])
        val_a_idx = np.array([team_idx.get(r["away_team_name"], 0) for r in val_data])
        val_hg = np.array([int(r["home_team_goals"]) for r in val_data])
        val_ag = np.array([int(r["away_team_goals"]) for r in val_data])

        val_lh = np.exp(np.clip(mu + att[val_h_idx] - defn[val_a_idx] + home_adv, -3.0, 3.0))
        val_la = np.exp(np.clip(mu + att[val_a_idx] - defn[val_h_idx], -3.0, 3.0))
        val_loss = float(np.mean((val_lh - val_hg)**2 + (val_la - val_ag)**2))

        # Update model weights dictionary
        attack_dict = {all_teams[i]: float(att[i]) for i in range(n_teams)}
        defense_dict = {all_teams[i]: float(defn[i]) for i in range(n_teams)}

        model.weights["home_advantage"] = home_adv
        model.weights["baseline_mu"] = mu
        model.weights["team_attack"] = attack_dict
        model.weights["team_defense"] = defense_dict
        model.weights["teams"] = all_teams
        model.weights["train_loss"] = float(opt_result.fun)
        model.weights["val_loss"] = val_loss

        return {
            "status": "completed",
            "train_loss": round(float(opt_result.fun), 2),
            "val_loss": round(val_loss, 4),
            "home_advantage": round(home_adv, 3),
            "baseline_mu": round(mu, 3),
            "teams_calibrated": n_teams,
            "train_samples": len(train_data),
            "val_samples": len(val_data),
        }
