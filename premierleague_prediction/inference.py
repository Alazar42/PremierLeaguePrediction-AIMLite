"""AIMLite Inference Handler: premierleague_prediction/inference.py"""

from typing import Any, Dict, List, Optional, Union
from aimlite import BaseInference, Model


class PremierLeagueInference(BaseInference):
    """Production serving handler for Premier League standings and championship predictions."""

    def run(self, model: Model, raw_input: Any, **kwargs: Any) -> Dict[str, Any]:
        """Normalizes request payloads, executes season simulation, and packages standings."""
        # 1. Parse and extract year from various input shapes
        year = 2026
        simulations = 500

        if isinstance(raw_input, dict):
            # Might be wrapped under {"inputs": ...} or {"features": ...} or direct {"year": 2026}
            data = raw_input.get("inputs") or raw_input.get("features") or raw_input
            if isinstance(data, dict):
                raw_y = data.get("year", data.get("season", 2026))
                simulations = int(data.get("simulations", 500))
            else:
                raw_y = data
            try:
                year = int(str(raw_y).replace("-", "/").split("/")[0])
            except Exception:
                year = 2026
        elif isinstance(raw_input, (int, float)):
            year = int(raw_input)
        elif isinstance(raw_input, str):
            try:
                year = int(raw_input.strip().replace("-", "/").split("/")[0])
            except Exception:
                year = 2026

        if year < 100:
            year += 2000

        # Enforce supported bounds 2019 - 2029
        year = max(2019, min(2029, year))

        prediction_result = model.predict({"year": year, "simulations": simulations})
        return prediction_result

    def get_available_seasons(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Returns preset season choices for rapid client selection."""
        seasons = [
            {"year": 2026, "label": "2025/26 (Upcoming)", "status": "forecast", "default": True},
            {"year": 2025, "label": "2024/25 Season", "status": "active"},
            {"year": 2024, "label": "2023/24 Season", "status": "historical"},
            {"year": 2023, "label": "2022/23 Season", "status": "historical"},
            {"year": 2022, "label": "2021/22 Season", "status": "historical"},
            {"year": 2021, "label": "2020/21 Season", "status": "historical"},
            {"year": 2020, "label": "2019/20 Season", "status": "historical"},
            {"year": 2019, "label": "2018/19 Season", "status": "historical"},
            {"year": 2027, "label": "2026/27 (Future)", "status": "projection"},
            {"year": 2028, "label": "2027/28 (Future)", "status": "projection"},
            {"year": 2029, "label": "2028/29 (Future)", "status": "projection"},
        ]
        return {"status": "success", "seasons": seasons}

    def get_routes(self) -> Dict[str, Any]:
        """Registers custom HTTP routes."""
        return {
            "POST /predict": self.run,
            "POST /": self.run,
            "GET /api/seasons": self.get_available_seasons,
            "GET /health": self.health,
            "GET /info": self.info,
        }
