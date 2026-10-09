"""AIMLite Dataset: premierleague_prediction/data.py"""

from typing import Any, Dict, List, Optional
from aimlite import Dataset


class PremierLeagueDataset(Dataset):
    """Dataset containing historical and modern Premier League match results.

    Contains 2,660+ competitive fixtures spanning 2018/19 through 2024/25,
    tracking round numbers, fixtures, dates, goals scored, and season indicators.
    """

    filename = "PremierLeague.csv"

    def __init__(self, name: str = "premier_league_data", config: Optional[Any] = None) -> None:
        super().__init__(name=name, config=config)

    def load(self, source: Optional[Any] = None, **kwargs: Any) -> List[Dict[str, Any]]:
        """Ingests all fixture records into memory."""
        records = super().load(source=source, **kwargs)
        return records
