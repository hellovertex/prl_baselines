import datetime
from typing import NamedTuple


class SnowieEpisode(NamedTuple):
    date: str
    game_id: int


snowie_episode = f"GameStart\n" \
    f"PokerClient: ExportFormat\n" \
    f"Date: {datetime.date.strftime(datetime.date.today(), '%d/%m/%y')}\n" \
    f"TimeZone: GMT\n" \
    f"Time: "