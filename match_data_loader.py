"""Module providing the data about matches (it downloads the data in case they are missing)."""
from collections import deque
import gc
import lzma
from pathlib import Path
import pickle
import random

import gdown
from riotwatcher import LolWatcher, ApiError

from env import RIOT_GAME_API


# Settings
summoner_region = 'na1'
match_region = "americas"
game_type = "RANKED_SOLO_5x5"
# tiers = ["DIAMOND", "PLATINUM", "GOLD", "SILVER", "BRONZE", "IRON"]
tiers = ["GOLD", "SILVER", "BRONZE"]
divisions = ["I", "II", "III", "IV"]
w = LolWatcher(RIOT_GAME_API)


SEED_FILE = Path("data/seed_player_ids.txt")
MATCHE_FILE = Path("data/matches.pkl.xz")
TMP_PLAYER_IDS = Path("data/tmp_player_ids.pkl.xz")
TMP_QUEUE = Path("data/tmp_queue.pkl.xz")
# URL with stored pre-downloaded data
DATA_URL = "https://drive.google.com/u/0/uc?id=1RbFKammoDtPr9vkLR46iseaKU3CxyLtC"


def _get_seed_player_ids() -> set:
    """Get 5 players from each league to perform further search for matches."""
    if SEED_FILE.exists():
        with SEED_FILE.open("r") as f:
            return set(f.read().split())

    player_ids = set()
    for tier in tiers:
        for division in divisions:
            players = w.league.entries(summoner_region, game_type, tier, division)
            # Select 5 random players from received players
            for p in random.choices(players, k=min(5, len(players))):
                try:
                    s = w.summoner.by_name(summoner_region, p["summonerName"])
                except ApiError as e:
                    print(f"API error - player skiped {p['summonerName']} ({tier}-{division})")
                    continue
                player_ids.add(s["puuid"])

    with SEED_FILE.open("wt+", encoding="utf-8") as f:
        f.write("\n".join(player_ids))

    print("Number of seed players:", len(player_ids))
    return player_ids


def _collect_matchlist(puuid):
    """Use API to collect ids of ranked solo 5x5 matches based on player id."""
    #  Perform 3 tries to collect matches
    for _ in range(3):
        try:
            # queue 420 should correspond to ranked solo 5x5 matches
            return w.match.matchlist_by_puuid(
                match_region, puuid, queue=420, start=0, count=50
            )
        except ApiError as e:
            print("Error collecting player's matches:", e)
    return []


def _collect_match(match_id):
    """Use API to collect match details based on match id."""
    for _ in range(3):
        try:
            return w.match.by_id(match_region, match_id)
        except ApiError as e:
            print("Error couldn't retriev match:", e)
    return None


def get_local_match_data():
    """Load data file with all matches."""
    if MATCHE_FILE.exists():
        with lzma.open(MATCHE_FILE, "r") as f:
            gc.disable()
            m =  pickle.load(f)
            gc.enable()
            return m
    return []


def _get_start_state():
    """Create init state or recover previous download state."""
    matches = get_local_match_data()

    if TMP_PLAYER_IDS.exists():
        with lzma.open(TMP_PLAYER_IDS, "r") as f:
            player_ids = pickle.load(f)
    else:
        player_ids = _get_seed_player_ids()

    if TMP_QUEUE.exists():
        with lzma.open(TMP_QUEUE, "r") as f:
            queue = pickle.load(f)
    else:
        queue = deque(player_ids)

    return matches, player_ids, queue


def _save_final_state(matches, player_ids, queue):
    """Save final state after downloading matches so it is possible to recover download."""
    with lzma.open(MATCHE_FILE, "w") as f:
        pickle.dump(matches, f)

    with lzma.open(TMP_PLAYER_IDS, "w") as f:
        pickle.dump(player_ids, f)

    with lzma.open(TMP_QUEUE, "w") as f:
        pickle.dump(queue, f)


def download_data(num_players=1e3):
    """Use seed players to download ranked games."""
    matches_list, player_ids, queue = _get_start_state()
    processed_matches = set(m["metadata"]["matchId"] for m in matches_list)

    for i in range(int(num_players)):
        player_id = queue.popleft()
        matches = _collect_matchlist(player_id)
        for match_id in [m for m in matches if m not in processed_matches]:
            match = _collect_match(match_id)
            if not match:
                continue

            matches_list.append(match)

            # Add players from match to search queue
            for puuid in match["metadata"]["participants"]:
                if puuid not in player_ids:
                    player_ids.add(puuid)
                    queue.append(puuid)

        if (i + 1) % 10 == 0:
            print(f"Saving matches... Number of players/matches: {i}/{len(matches_list)}")
            _save_final_state(matches_list, player_ids, queue)
            print(" -> Saved")

    _save_final_state(matches_list, player_ids, queue)
    return matches_list


def get_data():
    """Function for getting match data."""
    if not MATCHE_FILE.exists():
        gdown.download(DATA_URL, str(MATCHE_FILE), quiet=False)
    return get_local_match_data()


if __name__ == "__main__":
    download_data(num_players=1e4)
