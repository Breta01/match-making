"""Module providing the data about players (it downloads the data in case they are missing)."""
from pathlib import Path

import pandas as pd
from riotwatcher import LolWatcher, ApiError

from match_data_loader import get_data
from env import RIOT_GAME_API


watcher = LolWatcher(RIOT_GAME_API)
match_region = "americas"
summoner_region = 'na1'

base_path = Path(__file__).parent
PROCESSED_PLAYERS_FILE = base_path.joinpath("data/players_processed.csv")
RANK_PLAYERS_FILE = base_path.joinpath("data/players_rank.csv")


def get_local_player_rank_data():
    """Obtain data about rank of a player."""
    df = pd.read_csv(RANK_PLAYERS_FILE)

    rank_map = {
        "IRON":        1,
        "BRONZE":      2,
        "SILVER":      3,
        "GOLD":        4,
        "PLATINUM":    5,
        "DIAMOND":     6,
        "MASTER":      7,
        "GRANDMASTER": 8,
        "CHALLENGER":  9,
    }

    for col in rank_map.keys():
        df[f"tier_{col}"] = 0

    dummies = pd.get_dummies(df["tier"], prefix="tier")
    df[dummies.columns] = dummies

    df.replace({"tier": rank_map}, inplace=True)
    df = df.set_index(['summonerId'])

    return df[filter(lambda x: "tier" in x, df.columns)]


def get_local_player_data():
    """Load data file with all players."""
    df = pd.read_csv(PROCESSED_PLAYERS_FILE)
    df.dropna(inplace=True)
    # Reindex to puuid and drop original indexes in column "Unnamed: 0"
    df = df.set_index(['puuid']).drop(columns="Unnamed: 0")

    df_ranks = get_local_player_rank_data()
    return df.join(df_ranks, rsuffix="x", on="summoner_id")


def _get_matches_data(matches, puuid) :
    games_df = pd.DataFrame()
    for match in matches:
        try :
            game_df = pd.DataFrame(watcher.match.by_id(match_region, match)['info']['participants'])
            game_df = game_df[game_df['puuid'] == puuid]
            first_column = game_df.pop('puuid')
            game_df.insert(0, 'puuid', first_column)
            games_df = pd.concat([games_df, game_df], ignore_index = True)

        except ApiError as e:
            print("Error collecting player's matches:", e)
            
    return games_df


def get_mean_max_min(col) :
    return col.mean(), col.max(), col.min()


def _calculate_aggregations(players_info, players_data, full_matches_data, number_of_matches):
    unique_puuids = full_matches_data['puuid'].unique()

    players_dict_data = []
    for puuid in unique_puuids :
        
        player_id = players_info[players_info['puuid'] == puuid]['id'].values[0]
        
        player_data = full_matches_data[full_matches_data['puuid'] == puuid]
        
        mean_kills, max_kills, min_kills =  get_mean_max_min(player_data['kills'])
        
        mean_assists, max_assists, min_assists =  get_mean_max_min(player_data['assists'])

        mean_deaths, max_deaths, min_deaths = get_mean_max_min(player_data['deaths'])
        
        position_top = 0
        position_carry = 0
        position_mid = 0
        position_jungle = 0
        position_support = 0
        row = 0
        for position in player_data['individualPosition']:
            if position == 'TOP':
                position_top += 1
            elif position == 'MIDDLE':
                position_mid += 1
            elif position == 'JUNGLE':
                position_jungle += 1
            elif position == 'BOTTOM':
                position_carry += 1
            elif position == 'UTILITY':
                position_support += 1
            elif position == 'Invalid':
                second_position = player_data['lane'].iloc[row]
                if second_position == 'BOTTOM' :
                    third_position = player_data['role'].iloc[row]
                    if third_position == 'SOLO' or third_position == 'DUO':
                        position_carry += 1
                    elif third_position == 'SUPPORT':
                        position_support += 1
                elif second_position == 'TOP':
                    position_top += 1
                elif second_position == 'MIDDLE':
                    position_mid += 1
                elif second_position == 'JUNGLE':
                    position_jungle += 1
                elif second_position == 'NONE':
                    third_position = player_data['role'].iloc[row]
                    if third_position == 'SUPPORT':
                        position_support += 1
                    elif third_position == 'DUO':
                        position_carry += 1
            row = row + 1
            
            
        position_top = position_top / number_of_matches
        position_mid = position_mid / number_of_matches
        position_carry = position_carry / number_of_matches
        position_jungle = position_jungle / number_of_matches
        position_support = position_support / number_of_matches
        
        win_ratio = len(player_data['win'][player_data['win'] == True]) / number_of_matches
        
        gold_per_time = player_data['goldEarned'] / player_data['timePlayed']
        mean_gold_per_time, max_gold_per_time, min_gold_per_time = get_mean_max_min(gold_per_time)

        total_damage_to_champions_per_time = player_data['totalDamageDealtToChampions'] / player_data['timePlayed']
        mean_total_damage_to_champions_per_time, max_total_damage_to_champions_per_time, min_total_damage_to_champions_per_time = get_mean_max_min(total_damage_to_champions_per_time)
        
        total_damage_taken_per_time = player_data['totalDamageTaken'] / player_data['timePlayed']
        mean_total_damage_taken_per_time, max_total_damage_taken_per_time, min_total_damage_taken_per_time = get_mean_max_min(total_damage_taken_per_time)    
        
        total_minions_killed_per_time = player_data['totalMinionsKilled'] / player_data['timePlayed']
        mean_total_minions_killed_per_time, max_total_minions_killed_per_time, min_total_minions_killed_per_time = get_mean_max_min(total_minions_killed_per_time)
        
        total_wards_per_time = player_data['wardsPlaced'] / player_data['timePlayed']
        mean_total_wards_per_time, max_total_wards_per_time, min_total_wards_per_time = get_mean_max_min(total_wards_per_time)

        total_neutral_minions_killed_per_time = player_data['neutralMinionsKilled'] / player_data['timePlayed']
        mean_total_neutral_minions_killed_per_time, max_total_neutral_minions_killed_per_time, min_total_neutral_minions_killed_per_time = get_mean_max_min(total_neutral_minions_killed_per_time)

        turret_take_downs = player_data['turretTakedowns']
        mean_turret_take_downs, max_turret_take_downs, min_turret_take_downs = get_mean_max_min(turret_take_downs)
        
        turrets_lost = player_data['turretsLost']
        mean_turrets_lost, max_turrets_lost, min_turrets_lost = get_mean_max_min(turrets_lost)
        
        killing_spree = player_data['killingSprees']
        mean_killing_spree, max_killing_spree, min_killing_spree = get_mean_max_min(killing_spree)
        
        champ_level_per_time = player_data['champLevel'] / player_data['timePlayed']
        mean_champ_level_per_time, max_champ_level_per_time, min_champ_level_per_time = get_mean_max_min(champ_level_per_time)
        
        player_dict_data = {
            'summoner_id' : player_id,
            'puuid' : puuid,
            'mean_kills' : mean_kills,
            'max_kills' : max_kills,
            'min_kills' : min_kills,
            'mean_assists' : mean_assists,
            'max_assists' : max_assists,
            'min_assists' : min_assists,
            'mean_deaths' : mean_deaths,
            'max_deaths' : max_deaths,
            'min_deaths' : min_deaths,
            'position_top' : position_top,
            'position_mid' : position_mid,
            'position_jungle' : position_jungle,
            'position_carry' : position_carry,
            'position_support' : position_support,
            'win_ratio' : win_ratio,
            'mean_gold_per_time' : mean_gold_per_time,
            'max_gold_per_time' : max_gold_per_time,
            'min_gold_per_time' : min_gold_per_time,
            'mean_total_damage_to_champions_per_time' : mean_total_damage_to_champions_per_time,
            'max_total_damage_to_champions_per_time' : max_total_damage_to_champions_per_time,
            'min_total_damage_to_champions_per_time' : min_total_damage_to_champions_per_time,
            'mean_total_damage_taken_per_time' : mean_total_damage_taken_per_time,
            'max_total_damage_taken_per_time' : max_total_damage_taken_per_time,
            'min_total_damage_taken_per_time' : min_total_damage_taken_per_time,
            'mean_total_minions_killed_per_time' : mean_total_minions_killed_per_time,
            'max_total_minions_killed_per_time' : max_total_minions_killed_per_time,
            'min_total_minions_killed_per_time' : min_total_minions_killed_per_time,
            'mean_total_wards_per_time' : mean_total_wards_per_time,
            'max_total_wards_per_time' : max_total_wards_per_time,
            'min_total_wards_per_time' : min_total_wards_per_time,
            'mean_total_neutral_minions_killed_per_time' :  mean_total_neutral_minions_killed_per_time,
            'max_total_neutral_minions_killed_per_time' :  max_total_neutral_minions_killed_per_time,
            'min_total_neutral_minions_killed_per_time' :  min_total_neutral_minions_killed_per_time,
            'mean_turret_take_downs' : mean_turret_take_downs,
            'max_turret_take_downs' : max_turret_take_downs,
            'min_turret_take_downs' : min_turret_take_downs,
            'mean_turrets_lost' : mean_turrets_lost,
            'max_turrets_lost' : max_turrets_lost,
            'min_turrets_lost' : min_turrets_lost,
            'mean_killing_spree' : mean_killing_spree,
            'max_killing_spree' : max_killing_spree,
            'min_killing_spree' : min_killing_spree,
            'mean_champ_level_per_time' : mean_champ_level_per_time, 
            'max_champ_level_per_time' : max_champ_level_per_time,  
            'min_champ_level_per_time' : min_champ_level_per_time,
            'wins' : players_data[players_data['summonerId'] == player_id]['wins'].mean(),
            'losses' : players_data[players_data['summonerId'] == player_id]['losses'].mean(),
            'player_level' : player_data['summonerLevel'].max()
        }
        players_dict_data.append(player_dict_data)
    


def download_data(matches):
    players_matches = {}
    for match_idx, match in enumerate(matches):
        for player_id in match["metadata"]["participants"]:
            if player_id not in players_matches:
                players_matches[player_id] = []
            players_matches[player_id].append(match_idx)

    # Collect data for mapping summoner puuid to sommoner_id
    players_info = []
    for id in players_matches.keys():
        player_info = watcher.summoner.by_puuid(summoner_region,id)
        players_info.append(player_info)

    players_info = pd.DataFrame(players_info)

    # Collect players league data
    players_data = []
    for i, id in enumerate(players_info['id']):
        try:
            player_data = watcher.league.by_summoner(summoner_region,id) 
            if len(player_data) > 0:
                players_data.append(player_data[0])
        except ApiError as e:
            print("Error collecting player's matches:", e)
        
        print(f"Current progress: {i+1 / len(players_matches) * 100 : .2f}%", end="\r")   

    players_data_csv = pd.DataFrame(players_data)
    players_data_csv.to_csv(RANK_PLAYERS_FILE)

    # Download 5 past matches for each player
    number_of_matches = 5
    full_matches_data = pd.DataFrame()
    for i, puuid in enumerate(players_info['puuid']):
        if puuid not in full_matches_data['puuid'].values:
            try :
                player_matches = watcher.match.matchlist_by_puuid(match_region, puuid, count=number_of_matches)
                matches_data = _get_matches_data(player_matches, puuid)
                full_matches_data = pd.concat([full_matches_data, matches_data], ignore_index=True)

            except ApiError as e:
                    print("Error collecting player's matches:", e)
                
        print("Current progress:", i+1, "/", len(players_info['puuid']), end="\r")   

    # Calculate aggregated statistics
    players_agreg = _calculate_aggregations(player_info, players_data, full_matches_data, number_of_matches)
    pd.DataFrame(players_agreg).to_csv(PROCESSED_PLAYERS_FILE)


if __name__ == "__main__":
    matches = get_data()
    download_data(matches)
    print("Collecting data finished!")
