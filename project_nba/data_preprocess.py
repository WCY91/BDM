import numpy as np
import re
import pandas as pd

#Merge_inner = pd.merge(customer, TXN, how='inner', on=['ID'])

def convert_salary_to_number(s):
    s = re.sub(r'[\$,]', '', s)
    return int(s) / 1000

def time_to_minutes(time_str):
    if ':' in time_str:
        parts = time_str.split(':')
        if len(parts) == 3:  
            hours, minutes, seconds = int(parts[0]) , int(parts[1]), int(parts[2])
            return int(hours * 60 + minutes + seconds / 60)
        elif len(parts) == 2:  
            minutes, seconds = int(parts[0]) , int(parts[1])
            return int(minutes + seconds / 60)
    return 0

def remove_box_scores_non_number(df):
    cols = ["FG", "FGA", "3P", "3PA", "FT", "FTA", "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')
    df.dropna(subset=cols, inplace=True)
    df[cols] = df[cols].astype(int)
    return df

def generate_target_label(num):
    if num < 300:
        return int(num/50)
    else:
        return int(num / 1500) + 6
        

all_time_salaries_df = pd.read_csv("dataset\\archive\\salaries.csv",encoding='utf-8', usecols=['playerName', 'seasonStartYear','inflationAdjSalary'])
all_time_salaries_df.dropna()
all_time_salaries_df['inflationAdjSalary'] = all_time_salaries_df['inflationAdjSalary'].apply(convert_salary_to_number)
salaries_df = all_time_salaries_df.groupby('playerName').mean().astype(int).sort_values(by=['inflationAdjSalary'], ascending=False)
#salaries_df.to_csv("./cleaned_dataset/salaries.csv")
print(len(salaries_df))

all_time_player_scores_df = pd.read_csv("dataset\\archive\\boxscore.csv",encoding='utf-8',usecols=[i for i in range(19)])
all_time_player_scores_df.dropna()
all_time_player_scores_df["MP"]=all_time_player_scores_df["MP"].apply(time_to_minutes)
player_scores_df = remove_box_scores_non_number(all_time_player_scores_df)
player_scores_df = all_time_player_scores_df.groupby("playerName").mean().astype(int).sort_values(by=['MP'],ascending=False)
del player_scores_df['game_id']
#player_scores_df.to_csv("./cleaned_dataset/player_scores_df.csv")

salaries_and_scores_df = pd.merge(salaries_df,player_scores_df,how='inner',on=['playerName'])
salaries_and_scores_df['target'] = salaries_and_scores_df['inflationAdjSalary'].apply(generate_target_label)
salaries_and_scores_df.to_csv("./cleaned_dataset/salaries_and_scores.csv")


game_id_and_sesson_df = pd.read_csv("dataset\\archive\\games.csv",encoding='utf-8', usecols=['seasonStartYear', 'game_id'])
all_time_scores_game_session_df = pd.merge(all_time_player_scores_df,game_id_and_sesson_df,how="left",on=["game_id"])
del all_time_scores_game_session_df['game_id']
grouped_avg_df = all_time_scores_game_session_df.groupby(['playerName', 'seasonStartYear']).mean().reset_index().round(1).sort_values(by=['MP'],ascending=False)
print(all_time_salaries_df)

all_time_salaries_and_avg_scores_df = pd.merge(grouped_avg_df,all_time_salaries_df,how='inner',on=['playerName','seasonStartYear'])
all_time_salaries_and_avg_scores_df.to_csv("./cleaned_dataset/all_time_salaries_and_scores.csv")

player_info_df = pd.read_csv("dataset\\archive\\player_info.csv",encoding='utf-8',usecols=["playerName","Pos"])
info_and_scores_df = pd.merge(player_scores_df,player_info_df,how="inner",on=["playerName"])
info_and_scores_df.to_csv("./cleaned_dataset/info_and_scores.csv")





