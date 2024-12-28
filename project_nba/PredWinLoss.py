import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import joblib
# Read the data
data = pd.read_csv('games.csv')

# Filter playoff data
playoff_data = data[data['isRegular'] == 0]

# Filter regular season data (2017-2019)
regular_season_data = data[(data['isRegular'] == 1) & (data['seasonStartYear'].isin([2017, 2018, 2019]))]

# Calculate win-loss data (with additional features)
def calculate_win_loss(data):
    win_loss = []
    for _, row in data.iterrows():
        if row['pointsHome'] > row['pointsAway']:
            win_loss.append({'team1': row['homeTeam'], 'team2': row['awayTeam'], 'win': 1, 'home_team': row['homeTeam'], 'away_team': row['awayTeam']})
            win_loss.append({'team1': row['awayTeam'], 'team2': row['homeTeam'], 'win': 0, 'home_team': row['homeTeam'], 'away_team': row['awayTeam']})
        else:
            win_loss.append({'team1': row['homeTeam'], 'team2': row['awayTeam'], 'win': 0, 'home_team': row['homeTeam'], 'away_team': row['awayTeam']})
            win_loss.append({'team1': row['awayTeam'], 'team2': row['homeTeam'], 'win': 1, 'home_team': row['homeTeam'], 'away_team': row['awayTeam']})
    return pd.DataFrame(win_loss)

win_loss_data = calculate_win_loss(regular_season_data)

# Prepare the dataset for training with additional features
def prepare_dataset(win_loss_data, teams, year):
    X = []
    y = []
    
    # Add team statistics such as average points per game, win/loss record, etc.
    for team1 in teams:
        for team2 in teams:
            if team1 != team2:
                matches = win_loss_data[(win_loss_data['team1'] == team1) & (win_loss_data['team2'] == team2)]
                if not matches.empty:
                    # Extract win/loss info along with additional features
                    win_percentage_team1 = win_loss_data[win_loss_data['team1'] == team1]['win'].mean()
                    win_percentage_team2 = win_loss_data[win_loss_data['team1'] == team2]['win'].mean()
                    
                    features = [
                        # Example: Win percentage of team1 and team2
                        win_percentage_team1, 
                        win_percentage_team2,
                        # Home vs Away performance could also be added here
                    ]
                    X.append(features)
                    y.append(matches['win'].values[0])  # Win or loss
    return np.array(X), np.array(y)

models = {
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'Support Vector Machine': SVC(probability=True)
}

# Train and predict for each model
for name, model in models.items():
    print(f'Training {name} model...')
    accuracies = []
    
    # Iterate over each year (2017-2019)
    for year in [2017, 2018, 2019]:
        playoff_teams = playoff_data[playoff_data['seasonStartYear'] == year]['homeTeam'].unique()
        X_train, y_train = prepare_dataset(win_loss_data, playoff_teams, year)
        if len(X_train) > 0:
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_val)
            joblib.dump( model,f'{year}_{name}_model.pkl')
            
            accuracy = accuracy_score(y_val, y_pred)
            accuracies.append(accuracy)
        
            # Predict the champion for the year
            team_predictions = {team: model.predict_proba([[win_loss_data[win_loss_data['team1'] == team]['win'].mean(), 
                                                            win_loss_data[win_loss_data['team2'] == team]['win'].mean()]])[:, 1].mean()
                                for team in playoff_teams}
            
            predicted_champion = max(team_predictions, key=team_predictions.get)
            print(f'{year} Champion Prediction ({name}): {predicted_champion}')
    
    print(f'{name} model average accuracy: {np.mean(accuracies):.2f}')
    
    # Visualize the predictions for each year
    for year in [2017, 2018, 2019]:
        playoff_teams = playoff_data[playoff_data['seasonStartYear'] == year]['homeTeam'].unique()
        
        # Calculate the predicted probabilities for each team
        team_predictions = {team: model.predict_proba([[win_loss_data[win_loss_data['team1'] == team]['win'].mean(), 
                                                        win_loss_data[win_loss_data['team2'] == team]['win'].mean()]])[:, 1].mean()
                            for team in playoff_teams}
        
        # Plotting the probabilities
        plt.figure(figsize=(10, 6))
        plt.bar(team_predictions.keys(), team_predictions.values())
        plt.title(f'{year} Playoff Championship Prediction ({name})')
        plt.xlabel('Team')
        plt.ylabel('Championship Probability')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{name}_prediction_{year}.png')
        plt.show()



