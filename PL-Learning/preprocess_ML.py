import pandas as pd
import json
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_enemy_distance(player_x, player_y, enemies_positions):
    player_position = {'x': player_x, 'y': player_y}
    distances = [
        np.sqrt((player_position['x'] - enemy['x'])**2 + (player_position['y'] - enemy['y'])**2)
        for enemy in enemies_positions if enemy['active']
    ]
    return min(distances) if distances else np.nan

def load_data():
    logger.info('Loading data')
    with open('gameplay_data.json') as f:
        data = [json.loads(line) for line in f]

    logger.info('Converting data to DataFrame')
    df = pd.json_normalize(data)

    logger.info('Extracting features')
    df['player_x'] = df['state.player_position.x']
    df['player_y'] = df['state.player_position.y']
    df['coins_collected'] = df['state.coins_collected']
    df['lives_left'] = df['state.lives_left']
    df['enemy_distance'] = df.apply(lambda row: calculate_enemy_distance(row['state.player_position.x'], row['state.player_position.y'], row['state.enemies_positions']), axis=1)
    df['rewards_collected'] = df['reward']

    features = df[['player_x', 'player_y', 'coins_collected', 'lives_left', 'enemy_distance', 'rewards_collected']]
    labels = df['action']

    script_dir = os.path.dirname(__file__)
    csv_directory = os.path.join(script_dir, 'csv')
    if not os.path.exists(csv_directory):
        os.makedirs(csv_directory)

    features_path = os.path.join(csv_directory, 'features.csv')
    labels_path = os.path.join(csv_directory, 'labels.csv')

    logger.info('Saving preprocessed data')
    features.to_csv(features_path, index=False)
    labels.to_csv(labels_path, index=False)

    logger.info('Data saved')
    return features, labels

if __name__ == '__main__':
    load_data()
