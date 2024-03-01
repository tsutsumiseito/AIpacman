import pandas as pd
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_enemy_distance(player_x, player_y, enemies_positions):
    # プレイヤーの位置
    player_position = {'x': player_x, 'y': player_y}
    # プレイヤーと各敵との距離を計算し、最も近い敵との距離を返す
    distances = [
        np.sqrt((player_position['x'] - enemy['x'])**2 + (player_position['y'] - enemy['y'])**2)
        for enemy in enemies_positions if enemy['active']
    ]
    return min(distances) if distances else np.nan  # 敵がいなければNaN

def load_data():
    logger.info('Loading data')
    with open('gameplay_data.json') as f:
        data = [json.loads(line) for line in f]

    logger.info('Converting data to DataFrame')
    df = pd.json_normalize(data)

    logger.info('Extracting features')
    # プレイヤーの位置
    df['player_x'] = df['state.player_position.x']
    df['player_y'] = df['state.player_position.y']
    # コイン取得数
    df['coins_collected'] = df['state.coins_collected']
    # 残りライフ
    df['lives_left'] = df['state.lives_left']
    # 敵との距離
    df['enemy_distance'] = df.apply(lambda row: calculate_enemy_distance(
                                row['state.player_position.x'], 
                                row['state.player_position.y'], 
                                row['state.enemies_positions']), axis=1)

    # 褒美獲得数（reward）
    df['rewards_collected'] = df['reward']

    features = df[['player_x', 'player_y', 'coins_collected', 'lives_left', 'enemy_distance', 'rewards_collected']]
    labels = df['action']

    logger.info('Saving preprocessed data')
    features.to_csv('csv/features.csv', index=False)
    labels.to_csv('csv/labels.csv', index=False)

    logger.info('Data saved')
    return features, labels

if __name__ == '__main__':
    load_data()
