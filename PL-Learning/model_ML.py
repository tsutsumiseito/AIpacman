import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from preprocess_ML import load_data  # preprocess.pyからのデータロード関数

# データのロード
features, labels = load_data()

# 特徴量のスケーリング
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42)

# SMOTEでデータの不均衡を処理
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# 勾配ブースティングモデルのハイパーパラメータチューニング
parameters = {
    'n_estimators': [50, 100, 200],  # 決定木の数
    'max_depth': [3, 5, 7],  # 決定木の深さ
    'learning_rate': [0.01, 0.05, 0.1]  # 学習率
}
gbc = GradientBoostingClassifier(random_state=42)
clf = GridSearchCV(gbc, parameters, cv=5, scoring='accuracy')
clf.fit(X_train_smote, y_train_smote)

# 最適なパラメータとクロスバリデーションのスコアを表示
print(f"Best parameters: {clf.best_params_}")
print(f"Best cross-validation score: {clf.best_score_}")

# テストデータでの予測と評価
predictions = clf.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions)}")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
