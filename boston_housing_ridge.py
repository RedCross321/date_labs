from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from keras.datasets import boston_housing


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std
# housing = fetch_california_housing()

# train_data, test_data, train_targets, test_targets = train_test_split(
#     housing.data, housing.target, test_size=0.2, random_state=42)

# scaler = StandardScaler()
# train_data = scaler.fit_transform(train_data)

# test_data = scaler.transform(test_data)

model = Ridge(alpha=100)
model.fit(train_data, train_targets)

test_predictions = model.predict(test_data)
print(model.score(test_data, test_targets))
plt.figure(figsize=(10, 6))
plt.scatter(test_targets, test_predictions, alpha=0.5)
plt.plot([0, max(test_targets)], [0, max(test_targets)], 'r--')
plt.xlabel('Фактические цены')
plt.ylabel('Предсказанные цены')
plt.title('Сравнение предсказаний модели Ridge с фактическими ценами')
plt.grid(True)
plt.savefig('ridge_predictions.png')
plt.show()

