import joblib
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from tensorboardX import SummaryWriter
from torch.utils.data import random_split

from prl.baselines.supervised_learning.training.dataset import InMemoryDataset

input_dir = "/home/sascha/Documents/github.com/prl_baselines/data/03_preprocessed/0.25-0.50"
dataset = InMemoryDataset(input_dir)
total_len = len(dataset)
train_len = int(total_len * 0.85)
test_len = int(total_len * 0.1)
val_len = int(total_len * 0.05)
# add residuals to val_len to add up to total_len
val_len += total_len - (int(train_len) + int(test_len) + int(val_len))
# set seed
gen = torch.Generator().manual_seed(1)

# Splitting the dataset into train, test, and validation sets
train, test, val = random_split(dataset, [train_len, test_len, val_len], generator=gen)

# Defining the Random Forest model
# # Loading the saved model
# filename = 'random_forest.sav'
# best_model = joblib.load(filename)
model = RandomForestClassifier()

# Defining the hyperparameters to tune
param_grid = {
    'n_estimators': [10, 50, 100, 200],
    'max_depth': [5, 10, 15, 20],
    'min_samples_split': [2, 5, 10, 20],
}

# Initializing the SummaryWriter
writer = SummaryWriter()

# Performing cross-validation and hyperparameter tuning using GridSearchCV
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(train.x, train.y)
# # Evaluating the updated model's generalization performance
# train_predictions = best_model.predict(train)
# test_predictions = best_model.predict(test)
#
# train_accuracy = accuracy_score(train_labels, train_predictions)
# test_accuracy = accuracy_score(test_labels, test_predictions)
# train_precision = precision_score(train_labels, train_predictions)
# test_precision = precision_score(test_labels, test_predictions)
# train_recall = recall_score(train_labels, train_predictions)
# test_recall = recall_score(test_labels, test_predictions)
# train_f1 = f1_score(train_labels, train_predictions)
# test_f1 = f1_score(test_labels, test_predictions)
#
# # Writing the metrics to TensorBoard
# writer.add_scalar('Train/Accuracy', train_accuracy, 0)
# writer.add_scalar('Test/Accuracy', test_accuracy, 0)
# writer.add_scalar('Train/Precision', train_precision, 0)
# writer.add_scalar('Test/Precision', test_precision, 0)
# writer.add_scalar('Train/Recall', train_recall, 0)
# writer.add_scalar('Test/Recall', test_recall, 0)
# writer.add_scalar('Train/F1', train_f1, 0)
# writer.add_scalar('Test/F1', test_f1, 0)

writer.close()
# Selecting the best model
best_model = grid_search.best_estimator_

# Saving the model
filename = 'random_forest.sav'
joblib.dump(best_model, filename)

print("Random Forest model saved as", filename)

# Evaluating the model's generalization performance using cross-validation
scores = cross_val_score(best_model, test.x, test.y, cv=5)
print("Cross validation scores: ", scores)
print("Mean cross validation score: ", scores.mean())
