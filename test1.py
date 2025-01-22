import pickle

with open('folds.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

print(loaded_data)