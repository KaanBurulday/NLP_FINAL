import DistanceCalculator

class KNN:
    def __init__(self, **kwargs):
        self.distances = kwargs.get('distances', None)
        self.train_data = kwargs.get('train_data', None)
        if self.train_data is None:
            raise Exception('train_data must not be None')
        self.k = kwargs.get('k', 5)
        self.tf_idf_table = kwargs.get('tf_idf_table', None)
        if self.tf_idf_table is None:
            raise Exception('tf_idf_table must not be None')
        self.method = kwargs.get('method', DistanceCalculator.calculate_distances_between_query_and_train)


    def predict(self, query):
        self.distances = self.method(train_data=self.train_data, query=query, tf_idf_table=self.tf_idf_table)
        top_k_neighbors = dict(sorted(self.distances.items(), key=lambda item: item[1][0], reverse=True)[:self.k])
        top_k_classes = [neighbor[1][1] for neighbor in top_k_neighbors.items()]

        predicted_label = max(set(top_k_classes), key=top_k_classes.count)
        return predicted_label


    def predict_bulk(self, test_data):
        predictions = []
        actual = []
        for query in test_data:
            self.distances = self.method(train_data=self.train_data, query=query, tf_idf_table=self.tf_idf_table)
            top_k_neighbors = dict(sorted(self.distances.items(), key=lambda item: item[1][0], reverse=True)[:self.k])
            top_k_classes = [neighbor[1][1] for neighbor in top_k_neighbors.items()]

            predicted_label = max(set(top_k_classes), key=top_k_classes.count)
            predictions.append(predicted_label)
            actual.append(query[-1])
        return {
            'predictions': predictions,
            'actual': actual
        }


