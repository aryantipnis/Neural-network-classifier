import sys
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwork: 
    def __init__(self, neuronsPerLayer, weights=None):
        self.inputs = neuronsPerLayer[0]
        self.outputs = neuronsPerLayer[-1]
        self.hidden_layers = len(neuronsPerLayer)-2
        self.neurons = neuronsPerLayer
        self.weights = self.initialize_weights() if weights is None else weights

    def initialize_weights(self):
        weights = []
        weights.append(np.random.uniform(-1, 1, size=(self.neurons[1], self.inputs + 1)))
        for i in range(1, self.hidden_layers):
            weights.append(np.random.uniform(-1, 1, size=(self.neurons[i + 1], self.neurons[i] + 1)))
        weights.append(np.random.uniform(-1, 1, size=(self.outputs, self.neurons[-2] + 1)))

        return weights

    def g(self, x):
        return 1 / (1 + np.exp(-x))

    def forward_prop(self, x, weights):
        a = [np.vstack([np.array([[1]]), np.array([x]).reshape(-1, 1)])]
        for i in range(len(weights) - 1):
            z = np.dot(weights[i], a[-1])
            a.append(np.vstack([np.array([[1]]), self.g(z)]))
        z = np.dot(weights[-1], a[-1]) 
        a.append(self.g(z))
        return a 

    def compute_cost(self, x, y, weights, lam, p = False): #variable p determines if the values are to be printed or not
        n = len(x)
        J = 0
        for i in range(n):
            a = self.forward_prop(x[i], weights)
            cost = -np.sum(np.array([y[i]]).reshape(-1, 1) * np.log(a[-1]) + (1 - np.array([y[i]]).reshape(-1, 1)) * np.log(1 - a[-1]))
            if p: 
                print(f'\tCost, J, associated with instance {i+1}: {round(cost, 3)}')
            J += cost
        J /= n
        S = (lam / (2 * n)) * sum(np.sum(np.square(theta[:, 1:])) for theta in weights)
        return J + S

    def back_prop(self, x, y, weights, lam, alpha, epsilon, p = False): #variable p determines if the values are to be printed or not
        n = len(x)
        num_layers = len(weights)
        D = [np.zeros_like(theta) for theta in weights]

        while self.compute_cost(x, y, weights, lam) > epsilon: 
            # print(self.compute_cost(x, y, weights, lam))
            for i in range(n):
                deltas = []

                a = self.forward_prop(x[i], weights)
                delta = a[-1] - np.array([y[i]]).reshape(-1, 1)
                deltas.append(delta)

                for k in range(num_layers - 1, 0, -1):
                    delta = np.dot(weights[k].T, delta) * a[k] * (1 - a[k])
                    delta = delta[1:]
                    deltas.append(delta)
                
                if p : 
                    print(f'\n\tComputing gradients based on training instance {i+1}:')
                    for j in range(len(deltas)): 
                        print(f'\t\tdelta{len(deltas)-j+1}:\t{" ".join(map(lambda x: str(round(x, 5)), deltas[j].flatten()))}')

                deltas.reverse()

                for k in range(num_layers - 1 , -1, -1):
                    grad = np.dot(deltas[k], a[k].T)
                    D[k] += grad

                    if p:
                        print(f'\n\t\tGradients of Theta{k+1} based on training instance {i+1}:')
                        for row in grad:
                            print('\t\t', end='')
                            for element in row:
                                print(f'{element:.5f}', end='   ')
                            print() 
                    
            for k in range(num_layers):
                D[k] = (D[k] + lam *  np.hstack((np.zeros((weights[k].shape[0], 1)), weights[k][:, 1:]))) / n

            if p:
                print('\nThe entire training set has been processed. Computing the average (regularized) gradients:')
                for i, arr in enumerate(D, start=1):
                    print(f'\n\tFinal regularized gradients of Theta{i}:')
                    for row in arr:
                        print('\t\t', end='')
                        for element in row:
                            print(f'{element:.5f}', end='   ')
                        print()
                return
            
            for k in range(num_layers):
                weights[k] = weights[k] - alpha * D[k]

        return weights

    def train(self, trainingData, actualLabels, lam, alpha, epsilon, classes = None):
        unique_classes = np.unique(actualLabels) if classes is None else classes
        y = np.zeros((len(actualLabels), len(unique_classes)))
        
        for i, label in enumerate(actualLabels):
            y[i, np.where(unique_classes == label)] = 1

        self.weights = self.back_prop(trainingData, y, self.weights, lam, alpha, epsilon)

    def predict(self, data, actualLabels, classes = None):
        unique_classes = np.unique(actualLabels) if classes is None else classes
        predictedLabels = []
        for row in data: 
            probabilites = self.forward_prop(row, self.weights)[-1]
            predicted_class = unique_classes[np.argmax(probabilites)]
            predictedLabels.append(predicted_class)
        return predictedLabels
    
def evaluate_results(actualLabels, predictedLabels): 
    classes = np.unique(actualLabels)
    n = len(classes)
    confusion_matrix = np.zeros((n, n), dtype=int)
    class_map = {cls: i for i, cls in enumerate(classes)}

    for actual, predicted in zip(actualLabels, predictedLabels):
        confusion_matrix[class_map[actual], class_map[predicted]] += 1

    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    precision = np.zeros(n)
    recall = np.zeros(n)
    F1_score = np.zeros(n)
    for i in range(n):
        TP = confusion_matrix[i, i]
        FP = np.sum(confusion_matrix[:, i]) - TP
        FN = np.sum(confusion_matrix[i, :]) - TP
        
        precision[i] = TP / (TP + FP) if TP + FP > 0 else 0
        recall[i] = TP / (TP + FN) if TP + FN > 0 else 0
        F1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if precision[i] + recall[i] > 0 else 0

    return accuracy, np.mean(F1_score)

def normalize(data, targetName): 
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    maxVal = np.max(features, axis=0)
    minVal = np.min(features, axis=0)
    
    features = (features - minVal) / (maxVal - minVal)

    normalizedData = pd.DataFrame(features, columns=data.columns[:-1])
    normalizedData[targetName] = labels

    return normalizedData

def process_categorical(data, targetName): 
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values

    ohe = OneHotEncoder()
    transformed = ohe.fit_transform(features)
    
    transformedArray = transformed.toarray()
    transformedData = pd.DataFrame(transformedArray)
    transformedData[targetName] = labels

    return transformedData

def makeFolds(dataset, k, targetName):
    classes = np.unique(dataset[targetName])
    class_subsets = [dataset[dataset[targetName] == cls] for cls in classes]
    subset_len = [len(subset) for subset in class_subsets]
    fold_size = len(dataset)//k

    folds = []
    for _ in range(k):
        fold = pd.DataFrame(columns=dataset.columns)
        for i, subset in enumerate(class_subsets):
            num_instances = fold_size * subset_len[i] // len(dataset)
            chosen_instances = subset.sample(n=num_instances)
            fold = pd.concat([fold, chosen_instances])
            class_subsets[i] = subset.drop(chosen_instances.index)
        folds.append(fold)

    rem_instances = pd.concat(class_subsets)
    
    if len(rem_instances) != 0:
        fold_index = 0
        for instance in rem_instances.itertuples(index=False):
            folds[fold_index] = pd.concat([folds[fold_index], pd.DataFrame([instance], columns=dataset.columns)])
            fold_index = (fold_index + 1) % k 
    
    return folds

def cross_validation(folds, neuronsPerLayer, lam, alpha, epsilon):
    accuracies = []
    f1_scores = []
    for i, fold in enumerate(folds):
        print(i)
        testingSet = shuffle(fold)
        trainingSet = shuffle(pd.concat(folds[:i] + folds[i+1:]))

        trainingData = trainingSet.iloc[:, :-1].values
        trainingLabels = trainingSet.iloc[:, -1].values

        testingData = testingSet.iloc[:, :-1].values
        testingLabels = testingSet.iloc[:, -1].values

        a = NeuralNetwork(neuronsPerLayer)
        a.train(trainingData, trainingLabels, lam, alpha, epsilon)
        predictedLabels = a.predict(testingData, trainingLabels)
        accuracy, f1_score = evaluate_results(testingLabels, predictedLabels)
        accuracies.append(accuracy)
        f1_scores.append(f1_score)
    print(f'Accuracy: {np.mean(accuracies)}')
    print(f'F1-score: {np.mean(f1_scores)}')


def learningCurve(dataset, neuronsPerLayer, lam, alpha, epsilon, step_size=10):
    trainData, testData = train_test_split(dataset, train_size=0.7, test_size=0.3)
    n = NeuralNetwork(neuronsPerLayer)

    testingData = testData.iloc[:, :-1].values
    testingLabels = testData.iloc[:, -1].values
    classes = np.unique(dataset.iloc[:, -1].values)
    
    costs = []
    training_sizes = []

    for i in range(0, len(trainData), step_size):
        trainingData = trainData.iloc[i:i+10, :-1].values
        trainingLabels = trainData.iloc[i:i+10, -1].values

        y = np.zeros((len(testingLabels), len(classes)))
        
        for j, label in enumerate(testingLabels):
            y[j, np.where(classes == label)] = 1
        
        n.train(trainingData, trainingLabels, lam, alpha, epsilon, classes)
        cost = n.compute_cost(testingData, y, n.weights, lam)
        costs.append(cost)
        training_sizes.append(i+10)

    plt.plot(training_sizes, costs)
    plt.xlabel('Number of training instances')
    plt.ylabel('Cost')
    plt.title('Learning Curve')
    plt.show()


def examples(x, y, weights, lam, neuronsPerLayer):
    n = NeuralNetwork(neuronsPerLayer, weights)
    for i in range(len(x)):
        a = n.forward_prop(x[i], weights)
        print(f"Training instance {i + 1}:")
        for layer_idx, layer_output in enumerate(a):
            print(f"a{layer_idx + 1}:", end="")
            for neuron_output in layer_output:
                print(f"\t\t{round(neuron_output[0], 5)}", end="")
            print('\n')

        print(f'Predicted Output for instance {i+1}: {" ".join(map(lambda x: str(round(x, 5)), a[-1].flatten()))}')
        print(f'Expected Output for instance {i+1}: {y[i]}\n')
    
    print('Cost Function: ')
    totalCost = n.compute_cost(x, y, weights, lam, True)
    print(f'\tFinal (regularized) cost, J, based on the complete training set: {round(totalCost, 5)}')

    print('\nRunning Backpropogation: ')
    n.back_prop(x, y, weights, lam, 1, 0.1, True)

if __name__ == '__main__':
    argv_len = len(sys.argv)

    datasetName = sys.argv[1] if argv_len == 2 else 'backprop_example1.txt'

    if datasetName == 'backprop_example1.txt': 
        theta1 = np.array([[0.4 , 0.1], [0.3 , 0.2]])
        theta2 =  np.array([[0.7, 0.5, 0.6]])
        weights = [theta1 , theta2]
        neuronsPerLayer = [1, 2, 1]
        examples(np.array([0.13, 0.42]), np.array([0.9, 0.23]), weights, 0, neuronsPerLayer)
        sys.exit()
    elif datasetName == 'backprop_example2.txt':
        theta1 = np.array([[0.42 , 0.15, 0.4], [0.72 , 0.10, 0.54], [0.01, 0.19, 0.42], [0.3, 0.35, 0.68]])
        theta2 = np.array([[0.21, 0.67, 0.14, 0.96, 0.87 ], [0.87, 0.42, 0.2, 0.32, 0.89 ], [0.03, 0.56, 0.8, 0.69, 0.09]])
        theta3 = np.array([[0.04, 0.87, 0.42, 0.53 ] , [0.17, 0.1, 0.95, 0.69 ]])
        weights = [theta1, theta2, theta3] 
        neuronsPerLayer = [2, 4, 3, 2]
        examples(np.array([[0.32 , 0.68], [0.83, 0.02]]), np.array([[0.75, 0.98], [0.75, 0.28]]), weights, 0.250, neuronsPerLayer)
        sys.exit()
    elif datasetName == 'hw3_house_votes_84.csv': 
        dataset = pd.read_csv(f'datasets/{datasetName}')
        targetName = 'class'
        dataset = process_categorical(dataset, targetName)
        neuronsPerLayer = [48, 8, 2]
    elif datasetName == 'hw3_wine.csv': 
        dataset = pd.read_csv(f'datasets/{datasetName}', sep='\t')
        targetName = '# class'
        dataset = dataset[[col for col in dataset.columns if col != targetName] + [targetName]]
        dataset = normalize(dataset, targetName)
        neuronsPerLayer = [13, 8, 3]
    elif datasetName == 'hw3_cancer.csv':
        dataset = pd.read_csv(f'datasets/{datasetName}', sep='\t')
        targetName = 'Class'
        dataset = normalize(dataset, targetName)
        neuronsPerLayer = [9, 8, 2]
    
    folds = makeFolds(dataset, 10, targetName)
    cross_validation(folds, neuronsPerLayer, 0.3, 0.9, 0.5) #lambda, alpha, epsilon
    # learningCurve(dataset, neuronsPerLayer, 0.001, 0.9, 0.2)

    
