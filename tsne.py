import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def read_data(file_path):
    data_list = [] 
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(val) for val in line.strip().split('\t')]
            data_list.append(values)
    return np.array(data_list)

def visualize_tsne(data, labels, n_components=2, perplexity=4):
    tsne = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    data_2d = tsne.fit_transform(data)

    plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='plasma')
    plt.title('SGD')
    plt.colorbar(label='Initial W Values')
    plt.show()

if __name__ == "__main__":
    # Read data for class 1
    data_class1 = read_data('w1.txt')
    labels_class1 = np.ones(data_class1.shape[0])

    # Read data for class 2
    data_class2 = read_data('w2.txt')
    labels_class2 = 2 * np.ones(data_class2.shape[0])

    # Read data for class 3
    data_class3 = read_data('w3.txt')
    labels_class3 = 3 * np.ones(data_class3.shape[0])

    # Read data for class 4
    data_class4 = read_data('w4.txt')
    labels_class4 = 4 * np.ones(data_class4.shape[0])

    # Read data for class 5
    data_class5 = read_data('w5.txt')
    labels_class5 = 5 * np.ones(data_class5.shape[0])

    # Combine data and labels for all classes
    all_data = np.vstack([data_class1, data_class2, data_class3, data_class4, data_class5])
    all_labels = np.concatenate([labels_class1, labels_class2, labels_class3, labels_class4, labels_class5])

    # Visualize T-SNE
    visualize_tsne(all_data, all_labels)
