import csv
import numpy as np
import matplotlib.pyplot as plt


INGREDIENT = "chickpea"
FILE_DIR = f"logs/{INGREDIENT}.csv"
TOLERANCE = 20


def generate_plot():
    """
    Generate performance evaluation plots from log data
    """
    data = []
    with open(FILE_DIR) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count != 0:
                data.append([float(row[1]), float(row[2])])
            line_count += 1
    data = np.array(data)

    colors = ['r' if (np.abs(item[0] - item[1]) > TOLERANCE) else 'g' for item in data]
    min_pt = np.min(data[:, 0])
    max_pt = np.max(data[:, 0])
    start_pt = min_pt - 0.05 * (max_pt - min_pt)
    end_pt = max_pt + 0.05 * (max_pt - min_pt)
    plt.scatter(data[:, 0], data[:, 1], c=colors, s=200, alpha=0.5)
    plt.plot([start_pt, end_pt], [start_pt, end_pt], c='b', label='Ideal dispensing line')
    plt.xlabel("Requested Weight (g)")
    plt.ylabel("Dispensed Weight (g)")
    plt.grid()
    plt.title(INGREDIENT)
    plt.legend()
    plt.show()
    

if __name__ == "__main__":
    generate_plot()
