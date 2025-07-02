import matplotlib.pyplot as plt

x = [1, 1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 5]

plt.hist(x, bins=30, edgecolor='black',align='mid')

plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Values')

plt.show()