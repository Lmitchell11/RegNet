import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# Specify the directory containing the CSV files
directory = '/home/liam/Documents/regnet/results/csvFolder'

# List of CSV files to plot with their descriptive names for the legend
csv_files = [
    ('Losstrain-.-run_00005.csv', "Authors original attempt"),
    ('Losstrain-.-run_00007.csv', "Ours - B_size=8, lr=3e-4"),
    ('Losstrain-.-run_00008.csv', "Ours - B_size=16, lr=1e-3"),
]



# Create a new figure with high resolution
plt.figure(figsize=(8, 5), dpi=300)

# Styling
plt.style.use('ggplot')
font = {'family': 'serif',
        'weight': 'normal',
        'size': 12}
plt.rc('font', **font)
plt.rc('lines', linewidth=2)
colors = ['blue', 'green', 'red', 'purple']  # Add more colors if more lines are present

# Loop over the CSV files and their corresponding legend labels
for (filename, label), color in zip(csv_files, colors):
    file_path = os.path.join(directory, filename)
    df = pd.read_csv(file_path, delimiter=';')
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], label=label, color=color)

# Enhance the plot
plt.title('Comparison of Training Loss Over Epochs', fontsize=14)
plt.xlabel('Batch #', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(title='Configuration', fontsize=10, title_fontsize='13')
plt.grid(True, linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('/home/liam/Documents/regnet/results/TrainingLossComparison.png', format='png')
plt.show()
