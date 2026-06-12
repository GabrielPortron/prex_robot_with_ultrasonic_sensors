import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

# Read CSV file into a DataFrame

# Read the filename directly form the command line
if len(sys.argv) == 2:
    file_name = sys.argv[1]
elif len(sys.argv) == 3:
    file_name = sys.argv[1]
    folder_name = sys.argv[2]
else:
    file_name = input("insert the name of the csv file without extension") + ".csv"
    folder_name = "images"

print(f"Reading file: {file_name}")
df = pd.read_csv(file_name)

# Display basic statistics for the data
print("Basic statistics:")
print(df.describe())

test_cases = df["angle_distance"].unique()

# Create folder to store images
if not os.path.exists(folder_name):
    os.makedirs(folder_name)

for test_case in test_cases:
    # Create a figure with GridSpec: 2 rows, 2 columns
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2)  # 2 rows, 2 columns
    selection = df[df["angle_distance"] == test_case]
    values = selection["measure"]

    # First subplot: large one in the first row (line plot)
    ax1 = fig.add_subplot(gs[0, :])  # Span across the whole first row
    ax1.plot(values, label="Measure", marker="o")
    ax1.set_title("Plot of Measures for: " + test_case)
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Measure (cm)")
    ax1.legend()
    ax1.grid(True)

    # Second subplot: smaller one in the second row, first column (Histogram)
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax2.hist(values, bins=10, edgecolor="black")
    ax2.set_title("Histogram of Measures for " + test_case)
    ax2.set_xlabel("Measure (cm)")
    ax2.set_ylabel("Frequency")
    ax2.grid(True)

    # Third subplot: smaller one in the second row, second column (Boxplot)
    ax3 = fig.add_subplot(gs[1, 1])  # Bottom right
    ax3.boxplot(
        values, vert=True, patch_artist=True, boxprops=dict(facecolor="lightblue")
    )
    ax3.set_title("Boxplot of Measures for " + test_case)
    ax3.set_xlabel("Value")

    # Set title for the entire figure
    fig.suptitle("Measures for test case: " + test_case, fontsize=16)

    # Adjust layout to prevent overlap of titles and labels
    plt.tight_layout()

    # Show the plots
    plt.show()

    # Save picture
    fig.savefig(os.path.join(folder_name, test_case))
