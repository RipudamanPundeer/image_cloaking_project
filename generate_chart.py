import csv
import numpy as np
import matplotlib.pyplot as plt

RESULTS_FILE = "master_transferability_results.csv"


def generate_chart():
    print(f"Reading data from {RESULTS_FILE}...")

    metrics_data = {}

    try:
        # 1. Read the CSV file directly
        with open(RESULTS_FILE, mode='r') as file:
            reader = csv.DictReader(file)

            # Get the metric names (skipping the 'Image' column)
            metric_names = [field for field in reader.fieldnames if field != "Image"]

            for metric in metric_names:
                metrics_data[metric] = []

            # Extract the data
            for row in reader:
                for metric in metric_names:
                    val = row[metric]
                    if val != "ERROR":
                        metrics_data[metric].append(float(val))

    except FileNotFoundError:
        print(f"❌ Error: {RESULTS_FILE} not found. Please check the file name.")
        return

    # 2. Calculate Averages
    final_metrics = []
    final_scores = []

    print("\nCalculated Averages from CSV:")
    for metric in metric_names:
        if metrics_data[metric]:
            avg_score = np.mean(metrics_data[metric])
            print(f"{metric.ljust(35)} : {avg_score: .4f}")

            # Format names nicely for the chart (adds line breaks)
            chart_name = metric.replace(" (", "\n(")
            final_metrics.append(chart_name)
            final_scores.append(avg_score)

    # ==========================================
    # 3. GRAPH GENERATION (BAR CHART)
    # ==========================================
    print("\nGenerating Transferability Bar Chart...")
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))

    # Sleek color palette for academic reports
    colors = ['#E24A33', '#348ABD', '#988ED5', '#FBC15E', '#8EBA42']

    # Create the bars
    bars = plt.bar(final_metrics, final_scores, color=colors[:len(final_metrics)],
                   edgecolor='black', linewidth=1.2, alpha=0.9)

    # Add a baseline indicating the typical "Recognition Threshold"
    plt.axhline(y=0.40, color='red', linestyle='--', alpha=0.7, label='Estimated Recognition Threshold')

    # Add the exact numbers on top of each bar
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.015, f'{yval:.3f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Styling labels and title
    plt.title("Black-Box Transferability: Attack Success Across Architectures", fontsize=16, fontweight='bold', pad=15)
    plt.ylabel("Average Cosine Similarity (Lower is Better)", fontsize=13)

    # Scale Y-axis automatically to leave room for the text labels
    plt.ylim(0, max(final_scores) + 0.2)

    plt.legend(loc='upper right')
    plt.tight_layout()

    # Save a high-resolution copy for your thesis document
    plt.savefig("transferability_chart.png", dpi=300)
    print("✅ Chart saved successfully as 'transferability_chart.png'")
    plt.show()


if __name__ == "__main__":
    generate_chart()