# Customer Segmentation Project

This project demonstrates a practical customer segmentation workflow using Python and scikit-learn. It loads synthetic customer data, performs K-means clustering based on demographics and purchase behavior, analyzes segment characteristics, and visualizes the outcomes.

## Objectives
- Segment customers into meaningful groups based on behavior and demographics
- Identify purchase patterns and customer preferences
- Generate targeted insights for marketing and engagement strategies

## Project Structure
- `data/customers.csv` – synthetic customer dataset
- `customer_segmentation.py` – Python script for clustering and visualization
- `requirements.txt` – dependencies required to run the project
- `output/` – generated charts and summaries

## Setup
1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the analysis:
   ```bash
   python customer_segmentation.py
   ```

## Expected Output
- Cluster labels added to the dataset
- Summary of segment profiles saved to `output/segment_summary.csv`
- Visualizations saved to `output/cluster_plot.png` and `output/segment_profile.png`
