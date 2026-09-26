import pandas as pd
import os

print("Generating small preview files for easy viewing...")
s1_path = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_source1.tsv"
gt_path = "6ab10eb3b23ba_student_resource/student_resource/dataset/train/train_ground_truth.tsv"

df_s1 = pd.read_csv(s1_path, sep="\t", nrows=100)
df_gt = pd.read_csv(gt_path, sep="\t", nrows=100)

# Save to CSV for easy opening in Excel
df_s1.to_csv("preview_source1_100rows.csv", index=False)
df_gt.to_csv("preview_ground_truth_100rows.csv", index=False)

html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Amazon ML Challenge 2026 - Data Preview</title>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        background-color: #0f172a;
        color: #f8fafc;
        padding: 24px;
        margin: 0;
    }}
    h1 {{ color: #38bdf8; margin-bottom: 8px; }}
    p {{ color: #94a3b8; font-size: 15px; margin-top: 0; }}
    .card {{
        background-color: #1e293b;
        border-radius: 8px;
        padding: 20px;
        margin-bottom: 28px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.4);
        overflow-x: auto;
    }}
    table {{
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
        margin-top: 12px;
    }}
    th, td {{
        border: 1px solid #334155;
        padding: 10px 14px;
        text-align: left;
    }}
    th {{
        background-color: #0284c7;
        color: #ffffff;
        font-weight: 600;
        position: sticky;
        top: 0;
    }}
    tr:nth-child(even) {{ background-color: #1e293b; }}
    tr:nth-child(odd) {{ background-color: #0f172a; }}
    tr:hover {{ background-color: #334155; }}
    code {{ background: #334155; padding: 2px 6px; border-radius: 4px; font-family: monospace; }}
</style>
</head>
<body>
    <h1>Amazon ML Challenge 2026: Dataset Explorer</h1>
    <p>The original TSV files are 200MB - 500MB and contain over 2,000,000 rows, which freezes Excel or Notepad. Here is an interactive preview of the first 25 records.</p>
    
    <div class="card">
        <h2>Source 1 Sample (Deduplicated Ground Reference)</h2>
        {df_s1.head(25).to_html(classes="preview-table", index=False)}
    </div>

    <div class="card">
        <h2>Ground Truth Matching Labels Sample</h2>
        {df_gt.head(25).to_html(classes="preview-table", index=False)}
    </div>
</body>
</html>
"""

with open("preview_dataset.html", "w", encoding="utf-8") as f:
    f.write(html_template)

print("Created preview_source1_100rows.csv, preview_ground_truth_100rows.csv, and preview_dataset.html!")
