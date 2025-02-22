import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import re

# Ensure the logs directory exists
LOG_DIR = r"C:\Users\naik62\PycharmProjects\TrialByFire\LogAnalysis\Logs"
os.makedirs(LOG_DIR, exist_ok=True)

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from datetime import datetime
import re

# Ensure the logs directory exists
LOG_DIR = r"C:\Users\naik62\PycharmProjects\TrialByFire\LogAnalysis\Logs"
os.makedirs(LOG_DIR, exist_ok=True)

def parse_log_line(line, filename):
    try:
        match = re.search(r'<\d+>([A-Za-z]{3}) (\d+) (\d+:\d+:\d+) .*? \|  :(.*)', line)
        if not match:
            return None  # Skip invalid lines

        month, day, time, message = match.groups()

        # Convert month name to number
        month_map = {
            "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
            "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"
        }
        month_num = month_map.get(month, "01")

        timestamp_str = f"2025-{month_num}-{day} {time}"
        timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")

        # Extract log level from the message
        log_level_match = re.search(r'(INFO|WARNING|ERROR|FINE|CRITICAL)', message)
        log_level = log_level_match.group(1) if log_level_match else "INFO"

        return [timestamp, log_level, message.strip(), filename]

    except Exception as e:
        print(f"Parsing error: {e} for line: {line.strip()}")
        return None


def suggest_corrective_action(message, score):
    if "database connection failed" in message.lower():
        return "Check database server status and credentials."
    elif "timeout" in message.lower():
        return "Increase timeout settings or optimize query performance."
    elif "disk full" in message.lower():
        return "Free up disk space or increase storage allocation."
    elif score < -0.5:
        return "Immediate investigation needed. Check system logs for failures."
    else:
        return "Monitor logs for recurrence. No immediate action required."

def read_logs():
    log_entries = []
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".log") or f.endswith(".txt")]

    if not log_files:
        print("No log files found in", LOG_DIR)
        return pd.DataFrame(), []

    for file in log_files:
        file_path = os.path.join(LOG_DIR, file)
        print(f"Reading log file: {file_path}")
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                parsed_line = parse_log_line(line, file)
                if parsed_line:
                    log_entries.append(parsed_line)

    df = pd.DataFrame(log_entries, columns=["timestamp", "log_level", "message", "filename"])
    if df.empty:
        print("No valid log entries found in", LOG_DIR)
    return df, log_files

def preprocess_logs(df):
    df.dropna(inplace=True)
    df = df[df["timestamp"].notnull()]
    if df.empty:
        print("No valid log data after preprocessing.")
        return None

    df["timestamp"] = df["timestamp"].apply(lambda x: x.timestamp())
    df["log_length"] = df["message"].apply(len)
    df["error_flag"] = df["log_level"].apply(lambda x: 1 if x in ["ERROR", "CRITICAL", "WARNING"] else 0)
    return df[["timestamp", "log_length", "error_flag"]]


def train_anomaly_model(features):
    if features is None or features.empty:
        print("Skipping model training due to insufficient data.")
        return None
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(features)
    return model


def detect_anomalies(model, features, df):
    if model is None:
        print("Skipping anomaly detection due to no trained model.")
        return
    df["anomaly_score"] = model.decision_function(features)
    df["anomaly"] = model.predict(features)
    df["corrective_action"] = df.apply(lambda row: suggest_corrective_action(row["message"], row["anomaly_score"]), axis=1)
    anomalies = df[df["anomaly"] == -1]
    if anomalies.empty:
        print("No anomalies detected.")
    else:
        anomalies.to_csv("anomalies.csv", index=False)
        print(f"Anomalies saved to anomalies.csv ({len(anomalies)} detected)")


def plot_anomaly_trend(anomalies):
    if "timestamp" not in anomalies.columns:
        print("Skipping anomaly trend plot: 'timestamp' column is missing.")
        return

    plt.figure(figsize=(10, 5))
    anomalies = anomalies.copy()
    anomalies.loc[:, "timestamp"] = pd.to_datetime(anomalies["timestamp"])
    anomalies.set_index("timestamp", inplace=True)
    anomalies.resample('h').count()["message"].plot(kind='line', color='red', marker='o', label='Anomalies Per Hour')
    plt.xlabel("Timestamp")
    plt.ylabel("Anomaly Count")
    plt.title("Anomaly Trend Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("anomaly_trend.png")
    plt.close()


def generate_html_report(df, anomalies, log_files):
    report_file = "anomaly_report.html"
    top_5_errors = df[df["log_level"] == "ERROR"][["message", "filename"]].value_counts().reset_index()
    top_5_errors.columns = ["Error Message", "File Name", "Count"]
    top_5_errors = top_5_errors.head(5)

    if not anomalies.empty and "timestamp" in anomalies.columns:
        plot_anomaly_trend(anomalies)

    with open(report_file, "w", encoding="utf-8") as f:
        f.write("<html><head><title>Anomaly Report</title>")
        f.write("<style>")
        f.write("body { font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }")
        f.write("h1, h2 { color: #333; }")
        f.write("table { border-collapse: collapse; width: 100%; background: white; }")
        f.write("th, td { border: 1px solid #ddd; padding: 8px; text-align: center; }")
        f.write("th { background-color: #4CAF50; color: white; }")
        f.write("tr:nth-child(even) { background-color: #f2f2f2; }")
        f.write("tr:hover { background-color: #ddd; }")
        f.write("</style>")
        f.write("</head><body>")
        f.write("<h1>Anomaly Detection Report</h1>")
        f.write(f"<p>Total Log Lines Processed: {len(df)}</p>")
        f.write(f"<p>Total Anomalies Detected: {len(anomalies)}</p>")

        if log_files:
            f.write("<h2>Log Files Analyzed</h2>")
            f.write("<ul>")
            for log_file in log_files:
                f.write(f"<li>{log_file}</li>")
            f.write("</ul>")

        if not top_5_errors.empty:
            f.write("<h2>Top 5 Errors and Files</h2>")
            f.write(top_5_errors.to_html(index=False, escape=False))

        if not anomalies.empty and "timestamp" in anomalies.columns:
            f.write("<h2>Anomaly Trend Over Time</h2>")
            f.write("<img src='anomaly_trend.png' alt='Anomaly Trend' width='100%'>")
            f.write("<h2>Sample Anomalies</h2>")
            f.write(
                anomalies[["timestamp", "message", "filename", "anomaly_score", "corrective_action"]].head().to_html(
                    index=False, escape=False))

        f.write("</body></html>")
    print(f"HTML report saved to {report_file}")


if __name__ == "__main__":
    log_data, log_files = read_logs()
    if not log_data.empty:
        processed_data = preprocess_logs(log_data)
        if processed_data is not None and not processed_data.empty:
            model = train_anomaly_model(processed_data)
            detect_anomalies(model, processed_data, log_data)
            generate_html_report(log_data, log_data[log_data["anomaly"] == -1], log_files)
