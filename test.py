def read_logs():
    log_entries = []
    log_files = [f for f in os.listdir(LOG_DIR) if f.endswith(".log") or f.endswith(".txt")]

    if not log_files:
        print("No log files found in", LOG_DIR)
        return pd.DataFrame()

    for file in log_files:
        file_path = os.path.join(LOG_DIR, file)
        print(f"Reading log file: {file_path}")  # Debugging output
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                print(f"Raw log line: {line.strip()}")  # 👈 Print raw logs
                parsed_line = parse_log_line(line)
                if parsed_line:
                    log_entries.append(parsed_line)

    df = pd.DataFrame(log_entries, columns=["timestamp", "log_level", "message"])
    if df.empty:
        print("No valid log entries found in", LOG_DIR)
    return df
