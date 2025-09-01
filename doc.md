# TradingLinesMonitor

Monitors a chart region, detects line colors, and sends alerts (including Telegram). Core logic lives in `mult-ui-screen.py`.

## Setup

1. Python 3.12+
2. Install deps:
   ```
   pip install -r requirements.txt
   ```
3. Configure Telegram (if used) as expected by your `TelegramNotifier` in `mult-ui-screen.py`.

## Run

```
python mult-ui-screen.py
```

See `send_alert()` in `mult-ui-screen.py` for alert behavior.
