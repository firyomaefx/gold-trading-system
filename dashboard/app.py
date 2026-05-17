import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from dashboard.layout import app, build_layout
from dashboard.callbacks import register_callbacks
from dashboard.data_provider import DashboardDataProvider
from config.settings import GOLD_CONFIG


def create_dashboard(provider: DashboardDataProvider = None, refresh_interval_ms: int = 5000):
    if provider is None:
        provider = DashboardDataProvider(GOLD_CONFIG)

    if not provider.connected:
        if not provider.connect():
            print("WARNING: Could not connect to MT5. Dashboard will start in offline mode.")
            print("  Click START after MT5 is running and chart is open.")

    app.layout = build_layout(refresh_interval_ms)
    register_callbacks(app, provider)

    return app


def main():
    provider = DashboardDataProvider(GOLD_CONFIG)

    print("Connecting to MT5...")
    connected = provider.connect()

    if connected:
        provider.refresh()
        acc = provider.mt5.get_account_info()
        print(f"\n  Account: #{acc.get('login', '?')}")
        print(f"  Balance: ${acc.get('balance', 0):.2f}")
        print(f"  Equity:  ${acc.get('equity', 0):.2f}")
        print(f"  Leverage: 1:{acc.get('leverage', 0)}")
    else:
        print("WARNING: Cannot connect to MT5.")
        print("  Dashboard will start anyway. Open MT5 and refresh browser.")
    print()

    dash_app = create_dashboard(provider)

    print("=" * 55)
    print("  DASHBOARD STARTING")
    print("  Open http://127.0.0.1:8050 in your browser")
    print("=" * 55)

    try:
        dash_app.run(debug=False, host="127.0.0.1", port=8050)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        provider.disconnect()


if __name__ == "__main__":
    main()