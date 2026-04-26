import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dashboard.layout_v2 import app, build_layout
from dashboard.callbacks_v2 import register_callbacks
from dashboard.data_provider_v2 import DashboardDataProviderV2
from config.settings_v2 import V2Config


def create_dashboard(provider=None, refresh_interval_ms=5000):
    if provider is None:
        provider = DashboardDataProviderV2(V2Config())

    if not provider.connected:
        provider.connect_all()

    app.layout = build_layout(refresh_interval_ms)
    register_callbacks(app, provider)
    return app


def main():
    provider = DashboardDataProviderV2(V2Config())

    print("Connecting to MT5 + Rithmic...")
    if not provider.connect_all():
        print("WARNING: One or both data sources unavailable.")
        print("MT5: " + ("OK" if provider._connected_mt5 else "FAIL"))
        print("Rithmic: " + ("OK" if provider._connected_rithmic else "SYNTHETIC (simulated DOM)"))

    provider.refresh()
    if provider._connected_mt5:
        acc = provider.mt5.get_account_info()
        print(f"\n  Account: #{acc.get('login', '?')}")
        print(f"  Balance: ${acc.get('balance', 0):.2f}")

    app = create_dashboard(provider)

    print(f"\n{'=' * 55}")
    print(f"  V2 DOM-VALIDATED DASHBOARD")
    print(f"  Open http://127.0.0.1:8050")
    print(f"  DOM: {'RITHMIC LIVE' if provider._connected_rithmic else 'SIMULATED (synthetic)'}")
    print(f"{'=' * 55}")

    app.run(debug=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
