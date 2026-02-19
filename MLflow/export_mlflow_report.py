from pathlib import Path
import traceback

import mlflow
import pandas as pd

EXPERIMENT_NAME = "titanic-experiments"


def main():
    # report.html — рядом со скриптом
    script_dir = Path(__file__).resolve().parent
    out_path = script_dir / "report.html"

    print(f"[INFO] Script dir: {script_dir}")
    print(f"[INFO] Report path: {out_path}")

    # Tracking туда же, где у меня mlruns
    mlruns_dir = script_dir / "mlruns"
    mlflow.set_tracking_uri(f"file:{mlruns_dir.as_posix()}")
    print(f"[INFO] MLFLOW_TRACKING_URI: {mlflow.get_tracking_uri()}")

    mlflow.set_experiment(EXPERIMENT_NAME)

    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        print(f"[ERROR] Experiment '{EXPERIMENT_NAME}' не найден.")
        print("[INFO] Available experiments:")
        for e in mlflow.search_experiments():
            print(f"  - {e.name} (id={e.experiment_id})")
        raise SystemExit(1)

    df = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=5000
    )

    print(f"[INFO] Found runs: {len(df)}")

    # Даже если runs = 0, всё равно создаю HTML с сообщением
    keep_cols = [c for c in df.columns if c in ("run_name", "status") or c.startswith(("metrics.", "params.", "tags.mlflow.runName"))]
    df_small = df[keep_cols].copy() if len(df) else df

    html = list()
    html.append(f"<h1>MLflow report: {EXPERIMENT_NAME}</h1>")
    html.append(f"<p><b>Tracking URI</b>: {mlflow.get_tracking_uri()}</p>")
    html.append(f"<p><b>Всего runs</b>: {len(df)}</p>")

    if len(df) > 0:
        best = df_small.iloc[0]
        html.append("<h2>Лучший run</h2><ul>")
        html.append(f"<li><b>run_name</b>: {best.get('tags.mlflow.runName','')}</li>")
        for m in ["metrics.accuracy", "metrics.f1", "metrics.precision", "metrics.recall"]:
            if m in df_small.columns and pd.notna(best.get(m)):
                html.append(f"<li><b>{m}</b>: {float(best.get(m)):.6f}</li>")
        html.append("</ul>")

        html.append("<h2>Все runs</h2>")
        html.append(df_small.to_html(index=False, escape=True))
    else:
        html.append("<p><b>No runs found.</b></p>")

    out_path.write_text("\n".join(html), encoding="utf-8")
    print(f"[OK] Saved report: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        print("[FATAL] Exception occurred:")
        traceback.print_exc()
        raise
