"""Turn the completed interval sweep CSV into a readable comparison report.

Run from SAC/ after script/run_interval_sweep.py finishes. ``--wait`` keeps a small
background process alive until all five test results have been written.
"""

import argparse
import csv
import os
import time


def load_rows(csv_path):
    if not os.path.isfile(csv_path):
        return []
    with open(csv_path, newline="", encoding="utf-8") as summary_file:
        rows = list(csv.DictReader(summary_file))
    return rows if len(rows) == 5 else []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir",
                        default="./results/interval_sweep_2m")
    parser.add_argument("--wait", action="store_true")
    args = parser.parse_args()

    csv_path = os.path.join(args.results_dir, "summary.csv")
    rows = load_rows(csv_path)
    deadline = time.monotonic() + 36 * 3600
    while args.wait and not rows:
        if time.monotonic() >= deadline:
            raise TimeoutError("Sweep did not finish within 36 hours")
        time.sleep(60)
        rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("Expected five completed conditions in %s" % csv_path)

    rows.sort(key=lambda row: int(row["intervals"]))
    best = max(rows, key=lambda row: (
        float(row["success_rate"]),
        -float(row["collision_rate"]),
        float(row["average_speed_m_s"]),
    ))
    episodes = int(rows[0]["episodes"])
    lines = [
        "# 区间射线数量对比",
        "",
        "每组从零训练 %d 个计入步，使用 %d 条回放缓存；训练种子 %s，"
        "测试种子 %s。各组测试 %d 个完成回合。" % (
            int(rows[0]["steps"]), int(rows[0]["buffer_size"]),
            rows[0]["train_seed"], rows[0]["test_seed"], episodes,
        ),
        "",
        "| 区间数 | 成功率 | 碰撞率 | 平均速度 (m/s) | 卡死重置率 | 超时率 | 训练时间 (h) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| %s | %.1f%% | %.1f%% | %.3f | %.1f%% | %.1f%% | %.2f |" % (
                row["intervals"],
                100 * float(row["success_rate"]),
                100 * float(row["collision_rate"]),
                float(row["average_speed_m_s"]),
                100 * float(row["stuck_reset_rate"]),
                100 * float(row["timeout_rate"]),
                float(row["training_hours"]),
            )
        )
    lines.extend([
        "",
        "按成功率优先、碰撞率次之排序，%s 区间在本次测试中表现最好。" % best["intervals"],
        "碰撞率是回合中至少发生一次激光检测接触的比例，与成功率可能重叠；"
        "卡死重置率对应 train.py 的碰撞指标。平均速度为完成回合的绝对线速度"
        "按仿真时间加权平均。",
        "这是一组训练种子和一组测试种子的比较；较小差异需要多种子复验。",
        "",
    ])
    report_path = os.path.join(args.results_dir, "comparison.md")
    with open(report_path, "w", encoding="utf-8") as report_file:
        report_file.write("\n".join(lines))
    print("Comparison saved to %s" % report_path, flush=True)


if __name__ == "__main__":
    main()
