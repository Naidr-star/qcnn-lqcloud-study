"""Regenerate study statistics from released outcomes, with strict input checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

from .statistics import accuracy_report, paired_exact, wilson_interval


INPUTS = ("benchmark_reference.json", "holdout_predictions.json", "study.json", "temporal_predictions.json")
HOLDOUT_COLUMNS = ("qpu_original_head", "qpu_calibrated", "ideal_calibrated")
TEMPORAL_COLUMNS = ("initial", "session1_frozen", "session1_refitted", "session2_frozen", "session2_refitted")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_inputs(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    require(manifest.get("schema_version") == 1, "Unsupported manifest version")
    require(set(manifest["sha256"]) == set(INPUTS), "Manifest must cover exactly the four study inputs")
    result = {}
    for name in INPUTS:
        content = (directory / name).read_bytes()
        require(hashlib.sha256(content).hexdigest() == manifest["sha256"][name], f"Input integrity mismatch: {name}")
        result[name.removesuffix(".json")] = json.loads(content)
    return result


def validate_records(block, columns, expected_size):
    require(set(block) == {"schema_version", "records"} and block["schema_version"] == 1, "Unexpected outcome schema")
    rows = block["records"]
    require(isinstance(rows, list) and len(rows) == expected_size, "Incomplete or extra outcome records")
    fields = {"sample_id", "label", *columns}
    seen = set()
    for row in rows:
        require(isinstance(row, dict) and set(row) == fields, "Unexpected outcome fields")
        sample_id = row["sample_id"]
        require(isinstance(sample_id, str) and re.fullmatch(r"H\d{3}", sample_id) is not None, "Invalid public sample identifier")
        require(sample_id not in seen, "Duplicate sample identifier")
        seen.add(sample_id)
        for key in ("label", *columns):
            require(type(row[key]) is int and row[key] in (0, 1), "Outcomes must be integer binary classes")
    return rows


def evaluate_rows(rows, columns):
    labels = [row["label"] for row in rows]
    result = {}
    for column in columns:
        predictions = [row[column] for row in rows]
        result[column] = accuracy_report(predictions, labels)
        result[column]["prediction_class_counts"] = [predictions.count(0), predictions.count(1)]
    return result


def analyze(inputs):
    study = inputs["study"]
    require(study["schema_version"] == 1 and study["confidence"] == 0.95, "Unsupported study specification")
    require(study["calibration_images"] == sum(study["calibration_class_counts"]), "Calibration count mismatch")
    h = validate_records(inputs["holdout_predictions"], HOLDOUT_COLUMNS, study["holdout_images"])
    t = validate_records(inputs["temporal_predictions"], TEMPORAL_COLUMNS, study["temporal_panel_images"])
    require([sum(r["label"] == c for r in h) for c in (0, 1)] == study["holdout_class_counts"], "Holdout class composition mismatch")
    require([sum(r["label"] == c for r in t) for c in (0, 1)] == study["temporal_panel_class_counts"], "Temporal class composition mismatch")
    by_id = {row["sample_id"]: row for row in h}
    for row in t:
        require(row["sample_id"] in by_id, "Temporal sample absent from original holdout")
        original = by_id[row["sample_id"]]
        require(row["label"] == original["label"] and row["initial"] == original["qpu_calibrated"], "Temporal baseline/identity mismatch")
    holdout = evaluate_rows(h, HOLDOUT_COLUMNS)
    temporal = evaluate_rows(t, TEMPORAL_COLUMNS)
    holdout_pair = paired_exact([r["ideal_calibrated"] for r in h], [r["qpu_calibrated"] for r in h], [r["label"] for r in h])
    temporal_pairs = {
        f"session{i}": paired_exact([r[f"session{i}_frozen"] for r in t], [r[f"session{i}_refitted"] for r in t], [r["label"] for r in t])
        for i in (1, 2)
    }
    benchmark = inputs["benchmark_reference"]
    require(benchmark["schema_version"] == 1, "Unexpected benchmark schema")
    cases = benchmark["cases"]
    require(len(cases) == 4 and len({c["case"] for c in cases}) == 4, "Benchmark case coverage mismatch")
    for case in cases:
        for key in ("lqcloud_raw", "local_aer_finite_sample"):
            row = case[key]
            wilson_interval(row["correct"], row["total"])
            require(row["total"] == sum(case["local_asset_label_counts"]) == 50, "Benchmark denominator mismatch")
        for key in ("paper_ideal_accuracy_approx", "paper_aria_em_accuracy_approx"):
            require(type(case[key]) in (int, float) and 0 <= case[key] <= 1, "Invalid digitized literature accuracy")
    return {
        "schema_version": 1,
        "holdout": holdout,
        "qpu_minus_ideal_paired": holdout_pair,
        "temporal": temporal,
        "refitted_minus_frozen_paired": temporal_pairs,
        "benchmark_reference": cases,
        "reported_sensitivity_not_recomputed": study["reported_sensitivity"],
        "reproduction_scope": "Holdout and temporal statistics are recalculated from anonymized outcome records. Literature and sensitivity aggregates are referenced inputs, not newly estimated hardware results.",
    }


def pct(value):
    return f"{100*value:.2f}%"


def render_tables(result):
    lines = ["# 可复算统计结果", "", "由 `analysis/reproduce.py` 基于公开数据生成。分类标签顺序为 `[7, 1]`。", "",
             "## 独立留出评估", "", "| 条件 | 正确数 | 准确率 | Wilson 95% 区间 |", "|---|---:|---:|---:|"]
    labels = {"qpu_original_head": "原分类头／实机", "qpu_calibrated": "监督校准与固定聚合／实机", "ideal_calibrated": "对应校准规则／精确无噪声参考"}
    for key, row in result["holdout"].items():
        bounds = "–".join(pct(v) for v in row["wilson_95"])
        lines.append(f"| {labels[key]} | {row['correct']}/{row['total']} | {pct(row['accuracy'])} | {bounds} |")
    lines += ["", "原分类头在平衡测试集上输出单一类别；其区间仅为二项公式的条件性计算，不构成有效分类能力的证据。", "",
              "所有区间均未包含全部校准、相关测量误差或时变不确定性。精确模拟一行的区间反映有限图片样本，而非模拟器采样误差。", "",
              "## 同图配对分析", ""]
    pair = result["qpu_minus_ideal_paired"]
    lines += [f"实机独自正确 {pair['candidate_only_correct']} 张；理想参考独自正确 {pair['baseline_only_correct']} 张；两者共同正确 {pair['both_correct']} 张，共同错误 {pair['both_wrong']} 张。",
              f"实机减理想参考的准确率差为 {100*pair['accuracy_difference']:.2f} 个百分点；描述性精确 McNemar 双侧 p={pair['exact_mcnemar_p']:.6f}。该检验不构成等效性检验。", "",
              "## 时间重复测量", "", "| 条件 | 正确数 | 准确率 | Wilson 95% 区间 |", "|---|---:|---:|---:|"]
    names = {"initial": "初始同图子集", "session1_frozen": "时段一／原校准", "session1_refitted": "时段一／重新校准", "session2_frozen": "时段二／原校准", "session2_refitted": "时段二／重新校准"}
    for key, row in result["temporal"].items():
        bounds = "–".join(pct(v) for v in row["wilson_95"])
        lines.append(f"| {names[key]} | {row['correct']}/{row['total']} | {pct(row['accuracy'])} | {bounds} |")
    lines += ["", "各行重复使用同一 80 张图片，不合并为新增独立样本。", ""]
    for key, pair in result["refitted_minus_frozen_paired"].items():
        lines.append(f"- {key}：重新校准减原校准为 {100*pair['accuracy_difference']:.2f} 个百分点；独对/独错为 {pair['candidate_only_correct']}/{pair['baseline_only_correct']}；精确 McNemar p={pair['exact_mcnemar_p']:.6f}。")
    lines += ["", "## 历史基准与文献参照", "", "| 任务 | LQCloud raw | 本地 Aer 有限采样 | 文献理想模拟（约） | IonQ Aria 1＋EM（约） |", "|---|---:|---:|---:|---:|"]
    for case in result["benchmark_reference"]:
        raw, aer = case["lqcloud_raw"], case["local_aer_finite_sample"]
        lines.append(f"| {case['case']} | {raw['correct']}/{raw['total']} | {aer['correct']}/{aer['total']} | {100*case['paper_ideal_accuracy_approx']:.0f}% | {100*case['paper_aria_em_accuracy_approx']:.0f}% |")
    lines += ["", "文献来源：[IonQ 白皮书，第 2.5 节、图 8](https://www.ionq.com/resources/benchmarking-white-paper)。文献数值为图形读取近似值；历史条目不构成同协议配对硬件评估。", ""]
    return "\n".join(lines)
