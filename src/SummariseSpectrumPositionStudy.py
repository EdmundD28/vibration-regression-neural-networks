"""Summarise selection and frozen-candidate confirmation without significance claims."""
import argparse
import base64
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np


LABELS = {
    "spectrum_gap_control": "原频谱模型：全局平均",
    "spectrum_flatten": "频谱：直接展平",
    "spectrum_band_pooling": "频谱：分频段汇总",
    "spectrum_compact_flatten": "频谱：紧凑展平",
    "spectrum_gap_wide": "全局平均：参数量对照",
    "time_raw_reference": "原始时序卷积网络",
    "spectrum_flatten_linear": "频谱：展平后线性输出",
    "spectrum_fine_pooling_linear": "频谱：精细池化后线性输出",
}
METRICS = ("mean_normalised_rmse", "voltage_V_rmse", "position_cm_rmse")
TITLES = ("平均归一化误差", "电压均方根误差 / V", "位置均方根误差 / cm")
POSITION_MODELS = ("spectrum_flatten", "spectrum_band_pooling", "spectrum_compact_flatten",
                   "spectrum_flatten_linear", "spectrum_fine_pooling_linear")


def readRows(folder):
    manifest = json.loads((folder / "study_manifest.json").read_text(encoding="utf-8"))
    if manifest["status"] != "complete":
        raise ValueError(f"Incomplete study: {folder}")
    with (folder / "run_metrics.csv").open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def statistics(rows):
    result = {}
    for name in sorted({row["experiment_id"] for row in rows}):
        current = [row for row in rows if row["experiment_id"] == name]
        result[name] = {"label": LABELS[name], "runs": len(current),
                        "parameters": int(current[0]["parameters"])}
        for metric in METRICS:
            values = np.array([float(row[metric]) for row in current])
            result[name][metric] = {"mean": float(values.mean()), "sd": float(values.std(ddof=1)),
                                    "per_seed": {row["seed"]: float(row[metric]) for row in current}}
    return result


def comparisons(stats, candidate):
    result = {}
    for control in ("spectrum_gap_control", "spectrum_gap_wide", "time_raw_reference"):
        if control not in stats:
            continue
        result[control] = {}
        for metric in METRICS:
            baseline = stats[control][metric]
            selected = stats[candidate][metric]
            assert selected["per_seed"].keys() == baseline["per_seed"].keys()
            differences = {seed: selected["per_seed"][seed] - value for seed, value in baseline["per_seed"].items()}
            result[control][metric] = {
                "relative_reduction_percent": 100 * (1 - selected["mean"] / baseline["mean"]),
                "wins": sum(value < 0 for value in differences.values()),
                "paired_difference_per_seed": differences,
            }
    return result


def table(stats):
    rows = []
    for name, value in sorted(stats.items(), key=lambda item: item[1][METRICS[0]]["mean"]):
        cells = [html.escape(LABELS[name])]
        cells.extend(f'{value[metric]["mean"]:.4f} ± {value[metric]["sd"]:.4f}' for metric in METRICS)
        cells.append(f'{value["parameters"]:,}')
        rows.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
    headings = ("结构", *TITLES, "参数量")
    return "<table><thead><tr>" + "".join(f"<th>{value}</th>" for value in headings) + "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--additional-selection", type=Path)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selectionRows = readRows(args.selection)
    if args.additional_selection:
        selectionRows.extend(readRows(args.additional_selection))
    selection = statistics(selectionRows)
    candidate = min((name for name in POSITION_MODELS if name in selection), key=lambda name: selection[name][METRICS[0]]["mean"])
    confirmation = statistics(readRows(args.confirmation))
    confirmationConfig = json.loads((args.confirmation / "study_provenance.json").read_text(encoding="utf-8"))["configuration"]
    assert confirmationConfig["research_protocol"]["frozen_candidate"] == candidate
    assert set(selection[candidate][METRICS[0]]["per_seed"]).isdisjoint(confirmation[candidate][METRICS[0]]["per_seed"])
    outcome = {
        "selected_experiment": candidate,
        "selection_folder": str(args.selection.resolve()),
        "additional_selection_folder": str(args.additional_selection.resolve()) if args.additional_selection else None,
        "confirmation_folder": str(args.confirmation.resolve()),
        "selection": selection, "confirmation": confirmation,
        "selection_comparisons": comparisons(selection, candidate),
        "confirmation_comparisons": comparisons(confirmation, candidate),
        "limitations": "同一组200条记录的重复80/20划分，划分间重叠；验证集同时用于早停。新的随机划分是稳定性检查，不是独立新数据，不能据此宣称显著性或隐藏测试集准确率。",
    }
    outcome["conclusions"] = {
        "aggregation_bottleneck": "保持幅值谱输入不变，保留频率位置后大幅降低数值误差；这有力支持全局频率聚合是重要结构瓶颈，但不构成统计显著性或唯一因果结论。",
        "phase_loss": "幅值谱确实舍弃相位，但本实验没有证明相位缺失导致原模型失利或是本任务必须的信息。",
        "raw_time_comparison": "新频谱模型的位置表现接近原始时序模型，尚未稳定超过；电压及整体表现仍较差，未达到预设5%的整体追平范围。",
        "recommendation": "紧凑七特征全连接网络仍优先用于电压精度和小模型需求；原始时序网络仍是整体表现更强的卷积方案；分频段频谱模型可作为位置预测备选。",
        "legacy_evidence_scope": "旧频谱模型的增益扰动、频段敏感性和残差分析仅适用于旧结构，不能直接归属于新模型。",
    }
    (args.output_dir / "research_summary.json").write_text(json.dumps(outcome, ensure_ascii=False, indent=2), encoding="utf-8")
    flatRows = []
    for phase, stats in (("selection", selection), ("confirmation", confirmation)):
        for name, value in stats.items():
            flatRows.append({"phase": phase, "experiment_id": name, "parameters": value["parameters"],
                             **{f"{metric}_{kind}": value[metric][kind] for metric in METRICS for kind in ("mean", "sd")}})
    with (args.output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(flatRows[0]))
        writer.writeheader()
        writer.writerows(flatRows)
    fontPath = Path("C:/Windows/Fonts/msyh.ttc")
    if fontPath.exists():
        font = FontProperties(fname=str(fontPath))
        plt.rcParams["font.family"] = font.get_name()
    plt.rcParams["axes.unicode_minus"] = False
    names = ("spectrum_gap_control", "spectrum_gap_wide", candidate, "time_raw_reference")
    colors = ("#64748b", "#94a3b8", "#0d9488", "#2563eb")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, metric, title in zip(axes, METRICS, TITLES):
        positions = np.arange(len(names))
        values = [confirmation[name][metric]["mean"] for name in names]
        deviations = [confirmation[name][metric]["sd"] for name in names]
        axis.bar(positions, values, yerr=deviations, capsize=4, color=colors)
        axis.set_xticks(positions, [LABELS[name] for name in names], rotation=25, ha="right", fontsize=9)
        axis.set_title(title)
        axis.set_ylim(0, max(value + sd for value, sd in zip(values, deviations)) * 1.2)
        for position, value in zip(positions, values):
            axis.text(position, value * 0.5, f"{value:.3f}", ha="center", color="white", weight="bold")
        axis.grid(axis="y", alpha=0.15)
        axis.set_axisbelow(True)
    figure.suptitle("冻结结构后的五组新划分：均值与标准差，误差越低越好", fontsize=14)
    figure.tight_layout()
    figure.savefig(args.output_dir / "confirmation_comparison.png", dpi=180)
    plt.close(figure)
    gapReduction = outcome["confirmation_comparisons"]["spectrum_gap_control"][METRICS[0]]["relative_reduction_percent"]
    rawReduction = outcome["confirmation_comparisons"]["time_raw_reference"][METRICS[0]]["relative_reduction_percent"]
    headline = f"{LABELS[candidate]}：相对原频谱模型误差降低 {gapReduction:.1f}%；相对原始时序模型误差变化 {-rawReduction:+.1f}%。"
    imageData = base64.b64encode((args.output_dir / "confirmation_comparison.png").read_bytes()).decode("ascii")
    report = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>频谱网络：频率位置保留实验</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.7 "Microsoft YaHei",sans-serif;color:#193044;background:#f8fafc}}h1{{font-size:30px}}h2{{margin-top:34px;font-size:21px}}table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #dbe3ea;text-align:left}}th{{background:#eaf1f5}}.lead{{padding:18px;background:#e0f2ef;border-radius:12px}}img{{width:100%;margin:20px 0}}small{{color:#53667c}}code{{background:#eaf1f5;padding:2px 5px}}</style>
<h1>频谱卷积网络：保留频率位置后的表现</h1><p class="lead">{headline}</p>
<p>先在固定五组划分中比较三种保留位置的结构；观察到明显改善但尚未追平时序模型后，再在相同五组划分中测试简化输出层与更细局部池化。随后按平均归一化误差冻结最优结构，在另外五组划分中与两种全局平均池化对照及原始时序模型比较，后续不再调参。所有模型使用相同的200条记录、目标分层80/20划分、训练子集标准化、Adam学习率0.001、最多200轮、批量16、早停耐心20。频谱输入始终为0–500 Hz、0.1 Hz分辨率的对数幅值谱；频谱噪声0.01，时序噪声0。</p>
<img alt="新划分验证结果" src="data:image/png;base64,{imageData}">
<h2>更新后的项目结论</h2><p>保持同一幅值谱输入，保留频率位置后获得大幅改善，支持全局频率聚合是重要结构瓶颈。参数量更大的全局平均池化模型仍明显较弱，单纯增加模型容量不能解释新结构的收益。幅值谱舍弃相位是事实，但本实验没有证明相位缺失导致原模型失利，也没有证明池化是唯一原因。</p>
<p>新频谱模型的位置表现接近时序模型，平均误差低3.6%，仅在五组配对划分中的三组获胜；电压误差高33.2%，总体归一化误差高9.4%，未达到预设5%的整体追平范围。紧凑七特征全连接网络仍优先用于电压精度与小模型需求，原始时序模型仍是整体表现更强的卷积方案。分频段频谱模型成为可信的位置预测备选，但需要自身的扰动测试和可解释性分析；旧结构的相关结果不能直接用于新结构。</p>
<h2>第一阶段：结构选择</h2>{table(selection)}
<h2>第二阶段：冻结结构后的新划分</h2>{table(confirmation)}
<h2>结构与机制</h2><p>共同频谱卷积部分：16/32/64个卷积核，核宽21/11/7，三次宽度4的局部最大池化，得到78×64个响应。原模型对78个位置取平均；直接展平保留4992个响应；分频段汇总按6个相邻位置平均后拼接，保留13×64个响应；紧凑展平先用1×1卷积压到4个通道，再保留78×4个响应。全局平均参数量对照把隐藏层扩到2386个神经元，参数总量与直接展平相差22个。</p>
<p>结构检查确认共同卷积部分初始权重一致，三个新读出能够区分跨频率位置移动的相同响应，模型保存和重新加载后的预测一致。</p>
<p>补充结构：展平后线性输出保留原有局部池化，只删除32单元隐藏层，直接把4992个响应映射为两个输出，共30402个参数；精细池化后线性输出把三次局部池化宽度改为2，保留625×64个响应，直接映射为两个输出，共100418个参数。</p>
<h2>证据边界</h2><p>{outcome['limitations']}</p><p>参数量对照能帮助排除“仅仅因为网络更大”的解释；它不能把优化过程、通道压缩和位置信息的作用完全分离。结论适用于本项目的已测试结构，不应推广为所有频谱任务的结论。</p>
<small>第一阶段：{html.escape(str(args.selection.resolve()))}<br>第二阶段：{html.escape(str(args.confirmation.resolve()))}<br>完整配置、代码快照、数据文件摘要、逐次指标和逐样本预测保存在对应研究目录。</small></html>'''
    (args.output_dir / "research_report.html").write_text(report, encoding="utf-8")
    print(json.dumps({"selected_experiment": candidate, "confirmation_comparisons": outcome["confirmation_comparisons"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
