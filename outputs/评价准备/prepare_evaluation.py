"""核验既有清洗并冻结记录级评价边界；不清洗、不做EDA、不训练模型。"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SPLITS = ["train", "validation", "test"]
TIME_FORMAT = "%Y-%m-%d %H:%M:%S.%f"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_csv(path, frame):
    frame.to_csv(path, index=False, encoding="utf-8-sig", lineterminator="\n",
                 date_format=TIME_FORMAT, float_format="%.10g")


def verify_inputs():
    """只校验归档字节与既有验收，不重运行清洗。"""
    clean = PROJECT / "outputs/数据清洗"
    manifest = clean / "交付文件哈希.json"
    records = []
    for name, expected in read_json(manifest).items():
        path = clean / Path(name.replace("\\", "/"))
        actual = sha256(path)
        require(actual == expected, f"归档文件哈希变化：{name}")
        records.append({"role": "cleaning_archive", "path": str(path.relative_to(PROJECT)),
                        "expected_sha256": expected, "actual_sha256": actual, "matched": True})
    for item in read_json(PROJECT / "outputs/项目管理/输入来源.json")["inputs"]:
        path = Path(item["path"])
        actual = sha256(path)
        require(actual == item["sha256"], f"原始输入哈希变化：{path}")
        records.append({"role": item["role"], "path": str(path),
                        "expected_sha256": item["sha256"], "actual_sha256": actual, "matched": True})
    audit = read_json(clean / "results/清洗验收.json")
    require(all(audit["checks"].values()), "既有清洗验收包含失败项")
    return clean / "results", records


def assign_segments(meta, segments, rules):
    """只接收身份、时间及片段计数，不读取标签或测量值。"""
    counts = meta.groupby(["vehicle", "segment_id"]).size().rename("candidate_rows")
    result = segments.merge(counts, on=["vehicle", "segment_id"], how="left", validate="one_to_one")
    result["candidate_rows"] = result.candidate_rows.fillna(0).astype(int)
    result = result.sort_values(["vehicle", "start", "segment_id"]).reset_index(drop=True)
    result["split"] = ""
    decisions = []
    for vehicle, group in result.groupby("vehicle", sort=True):
        active = group[group.candidate_rows > 0]
        n = len(active)
        require(n > 0, f"{vehicle}没有候选记录")
        if n == 1:
            train_end, val_end, policy = 1, 1, "single_segment_train_only"
        elif n == 2:
            train_end, val_end, policy = 1, 1, "two_segments_no_validation"
        else:
            # 整数目标(3/5,1/5,1/5)，避免浮点平局导致平台差异。
            require(rules["target_ratios"] == [0.6, 0.2, 0.2], "当前算法冻结为60/20/20")
            values = active.candidate_rows.to_numpy()
            cumulative = np.cumsum(values)
            total = int(cumulative[-1])
            candidates = []
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    parts = [int(cumulative[i-1]), int(cumulative[j-1]-cumulative[i-1]),
                             int(total-cumulative[j-1])]
                    score = sum(abs(5*x - weight*total) for x, weight in zip(parts, [3, 1, 1]))
                    candidates.append((score, i, j))
            _, train_end, val_end = min(candidates)
            policy = "whole_segments_count_target_60_20_20"
        validation_start = active.iloc[train_end].start if val_end > train_end else None
        test_start = active.iloc[val_end].start if val_end < n else None
        for ix, row in group.iterrows():
            # 仅GPS片段也按固定时间边界分配，供后续全长表/宽表取开发范围。
            split = "train"
            if validation_start is not None and row.start >= validation_start:
                split = "validation"
            if test_start is not None and row.start >= test_start:
                split = "test"
            result.loc[ix, "split"] = split
        decisions.append({"vehicle": vehicle, "policy": policy, "candidate_segments": n,
                          "train_segment_count": train_end,
                          "validation_segment_count": val_end-train_end,
                          "test_segment_count": n-val_end,
                          "validation_start": str(validation_start) if validation_start is not None else None,
                          "test_start": str(test_start) if test_start is not None else None})
    return result, decisions


def validate_boundaries(assigned, guard_seconds):
    """使用全长表片段的末/首次时间，不只看候选记录的间隔。"""
    checks = []
    for vehicle, g in assigned.groupby("vehicle", sort=True):
        g = g.sort_values("start")
        for left, right in zip(g.iloc[:-1].itertuples(), g.iloc[1:].itertuples()):
            if left.split == right.split:
                continue
            gap = (right.start-left.end).total_seconds()
            require(gap > guard_seconds, f"{vehicle}划分间隔{gap}不满足隔离要求")
            checks.append({"vehicle": vehicle, "left_split": left.split, "right_split": right.split,
                           "left_segment": left.segment_id, "right_segment": right.segment_id,
                           "left_last_raw_time": left.end, "right_first_raw_time": right.start,
                           "raw_gap_seconds": gap, "required_guard_seconds": guard_seconds,
                           "passed": True})
    return pd.DataFrame(checks)


def self_test():
    rules = read_json(HERE / "划分规则.json")
    segments, records = [], []
    source_id = 0
    for vehicle, sizes in [("A", [6, 2, 2]), ("B", [7, 3]), ("C", [5])]:
        for i, size in enumerate(sizes):
            start = pd.Timestamp("2020-01-01") + pd.Timedelta(seconds=i*1000)
            segment_id = f"{vehicle}_{i}"
            segments.append({"vehicle": vehicle, "segment_id": segment_id,
                             "start": start, "end": start+pd.Timedelta(seconds=size)})
            for k in range(size):
                source_id += 1
                records.append({"source_row_id": source_id, "vehicle": vehicle,
                                "segment_id": segment_id, "time_local": start+pd.Timedelta(seconds=k)})
    meta, seg = pd.DataFrame(records), pd.DataFrame(segments)
    assigned, _ = assign_segments(meta, seg, rules)
    require(assigned.split.tolist() == ["train", "validation", "test", "train", "test", "train"],
            "完整片段/两片段/单片段测试失败")
    require(len(validate_boundaries(assigned, 120)) == 3, "边界数错误")
    shuffled, _ = assign_segments(meta.sample(frac=1, random_state=42), seg, rules)
    pd.testing.assert_frame_equal(assigned, shuffled)
    # 必须拒绝刚好120秒的边界，防止含端点历史重用。
    bad = assigned.copy()
    bad.loc[0, "end"] = bad.loc[1, "start"]-pd.Timedelta(seconds=120)
    try:
        validate_boundaries(bad, 120)
    except ValueError:
        pass
    else:
        raise ValueError("未拒绝不足隔离的边界")
    # 元数据入口不需要目标列；同车同刻多变量不能打散。
    duplicate_time = pd.concat([meta, meta.iloc[[0]].assign(source_row_id=999)], ignore_index=True)
    extra, _ = assign_segments(duplicate_time, seg, rules)
    require(extra[extra.vehicle == "A"].split.tolist() == ["train", "validation", "test"], "同刻分配错误")
    return {"status": "PASS", "checks": ["whole_segment_assignment", "two_segment_exception",
            "one_segment_exception", "row_order_invariance", "strict_guard_endpoint",
            "same_timestamp_assignment", "label_free_input"]}


def run(output):
    rules_path = HERE / "划分规则.json"
    rules = read_json(rules_path)
    require(rules["guard_seconds"] >= rules["history_window_seconds"]+rules["maximum_asof_ttl_seconds"],
            "隔离带不足以覆盖预登记最大回溯")
    output = Path(output).resolve()
    require(output.is_relative_to(PROJECT), "输出必须位于项目内")
    require(not output.is_relative_to(PROJECT / "outputs/数据清洗"), "不得写入清洗归档")
    data_path, hash_records = verify_inputs()
    # 第一次只读结构元数据；边界生成之后才读取标签进行覆盖审计。
    meta = pd.read_csv(data_path / "清洗长表.csv", usecols=["source_row_id", "deviceId", "vehicle",
        "segment_id", "time_local", "variable", "variable_key", "value_type"])
    meta["time_local"] = pd.to_datetime(meta.time_local, format=TIME_FORMAT)
    require(len(meta) == 173853 and meta.source_row_id.is_unique, "原行数/索引不符")
    require(set(meta.source_row_id) == set(range(1, 173854)), "原始行索引不连续完整")
    require(not meta.isna().any().any(), "身份/时间/字典字段出现空值")
    dictionary = pd.read_csv(data_path / "变量字典.csv")
    require(len(dictionary) == 52 and dictionary.variable.is_unique and dictionary.variable_key.is_unique,
            "变量字典键不唯一")
    mapping = meta[["variable", "variable_key", "value_type"]].drop_duplicates()
    require(len(mapping) == 52, "清洗长表字典映射不唯一")
    joined = mapping.merge(dictionary, on=["variable", "variable_key", "value_type"], how="inner")
    require(len(joined) == 52, "字典与长表映射不一致")
    vehicles = meta[["vehicle", "deviceId"]].drop_duplicates().sort_values("vehicle")
    require(len(vehicles) == 3 and vehicles.vehicle.is_unique and vehicles.deviceId.is_unique,
            "车辆编码不唯一")
    candidate = meta[~meta.variable.isin(rules["exclude_variables"])].copy()
    require(len(candidate) == 83463 and candidate.variable.nunique() == 50, "候选样本规模不符")
    variables = mapping[~mapping.variable.isin(rules["exclude_variables"])].sort_values("variable_key").copy()
    variables.insert(0, "variable_code", range(1, len(variables)+1))
    candidate = candidate.merge(variables, on=["variable", "variable_key", "value_type"], validate="many_to_one")
    segments = pd.read_csv(data_path / "观测片段.csv")
    for col in ["start", "end"]:
        segments[col] = pd.to_datetime(segments[col], format=TIME_FORMAT)
    actual_segments = meta.groupby(["vehicle", "segment_id"]).agg(
        start=("time_local", "min"), end=("time_local", "max"), raw_rows=("source_row_id", "size")).reset_index()
    require(len(segments) == 44, "观测片段数变化")
    pd.testing.assert_frame_equal(segments[actual_segments.columns].sort_values("segment_id").reset_index(drop=True),
                                  actual_segments.sort_values("segment_id").reset_index(drop=True))
    assigned, decisions = assign_segments(candidate[["vehicle", "segment_id", "time_local"]], segments, rules)
    boundary_checks = validate_boundaries(assigned, rules["guard_seconds"])
    candidate = candidate.merge(assigned[["vehicle", "segment_id", "split"]],
                                on=["vehicle", "segment_id"], validate="many_to_one")
    require(candidate.groupby(["vehicle", "time_local"]).split.nunique().max() == 1, "同车同刻被拆分")
    require(candidate.groupby(["vehicle", "segment_id"]).split.nunique().max() == 1, "完整片段被拆分")
    require(set(candidate.split) == set(SPLITS), "整体缺少训练/验证/测试集合")
    frozen = {"schema_version": 1, "rules_sha256": sha256(rules_path),
              "clean_long_sha256": sha256(data_path / "清洗长表.csv"), "decisions": decisions,
              "segment_assignments": assigned[["vehicle", "segment_id", "split"]].to_dict("records")}
    output.mkdir(parents=True, exist_ok=True)
    frozen_path = output / "冻结边界.json"
    if frozen_path.exists():
        require(read_json(frozen_path) == frozen, "冻结边界有变化；必须另建版本，不得覆盖")
    write_json(frozen_path, frozen)
    index_cols = ["source_row_id", "vehicle", "segment_id", "time_local", "variable_code", "split"]
    write_csv(output / "候选样本索引.csv", candidate[index_cols].sort_values("source_row_id"))
    write_csv(output / "观测片段划分.csv", assigned)
    write_csv(output / "时间边界核查.csv", boundary_checks)
    write_csv(output / "车辆编码.csv", vehicles)
    write_csv(output / "变量编码.csv", variables)
    write_csv(output / "输入哈希核验.csv", pd.DataFrame(hash_records))
    # 以下标签只用于审计支持数，不反馈给选边界算法。
    labels = pd.read_csv(data_path / "清洗长表.csv", usecols=["source_row_id", "alarm_class", "value_num"])
    labeled = candidate.merge(labels, on="source_row_id", validate="one_to_one")
    require(set(labels.alarm_class) == set(rules["labels"]), "类别范围变化")
    require(np.isfinite(labeled.value_num).all(), "候选当前值存在非有限值，应先调查")
    grid = pd.MultiIndex.from_product([vehicles.vehicle, SPLITS, rules["labels"]],
                                     names=["vehicle", "split", "alarm_class"])
    coverage = labeled.groupby(["vehicle", "split", "alarm_class"]).size().reindex(grid, fill_value=0)
    write_csv(output / "类别支持数.csv", coverage.rename("records").reset_index())
    summary = labeled.groupby(["vehicle", "split"]).agg(records=("source_row_id", "size"),
        segments=("segment_id", "nunique"), first_time=("time_local", "min"), last_time=("time_local", "max"))
    summary = summary.reindex(pd.MultiIndex.from_product([vehicles.vehicle, SPLITS], names=["vehicle", "split"]))
    summary[["records", "segments"]] = summary[["records", "segments"]].fillna(0).astype(int)
    summary = summary.reset_index()
    summary["within_vehicle_fraction"] = summary.records / summary.groupby("vehicle").records.transform("sum")
    write_csv(output / "划分汇总.csv", summary)
    variable_grid = pd.MultiIndex.from_product([variables.variable_code, SPLITS], names=["variable_code", "split"])
    variable_support = labeled.groupby(["variable_code", "split"]).size().reindex(variable_grid, fill_value=0)
    write_csv(output / "变量支持数.csv", variable_support.rename("records").reset_index())
    wide = pd.read_csv(data_path / "状态宽表_10秒_未填补.csv")
    filled = pd.read_csv(data_path / "状态宽表_10秒_有限历史对齐.csv")
    keys = ["sample_id", "vehicle", "segment_id", "reference_time"]
    require(wide.shape == (3132, 57) and filled.shape == wide.shape, "状态宽表结构变化")
    pd.testing.assert_frame_equal(wide[keys], filled[keys])
    require(wide.sample_id.is_unique and not wide.duplicated(keys[1:]).any(), "宽表主键重复")
    wide_scope = wide[keys].merge(assigned[["vehicle", "segment_id", "split"]],
                                on=["vehicle", "segment_id"], how="left", validate="many_to_one")
    require(not wide_scope.split.isna().any(), "宽表分区未覆盖")
    write_csv(output / "状态样本分析范围.csv", wide_scope)
    # 记录排除口径；未删除任何原始或清洗行。
    exclusions = [{"reason": "POSITION_not_scalar_target", "records": int((meta.variable == "POSITION").sum())},
                  {"reason": "unknown_PID", "records": int((meta.variable == "PIDNameNotAvailable").sum())},
                  {"reason": "temporal_guard", "records": 0},
                  {"reason": "cross_boundary_episode", "records": 0}]
    write_csv(output / "样本排除计数.csv", pd.DataFrame(exclusions))
    folds = []
    for held in vehicles.vehicle:
        for vehicle in vehicles.vehicle:
            for split in SPLITS:
                role = "test" if vehicle == held else (split if split != "test" else "unused")
                n = int(summary.loc[(summary.vehicle == vehicle) & (summary.split == split), "records"].iloc[0])
                folds.append({"held_vehicle": held, "vehicle": vehicle, "main_split": split,
                              "fold_role": role, "records": n})
    write_csv(output / "留一车辆预登记.csv", pd.DataFrame(folds))
    checks = {"input_and_archive_hashes_match": True, "raw_row_ids_complete_unique": True,
              "dictionary_vehicle_mapping_consistent": True, "segment_table_matches_long": True,
              "same_vehicle_time_not_split": True, "whole_segment_not_split": True,
              "chronological_order_guard_passed": True, "candidate_numeric_values_finite": True,
              "wide_keys_and_scope_consistent": True, "labels_not_used_for_cut_selection": True}
    train_labels = set(labeled.loc[labeled.split == "train", "alarm_class"])
    test_labels = set(labeled.loc[labeled.split == "test", "alarm_class"])
    write_json(output / "准备验收.json", {"status": "PASS", "checks": checks,
        "raw_rows": len(meta), "candidate_rows": len(candidate), "candidate_variables": len(variables),
        "all_segments": len(assigned), "candidate_segments": int((assigned.candidate_rows > 0).sum()),
        "state_rows": len(wide), "hash_checked_files": len(hash_records),
        "split_totals": {s: int((candidate.split == s).sum()) for s in SPLITS},
        "minimum_boundary_gap_seconds": float(boundary_checks.raw_gap_seconds.min()),
        "train_absent_labels": sorted(set(rules["labels"])-train_labels),
        "test_absent_labels": sorted(set(rules["labels"])-test_labels),
        "zero_validation_vehicles": summary.loc[(summary.split == "validation") & (summary.records == 0), "vehicle"].tolist(),
        "temporal_guard_exclusions": 0, "cross_boundary_episode_exclusions": 0,
        "episode_check_status": "structurally_non_crossing_under_same_segment_rule; recheck_membership_when_episodes_exist",
        "evaluation_type": "internal_temporal_holdout_per_vehicle_not_global_calendar_future",
        "prior_exposure": "full_data_quality_diagnostics_and_label_counts_already_seen; not_external_blind_validation",
        "not_executed": ["EDA", "episode_detection", "feature_fitting", "model_training", "final_test_evaluation"]})
    # 包含所有正式计算输入与输出哈希；复现清单的运行时元数据另由技能工具生成。
    write_json(output / "运行环境.json", {"python": sys.version.split()[0], "platform": platform.platform(),
        "pandas": pd.__version__, "numpy": np.__version__, "random_seed": rules["random_seed"], "randomness_used": False})
    inputs = [HERE / "prepare_evaluation.py", rules_path, PROJECT / "outputs/项目管理/输入来源.json",
              PROJECT / "outputs/数据清洗/交付文件哈希.json"]
    write_json(output / "输入代码哈希.json", {str(p.relative_to(PROJECT)): sha256(p) for p in inputs})
    names = ["冻结边界.json", "候选样本索引.csv", "观测片段划分.csv", "时间边界核查.csv", "车辆编码.csv",
             "变量编码.csv", "输入哈希核验.csv", "类别支持数.csv", "划分汇总.csv", "变量支持数.csv",
             "状态样本分析范围.csv", "样本排除计数.csv", "留一车辆预登记.csv", "准备验收.json",
             "运行环境.json", "输入代码哈希.json"]
    write_json(output / "输出哈希.json", {name: sha256(output / name) for name in names})
    print(json.dumps(read_json(output / "准备验收.json"), ensure_ascii=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(self_test(), ensure_ascii=True))
    else:
        run(args.output)
