import json
import os


class EditGes:
    def __init__(self, gesture_tiers: dict, segments: list = None, config: dict = None):
        """
        :param gesture_tiers: VTLUtterance.gesture_tiers と同じ構造の辞書
                              {tier_name: {"unit": str, "gestures": [dict]}}
        :param segments: VTLUtterance.segments と同じ構造のリスト (optional)
        """
        self.gesture_tiers = gesture_tiers
        self.segments = segments if segments is not None else []
        # ジェスチャーに開始・終了時間を付与（操作用）
        self._add_timing_info()
        if config is None:
            self.phoneme_config = self._load_phoneme_config()
        else:
            self.phoneme_config = config

        self.map = self.get_gestures_map()

    def _load_phoneme_config(self):
        json_path = os.path.join(os.path.dirname(__file__), "phoneme_properties.json")
        if not os.path.exists(json_path):
            return {}
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _add_timing_info(self):
        """各ジェスチャーに _start_time, _end_time を付与する"""
        for tier in self.gesture_tiers.values():
            current_t = 0.0
            for g in tier["gestures"]:
                g["_start_time"] = current_t
                current_t += float(g["duration_s"])
                g["_end_time"] = current_t

    def _remove_timing_info(self):
        """一時的な時間情報を削除する"""
        for tier in self.gesture_tiers.values():
            for g in tier["gestures"]:
                g.pop("_start_time", None)
                g.pop("_end_time", None)

    def get_data(self):
        """編集後のデータを取得する（時間情報は削除される）"""
        self._remove_timing_info()
        return self.gesture_tiers

    def select_gesture(self, tier_name: str, start: float, end: float, value=None):
        """
        指定した時間範囲（セグメント）に対応するジェスチャーを取得する。
        セグメントの開始時刻を含むジェスチャーを対象とする。
        """
        gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])
        if not gestures:
            return -1, None

        cand = -1
        target_time = start
        for i, g in enumerate(gestures):
            if g["_start_time"] <= target_time < g["_end_time"]:
                cand = i
                break

        if cand == -1:
            # Overlap not found, find closest
            min_dist = float("inf")
            for i, g in enumerate(gestures):
                dist = min(
                    abs(g["_start_time"] - target_time),
                    abs(g["_end_time"] - target_time),
                )
                if dist < min_dist:
                    min_dist = dist
                    cand = i

        # 0, -1, 1, -2, 2 ... の順でiの周りを探索して，valueが一致する最も近いジェスチャーを返す
        for offset in range(len(gestures)):
            for sign in [-1, 1]:
                if offset == 0 and sign == 1:
                    continue
                i = cand + sign * offset
                if 0 <= i < len(gestures):
                    print(
                        "  check",
                        i,
                        cand,
                        sign * offset,
                        gestures[i]["value"],
                        "for value",
                        value,
                        gestures[i]["value"] == value,
                    )
                    if gestures[i]["value"] in ["stop"] or gestures[i]["neutral"]:
                        continue
                    if value is None or gestures[i]["value"] == value:
                        return i, gestures[i]

        return -1, None

    def s2g(self, idx):
        """セグメントインデックスから対応するジェスチャーを取得する"""
        map_ = self.map
        info = map_[idx]
        tier_name = info["tier"]
        if info["tier"] != tier_name:
            return None
        ges_idx = info["index"]
        if ges_idx == -1:
            return None
        gestures = self.gesture_tiers[tier_name]["gestures"]
        return gestures[ges_idx]

    def s2gls(self, idx):
        """セグメントインデックスから対応する声門形状ジェスチャーを取得する"""
        map_ = self.map
        info = map_[idx]
        ges_idx = info["glottal_shape_index"]
        if ges_idx == -1:
            return None
        gestures = self.gesture_tiers["glottal-shape-gestures"]["gestures"]
        return gestures[ges_idx]

    def adjust_gesture_timing(
        self, ges_idx: int, tier_name: str, delta_start: float, delta_end: float
    ):
        """指定したジェスチャーの開始・終了時間を調整する"""
        gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])
        if not (0 <= ges_idx < len(gestures)):
            return
        ges = gestures[ges_idx]

        # 開始時間の調整
        new_start = ges["_start_time"] + delta_start
        if ges_idx > 0:
            prev_ges = gestures[ges_idx - 1]
            prev_end = prev_ges["_end_time"]
            if new_start < prev_end:
                new_start = prev_end
            prev_ges["duration_s"] = new_start - prev_ges["_start_time"]
            prev_ges["_end_time"] = new_start
        ges["duration_s"] = ges["_end_time"] - new_start
        ges["_start_time"] = new_start

        # 終了時間の調整
        new_end = ges["_end_time"] + delta_end
        if ges_idx + 1 < len(gestures):
            next_ges = gestures[ges_idx + 1]
            next_start = next_ges["_start_time"]
            if new_end > next_start:
                new_end = next_start
            next_ges["duration_s"] = next_ges["_end_time"] - new_end
            next_ges["_start_time"] = new_end
        ges["duration_s"] = new_end - ges["_start_time"]
        ges["_end_time"] = new_end

    def process_replacements(self):
        map_ = self.map

        for i, seg in enumerate(self.segments):
            replace_from = seg.get("replace_from")
            if not replace_from:
                name = seg.get("name")
                if name == "b":
                    self.adjust_gesture_timing(
                        map_[i]["index"], map_[i]["tier"], 0.01, -0.01
                    )
                continue

            ges = self.s2g(i)

            if replace_from == "M":
                ges["value"] = "M"

            elif replace_from == "w":
                ges["value"] = "ll-labial-approx"
                ges_gs = self.s2gls(i)
                if ges_gs:
                    ges_gs["value"] = "modal"

            elif replace_from in ["φ", "β"]:
                ges["value"] = "ll-labial-fricative"

            elif replace_from in ["u_", "i_"]:
                ges["value"] = "u" if replace_from == "u_" else "i"

                # 一つ前の音素の終了時間よりも前に終了するように調整
                ges_before = self.s2g(i - 1)
                print("ges_before", ges_before)
                end_time_new = ges_before["_end_time"] - 0.02
                shorten_length = ges["_end_time"] - end_time_new
                self.adjust_gesture_timing(
                    map_[i]["index"], map_[i]["tier"], 0.0, -shorten_length
                )
                print(map_[i])
                ges_gs = self.s2gls(i)
                ges_gs["value"] = "modal"

            elif replace_from == "ɾ":
                ges["value"] = "tt-alveolar-flap"
                # 母音の開点を中心に半分ずつ
                total_length = 0.05
                ges_after = self.s2g(i + 1)
                mid_time_new = ges_after["_start_time"]
                start_time_new = mid_time_new - total_length / 2
                end_time_new = mid_time_new + total_length / 2

                self.adjust_gesture_timing(
                    map_[i]["index"],
                    map_[i]["tier"],
                    start_time_new - ges["_start_time"],
                    end_time_new - ges["_end_time"],
                )
                # 舌の動きを速くする
                ges["time_constant_s"] = 0.009

                ges_gs = self.s2gls(i)
                ges_gs["value"] = "modal"

    def get_gestures_map(self):
        map = []
        current_t = 0.0
        for i in range(len(self.segments)):
            seg = self.segments[i]
            seg_start = current_t
            seg_end = seg_start + float(seg.get("duration_s", 0.0))
            current_t = seg_end
            tier = self.phoneme_config[seg["name"]].get("has_gesture_tiers")
            value = self.phoneme_config[seg["name"]].get("has_gesture_value")
            print("seg", i, seg["name"], tier, value)
            ges_idx, ges = self.select_gesture(tier, seg_start, seg_end, value=value)
            ges_gs_idx, ges_gs = self.select_gesture(
                "glottal-shape-gestures", seg_start, seg_end
            )
            map.append(
                {
                    "segment_index": i,
                    "segment_start": seg_start,
                    "segment_end": seg_end,
                    "index": ges_idx,
                    "tier": tier,
                    "gesture": ges,
                    "glottal_shape_index": ges_gs_idx,
                    "glottal_shape_gesture": ges_gs,
                }
            )
        return map
