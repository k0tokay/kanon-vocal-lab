class EditGes:
    def __init__(self, gesture_tiers: dict, segments: list = None):
        """
        :param gesture_tiers: VTLUtterance.gesture_tiers と同じ構造の辞書
                              {tier_name: {"unit": str, "gestures": [dict]}}
        :param segments: VTLUtterance.segments と同じ構造のリスト (optional)
        """
        self.gesture_tiers = gesture_tiers
        self.segments = segments if segments is not None else []
        # ジェスチャーに開始・終了時間を付与（操作用）
        self._add_timing_info()
        self.phoneme_config = self._load_phoneme_config()

    def _load_phoneme_config(self):
        import json
        import os

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

    def select_gesture(self, tier_name: str, start: float, end: float):
        """
        指定した時間範囲（セグメント）に対応するジェスチャーを取得する。
        セグメントの開始時刻を含むジェスチャーを対象とする。
        """
        gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])
        target_time = start
        for i, g in enumerate(gestures):
            if g["_start_time"] <= target_time < g["_end_time"]:
                return i, gestures[i]
        return -1, None

    def update_value(self, tier_name: str, start: float, end: float, new_value: str):
        """
        指定した時間範囲（セグメント）に対応するジェスチャーの値を更新する。
        セグメントの中央時刻を含むジェスチャーを対象とする。
        """
        i, g = self.select_gesture(tier_name, start, end)
        if g:
            print(
                f"Updating gesture value in tier '{tier_name}' from {g['value']} to {new_value}"
            )
            gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])
            gestures[i]["value"] = new_value

    def adjust_length(
        self, tier_name: str, start: float, end: float, adjust_length: float
    ):
        """
        指定した時間範囲に対応するジェスチャーの長さを変更する。
        対象ジェスチャーの長さを固定値(adjust_length)にし、
        その差分を次のジェスチャーで吸収する。
        """
        i, g = self.select_gesture(tier_name, start, end)
        if g is None:
            return

        gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])

        if i + 1 >= len(gestures):
            print("Warning: Cannot adjust length, no next gesture to absorb diff.")
            return

        print(
            f"Adjusting length of gesture in tier '{tier_name}' from {g['duration_s']} to {adjust_length}"
        )

        t_before = float(gestures[i]["duration_s"])
        diff = adjust_length - t_before

        # 長さ更新
        gestures[i]["duration_s"] = adjust_length
        gestures[i + 1]["duration_s"] = float(gestures[i + 1]["duration_s"]) - diff

        # 時間情報の再計算が必要
        self._add_timing_info()

    def scale_duration(
        self,
        tier_name: str,
        start: float,
        end: float,
        left_scale: float = 1.0,
        right_scale: float = 1.0,
    ):
        """
        指定したジェスチャーの長さを左右それぞれスケール倍する。
        左側を伸ばすと前のジェスチャーが縮み、右側を伸ばすと次のジェスチャーが縮む。
        :param left_scale: 左側の伸縮率 (1.0 = 変化なし, >1.0 = 伸びる/前が縮む)
        :param right_scale: 右側の伸縮率
        """
        i, g = self.select_gesture(tier_name, start, end)
        if g is None:
            return

        gestures = self.gesture_tiers.get(tier_name, {}).get("gestures", [])
        current_duration = float(gestures[i]["duration_s"])

        # 左右それぞれの変化量を計算（単純に期間を半分ずつ担当していると仮定して計算するか、
        # あるいは現在の期間全体に対して適用するかだが、ここでは「現在の期間」を基準にする）
        # ただし「左を伸ばす」=「開始時刻を早める」=「前のジェスチャーを削る」

        # 変化量 (正なら自分が増える)
        # ここでは単純に「現在の長さ」に対してスケールを掛けるのではなく、
        # 「現在の長さ」を維持しつつ、境界を移動させるイメージで実装する

        # 左境界の移動量: (scale - 1.0) * (duration / 2) みたいな感じ？
        # 指示が「n%拡大縮小」なので、duration自体が変わる。

        # 解釈:
        # left_scale=1.2 -> 左側に20%伸びる（前のジェスチャーがその分減る）
        # right_scale=0.8 -> 右側が20%縮む（次のジェスチャーがその分増える）
        # 基準となる長さは現在の duration_s とする

        delta_left = current_duration * (left_scale - 1.0)
        delta_right = current_duration * (right_scale - 1.0)

        # 前のジェスチャーチェック
        if i > 0:
            prev_g = gestures[i - 1]
            prev_dur = float(prev_g["duration_s"])
            if prev_dur - delta_left < 0:
                print(f"Warning: Previous gesture too short to shrink by {delta_left}")
                delta_left = prev_dur  # 限界まで縮める

            gestures[i - 1]["duration_s"] = prev_dur - delta_left
        else:
            delta_left = 0  # 前がないなら伸ばせない

        # 次のジェスチャーチェック
        if i + 1 < len(gestures):
            next_g = gestures[i + 1]
            next_dur = float(next_g["duration_s"])
            if next_dur - delta_right < 0:
                print(f"Warning: Next gesture too short to shrink by {delta_right}")
                delta_right = next_dur

            gestures[i + 1]["duration_s"] = next_dur - delta_right
        else:
            delta_right = 0

        # 自身の長さを更新
        new_duration = current_duration + delta_left + delta_right
        gestures[i]["duration_s"] = new_duration

        print(
            f"Scaled gesture in '{tier_name}': {current_duration:.4f} -> {new_duration:.4f} (L:{left_scale}, R:{right_scale})"
        )

        self._add_timing_info()

    def get_gestures(self, segment: dict):
        """
        指定されたセグメントに対応するジェスチャーを取得する。
        セグメントは self.segments に含まれている必要がある。
        """
        if not self.segments:
            return []

        try:
            # 同一オブジェクト検索
            idx = -1
            for i, s in enumerate(self.segments):
                if s is segment:
                    idx = i
                    break

            if idx == -1:
                # 見つからない場合は値で検索（fallback）
                idx = self.segments.index(segment)

        except ValueError:
            print("Segment not found in editor.segments")
            return []

        # 開始時間を計算
        seg_start = 0.0
        for i in range(idx):
            seg_start += float(self.segments[i].get("duration_s", 0.0))

        name = segment.get("name")

        found_gestures = []
        if name in self.phoneme_config:
            config = self.phoneme_config[name]
            tier = config.get("has_gesture_tiers")
            expected_value = config.get("has_gesture_value")

            if tier:
                gestures = self.gesture_tiers.get(tier, {}).get("gestures", [])
                for g in gestures:
                    if g["_start_time"] <= seg_start < g["_end_time"]:
                        found_gestures.append(
                            {
                                "tier": tier,
                                "gesture": g,
                                "expected_value": expected_value,
                                "start": g["_start_time"],
                                "end": g["_end_time"],
                            }
                        )
                        break

        return found_gestures

    def get_segment_gesture_timings(self, segments):
        """
        セグメント列を受け取り、各セグメントに対応するジェスチャーの時間情報を取得する。
        ジェスチャーの特定には phoneme_properties.json の設定を使用する。
        """
        import json
        import os

        json_path = os.path.join(os.path.dirname(__file__), "phoneme_properties.json")
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found.")
            return []

        with open(json_path, "r", encoding="utf-8") as f:
            phoneme_config = json.load(f)

        # ジェスチャーのタイミングを計算
        tier_timings = {}
        for tier_name, tier_data in self.gesture_tiers.items():
            timings = []
            t = 0.0
            for g in tier_data["gestures"]:
                duration = float(g["duration_s"])
                timings.append({"start": t, "end": t + duration, "gesture": g})
                t += duration
            tier_timings[tier_name] = timings

        results = []
        current_time = 0.0

        for seg in segments:
            name = seg.get("name")
            duration = float(seg.get("duration_s", 0.0))
            seg_start = current_time
            seg_end = current_time + duration

            info = {
                "segment_index": len(results),
                "name": name,
                "segment_start": seg_start,
                "segment_end": seg_end,
                "gestures": [],
            }

            if name in phoneme_config:
                config = phoneme_config[name]
                tier = config.get("has_gesture_tiers")
                expected_value = config.get("has_gesture_value")

                if tier and tier in tier_timings:
                    # セグメントの開始時刻を含むジェスチャーを探す
                    found = None
                    for item in tier_timings[tier]:
                        if item["start"] <= seg_start < item["end"]:
                            found = item
                            break

                    if found:
                        info["gestures"].append(
                            {
                                "tier": tier,
                                "start": found["start"],
                                "end": found["end"],
                                "value": found["gesture"]["value"],
                                "expected_value": expected_value,
                            }
                        )

            results.append(info)
            current_time += duration

        return results
