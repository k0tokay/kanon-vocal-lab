import json
import os
import re
import xml.etree.ElementTree as ET
from xml.dom import minidom
import numpy as np

import sys

# vtl_apiへのパスを通す
sys.path.append(os.path.join(os.path.dirname(__file__), "../vtl_api"))
from vtl_api import VocalTractLab


class VTLUtterance:
    def __init__(self):
        self.total_duration_s = 0.0
        self.linguistic_blocks = []  # .ling用
        self.segments = []  # .seg用
        self.gesture_tiers = {}  # .ges用: {type: [gestures]}

    @staticmethod
    def gesture_tiers_template():
        return {
            "vowel-gestures": {"unit": "", "gestures": []},
            "lip-gestures": {"unit": "", "gestures": []},
            "tongue-tip-gestures": {"unit": "", "gestures": []},
            "tongue-body-gestures": {"unit": "", "gestures": []},
            "velic-gestures": {"unit": "", "gestures": []},
            "glottal-shape-gestures": {"unit": "", "gestures": []},
            "f0-gestures": {"unit": "st", "gestures": []},
            "lung-pressure-gestures": {"unit": "dPa", "gestures": []},
        }

    @staticmethod
    def merge_gesture_tiers(ges1, ges2):
        """2つのジェスチャースコアを時間軸に沿って結合する"""
        merged = VTLUtterance.gesture_tiers_template()
        for tier_type in merged.keys():
            g1 = ges1.get(tier_type, {}).get("gestures", [])
            g2 = ges2.get(tier_type, {}).get("gestures", [])
            merged[tier_type]["unit"] = ges1.get(tier_type, {}).get(
                "unit", ""
            ) or ges2.get(tier_type, {}).get("unit", "")
            merged[tier_type]["gestures"] = g1 + g2
        return merged

    def load_ling(self, filepath: str):
        """
        Linguistic score file (XML) を読み込む
        """
        self.linguistic_blocks = []
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()

            if root.tag != "linguistic_score":
                raise ValueError(f"Invalid root element: {root.tag}")

            for block in root.findall("block"):
                block_data = {
                    "context": block.find("context").attrib
                    if block.find("context") is not None
                    else {},
                    "baseline": block.find("baseline").attrib
                    if block.find("baseline") is not None
                    else {},
                    "content": block.find("content").text.strip()
                    if block.find("content") is not None
                    else "",
                }

                if block.find("melody") is not None:
                    block_data["melody"] = block.find("melody").text.strip()

                self.linguistic_blocks.append(block_data)

            print(
                f"Successfully loaded {len(self.linguistic_blocks)} linguistic blocks."
            )
        except Exception as e:
            print(f"Error loading linguistic score: {e}")

    def parse_ling(self):
        """
        Linguistic score の内容を解析する
        """
        gesture_tiers = self.gesture_tiers_template()

        for block in self.linguistic_blocks:
            context = block.get("context", {})
            lang = context.get("lang", "unknown")
            type_ = context.get("type", "unknown")

            content = block.get("content", "")

            if lang == "jp":
                parser = JPParser()
                block_gesture_tiers = parser.parse(content)
                gesture_tiers = self.merge_gesture_tiers(
                    gesture_tiers, block_gesture_tiers
                )
            else:
                print(f"Unsupported language: {lang}")

        self.gesture_tiers = gesture_tiers
        return gesture_tiers

    def load_seg(self, filepath: str):
        """
        Segment sequence file (*.seg) を読み込む
        """
        self.segments = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if not lines:
                    return

                # 各セグメント（音素）の定義
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # セミコロンで区切られた属性ペアをパース
                    attributes = {}
                    parts = line.split(";")
                    for part in parts:
                        if "=" in part:
                            key, val = part.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            # 数値として扱えるものは変換
                            try:
                                if "." in val:
                                    val = float(val)
                                elif val.isdigit():
                                    val = int(val)
                            except ValueError:
                                pass
                            attributes[key] = val

                    if attributes:
                        self.segments.append(attributes)

            print(f"Loaded {len(self.segments)} segments from {filepath}")
        except Exception as e:
            print(f"Error loading .seg file: {e}")

    def load_ges(self, filepath: str):
        """
        Gestural score file (*.ges) を読み込む。 [cite: 1196]
        XML形式で、8つのジェスチャーシーケンス（ティア）が含まれる。 [cite: 1197, 1199]
        """
        self.gesture_tiers = {}
        try:
            tree = ET.parse(filepath)
            root = tree.getroot()  # ルートは gestural_score

            # gesture_sequence 要素をループ [cite: 1199]
            for seq_elem in root.findall("gesture_sequence"):
                tier_type = seq_elem.get(
                    "type"
                )  # vowel, lip, tongue_tip 等 [cite: 870]
                unit = seq_elem.get("unit")

                gestures = []
                current_time = 0.0  # 開始時間は前のジェスチャ期間の合計

                for g_elem in seq_elem.findall("gesture"):
                    duration = float(g_elem.get("duration_s", 0))

                    gesture_data = {
                        "value": g_elem.get("value"),
                        "slope": float(g_elem.get("slope", 0))
                        if g_elem.get("slope")
                        else 0.0,
                        "duration_s": duration,
                        "time_constant_s": float(g_elem.get("time_constant_s", 0)),
                        "neutral": g_elem.get("neutral") == "1",
                        "start_time": current_time,  # 暗黙的な開始時間を計算
                    }
                    gestures.append(gesture_data)
                    current_time += duration

                self.gesture_tiers[tier_type] = {"unit": unit, "gestures": gestures}

            print(f"Loaded {len(self.gesture_tiers)} tiers from {filepath}")
        except Exception as e:
            print(f"Error loading .ges file: {e}")

    def save_ling(self, filepath: str):
        """
        Linguistic score file (XML) として保存する。
        """
        try:
            # ルート要素の作成
            root = ET.Element("linguistic_score")

            for block_data in self.linguistic_blocks:
                block_node = ET.SubElement(root, "block")

                # context 要素の作成（属性として格納）
                context_attrs = {
                    k: str(v) for k, v in block_data.get("context", {}).items()
                }
                ET.SubElement(block_node, "context", context_attrs)

                # baseline 要素の作成（属性として格納）
                baseline_attrs = {
                    k: str(v) for k, v in block_data.get("baseline", {}).items()
                }
                ET.SubElement(block_node, "baseline", baseline_attrs)

                # content 要素の作成（テキストとして格納）
                content_node = ET.SubElement(block_node, "content")
                content_node.text = block_data.get("content", "")

                # melody 要素の作成（存在する場合のみ、テキストとして格納）
                if "melody" in block_data:
                    melody_node = ET.SubElement(block_node, "melody")
                    melody_node.text = block_data["melody"]

            # XMLを文字列に変換し、整形して保存
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

            print(
                f"Successfully saved {len(self.linguistic_blocks)} linguistic blocks to {filepath}"
            )

        except Exception as e:
            print(f"Error saving linguistic score: {e}")

    def save_seg(self, filepath: str, data=None):
        """
        Segment sequence file (*.seg) として保存。
        VTLの仕様に基づき、セミコロン区切りのテキスト形式で出力する。
        """
        if data is None:
            data = self.segments

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                for seg in data:
                    name = seg.get("name", "")
                    duration = seg.get("duration_s", 0.0)

                    attr_parts = [f"name = {name}", f"duration_s = {duration:.6f}"]

                    for key, val in seg.items():
                        if key not in ["name", "duration_s"]:
                            attr_parts.append(f"{key} = {val}")

                    f.write("; ".join(attr_parts) + ";\n")

            print(f"Successfully saved segment sequence to {filepath}")
        except Exception as e:
            print(f"Error saving .seg file: {e}")

    def save_ges(self, filepath: str, data=None):
        """
        Gestural score file (*.ges) として保存。
        XML形式で出力する。
        """
        if data is None:
            data = self.gesture_tiers

        try:
            # ルート要素の作成 [cite: 1198, 1202]
            root = ET.Element("gestural_score")

            # 各ティア (gesture_sequence) を作成 [cite: 1199, 1204]
            # 8つのティア（vowel, lip, tongue tip, tongue body, velic, glottal shape, F0, lung pressure）
            for tier_type, content in data.items():
                seq_elem = ET.SubElement(
                    root,
                    "gesture_sequence",
                    {"type": tier_type, "unit": content.get("unit", "")},
                )

                # 各ジェスチャを作成
                for g in content.get("gestures", []):
                    # XML属性はすべて文字列である必要がある
                    attr = {
                        "value": str(g.get("value", "")),
                        "slope": f"{float(g.get('slope', 0)):.6f}",
                        "duration_s": f"{float(g.get('duration_s', 0)):.6f}",
                        "time_constant_s": f"{float(g.get('time_constant_s', 0)):.6f}",
                        "neutral": "1" if g.get("neutral") else "0",
                    }
                    ET.SubElement(seq_elem, "gesture", attr)

            # きれいに整形（インデント付与）して保存
            xml_str = ET.tostring(root, encoding="utf-8")
            pretty_xml = minidom.parseString(xml_str).toprettyxml(indent="  ")

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(pretty_xml)

            print(f"Successfully saved gestural score to {filepath}")
        except Exception as e:
            print(f"Error saving .ges file: {e}")

    def ling_to_seg(self):
        self.segments = []
        for block in self.linguistic_blocks:
            context = block.get("context", {})
            lang = context.get("lang", "unknown")
            content = block.get("content", "")
            baseline = block.get("baseline", {})
            tempo = float(baseline.get("tempo", 100))
            f0 = float(baseline.get("f0", 250))

            if lang == "jp":
                parser = JPParser()
                segments = parser.parse_to_segments(content, tempo, f0)
                self.segments.extend(segments)
            else:
                print(f"Unsupported language: {lang}")

        print(f"Generated {len(self.segments)} segments from linguistic score.")

    def _preprocess_s2g(self, seg_data=None):
        if seg_data is None:
            seg_data = self.segments

        # 母音の処理
        replace_map = {"M": "u"}
        tmp_segments = []
        for seg in seg_data:
            seg_copy = seg.copy()
            if seg_copy.get("name") in replace_map:
                seg_copy["name"] = replace_map[seg_copy["name"]]
            tmp_segments.append(seg_copy)

        return tmp_segments

    def _postprocess_s2g(self, ges_data=None):
        if ges_data is None:
            ges_data = self.gesture_tiers

        # 母音の処理を元に戻す
        reverse_map = {"u": "M"}
        vowel_gestures = ges_data.get("vowel-gestures", {}).get("gestures", [])
        for g in vowel_gestures:
            if g.get("value") in reverse_map:
                g["value"] = reverse_map[g["value"]]

        return ges_data

    def _f0_gestures_s2g(self, seg_data=None):
        if seg_data is None:
            seg_data = self.segments

        f0_gestures = []
        current_time = 0.0

        for seg in seg_data:
            if "f0" not in seg:
                continue
            duration = float(seg.get("duration_s", 0.0))
            f0_value = float(seg.get("f0", 0.0))
            f0_value_st = np.log2(f0_value) * 12  # st単位に変換
            gesture = {
                "value": f0_value_st,
                "slope": 0.0,
                "duration_s": duration,
                "time_constant_s": 0.010,
                "neutral": False,
            }
            f0_gestures.append(gesture)
            current_time += duration

        return f0_gestures

    def seg_to_ges(self, ges_file: str, speaker_file: str = None):
        f0_gestures = self._f0_gestures_s2g()
        tmp_segments = self._preprocess_s2g()
        seg_file = os.path.splitext(ges_file)[0] + ".seg"
        self.save_seg(seg_file, data=tmp_segments)

        try:
            with VocalTractLab() as vtl:
                vtl.seg_to_ges(seg_file, ges_file)
            print(f"Successfully converted to {ges_file}")

            self.load_ges(ges_file)
            ges_data = self._postprocess_s2g()
            ges_data["f0-gestures"]["gestures"] = f0_gestures

            self.save_ges(ges_file, data=ges_data)
        except Exception as e:
            print(f"Error converting seg to ges: {e}")
            import traceback

            traceback.print_exc()

    def ling_to_ges(self, ling_file: str, ges_file: str, speaker_file: str = None):
        """
        .lingファイルを読み込み、.segを経由して.gesファイルを作成する
        """
        self.load_ling(ling_file)
        self.ling_to_seg()
        self.seg_to_ges(ges_file, speaker_file)

    def ges_to_wav(self, ges_file: str, wav_file: str, speaker_file: str):
        """
        .gesファイルを読み込み、.tractを経由して.wavファイルを作成する
        """
        # 1. .tractファイルのパスを決定
        tract_file = os.path.splitext(wav_file)[0] + ".tract"

        # 2. VTL APIを使って変換
        try:
            with VocalTractLab(speaker_file) as vtl:
                # .ges -> .tract
                vtl.ges_to_tract(ges_file, tract_file)
                print(f"Successfully converted to {tract_file}")

                # .tract -> .wav
                vtl.tract_to_audio(tract_file, wav_file)
                print(f"Successfully converted to {wav_file}")

        except Exception as e:
            print(f"Error converting ges to wav: {e}")
            import traceback

            traceback.print_exc()

    def ling_to_wav(self, ling_file: str, wav_file: str, speaker_file: str):
        """
        .lingファイルを読み込み、.seg, .ges, .tractを経由して.wavファイルを作成する
        """
        # 1. .gesファイルのパスを決定
        ges_file = os.path.splitext(wav_file)[0] + ".ges"

        # 2. .ling -> .ges
        self.ling_to_ges(ling_file, ges_file, speaker_file)

        # 3. .ges -> .wav
        self.ges_to_wav(ges_file, wav_file, speaker_file)


class JPParser:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.kana_map = self._load_json(os.path.join(self.base_dir, "kana_map.json"))
        self.phoneme_durations = self._load_json(
            os.path.join(self.base_dir, "phoneme_durations.json")
        )

    def _load_json(self, path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return {}

    def parse_to_segments(self, text, bpm, f0):
        """
        Parse Japanese text into segments with durations.
        """
        moras = self._parse_text_to_moras(text)
        segments = self._generate_segments(moras, bpm, f0)
        return segments

    def _parse_text_to_moras(self, text):
        """
        Parse Japanese text into a list of moras (list of phonemes).
        """
        # Remove @devoc{...} wrapper
        text = re.sub(r"@devoc\{([^}]+)\}", r"\1", text)

        # Remove whitespace
        text = text.replace(" ", "").replace("　", "")

        moras = []
        i = 0
        while i < len(text):
            char = text[i]

            if char == "'":
                if moras and not moras[-1].get("pause"):
                    moras[-1]["accent"] = True
                i += 1
                continue

            if char == ",":
                if moras and not moras[-1].get("pause"):
                    moras[-1]["word_break"] = True
                i += 1
                continue

            if char == "/":
                if moras and not moras[-1].get("pause"):
                    moras[-1]["phrase_break"] = True

                moras.append({"phonemes": ["PAUSE"], "pause": True, "accent": False})
                i += 1
                continue

            if i + 1 < len(text):
                sub = text[i : i + 2]
                if sub in self.kana_map:
                    moras.append(
                        {
                            "phonemes": self.kana_map[sub],
                            "pause": False,
                            "accent": False,
                            "word_break": False,
                            "phrase_break": False,
                        }
                    )
                    i += 2
                    continue

            # Check 1 char
            sub = text[i]
            if sub in self.kana_map:
                moras.append(
                    {
                        "phonemes": self.kana_map[sub],
                        "pause": False,
                        "accent": False,
                        "word_break": False,
                        "phrase_break": False,
                    }
                )
                i += 1
                continue
            print(f"Warning: Unknown character '{sub}' at index {i}")
            i += 1

        return moras

    def _generate_segments(self, moras, bpm, f0):
        """
        Convert moras to segments with durations.
        """
        duration_per_mora = 60.0 / bpm / 4  # mora1個は4分音符分とする
        segments = []

        # イントネーションパラメータ
        # フレーズごとに最高音(High)と最低音(Low)を一定にする
        # 基準f0に対して、Highは +6 semitones, Lowは 0 semitones (基準値) とする例
        high_st = 6.0
        low_st = 0.0

        f0_high = f0 * np.power(2, high_st / 12)
        f0_low = f0 * np.power(2, low_st / 12)

        # モーラをフレーズごとに分割する
        phrases = []
        current_phrase = []
        for mora in moras:
            current_phrase.append(mora)
            # フレーズ切れ目判定: PAUSE または phrase_breakフラグ
            if mora.get("pause") or mora.get("phrase_break"):
                phrases.append(current_phrase)
                current_phrase = []

        if current_phrase:
            phrases.append(current_phrase)

        # フレーズごとに処理
        for i, phrase in enumerate(phrases):
            # このフレーズ内のモーラ（PAUSE除く）を抽出
            content_moras = [m for m in phrase if not m.get("pause")]

            # アクセント位置の特定
            # accent=True となっているモーラがアクセント核（下がるところの直前）
            accent_index = -1
            for j, m in enumerate(content_moras):
                if m.get("accent"):
                    accent_index = j
                    break

            # High/Low パターンの生成
            # 0: Low, 1: High
            hl_pattern = [0] * len(content_moras)

            if content_moras:
                if accent_index == 0:
                    # 頭高型: H L L ...
                    hl_pattern[0] = 1
                    # 残りは0 (Low)
                else:
                    # 平板 or 中高 or 尾高
                    # 1モーラ目はLow, 2モーラ目からアクセント核までHigh
                    hl_pattern[0] = 0

                    # 2モーラ目以降の処理
                    limit = (
                        accent_index if accent_index != -1 else len(content_moras) - 1
                    )

                    for j in range(1, len(content_moras)):
                        if j <= limit:
                            hl_pattern[j] = 1
                        else:
                            hl_pattern[j] = 0

            # 次のフレーズの先頭F0を決定する（ポーズ用）
            next_phrase_start_f0 = f0_low
            if i + 1 < len(phrases):
                next_phrase = phrases[i + 1]
                next_content_moras = [m for m in next_phrase if not m.get("pause")]
                if next_content_moras:
                    next_accent_index = -1
                    for j, m in enumerate(next_content_moras):
                        if m.get("accent"):
                            next_accent_index = j
                            break
                    # 頭高型ならHighスタート、それ以外はLowスタート
                    if next_accent_index == 0:
                        next_phrase_start_f0 = f0_high
                    else:
                        next_phrase_start_f0 = f0_low

            # 各モーラへのF0割り当てとセグメント生成
            content_mora_idx = 0
            is_next_mora_start_of_word = True

            for mora in phrase:
                phonemes = mora["phonemes"]

                if mora.get("pause"):
                    # Pause
                    segments.append(
                        {
                            "name": "?",
                            "duration_s": duration_per_mora,
                            "f0": next_phrase_start_f0,
                        }
                    )
                    is_next_mora_start_of_word = True
                    continue

                # Calculate duration for each phoneme in the mora
                duration_phs = [None] * len(phonemes)

                # Assign fixed durations for consonants
                for i, ph in enumerate(phonemes):
                    if ph in self.phoneme_durations and len(phonemes) > 1:
                        duration_phs[i] = self.phoneme_durations.get(ph, 0.060)

                # Calculate remaining time for vowels
                fixed_duration = sum([d for d in duration_phs if d is not None])
                remaining_duration = duration_per_mora - fixed_duration

                if remaining_duration < 0:
                    remaining_duration = 0.010  # Minimum duration

                # Distribute remaining duration among undefined phonemes (usually vowels)
                undefined_count = duration_phs.count(None)
                if undefined_count > 0:
                    val = remaining_duration / undefined_count
                    for i in range(len(duration_phs)):
                        if duration_phs[i] is None:
                            duration_phs[i] = val

                # Determine F0 for this mora
                is_high = hl_pattern[content_mora_idx] == 1
                target_f0 = f0_high if is_high else f0_low
                content_mora_idx += 1

                # Create segments
                for i, (ph, dur) in enumerate(zip(phonemes, duration_phs)):
                    seg = {"name": ph, "duration_s": dur, "f0": target_f0}

                    if i == 0:
                        seg["start_of_syllable"] = 1
                        if is_next_mora_start_of_word:
                            seg["start_of_word"] = 1
                            is_next_mora_start_of_word = False

                    # 必要な属性をコピー
                    for key in mora.keys():
                        if key not in ["phonemes", "pause"]:
                            seg[key] = mora[key]
                    segments.append(seg)

                if mora.get("word_break"):
                    is_next_mora_start_of_word = True

        # Add final silence
        segments.append({"name": "?", "duration_s": 0.5, "f0": f0_low})

        return segments


if __name__ == "__main__":
    # テストコード
    utterance = VTLUtterance()
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ling_path = os.path.join(base_dir, "examples", "konnichiwa.ling")
    wav_out_path = os.path.join(base_dir, "examples", "konnichiwa_out.wav")

    # スピーカーファイルのパス (環境に合わせて調整してください)
    speaker_path = os.path.join(
        base_dir, "vtl_gui", "speakers", "降久嘉音_alpha.speaker"
    )
    if not os.path.exists(speaker_path):
        speaker_path = os.path.join(base_dir, "vtl_gui", "speakers", "JD2.speaker")

    print(f"Converting {ling_path} to {wav_out_path} using {speaker_path}")
    utterance.ling_to_wav(ling_path, wav_out_path, speaker_path)
