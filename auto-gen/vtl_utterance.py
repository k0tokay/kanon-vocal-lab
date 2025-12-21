import xml.etree.ElementTree as ET
from xml.dom import minidom


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


class JPParser:
    def __init__(self, data):
        self.data = data
        self.parse(data)

    def parse(self, data):
        context = data.get("context", {})
        self.lang = context.get("lang", "unknown")
        self.type = context.get("type", "unknown")

        if self.lang != "jp":
            raise ValueError(f"Unsupported language for JPParser: {self.lang}")

        content = data.get("content", "")
        gesture_tiers = VTLUtterance.gesture_tiers_template()

        return gesture_tiers


if __name__ == "__main__":
    # テストコード
    utterance = VTLUtterance()
    ling = utterance.load_ling("../examples/konnichiwa.ling")
    print(utterance.linguistic_blocks)
    utterance.save_ling("../examples/konnichiwa_out.ling")
    ges = utterance.load_ges("../examples/konnichiwa.ges")
    print(utterance.gesture_tiers)
    utterance.save_ges("../examples/konnichiwa_out.ges")
