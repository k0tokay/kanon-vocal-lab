import os
import json
import re
import numpy as np


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

    def _generate_segments(
        self, moras, bpm, f0, pause_at_beginning=True, pause_at_end=True
    ):
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

        if pause_at_beginning and phrases:
            segments.append(
                {
                    "name": "?",
                    "duration_s": 0.5,
                    "f0": f0,
                }
            )

        # フレーズごとに処理
        for i, phrase in enumerate(phrases):
            # このフレーズ内のモーラ（PAUSE除く）を抽出
            content_moras = [m for m in phrase if not m.get("pause")]

            # フレーズを単語に分割
            accent_phrases = []
            current_accent_phrase = []
            for m in content_moras:
                current_accent_phrase.append(m)
                if m.get("word_break"):
                    accent_phrases.append(current_accent_phrase)
                    current_accent_phrase = []
            if current_accent_phrase:
                accent_phrases.append(current_accent_phrase)

            # 各アクセント句(?)ごとにHLパターンを生成して結合
            hl_pattern = []
            for word_moras in accent_phrases:
                # アクセント位置特定
                accent_idx = -1
                for j, m in enumerate(word_moras):
                    if m.get("accent"):
                        accent_idx = j
                        break

                word_hl = [0] * len(word_moras)
                if word_moras:
                    if accent_idx == 0:  # 頭高型: H L L ...
                        word_hl[0] = 1
                        # 残りは0
                    else:  # 平板・中高・尾高: L H H ... (L)
                        word_hl[0] = 0
                        limit = accent_idx if accent_idx != -1 else len(word_moras) - 1
                        for k in range(1, len(word_moras)):
                            if k <= limit:
                                word_hl[k] = 1
                            else:
                                word_hl[k] = 0
                hl_pattern.extend(word_hl)

            # 次のフレーズの先頭F0を決定する（ポーズ用）
            next_phrase_start_f0 = f0
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
                        next_phrase_start_f0 = f0 * np.power(2, 6.0 / 12)
                    else:
                        next_phrase_start_f0 = f0

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
                # フレーズ全体の下降 (Declination): 1モーラあたり -0.7 semitones
                declination_st = -0.7 * content_mora_idx
                baseline_f0 = f0 * np.power(2, declination_st / 12)

                # アクセントの重畳: Highなら +6.0 semitones
                is_high = hl_pattern[content_mora_idx] == 1
                accent_lift_st = 6.0 if is_high else 0.0

                target_f0 = baseline_f0 * np.power(2, accent_lift_st / 12)
                content_mora_idx += 1

                # 母音のみの音節が単語頭に来る場合、短いポーズを挿入
                if (
                    is_next_mora_start_of_word
                    and len(phonemes) == 1
                    and phonemes[0] in ["a", "i", "M", "e", "o"]
                ):
                    segments.append(
                        {
                            "name": "?",
                            "duration_s": 0.060,
                            "f0": target_f0,
                        }
                    )
                    # 短いポーズ分だけ母音の長さを短くする
                    duration_phs = [
                        d - 0.060 if d is not None else None for d in duration_phs
                    ]

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
        if pause_at_end:
            segments.append(
                {
                    "name": "?",
                    "duration_s": 0.5,
                    "f0": f0,
                }
            )

        return segments
