from itertools import product


class Phoneme:
    def __init__(self, symbol: str):
        self.symbol = symbol


PHONEME_ONTOLOGY = {
    "concepts": {
        "phonemes": {
            "vowels": {
                "a",
                "o",
                "e",
                "i",
                "y",
                "M",
                "u",
            },
            "consonants": {
                "fricatives": {
                    "φ",
                    "β",
                    "f",
                    "v",
                    "T",
                    "D",
                    "s",
                    "z",
                    "S",
                    "Z",
                    "C",
                    "x",
                    "r",
                    "R",
                    "h",
                },
                "approximants": {
                    "j",
                    "w",
                },
                "flaps": {
                    "ɾ",
                },
                "plosives": {
                    "?",
                    "p",
                    "b",
                    "t",
                    "d",
                    "k",
                    "g",
                },
                "nasals": {
                    "m",
                    "n",
                    "N",
                },
                "affricates": {
                    "tS",
                    "dZ",
                    "ts",
                    "dz",
                },
                "laterals": {
                    "l",
                },
            },
        },
        "vtl_objects": {
            "gesture_tiers": {
                "vowel-gestures",
                "lip-gestures",
                "tongue-tip-gestures",
                "tongue-body-gestures",
                "velic-gestures",
                "glottal-shape-gestures",
                "f0-gestures",
                "lung-pressure-gestures",
            },
            "glottal_shapes": {
                "modal",
                "stop",
                "voiced-fricative",
                "voiceless-fricative",
                "voiced-plosive",
                "voiceless-plosive",
                "h",
            },
        },
        "features": {
            "voicing": {
                "voiced",
                "voiceless",
            },
            "place_of_articulation": {
                "labial",
                "labiodental",
                "dental",
                "alveolar",
                "postalveolar",
                "palatal",
                "velar",
                "glottal",
            },
        },
    },
    "relations": {
        "has_feature": {
            "has_voicing": {
                *[
                    (phoneme, "voiceless")
                    for phoneme in {
                        "φ",
                        "f",
                        "T",
                        "s",
                        "S",
                        "C",
                        "x",
                        "h",
                        "?",
                        "t",
                        "k",
                        "ts",
                        "tS",
                    }
                ],
                *[
                    (phoneme, "voiced")
                    for phoneme in {
                        "β",
                        "v",
                        "D",
                        "z",
                        "Z",
                        "j",
                        "r",
                        "R",
                        "b",
                        "d",
                        "g",
                        "dz",
                        "dZ",
                    }
                ],
            },
            "has_place_of_articulation": {
                *[
                    (phoneme, "labial")
                    for phoneme in {"p", "b", "m", "φ", "f", "β", "v", "w"}
                ],
                *[(phoneme, "labiodental") for phoneme in {"f", "v"}],
                *[(phoneme, "dental") for phoneme in {"T", "D"}],
                *[
                    (phoneme, "alveolar")
                    for phoneme in {
                        "t",
                        "d",
                        "ts",
                        "dz",
                        "n",
                        "s",
                        "z",
                        "l",
                        "r",
                    }
                ],
                *[(phoneme, "dental") for phoneme in {"T", "D"}],
                *[(phoneme, "postalveolar") for phoneme in {"S", "Z", "tS", "dZ"}],
                *[(phoneme, "palatal") for phoneme in {"j", "C"}],
                *[(phoneme, "velar") for phoneme in {"k", "g", "N", "x"}],
                *[(phoneme, "glottal") for phoneme in {"h", "?"}],
            },
        },
        "has_gesture_tiers": {
            # 調音点とジェスチャーティアの対応
            ("vowels", "vowel-gestures"),
            ("labial@", "lip-gestures"),
            ("labiodental@", "lip-gestures"),
            ("alveolar@", "tongue-tip-gestures"),
            ("dental@", "tongue-tip-gestures"),
            ("postalveolar@", "tongue-tip-gestures"),
            ("palatal@", "tongue-body-gestures"),
            ("velar@", "tongue-body-gestures"),
            ("glottal@", "glottal-shape-gestures"),
        },
        "has_gesture_value": {
            ("a", "[str]a"),
            ("o", "[str]o"),
            ("e", "[str]e"),
            ("i", "[str]i"),
            ("y", "[str]y"),
            ("M", "[str]M"),
            ("u", "[str]u"),
            *[(phoneme, "[str]ll-labial-closure") for phoneme in {"p", "b", "m"}],
            *[(phoneme, "[str]ll-labial-approx") for phoneme in {"w"}],
            *[(phoneme, "[str]ll-labial-fricative") for phoneme in {"φ", "β"}],
            *[(phoneme, "[str]ll-dental-fricative") for phoneme in {"f", "v"}],
            *[(phoneme, "[str]tt-dental-fricative") for phoneme in {"T", "D"}],
            *[(phoneme, "[str]tt-alveolar-closure") for phoneme in {"t", "d", "n"}],
            *[(phoneme, "[str]tt-alveolar-fricative") for phoneme in {"s", "z"}],
            *[(phoneme, "[str]tt-postalveolar-fricative") for phoneme in {"S", "Z"}],
            *[(phoneme, "[str]tb-velar-closure") for phoneme in {"k", "g", "N"}],
        },
        "has_glottal_shape": {
            ("voiced", "voiced-fricative"),
            ("voiced", "voiced-plosive"),
            ("voiceless", "voiceless-fricative"),
            ("voiceless", "voiceless-plosive"),
            ("plosives", "voiced-plosive"),
            ("plosives", "voiceless-plosive"),
            ("fricatives", "voiced-fricative"),
            ("fricatives", "voiceless-fricative"),
            ("h", "h"),  # 例外処理ができないからダメそう
            ("?", "stop"),
        },
    },
}


class Ontology:
    def __init__(self, tree: dict = None, flattened_tree: list = None):
        self.tree = {}
        self.flattened_tree = []
        if tree is not None:
            self.load_tree(tree)
            self._flatten_tree(self.tree)
        elif flattened_tree is not None:
            self.flattened_tree = flattened_tree
        self.reversed_name_map = None

    def load_tree(self, tree: dict):
        self.tree = tree
        self._flatten_tree(self.tree)

    def _flatten_tree(self, subtree: dict):
        results = []

        def _rec(pid, current_tree: dict, is_instance=False):
            for key, value in current_tree.items():
                id = len(results)
                elem = {
                    "id": id,
                    "name": key,
                    "parents": [],
                    "children": [],
                    "is_instance": is_instance,
                }
                results.append(elem)

                if pid is not None:
                    results[id]["parents"].append(pid)
                    results[pid]["children"].append(id)

                if isinstance(value, dict):
                    _rec(id, value)
                elif isinstance(value, set):
                    for v in value:
                        _rec(id, {v: {}}, is_instance=True)

        _rec(None, subtree)

        self.flattened_tree = results

    def i2n(self, id: int) -> str:
        return self.flattened_tree[id]["name"]

    def n2i(self, name: str) -> int:
        if self.reversed_name_map is None:
            self.reversed_name_map = {
                item["name"]: item["id"]
                for item in self.flattened_tree
                if item is not None
            }
        return self.reversed_name_map[name]

    def _recursive_visitor(self, func):
        """デコレータ: 関数を子要素へ再帰的に適用するようにラップする"""

        def wrapper(current_id: int):
            # まず現在のノードに処理を適用
            func(current_id)

            # ノードが削除されていなければ、子要素へ再帰
            node = self.flattened_tree[current_id]
            if node is not None:
                # 処理中にchildrenが変更される可能性を考慮してコピーで回す
                for child_id in list(node["children"]):
                    wrapper(child_id)

        return wrapper

    def use_denotation(self, name: str) -> bool:
        """名前が値を指示しているかどうかを判定する"""
        if isinstance(name, tuple):
            return False
        denotations = {"[str]", "[int]", "[float]"}
        for key in denotations:
            if name.startswith(key):
                return True
        return False

    def expand(self, name: str) -> list:
        """ある概念のインスタンスをすべて取得する"""
        results = set()

        def _rec(current: str):
            if self.use_denotation(current):
                results.add(current)
                return
            if isinstance(current, tuple):
                results.add(current)
                return
            current_id = self.n2i(current)
            if self.flattened_tree[current_id]["is_instance"]:
                results.add(self.flattened_tree[current_id]["name"])
            for child_id in self.flattened_tree[current_id]["children"]:
                _rec(self.i2n(child_id))

        _rec(name)
        return results

    def expand_rect(self, name: str) -> list:
        """(relationsのみ) 非インスタンスのタプルも分解する"""
        epanded = self.expand(name)
        results = set()
        for item in epanded:
            if isinstance(item, tuple):
                axes = list(item)
                for i, axis in enumerate(axes):
                    axes[i] = self.expand(axis)
                results.update(set(product(*axes)))
            else:
                results.add(item)
        return results

    def remove_recursively(self, name: str):
        subtree = self.get_subtree(name)
        print("Removing subtree:", subtree)
        for node in subtree:
            if node is None:
                continue
            for pid in node["parents"]:
                if self.flattened_tree[pid] is None:
                    continue
                if node["id"] in self.flattened_tree[pid]["children"]:
                    self.flattened_tree[pid]["children"].remove(node["id"])
        for node in subtree:
            self.flattened_tree[node["id"]] = None

    def get_subtree(self, name: str) -> list:
        """ある概念以下の部分木を取得する"""
        id = self.n2i(name)
        results = []

        @self._recursive_visitor
        def _collect(current_id: int):
            node = self.flattened_tree[current_id]
            if node is not None:
                results.append(node)

        _collect(id)
        return results

    def visualize(self, filename="ontology_graph", stagger=8, root=None):
        """
        Ontologyを可視化して画像を出力する。
        実行には `pip install graphviz` と、OSへのGraphvizのインストール(brew install graphviz等)が必要です。
        """
        try:
            from graphviz import Digraph
        except ImportError:
            print(
                "Error: graphviz library not found. Please install it with `pip install graphviz`."
            )
            return

        dot = Digraph(comment="Phoneme Ontology")
        dot.attr(rankdir="TB")  # Top to Bottom レイアウト

        if root is not None:
            subtree = self.get_subtree(root)
        else:
            subtree = self.flattened_tree

        for node in subtree:
            if node is None:
                continue

            # ノードのスタイル設定
            # インスタンス（具体的な音素）は楕円、概念（カテゴリ）は箱
            shape = "ellipse" if node["is_instance"] else "box"
            style = "filled" if node["is_instance"] else ""
            fillcolor = "lightblue" if node["is_instance"] else "white"

            dot.node(
                str(node["id"]),
                label=str(node["name"]),
                shape=shape,
                style=style,
                fillcolor=fillcolor,
            )

            # エッジの追加（親 -> 子）
            for child_id in node["children"]:
                # 削除されたノードへの参照が残っている場合のガード
                if (
                    child_id < len(self.flattened_tree)
                    and self.flattened_tree[child_id] is not None
                ):
                    dot.edge(str(node["id"]), str(child_id))

        # 出力
        try:
            dot = dot.unflatten(stagger=stagger)
            output_path = dot.render(filename, view=True, format="png", cleanup=True)
            print(f"Graph rendered to {output_path}")
        except Exception as e:
            print(f"Failed to render graph: {e}")
            print(
                "Make sure Graphviz is installed on your system (e.g., `brew install graphviz` on macOS)."
            )

    def remove_cp(self, child_name: str, parent_name: str):
        """child-parent関係を削除する"""
        cid = self.n2i(child_name)
        pid = self.n2i(parent_name)
        if pid in self.flattened_tree[cid]["parents"]:
            self.flattened_tree[cid]["parents"].remove(pid)
        if cid in self.flattened_tree[pid]["children"]:
            self.flattened_tree[pid]["children"].remove(cid)

    def add_cp(self, child_name: str, parent_name: str):
        """child-parent関係を追加する"""
        cid = self.n2i(child_name)
        pid = self.n2i(parent_name)
        if pid not in self.flattened_tree[cid]["parents"]:
            self.flattened_tree[cid]["parents"].append(pid)
        if cid not in self.flattened_tree[pid]["children"]:
            self.flattened_tree[pid]["children"].append(cid)

    def add_tree(self, subtree, shift_ids=True, name_prefix="", name_suffix="@"):
        import copy

        # subtreeは参照渡しされるので、ディープコピーして元のオブジェクトを破壊しないようにする
        subtree = copy.deepcopy(subtree)

        delta_id = len(self.flattened_tree) if shift_ids else 0

        # IDのマッピングを作成 (old_id -> new_id)
        # もとのIDは使わない
        id_map = {}
        for i, node in enumerate(subtree):
            if node is not None:
                id_map[node["id"]] = i + delta_id

        for i, node in enumerate(subtree):
            if node is None:
                continue

            # IDと名前の更新
            old_id = node["id"]
            new_id = id_map[old_id]
            node["id"] = new_id
            node["name"] = f"{name_prefix}{node['name']}{name_suffix}"

            # 親子関係のIDを更新
            # parents/childrenリスト内のIDを新しいIDに書き換える
            # ただし、subtree内に含まれないID（外部への参照）はそのままにするか、削除するか？
            # ここでは「subtree内に含まれるものは書き換え、そうでないものはそのまま」とする

            new_parents = []
            for pid in node["parents"]:
                if pid in id_map:
                    new_parents.append(id_map[pid])
                else:
                    # 部分木の外への参照は、コピー時には切断するのが一般的だが、
                    # 文脈によっては維持したい場合もある。
                    # ここでは「新しいツリーとして独立させる」ために外への参照は削除する方針をとる
                    # もし接続したいなら後で add_cp する
                    pass
            node["parents"] = new_parents

            new_children = []
            for cid in node["children"]:
                if cid in id_map:
                    new_children.append(id_map[cid])
                else:
                    pass
            node["children"] = new_children

            # flattened_treeを拡張して格納
            # リストのインデックス = ID となるように調整が必要
            while len(self.flattened_tree) <= new_id:
                self.flattened_tree.append(None)
            self.flattened_tree[new_id] = node

        # reversed_name_mapをリセット
        self.reversed_name_map = None

    def lift_to_concept(
        self,
        relation: str,
        dom: str,
        cod: str,
        direction=0,
    ):
        """ある二項関係をis-a関係に同一視する．もとの関係は削除するかどうかで処理が変わる．"""
        if direction == 1:
            dom, cod = cod, dom

        subtree = self.get_subtree(cod)
        suffix = "@"
        self.add_tree(subtree, name_suffix=suffix)

        cod_new_name = cod + suffix
        cod_new_id = self.n2i(cod_new_name)
        cod_new_node = self.flattened_tree[cod_new_id]

        for child_id in cod_new_node["children"]:
            child_name = self.i2n(child_id)
            self.add_cp(child_name, dom)

        self.flattened_tree[cod_new_id] = None

        # 3. 元の relation を見て、新しい接続を作る
        relation_set = self.expand(relation)
        for rel in relation_set:
            if direction == 1:
                rel = rel[::-1]
            child_name, parent_name = rel

            parent_name_new = parent_name + suffix

            self.add_cp(child_name, parent_name_new)

            self.flattened_tree[self.n2i(parent_name_new)]["is_instance"] = False

    def to_json(self, filename="ontology.json"):
        """flattened_treeをJSONファイルに書き出す"""
        import json

        # Noneを除去してIDを振り直す必要があるが、
        # ここでは単純にNone以外のリストとして書き出す
        # (IDの整合性を保つなら再構築が必要だが、簡易的なダンプとする)
        valid_nodes = [node for node in self.flattened_tree if node is not None]

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(valid_nodes, f, indent=2, ensure_ascii=False)
        print(f"Ontology saved to {filename}")

    def get_denotation(self, name):
        """(分類木以外の値を指示する場合に) 名前から値を取得する"""
        denotations = {"[str]": str, "[int]": int, "[float]": float}
        for key in denotations.keys():
            if name.startswith(key):
                value = name[len(key) :]
                return denotations[key](value)
        return name

    def is_functional(self, relation: str, index=1) -> bool:
        """n項関係が第index項目に関して関数的か(domainは指定しないので一意性のみ)どうかを判定する"""
        relation_set = self.expand(relation)
        cod_values = set()
        for rel in relation_set:
            cod = rel[index]
            if cod in cod_values:
                return False
            cod_values.add(cod)
        return True

    # これn項関係に一般化できるので将来的に拡張する
    def iota(self, relation: str, dom_value: str, index=0):
        """n項関係において、第index項目がdom_valueであるタプルのうち一意に定まるcodを取得する"""
        relation_set = self.expand_rect(relation)
        results = []
        for rel in relation_set:
            dom = rel[index]
            if dom == dom_value:
                results.append(rel)
        if len(results) == 0:
            # raise ValueError(
            #     f"No matching tuple found in relation '{relation}' for dom_value '{dom_value}' at index {index}."
            # )
            return None
        if len(results) > 1:
            raise ValueError(
                f"Multiple matching tuples found in relation '{relation}' for dom_value '{dom_value}': {results}."
            )
        return results[0]


class PhonemeOntology(Ontology):
    def __init__(self, tree: dict = None, flattened_tree: list = None):
        super().__init__(tree, flattened_tree)

        # lift relations to is-a
        self.suffix = "@"
        self.lift_to_concept(
            relation="has_voicing", dom="phonemes", cod="voicing", direction=0
        )
        self.lift_to_concept(
            relation="has_place_of_articulation",
            dom="phonemes",
            cod="place_of_articulation",
            direction=0,
        )
        self.phoneme_dict = self.calc_phoneme_dict()

    def calc_phoneme_dict(self):
        phoneme_dict = {}

        # "あるクラスに入っているかどうか"で判断する属性の集まり
        attr_config = [
            {"name": "type", "classes": ["vowels", "consonants"], "suffix": ""},
            {
                "name": "manner_of_articulation",
                "classes": [
                    "plosives",
                    "fricatives",
                    "affricates",
                    "nasals",
                    "approximants",
                    "flaps",
                    "laterals",
                ],
                "suffix": "",
            },
            {
                "name": "place_of_articulation",
                "classes": self.expand("place_of_articulation"),
                "suffix": self.suffix,
            },
            {
                "name": "voicing",
                "classes": self.expand("voicing"),
                "suffix": self.suffix,
            },
        ]

        # (部分)写像
        map_config = [
            {
                "name": "has_gesture_tiers",
            },
            {
                "name": "has_gesture_value",
            },
        ]

        phonemes = self.expand("phonemes")
        for name in phonemes:
            phoneme_dict[name] = {"symbol": name}
            node = self.flattened_tree[self.n2i(name)]
            for attr in attr_config:
                for class_name in attr["classes"]:
                    class_id = self.n2i(class_name + attr["suffix"])
                    if class_id in node["parents"]:
                        phoneme_dict[name][attr["name"]] = class_name
            for map_item in map_config:
                value = self.iota(
                    relation=map_item["name"],
                    dom_value=name,
                    index=0,
                )
                if value is not None:
                    _, value = value
                    value = self.get_denotation(value)
                    phoneme_dict[name][map_item["name"]] = value

        return phoneme_dict

    def export_phoneme_dict(self, filename="phoneme_properties.json"):
        import json

        phoneme_dict = self.calc_phoneme_dict()
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(phoneme_dict, f, indent=2, ensure_ascii=False)
        print(f"Phoneme properties saved to {filename}")

    def _get_ancestor_names(self, node_id):
        """指定されたノードのすべての祖先の名前を取得する"""
        ancestors = set()

        def _rec(current_id):
            node = self.flattened_tree[current_id]
            if node is None:
                return

            # 自分自身は属性に含めない（必要なら含める）
            if current_id != node_id:
                ancestors.add(node["name"])

            for parent_id in node["parents"]:
                _rec(parent_id)

        _rec(node_id)
        return ancestors


ontology = PhonemeOntology(PHONEME_ONTOLOGY)

ontology.to_json("phoneme_ontology.json")
ontology.export_phoneme_dict("phoneme_properties.json")
