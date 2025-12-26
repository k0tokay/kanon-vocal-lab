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
                    "j",
                    "x",
                    "r",
                    "R",
                    "h",
                },
                "approximants": {
                    "j",
                    "w",
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
        }
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
                item["name"]: item["id"] for item in self.flattened_tree
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

    def expand(self, name: str) -> list:
        """ある概念のインスタンスをすべて取得する"""
        start_id = self.n2i(name)
        results = set()

        def _rec(current_id: int):
            if self.flattened_tree[current_id]["is_instance"]:
                results.add(self.flattened_tree[current_id]["name"])
            for child_id in self.flattened_tree[current_id]["children"]:
                _rec(child_id)

        _rec(start_id)
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

                for prod in product(*axes):
                    results.update(prod)
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
            root_id = self.n2i(root)
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

    def lift_to_concept(
        self, relation: str, dom: str, cod: str, dom_replace_from=None, direction=0
    ):
        """ある二項関係をis-a関係に同一視する"""
        if direction == 1:
            dom, cod = cod, dom
        # 元の親から外す
        parents_cod = self.flattened_tree[self.n2i(cod)]["parents"]
        if dom_replace_from is not None:
            pass
        elif len(parents_cod) == 0:  # 滅多にない(?)
            pass
        elif len(parents_cod) == 1:
            dom_replace_from = self.i2n(parents_cod[0])
        else:
            raise ValueError(
                f"Cannot determine dom_replace_from for {cod} because it has multiple parents: {[self.i2n(pid) for pid in parents_cod]}"
            )
        self.remove_cp(cod, dom_replace_from)
        # dom-codを親子関係に追加
        self.add_cp(cod, dom)
        # 関係をis-a関係に変える
        relation_set = self.expand(relation)
        for rel in relation_set:
            if direction == 1:
                rel = rel[::-1]
            child_name, parent_name = rel
            self.add_cp(child_name, parent_name)
            self.flattened_tree[self.n2i(parent_name)]["is_instance"] = False
        self.remove_recursively(relation)


ontology = Ontology(PHONEME_ONTOLOGY)

ontology.visualize(root="has_voicing")

ontology.lift_to_concept(
    relation="has_voicing", dom="phonemes", cod="voicing", direction=0
)
ontology.lift_to_concept(
    relation="has_place_of_articulation",
    dom="phonemes",
    cod="place_of_articulation",
    direction=0,
)
ontology.visualize("ontology_after_lifting")
