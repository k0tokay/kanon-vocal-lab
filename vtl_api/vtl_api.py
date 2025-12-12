import ctypes as C
import os
import sys
from pathlib import Path


class VocalTractLab:
    def __init__(self, speaker_file: str, lib_path: str = None):
        self.lib = self._load_library(lib_path)
        self.speaker_file = speaker_file
        self._setup_signatures()

        # 内部状態
        self.audio_sr = C.c_int(0)
        self.num_tube = C.c_int(0)
        self.num_vtp = C.c_int(0)
        self.num_glot = C.c_int(0)

    def _load_library(self, lib_path):
        """OSに合わせてライブラリを読み込む"""
        if lib_path:
            return C.cdll.LoadLibrary(lib_path)

        here = os.path.abspath(os.path.dirname(__file__))
        if sys.platform == "win32":
            names = ["VocalTractLabApi.dll"]
        elif sys.platform == "darwin":
            names = ["VocalTractLabApi.dylib"]
        else:
            names = ["VocalTractLabApi.so"]

        candidates = [os.path.join(here, n) for n in names] + names

        for path in candidates:
            try:
                return C.cdll.LoadLibrary(path)
            except OSError:
                continue
        raise OSError("VocalTractLabApi load failed.")

    def _setup_signatures(self):
        """関数の型定義を一括で行う"""
        self.lib.vtlInitialize.argtypes = [C.c_char_p]
        self.lib.vtlInitialize.restype = C.c_int

        self.lib.vtlGetConstants.argtypes = [C.POINTER(C.c_int)] * 4
        self.lib.vtlSegmentSequenceToGesturalScore.argtypes = [C.c_char_p, C.c_char_p]
        self.lib.vtlSegmentSequenceToGesturalScore.restype = C.c_int

        self.lib.vtlGesturalScoreToTractSequence.argtypes = [C.c_char_p, C.c_char_p]
        self.lib.vtlGesturalScoreToTractSequence.restype = C.c_int

        self.lib.vtlTractSequenceToAudio.argtypes = [
            C.c_char_p,
            C.c_char_p,
            C.POINTER(C.c_double),
            C.POINTER(C.c_int),
        ]
        self.lib.vtlTractSequenceToAudio.restype = C.c_int

        self.lib.vtlClose.argtypes = []
        self.lib.vtlClose.restype = None

    def __enter__(self):
        """with構文の開始時に初期化"""
        if not os.path.exists(self.speaker_file):
            raise FileNotFoundError(f"Speaker file not found: {self.speaker_file}")

        path_str = str(self.speaker_file)

        # Windowsの場合、DLL(char*)はシステムロケール(Shift-JISなど)を期待する
        if sys.platform == "win32":
            # "mbcs" はWindows専用。現在のシステムロケール(CP932)に変換する
            path_bytes = path_str.encode("mbcs")
        else:
            # macOS / Linux は通常 UTF-8
            path_bytes = path_str.encode("utf-8")

        ret = self.lib.vtlInitialize(path_bytes)
        if ret != 0:
            raise RuntimeError(f"vtlInitialize failed: {ret}")

        self.lib.vtlGetConstants(
            C.byref(self.audio_sr),
            C.byref(self.num_tube),
            C.byref(self.num_vtp),
            C.byref(self.num_glot),
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """with構文の終了時にクローズ"""
        self.lib.vtlClose()


if __name__ == "__main__":
    speaker = (
        Path(__file__).parent.parent / "vtl" / "speakers" / "降久嘉音_alpha.speaker"
    )
    print(f"Using speaker file: {speaker}")
    with VocalTractLab(speaker) as vtl:
        print(f"Audio Sample Rate: {vtl.audio_sr.value}")
        print(f"Number of Tubes: {vtl.num_tube.value}")
        print(f"Number of VTPs: {vtl.num_vtp.value}")
        print(f"Number of Glottal Models: {vtl.num_glot.value}")
