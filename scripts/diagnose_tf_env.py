#!/usr/bin/env python3
"""
Quick self-check for the most common onboarding issue:
EfficientNet ImageNet weight load failures caused by TF/Keras mismatch or stale caches.

Run:
  python scripts/diagnose_tf_env.py
"""

import os
import platform
import sys
from pathlib import Path


def _try_import_tf():
    try:
        import tensorflow as tf  # type: ignore

        return tf, None
    except Exception as e:  # pragma: no cover
        return None, e


def main() -> None:
    print("=== Python / Platform ===")
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("platform:", platform.platform())
    print("machine:", platform.machine())
    print("KERAS_HOME:", os.getenv("KERAS_HOME", "<unset>"))

    tf, err = _try_import_tf()
    if tf is None:
        print("\n=== TensorFlow import FAILED ===")
        print(repr(err))
        print("\nFix:")
        print("- Install the correct TensorFlow for your OS (see README)")
        return

    print("\n=== TensorFlow / Keras ===")
    print("tensorflow:", getattr(tf, "__version__", "<unknown>"))
    try:
        print("keras:", tf.keras.__version__)  # TF 2.16+ exposes this
    except Exception:
        pass

    print("\n=== EfficientNetB0 ImageNet weights check ===")
    try:
        from tensorflow.keras import applications  # type: ignore

        m = applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_shape=(224, 224, 3),
        )
        stem = m.get_layer("stem_conv")
        print("OK: loaded weights='imagenet'")
        print("stem_conv kernel shape:", tuple(stem.weights[0].shape))
    except Exception as e:
        print("FAILED: could not load weights='imagenet'")
        print(repr(e))

        home = Path(os.getenv("KERAS_HOME") or (Path.home() / ".keras"))
        print("\nLikely causes:")
        print("- Wrong TensorFlow build for your machine (Apple Silicon vs x86_64)")
        print("- Stale cached EfficientNet weights from a different environment")
        print("\nFast fixes:")
        print(f"- (Recommended) Set per-project cache: export KERAS_HOME='{(Path.cwd() / '.keras').resolve()}'")
        print(f"- Clear cached weights under: {home / 'models'}")
        print("- Then retry: python scripts/diagnose_tf_env.py")
        print("\nYou can also run: python scripts/clear_keras_cache.py")


if __name__ == "__main__":
    main()

