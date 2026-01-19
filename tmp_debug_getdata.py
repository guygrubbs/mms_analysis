import numpy as np

import mms_mp as mp

try:
    from pyspedas import get_data
except Exception as e:  # pragma: no cover
    print("pyspedas import failed:", e)
    raise


TRANGE = ("2019-01-27/12:15:00", "2019-01-27/12:55:00")


def main():
    info = mp.data_loader.force_load_all_plasma_spectrometers(
        list(TRANGE), probes=["1"], rates=["fast", "srvy"], verbose=True
    )
    dis = info["1"]["dis"]
    print("DIS omni_var:", dis.get("omni_var"))
    print("DIS energy_var:", dis.get("energy_var"))

    omni = get_data(dis["omni_var"]) if dis.get("omni_var") else None
    energy = get_data(dis["energy_var"]) if dis.get("energy_var") else None

    def describe(name, obj):
        print(f"{name}: type={type(obj)}")
        if isinstance(obj, tuple):
            print(f"  len={len(obj)}")
            for i, item in enumerate(obj):
                shape = getattr(item, "shape", None)
                print(f"  [{i}] shape={shape} dtype={getattr(item, 'dtype', None)}")
        else:
            shape = getattr(obj, "shape", None)
            print(f"  shape={shape} dtype={getattr(obj, 'dtype', None)}")

    describe("omni", omni)
    describe("energy", energy)


if __name__ == "__main__":
    main()

