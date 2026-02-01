import numpy as np
import rasterio


paths = {
    2016: "data/inputs/deforestation_maps/primary_deforestation_2016.tif",
    2017: "data/inputs/deforestation_maps/primary_deforestation_2017.tif",
    2018: "data/inputs/deforestation_maps/primary_deforestation_2018.tif",
}


years = sorted(paths)

with rasterio.open(paths[2016]) as ref:
    profile = ref.profile.copy()
    profile.update(dtype="int8", count=1, nodata=-1, compress="LZW", BIGTIFF="YES")

    srcs = {y: rasterio.open(paths[y]) for y in years}
    outs = {y: rasterio.open(f"data/accumulated_deforestation/accum_{y}.tif", "w", **profile) for y in years}

    try:
        for _, w in ref.block_windows(1):
            seen1 = np.zeros((w.height, w.width), dtype=bool)
            seen0 = np.zeros((w.height, w.width), dtype=bool)
            seen0_then_neg1 = np.zeros((w.height, w.width), dtype=bool)

            for y in years:
                a = srcs[y].read(1, window=w).astype(np.int8)

                seen1 |= (a == 1)
                # mark transition "have seen 0 before" AND "now see -1"
                seen0_then_neg1 |= seen0 & (a == -1)
                seen0 |= (a == 0)

                out = np.zeros(a.shape, dtype=np.int8)     # default 0
                out[~seen0 & ~seen1] = -1                  # all -1 so far
                out[seen0_then_neg1] = -1                  # 0 followed by -1
                out[seen1] = 1                             # 1 overrides everything

                outs[y].write(out, 1, window=w)
    finally:
        for ds in outs.values():
            ds.close()
        for ds in srcs.values():
            ds.close()