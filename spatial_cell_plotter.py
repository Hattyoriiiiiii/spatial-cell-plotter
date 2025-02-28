import os
from typing import List
from memory_profiler import profile
import psutil
import gc

import pandas as pd
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patch
import seaborn as sns

import json
import zarr
import tifffile

import scanpy as sc


from functools import wraps
import time

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        tick = time.time()
        mem_before = psutil.virtual_memory().used / 10**9
        result = func(*args, **kargs)

        tock = time.time()
        mem_after = psutil.virtual_memory().used / 10**9

        print(f"< {func.__name__} > took .... {tock - tick:.4f} sec")
        print(f"Memory usage: {mem_after - mem_before:.3f} GB")
        print()
        return result
    return wrapper


class SpatialCellPlotter:
    # @profile
    def __init__(
        self,
        data_path: str = "../data",
        gene_panel_path: str = None,
        experiment_path: str = None,
        cells_csv_path: str = None,
        cells_zarr_path: str = None,
        transcripts_parquet_path: str = None,
        cell_boundaries_parquet_path: str = None,
        nucleus_boundaries_parquet_path: str = None,
        images_path: str = None,
        adata_path: str = None,
        load_mask: bool = False
    ) -> None:
        self.data_path = data_path
        self.gene_panel_path = gene_panel_path or os.path.join(data_path, "gene_panel.json")
        self.experiment_path = experiment_path or os.path.join(data_path, "experiment.xenium")
        self.cells_csv_path = cells_csv_path or os.path.join(data_path, "cells.csv.gz")
        self.cells_zarr_path = cells_zarr_path or os.path.join(data_path, "cells.zarr.zip")
        self.transcripts_parquet_path = transcripts_parquet_path or os.path.join(data_path, "transcripts.parquet")
        self.cell_boundaries_parquet_path = cell_boundaries_parquet_path or os.path.join(data_path, "cell_boundaries.parquet")
        self.nucleus_boundaries_parquet_path = nucleus_boundaries_parquet_path or os.path.join(data_path, "nucleus_boundaries.parquet")
        self.images_path = images_path or os.path.join(data_path, "morphology_focus")
        self.adata_path = adata_path or os.path.join(data_path, "adata.h5ad")

        mem_before = psutil.virtual_memory().used / 10**9
        psutil.cpu_percent()

        self.gene_panel = self._load_json(self.gene_panel_path)
        self.experiment = self._load_json(self.experiment_path)

        xoa_version = self.experiment["analysis_sw_version"].split("-")[1]
        major, minor, patch, _ = map(int, xoa_version.split("."))
        self.multi_stain = True if major >= 2 else False

        self.cells_meta = self._load_cells_meta(self.cells_csv_path)

        if load_mask:
            self.cellseg_mask, self.cellseg_mask_binary, self.nucseg_mask, self.nucseg_mask_binary = self._load_mask(self.cells_zarr_path)
        else:
            self.cellseg_mask, self.cellseg_mask_binary, self.nucseg_mask, self.nucseg_mask_binary = None, None, None, None

        self.transcripts = self._load_transcripts(self.transcripts_parquet_path)
        self.cell_boundaries = self._load_boundaries(self.cell_boundaries_parquet_path)
        self.nucleus_boundaries = self._load_boundaries(self.nucleus_boundaries_parquet_path)
        self.images = self._load_images(self.images_path)
        self.adata = sc.read_h5ad(self.adata_path)


        mem_after = psutil.virtual_memory().used / 10**9
        cpu_percent = psutil.cpu_percent(None, percpu=True)
        print(f"\nTotal memory usage: {mem_after - mem_before:.3f} GB")
        print(f"CPU usage: {cpu_percent} %")

    @stop_watch
    def _load_json(self, path: str) -> dict:
        with open(path, "r") as f:
            return json.load(f)
    
    @stop_watch
    def _load_cells_meta(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        df["x"] = df["x_centroid"] / self.experiment["pixel_size"]
        df["y"] = df["y_centroid"] / self.experiment["pixel_size"]
        print(f'Pixel size: {self.experiment["pixel_size"]}')
        return df
    
    @stop_watch
    def _load_mask(self, path: str) -> tuple:
        cells = zarr.open(path, mode="r")

        cellseg_mask = np.array(cells["masks"][1])
        cellseg_mask_binary = cellseg_mask.copy()
        cellseg_mask_binary[cellseg_mask_binary != 0] = 1

        nucseg_mask = np.array(cells["masks"][0])
        nucseg_mask_binary = nucseg_mask.copy()
        nucseg_mask_binary[nucseg_mask_binary != 0] = 1

        ##################################################
        ##### maskとcell_idの対応を取得するためのコード
        # # pl.DataFrame({"idx": sdata.tables["table"].obs_names + 1,
        # #               "cell_id": sdata.tables["table"].obs.cell_id})
        # cells = zarr.open(cp.cells_zarr_path, mode="r")
        # cellseg_mask = np.array(cells["masks"][1])
        # sdata = xenium("../data")
        # idx_to_cellid = ["backgound"] + sdata.tables["table"].obs.cell_id.tolist()
        # idx = idx_to_cellid.index("ohpbjlnn-1")
        ##################################################

        return cellseg_mask, cellseg_mask_binary, nucseg_mask, nucseg_mask_binary
    
    @stop_watch
    def _load_transcripts(self, path: str) -> pl.DataFrame:
        """_summary_

        Args:
            path (str): _description_

        Returns:
            pl.DataFrame: _description_
        """
        df = pl.read_parquet(path)
        return df.with_columns([
            (pl.col("x_location") / self.experiment["pixel_size"]).alias("x"),
            (pl.col("y_location") / self.experiment["pixel_size"]).alias("y"),
            # (pl.col("z_location") / self.experiment["z_step_size"]).alias("z"),
            (pl.col("z_location") / self.experiment["pixel_size"]).alias("z"),
        ])

    @stop_watch
    def _load_boundaries(self, path: str) -> pl.DataFrame:
        df = pl.read_parquet(path)
        return df.with_columns([
            (pl.col("vertex_x") / self.experiment["pixel_size"]).alias("x"),
            (pl.col("vertex_y") / self.experiment["pixel_size"]).alias("y"),
        ])

    @stop_watch
    def _load_images(self, path: str) -> List[np.array]:
        """
        XOA v2.0以降で`morphology_focus`ディレクトリが存在する
        ref: https://www.10xgenomics.com/support/software/xenium-onboard-analysis/latest/release-notes/release-notes-for-xoa#v2-0
        """

        if self.multi_stain:
            ret = [tifffile.imread(f"{path}/morphology_focus_{i:04d}.ome.tif", is_ome=False, level=0) for i in range(4)]
        else:
            ret = [tifffile.imread(f"{self.data_path}/morphology_focus.ome.tif", is_ome=True, level=0)]
        return ret

    @stop_watch
    def plot_cell(
        self,
        cell_id: str,
        expand: int = 100,
        location: bool = True,
        same_cluster: bool = True,
        cell_boundary: bool = True,
        # boundary: str = "both",
        random_state: int = 0
        ) -> None:
        """_summary_

        Args:
            cell_id (str): `"xxxxx-1"`のようなcell ID
            expand (int, optional): 中央からの拡張範囲. Defaults to 100.
            location (bool, optional): 指定した細胞の切片上・UMAPでの位置を可視化. Defaults to True.
            same_cluster (bool, optional): 同じクラスターに属する他の細胞を可視化. Defaults to True.
            random_state (int, optional): _description_. Defaults to 0.
        """
        x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cell_id]["x"]
        y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cell_id]["y"]
        x_min, x_max = int(round(x_centroid - expand)), int(round(x_centroid + expand))
        y_min, y_max = int(round(y_centroid - expand)), int(round(y_centroid + expand))
        print(x_min, x_max, y_min, y_max)

        cropped_cellmask = self.cellseg_mask_binary[y_min:y_max, x_min:x_max]
        cropped_nucmask = self.nucseg_mask_binary[y_min:y_max, x_min:x_max]
        cropped_images = [img[y_min:y_max, x_min:x_max] for img in self.images] # if self.multi_stain else [self.images[y_min:y_max, x_min:x_max]]

        cropped_cells_meta = self.cells_meta.loc[
            (self.cells_meta["x"] >= x_min) & (self.cells_meta["x"] <= x_max) &
            (self.cells_meta["y"] >= y_min) & (self.cells_meta["y"] <= y_max)].copy()
        
        cropped_cells_meta["x"] = cropped_cells_meta["x"] - x_min
        cropped_cells_meta["y"] = cropped_cells_meta["y"] - y_min

        cropped_boundaries = self.cell_boundaries.filter(
            (pl.col("x") >= x_min) & (pl.col("x") <= x_max) &
            (pl.col("y") >= y_min) & (pl.col("y") <= y_max)
        ).with_columns([
            (pl.col("x") - x_min).alias("x"),
            (pl.col("y") - y_min).alias("y"),])

        if location:
            self._plot_location(x_centroid, y_centroid, cell_id)
        
        if same_cluster:
            self._plot_same_cluster(cell_id, expand, random_state=random_state)
        
        self._plot_segmentation(
            cell_id, 
            expand,
            cell_boundary,
            cropped_nucmask, 
            cropped_cellmask, 
            cropped_images, 
            cropped_cells_meta, 
            cropped_boundaries
        )

        cid_in_crop = [cid for cid in cropped_cells_meta["cell_id"].unique()]
        print(f"{len(cid_in_crop)} cells in the cropped area")

        del cropped_cellmask, cropped_nucmask, cropped_images, cropped_cells_meta, cropped_boundaries
        gc.collect()
        
        return cid_in_crop


    @stop_watch
    def _plot_location(self, x_centroid: float, y_centroid: float, cell_id: str):
        """
        privateメソッド: クラス内でのみ使用されるメソッド。
        公開メソッド (ここではplot_cell) の下に配置することが多い。
        """
        print(f"Plotting location for {cell_id}")
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"Cell ID: {cell_id}", fontweight="bold", fontsize=20)
        
        rect = patch.Circle(xy=(x_centroid, y_centroid), radius=100, ec="r", fc="w", linewidth=5)
        axes[0].invert_yaxis()
        axes[0].add_patch(rect)
        axes[0].imshow(self.cellseg_mask_binary)
        
        umap_coords = self.adata.obsm["X_umap"]
        cell_names = self.adata.obs_names

        df_umap = pl.DataFrame({
            "index": cell_names,
            "UMAP1": umap_coords[:, 0],
            "UMAP2": umap_coords[:, 1],
            "flag": [False] * len(cell_names)
        }).with_columns(
            pl.when(pl.col("index") == cell_id)
            .then(True)
            .otherwise(pl.col("flag"))
            .alias("flag")
        )

        axes[1].scatter(data=df_umap.filter(pl.col("flag") == False), x="UMAP1", y="UMAP2", c="grey", s=3)
        axes[1].scatter(data=df_umap.filter(pl.col("flag") == True), x="UMAP1", y="UMAP2", c="red", s=50, marker="*")
        axes[1].set_xlabel("UMAP1")
        axes[1].set_ylabel("UMAP2")
        plt.show()

    @stop_watch
    def _plot_same_cluster(
        self, 
        cell_id: str, 
        expand: int = 100,
        cluster: str = "leiden",
        random_state: int = 0
        ) -> None:

        n_cluster = self.adata.obs[self.adata.obs["cell_id"] == cell_id][cluster].values[0]
        print(f"Plotting same cluster-{n_cluster} cells for {cell_id} with expand {expand}")

        subset_adata = sc.pp.subsample(
            self.adata[self.adata.obs[cluster] == n_cluster],
            n_obs=3,
            random_state=random_state,
            copy=True)
        cell_ids = [cell_id] + subset_adata.obs["cell_id"].tolist()
        print(f"Picked cell_ids: {cell_ids}")

        fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex="all", sharey="all")
        fig.suptitle(f"Cells in the same cluster {n_cluster}", fontweight="bold", fontsize=20)

        for i, cid in enumerate(cell_ids):

            cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"]
            cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"]
            cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
            cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
            cid_cropped_img_tiff_0000 = self.images[0][cid_y_min:cid_y_max, cid_x_min:cid_x_max]
            cid_cropped_img_tiff_0002 = self.images[2][cid_y_min:cid_y_max, cid_x_min:cid_x_max]

            # Nuclei mask
            cid_cropped_nucmask = self.nucseg_mask[cid_y_min:cid_y_max, cid_x_min:cid_x_max]
            cid_cropped_nucmask[cid_cropped_nucmask != 0] = 1

            cid_cropped_cells_meta = self.cells_meta.loc[
                (self.cells_meta["x"] >= cid_x_min) & (self.cells_meta["x"] <= cid_x_max) &
                (self.cells_meta["y"] >= cid_y_min) & (self.cells_meta["y"] <= cid_y_max)].copy()
            cid_cropped_cells_meta["x"] = cid_cropped_cells_meta["x"] - cid_x_min
            cid_cropped_cells_meta["y"] = cid_cropped_cells_meta["y"] - cid_y_min

            axes[0][i].imshow(cid_cropped_img_tiff_0000, cmap="grey")
            axes[0][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="yellow", marker="*", s=100)
            axes[0][i].set_title(f"DAPI ({cid})")

            axes[1][i].imshow(cid_cropped_nucmask, cmap="grey")
            axes[1][i].scatter(data=cid_cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
            axes[1][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="red", marker="*", s=100)
            axes[1][i].set_title("Nucleus segmentation")
            
            axes[2][i].imshow(cid_cropped_img_tiff_0002, cmap="grey")
            axes[2][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="yellow", marker="*", s=100)
            axes[2][i].set_title(f"RNA(18S) ({cid})")

        plt.tight_layout()
        plt.show()


    @stop_watch
    def _plot_segmentation(
        self, 
        cell_id: str, 
        expand: int,
        cell_boundary: bool,
        cropped_nucmask: np.ndarray, 
        cropped_cellmask: np.ndarray, 
        cropped_images: List[np.ndarray], 
        cropped_cells_meta: pd.DataFrame, 
        cropped_boundaries: pl.DataFrame
        ) -> None:

        ### Fig作成
        fig, axes = plt.subplots(2, 3, figsize=(12, 9), sharex="all", sharey="all")
        fig.suptitle(f"Cell ID: {cell_id}", fontweight="bold", fontsize=20)

        # セグメンテーションマスクを表示
        axes[0][0].imshow(cropped_nucmask, cmap = "grey")
        # axes[0][0].scatter(data=cropped_boundaries, x="x", y="y", color="yellow", s=1)
        axes[0][0].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[0][0].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="red", marker="*", s=100)
        axes[0][0].set_title("Nucleus segmentation")
        axes[0][0].set_xlim(0, expand*2)
        axes[0][0].set_ylim(expand*2, 0)

        axes[0][1].invert_yaxis()   # y座標を反転
        axes[0][1].imshow(cropped_cellmask, cmap = "grey")
        # axes[0][1].scatter(data=cropped_boundaries, x="x", y="y", color="blue", s=1)
        axes[0][1].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[0][1].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="red", marker="*", s=100)
        axes[0][1].set_title("Cell segmentation")
        axes[0][1].set_xlim(0, expand*2)
        axes[0][1].set_ylim(0, expand*2)
        # axes[0][1].set_aspect("equal")


        axes[0][2].imshow(cropped_images[0], cmap="grey")
        axes[0][2].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[0][2].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="yellow", marker="*", s=100)
        axes[0][2].set_title("DAPI")
        # axes[0][2].set_xlim(0, expand*2)
        # axes[0][2].set_ylim(0, expand*2)
        # axes[0][2].set_aspect("equal")


        axes[1][0].imshow(cropped_images[1], cmap="grey")
        axes[1][0].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[1][0].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="yellow", marker="*", s=100)
        axes[1][0].set_title("ATP1A1/E-Cadherin/CD45")
        # axes[1][0].set_xlim(0, expand*2)
        # axes[1][0].set_ylim(0, expand*2)
        # axes[1][0].set_aspect("equal")

        axes[1][1].imshow(cropped_images[2], cmap="grey")
        axes[1][1].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[1][1].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="yellow", marker="*", s=100)
        axes[1][1].set_title("RNA(18S)")
        # axes[1][1].set_xlim(0, expand*2)
        # axes[1][1].set_ylim(0, expand*2)
        # axes[1][1].set_aspect("equal")

        axes[1][2].imshow(cropped_images[3], cmap="grey")
        axes[1][2].scatter(data=cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
        axes[1][2].scatter(data=cropped_cells_meta[cropped_cells_meta["cell_id"] == cell_id], x="x", y="y", color="yellow", marker="*", s=100)
        axes[1][2].set_title("alphaSMA/Vimentin")
        # axes[1][2].set_xlim(0, expand*2)
        # axes[1][2].set_ylim(0, expand*2)
        # axes[1][2].set_aspect("equal")

        if cell_boundary:
            for cell_id, group in cropped_boundaries.group_by("cell_id"):
                x = group["x"]
                y = group["y"]
                axes[0][0].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                axes[0][1].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                axes[0][2].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                axes[1][0].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                axes[1][1].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                axes[1][2].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)

        plt.tight_layout()
        plt.show()

    @stop_watch
    def plot_transcripts(self, cell_id: str, expand: int = 100, transcripts: str = "COL1A1"):
        """
        Cell IDの周辺のTranscriptsをプロットする(左側), DAPI + cell segmentation polygon + transcriptの位置
        指定したtranscriptについて、各クラスターでの発現量をviolin plotで表示する(右側)
        """

        df = self.transcripts.filter(pl.col("feature_name") == transcripts)

        return None

    @stop_watch
    def plot_cell_dapi(
        self,
        cell_id: str,
        expand: int = 100,
        n_rows: int = 2,
        n_cols: int = 3,
        same_cluster: bool = True,
        cluster: str = "leiden",
        boundary: str = "both",
        transcripts: List[str] = None,
        random_state: int = 0
        ) -> None:
        """
        DAPIのみをプロットする
        """
        n_cells = n_rows * n_cols

        if same_cluster:
            n_cluster = self.adata.obs[self.adata.obs["cell_id"] == cell_id][cluster].values[0]
            subset_adata = sc.pp.subsample(
                self.adata[self.adata.obs[cluster] == n_cluster],
                n_obs=n_cells - 1,
                random_state=random_state,
                copy=True)
            cell_ids = [cell_id] + subset_adata.obs["cell_id"].tolist()
            print(f"Picked cell_ids: {cell_ids}")
        else:
            subset_adata = sc.pp.subsample(
                self.adata,
                n_obs=n_cells - 1,
                random_state=random_state,
                copy=True)
            cell_ids = [cell_id] + subset_adata.obs["cell_id"].tolist()
            print(f"Picked cell_ids: {cell_ids}")
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(round(3.5*n_cols), round(2.5*n_cols)), sharex="all", sharey="all")
        fig.suptitle(f"DAPI for Cell ID: {cell_id}", fontweight="bold", fontsize=20)

        for i, cid in enumerate(cell_ids):
            cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"]
            cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"]
            cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
            cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
            cid_cropped_img_tiff_0000 = self.images[0][cid_y_min:cid_y_max, cid_x_min:cid_x_max]

            cid_cropped_cells_meta = self.cells_meta.loc[
                (self.cells_meta["x"] >= cid_x_min) & (self.cells_meta["x"] <= cid_x_max) &
                (self.cells_meta["y"] >= cid_y_min) & (self.cells_meta["y"] <= cid_y_max)].copy()
            cid_cropped_cells_meta["x"] = cid_cropped_cells_meta["x"] - cid_x_min
            cid_cropped_cells_meta["y"] = cid_cropped_cells_meta["y"] - cid_y_min
            cell_ids_cropped = cid_cropped_cells_meta["cell_id"].unique()

            axes[i // n_cols][i % n_cols].imshow(cid_cropped_img_tiff_0000, cmap="gray")
            axes[i // n_cols][i % n_cols].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="yellow", marker="*", s=100)
            axes[i // n_cols][i % n_cols].set_title(f"DAPI ({cid})")
            axes[i // n_cols][i % n_cols].set_xlim(0, expand*2)
            axes[i // n_cols][i % n_cols].set_ylim(expand*2, 0)


            # def plot_clipped_boundary(boundary_df, alpha_val):
            #     """
            #     指定したcell_idのboundaryをプロットする
            #     """
            #     from shapely.geometry import Polygon, box
            #     from shapely.ops import clip_by_rect

            #     boundaries = boundary_df.filter(pl.col("cell_id") == cid)
            #     polygon = Polygon(zip(boundaries["x"], boundaries["y"]))

            #     clipping_box = box(cid_x_min, cid_y_min, cid_x_max, cid_y_max)
            #     clipped_polygon = polygon.intersection(clipping_box)

            #     if not clipped_polygon.is_empty:
            #         x, y = clipped_polygon.exterior.xy
            #         x_shifted = [coord - cid_x_min for coord in x]
            #         y_shifted = [coord - cid_y_min for coord in y]
            #         axes[i // n_cols][i % n_cols].plot(x_shifted, y_shifted, label=f"Cell {cid}", alpha=alpha_val)
                
            # if boundary in ["cell", "both"]:
            #     plot_clipped_boundary(self.cell_boundaries, 0.7)

            # if boundary in ["nuclei", "both"]:
            #     plot_clipped_boundary(self.nucleus_boundaries, 0.4)

            if boundary == "cell":
                # cid_cropped_boundaries = self.cell_boundaries.filter(
                #     (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                #     (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
                # ).with_columns([
                #     (pl.col("x") - cid_x_min).alias("x"),
                #     (pl.col("y") - cid_y_min).alias("y"),])
                cid_cropped_boundaries = self.cell_boundaries.filter(
                    pl.col("cell_id").is_in(cell_ids_cropped)
                ).with_columns([
                    (pl.col("x") - cid_x_min).alias("x"),
                    (pl.col("y") - cid_y_min).alias("y"),])

                for cell_id, group in cid_cropped_boundaries.group_by("cell_id"):
                    x = group["x"]
                    y = group["y"]
                    axes[i // n_cols][i % n_cols].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                
                del cid_cropped_boundaries
            
            elif boundary == "nuclei":
                # cid_cropped_boundaries = self.nucleus_boundaries.filter(
                #     (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                #     (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
                # ).with_columns([
                #     (pl.col("x") - cid_x_min).alias("x"),
                #     (pl.col("y") - cid_y_min).alias("y"),])
                cid_cropped_boundaries = self.nucleus_boundaries.filter(
                    pl.col("cell_id").is_in(cell_ids_cropped)
                ).with_columns([
                    (pl.col("x") - cid_x_min).alias("x"),
                    (pl.col("y") - cid_y_min).alias("y"),])

                for cell_id, group in cid_cropped_boundaries.group_by("cell_id"):
                    x = group["x"]
                    y = group["y"]
                    axes[i // n_cols][i % n_cols].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                
                del cid_cropped_boundaries
            
            elif boundary == "both":
                # cid_cropped_boundaries_cell = self.cell_boundaries.filter(
                #     (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                #     (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
                # ).with_columns([
                #     (pl.col("x") - cid_x_min).alias("x"),
                #     (pl.col("y") - cid_y_min).alias("y"),])
                cid_cropped_boundaries_cell = self.cell_boundaries.filter(
                    pl.col("cell_id").is_in(cell_ids_cropped)
                ).with_columns([
                    (pl.col("x") - cid_x_min).alias("x"),
                    (pl.col("y") - cid_y_min).alias("y"),])

                # cid_cropped_boundaries_nuclei = self.nucleus_boundaries.filter(
                #     (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                #     (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
                # ).with_columns([
                #     (pl.col("x") - cid_x_min).alias("x"),
                #     (pl.col("y") - cid_y_min).alias("y"),])
                cid_cropped_boundaries_nuclei = self.nucleus_boundaries.filter(
                    pl.col("cell_id").is_in(cell_ids_cropped)
                ).with_columns([
                    (pl.col("x") - cid_x_min).alias("x"),
                    (pl.col("y") - cid_y_min).alias("y"),])

                for cell_id, group in cid_cropped_boundaries_cell.group_by("cell_id"):
                    x = group["x"]
                    y = group["y"]
                    axes[i // n_cols][i % n_cols].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                
                for cell_id, group in cid_cropped_boundaries_nuclei.group_by("cell_id"):
                    x = group["x"]
                    y = group["y"]
                    axes[i // n_cols][i % n_cols].plot(x, y, label=f"Cell {cell_id}", alpha=0.4)
                
                del cid_cropped_boundaries_cell, cid_cropped_boundaries_nuclei

        if transcripts is not None:
            transcript_df = self.transcripts.filter(pl.col("feature_name").is_in(transcripts))
            # transcriptごとに色を分けて、plotする。legendは右上に表示
            for transcript in transcripts:
                for i, cid in enumerate(cell_ids):
                    cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"]
                    cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"]
                    cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
                    cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
                    cid_cropped_transcript = transcript_df.filter(
                        (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                        (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
                    ).with_columns([
                        (pl.col("x") - cid_x_min).alias("x"),
                        (pl.col("y") - cid_y_min).alias("y"),])
                    axes[i // n_cols][i % n_cols].scatter(data=cid_cropped_transcript, x="x", y="y", color="red", s=100)
                    print(f"Transcript {transcript} for {cid} is plotted: {len(cid_cropped_transcript)} points")
    
        # if transcript in not None:
        #     transcript_df = self.transcripts.filter(pl.col("feature_name") == transcript)
        #     for i, cid in enumerate(cell_ids):
        #         cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"]
        #         cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"]
        #         cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
        #         cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
        #         cid_cropped_transcript = transcript_df.filter(
        #             (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
        #             (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
        #         ).with_columns([
        #             (pl.col("x") - cid_x_min).alias("x"),
        #             (pl.col("y") - cid_y_min).alias("y"),])
        #         axes[i // n_cols][i % n_cols].scatter(data=cid_cropped_transcript, x="x", y="y", color="red", s=100)
        #         print(f"Transcript {transcript} for {cid} is plotted: {len(cid_cropped_transcript)} points")

        plt.tight_layout()
        plt.show()

        del cid_cropped_img_tiff_0000, cid_cropped_cells_meta
        gc.collect()


    @stop_watch
    def plot_cluster(
        self, 
        cluster_id: str = None, 
        expand: int = 100,
        cell_boundary: bool = True,
        n_cells: int = 4,
        cluster: str = "leiden",
        random_state: int = 0
        ) -> None:
        """
        クラスターごとにセルをプロットする
        行方向: DAPI, Nucleus segmentation, RNA(18S)
        列方向: n_cells個
        plot_cellはcell_idを指定していたが、こちらはクラスターidを指定する
        """
        print(f"Plotting same cluster-{cluster_id} with expand {expand}")
        subset_adata = sc.pp.subsample(
            self.adata[self.adata.obs[cluster] == cluster_id],
            n_obs=n_cells,
            random_state=random_state,
            copy=True)
        cell_ids = subset_adata.obs["cell_id"].tolist()
        print(f"Picked cell_ids: {cell_ids}")

        fig, axes = plt.subplots(3, n_cells, figsize=(round(3.5*n_cells), round(2.5*n_cells)), sharex="all", sharey="all")
        fig.suptitle(f"Cells in the same cluster {cluster_id}", fontweight="bold", fontsize=20)

        for i, cid in enumerate(cell_ids):
            cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"]
            cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"]
            cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
            cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
            cid_cropped_img_tiff_0000 = self.images[0][cid_y_min:cid_y_max, cid_x_min:cid_x_max]
            cid_cropped_img_tiff_0002 = self.images[2][cid_y_min:cid_y_max, cid_x_min:cid_x_max]

            # Nuclei mask
            cid_cropped_nucmask = self.nucseg_mask[cid_y_min:cid_y_max, cid_x_min:cid_x_max]
            cid_cropped_nucmask[cid_cropped_nucmask != 0] = 1

            cid_cropped_cells_meta = self.cells_meta.loc[
                (self.cells_meta["x"] >= cid_x_min) & (self.cells_meta["x"] <= cid_x_max) &
                (self.cells_meta["y"] >= cid_y_min) & (self.cells_meta["y"] <= cid_y_max)].copy()
            cid_cropped_cells_meta["x"] = cid_cropped_cells_meta["x"] - cid_x_min
            cid_cropped_cells_meta["y"] = cid_cropped_cells_meta["y"] - cid_y_min

            cid_cropped_boundaries = self.cell_boundaries.filter(
                (pl.col("x") >= cid_x_min) & (pl.col("x") <= cid_x_max) &
                (pl.col("y") >= cid_y_min) & (pl.col("y") <= cid_y_max)
            ).with_columns([
                (pl.col("x") - cid_x_min).alias("x"),
                (pl.col("y") - cid_y_min).alias("y"),])

            axes[0][i].imshow(cid_cropped_img_tiff_0000, cmap="grey")                
            axes[0][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="yellow", marker="*", s=100)
            axes[0][i].set_title(f"DAPI ({cid})")
            axes[0][i].set_xlim(0, expand*2)
            axes[0][i].set_ylim(expand*2, 0)
            
            axes[1][i].imshow(cid_cropped_nucmask, cmap="grey")
            axes[1][i].scatter(data=cid_cropped_cells_meta, x="x", y="y", color="red", marker="*", s=40)
            axes[1][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="red", marker="*", s=100)
            axes[1][i].set_title("Nucleus segmentation")
            axes[1][i].set_xlim(0, expand*2)
            axes[1][i].set_ylim(expand*2, 0)

            axes[2][i].imshow(cid_cropped_img_tiff_0002, cmap="grey")
            axes[2][i].scatter(data=cid_cropped_cells_meta[cid_cropped_cells_meta["cell_id"] == cid], x="x", y="y", color="yellow", marker="*", s=100)
            axes[2][i].set_title(f"RNA(18S) ({cid})")
            axes[2][i].set_xlim(0, expand*2)
            axes[2][i].set_ylim(expand*2, 0)

            if cell_boundary:
                for cell_id, group in cid_cropped_boundaries.group_by("cell_id"):
                    x = group["x"]
                    y = group["y"]
                    axes[0][i].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                    axes[1][i].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)
                    axes[2][i].plot(x, y, label=f"Cell {cell_id}", alpha=0.7)

        plt.tight_layout()
        plt.show()

        del cid_cropped_img_tiff_0000, cid_cropped_img_tiff_0002, cid_cropped_nucmask, cid_cropped_cells_meta, cid_cropped_boundaries
        gc.collect()

        # return axes


    @stop_watch
    def plot_cell_hist(
        self, 
        # data: pd.DataFrame,
        left: str = "nucleus_area",
        right: str = "cell_area",
        log_scale: bool = True
        ) -> None:
        """nucleus_areaとcell_areaのヒストグラムをプロットする

        Args:
            data (pd.DataFrame): cells_meta (cells.csv.gz)のデータフレーム
        """
        n_total = len(self.cells_meta)
        n_bins = round(1+np.log2(n_total))
        fig, axes = plt.subplots(1, 2, figsize=(9,4))
        if log_scale:
            sns.histplot(self.cells_meta[left], bins=n_bins*2, color="Green", ax=axes[0], log_scale=True)
            sns.histplot(self.cells_meta[right], bins=n_bins*2, color="Green", ax=axes[1], log_scale=True)
            if left == "nucleus_area":
                axes[0].set_title("Nucleus area (log scale)")
            if right == "cell_area":
                axes[1].set_title("Cell area (log scale)")
            if right == "transcript_counts":
                axes[1].set_title("Transcript count (log scale)")
        else:
            sns.histplot(self.cells_meta[left], bins=n_bins*2, color="Green", ax=axes[0])
            sns.histplot(self.cells_meta[right], bins=n_bins*2, color="Green", ax=axes[1])
            if left == "nucleus_area":
                axes[0].set_title("Nucleus area (µm$^2$)")
            if right == "cell_area":
                axes[1].set_title("Cell area (µm$^2$)")
            if right == "transcript_counts":
                axes[1].set_title("Transcript count")
        plt.tight_layout()
        plt.show()


    @stop_watch
    def calc_ssgsea(self):
        """
        Single-sample GSEA (ssGSEA)を計算する
        """
        # from gseapy import ssgsea
        # import os
        # from concurrent.futures import ProcessPoolExecutor

        # max_workers = os.cpu_count() or 4
        # with ProcessPoolExecutor(max_workers=max_workers) as executor:
        #     for gene in self.gene_panel:
        #         executor.submit(self._calc_ssgsea, gene)

        return None


    def process_and_cluster_nuclei(self, cell_ids, expand=100, n_clusters=3):
        """
        クロップしたnucleiのboundariesを使ってマスクを作成し、形状特徴量を抽出してクラスタリングを行う。
        """
        import numpy as np
        from shapely.geometry import Polygon
        from skimage.draw import polygon
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        from skimage.measure import label, regionprops
        import pandas as pd

        features = []  # 特徴量を格納するリスト

        for cid in cell_ids:
            # セルの中心座標を取得
            cid_x_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["x"].iloc[0]
            cid_y_centroid = self.cells_meta[self.cells_meta["cell_id"] == cid]["y"].iloc[0]
            cid_x_min, cid_x_max = int(round(cid_x_centroid - expand)), int(round(cid_x_centroid + expand))
            cid_y_min, cid_y_max = int(round(cid_y_centroid - expand)), int(round(cid_y_centroid + expand))
            cropped_img = self.images[0][cid_y_min:cid_y_max, cid_x_min:cid_x_max]

            # クロップされたnucleiのboundariesを取得
            cropped_nuclei_boundaries = self.nucleus_boundaries.filter(
                pl.col("cell_id") == cid
            )

            # マスク画像を作成
            mask = np.zeros_like(cropped_img, dtype=np.uint8)
            for _, group in cropped_nuclei_boundaries.group_by("cell_id"):
                # ポリゴンを作成
                polygon_coords = Polygon(zip(group["x"] - cid_x_min, group["y"] - cid_y_min))
                if not polygon_coords.is_empty:
                    rr, cc = polygon(np.array(polygon_coords.exterior.xy[1]), 
                                    np.array(polygon_coords.exterior.xy[0]), 
                                    shape=mask.shape)
                    mask[rr, cc] = 1  # 対象領域を1に設定

            # マスク画像を用いた領域の形状特徴量を計算
            labeled_mask = label(mask)  # ラベル付け（複数領域がある場合の対応）
            for region in regionprops(labeled_mask):
                # 各領域から形状特徴量を抽出
                features.append({
                    "cell_id": cid,
                    "area": region.area,  # 面積
                    "perimeter": region.perimeter,  # 周囲長
                    "eccentricity": region.eccentricity,  # 偏心率
                    "solidity": region.solidity,  # 凸充実度
                    "extent": region.extent,  # 広がり
                    "aspect_ratio": region.bbox[3] / region.bbox[2]  # アスペクト比
                })

        # 特徴量をデータフレームに変換
        features_df = pd.DataFrame(features)

        # クラスタリングの実施
        scaler = StandardScaler()
        clustering_features = features_df[["area", "perimeter", "eccentricity", "solidity", "extent", "aspect_ratio"]]
        clustering_features_scaled = StandardScaler().fit_transform(clustering_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        features_df["cluster"] = kmeans.fit_predict(clustering_features_scaled)

        return features_df


    def get_count_from_nuclei(self, cell_ids, expand=100):
        """
        クロップしたnucleiのboundariesを使ってマスクを作成し、形状特徴量を抽出してクラスタリングを行う。
        """
        # for i in [0, 1, 2, 3, 4, 5]:
        #     df = pl.DataFrame(
        #         scipy.sparse.csr_matrix(cp.adata.layers[f"count_{i}um_from_nuclei"]).toarray())
        #     df.columns = cp.adata.var_names
        #     cp.adata.obs[f"count_{i}um_from_nuclei"] = df.sum_horizontal()
        return None


def ssGSEA_single(cell_data, gene_set, alpha=0.25, scale=True, single=True):
    """
    ssGSEA (single-sample GSEA) scoreを計算する (single cell, single gene set)

    Parameters:
        cell_data (pl.DataFrame): Filtered data for a single cell (genes x expression).
        gene_set (list): List of genes in the gene set.
        alpha (float): Weighting factor for enrichment score.
        scale (bool): Whether to scale the enrichment score by the number of genes.
        single (bool): Whether to return ssGSEA score or the maximum enrichment score.

    Returns:
        float: ssGSEA score for the cell and gene set.
    """
    # Filter genes in the gene set
    gene_set_data = cell_data.filter(pl.col("gene").is_in(gene_set))

    # If no genes from the gene set are present, return 0
    if gene_set_data.height == 0:
        return 0.0

    # Rank genes by expression
    gene_set_data = gene_set_data.sort("expression", descending=True)
    gene_set_data = gene_set_data.with_columns(
        pl.arange(1, gene_set_data.height + 1).alias("rank")  # Rank starts at 1
    )

    # Calculate ranks
    ranks = gene_set_data["rank"].to_numpy()
    expressions = gene_set_data["expression"].to_numpy()

    # Apply weight to ranks (defaultでalpha = 0.25, GSVAではalpha=1)
    weights = (expressions ** alpha)

    # Step CDFs
    step_cdf_pos = np.cumsum(weights) / np.sum(weights)
    step_cdf_neg = np.cumsum(np.ones_like(weights)) / len(ranks)

    # Enrichment score
    step_cdf_diff = step_cdf_pos - step_cdf_neg

    if scale:
        step_cdf_diff /= len(ranks)

    if single:
        return np.sum(step_cdf_diff)
    else:
        return step_cdf_diff[np.argmax(np.abs(step_cdf_diff))]


def process_ssGSEA(args):
    """
    ssGSEAの計算を並列化するための関数

    Parameters:
        args (tuple): (cell_id, geneset, df_long)

    Returns:
        dict: ssGSEA score for the cell and gene set.
    """
    cell_id, geneset, df_long = args

    # Filter data for the given cell_id
    cell_data = df_long.filter(pl.col("cell_id") == cell_id)

    # Calculate ssGSEA score
    score = ssGSEA_single(cell_data, geneset)

    return {
        "cell_id": cell_id,
        "geneset": geneset,
        "score": score
    }