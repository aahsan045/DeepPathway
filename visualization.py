import spatialdata_plot as sd
from PIL import Image
from spatialdata import read_zarr
import spatialdata as sd
from spatialdata.models import TableModel, Image2DModel
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import geopandas as gpd
from shapely.geometry import Point
from spatialdata.models import Image2DModel, ShapesModel, TableModel
import contextlib
import config as cfg

def plot_sdata(sdata,sample_id,top_k_pathway_names=['Hypoxia','DNA Repair','Myc Targets V1']):
    image_path = cfg.root_path + "wsis/" + sample_id + ".tif"
    full_res_image=cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    image_for_sdata =Image2DModel.parse(data=full_res_image,dims=('y','x','c'))
    adata_true=sdata['adata_pathways']
    if "+" in cfg.method:
        method = cfg.method.replace("+","_")
    else:
        method = cfg.method
    adata_preds=sdata['predictions_'+method]
    radius = adata_true.uns['spatial']['ST']['scalefactors']['spot_diameter_fullres']/2
    centers = adata_true.obsm["spatial"]
    # centers_new = np.array([[y,x] for x,y in adata_true.obsm["spatial"]])
    # spot_df = pd.concat([adata_true.obs[["array_col", "array_row"]].reset_index(drop=True),
    #                      pd.DataFrame(centers_new, columns=["x", "y"])],axis=1,ignore_index=True,
    #                    )
    # spot_df.columns = ["array_col", "array_row", "spot_center_x", "spot_center_y"]
    # fixed_row = spot_df.array_row.iloc[0].item()
    # cols = spot_df.query(f"array_row == {fixed_row}").array_col
    # min_col, max_col = cols.min().item(), cols.max().item()
    # xs = spot_df.query(f"array_row == {fixed_row}").spot_center_x
    # min_x, max_x = xs.min().item(), xs.max().item()
    # px_per_um = (max_x - min_x) / ((max_col - min_col) / 2) / 100
    # radius = px_per_um * 55 / 2
    df = pd.DataFrame([radius] * len(centers), columns=["radius"])
    gdf = gpd.GeoDataFrame(df, geometry=[Point(x, y) for x, y in centers])
    shapes_for_sdata = ShapesModel.parse(gdf)
    with contextlib.suppress(KeyError):
        del adata_true.uns["spatial"]
        del adata_preds.uns["spatial"]
    with contextlib.suppress(KeyError):
        del adata_true.obsm["spatial"]
        del adata_preds.obsm["spatial"]
    adata_true_for_sdata = TableModel.parse(adata_true)
    adata_preds_for_sdata = TableModel.parse(adata_preds)
    adata_true_for_sdata.uns["spatialdata_attrs"] = {"region": "spots",  # name of the Shapes element we will use later (i.e. the object with centers and radii of the Visium spots)
                                                "region_key": "region",  # column in adata.obs that will link a given obs to the elements it annotates
                                                "instance_key": "spot_id",  # column that matches a given obs in the table to a given circle
                                               }
    adata_preds_for_sdata.uns["spatialdata_attrs"] = {"region": "spots",
                                                     # name of the Shapes element we will use later (i.e. the object with centers and radii of the Visium spots)
                                                     "region_key": "region",
                                                     # column in adata.obs that will link a given obs to the elements it annotates
                                                     "instance_key": "spot_id",
                                                     # column that matches a given obs in the table to a given circle
                                                     }
    # all the rows of adata annotate the same element, called "spots" (as we declared above)
    adata_true_for_sdata.obs["region"] = pd.Categorical(["spots"] * len(adata_true))
    adata_preds_for_sdata.obs["region"] = pd.Categorical(["spots"] * len(adata_preds))
    adata_true_for_sdata.obs["spot_id"] = shapes_for_sdata.index
    adata_preds_for_sdata.obs["spot_id"] = shapes_for_sdata.index
    new_sdata=sd.SpatialData(images={"full_res_image": image_for_sdata},
                          shapes={"spots": shapes_for_sdata},
                          tables={"adata_true": adata_true_for_sdata},
                         )
    print(adata_preds_for_sdata)
    new_sdata.tables['adata_pred'] = adata_preds_for_sdata
    fig, axs = plt.subplots(1, len(top_k_pathway_names), figsize=(18, 5))
    for i in range(0,len(top_k_pathway_names)):
        new_sdata.pl.render_images(alpha=0.3).pl.render_shapes(color=top_k_pathway_names[i],scale=1.5,cmap=mpl.cm.Reds,alpha=0.7,table_name='adata_true').pl.show(ax=axs[i], title=top_k_pathway_names[i])

    fig, axs = plt.subplots(1, len(top_k_pathway_names), figsize=(18, 5))
    for i in range(0, len(top_k_pathway_names)):
        new_sdata.pl.render_images(alpha=0.3).pl.render_shapes(color=top_k_pathway_names[i], scale=1.5,
                                                               cmap=mpl.cm.Reds, alpha=0.7,
                                                               table_name='adata_pred').pl.show(ax=axs[i], title=top_k_pathway_names[i])
    return