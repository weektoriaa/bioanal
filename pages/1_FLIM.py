import itertools
import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tifffile
from scipy import ndimage as ndi
from scipy import stats
from skimage import exposure, filters, measure, morphology, segmentation
from statsmodels.stats.multitest import multipletests

wanted = ["tm", "t1", "t2", "a1"]
encodings = ["utf-8", "cp1251", "latin-1", "cp866"]
name_pattern = re.compile(
    r"^(?P<base>.+)_(?P<num>\d+)(?:\._sdt_statistic_all|\.sdt_statistic_all|_statistic_all|_statistics)\.asc(?:_statistic_all\.asc)?$",
    re.IGNORECASE,
)
roi_pattern = re.compile(r"ROI\s*([0-9]+)", re.IGNORECASE)
safe_filename_pattern = re.compile(r"[^A-Za-z0-9._-]+")


def extract_roi_number(line: str):
    match = roi_pattern.search(line)
    if match:
        return int(match.group(1))
    return line.strip()


def sanitize_filename(name: str, fallback: str = "flim_statistics") -> str:
    clean_name = safe_filename_pattern.sub("_", name.strip()).strip("._")
    return clean_name or fallback


def parse_uploaded_file(uploaded_file):
    raw = uploaded_file.getvalue()
    text = None
    used_encoding = None
    for enc in encodings:
        try:
            text = raw.decode(enc)
            used_encoding = enc
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        return None, None, "Could not decode file with available encodings."

    current_roi = None
    rows = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if line.startswith("ROI"):
            current_roi = extract_roi_number(line.replace(",", ""))
            continue

        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue

        metric = parts[0]
        if metric in wanted:
            try:
                mu = float(parts[1])
                rows.append([current_roi, metric, mu])
            except ValueError:
                continue

    return rows, used_encoding, None


def p_to_stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "ns"


def format_p_value(p_value: float) -> str:
    if pd.isna(p_value):
        return "p = NA"
    if p_value < 0.0001:
        return "p < 0.0001"
    return f"p = {p_value:.4f}"


def build_pairwise_options(pairwise_df: pd.DataFrame):
    options = []
    for _, row in pairwise_df.iterrows():
        options.append(f"{row['group_1']} vs {row['group_2']}")
    return options


def get_pairwise_annotation_rows(pairwise_df: pd.DataFrame, selected_pairs):
    if pairwise_df.empty or not selected_pairs:
        return []

    selected_set = set(selected_pairs)
    rows = []
    for _, row in pairwise_df.iterrows():
        pair_label = f"{row['group_1']} vs {row['group_2']}"
        if pair_label in selected_set:
            rows.append(
                {
                    "group_1": row["group_1"],
                    "group_2": row["group_2"],
                    "p_value": row["p_adj_fdr_bh"] if "p_adj_fdr_bh" in row and pd.notna(row["p_adj_fdr_bh"]) else row["p_raw"],
                }
            )
    return rows


def build_plot_df(group_tables):
    plot_frames = []
    for base_name, table in group_tables.items():
        plot_table = table.reset_index().copy()
        plot_table.insert(0, "name", base_name)
        plot_frames.append(plot_table)

    if not plot_frames:
        return pd.DataFrame(columns=["name", "FileNum", "ROI"] + wanted)

    plot_df = pd.concat(plot_frames, ignore_index=True)
    ordered_columns = ["name", "FileNum", "ROI"] + [metric for metric in wanted if metric in plot_df.columns]
    return plot_df[ordered_columns]


def remove_zero_values(plot_df: pd.DataFrame, metric_columns):
    cleaned_df = plot_df.copy()
    for metric in metric_columns:
        if metric in cleaned_df.columns:
            cleaned_df[metric] = cleaned_df[metric].where(cleaned_df[metric] != 0)
    return cleaned_df


def calculate_statistics(plot_df: pd.DataFrame, x_column: str, y_column: str):
    stats_df = plot_df[[x_column, y_column]].dropna().copy()
    if stats_df.empty:
        empty_df = pd.DataFrame()
        return empty_df, empty_df, empty_df

    grouped = (
        stats_df.groupby(x_column, sort=False)[y_column]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .reset_index()
        .rename(
            columns={
                "count": "n",
                "std": "sd",
            }
        )
    )
    grouped["mean +/- sd"] = grouped.apply(
        lambda row: f"{row['mean']:.2f} +/- {row['sd']:.2f}"
        if pd.notna(row["sd"])
        else f"{row['mean']:.2f} +/- NA",
        axis=1,
    )
    grouped["parameter"] = y_column
    grouped["description"] = "Descriptive statistics for each group"

    grouped_data = {
        name: group[y_column].dropna().values
        for name, group in stats_df.groupby(x_column, sort=False)
    }
    group_names = list(grouped_data.keys())
    groups = list(grouped_data.values())

    overall_rows = []
    if len(groups) >= 2:
        statistic, p_value = stats.kruskal(*groups)
        overall_rows.append(
            {
                "parameter": y_column,
                "comparison_scope": "all groups",
                "test": "Kruskal-Wallis",
                "statistic": statistic,
                "p_value": p_value,
                "significance": p_to_stars(p_value),
                "description": "Overall nonparametric comparison across all groups",
            }
        )
    overall_df = pd.DataFrame(overall_rows)

    pairwise_rows = []
    for group_1, group_2 in itertools.combinations(group_names, 2):
        values_1 = grouped_data[group_1]
        values_2 = grouped_data[group_2]
        statistic, p_raw = stats.mannwhitneyu(values_1, values_2, alternative="two-sided")
        pairwise_rows.append(
            {
                "parameter": y_column,
                "group_1": group_1,
                "group_2": group_2,
                "test": "Mann-Whitney U",
                "statistic": statistic,
                "p_raw": p_raw,
            }
        )

    pairwise_df = pd.DataFrame(pairwise_rows)
    if not pairwise_df.empty:
        reject, p_adj, _, _ = multipletests(pairwise_df["p_raw"], method="fdr_bh")
        pairwise_df["p_adj_fdr_bh"] = p_adj
        pairwise_df["significant_after_fdr"] = reject
        pairwise_df["significance_raw"] = pairwise_df["p_raw"].apply(p_to_stars)
        pairwise_df["significance_adj"] = pairwise_df["p_adj_fdr_bh"].apply(p_to_stars)
        pairwise_df["description"] = "Pairwise nonparametric comparison with FDR correction"

    return grouped, overall_df, pairwise_df


def build_summary_df(y_column: str):
    summary_rows = [
        ["Sheet used", "plot"],
        ["Grouping column", "name"],
        ["Measured column", y_column],
        ["Overall test", "Kruskal-Wallis"],
        ["Pairwise test", "Mann-Whitney U"],
        ["Multiple testing correction", "FDR Benjamini-Hochberg"],
        ["Meaning of significance labels", "ns: p>=0.05, *: p<0.05, **: p<0.01, ***: p<0.001"],
        ["Descriptive stats note", "mean +/- sd shown on chart and in descriptive_stats"],
        ["Pairwise stats note", "p_raw is uncorrected p-value, p_adj_fdr_bh is corrected p-value"],
    ]
    return pd.DataFrame(summary_rows, columns=["field", "value"])


def build_scatter_plot(
    plot_df: pd.DataFrame,
    grouped: pd.DataFrame,
    overall_df: pd.DataFrame,
    pairwise_annotations,
    x_column: str,
    y_column: str,
    title: str,
    x_axis_label: str,
    y_axis_label: str,
    palette,
    chart_type: str,
    point_alpha: float,
    box_alpha: float,
    jitter_amount: float,
    show_mean_sd_labels: bool,
    group_label_map,
):
    stats_df = plot_df[[x_column, y_column]].dropna().copy()
    order = grouped[x_column].tolist()
    group_count = len(order)
    display_labels = [group_label_map.get(group_name, group_name) for group_name in order]

    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7))

    if chart_type == "Box plot":
        sns.boxplot(
            data=stats_df,
            x=x_column,
            y=y_column,
            order=order,
            palette=palette[: len(order)],
            width=0.5,
            fliersize=0,
            linewidth=1.6,
            boxprops={"alpha": box_alpha},
            ax=ax,
        )
    elif chart_type == "Violin plot":
        sns.violinplot(
            data=stats_df,
            x=x_column,
            y=y_column,
            order=order,
            palette=palette[: len(order)],
            cut=0,
            inner=None,
            linewidth=0,
            saturation=1,
            ax=ax,
        )
        for collection in ax.collections:
            collection.set_alpha(box_alpha)
    else:
        sns.stripplot(
            data=stats_df,
            x=x_column,
            y=y_column,
            hue=x_column,
            order=order,
            hue_order=order,
            palette=palette[: len(order)],
            jitter=jitter_amount,
            alpha=point_alpha,
            size=7,
            linewidth=0,
            dodge=False,
            ax=ax,
        )

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()

    labels = grouped["mean +/- sd"].tolist()
    for i, row in grouped.iterrows():
        mean_value = row["mean"]
        sd_value = row["sd"]

        if pd.notna(sd_value):
            ax.errorbar(
                x=i,
                y=mean_value,
                yerr=sd_value,
                fmt="none",
                ecolor="black",
                elinewidth=2,
                capsize=6,
                capthick=2,
                zorder=10,
            )

        ax.hlines(
            y=mean_value,
            xmin=i - 0.18,
            xmax=i + 0.18,
            colors="black",
            linewidth=2.2,
            zorder=11,
        )

        if show_mean_sd_labels:
            ax.text(
                i,
                -0.18,
                labels[i],
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=10,
                fontweight="bold",
                color="black",
            )

    if not overall_df.empty and group_count == 2 and not pairwise_annotations:
        p_value = overall_df.loc[0, "p_value"]
        ax.text(
            0.5,
            0.95,
            format_p_value(p_value),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            color="red",
        )

    y_values = stats_df[y_column]
    annotation_count = len(pairwise_annotations)
    y_min = None
    y_max = None
    if not y_values.empty:
        y_min = float(y_values.min())
        y_max = float(y_values.max())
        y_range = y_max - y_min
        if y_range == 0:
            padding = max(abs(y_max) * 0.35, 1.0)
        else:
            padding = y_range * 0.6
        lower = max(0.0, y_min - padding * 0.65)
        upper = y_max + padding + (annotation_count * max(y_range * 0.12, 0.6))
        if lower == upper:
            upper = lower + 1.0
        ax.set_ylim(lower, upper)

    if pairwise_annotations and y_max is not None:
        base_range = max((y_max - y_min) if y_min is not None else 0.0, 1.0)
        bracket_height = base_range * 0.05
        step_height = base_range * 0.12
        start_y = y_max + base_range * 0.08
        x_positions = {group_name: index for index, group_name in enumerate(order)}

        for idx, annotation in enumerate(pairwise_annotations):
            left_name = annotation["group_1"]
            right_name = annotation["group_2"]
            if left_name not in x_positions or right_name not in x_positions:
                continue

            left_x = x_positions[left_name]
            right_x = x_positions[right_name]
            if left_x > right_x:
                left_x, right_x = right_x, left_x

            y_level = start_y + idx * step_height
            ax.plot(
                [left_x, left_x, right_x, right_x],
                [y_level, y_level + bracket_height, y_level + bracket_height, y_level],
                lw=1.6,
                c="black",
            )
            ax.text(
                (left_x + right_x) / 2,
                y_level + bracket_height + base_range * 0.02,
                format_p_value(annotation["p_value"]),
                ha="center",
                va="bottom",
                fontsize=10,
                color="black",
            )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=16)
    ax.set_xlabel(x_axis_label, fontsize=12)
    ax.set_ylabel(y_axis_label, fontsize=13)
    ax.set_xticklabels(display_labels)
    ax.tick_params(axis="x", rotation=15, labelsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    ax.grid(axis="x", visible=False)

    sns.despine(offset=10, trim=True)
    plt.subplots_adjust(bottom=0.28 if show_mean_sd_labels else 0.14)
    return fig


def figure_to_png_bytes(fig) -> BytesIO:
    output = BytesIO()
    fig.savefig(output, format="png", dpi=300, bbox_inches="tight")
    output.seek(0)
    return output


def figure_to_pdf_bytes(fig) -> BytesIO:
    output = BytesIO()
    fig.savefig(output, format="pdf", bbox_inches="tight")
    output.seek(0)
    return output


def build_excel_export(
    group_tables,
    plot_df,
    export_options,
    stats_sheets,
):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        if export_options["include_all_groups_sheet"]:
            sheet_name = "All_Groups"
            start_col = 0

            for base_name in sorted(group_tables.keys()):
                table = group_tables[base_name].reset_index()

                pd.DataFrame([[base_name]]).to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    header=False,
                    startrow=0,
                    startcol=start_col,
                )

                table.to_excel(
                    writer,
                    sheet_name=sheet_name,
                    index=False,
                    startrow=1,
                    startcol=start_col,
                )

                start_col += table.shape[1] + 3

        if export_options["include_plot_sheet"]:
            plot_df.to_excel(writer, sheet_name="plot", index=False)

        if export_options["include_summary_sheet"] and not stats_sheets["summary"].empty:
            stats_sheets["summary"].to_excel(writer, sheet_name="summary", index=False)

        if export_options["include_descriptive_sheet"] and not stats_sheets["descriptive"].empty:
            stats_sheets["descriptive"].to_excel(writer, sheet_name="descriptive_stats", index=False)

        if export_options["include_overall_sheet"] and not stats_sheets["overall"].empty:
            stats_sheets["overall"].to_excel(writer, sheet_name="overall_test", index=False)

        if export_options["include_pairwise_sheet"] and not stats_sheets["pairwise"].empty:
            stats_sheets["pairwise"].to_excel(writer, sheet_name="pairwise_stats", index=False)

    output.seek(0)
    return output


def read_tiff_array(uploaded_file):
    return tifffile.imread(BytesIO(uploaded_file.getvalue()))


def extract_scalar_plane(image_array: np.ndarray):
    image = np.asarray(image_array)
    if image.ndim == 2:
        scalar = image.astype(float)
    elif image.ndim == 3:
        if image.shape[-1] in (3, 4):
            scalar = image[..., :3].astype(float).mean(axis=2)
        else:
            scalar = image[0].astype(float)
    else:
        scalar = image.squeeze().astype(float)
        if scalar.ndim > 2:
            scalar = scalar[0]

    return np.nan_to_num(scalar, nan=0.0)


def to_display_image(image_array: np.ndarray):
    display = extract_scalar_plane(image_array)
    low, high = np.percentile(display, [1, 99]) if np.any(display) else (0.0, 1.0)
    if high <= low:
        high = low + 1.0
    normalized = np.clip((display - low) / (high - low), 0, 1)
    return normalized


@st.cache_resource(show_spinner=False)
def get_cellpose_model():
    from cellpose import models

    return models.Cellpose(model_type="cyto")


def exclude_nucleus_like_regions(labels: np.ndarray, smoothed_image: np.ndarray, min_cell_area: int, nucleus_darkness: float):
    cell_mask = labels > 0
    cytoplasm_mask = np.zeros_like(cell_mask, dtype=bool)

    for region in measure.regionprops(labels, intensity_image=smoothed_image):
        if region.area < min_cell_area:
            continue

        region_mask = labels == region.label
        region_values = smoothed_image[region_mask]
        dark_threshold = np.quantile(region_values, nucleus_darkness)
        nucleus_candidate = region_mask & (smoothed_image <= dark_threshold)
        nucleus_candidate = morphology.remove_small_objects(nucleus_candidate, min_size=max(region.area // 20, 20))
        nucleus_candidate = morphology.binary_opening(nucleus_candidate, morphology.disk(2))

        cytoplasm_region = region_mask & ~nucleus_candidate
        cytoplasm_region = morphology.remove_small_objects(cytoplasm_region, min_size=max(region.area // 3, 30))

        if cytoplasm_region.sum() == 0:
            continue

        cytoplasm_mask |= cytoplasm_region

    filtered_labels = measure.label(cytoplasm_mask)
    return cell_mask, cytoplasm_mask, filtered_labels


def segment_monoculture_cells_classical(display_image: np.ndarray, min_cell_area: int, nucleus_darkness: float):
    smoothed = filters.gaussian(display_image, sigma=1.0)
    threshold = filters.threshold_otsu(smoothed)
    cell_mask = smoothed > threshold
    cell_mask = morphology.remove_small_objects(cell_mask, min_size=min_cell_area)
    cell_mask = morphology.remove_small_holes(cell_mask, area_threshold=max(min_cell_area // 2, 64))
    cell_mask = segmentation.clear_border(cell_mask)

    labels = measure.label(cell_mask)
    return exclude_nucleus_like_regions(labels, smoothed, min_cell_area, nucleus_darkness)


def segment_monoculture_cells_cellpose(
    display_image: np.ndarray,
    min_cell_area: int,
    nucleus_darkness: float,
    cellpose_diameter: float,
):
    smoothed = filters.gaussian(display_image, sigma=1.0)
    model = get_cellpose_model()
    eval_image = (smoothed * 255).astype(np.uint8)
    masks, _, _, _ = model.eval(
        [eval_image],
        channels=[0, 0],
        diameter=cellpose_diameter if cellpose_diameter > 0 else None,
        do_3D=False,
    )
    labels = np.asarray(masks[0], dtype=np.int32)
    if labels.max() == 0:
        return np.zeros_like(display_image, dtype=bool), np.zeros_like(display_image, dtype=bool), labels

    # Clean tiny or border-touching regions before nucleus exclusion.
    cleaned_mask = morphology.remove_small_objects(labels > 0, min_size=min_cell_area)
    cleaned_mask = segmentation.clear_border(cleaned_mask)
    labels = measure.label(cleaned_mask)
    return exclude_nucleus_like_regions(labels, smoothed, min_cell_area, nucleus_darkness)


def build_segmentation_overlay(display_image: np.ndarray, cell_mask: np.ndarray, cytoplasm_mask: np.ndarray, labels: np.ndarray):
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(display_image, cmap="gray")
    ax.contour(cell_mask.astype(float), levels=[0.5], colors=["#ff89b5"], linewidths=1.2)
    ax.contour(cytoplasm_mask.astype(float), levels=[0.5], colors=["#18c37e"], linewidths=1.0)

    for region in measure.regionprops(labels):
        y_pos, x_pos = region.centroid
        ax.text(x_pos, y_pos, str(region.label), color="white", fontsize=8, ha="center", va="center")

    ax.set_axis_off()
    ax.set_title("Segmentation preview", fontsize=12)
    fig.tight_layout()
    return fig


def build_image_analysis_table(labels: np.ndarray, display_image: np.ndarray, metric_maps):
    rows = []
    for region in measure.regionprops(labels, intensity_image=display_image):
        mask = labels == region.label
        row = {
            "name": "TIFF_monoculture",
            "ROI": int(region.label),
            "area_px": int(region.area),
            "mean_intensity": float(display_image[mask].mean()),
        }

        for metric_name in wanted:
            metric_image = metric_maps.get(metric_name)
            if metric_image is not None:
                row[metric_name] = float(metric_image[mask].mean())
            else:
                row[metric_name] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def render_tiff_monoculture_analysis(export_basename: str):
    st.subheader("TIFF monoculture MVP")
    st.caption(
        "Test pipeline for cell monolayer TIFF images: segment cells, exclude nucleus-like dark regions, "
        "preview selected cytoplasm, and optionally compute a1/t1/t2/tm from uploaded parameter maps."
    )

    settings_col, upload_col = st.columns([1, 1.2])
    with settings_col:
        segmentation_engine = st.radio(
            "Segmentation engine",
            options=["Cellpose", "Classical thresholding"],
            index=0,
            horizontal=True,
        )
        min_cell_area = st.slider("Minimum cell area (px)", min_value=150, max_value=5000, value=900, step=50)
        nucleus_darkness = st.slider("Nucleus exclusion strength", min_value=0.05, max_value=0.40, value=0.18, step=0.01)
        cellpose_diameter = st.slider("Cellpose diameter", min_value=0, max_value=200, value=80, step=5)
        st.caption("Lower values exclude only the darkest inner pixels. Higher values carve out larger dark zones.")

    with upload_col:
        base_tiff = st.file_uploader("Segmentation source TIFF", type=["tif", "tiff"], key="base_tiff_uploader")
        st.caption("Optional maps can drive a1, t1, t2, tm extraction per selected cell.")
        metric_uploads = {}
        metric_cols = st.columns(2)
        for index, metric in enumerate(wanted):
            with metric_cols[index % 2]:
                metric_uploads[metric] = st.file_uploader(
                    f"{metric} map TIFF (optional)",
                    type=["tif", "tiff"],
                    key=f"{metric}_map_uploader",
                )

    if not base_tiff:
        st.info("Upload a segmentation source TIFF to preview the monoculture pipeline.")
        return

    try:
        base_image = read_tiff_array(base_tiff)
    except Exception as exc:
        st.error(f"Could not read TIFF: {exc}")
        return

    display_image = to_display_image(base_image)

    metric_maps = {}
    for metric, metric_file in metric_uploads.items():
        if metric_file is None:
            continue
        try:
            metric_map = extract_scalar_plane(read_tiff_array(metric_file))
            if metric_map.shape != display_image.shape:
                st.warning(f"{metric} map shape {metric_map.shape} does not match source TIFF shape {display_image.shape}.")
                continue
            metric_maps[metric] = metric_map
        except Exception as exc:
            st.warning(f"Could not read {metric} map: {exc}")

    try:
        if segmentation_engine == "Cellpose":
            cell_mask, cytoplasm_mask, labels = segment_monoculture_cells_cellpose(
                display_image,
                min_cell_area,
                nucleus_darkness,
                cellpose_diameter,
            )
        else:
            cell_mask, cytoplasm_mask, labels = segment_monoculture_cells_classical(
                display_image,
                min_cell_area,
                nucleus_darkness,
            )
    except Exception as exc:
        st.error(f"Segmentation failed: {exc}")
        st.info(
            "If this happened with Cellpose, the most likely reason is that the dependency is not installed yet "
            "or the runtime does not support it."
        )
        return

    overlay_figure = build_segmentation_overlay(display_image, cell_mask, cytoplasm_mask, labels)
    overlay_pdf = figure_to_pdf_bytes(overlay_figure)

    preview_col_1, preview_col_2 = st.columns(2)
    with preview_col_1:
        st.markdown("**Source image**")
        st.image(display_image, clamp=True, use_container_width=True)
    with preview_col_2:
        st.markdown("**Selected cells and cytoplasm mask**")
        st.pyplot(overlay_figure, use_container_width=True)

    results_df = build_image_analysis_table(labels, display_image, metric_maps)
    plt.close(overlay_figure)

    if results_df.empty:
        st.warning("No cells passed the current segmentation settings.")
        return

    stats_ready_columns = ["name", "ROI"] + wanted
    stats_ready_df = results_df[stats_ready_columns].copy()
    stats_ready_df = remove_zero_values(stats_ready_df, wanted)

    st.markdown("**Segmentation results**")
    st.dataframe(results_df, use_container_width=True)

    if any(metric in metric_maps for metric in wanted):
        available_metrics = [metric for metric in wanted if stats_ready_df[metric].notna().any()]
        if available_metrics:
            st.markdown("**Stats-ready preview**")
            st.dataframe(stats_ready_df[["name", "ROI"] + available_metrics], use_container_width=True)
            st.info(
                "Metric maps were provided, so these selected cells can already flow into the same statistics pipeline "
                "used by notebook analysis."
            )
        else:
            st.info("Metric map files were uploaded, but no usable values were extracted from the current masks.")
    else:
        st.info(
            "This test version already segments cells and excludes nucleus-like regions. "
            "To populate a1, t1, t2, tm, upload the corresponding parameter map TIFFs."
        )

    export_name = sanitize_filename(export_basename)
    export_buffer = BytesIO()
    with pd.ExcelWriter(export_buffer, engine="openpyxl") as writer:
        results_df.to_excel(writer, sheet_name="image_analysis", index=False)
        stats_ready_df.to_excel(writer, sheet_name="plot", index=False)
    export_buffer.seek(0)

    download_col_1, download_col_2 = st.columns(2)
    with download_col_1:
        st.download_button(
            label="Download image analysis Excel",
            data=export_buffer,
            file_name=f"{export_name}_tiff_monoculture.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_col_2:
        st.download_button(
            label="Download segmentation overlay PDF",
            data=overlay_pdf,
            file_name=f"{export_name}_tiff_overlay.pdf",
            mime="application/pdf",
        )


def render_image_analysis_placeholder():
    st.subheader("FLIM Image Analysis")
    st.caption(
        "This workspace will grow into the raw-image pipeline: start with TIFF analysis for "
        "monoculture or tissue, then add direct SDT support and Cellpose-based segmentation."
    )

    mode_col, input_col = st.columns(2)
    with mode_col:
        sample_type = st.radio(
            "Sample type",
            options=["Cell monolayer", "Tissue section"],
            index=0,
            horizontal=True,
            disabled=True,
        )
    with input_col:
        input_type = st.radio(
            "Input type",
            options=["TIFF workflow", "SDT workflow"],
            index=0,
            horizontal=True,
            disabled=True,
        )

    st.info(
        f"Planned first milestone: `{input_type}` for `{sample_type}` with cytoplasm masks, "
        "nucleus exclusion, and export into the same statistics module used by processed data."
    )

    roadmap_col_1, roadmap_col_2 = st.columns(2)
    with roadmap_col_1:
        st.markdown("**Phase 1**")
        st.write("Upload TIFF maps or exported SPCImage images.")
        st.write("Preview image and basic channels.")
        st.write("Segment monoculture or tissue regions.")
        st.write("Extract FLIM values into a table for the current stats/plot engine.")

    with roadmap_col_2:
        st.markdown("**Phase 2**")
        st.write("Add direct SDT reading when the format path is validated.")
        st.write("Use Cellpose for cell segmentation and cytoplasm masking.")
        st.write("Add coculture mode with exclusion of brightly labeled cells.")
        st.write("Keep manual QC overlays before export.")

    st.markdown("**Planned controls**")
    planned_controls = pd.DataFrame(
        [
            {"future_block": "Input", "planned_options": "TIFF, SDT"},
            {"future_block": "Sample type", "planned_options": "Cell monolayer, tissue section"},
            {"future_block": "Culture type", "planned_options": "Monoculture first, coculture later"},
            {"future_block": "Segmentation engine", "planned_options": "Thresholding, Cellpose"},
            {"future_block": "Output", "planned_options": "Mask preview, ROI table, stats-ready export"},
        ]
    )
    st.dataframe(planned_controls, use_container_width=True, hide_index=True)


def render_sdt_workflow_placeholder(export_basename: str):
    st.subheader("SDT workflow (scaffold)")
    st.caption("SDT parsing/processing is not implemented yet. TIFF workflow is disabled to keep the app fast and stable.")

    st.info(
        "Когда SDT-парсер будет готов, он будет возвращать те же метрики (tm/t1/t2/a1), "
        "что сейчас использует модуль статистики и построения графиков."
    )

    uploaded_files = st.file_uploader(
        "Upload SDT files",
        type=["sdt", "SDT"],
        accept_multiple_files=True,
        key="sdt_uploader",
    )

    if uploaded_files:
        st.success(f"Received {len(uploaded_files)} file(s). Processing is not enabled yet.")
        with st.expander("Received file names"):
            for f in uploaded_files:
                st.write(f.name)

    st.divider()
    st.markdown("**Export**")
    st.caption("Сейчас экспорт отключен, т.к. обработка SDT не реализована.")


def render_processed_analysis(export_basename: str):
    uploaded_files = st.file_uploader(
        "Drop .asc files here",
        type=["asc"],
        accept_multiple_files=True,
        key="processed_uploader",
    )

    if not uploaded_files:
        st.info("Upload one or more .asc files to start.")
        return

    groups = {}
    skipped = []
    logs = []

    for uploaded in uploaded_files:
        filename = uploaded.name
        match = name_pattern.match(filename)
        if not match:
            skipped.append(f"{filename} (name does not match required pattern)")
            continue

        base = match.group("base")
        num = int(match.group("num"))

        rows, used_encoding, err = parse_uploaded_file(uploaded)
        if err:
            skipped.append(f"{filename} ({err})")
            continue

        logs.append(f"{filename}: {used_encoding}")
        for roi, metric, mu in rows:
            groups.setdefault(base, []).append([num, filename, roi, metric, mu])

    if logs:
        with st.expander("Read log"):
            for line in logs:
                st.write(line)

    if skipped:
        st.warning("Some files were skipped:")
        for item in skipped:
            st.write(f"- {item}")

    if not groups:
        st.error("No valid data found.")
        return

    st.success(f"Groups found: {len(groups)}")

    group_tables = {}
    for base_name in sorted(groups.keys()):
        df = pd.DataFrame(
            groups[base_name],
            columns=["FileNum", "File", "ROI", "type", "mu"],
        )

        table = df.pivot_table(
            index=["FileNum", "ROI"],
            columns="type",
            values="mu",
            aggfunc="first",
        ).round(3)

        group_tables[base_name] = table

    plot_df = build_plot_df(group_tables)
    plot_metrics = [metric for metric in wanted if metric in plot_df.columns]
    plot_df = remove_zero_values(plot_df, plot_metrics)

    st.subheader("Preview")
    preview_group = st.selectbox("Select group", list(group_tables.keys()))
    st.dataframe(group_tables[preview_group], use_container_width=True)

    st.subheader("Plot data")
    st.dataframe(plot_df, use_container_width=True)

    if not plot_metrics:
        st.warning("No metric columns were parsed for statistics or scatter plot.")
        return

    st.subheader("Statistics and chart")
    controls_col, sheet_col = st.columns([1.2, 1])

    with controls_col:
        selected_metric = st.selectbox("Metric for stats and chart", plot_metrics, index=0)
        chart_type = st.radio(
            "Chart type",
            options=["Scatter plot", "Box plot", "Violin plot"],
            index=0,
            horizontal=True,
        )
        default_title = f"{selected_metric} across groups"
        chart_title = st.text_input("Chart title", value=default_title)
        x_axis_label = st.text_input("X axis label", value="Cell type")
        y_axis_label = st.text_input("Y axis label", value=selected_metric)
        point_alpha = st.slider("Point opacity", min_value=0.10, max_value=1.00, value=0.85, step=0.05)
        box_alpha = st.slider("Box opacity", min_value=0.10, max_value=1.00, value=0.55, step=0.05)
        jitter_amount = st.slider("Point spread", min_value=0.00, max_value=0.50, value=0.22, step=0.01)
        show_mean_sd_labels = st.checkbox("Show mean +/- sd under chart", value=True)

    with sheet_col:
        include_all_groups_sheet = st.checkbox("Include All_Groups sheet", value=True)
        include_plot_sheet = st.checkbox("Include plot sheet", value=True)
        include_summary_sheet = st.checkbox("Include summary sheet", value=True)
        include_descriptive_sheet = st.checkbox("Include descriptive_stats sheet", value=True)
        include_overall_sheet = st.checkbox("Include overall_test sheet", value=True)
        include_pairwise_sheet = st.checkbox("Include pairwise_stats sheet", value=True)

    grouped_stats, overall_df, pairwise_df = calculate_statistics(plot_df, "name", selected_metric)
    summary_df = build_summary_df(selected_metric)
    group_label_map = {}

    with st.expander("Group labels on chart"):
        label_columns = st.columns(min(len(group_tables), 2) or 1)
        for index, group_name in enumerate(group_tables.keys()):
            with label_columns[index % len(label_columns)]:
                group_label_map[group_name] = st.text_input(
                    f"Label for {group_name}",
                    value=group_name,
                    key=f"label_{group_name}",
                )

    if grouped_stats.empty:
        st.warning("All values for this metric are zero or missing, so the app skipped stats and charting.")
        plot_pdf = None
    else:
        st.markdown("**Descriptive statistics**")
        st.dataframe(grouped_stats, use_container_width=True)

        if not overall_df.empty:
            st.markdown("**Overall test**")
            st.dataframe(overall_df, use_container_width=True)

        if not pairwise_df.empty:
            st.markdown("**Pairwise comparisons**")
            st.dataframe(pairwise_df, use_container_width=True)

        pairwise_options = build_pairwise_options(pairwise_df)
        selected_pairs = []
        if pairwise_options:
            selected_pairs = st.multiselect(
                "Pairwise brackets on chart",
                options=pairwise_options,
                help="Selected comparisons will be shown as brackets with adjusted p-values when available.",
            )
        pairwise_annotations = get_pairwise_annotation_rows(pairwise_df, selected_pairs)

        st.markdown("**Chart builder**")
        palette_columns = st.columns(min(len(group_tables), 4) or 1)
        palette = []
        for index, group_name in enumerate(group_tables.keys()):
            picker_column = palette_columns[index % len(palette_columns)]
            default_palette = [
                "#081f5c",
                "#334eac",
                "#488db4",
                "#780000",
                "#c1121f",
                "#022e17",
                "#88a33b",
                "#7a4cc2",
            ]
            with picker_column:
                palette.append(
                    st.color_picker(
                        f"{group_name} color",
                        value=default_palette[index % len(default_palette)],
                        key=f"color_{group_name}",
                    )
                )

        figure = build_scatter_plot(
            plot_df=plot_df,
            grouped=grouped_stats,
            overall_df=overall_df,
            pairwise_annotations=pairwise_annotations,
            x_column="name",
            y_column=selected_metric,
            title=chart_title,
            x_axis_label=x_axis_label,
            y_axis_label=y_axis_label,
            palette=palette,
            chart_type=chart_type,
            point_alpha=point_alpha,
            box_alpha=box_alpha,
            jitter_amount=jitter_amount,
            show_mean_sd_labels=show_mean_sd_labels,
            group_label_map=group_label_map,
        )
        st.pyplot(figure, use_container_width=True)
        plot_pdf = figure_to_pdf_bytes(figure)
        plt.close(figure)

    export_name = sanitize_filename(export_basename)
    stats_sheets = {
        "summary": summary_df,
        "descriptive": grouped_stats,
        "overall": overall_df,
        "pairwise": pairwise_df,
    }
    export_options = {
        "include_all_groups_sheet": include_all_groups_sheet,
        "include_plot_sheet": include_plot_sheet,
        "include_summary_sheet": include_summary_sheet,
        "include_descriptive_sheet": include_descriptive_sheet,
        "include_overall_sheet": include_overall_sheet,
        "include_pairwise_sheet": include_pairwise_sheet,
    }

    excel_data = build_excel_export(
        group_tables=group_tables,
        plot_df=plot_df,
        export_options=export_options,
        stats_sheets=stats_sheets,
    )

    download_col_1, download_col_2 = st.columns(2)
    with download_col_1:
        st.download_button(
            label="Download Excel",
            data=excel_data,
            file_name=f"{export_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with download_col_2:
        if plot_pdf is not None:
            st.download_button(
                label="Download chart PDF",
                data=plot_pdf,
                file_name=f"{export_name}_{selected_metric}_{chart_type.lower().replace(' ', '_')}.pdf",
                mime="application/pdf",
            )


st.set_page_config(page_title="ASC Grouper", layout="wide")
st.title("ASC Grouper")
st.caption(
    "Upload .asc files with pattern name_number_statistic_all.asc. "
    "Build Excel tables, optional statistics sheets, and a customizable FLIM chart."
)

st.markdown(
    """
    <style>
    .flim-panel {
        padding: 1.2rem 1.2rem 0.5rem 1.2rem;
        border-radius: 14px;
        border: 1px solid rgba(212, 83, 126, 0.22);
        background: linear-gradient(180deg, rgba(255, 241, 246, 0.9), rgba(255, 249, 251, 0.96));
        margin-bottom: 0.85rem;
        min-height: 148px;
    }
    .flim-panel h3 {
        margin: 0;
        color: #5b2c40;
        font-size: 1.2rem;
    }
    .flim-panel p {
        margin-top: 0.45rem;
        margin-bottom: 0;
        color: #875a6a;
        font-size: 0.94rem;
    }
    .flim-panel-tag {
        display: inline-block;
        margin-top: 0.8rem;
        padding: 0.22rem 0.52rem;
        border-radius: 999px;
        background: rgba(212, 83, 126, 0.1);
        color: #9a4b6d;
        font-size: 0.8rem;
    }
    div[data-testid="stButton"] > button[kind="secondary"] {
        width: 100%;
        min-height: 3rem;
        border-radius: 12px;
        border: 1px solid rgba(212, 83, 126, 0.3);
        color: #5b2c40;
        background: linear-gradient(180deg, rgba(255, 236, 244, 0.95), rgba(255, 247, 251, 0.98));
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Export setup")
    export_basename = st.text_input("Output file name", value="flim_statistics")
    st.caption("The app adds the file extension automatically.")
    st.divider()
    st.subheader("How it works")
    st.write("1. Upload one or more `.asc` files.")
    st.write("2. Choose what goes into the Excel export.")
    st.write("3. Customize and download the chart if you want one.")
    st.divider()
    st.caption("Zero values are ignored in export, statistics, and charting.")

if "flim_mode" not in st.session_state:
    st.session_state.flim_mode = None

st.markdown("**Choose a FLIM workflow**")

if st.session_state.flim_mode is None:
    workflow_col_1, workflow_col_2 = st.columns(2)

    with workflow_col_1:
        st.markdown(
            """
            <div class="flim-panel">
                <h3>Notebook analysis</h3>
                <p>Upload processed FLIM notebooks, review parsed groups, build statistics, and export charts.</p>
                <div class="flim-panel-tag">ready now</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open notebook analysis", key="open_notebook_analysis", type="secondary"):
            st.session_state.flim_mode = "processed"
            st.rerun()

    with workflow_col_2:
        st.markdown(
            """
            <div class="flim-panel">
                <h3>SDT workflow</h3>
                <p>SDT inputs only (placeholder). TIFF pipeline is disabled to keep the app fast and stable.</p>
                <div class="flim-panel-tag">sdt scaffold</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open SDT workflow", key="open_sdt_workflow", type="secondary"):
            st.session_state.flim_mode = "sdt"
            st.rerun()

else:
    back_col, _ = st.columns([1, 8])
    with back_col:
        if st.button("Back", key="back_to_flim_choice"):
            st.session_state.flim_mode = None
            st.rerun()

    if st.session_state.flim_mode == "processed":
        st.markdown(
            """
            <div class="flim-panel">
                <h3>Notebook analysis</h3>
                <p>Processed FLIM tables, statistics, and export.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_processed_analysis(export_basename)
    else:
        st.markdown(
            """
            <div class="flim-panel">
                <h3>SDT workflow</h3>
                <p>SDT scaffold (TIFF workflow is disabled).</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        from SDT_workflow import render_sdt_workflow
        render_sdt_workflow(export_basename)
