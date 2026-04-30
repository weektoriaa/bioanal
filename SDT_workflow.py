# sdt_workflow.py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import tempfile, os
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import curve_fit
from scipy.ndimage import uniform_filter, binary_dilation, label
from cellpose import models as cellpose_models
import streamlit as st
import sdtfile

# ============================================================
# FITTING
# ============================================================

def biexp(t, a1, T1, a2, T2, bg):
    return a1 * np.exp(-t / T1) + a2 * np.exp(-t / T2) + bg

def fit_row(args):
    """Один поток = одна строка пикселей"""
    row_idx, row_data, times_local, n_photons_min = args
    results = []
    for j in range(row_data.shape[0]):
        curve    = row_data[j]
        peak_idx = np.argmax(curve)
        t_fit    = times_local[peak_idx:] - times_local[peak_idx]
        y_fit    = curve[peak_idx:]
        if y_fit.sum() < n_photons_min:
            results.append(None)
            continue
        try:
            popt, _ = curve_fit(
                biexp, t_fit, y_fit,
                p0=[50, 0.4, 50, 2.0, 0.1],
                bounds=([0, 0.1, 0, 0.5, 0], [np.inf, 2.0, np.inf, 5.0, 10]),
                maxfev=3000,
            )
            a1, T1, a2, T2, bg = popt
            a1_pct = a1 / (a1 + a2) * 100
            Tm     = (a1 * T1 + a2 * T2) / (a1 + a2)
            results.append((a1_pct, T1, T2, Tm))
        except Exception:
            results.append(None)
    return row_idx, results

# ============================================================
# ОСНОВНЫЕ ФУНКЦИИ
# ============================================================

def load_sdt(file_bytes):
    """Загружает .sdt из байтов"""
    with tempfile.NamedTemporaryFile(suffix='.sdt', delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        sdt   = sdtfile.SdtFile(tmp_path)
        data  = sdt.data[0].astype(np.float32)
        times = sdt.times[0] * 1e9
    finally:
        os.unlink(tmp_path)
    return data, times

def run_binning(data, bin_size=3):
    binned = uniform_filter(data, size=(bin_size, bin_size, 1)) * (bin_size ** 2)
    return binned

def run_fitting_parallel(binned, times, n_photons_min=50, progress_bar=None, status_text=None):
    h, w      = binned.shape[:2]
    args_list = [(i, binned[i], times, n_photons_min) for i in range(h)]

    map_a1 = np.full((h, w), np.nan)
    map_T1 = np.full((h, w), np.nan)
    map_T2 = np.full((h, w), np.nan)
    map_Tm = np.full((h, w), np.nan)

    completed = 0
   with ThreadPoolExecutor() as executor:
        futures = {executor.submit(fit_row, args): args[0] for args in args_list}
        for future in as_completed(futures):
            row_idx, row_results = future.result()
            for j, res in enumerate(row_results):
                if res:
                    map_a1[row_idx, j] = res[0]
                    map_T1[row_idx, j] = res[1]
                    map_T2[row_idx, j] = res[2]
                    map_Tm[row_idx, j] = res[3]
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(completed / h)
            if status_text is not None:
                status_text.text(f'Fitting: {completed}/{h} строк обработано...')

    intensity = binned.sum(axis=2)
    return map_a1, map_T1, map_T2, map_Tm, intensity

def normalize_image(img):
    p1, p99 = np.percentile(img, 1), np.percentile(img, 99)
    return np.clip((img - p1) / (p99 - p1), 0, 1) * 255

def run_cellpose(intensity):
    img_norm = normalize_image(intensity).astype(np.uint8)
    model    = cellpose_models.CellposeModel(model_type='nuclei', gpu=False)
    masks, _, _ = model.eval(
        img_norm, diameter=None, channels=[0, 0],
        flow_threshold=0.4, cellprob_threshold=0.0,
    )
    return masks, img_norm

def extract_per_cell(masks, intensity, map_a1, map_T1, map_T2, map_Tm, group_name, file_num):
    rows     = []
    n_cells  = masks.max()

    for cell_id in range(1, n_cells + 1):
        cell_mask = masks == cell_id
        if cell_mask.sum() < 20:
            continue

        cell_intensity    = intensity[cell_mask]
        nuclear_threshold = np.percentile(cell_intensity, 25)
        dark_mask         = cell_mask & (intensity < nuclear_threshold)
        labeled_dark, n_regions = label(dark_mask)

        if n_regions > 0:
            sizes   = [np.sum(labeled_dark == r) for r in range(1, n_regions + 1)]
            nucleus = labeled_dark == (np.argmax(sizes) + 1)
            nucleus = binary_dilation(nucleus, iterations=4)
        else:
            nucleus = np.zeros_like(cell_mask)

        cytoplasm = cell_mask & ~nucleus
        high      = np.percentile(cell_intensity, 95)
        cytoplasm = cytoplasm & (intensity < high)

        if cytoplasm.sum() < 20:
            continue

        a1 = np.nanmean(map_a1[cytoplasm])
        T1 = np.nanmean(map_T1[cytoplasm]) * 1000
        T2 = np.nanmean(map_T2[cytoplasm]) * 1000
        Tm = np.nanmean(map_Tm[cytoplasm]) * 1000

        if np.isnan(a1):
            continue

        rows.append({
            'name':    group_name,
            'FileNum': int(file_num),
            'ROI':     cell_id,
            'tm':      round(Tm, 2),
            't1':      round(T1, 2),
            't2':      round(T2, 2),
            'a1':      round(a1, 2),
        })

    return pd.DataFrame(rows)

def build_mask_figure(img_norm, masks, cytoplasm_display=True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(img_norm, cmap='hot')
    axes[0].set_title('Интенсивность НАДH')
    axes[0].axis('off')

    # Строим маску цитоплазмы для отображения
    cyto_display = np.zeros_like(masks)
    for cell_id in range(1, masks.max() + 1):
        cell_mask         = masks == cell_id
        cell_intensity    = img_norm[cell_mask] if img_norm is not None else None
        nuclear_threshold = np.percentile(img_norm[cell_mask], 25)
        dark_mask         = cell_mask & (img_norm < nuclear_threshold)
        labeled_dark, n  = label(dark_mask)
        if n > 0:
            sizes   = [np.sum(labeled_dark == r) for r in range(1, n + 1)]
            nucleus = labeled_dark == (np.argmax(sizes) + 1)
            nucleus = binary_dilation(nucleus, iterations=4)
        else:
            nucleus = np.zeros_like(cell_mask)
        cyto_display[cell_mask & ~nucleus] = cell_id

    axes[1].imshow(img_norm, cmap='hot')
    axes[1].imshow(
        np.where(cyto_display > 0, cyto_display, np.nan),
        cmap='tab20', alpha=0.45,
    )
    for cell_id in range(1, masks.max() + 1):
        m = cyto_display == cell_id
        if m.sum() == 0:
            continue
        ys, xs = np.where(m)
        axes[1].text(
            int(xs.mean()), int(ys.mean()), str(cell_id),
            color='white', fontsize=7, fontweight='bold',
            ha='center', va='center',
        )
    axes[1].set_title(f'Цитоплазма — {masks.max()} клеток')
    axes[1].axis('off')

    plt.tight_layout()
    return fig

def fig_to_pdf(fig):
    buf = BytesIO()
    fig.savefig(buf, format='pdf', bbox_inches='tight')
    buf.seek(0)
    return buf

def df_to_excel(df):
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='plot', index=False)
    buf.seek(0)
    return buf

# ============================================================
# STREAMLIT UI
# ============================================================

def render_sdt_workflow(export_basename: str):
    st.subheader("SDT workflow")
    st.caption(
        "Drag and drop .sdt file for automatical fitting, "
        "and export notebooks to analyse"
    )

    # Настройки
    col1, col2 = st.columns(2)
    with col1:
        group_name    = st.text_input("group name", value="tumour",
                                      help="example: tumour, CAFs, control")
        file_num      = st.number_input("file num", min_value=1, value=1, step=1)
        bin_size      = st.slider("Binning", 1, 7, 3, step=2,
                                  help="same binning from SPCImage")
    with col2:
        n_photons_min = st.slider("min photons", 20, 200, 50, step=10,
                                  help="no pixels above threshold")
        st.info("standart for NADPH - bin=3 min photons=50")

    uploaded = st.file_uploader(
        "drop your .sdt here", type=["sdt", "SDT"], key="sdt_main_uploader"
    )

    if not uploaded:
        st.info("drop .sdt to start")
        return

    st.success(f"uploaded: **{uploaded.name}**")

    if not st.button("▶ star analysis", type="primary"):
        return

    # ── Шаг 1: загрузка ──────────────────────────────────────
    with st.spinner("oh so beautiful let me see..."):
        data, times = load_sdt(uploaded.getvalue())
        binned      = run_binning(data, bin_size)
        del data
    st.success(f"✓ got it — {binned.shape[0]}×{binned.shape[1]} pixels")

    # ── Шаг 2: fitting ───────────────────────────────────────
    st.markdown("**now im fitting them**")
    st.caption("wait a bit")
    progress    = st.progress(0)
    status_text = st.empty()

    map_a1, map_T1, map_T2, map_Tm, intensity = run_fitting_parallel(
        binned, times, n_photons_min, progress, status_text
    )
    del binned
    status_text.empty()
    st.success("✓ Fitting is done!")

    # ── Шаг 3: Cellpose ──────────────────────────────────────
    with st.spinner("Cellpose is looking for your tiny shiny cells..."):
        masks, img_norm = run_cellpose(intensity)
    st.success(f"✓ we found: **{masks.max()}**")

    # ── Шаг 4: картинка с масками ────────────────────────────
    st.markdown("**i'll show you something**")
    fig = build_mask_figure(img_norm, masks)
    st.pyplot(fig, use_container_width=True)
    pdf_buf = fig_to_pdf(fig)
    plt.close(fig)

    # ── Шаг 5: параметры по клеткам ──────────────────────────
    with st.spinner("but first i need to calculate some a1, tm.. you know..."):
        df = extract_per_cell(
            masks, intensity, map_a1, map_T1, map_T2, map_Tm,
            group_name, file_num,
        )

    st.success(f"✓cells with data: **{len(df)}**")
    st.markdown("**results**")
    st.dataframe(df, use_container_width=True)

    # ── Шаг 6: экспорт ───────────────────────────────────────
    export_name = f"{export_basename}_{group_name}_file{int(file_num)}"
    excel_buf   = df_to_excel(df)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "📥 download exel",
            data=excel_buf,
            file_name=f"{export_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with dl2:
        st.download_button(
            "📥download masks PDF",
            data=pdf_buf,
            file_name=f"{export_name}_masks.pdf",
            mime="application/pdf",
        )

    st.info(
        "we got it! shine bright like a diamond!"
    )
