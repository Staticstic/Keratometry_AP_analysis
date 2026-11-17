import io
import os
import re
import zipfile
import warnings
import urllib.request

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy import stats
import streamlit as st
from statsmodels.formula.api import mixedlm

# ── 페이지 설정 ──────────────────────────────────────────────
st.set_page_config(page_title="Keratometry 분석 대시보드", page_icon="🔬", layout="wide")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans+KR:wght@300;400;600&family=IBM+Plex+Mono&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans KR', sans-serif; }
  .metric-card {
    background: #1c1f2e; border: 1px solid #2e3250;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 10px;
  }
  .metric-label { color: #8b92b8; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .metric-value { color: #e8eaf6; font-size: 1.4rem; font-weight: 600; font-family: 'IBM Plex Mono'; }
  .metric-sub   { color: #8b92b8; font-size: 0.72rem; margin-top: 2px; }
  .section-title {
    color: #7c83d0; font-size: 0.7rem; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 6px;
    border-bottom: 1px solid #2e3250; padding-bottom: 4px;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  상수
# ══════════════════════════════════════════════════════════════
DEV_T, DEV_P, DEV_P2 = "Test", "Predicate", "Predicate 2"
MEASURES = ["K1", "K2", "CYL", "AXIS", "J0", "J45"]
M_LABEL = {"K1": "K1", "K2": "K2", "CYL": "Cylinder", "AXIS": "Axis", "J0": "J0", "J45": "J45"}
M_UNIT = {"K1": "D", "K2": "D", "CYL": "D", "AXIS": "°", "J0": "D", "J45": "D"}
M_DEC = {"K1": 3, "K2": 3, "CYL": 3, "AXIS": 2, "J0": 3, "J45": 3}
CYL_CUTOFF = 0.75            # AXIS 분석 대상: |Predicate CYL| ≥ 0.75 D
ALL_GROUP = "전체"
ALL_EYES = "All eyes"         # Eye 컬럼이 없는 데이터
STUDY_EYE = "Study eye"       # 대상자별 지정 눈(OD 또는 OS 1안)

# PG — Agreement (LoA 95% CI 기준, ±)
BA_PG = {"K1": 1.0, "K2": 1.0, "CYL": 0.5, "AXIS": 15.0}
# PG — Precision (Repeatability CV% 기준)
PREC_PG = {"K1": 0.75, "K2": 0.75, "CYL": 22.5}

# 레이더 차트
RADAR_NAME = {"K1": "K1", "K2": "K2", "CYL": "Cylinder", "AXIS": "Axis"}
BA_RADAR_YLIM = 2.0
PREC_RADAR_YLIM = 3.0
PREC_RADAR_THR = {"K1": 0.75, "K2": 0.75, "Cylinder": 2.25}   # Cylinder는 CV/10 스케일

# 논문용 색상 (Okabe–Ito, 색각이상 친화)
DEV_COLOR = {DEV_T: "#000000", DEV_P: "#0072B2", DEV_P2: "#D55E00"}
PREV_COLORS = ["#D55E00", "#009E73"]
PG_COLOR = "#7F7F7F"

EYE_MAP = {"OD": "OD", "R": "OD", "RE": "OD", "RIGHT": "OD",
           "OS": "OS", "L": "OS", "LE": "OS", "LEFT": "OS"}
FIG_MIME = {"PNG": "image/png", "TIFF": "image/tiff", "PDF": "application/pdf", "SVG": "image/svg+xml"}

# ══════════════════════════════════════════════════════════════
#  논문 스타일 플롯 설정
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _korean_font():
    """한글 폰트 이름 반환 (범례 등에 한글이 들어갈 때 fallback 용)"""
    avail = {f.name for f in fm.fontManager.ttflist}
    for f in ["Malgun Gothic", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic", "Noto Sans CJK KR"]:
        if f in avail:
            return f
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "NanumGothic.ttf")
    try:
        if not os.path.exists(path):
            url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
            urllib.request.urlretrieve(url, path)
        fm.fontManager.addfont(path)
        return fm.FontProperties(fname=path).get_name()
    except Exception:
        return None


def apply_paper_style():
    avail = {f.name for f in fm.fontManager.ttflist}
    latin = next((f for f in ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"] if f in avail), "DejaVu Sans")
    families = [latin]
    kf = _korean_font()
    if kf:
        families.append(kf)
    plt.rcParams.update({
        "font.family": families,
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "black",
        "axes.labelcolor": "black",
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "xtick.color": "black",
        "ytick.color": "black",
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "axes.unicode_minus": True,
        "pdf.fonttype": 42,     # 저널 제출용: 텍스트를 편집 가능한 폰트로 저장
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def fig_bytes(fig, fmt, dpi):
    buf = io.BytesIO()
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if fmt == "TIFF" else {}
    fig.savefig(buf, format=fmt.lower(), dpi=dpi, bbox_inches="tight", facecolor="white", **kw)
    return buf.getvalue()


def show_fig(fig, fname, key, fmt, dpi, width=None):
    """플롯 표시 + 다운로드 버튼"""
    st.pyplot(fig, use_container_width=(width is None))
    st.download_button(f"⬇️ {fmt} ({dpi} dpi)", fig_bytes(fig, fmt, dpi),
                       f"{fname}.{fmt.lower()}", FIG_MIME[fmt], key=key)
    plt.close(fig)


def safe_name(s):
    return re.sub(r"[^0-9A-Za-z가-힣_+-]+", "_", str(s)).strip("_")

# ══════════════════════════════════════════════════════════════
#  데이터 로드 / 전처리
# ══════════════════════════════════════════════════════════════

def norm_pid(x):
    """PID 매칭용 키: 대소문자·구분자 무시, 숫자 앞 0 무시 (S001 = s-001 = s1)"""
    s = re.sub(r"[^0-9a-z]", "", str(x).strip().lower())
    return re.sub(r"\d+", lambda m: str(int(m.group())), s)


def natural_key(s):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(s))]


def _read_bytes(raw, fname):
    if fname.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))
    for enc in ("utf-8-sig", "cp949", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일 인코딩을 읽을 수 없습니다.")


def calc_jackson(df):
    if "CYL" in df.columns and "AXIS" in df.columns:
        df["J0"] = (-df["CYL"] / 2) * np.cos(2 * np.radians(df["AXIS"]))
        df["J45"] = (-df["CYL"] / 2) * np.sin(2 * np.radians(df["AXIS"]))
    return df


@st.cache_data(show_spinner=False)
def load_device(raw, fname, normalize=True, axis_offset=False):
    df = _read_bytes(raw, fname)
    df.columns = [str(c).strip().upper() for c in df.columns]
    if "PID" not in df.columns:
        raise ValueError(f"{fname}: PID 컬럼이 없습니다.")
    df = df[df["PID"].notna()].copy()
    df["PID"] = df["PID"].astype(str).str.strip()
    df["PID_KEY"] = df["PID"].map(norm_pid)
    if "EYE" in df.columns:
        df["EYE"] = df["EYE"].astype(str).str.strip().str.upper().map(lambda v: EYE_MAP.get(v, v))
    else:
        df["EYE"] = ALL_EYES
    if "SEQ" not in df.columns:
        df["SEQ"] = df.groupby(["PID_KEY", "EYE"]).cumcount() + 1
    df["SEQ"] = pd.to_numeric(df["SEQ"], errors="coerce")
    for c in ["K1", "K2", "CYL", "AXIS"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if normalize:
        # K1 = flat, K2 = steep
        if "K1" in df.columns and "K2" in df.columns:
            sw = df["K1"] > df["K2"]
            df.loc[sw, ["K1", "K2"]] = df.loc[sw, ["K2", "K1"]].values
        # 마이너스 실린더 표기로 통일 (plus → minus 전환)
        if "CYL" in df.columns and "AXIS" in df.columns:
            pos = df["CYL"] > 0
            df.loc[pos, "CYL"] = -df.loc[pos, "CYL"]
            df.loc[pos, "AXIS"] = df.loc[pos, "AXIS"] + 90
    if "AXIS" in df.columns:
        if axis_offset:
            df["AXIS"] = df["AXIS"] + 90
        df["AXIS"] = df["AXIS"] % 180

    df = calc_jackson(df)
    keep = ["PID", "PID_KEY", "EYE", "SEQ"] + [m for m in MEASURES if m in df.columns]
    return df[keep].sort_values(["PID_KEY", "EYE", "SEQ"]).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_group(raw, fname):
    g = _read_bytes(raw, fname)
    g.columns = [str(c).strip().upper() for c in g.columns]
    if "PID" not in g.columns:
        raise ValueError("그룹 파일에 PID 컬럼이 없습니다.")
    gcol = "GROUP" if "GROUP" in g.columns else next(
        (c for c in g.columns if c not in ("PID", "EYE") and _study_col([c]) is None), None)
    if gcol is None:
        return None     # GROUP 없음 (Study eye 정보만 있는 파일)
    g = g[g["PID"].notna() & g[gcol].notna()].copy()
    out = pd.DataFrame({"PID_KEY": g["PID"].map(norm_pid), "GROUP": g[gcol].astype(str).str.strip()})
    if "EYE" in g.columns:
        out["EYE"] = g["EYE"].astype(str).str.strip().str.upper().map(lambda v: EYE_MAP.get(v, v))
    return out.drop_duplicates()


def _study_col(cols):
    """'STUDY_EYE', 'STUDY EYE', 'STUDYEYE' 등 Study eye 컬럼 이름 찾기"""
    return next((c for c in cols if re.sub(r"[^A-Z]", "", str(c).upper()) in ("STUDYEYE", "STUDY")), None)


@st.cache_data(show_spinner=False)
def load_study_eye(raw, fname):
    """PID별 Study eye(OD/OS) → DataFrame[PID_KEY, STUDY_EYE]. 컬럼이 없으면 None"""
    s = _read_bytes(raw, fname)
    s.columns = [str(c).strip().upper() for c in s.columns]
    scol = _study_col(s.columns)
    if "PID" not in s.columns or scol is None:
        return None
    s = s[s["PID"].notna() & s[scol].notna()]
    out = pd.DataFrame({
        "PID_KEY": s["PID"].map(norm_pid),
        "STUDY_EYE": s[scol].astype(str).str.strip().str.upper().map(lambda v: EYE_MAP.get(v, v)),
    })
    out = out[out["STUDY_EYE"].isin(["OD", "OS"])]
    return out.drop_duplicates(subset=["PID_KEY"], keep="first")


def attach_study(df, sdf):
    df = df.drop(columns=["STUDY_EYE"], errors="ignore")
    if sdf is None:
        df["STUDY_EYE"] = np.nan
        return df
    return df.merge(sdf, on="PID_KEY", how="left")


def attach_group(df, gdf):
    df = df.drop(columns=["GROUP"], errors="ignore")
    if gdf is None:
        df["GROUP"] = np.nan
        return df
    keys = ["PID_KEY", "EYE"] if "EYE" in gdf.columns else ["PID_KEY"]
    return df.merge(gdf.drop_duplicates(subset=keys), on=keys, how="left")


def exclude(df, keys):
    return df[~df["PID_KEY"].isin(keys)] if keys else df


def circ_mean_axis(a):
    """축(0–180°) 원형 평균 — 2배각 사용 (예: 179°, 1° → 0°)"""
    a = np.asarray(a, float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return np.nan
    r = np.radians(2 * a)
    return (np.degrees(np.arctan2(np.sin(r).mean(), np.cos(r).mean())) / 2) % 180


def per_eye_value(df, method):
    """PID×EYE 당 1개 값: first = SEQ 최솟값 측정, mean = 반복 측정 평균"""
    keys = ["PID_KEY", "EYE"]
    if method == "first":
        idx = df.dropna(subset=["SEQ"]).groupby(keys)["SEQ"].idxmin()
        out = df.loc[idx].copy()
        out["N_REP"] = 1
        return out.reset_index(drop=True)
    lin = [m for m in ["K1", "K2", "CYL", "J0", "J45"] if m in df.columns]
    g = df.groupby(keys)
    out = g.agg(PID=("PID", "first"), N_REP=("SEQ", "count"), **{m: (m, "mean") for m in lin})
    if "AXIS" in df.columns:
        out["AXIS"] = g["AXIS"].apply(circ_mean_axis)
    if "GROUP" in df.columns:
        out["GROUP"] = g["GROUP"].first()
    return out.reset_index()


def build_agreement(t_df, p_df, method):
    t = per_eye_value(t_df, method)
    p = per_eye_value(p_df, method)
    vals = [c for c in MEASURES + ["N_REP"]]
    t = t[["PID", "PID_KEY", "EYE"] + [c for c in vals if c in t.columns]]
    p = p[["PID_KEY", "EYE"] + [c for c in vals if c in p.columns]]
    t = t.rename(columns={c: f"{c}_test" for c in vals})
    p = p.rename(columns={c: f"{c}_pred" for c in vals})
    m = t.merge(p, on=["PID_KEY", "EYE"], how="inner")
    if "AXIS_test" in m.columns and "AXIS_pred" in m.columns:
        d = m["AXIS_test"] - m["AXIS_pred"]
        m["AXIS_TRANS"] = m["AXIS_test"]
        m.loc[d > 90, "AXIS_TRANS"] = m["AXIS_test"] - 180
        m.loc[d < -90, "AXIS_TRANS"] = m["AXIS_test"] + 180
    return m


PAIR = {m: (f"{m}_test", f"{m}_pred") for m in MEASURES}
PAIR["AXIS"] = ("AXIS_TRANS", "AXIS_pred")


def subset_df(df, eye, group):
    d = df
    if eye in ("OD", "OS"):
        d = d[d["EYE"] == eye]
    elif eye == STUDY_EYE:
        d = d[d["EYE"] == d["STUDY_EYE"]]
    if group != ALL_GROUP:
        d = d[d["GROUP"] == group]
    return d


def subset_label(eye, group):
    return f"{eye} · {group}"


def axis_gap_check(t_df, p_df):
    """두 기기 AXIS 차이의 중앙값(°) — 약 90°면 기준 경선이 다를 가능성"""
    try:
        m = build_agreement(t_df, p_df, "first")
        m = m[m["CYL_pred"].abs() >= CYL_CUTOFF].dropna(subset=["AXIS_test", "AXIS_pred"])
        if len(m) < 5:
            return None
        d = ((m["AXIS_test"] - m["AXIS_pred"] + 90) % 180) - 90
        return float(np.median(np.abs(d)))
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════
#  Agreement (Bland-Altman) 통계
# ══════════════════════════════════════════════════════════════

def ccc(x, y):
    mx, my = np.mean(x), np.mean(y)
    sx, sy = np.std(x, ddof=1), np.std(y, ddof=1)
    r = np.corrcoef(x, y)[0, 1]
    return (2 * r * sx * sy) / (sx ** 2 + sy ** 2 + (mx - my) ** 2)


def pearson_ci(r, n, alpha=0.05):
    z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
    zc = stats.norm.ppf(1 - alpha / 2)
    return np.tanh(z - zc * se), np.tanh(z + zc * se)


def ba_data(d, m):
    x, y = PAIR[m]
    if x not in d.columns or y not in d.columns:
        return d.iloc[0:0]
    dd = d.dropna(subset=[x, y])
    if m == "AXIS":
        dd = dd[dd["CYL_pred"].abs() >= CYL_CUTOFF]
    return dd


def ba_stats(d, m):
    dd = ba_data(d, m)
    n = len(dd)
    if n < 3:
        return None
    x, y = PAIR[m]
    k = M_DEC[m]
    tv, pv = dd[x].values, dd[y].values
    diff = tv - pv
    md, sd = diff.mean(), diff.std(ddof=1)
    loa_u, loa_l = md + 1.96 * sd, md - 1.96 * sd
    ci = stats.t.interval(0.95, df=n - 1, loc=md, scale=sd / np.sqrt(n))
    se_loa = stats.t.ppf(0.975, df=n - 1) * sd * np.sqrt(3 / n)
    r, p_r = stats.pearsonr(tv, pv)
    r_lo, r_hi = pearson_ci(r, n)
    rd = lambda v: round(float(v), k)
    return {
        "Measure": M_LABEL[m], "Code": m, "Unit": M_UNIT[m], "N": n,
        "Test Mean": rd(tv.mean()), "Test SD": rd(tv.std(ddof=1)),
        "Test Min": rd(tv.min()), "Test Max": rd(tv.max()),
        "Predicate Mean": rd(pv.mean()), "Predicate SD": rd(pv.std(ddof=1)),
        "Predicate Min": rd(pv.min()), "Predicate Max": rd(pv.max()),
        "Mean Diff": rd(md), "SD Diff": rd(sd),
        "95% CI Lo": rd(ci[0]), "95% CI Hi": rd(ci[1]),
        "LoA Lower": rd(loa_l), "LoA Upper": rd(loa_u),
        "LoA Lower CI Lo": rd(loa_l - se_loa), "LoA Lower CI Hi": rd(loa_l + se_loa),
        "LoA Upper CI Lo": rd(loa_u - se_loa), "LoA Upper CI Hi": rd(loa_u + se_loa),
        "Pearson r": round(float(r), 3), "Pearson CI Lo": round(float(r_lo), 3),
        "Pearson CI Hi": round(float(r_hi), 3), "Pearson p": float(p_r),
        "CCC": round(float(ccc(tv, pv)), 3),
    }


def ba_pass(res):
    pg = BA_PG.get(res["Code"])
    if pg is None:
        return "—"
    return "PASS" if res["LoA Lower CI Lo"] >= -pg and res["LoA Upper CI Hi"] <= pg else "FAIL"


def ba_draw(ax, dd, m, res, title=None, show_ci=True, eye_marker=True, panel=None):
    """논문 스타일 Bland-Altman plot"""
    x, y = PAIR[m]
    diff = dd[x].values - dd[y].values
    avg = (dd[x].values + dd[y].values) / 2
    md, lu, ll = res["Mean Diff"], res["LoA Upper"], res["LoA Lower"]
    unit = M_UNIT[m]

    if show_ci:
        ax.axhspan(res["LoA Upper CI Lo"], res["LoA Upper CI Hi"], color="0.88", lw=0, zorder=0)
        ax.axhspan(res["LoA Lower CI Lo"], res["LoA Lower CI Hi"], color="0.88", lw=0, zorder=0)
        ax.axhspan(res["95% CI Lo"], res["95% CI Hi"], color="0.93", lw=0, zorder=0)

    eyes = dd["EYE"].values
    if eye_marker and set(eyes) >= {"OD", "OS"}:
        for eye, fc in [("OD", "black"), ("OS", "white")]:
            s = eyes == eye
            ax.scatter(avg[s], diff[s], s=14, facecolors=fc, edgecolors="black",
                       linewidths=0.6, zorder=3, label=eye)
        ax.legend(loc="upper left", handletextpad=0.1, borderaxespad=0.2, ncol=2, columnspacing=0.8)
    else:
        ax.scatter(avg, diff, s=14, facecolors="black", edgecolors="black", linewidths=0.6, zorder=3)

    ax.axhline(0, color="0.6", ls=":", lw=0.8, zorder=1)
    ax.axhline(md, color="black", ls="-", lw=1.0, zorder=2)
    ax.axhline(lu, color="black", ls="--", lw=0.9, zorder=2)
    ax.axhline(ll, color="black", ls="--", lw=0.9, zorder=2)

    span = max(abs(res["LoA Upper CI Hi"]), abs(res["LoA Lower CI Lo"]), np.abs(diff).max())
    lim = span * 1.3 if span > 0 else 1
    ax.set_ylim(-lim, lim)
    xr = avg.max() - avg.min() if len(avg) > 1 else 1
    ax.set_xlim(avg.min() - 0.05 * xr, avg.max() + 0.05 * xr)

    dec = 1 if m == "AXIS" else 2
    for yv, txt in [(lu, "+1.96 SD"), (md, "Mean"), (ll, "−1.96 SD")]:
        ax.annotate(f"{txt}\n{yv:.{dec}f}".replace("-", "−"), xy=(1.0, yv),
                    xycoords=("axes fraction", "data"), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7, annotation_clip=False)

    ax.set_xlabel(f"Mean of Test and Predicate ({unit})")
    ax.set_ylabel(f"Test − Predicate ({unit})")
    if title:
        ax.set_title(title, loc="left", fontweight="bold")
    if panel:
        ax.text(-0.18, 1.06, panel, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")


def ba_figure(dd, m, res, title, show_ci, eye_marker):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ba_draw(ax, dd, m, res, title=title, show_ci=show_ci, eye_marker=eye_marker)
    fig.tight_layout()
    return fig


def ba_multipanel(items, ncols, show_ci, eye_marker, suptitle=None):
    """items: [(dd, m, res)] → 패널 라벨(A, B, …)이 달린 다중 패널 그림"""
    n = len(items)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols, 3.1 * nrows), squeeze=False)
    for i, (dd, m, res) in enumerate(items):
        ba_draw(axes[i // ncols][i % ncols], dd, m, res, title=M_LABEL[m],
                show_ci=show_ci, eye_marker=eye_marker, panel=chr(65 + i))
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=9, x=0.01, ha="left")
    fig.tight_layout(w_pad=2.5, h_pad=1.5)
    return fig

# ══════════════════════════════════════════════════════════════
#  Precision (one-way random effect, statsmodels mixedlm)
# ══════════════════════════════════════════════════════════════

def anova_rep(df, cols, subj="SUBJ"):
    rows = []
    for c in cols:
        if c not in df.columns:
            continue
        d = df[[subj, c]].dropna().reset_index(drop=True)
        if d[subj].nunique() < 3 or len(d) < 3:
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = mixedlm(f"{c} ~ 1", d, groups=d[subj]).fit(reml=True, method="lbfgs", disp=False)
            res_var = fit.scale
            sub_var = float(fit.cov_re.iloc[0, 0])
            rep_sd = np.sqrt(res_var)
            mean_v = d[c].mean()
            icc = sub_var / (sub_var + res_var)
            n = d[subj].nunique()
            z = np.arctanh(np.clip(icc, -0.9999, 0.9999))
            z_se = 1 / np.sqrt(n - 3) if n > 3 else 0
            rows.append({
                "Measure": M_LABEL[c], "Code": c, "N (eyes)": n, "N (measurements)": len(d),
                "Mean": round(mean_v, 3), "SD": round(d[c].std(ddof=1), 3),
                "Repeatability_SD": round(rep_sd, 3),
                "Repeatability_limit": round(rep_sd * 2.8, 3),
                "Repeatability_CV": round(rep_sd / mean_v * 100, 3) if mean_v != 0 else np.nan,
                "ICC": round(icc, 3),
                "ICC_CI_lower": round(float(np.tanh(z - 1.96 * z_se)), 3),
                "ICC_CI_upper": round(float(np.tanh(z + 1.96 * z_se)), 3),
            })
        except Exception:
            continue
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def run_precision(t_df, p_df, cols):
    """{'Test': df, 'Predicate': df} 반환. AXIS는 |Predicate 첫 측정 CYL| ≥ 0.75 D 눈만 사용"""
    p_first = per_eye_value(p_df, "first")
    cyl_keys = set()
    if "CYL" in p_first.columns:
        sel = p_first[p_first["CYL"].abs() >= CYL_CUTOFF]
        cyl_keys = set(sel["PID_KEY"] + "|" + sel["EYE"])
    out = {}
    for label, d in [(DEV_T, t_df), (DEV_P, p_df)]:
        d = d.copy()
        d["SUBJ"] = d["PID_KEY"] + "|" + d["EYE"]
        if "CYL" in d.columns:
            d["CYL"] = d["CYL"].abs()
        if "AXIS" in d.columns:
            # 0/180° 경계를 넘는 반복측정을 같은 쪽으로 펼침 (눈별 원형평균 기준 ±90°)
            cm = d.groupby("SUBJ")["AXIS"].transform(circ_mean_axis)
            d["AXIS"] = cm + (((d["AXIS"] - cm) + 90) % 180 - 90)
        main = [c for c in cols if c != "AXIS"]
        parts = [anova_rep(d, main)]
        if "AXIS" in cols:
            parts.append(anova_rep(d[d["SUBJ"].isin(cyl_keys)], ["AXIS"]))
        r = pd.concat(parts, ignore_index=True)
        if not r.empty:
            r["_o"] = r["Code"].map(MEASURES.index)
            r = r.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        out[label] = r
    return out


def prec_pass(row):
    pg = PREC_PG.get(row["Code"])
    if pg is None or pd.isna(row["Repeatability_CV"]):
        return "—"
    return "PASS" if float(row["Repeatability_CV"]) <= pg else "FAIL"

# ══════════════════════════════════════════════════════════════
#  레이더 차트 (논문 스타일)
# ══════════════════════════════════════════════════════════════

def _radar_axes(categories, ylim_lo, ylim_hi, tick_labels):
    N = len(categories)
    angles = [n / N * 2 * np.pi for n in range(N)] + [0]
    fig, ax = plt.subplots(figsize=(4.6, 4.6), subplot_kw=dict(polar=True))
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_rlabel_position(30)
    ax.tick_params(axis="y", labelsize=7, colors="0.3")
    ax.grid(color="0.8", linewidth=0.6)
    ax.spines["polar"].set_color("0.5")
    ax.set_ylim(ylim_lo, ylim_hi)
    return fig, ax, angles


def plot_ba_radar(current, prev_list, ylim, legend_labels, title):
    """current / prev: {measure_name: {'lower': v, 'upper': v}} — LoA 95% CI 구간 음영"""
    cats = [RADAR_NAME[c] for c in ["K1", "K2", "CYL", "AXIS"]
            if RADAR_NAME[c] in current]
    if len(cats) < 3:
        return None
    fig, ax, angles = _radar_axes(cats, -ylim, ylim, cats)
    series = [{"label": "This study", "data": current}] + prev_list
    colors = [DEV_COLOR[DEV_T]] + PREV_COLORS
    for i, s in enumerate(series):
        col = colors[i % len(colors)]
        lo = [s["data"].get(c, {}).get("lower", np.nan) for c in cats]
        hi = [s["data"].get(c, {}).get("upper", np.nan) for c in cats]
        lo, hi = lo + lo[:1], hi + hi[:1]
        ax.fill_between(angles, lo, hi, color=col, alpha=0.12, lw=0)
        ax.plot(angles, hi, color=col, lw=1.4)
        ax.plot(angles, lo, color=col, lw=1.4, label=legend_labels.get(s["label"], s["label"]))
    pg_hi = [BA_PG.get({v: k for k, v in RADAR_NAME.items()}[c], (np.nan)) for c in cats]
    pg_lo = [-v for v in pg_hi]
    ax.plot(angles, pg_hi + pg_hi[:1], color=PG_COLOR, ls="--", lw=1.2, label=legend_labels.get("PG", "PG"))
    ax.plot(angles, pg_lo + pg_lo[:1], color=PG_COLOR, ls="--", lw=1.2)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    fig.tight_layout()
    return fig


def plot_prec_radar(cv_current, prev_list, ylim, legend_labels, title):
    """cv_current: {measure_name: {'Test': cv, 'Predicate': cv}} — Cylinder는 CV/10"""
    cats = ["K1", "K2", "Cylinder"]
    ticks = ["K1", "K2", "Cylinder\n(CV/10)"]
    fig, ax, angles = _radar_axes(cats, 0, ylim, ticks)
    for dev in [DEV_T, DEV_P]:
        vals = [cv_current.get(c, {}).get(dev, np.nan) for c in cats]
        if all(np.isnan(v) for v in vals):
            continue
        vals = [0 if np.isnan(v) else v for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, color=DEV_COLOR[dev], lw=1.6, label=legend_labels.get(dev, dev))
        ax.fill(angles, vals, color=DEV_COLOR[dev], alpha=0.08)
    for i, prev in enumerate(prev_list):
        col = PREV_COLORS[i % len(PREV_COLORS)]
        vals = [prev["data"].get(c, np.nan) for c in cats]
        vals = [v / 10 if c == "Cylinder" and not np.isnan(v) else v for c, v in zip(cats, vals)]
        vals = [0 if np.isnan(v) else v for v in vals]
        vals += vals[:1]
        ax.plot(angles, vals, color=col, lw=1.6, ls="-.", label=legend_labels.get(prev["label"], prev["label"]))
    thr = [PREC_RADAR_THR[c] for c in cats]
    ax.plot(angles, thr + thr[:1], color=PG_COLOR, ls="--", lw=1.2, label=legend_labels.get("PG", "PG"))
    ax.set_title(title, fontsize=10, fontweight="bold", pad=16)
    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.12))
    fig.tight_layout()
    return fig


def to_excel_bytes(sheets):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        for name, df in sheets.items():
            df.to_excel(w, sheet_name=name[:31], index=False)
    return buf.getvalue()


def parse_csv_list(text):
    return {norm_pid(p) for p in text.split(",") if p.strip()}

# ══════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════
apply_paper_style()

st.markdown("## 🔬 Keratometry 분석 대시보드")
st.markdown("### Test vs Predicate — Agreement & Precision Analysis")
st.divider()

st.markdown('<div class="section-title">Data Upload</div>', unsafe_allow_html=True)
cu1, cu2, cu3, cu4, cu5 = st.columns(5)
with cu1:
    test_file = st.file_uploader("📂 Test (시험기기)", type=["csv", "xlsx"])
with cu2:
    pred_file = st.file_uploader("📂 Predicate (대조기기)", type=["csv", "xlsx"])
with cu3:
    pred2_file = st.file_uploader("📂 Predicate 2 (산점도용, 선택)", type=["csv", "xlsx"])
with cu4:
    group_file = st.file_uploader("📂 그룹 정보 (선택, PID·GROUP)", type=["csv", "xlsx"])
with cu5:
    study_file = st.file_uploader("📂 Study eye 정보 (선택, PID·STUDY_EYE)", type=["csv", "xlsx"])
st.caption("기기 파일 필수 컬럼: PID, SEQ, EYE(OD/OS), K1, K2, CYL, AXIS  ·  그룹 파일: PID, GROUP (EYE 컬럼이 있으면 눈별 그룹 적용)  ·  "
           "Study eye: PID, STUDY_EYE(OD/OS) — 별도 파일 또는 그룹 파일에 STUDY_EYE 컬럼으로 추가  ·  "
           "PID는 대소문자·하이픈·앞자리 0을 무시하고 매칭합니다 (S001 = s-001).")
st.divider()

# ── 사이드바: 공통 설정 ─────────────────────────────────────
st.sidebar.markdown("### ⚙️ 데이터 설정")
normalize = st.sidebar.checkbox(
    "K1/K2·CYL 표기 정규화", value=True,
    help="K1 > K2인 행은 K1·K2를 교환해 K1=flat, K2=steep로 통일하고, CYL > 0인 행은 "
         "마이너스 실린더 표기로 전환합니다 (CYL 부호 반전, AXIS +90°).")
off_t = st.sidebar.checkbox("Test AXIS +90° 보정", value=False,
                            help="기기가 AXIS를 steep(K2) 경선 기준으로 출력하는 경우 체크")
off_p = st.sidebar.checkbox("Predicate AXIS +90° 보정", value=False,
                            help="기기가 AXIS를 steep(K2) 경선 기준으로 출력하는 경우 체크")
off_p2 = st.sidebar.checkbox("Predicate 2 AXIS +90° 보정", value=False) if pred2_file else False

st.sidebar.markdown("### 📐 분석 설정")
method = st.sidebar.radio(
    "일치도 분석 비교 기준", ["first", "mean"],
    format_func=lambda v: "첫 번째 측정값 (SEQ 최솟값)" if v == "first" else "반복 측정 평균",
    help="반복 측정 평균: K1·K2·CYL·J0·J45는 산술평균, AXIS는 원형평균(2배각)으로 계산합니다.")

st.sidebar.markdown("### 🖼️ 그림 설정")
FIG_FMT = st.sidebar.selectbox("저장 형식", list(FIG_MIME.keys()), index=0)
FIG_DPI = st.sidebar.select_slider("해상도 (dpi)", options=[300, 600, 1200], value=300)
SHOW_CI = st.sidebar.checkbox("BA plot: 95% CI 음영 표시", value=True)
EYE_MARKER = st.sidebar.checkbox("BA plot: OD(●)/OS(○) 마커 구분", value=True)

st.sidebar.markdown("### 🚫 분석에서 제외할 PID")
st.sidebar.caption("쉼표로 구분 (예: s-001, s-015)")
ba_ex = parse_csv_list(st.sidebar.text_input("Agreement 제외 PID", value="", key="ba_ex"))
prec_ex = parse_csv_list(st.sidebar.text_input("Precision 제외 PID", value="", key="prec_ex"))

# ── 데이터 로드 ─────────────────────────────────────────────
data_ready = bool(test_file and pred_file)
group_df, study_df, subsets = None, None, []
if data_ready:
    try:
        test_df = load_device(test_file.getvalue(), test_file.name, normalize, off_t)
        pred_df = load_device(pred_file.getvalue(), pred_file.name, normalize, off_p)
    except Exception as e:
        st.error(f"데이터 로드 오류: {e}")
        data_ready = False

if data_ready and group_file:
    try:
        group_df = load_group(group_file.getvalue(), group_file.name)
        study_df = load_study_eye(group_file.getvalue(), group_file.name)   # 그룹 파일의 STUDY_EYE 컬럼
    except Exception as e:
        st.error(f"그룹 파일 오류: {e}")
        group_df = None

if data_ready and study_file:
    try:
        study_df = load_study_eye(study_file.getvalue(), study_file.name)  # 별도 파일이 우선
        if study_df is None:
            st.error("Study eye 파일에 PID / STUDY_EYE 컬럼이 없습니다.")
    except Exception as e:
        st.error(f"Study eye 파일 오류: {e}")
        study_df = None

if data_ready:
    test_df = attach_study(attach_group(test_df, group_df), study_df)
    pred_df = attach_study(attach_group(pred_df, group_df), study_df)

    has_eye = bool({"OD", "OS"} & set(test_df["EYE"]))
    eye_options = ["OD+OS", "OD", "OS"] if has_eye else [ALL_EYES]
    if has_eye and study_df is not None:
        eye_options.append(STUDY_EYE)
    st.sidebar.markdown("### 👁️ 분석 단위")
    eye_sel = st.sidebar.multiselect("Eye", eye_options, default=eye_options,
                                     help="OD+OS: 양안을 독립된 눈으로 합산하여 분석 · "
                                          "Study eye: 대상자별로 지정된 눈(OD 또는 OS) 1안만 분석")
    groups = sorted(group_df["GROUP"].unique(), key=natural_key) if group_df is not None else []
    grp_options = [ALL_GROUP] + groups
    grp_sel = st.sidebar.multiselect("Group", grp_options, default=grp_options) if groups else [ALL_GROUP]
    subsets = [(e, g) for e in eye_sel for g in grp_sel]
    if not subsets:
        st.warning("사이드바에서 Eye / Group을 하나 이상 선택해주세요.")

    # 데이터 점검 메시지
    info = []
    tk = set(zip(test_df["PID_KEY"], test_df["EYE"]))
    pk = set(zip(pred_df["PID_KEY"], pred_df["EYE"]))
    info.append(f"Test {test_df['PID_KEY'].nunique()}명 / {len(tk)}안 · Predicate {pred_df['PID_KEY'].nunique()}명 / {len(pk)}안 · "
                f"공통 {len(tk & pk)}안")
    if tk ^ pk:
        only = sorted({f"{a}({b})" for a, b in tk ^ pk}, key=natural_key)
        info.append(f"한쪽 기기에만 있는 눈 {len(only)}개 (일치도 분석에서 제외): {', '.join(only[:20])}{' …' if len(only) > 20 else ''}")
    if group_df is not None:
        no_grp = sorted(set(test_df.loc[test_df["GROUP"].isna(), "PID"]), key=natural_key)
        if no_grp:
            info.append(f"그룹 정보가 없는 PID {len(no_grp)}명 ('{ALL_GROUP}'에만 포함): {', '.join(no_grp[:20])}{' …' if len(no_grp) > 20 else ''}")
    if study_df is not None and has_eye:
        se = test_df.drop_duplicates("PID_KEY")
        n_od, n_os = (se["STUDY_EYE"] == "OD").sum(), (se["STUDY_EYE"] == "OS").sum()
        info.append(f"Study eye: OD {n_od}명 · OS {n_os}명")
        no_se = sorted(set(se.loc[se["STUDY_EYE"].isna(), "PID"]), key=natural_key)
        if no_se:
            info.append(f"Study eye 정보가 없는 PID {len(no_se)}명 (Study eye 분석에서 제외): {', '.join(no_se[:20])}{' …' if len(no_se) > 20 else ''}")
    st.caption("  \n".join(info))

    gap = axis_gap_check(test_df, pred_df)
    if gap is not None and gap > 45:
        st.warning(f"⚠️ |CYL| ≥ {CYL_CUTOFF} D 눈에서 Test와 Predicate의 AXIS 차이 중앙값이 {gap:.0f}°입니다. "
                   "두 기기의 AXIS 기준 경선(flat/steep)이 다를 가능성이 높습니다. "
                   "기기 출력 사양을 확인한 뒤 사이드바의 ‘AXIS +90° 보정’을 적용하세요 (AXIS·J0·J45 결과에 영향).")

# ══════════════════════════════════════════════════════════════
#  메인 탭
# ══════════════════════════════════════════════════════════════
tab_ba, tab_prec, tab_sc = st.tabs(["📊 Agreement (Bland-Altman)", "🔁 Precision", "📉 측정값 산점도"])

# ─────────────────────────────────────────────────────────────
# TAB 1 : Agreement
# ─────────────────────────────────────────────────────────────
with tab_ba:
    if not data_ready:
        st.info("👆 Test와 Predicate 파일을 업로드하면 분석이 시작됩니다.")
    elif subsets:
        merged = build_agreement(exclude(test_df, ba_ex), exclude(pred_df, ba_ex), method)
        merged = attach_study(attach_group(merged, group_df), study_df)
        avail = [m for m in MEASURES if all(c in merged.columns for c in PAIR[m])]
        ba_sel = st.sidebar.multiselect("📊 Agreement 분석 항목", avail, default=avail,
                                        format_func=lambda m: M_LABEL[m], key="ba_sel")

        rows, store = [], {}
        for e, g in subsets:
            d = subset_df(merged, e, g)
            for m in ba_sel:
                res = ba_stats(d, m)
                if res:
                    store[(e, g, m)] = (ba_data(d, m), res)
                    rows.append({"Eye": e, "Group": g, **res, "PG (±)": BA_PG.get(m, np.nan), "Pass/Fail": ba_pass(res)})

        if not rows:
            st.error("유효한 데이터가 없습니다 (항목별 최소 3안 필요).")
        else:
            res_all = pd.DataFrame(rows)
            st.caption(f"비교 기준: **{'첫 번째 측정값' if method == 'first' else '반복 측정 평균'}** · "
                       f"Axis는 |Predicate CYL| ≥ {CYL_CUTOFF} D인 눈만 분석 · J0/J45는 전체 대상 · "
                       "PG 판정: LoA 95% CI가 PG 범위 이내")
            s1, s2, s3 = st.tabs(["📋 Subset별 결과", "📑 통합 결과 테이블", "📡 성능 비교 Chart"])

            # ── Subset별 결과 ───────────────────────────────
            with s1:
                labels = [subset_label(e, g) for e, g in subsets]
                pick = st.selectbox("분석 대상 (Eye · Group)", labels, key="ba_subset")
                e, g = subsets[labels.index(pick)]
                sub = res_all[(res_all["Eye"] == e) & (res_all["Group"] == g)]
                if sub.empty:
                    st.warning("이 subset에는 분석 가능한 데이터가 없습니다.")
                else:
                    st.markdown('<div class="section-title">Summary Overview</div>', unsafe_allow_html=True)
                    cols = st.columns(min(len(sub), 3))
                    for i, (_, r) in enumerate(sub.iterrows()):
                        with cols[i % len(cols)]:
                            pf = r["Pass/Fail"]
                            pg_html = ""
                            if pf != "—":
                                c = "#4caf8a" if pf == "PASS" else "#ff5252"
                                pg_html = f'<div class="metric-sub" style="color:{c};font-weight:600;">{"✅" if pf == "PASS" else "❌"} {pf} &nbsp;|&nbsp; PG ± {r["PG (±)"]}</div>'
                            st.markdown(f"""
                            <div class="metric-card">
                              <div class="metric-label">{r['Measure']} [{r['Unit']}] · N={r['N']}</div>
                              <div class="metric-value">{r['Mean Diff']} <span style="font-size:0.9rem;color:#8b92b8;">± {r['SD Diff']}</span></div>
                              <div class="metric-sub">Mean Diff ± SD &nbsp;|&nbsp; LoA [{r['LoA Lower']}, {r['LoA Upper']}]</div>
                              <div class="metric-sub">CCC {r['CCC']} &nbsp;|&nbsp; Pearson r {r['Pearson r']}</div>
                              {pg_html}
                            </div>""", unsafe_allow_html=True)

                    st.divider()
                    st.markdown('<div class="section-title">Bland-Altman Plots</div>', unsafe_allow_html=True)
                    layout = st.radio("표시 방식", ["개별 그림", "다중 패널 그림 (A, B, …)"], horizontal=True, key="ba_layout")
                    show_title = st.checkbox("그림 제목에 Eye·Group 표시", value=False, key="ba_title")
                    items = [(store[(e, g, m)][0], m, store[(e, g, m)][1]) for m in ba_sel if (e, g, m) in store]
                    fname_base = f"BA_{safe_name(e)}_{safe_name(g)}"

                    if layout.startswith("개별"):
                        zbuf = io.BytesIO()
                        with zipfile.ZipFile(zbuf, "w") as zf:
                            pcols = st.columns(2)
                            for i, (dd, m, res) in enumerate(items):
                                ttl = f"{M_LABEL[m]} ({e}, {g})" if show_title else M_LABEL[m]
                                fig = ba_figure(dd, m, res, ttl, SHOW_CI, EYE_MARKER)
                                zf.writestr(f"{fname_base}_{m}.{FIG_FMT.lower()}", fig_bytes(fig, FIG_FMT, FIG_DPI))
                                with pcols[i % 2]:
                                    show_fig(fig, f"{fname_base}_{m}", f"dl_ba_{e}_{g}_{m}", FIG_FMT, FIG_DPI)
                        st.download_button(f"⬇️ 이 subset 그림 전체 (ZIP, {FIG_FMT})", zbuf.getvalue(),
                                           f"{fname_base}.zip", "application/zip", key="ba_zip")
                    else:
                        ncol = st.slider("열 개수", 1, 3, 2, key="ba_ncol")
                        sup = f"{e}, {g}" if show_title else None
                        fig = ba_multipanel(items, ncol, SHOW_CI, EYE_MARKER, sup)
                        show_fig(fig, f"{fname_base}_panel", "dl_ba_panel", FIG_FMT, FIG_DPI)

                    st.divider()
                    st.markdown('<div class="section-title">Detailed Statistics Table</div>', unsafe_allow_html=True)
                    t1, t2, t3 = st.tabs(["📊 Agreement", "📈 Correlation", "📋 Descriptive"])
                    with t1:
                        st.dataframe(sub[["Measure", "Unit", "PG (±)", "Pass/Fail", "N", "Mean Diff", "SD Diff",
                                          "95% CI Lo", "95% CI Hi", "LoA Lower", "LoA Upper",
                                          "LoA Lower CI Lo", "LoA Lower CI Hi", "LoA Upper CI Lo", "LoA Upper CI Hi"]],
                                     use_container_width=True, hide_index=True)
                    with t2:
                        st.dataframe(sub[["Measure", "N", "Pearson r", "Pearson CI Lo", "Pearson CI Hi", "Pearson p", "CCC"]],
                                     use_container_width=True, hide_index=True)
                    with t3:
                        st.dataframe(sub[["Measure", "Unit", "N", "Test Mean", "Test SD", "Test Min", "Test Max",
                                          "Predicate Mean", "Predicate SD", "Predicate Min", "Predicate Max"]],
                                     use_container_width=True, hide_index=True)

            # ── 통합 결과 테이블 ────────────────────────────
            with s2:
                st.markdown('<div class="section-title">All Subsets</div>', unsafe_allow_html=True)
                show_cols = ["Eye", "Group", "Measure", "Unit", "N", "Mean Diff", "SD Diff", "95% CI Lo", "95% CI Hi",
                             "LoA Lower", "LoA Upper", "LoA Lower CI Lo", "LoA Upper CI Hi", "PG (±)", "Pass/Fail",
                             "CCC", "Pearson r"]
                st.dataframe(res_all[show_cols], use_container_width=True, hide_index=True)
                export = res_all.drop(columns=["Code"])
                export.insert(0, "Comparison", "First measurement" if method == "first" else "Mean of repeats")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇️ Agreement 결과 CSV", export.to_csv(index=False).encode("utf-8-sig"),
                                       "Agreement_result.csv", "text/csv")
                with c2:
                    st.download_button("⬇️ Agreement 결과 Excel (+분석 데이터)",
                                       to_excel_bytes({"Agreement": export, "Analysis data": merged}),
                                       "Agreement_result.xlsx",
                                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # ── 성능 비교 레이더 ────────────────────────────
            with s3:
                st.markdown("Lower LoA 95% CI 하한 ~ Upper LoA 95% CI 상한 구간을 음영으로 표시합니다.")
                labels = [subset_label(e, g) for e, g in subsets]
                pick_r = st.selectbox("대상 (Eye · Group)", labels, key="ba_radar_subset")
                er, gr = subsets[labels.index(pick_r)]

                cb1, cb2 = st.columns(2)
                with cb1:
                    ba_prev1 = st.file_uploader("📂 비교 결과 CSV #1 (선택)", type=["csv"], key="ba_prev1")
                with cb2:
                    ba_prev2 = st.file_uploader("📂 비교 결과 CSV #2 (선택)", type=["csv"], key="ba_prev2")
                st.caption("비교 CSV 형식: Measure(K1/K2/Cylinder/Axis), <label>_lower, <label>_upper")

                def parse_ba_prev(f):
                    df_p = pd.read_csv(f)
                    lo = next((c for c in df_p.columns if "lower" in c.lower()), None)
                    hi = next((c for c in df_p.columns if "upper" in c.lower()), None)
                    if not lo or not hi or "Measure" not in df_p.columns:
                        return None, None
                    label = re.sub(r"_?lower$", "", lo, flags=re.I) or "Previous"
                    return label, {r["Measure"]: {"lower": float(r[lo]), "upper": float(r[hi])} for _, r in df_p.iterrows()}

                prev_s = []
                for pf in [ba_prev1, ba_prev2]:
                    if pf is not None:
                        lbl, dat = parse_ba_prev(pf)
                        if lbl:
                            prev_s.append({"label": lbl, "data": dat})

                cur = {}
                for m in ["K1", "K2", "CYL", "AXIS"]:
                    if (er, gr, m) in store:
                        r = store[(er, gr, m)][1]
                        cur[RADAR_NAME[m]] = {"lower": r["LoA Lower CI Lo"], "upper": r["LoA Upper CI Hi"]}

                st.markdown("##### 범례 라벨 수정 (선택)")
                keys = ["This study"] + [p["label"] for p in prev_s] + ["PG"]
                lcols = st.columns(len(keys))
                leg = {k: lcols[i].text_input(f"라벨 {i + 1}", value=k, key=f"ba_leg_{i}") for i, k in enumerate(keys)}
                ylim = st.slider("축 범위 (±)", 0.5, 20.0, BA_RADAR_YLIM, 0.25, key="ylim_ba")
                fig = plot_ba_radar(cur, prev_s, ylim, leg, f"Keratometry ({er}, {gr})")
                if fig is None:
                    st.info("레이더 차트에는 K1/K2/Cylinder/Axis 중 3개 이상이 필요합니다.")
                else:
                    cc, _ = st.columns([1, 1])
                    with cc:
                        show_fig(fig, f"BA_radar_{safe_name(er)}_{safe_name(gr)}", "dl_ba_radar", FIG_FMT, FIG_DPI)

# ─────────────────────────────────────────────────────────────
# TAB 2 : Precision
# ─────────────────────────────────────────────────────────────
with tab_prec:
    st.markdown("One-way random effect ANOVA 기반 반복성(Repeatability) 및 ICC 분석 — 눈(PID×EYE) 단위")
    if not data_ready:
        st.info("👆 Test와 Predicate 파일을 업로드하면 분석이 시작됩니다.")
    elif subsets:
        p_avail = [m for m in MEASURES if m in test_df.columns or m in pred_df.columns]
        prec_sel = st.sidebar.multiselect("🔁 Precision 분석 항목", p_avail, default=p_avail,
                                          format_func=lambda m: M_LABEL[m], key="prec_sel")
        if not prec_sel:
            st.warning("사이드바에서 Precision 분석 항목을 하나 이상 선택해주세요.")
        else:
            t_p, p_p = exclude(test_df, prec_ex), exclude(pred_df, prec_ex)
            prec_store, all_rows = {}, []
            with st.spinner("🔄 Precision 분석 실행 중..."):
                for e, g in subsets:
                    r = run_precision(subset_df(t_p, e, g), subset_df(p_p, e, g), tuple(prec_sel))
                    prec_store[(e, g)] = r
                    for dev, df in r.items():
                        if not df.empty:
                            x = df.copy()
                            x.insert(0, "Device", dev)
                            x.insert(0, "Group", g)
                            x.insert(0, "Eye", e)
                            x["PG (CV%)"] = x["Code"].map(PREC_PG)
                            x["Pass/Fail"] = x.apply(prec_pass, axis=1)
                            all_rows.append(x)

            if not all_rows:
                st.warning("분석 결과가 없습니다. 데이터를 확인해주세요 (subset별 최소 3안 필요).")
            else:
                prec_all = pd.concat(all_rows, ignore_index=True)
                st.caption(f"CYL은 절댓값으로 분석 · Axis는 |Predicate 첫 측정 CYL| ≥ {CYL_CUTOFF} D인 눈만, "
                           "0/180° 경계는 눈별로 펼쳐서 계산 · PG 판정: Repeatability CV% ≤ PG")
                q1, q2, q3 = st.tabs(["📋 Subset별 결과", "📑 통합 결과 테이블", "📡 성능 비교 Chart"])

                with q1:
                    labels = [subset_label(e, g) for e, g in subsets]
                    pick = st.selectbox("분석 대상 (Eye · Group)", labels, key="prec_subset")
                    e, g = subsets[labels.index(pick)]
                    dtabs = st.tabs([f"🔵 {DEV_T}", f"🟠 {DEV_P}"])
                    for tab, dev in zip(dtabs, [DEV_T, DEV_P]):
                        with tab:
                            gdf = prec_all[(prec_all["Eye"] == e) & (prec_all["Group"] == g) & (prec_all["Device"] == dev)]
                            if gdf.empty:
                                st.info(f"{dev} 분석 결과가 없습니다.")
                                continue
                            cols = st.columns(min(len(gdf), 3))
                            for i, (_, r) in enumerate(gdf.iterrows()):
                                with cols[i % len(cols)]:
                                    pf = r["Pass/Fail"]
                                    if pf == "—":
                                        pg_html = '<div class="metric-sub">PG 없음</div>'
                                    else:
                                        c = "#4caf8a" if pf == "PASS" else "#ff5252"
                                        pg_html = f'<div class="metric-sub" style="color:{c};font-weight:600;">{"✅" if pf == "PASS" else "❌"} {pf} &nbsp;|&nbsp; PG ≤ {r["PG (CV%)"]}</div>'
                                    st.markdown(f'''<div class="metric-card">
                                      <div class="metric-label">{r["Measure"]} · N={r["N (eyes)"]}</div>
                                      <div class="metric-value">{r["Repeatability_CV"]} <span style="font-size:0.8rem;color:#8b92b8;">CV%</span></div>
                                      <div class="metric-sub">Sw: {r["Repeatability_SD"]} &nbsp;|&nbsp; Limit: {r["Repeatability_limit"]}</div>
                                      <div class="metric-sub">ICC: {r["ICC"]} ({r["ICC_CI_lower"]} – {r["ICC_CI_upper"]})</div>
                                      {pg_html}
                                    </div>''', unsafe_allow_html=True)
                            st.dataframe(gdf.drop(columns=["Eye", "Group", "Device", "Code"]),
                                         use_container_width=True, hide_index=True)

                with q2:
                    st.dataframe(prec_all.drop(columns=["Code"]), use_container_width=True, hide_index=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        st.download_button("⬇️ Precision 결과 CSV",
                                           prec_all.drop(columns=["Code"]).to_csv(index=False).encode("utf-8-sig"),
                                           "Precision_result.csv", "text/csv")
                    with c2:
                        st.download_button("⬇️ Precision 결과 Excel",
                                           to_excel_bytes({"Precision": prec_all.drop(columns=["Code"])}),
                                           "Precision_result.xlsx",
                                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

                with q3:
                    st.markdown("Repeatability CV%를 레이더 차트로 나타냅니다 (Cylinder는 CV/10).")
                    labels = [subset_label(e, g) for e, g in subsets]
                    pick_r = st.selectbox("대상 (Eye · Group)", labels, key="prec_radar_subset")
                    er, gr = subsets[labels.index(pick_r)]

                    cp1, cp2 = st.columns(2)
                    with cp1:
                        pv1 = st.file_uploader("📂 비교 결과 CSV #1 (선택)", type=["csv"], key="prev1")
                    with cp2:
                        pv2 = st.file_uploader("📂 비교 결과 CSV #2 (선택)", type=["csv"], key="prev2")
                    st.caption("비교 CSV 형식: Measure(K1/K2/Cylinder), <label> (CV% 원값)")

                    prev_s = []
                    for pf in [pv1, pv2]:
                        if pf is not None:
                            dfp = pd.read_csv(pf)
                            vc = [c for c in dfp.columns if c not in ("Mode", "Measure")]
                            if vc and "Measure" in dfp.columns:
                                prev_s.append({"label": vc[0], "data": dict(zip(dfp["Measure"], dfp[vc[0]].astype(float)))})

                    cv_cur = {}
                    for dev, df in prec_store[(er, gr)].items():
                        for _, r in df.iterrows():
                            if r["Code"] in ("K1", "K2", "CYL"):
                                v = r["Repeatability_CV"] / 10 if r["Code"] == "CYL" else r["Repeatability_CV"]
                                cv_cur.setdefault(RADAR_NAME[r["Code"]], {})[dev] = v

                    st.markdown("##### 범례 라벨 수정 (선택)")
                    keys = [DEV_T, DEV_P] + [p["label"] for p in prev_s] + ["PG"]
                    lcols = st.columns(len(keys))
                    leg = {k: lcols[i].text_input(f"라벨 {i + 1}", value=k, key=f"prec_leg_{i}") for i, k in enumerate(keys)}
                    ylim = st.slider("축 범위", 0.5, 10.0, PREC_RADAR_YLIM, 0.25, key="ylim_prec")
                    fig = plot_prec_radar(cv_cur, prev_s, ylim, leg, f"Keratometry ({er}, {gr})")
                    cc, _ = st.columns([1, 1])
                    with cc:
                        show_fig(fig, f"Precision_radar_{safe_name(er)}_{safe_name(gr)}", "dl_prec_radar", FIG_FMT, FIG_DPI)

# ─────────────────────────────────────────────────────────────
# TAB 3 : 측정값 산점도
# ─────────────────────────────────────────────────────────────
with tab_sc:
    st.markdown("PID별 반복 측정값을 기기별로 나타냅니다 (숫자 = SEQ).")
    if not data_ready:
        st.info("👆 Test와 Predicate 파일을 업로드하면 산점도가 표시됩니다.")
    else:
        devices = {DEV_T: test_df, DEV_P: pred_df}
        if pred2_file:
            try:
                devices[DEV_P2] = attach_study(attach_group(
                    load_device(pred2_file.getvalue(), pred2_file.name, normalize, off_p2), group_df), study_df)
            except Exception as e:
                st.error(f"Predicate 2 로드 오류: {e}")

        c1, c2, c3 = st.columns(3)
        with c1:
            sc_var = st.selectbox("측정 변수", [m for m in MEASURES if any(m in d.columns for d in devices.values())],
                                  format_func=lambda m: f"{M_LABEL[m]} [{M_UNIT[m]}]", key="sc_var")
        with c2:
            eyes_av = sorted(set(test_df["EYE"]))
            sc_eye_opts = (["OD+OS"] + eyes_av) if len(eyes_av) > 1 else eyes_av
            if study_df is not None and len(eyes_av) > 1:
                sc_eye_opts.append(STUDY_EYE)
            sc_eye = st.selectbox("Eye", sc_eye_opts, key="sc_eye")
        with c3:
            sc_grp = st.selectbox("Group", [ALL_GROUP] + (sorted(group_df["GROUP"].unique(), key=natural_key)
                                                          if group_df is not None else []), key="sc_grp")

        st.markdown("##### 범례 라벨 수정 (선택)")
        lcols = st.columns(len(devices) + 1)
        sc_leg = {k: lcols[i].text_input(f"라벨 {i + 1}", value=k, key=f"sc_leg_{i}") for i, k in enumerate(devices)}
        with lcols[-1]:
            legend_loc = st.selectbox("범례 위치", ["upper right", "upper left", "lower right", "lower left",
                                                  "upper center", "lower center", "outside right"], key="sc_loc")

        subs = {k: subset_df(v, sc_eye, sc_grp) for k, v in devices.items()}
        base = subs[DEV_T]
        units = sorted({(p, k, e) for p, k, e in zip(base["PID"], base["PID_KEY"], base["EYE"])},
                       key=lambda u: (natural_key(u[1]), u[2]))
        multi_eye = len({u[2] for u in units}) > 1
        xlabels = [f"{u[0]} {u[2]}" if multi_eye else u[0] for u in units]
        xidx = {(u[1], u[2]): i for i, u in enumerate(units)}

        if not units:
            st.warning("표시할 데이터가 없습니다.")
        else:
            fig, ax = plt.subplots(figsize=(max(6.5, 0.28 * len(units) + 1.5), 3.6))
            # 대상자별 세로 기준선 + 가로 보조 눈금선 (값 읽기용)
            ax.set_axisbelow(True)
            ax.grid(axis="x", color="0.85", lw=0.6, ls="-", zorder=0)
            ax.grid(axis="y", which="major", color="0.85", lw=0.5, ls=":", zorder=0)
            vals = []
            handles = []
            for dev, d in subs.items():
                off = 0   # 모든 기기를 같은 x 위치(한 줄)에 표기
                if sc_var not in d.columns:
                    continue
                col = DEV_COLOR.get(dev, "0.4")
                for _, r in d.dropna(subset=[sc_var]).iterrows():
                    k = (r["PID_KEY"], r["EYE"])
                    if k not in xidx:
                        continue
                    ax.text(xidx[k] + off, r[sc_var], f"{int(r['SEQ'])}" if pd.notna(r["SEQ"]) else "?",
                            color=col, fontsize=6.5, fontweight="bold", ha="center", va="center", zorder=3)
                    vals.append(r[sc_var])
                handles.append(ax.scatter([], [], color=col, s=18, label=sc_leg.get(dev, dev)))
            if vals:
                lo, hi = min(vals), max(vals)
                pad = (hi - lo) * 0.08 if hi > lo else 0.5
                ax.set_ylim(lo - pad, hi + pad)
            ax.set_xticks(range(len(units)))
            ax.set_xticklabels(xlabels, rotation=90, fontsize=7)
            ax.set_xlim(-0.8, len(units) - 0.2)
            ax.set_xlabel("Subject")
            ax.set_ylabel(f"{M_LABEL[sc_var]} ({M_UNIT[sc_var]})")
            if legend_loc == "outside right":
                ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5))
            else:
                ax.legend(handles=handles, loc=legend_loc)
            fig.tight_layout()
            show_fig(fig, f"Scatter_{sc_var}_{safe_name(sc_eye)}_{safe_name(sc_grp)}", "dl_sc", FIG_FMT, FIG_DPI)
