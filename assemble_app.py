"""
assemble_app.py v2 — writes complete dashboard/app.py with real SHAP values in drilldown
"""
code = r'''"""
app.py - PayU MerchantPulse Platform
Run: streamlit run dashboard/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="MerchantPulse | PayU", page_icon="P", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;}
.metric-card{background:linear-gradient(135deg,#1e2940,#162032);border:1px solid #2d3f5e;
  border-radius:12px;padding:20px;text-align:center;margin-bottom:8px;}
.metric-val{font-size:2rem;font-weight:700;color:#60a5fa;}
.metric-lbl{font-size:.78rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.08em;}
</style>""", unsafe_allow_html=True)

TIER = {"Red":"#f87171","Orange":"#fb923c","Yellow":"#fbbf24","Green":"#34d399"}
SEV  = {"CRITICAL":"#f87171","HIGH":"#fb923c","MEDIUM":"#fbbf24","LOW":"#34d399"}
DARK = dict(paper_bgcolor="#1e2940", plot_bgcolor="#162032", font_color="#e2e8f0")

@st.cache_data
def load():
    base  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    df    = pd.read_csv(os.path.join(base,"data/outputs/master_dataframe.csv"))
    wk    = pd.read_csv(os.path.join(base,"data/processed/features_engineered.csv"))
    cr    = pd.read_csv(os.path.join(base,"data/outputs/cluster_rar.csv"))
    shap  = pd.read_csv(os.path.join(base,"data/outputs/shap_values.csv"))
    return df, wk, cr, shap

df, weekly, cluster_rar, shap_df = load()

SHAP_FEATURES = [c for c in shap_df.columns
                 if c not in ["merchant_id","top_driver_1","top_driver_1_val",
                               "top_driver_2","top_driver_2_val","top_driver_3",
                               "top_driver_3_val","alert_text"]]

SCREENS = ["Overview","Cluster Explorer","Merchant Drilldown","Watchlist","What-If Simulator"]

with st.sidebar:
    st.markdown("## MerchantPulse")
    st.markdown("*PayU Risk Intelligence*")
    st.markdown("---")
    screen = st.radio("Navigate", SCREENS)
    st.markdown("---")
    st.markdown(f"**Merchants:** {len(df)}")
    st.markdown(f"**Avg Health:** {df['health_score'].mean():.1f} / 100")
    st.markdown(f"**Portfolio RaR:** Rs {df['revenue_at_risk'].sum()/1e7:.2f} Cr")
    st.markdown(f"**Critical:** {(df['alert_severity']=='CRITICAL').sum()}")

# S1 OVERVIEW ──────────────────────────────────────────────────────────────────
if screen == "Overview":
    st.markdown("# Portfolio Overview")
    st.caption("Week 24 Snapshot — MerchantPulse Risk Intelligence")
    c1,c2,c3,c4,c5 = st.columns(5)
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5],[
        ("Total RaR",  f"Rs {df['revenue_at_risk'].sum()/1e7:.1f} Cr"),
        ("Avg Health", f"{df['health_score'].mean():.0f} / 100"),
        ("Critical",   str((df['alert_severity']=='CRITICAL').sum())),
        ("High Drift", str((df['drift_composite']>0.4).sum())),
        ("Watchlist",  str(df['watchlist_flag'].sum())),
    ]):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{val}</div>'
                        f'<div class="metric-lbl">{lbl}</div></div>', unsafe_allow_html=True)
    st.markdown("---")
    col1,col2 = st.columns([1.4,1])
    with col1:
        fig = px.histogram(df,x="health_score",nbins=30,color="risk_tier",
                           color_discrete_map=TIER,title="Health Score Distribution",
                           template="plotly_dark")
        fig.update_layout(**DARK,height=320,title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        tc = df["risk_tier"].value_counts().reset_index()
        tc.columns = ["Tier","Count"]
        fig2 = px.pie(tc,names="Tier",values="Count",color="Tier",color_discrete_map=TIER,
                      hole=0.55,title="Risk Tier Distribution",template="plotly_dark")
        fig2.update_layout(**DARK,height=320,title_font_size=15)
        st.plotly_chart(fig2, use_container_width=True)
    col3,col4 = st.columns(2)
    with col3:
        fig3 = px.bar(cluster_rar.sort_values("total_rar",ascending=True),
                      x="total_rar",y="cluster_label",orientation="h",
                      title="Revenue at Risk by Cluster",color="avg_health",
                      color_continuous_scale="RdYlGn",template="plotly_dark")
        fig3.update_layout(**DARK,height=280,title_font_size=15)
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        top10 = df.nsmallest(10,"health_score")[
            ["merchant_id","segment","health_score","p_failure_4w","revenue_at_risk","alert_severity"]].copy()
        top10["revenue_at_risk"] = top10["revenue_at_risk"].apply(lambda x: f"Rs {x:,.0f}")
        top10["health_score"]    = top10["health_score"].round(1)
        st.markdown("#### Top 10 Riskiest Merchants")
        st.dataframe(top10.set_index("merchant_id"), use_container_width=True, height=260)
    st.markdown("---")
    st.markdown("### Executive Alerts")
    for _, row in df[df["alert_severity"].isin(["CRITICAL","HIGH"])].nsmallest(3,"health_score").iterrows():
        c = SEV.get(row["alert_severity"],"#94a3b8")
        st.markdown(f'<div style="background:#1e2940;border-left:4px solid {c};padding:12px 16px;'
                    f'border-radius:8px;margin-bottom:8px;"><b style="color:{c}">[{row["alert_severity"]}]</b> '
                    f'{row["merchant_id"]} ({row["segment"]}) | Health: {row["health_score"]:.0f} | '
                    f'RaR: Rs {row["revenue_at_risk"]:,.0f} | {row.get("alert_text_x","")}</div>',
                    unsafe_allow_html=True)

# S2 CLUSTER EXPLORER ──────────────────────────────────────────────────────────
elif screen == "Cluster Explorer":
    st.markdown("# Behavioral Cluster Explorer")
    cf1,cf2 = st.columns([1,3])
    with cf1:
        sel = st.multiselect("Clusters",df["cluster_label"].dropna().unique().tolist(),
                             default=df["cluster_label"].dropna().unique().tolist())
        cby = st.selectbox("Color by",["cluster_label","risk_tier","alert_severity","segment"])
    fdf  = df[df["cluster_label"].isin(sel)] if sel else df
    cmap = TIER if cby=="risk_tier" else SEV if cby=="alert_severity" else None
    with cf2:
        fig = px.scatter(fdf,x="umap_x",y="umap_y",color=cby,color_discrete_map=cmap,
                         hover_data={"merchant_id":True,"health_score":":.1f",
                                     "revenue_at_risk":":.0f","drift_composite":":.3f","segment":True},
                         title="UMAP Merchant Behavior Space",template="plotly_dark")
        fig.update_traces(marker=dict(size=7,opacity=0.8))
        fig.update_layout(**DARK,height=480,title_font_size=15)
        st.plotly_chart(fig, use_container_width=True)
    csum = fdf.groupby("cluster_label").agg(
        merchants=("merchant_id","count"),avg_health=("health_score","mean"),
        avg_fail=("p_failure_4w","mean"),total_rar=("revenue_at_risk","sum"),
        avg_drift=("drift_composite","mean")).round(3).reset_index()
    csum["total_rar"] = csum["total_rar"].apply(lambda x: f"Rs {x:,.0f}")
    st.dataframe(csum, use_container_width=True)

# S3 MERCHANT DRILLDOWN ────────────────────────────────────────────────────────
elif screen == "Merchant Drilldown":
    st.markdown("# Merchant Deep Dive")
    mid = st.selectbox("Select Merchant", df["merchant_id"].sort_values().tolist())
    row = df[df["merchant_id"]==mid].iloc[0]
    mw  = weekly[weekly["merchant_id"]==mid].sort_values("week")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Health Score",    f"{row['health_score']:.0f} / 100")
    c2.metric("Failure Prob 4w", f"{row['p_failure_4w']:.1%}")
    c3.metric("Revenue at Risk", f"Rs {row['revenue_at_risk']:,.0f}")
    c4.metric("Drift Score",     f"{row.get('drift_composite',0):.3f}")
    col_l,col_r = st.columns(2)
    with col_l:
        fig = px.area(mw,x="week",y="gmv",title="GMV Over Time",
                      template="plotly_dark",color_discrete_sequence=["#60a5fa"])
        fig.update_layout(**DARK,height=240)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mw["week"],y=mw["refund_rate"],
                                  name="Refund Rate",line=dict(color="#fb923c")))
        fig2.add_trace(go.Scatter(x=mw["week"],y=mw["chargeback_rate"],
                                  name="Chargeback Rate",line=dict(color="#f87171")))
        fig2.update_layout(title="Refund & Chargeback Rates",template="plotly_dark",
                           **DARK,height=240)
        st.plotly_chart(fig2, use_container_width=True)
    with col_r:
        # Real SHAP values
        shap_row = shap_df[shap_df["merchant_id"]==mid]
        if len(shap_row) > 0:
            sv = shap_row[SHAP_FEATURES].iloc[0]
            top_feats = sv.abs().nlargest(8).index.tolist()
            vals   = [float(sv[f]) for f in top_feats]
            colors = ["#f87171" if v<0 else "#34d399" for v in vals]
            fig3 = go.Figure(go.Bar(x=vals,y=top_feats,orientation="h",
                                    marker_color=colors,
                                    text=[f"{v:+.3f}" for v in vals],
                                    textposition="outside"))
            fig3.update_layout(title="SHAP — Top 8 Risk Drivers",
                               template="plotly_dark",**DARK,
                               height=320,xaxis_title="SHAP Value (impact on health score)")
            st.plotly_chart(fig3, use_container_width=True)
        sc = SEV.get(str(row.get("alert_severity","LOW")),"#34d399")
        at = row.get("alert_text_x", row.get("alert_text","No alerts."))
        st.markdown(f'<div style="background:#1e2940;border:1px solid {sc};'
                    f'border-radius:12px;padding:16px;margin-top:8px;">'
                    f'<b style="color:{sc}">{row.get("alert_severity","LOW")}</b><br>'
                    f'<b>Segment:</b> {row["segment"]} &nbsp;|&nbsp; '
                    f'<b>Cluster:</b> {row.get("cluster_label","")}<br>'
                    f'<b>Alert:</b> {at}</div>', unsafe_allow_html=True)

# S4 WATCHLIST ─────────────────────────────────────────────────────────────────
elif screen == "Watchlist":
    st.markdown("# Risk Watchlist")
    sf  = st.multiselect("Severity",["CRITICAL","HIGH","MEDIUM","LOW"],
                         default=["CRITICAL","HIGH","MEDIUM"])
    wdf = df[(df["watchlist_flag"]==1)&(df["alert_severity"].isin(sf))].sort_values(
        ["alert_severity","health_score"])
    st.markdown(f"**{len(wdf)} merchants on watchlist**")
    col1,col2 = st.columns([2,1])
    with col1:
        disp = wdf[["merchant_id","segment","health_score","p_failure_4w",
                    "revenue_at_risk","drift_composite","alert_severity"]].copy()
        disp["revenue_at_risk"] = disp["revenue_at_risk"].apply(lambda x: f"Rs {x:,.0f}")
        disp["health_score"]    = disp["health_score"].round(1)
        disp["drift_composite"] = disp["drift_composite"].round(3)
        st.dataframe(disp.set_index("merchant_id"), use_container_width=True, height=400)
    with col2:
        fig = px.bar(wdf["alert_severity"].value_counts().reset_index(),
                     x="alert_severity",y="count",color="alert_severity",
                     color_discrete_map=SEV,title="Alert Distribution",template="plotly_dark")
        fig.update_layout(**DARK,height=200,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(wdf,x="drift_composite",y="health_score",
                          color="alert_severity",color_discrete_map=SEV,
                          hover_data=["merchant_id","segment"],
                          title="Drift vs Health Score",template="plotly_dark")
        fig2.update_layout(**DARK,height=220)
        st.plotly_chart(fig2, use_container_width=True)

# S5 WHAT-IF SIMULATOR ─────────────────────────────────────────────────────────
elif screen == "What-If Simulator":
    st.markdown("# What-If Risk Simulator")
    st.caption("Simulate behavioral changes and see the impact on health score and Revenue at Risk.")
    mid = st.selectbox("Select Merchant", df["merchant_id"].sort_values().tolist())
    row = df[df["merchant_id"]==mid].iloc[0]
    ctrl,res = st.columns([1,1.5])
    with ctrl:
        st.markdown("#### Adjust Parameters")
        new_ref = st.slider("Refund Rate (%)",0.0,60.0,
                            float(row["refund_rate"]*100),0.5)/100
        new_cb  = st.slider("Chargeback Rate (%)",0.0,30.0,
                            float(row["chargeback_rate"]*100),0.1)/100
        gmv_chg = st.slider("GMV Change (%)",-80,50,0,5)/100
        dr_adj  = st.slider("Behavioral Drift",0.0,1.0,
                            float(row.get("drift_composite",0.2)),0.01)
        st.markdown("---")
        st.caption("Health = 0.35 x Survival + 0.25 x (1-Refund) + 0.20 x (1-CB) + 0.20 x (1-Drift)")
    orig = row["health_score"]
    surv = float(np.clip(
        float(row["survival_prob_4w"]) - gmv_chg*0.3 - (new_ref-row["refund_rate"])*2, 0, 1))
    ns   = round((0.35*surv + 0.25*max(0,1-new_ref/0.6) +
                  0.20*max(0,1-new_cb/0.3) + 0.20*max(0,1-dr_adj))*100, 1)
    ngmv = float(row.get("gmv_4w_avg",100000))*(1+gmv_chg)
    nrar = ngmv*4*(1-surv)*(1+new_cb)
    with res:
        st.markdown("#### Simulation Result")
        r1,r2 = st.columns(2)
        r1.metric("Original Score", f"{orig:.0f}",  None)
        r1.metric("New Score",      f"{ns:.0f}",  f"{ns-orig:+.1f}")
        r2.metric("Original RaR",   f"Rs {row['revenue_at_risk']:,.0f}")
        r2.metric("New RaR",        f"Rs {nrar:,.0f}", f"Rs {nrar-row['revenue_at_risk']:+,.0f}")
        fig = go.Figure(go.Bar(
            x=["Original","Simulated"], y=[orig,ns],
            marker_color=["#60a5fa","#f87171" if ns<50 else "#34d399"],
            text=[f"{orig:.0f}",f"{ns:.0f}"], textposition="outside",
        ))
        fig.update_layout(title="Health Score: Before vs After",template="plotly_dark",
                          **DARK,height=250,yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)
        tier  = "CRITICAL" if ns<25 else "HIGH" if ns<40 else "MEDIUM" if ns<55 else "LOW"
        color = SEV[tier]
        st.markdown(f'<div style="background:#1e2940;border-left:4px solid {color};'
                    f'padding:14px;border-radius:8px;">'
                    f'<b style="color:{color}">Projected Tier: {tier}</b><br>'
                    f'Survival Prob: {surv:.1%} &nbsp;|&nbsp; '
                    f'Refund: {new_ref:.1%} &nbsp;|&nbsp; '
                    f'Chargeback: {new_cb:.1%}</div>', unsafe_allow_html=True)
'''

with open("dashboard/app.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Written", len(code), "chars,", len(code.split("\n")), "lines")
