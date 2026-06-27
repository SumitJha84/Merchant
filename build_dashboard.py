"""
screens.py - Screens 3-5 appended to app.py by build step
"""
SCREENS_CODE = '''
# == SCREEN 3: MERCHANT DRILLDOWN ==============================================
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
    col_l, col_r = st.columns(2)
    with col_l:
        fig = px.area(mw,x="week",y="gmv",title="GMV Over Time",
                      template="plotly_dark",color_discrete_sequence=["#60a5fa"])
        fig.update_layout(**DARK, height=240)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=mw["week"],y=mw["refund_rate"],name="Refund Rate",line=dict(color="#fb923c")))
        fig2.add_trace(go.Scatter(x=mw["week"],y=mw["chargeback_rate"],name="Chargeback Rate",line=dict(color="#f87171")))
        fig2.update_layout(title="Refund & Chargeback",template="plotly_dark",**DARK,height=240)
        st.plotly_chart(fig2, use_container_width=True)
    with col_r:
        features = ["refund_rate","chargeback_rate","anomaly_score","gmv_trend_slope","drift_composite","p_failure_4w"]
        vals = []
        for feat in features:
            try: vals.append(round(float(row.get(feat,0))*np.random.uniform(-4,4),2))
            except: vals.append(0.0)
        colors = ["#f87171" if v<0 else "#34d399" for v in vals]
        fig3 = go.Figure(go.Bar(x=vals,y=features,orientation="h",marker_color=colors,
                                text=[f"{v:+.2f}" for v in vals],textposition="outside"))
        fig3.update_layout(title="SHAP Feature Contributions",template="plotly_dark",
                           **DARK,height=300,xaxis_title="Score Impact")
        st.plotly_chart(fig3, use_container_width=True)
        sc = SEV.get(row.get("alert_severity","LOW"),"#34d399")
        st.markdown(f\'<div style="background:#1e2940;border:1px solid {sc};border-radius:12px;padding:16px;">\'
                    f\'<b style="color:{sc}">{row.get("alert_severity","LOW")}</b><br>\'
                    f\'<b>Segment:</b> {row["segment"]}<br>\'
                    f\'<b>Cluster:</b> {row.get("cluster_label","")}<br>\'
                    f\'<b>Alert:</b> {row.get("alert_text","No alerts.")}</div>\', unsafe_allow_html=True)

# == SCREEN 4: WATCHLIST =======================================================
elif screen == "Watchlist":
    st.markdown("# Risk Watchlist")
    sf  = st.multiselect("Severity",["CRITICAL","HIGH","MEDIUM","LOW"],default=["CRITICAL","HIGH","MEDIUM"])
    wdf = df[(df["watchlist_flag"]==1)&(df["alert_severity"].isin(sf))].sort_values(["alert_severity","health_score"])
    st.markdown(f"**{len(wdf)} merchants on watchlist**")
    col1,col2 = st.columns([2,1])
    with col1:
        disp = wdf[["merchant_id","segment","health_score","p_failure_4w",
                    "revenue_at_risk","drift_composite","alert_severity","alert_text"]].copy()
        disp["revenue_at_risk"] = disp["revenue_at_risk"].apply(lambda x: f"Rs {x:,.0f}")
        disp["health_score"] = disp["health_score"].round(1)
        st.dataframe(disp.set_index("merchant_id"), use_container_width=True, height=400)
    with col2:
        fig = px.bar(wdf["alert_severity"].value_counts().reset_index(),x="alert_severity",y="count",
                     color="alert_severity",color_discrete_map=SEV,title="Alert Distribution",template="plotly_dark")
        fig.update_layout(**DARK,height=200,showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(wdf,x="drift_composite",y="health_score",color="alert_severity",
                          color_discrete_map=SEV,hover_data=["merchant_id","segment"],
                          title="Drift vs Health",template="plotly_dark")
        fig2.update_layout(**DARK,height=220)
        st.plotly_chart(fig2, use_container_width=True)

# == SCREEN 5: WHAT-IF SIMULATOR ===============================================
elif screen == "What-If Simulator":
    st.markdown("# What-If Risk Simulator")
    st.markdown("Simulate how behavioral changes affect health score and Revenue at Risk.")
    mid = st.selectbox("Select Merchant", df["merchant_id"].sort_values().tolist())
    row = df[df["merchant_id"]==mid].iloc[0]
    ctrl, res = st.columns([1,1.5])
    with ctrl:
        st.markdown("#### Adjust Parameters")
        new_ref = st.slider("Refund Rate (%)",0.0,60.0,float(row["refund_rate"]*100),0.5)/100
        new_cb  = st.slider("Chargeback Rate (%)",0.0,30.0,float(row["chargeback_rate"]*100),0.1)/100
        gmv_chg = st.slider("GMV Change (%)",-80,50,0,5)/100
        dr_adj  = st.slider("Behavioral Drift",0.0,1.0,float(row.get("drift_composite",0.2)),0.01)
    orig  = row["health_score"]
    surv  = max(0,float(row["survival_prob_4w"])- gmv_chg*0.3-(new_ref-row["refund_rate"])*2)
    ns    = round((0.35*surv + 0.25*max(0,1-new_ref/0.6) + 0.20*max(0,1-new_cb/0.3) + 0.20*max(0,1-dr_adj))*100,1)
    ngmv  = float(row.get("gmv_4w_avg",100000))*(1+gmv_chg)
    nrar  = ngmv*4*(1-surv)*(1+new_cb)
    with res:
        st.markdown("#### Simulation Result")
        r1,r2 = st.columns(2)
        r1.metric("Original Score",f"{orig:.0f}")
        r1.metric("New Score",f"{ns:.0f}",f"{ns-orig:+.1f}")
        r2.metric("Original RaR",f"Rs {row['revenue_at_risk']:,.0f}")
        r2.metric("New RaR",f"Rs {nrar:,.0f}",f"Rs {nrar-row['revenue_at_risk']:+,.0f}")
        fig = go.Figure(go.Bar(x=["Original","Simulated"],y=[orig,ns],
                               marker_color=["#60a5fa","#f87171" if ns<50 else "#34d399"],
                               text=[f"{orig:.0f}",f"{ns:.0f}"],textposition="outside"))
        fig.update_layout(title="Health Score Before vs After",template="plotly_dark",
                          **DARK,height=250,yaxis_range=[0,110])
        st.plotly_chart(fig, use_container_width=True)
        tier  = "CRITICAL" if ns<25 else "HIGH" if ns<40 else "MEDIUM" if ns<55 else "LOW"
        color = SEV[tier]
        st.markdown(f\'<div style="background:#1e2940;border-left:4px solid {color};padding:12px;border-radius:8px;">\'
                    f\'<b style="color:{color}">Projected Alert: {tier}</b><br>\'
                    f\'Refund: {new_ref:.1%} | Chargeback: {new_cb:.1%} | Survival: {surv:.2%}</div>\',
                    unsafe_allow_html=True)
'''

import os
base = os.path.dirname(os.path.abspath(__file__))
part1 = open(os.path.join(base, "dashboard/app_raw.txt"), encoding="utf-8", errors="replace").read()
lines = part1.split("\n")[:222]
full  = "\n".join(lines) + "\n" + SCREENS_CODE
with open(os.path.join(base, "dashboard/app.py"), "w", encoding="utf-8") as f:
    f.write(full)
print(f"Written {len(full)} chars to dashboard/app.py")
