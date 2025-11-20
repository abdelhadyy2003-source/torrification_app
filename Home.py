# ===== Flow Sheet Tab =====
with tabs[2]:
    st.subheader("Torrefaction Process Flow Sheet (All Simulations)")
    if st.session_state.simulations:
        # إعداد القيم لكل محاكاة
        labels = ["Input Waste", "Water Loss", "Gas & Volatiles", "Ash", "Biochar"]
        node_colors = ['#8B4513','#1E90FF','#FFA500','#808080','#2E8B57']

        # سنستخدم كل محاكاة كمجموعة روابط
        sources = []
        targets = []
        values = []
        colors = []

        for sim_index, sim in enumerate(st.session_state.simulations):
            # كل محاكاة تكون offset لتجنب التداخل
            offset = sim_index * 0.01  # قليل جدا لتوضيح الخطوط
            # روابط Sankey
            sources.extend([0,0,0,0])
            targets.extend([1,2,3,4])
            values.extend([sim['Water Loss (kg)'], sim['Gas & Volatiles (kg)'], sim['Ash (kg)'], sim['Biochar (kg)']])
            colors.extend(node_colors)

        fig_sankey = go.Figure(data=[go.Sankey(
            node=dict(label=labels, pad=15, thickness=20, color=node_colors),
            link=dict(source=sources, target=targets, value=values, color=colors)
        )])
        fig_sankey.update_layout(title_text="Torrefaction Process Flow Sheet (All Simulations)", font_size=12)
        st.plotly_chart(fig_sankey, use_container_width=True)
    else:
        st.info("Run a simulation to see the process flow sheet.")
