import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime
import pytz
import re


def render_move_analysis_tab(moves_df_master, aest_tz):
    """Renders the Move Analysis tab for detailed move inspection."""
    st.header("Detailed Move Analysis")
    st.info(
        "This section shows high-level move analysis. Use the filters to drill down.")

    analysis_df_orig = moves_df_master.copy()

    col1, col2, col3 = st.columns(3)
    with col1:
        move_types = sorted(analysis_df_orig['MoveType'].unique())
        selected_move_types = st.multiselect(
            "Filter by Move Type", options=move_types, default=move_types, key="ma_movetype")
    with col2:
        che_ids_orig = sorted(analysis_df_orig['CHEID'].dropna().unique())
        selected_che_ids_orig = st.multiselect(
            "Filter by CHE ID", options=che_ids_orig, default=che_ids_orig, key="ma_cheid_orig")
    with col3:
        size_classes = sorted(analysis_df_orig['SizeClass'].dropna().unique())
        selected_size_classes = st.multiselect(
            "Filter by Size Class", options=size_classes, default=size_classes, key="ma_sizeclass")

    filtered_moves_orig = analysis_df_orig[
        (analysis_df_orig['MoveType'].isin(selected_move_types)) &
        (analysis_df_orig['CHEID'].isin(selected_che_ids_orig)) &
        (analysis_df_orig['SizeClass'].isin(selected_size_classes))
    ].copy()

    st.metric("Total Moves Matching Filters", f"{len(filtered_moves_orig):,}")

    if filtered_moves_orig.empty:
        st.warning("No moves match the current filter criteria.")
    else:
        st.subheader("Analysis of Filtered Moves")
        che_counts = filtered_moves_orig['CHEID'].value_counts().reset_index()
        fig_che_bar = px.bar(che_counts, x='CHEID', y='count', title="Total Moves per CHE", labels={
                             'count': 'Number of Moves'}, text='count')
        fig_che_bar.update_traces(textposition='outside')
        st.session_state.report_figs['Total Moves per CHE'] = fig_che_bar
        st.plotly_chart(fig_che_bar, use_container_width=True)

        fig_dur_hist = px.histogram(filtered_moves_orig, x="OverallDurationMinutes",
                                    color="MoveType", title="Move Duration by Type (minutes)")
        st.session_state.report_figs['Move Duration by Type'] = fig_dur_hist
        st.plotly_chart(fig_dur_hist, use_container_width=True)

        # Anomaly Detection
        st.subheader("Anomaly Detection for Move Durations")
        filtered_moves_orig['rolling_avg'] = filtered_moves_orig['OverallDurationMinutes'].rolling(
            window=20, min_periods=1).mean()
        filtered_moves_orig['rolling_std'] = filtered_moves_orig['OverallDurationMinutes'].rolling(
            window=20, min_periods=1).std()
        filtered_moves_orig['anomaly'] = filtered_moves_orig['OverallDurationMinutes'] > (
            filtered_moves_orig['rolling_avg'] + 3 * filtered_moves_orig['rolling_std'])
        anomalies = filtered_moves_orig[filtered_moves_orig['anomaly']]

        st.write(
            f"Found {len(anomalies)} anomalous moves (more than 3 standard deviations from rolling average).")
        st.dataframe(anomalies[['ContainerMoveID', 'CHEID',
                     'OverallDurationMinutes', 'rolling_avg', 'rolling_std']])

    st.markdown("---")

    st.subheader("Move Cycle Time Breakdown")
    st.info("Analyze the time taken for each segment of a move. This section is independent of the filters above.")

    cycle_col1, cycle_col2 = st.columns(2)
    with cycle_col1:
        min_ts_date = moves_df_master['CompletionTimestamp'].min().date()
        max_ts_date = moves_df_master['CompletionTimestamp'].max().date()

        proposed_default_start_date = max_ts_date - datetime.timedelta(days=7)

        default_start_date = max(proposed_default_start_date, min_ts_date)

        cycle_date_range = st.date_input(
            "Select Date Range",
            value=(default_start_date, max_ts_date),
            min_value=min_ts_date,
            max_value=max_ts_date,
            key="cycle_date_range"
        )

    with cycle_col2:
        all_che_ids = sorted(moves_df_master['CHEID'].dropna().unique())
        selected_che_ids_cycle = st.multiselect(
            "Filter by CHE ID", options=all_che_ids, default=all_che_ids, key="ma_cheid_cycle")

    chart_type = st.radio("Select Chart Type", ("Bar Chart",
                          "Line Chart"), key="chart_type_selector", horizontal=True)

    if len(cycle_date_range) == 2:
        cycle_start_date = aest_tz.localize(
            datetime.datetime.combine(cycle_date_range[0], datetime.time.min))
        cycle_end_date = aest_tz.localize(
            datetime.datetime.combine(cycle_date_range[1], datetime.time.max))

        cycle_analysis_df = moves_df_master[
            (moves_df_master['CompletionTimestamp'].between(cycle_start_date, cycle_end_date)) &
            (moves_df_master['CHEID'].isin(selected_che_ids_cycle))
        ].copy()

        # Using .loc to prevent SettingWithCopyWarning
        cycle_analysis_df.loc[:, 'Request to Assigned'] = (
            cycle_analysis_df['AssignedTimestamp'] - cycle_analysis_df['ACMRTimestamp']).dt.total_seconds() / 60
        cycle_analysis_df.loc[:, 'Assign to Pick'] = (
            cycle_analysis_df['PickedTimestamp'] - cycle_analysis_df['AssignedTimestamp']).dt.total_seconds() / 60
        cycle_analysis_df.loc[:, 'Pick to Ground'] = (
            cycle_analysis_df['GroundedTimestamp'] - cycle_analysis_df['PickedTimestamp']).dt.total_seconds() / 60

        duration_cols = ['Request to Assigned',
                         'Assign to Pick', 'Pick to Ground']

        cycle_analysis_df.dropna(subset=[
                                 'ACMRTimestamp', 'AssignedTimestamp', 'PickedTimestamp', 'GroundedTimestamp'], inplace=True)
        for col in duration_cols:
            cycle_analysis_df[col] = pd.to_numeric(
                cycle_analysis_df[col], errors='coerce')
        cycle_analysis_df.dropna(subset=duration_cols, inplace=True)

        if cycle_analysis_df.empty:
            st.warning(
                "No data available for the selected date range and CHEs.")
        else:
            cycle_analysis_df['Hour'] = cycle_analysis_df['CompletionTimestamp'].dt.hour
            hourly_avg = cycle_analysis_df.groupby(
                'Hour')[duration_cols].mean().reset_index()

            color_palette = {'Request to Assigned': '#D6A1A1',
                             'Assign to Pick': '#A1B5D6', 'Pick to Ground': '#A1D6B8'}

            if chart_type == "Bar Chart":
                fig_cycle = go.Figure()
                for col in duration_cols:
                    fig_cycle.add_trace(go.Bar(
                        x=hourly_avg['Hour'],
                        y=hourly_avg[col],
                        name=col,
                        marker_color=color_palette[col]
                    ))
                fig_cycle.update_layout(
                    barmode='stack',
                    title='Average Move Cycle Time Breakdown by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Average Duration (minutes)',
                    legend_title='Move Segment'
                )
                st.session_state.report_figs['Move Cycle Time Bar Chart'] = fig_cycle
                st.plotly_chart(fig_cycle, use_container_width=True)

            elif chart_type == "Line Chart":
                fig_cycle = go.Figure()
                for col in duration_cols:
                    fig_cycle.add_trace(go.Scatter(
                        x=hourly_avg['Hour'],
                        y=hourly_avg[col],
                        mode='lines+markers',
                        name=col,
                        line_color=color_palette[col]
                    ))
                fig_cycle.update_layout(
                    title='Average Move Cycle Time Trends by Hour',
                    xaxis_title='Hour of Day',
                    yaxis_title='Average Duration (minutes)',
                    legend_title='Move Segment'
                )
                st.session_state.report_figs['Move Cycle Time Line Chart'] = fig_cycle
                st.plotly_chart(fig_cycle, use_container_width=True)
    else:
        st.info("Please select a valid date range.")


def render_efficiency_analysis_tab(moves_df_time_filtered):
    """Renders the Efficiency Analysis tab with CHE performance and idle time."""
    st.header("Efficiency Analysis")
    st.info("This section provides additional metrics for deeper analysis of operational efficiency.")
    if moves_df_time_filtered.empty:
        st.warning("No data available for the selected time range.")
    else:
        st.subheader("CHE (Crane) Performance: Assign-to-Completion")
        moves_df_exp = moves_df_time_filtered.dropna(
            subset=['AssignedTimestamp', 'CompletionTimestamp', 'CHEID']).copy()
        if not moves_df_exp.empty:
            moves_df_exp['AssignedToCompletionDurationMinutes'] = (
                moves_df_exp['CompletionTimestamp'] - moves_df_exp['AssignedTimestamp']).dt.total_seconds() / 60
            che_productivity = moves_df_exp.groupby('CHEID').agg(Total_Moves=(
                'ContainerMoveID', 'count'), Avg_Assign_to_Complete_min=('AssignedToCompletionDurationMinutes', 'mean')).reset_index()
            che_productivity['Avg_Assign_to_Complete_min'] = che_productivity['Avg_Assign_to_Complete_min'].round(
                2)
            st.write("Productivity metrics per crane:")
            st.dataframe(che_productivity)
            fig_che_perf_eff = px.bar(che_productivity, x='CHEID', y='Total_Moves', color='Avg_Assign_to_Complete_min', title='Total Moves and Average Assign-to-Completion Duration per CHE',
                                      labels={'Total_Moves': 'Total Moves', 'Avg_Assign_to_Complete_min': 'Avg. Assign-to-Completion (min)'}, text='Total_Moves')
            fig_che_perf_eff.update_traces(textposition='outside')
            st.session_state.report_figs['CHE Assign-to-Completion'] = fig_che_perf_eff
            st.plotly_chart(fig_che_perf_eff, use_container_width=True)

            # Crane Idle Time Analysis
            st.subheader("Crane Idle Time Analysis")
            idle_time_data = []
            for che in moves_df_exp['CHEID'].unique():
                che_moves = moves_df_exp[moves_df_exp['CHEID'] == che].sort_values(
                    by='CompletionTimestamp').reset_index()
                if len(che_moves) > 1:
                    che_moves['next_assigned'] = che_moves['AssignedTimestamp'].shift(
                        -1)
                    che_moves['idle_time_minutes'] = (
                        che_moves['next_assigned'] - che_moves['CompletionTimestamp']).dt.total_seconds() / 60
                    total_idle_time = che_moves['idle_time_minutes'].sum()
                    total_active_time = che_moves['AssignedToCompletionDurationMinutes'].sum(
                    )
                    idle_time_data.append(
                        {'CHEID': che, 'Total Idle Time (min)': total_idle_time, 'Total Active Time (min)': total_active_time})

            if idle_time_data:
                idle_time_df = pd.DataFrame(idle_time_data)
                fig_idle = px.bar(idle_time_df, x='CHEID', y=[
                                  'Total Active Time (min)', 'Total Idle Time (min)'], title="Crane Active vs. Idle Time", barmode='stack')
                st.session_state.report_figs['Crane Idle Time'] = fig_idle
                st.plotly_chart(fig_idle, use_container_width=True)
            else:
                st.warning("Not enough data to calculate idle time.")

        else:
            st.warning(
                "No moves with both 'Assigned' and 'Completion' timestamps found for this analysis.")
        st.markdown("---")
        st.subheader("Move Type Efficiency")
        move_type_counts = moves_df_time_filtered['MoveType'].value_counts()
        total_moves_val = len(moves_df_time_filtered)
        productive_pct = (move_type_counts.get(
            'Productive', 0) / total_moves_val) * 100 if total_moves_val > 0 else 0
        housekeeping_pct = (move_type_counts.get(
            'Housekeeping', 0) / total_moves_val) * 100 if total_moves_val > 0 else 0
        other_pct = 100 - productive_pct - housekeeping_pct
        eff_col1, eff_col2, eff_col3 = st.columns(3)
        eff_col1.metric("Productive Move %", f"{productive_pct:.2f}%")
        eff_col2.metric("Housekeeping Move %", f"{housekeeping_pct:.2f}%")
        eff_col3.metric("Other Move %", f"{other_pct:.2f}%")
        fig_move_eff = px.pie(move_type_counts, values=move_type_counts.values,
                              names=move_type_counts.index, title="Overall Move Type Distribution", hole=0.3)
        fig_move_eff.update_traces(
            textposition='inside', textinfo='percent+label')
        st.session_state.report_figs['Move Type Efficiency Pie'] = fig_move_eff
        st.plotly_chart(fig_move_eff, use_container_width=True)


def render_yard_analysis_tab(moves_df_master):
    """Renders the Yard Analysis tab with heatmaps of yard activity."""
    st.header("Yard Layout and Congestion Analysis")

    YARD_ZONES = [
        {"name": "ASI East", "codes": [], "rows": []},
        {"name": "IMEX EAST (CASC EAST)", "codes": ['ME'], "rows": []},
        {"name": "Interchange EAST", "codes": [], "rows": []},
        {"name": "RAIL 1", "codes": ['IE'], "rows": ['A']},
        {"name": "RAIL 2", "codes": ['IE'], "rows": ['B']},
        {"name": "RAIL 3", "codes": ['IE'], "rows": ['C']},
        {"name": "RAIL 4", "codes": ['IE'], "rows": ['D']},
        {"name": "Interchange WEST", "codes": [], "rows": []},
        {"name": "IMEX WEST (CASC WEST)", "codes": ['MW'], "rows": []},
        {"name": "ASI WEST", "codes": [], "rows": []},
    ]

    def parse_location(location_str):
        if pd.isna(location_str):
            return None, None, None, None
        match = re.match(r"Slot: (\w+)/(\w+)/(\d+)-(\d+)", str(location_str))
        if match:
            return match.groups()
        return None, None, None, None

    def create_parallel_heatmap(df, title, all_bays, specific_rows=None):
        if df.empty:
            return None

        df_copy = df.copy()
        df_copy.loc[:, 'Bay'] = pd.to_numeric(df_copy['Bay'])

        rows_to_display = specific_rows if specific_rows is not None else sorted(
            df_copy['Row'].unique())

        pivot_df = df_copy.pivot_table(
            index='Row', columns='Bay', values='counts', aggfunc='sum')
        pivot_df = pivot_df.reindex(
            index=rows_to_display, columns=all_bays).fillna(0)

        if pivot_df.empty or pivot_df.sum().sum() == 0:
            return None

        fig = go.Figure(data=go.Heatmap(
            z=pivot_df.values,
            x=pivot_df.columns,
            y=pivot_df.index,
            hoverongaps=False,
            colorscale='Reds',
            showscale=False))

        fig.update_layout(
            height=len(rows_to_display) * 35,
            margin=dict(l=0, r=0, t=0, b=0),
            yaxis=dict(showticklabels=False),
            xaxis=dict(showticklabels=False)
        )
        return fig

    moves_df_yard = moves_df_master.copy()
    source_locs = pd.DataFrame(moves_df_yard['SourceLocation'].apply(
        parse_location).tolist(), columns=['Area', 'Row', 'Bay', 'Tier'])
    target_locs = pd.DataFrame(moves_df_yard['TargetLocation'].apply(
        parse_location).tolist(), columns=['Area', 'Row', 'Bay', 'Tier'])

    heatmap_type = st.selectbox("Select Heatmap View", [
                                "Total Moves", "Pick-ups (Source)", "Set-downs (Target)"])

    if heatmap_type == "Pick-ups (Source)":
        heatmap_data = source_locs.dropna(subset=['Area', 'Bay', 'Row'])
    elif heatmap_type == "Set-downs (Target)":
        heatmap_data = target_locs.dropna(subset=['Area', 'Bay', 'Row'])
    else:  # Total Moves
        heatmap_data = pd.concat([
            source_locs.dropna(subset=['Area', 'Bay', 'Row']),
            target_locs.dropna(subset=['Area', 'Bay', 'Row'])
        ])

    if not heatmap_data.empty:
        move_counts = heatmap_data.groupby(
            ['Area', 'Bay', 'Row']).size().reset_index(name='counts')
        min_bay_overall = int(pd.to_numeric(
            heatmap_data['Bay'], errors='coerce').min())
        max_bay_overall = int(pd.to_numeric(
            heatmap_data['Bay'], errors='coerce').max())
        full_bay_range = range(min_bay_overall, max_bay_overall + 1)
    else:
        move_counts = pd.DataFrame()
        full_bay_range = range(1, 101)

    st.markdown("---")

    name_col, heat_col = st.columns([1, 10])

    for zone in YARD_ZONES:
        with name_col:
            st.markdown(
                f'<div style="height: {len(zone["rows"])*35 if zone["rows"] else 50}px; display: flex; align-items: center; justify-content: flex-end; padding-right: 10px;"><b>{zone["name"]}</b></div>', unsafe_allow_html=True)

        with heat_col:
            area_df = pd.DataFrame()
            if not move_counts.empty and zone["codes"]:
                area_df_filtered = move_counts[move_counts['Area'].isin(
                    zone["codes"])]
                if zone["rows"]:
                    area_df = area_df_filtered[area_df_filtered['Row'].isin(
                        zone["rows"])]
                else:
                    area_df = area_df_filtered

            fig = create_parallel_heatmap(
                area_df, title=zone["name"], all_bays=full_bay_range, specific_rows=zone.get("rows") or None)

            if fig:
                st.plotly_chart(fig, use_container_width=True,
                                config={'displayModeBar': False})
            else:
                height = len(zone["rows"]) * 35 if zone["rows"] else 50
                st.markdown(
                    f'<div style="height: {height}px; display: flex; align-items: center; justify-content: center; background-color: #262730; border: 1px dashed #444;">No Activity</div>', unsafe_allow_html=True)

    fig_axis = go.Figure()
    fig_axis.update_layout(
        height=30, margin=dict(l=0, r=0, t=0, b=30),
        xaxis=dict(
            range=[min(full_bay_range), max(full_bay_range)],
            showticklabels=True,
            title="Bay Number"
        ),
        yaxis=dict(showticklabels=False, visible=False),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117'
    )
    st.plotly_chart(fig_axis, use_container_width=True,
                    config={'displayModeBar': False})


def render_data_explorer_tab(all_data):
    """Renders the Data Explorer tab for viewing and filtering raw data tables."""
    st.header("Raw Data Explorer")

    table_to_view = st.selectbox(
        "Select a table to view:", options=list(all_data.keys()))

    if table_to_view:
        df_to_filter = all_data[table_to_view].copy()
        st.subheader(f"Displaying Table: {table_to_view}")
        st.info("Use the filters in the sidebar to narrow down the data in this table.")

        st.sidebar.header("Data Explorer Filters")
        for column in df_to_filter.columns:
            if df_to_filter[column].dtype == 'object' and 1 < df_to_filter[column].nunique() < 100:
                unique_vals = sorted(df_to_filter[column].dropna().unique())
                selected_val = st.sidebar.multiselect(
                    f'Filter by {column}', options=unique_vals, default=unique_vals, key=f"de_{column}")
                df_to_filter = df_to_filter[df_to_filter[column].isin(
                    selected_val)]
            elif pd.api.types.is_numeric_dtype(df_to_filter[column]) and df_to_filter[column].nunique() > 1:
                min_val, max_val = float(df_to_filter[column].min()), float(
                    df_to_filter[column].max())
                if min_val < max_val:
                    selected_range = st.sidebar.slider(
                        f'Filter by {column}', min_val, max_val, (min_val, max_val), key=f"de_slider_{column}")
                    df_to_filter = df_to_filter[df_to_filter[column].between(
                        selected_range[0], selected_range[1])]
            elif 'timestamp' in column.lower() and not pd.api.types.is_string_dtype(df_to_filter[column]) and df_to_filter[column].nunique() > 1:
                try:
                    min_date_de, max_date_de = df_to_filter[column].min(
                    ).date(), df_to_filter[column].max().date()
                    if min_date_de < max_date_de:
                        selected_date = st.sidebar.date_input(f'Filter by {column} date', value=(
                            min_date_de, max_date_de), min_value=min_date_de, max_value=max_date_de, key=f"de_date_{column}")
                        if len(selected_date) == 2:
                            start_date_de = pd.to_datetime(selected_date[0])
                            end_date_de = pd.to_datetime(
                                selected_date[1]) + pd.Timedelta(days=1)
                            df_to_filter = df_to_filter[df_to_filter[column].between(
                                start_date_de, end_date_de)]
                except Exception:
                    pass

        st.dataframe(df_to_filter)
