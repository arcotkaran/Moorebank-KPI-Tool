import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import datetime
import pytz


def render_terminal_performance_tab(moves_df_master, aest_tz):
    """Renders the Terminal Performance tab with GMPH and crane performance charts."""
    st.header("Terminal Performance")

    def categorize_move(row):
        if pd.notna(row['TrainVisitID']):
            if row['LoadDischargeStatus'] == 'Discharged':
                return 'Train Discharge'
            elif row['LoadDischargeStatus'] == 'Loaded':
                return 'Train Export'
        if row['VirtualBufferID'] == 'GATE':
            return 'Truck Move'
        if row['MoveType'] in ['Housekeeping', 'Shuffle']:
            return 'Stack Move'
        return 'Other'

    moves_df_categorized = moves_df_master.copy()
    moves_df_categorized['MoveCategory'] = moves_df_categorized.apply(
        categorize_move, axis=1)

    min_chart_date = moves_df_categorized['CompletionTimestamp'].min().date()
    max_chart_date = moves_df_categorized['CompletionTimestamp'].max().date()

    chart_date_range = st.date_input(
        "Select Date Range for Chart",
        value=(min_chart_date, max_chart_date),
        min_value=min_chart_date,
        max_value=max_chart_date,
        key="terminal_perf_date_range"
    )

    if len(chart_date_range) == 2:
        chart_start_date = aest_tz.localize(
            datetime.datetime.combine(chart_date_range[0], datetime.time.min))
        chart_end_date = aest_tz.localize(
            datetime.datetime.combine(chart_date_range[1], datetime.time.max))
        time_delta = chart_end_date - chart_start_date

        filtered_df = moves_df_categorized[
            (moves_df_categorized['CompletionTimestamp'] >= chart_start_date) &
            (moves_df_categorized['CompletionTimestamp'] <= chart_end_date)
        ].copy()

        # Dynamic granularity based on date range
        if time_delta.days < 2:
            granularity = 'Hour'
            filtered_df['time_group'] = filtered_df['CompletionTimestamp'].dt.floor(
                'h')
            hours_in_period = 1
            xaxis_title = 'Hour'
            tick_format = '%Y-%m-%d %H:00'
        elif time_delta.days <= 14:
            granularity = 'Day'
            filtered_df['time_group'] = filtered_df['CompletionTimestamp'].dt.date
            hours_in_period = 24
            xaxis_title = 'Date'
            tick_format = '%Y-%m-%d'
        elif time_delta.days <= 90:
            granularity = 'Week'
            filtered_df['time_group'] = filtered_df['CompletionTimestamp'].dt.to_period(
                'W').apply(lambda p: p.start_time).dt.date
            hours_in_period = 24 * 7
            xaxis_title = 'Week Starting'
            tick_format = '%Y-%m-%d'
        else:
            granularity = 'Month'
            filtered_df['time_group'] = filtered_df['CompletionTimestamp'].dt.to_period(
                'M').apply(lambda p: p.start_time).dt.date
            hours_in_period = None
            xaxis_title = 'Month'
            tick_format = '%Y-%m'

        if not filtered_df.empty:
            # --- Chart 1: Daily Moves and GMPH ---
            daily_summary = filtered_df.groupby(
                ['time_group', 'MoveCategory']).size().unstack(fill_value=0)

            all_categories = ['Train Discharge',
                              'Train Export', 'Truck Move', 'Stack Move']
            for cat in all_categories:
                if cat not in daily_summary.columns:
                    daily_summary[cat] = 0

            daily_summary['TotalMoves'] = daily_summary[all_categories].sum(
                axis=1)

            if granularity == 'Month':
                monthly_hours = pd.to_datetime(
                    daily_summary.index).to_series().dt.days_in_month * 24
                daily_summary['GMPH'] = daily_summary['TotalMoves'] / \
                    monthly_hours.values
            else:
                daily_summary['GMPH'] = daily_summary['TotalMoves'] / \
                    hours_in_period

            daily_summary.index = pd.to_datetime(daily_summary.index)

            fig_gmph = make_subplots(specs=[[{"secondary_y": True}]])
            colors = {'Train Discharge': '#F2B8B8', 'Train Export': '#AED6F1',
                      'Truck Move': '#B4D2B1', 'Stack Move': '#FAD7A0'}

            for category in all_categories:
                fig_gmph.add_trace(go.Bar(
                    x=daily_summary.index,
                    y=daily_summary[category],
                    name=category,
                    marker_color=colors.get(category)
                ), secondary_y=False)

            fig_gmph.add_trace(go.Scatter(
                x=daily_summary.index,
                y=daily_summary['GMPH'],
                name='GMPH',
                mode='lines+markers',
                line=dict(color='#74A4BC', width=3)
            ), secondary_y=True)

            fig_gmph.update_layout(
                barmode='stack',
                title_text=f'Moves and Gross Moves Per Hour (GMPH) by {granularity}',
                legend_title_text='Move Type',
                xaxis_title=xaxis_title,
                xaxis=dict(tickformat=tick_format)
            )
            fig_gmph.update_yaxes(title_text="Total Moves", secondary_y=False)
            fig_gmph.update_yaxes(title_text="GMPH", secondary_y=True)
            st.session_state.report_figs['GMPH Chart'] = fig_gmph
            st.plotly_chart(fig_gmph, use_container_width=True)

            # --- Chart 2: Crane Performance ---
            st.markdown("---")
            st.header("Crane Performance")

            crane_summary = filtered_df.groupby(
                ['time_group', 'CHEID']).size().unstack(fill_value=0)
            total_moves_per_crane = filtered_df.groupby('CHEID').size()

            fig_crane_perf = go.Figure()

            crane_colors = px.colors.qualitative.Plotly

            for i, che_id in enumerate(crane_summary.columns):
                legend_name = f"{che_id} -> [{total_moves_per_crane.get(che_id, 0)}]"
                fig_crane_perf.add_trace(go.Bar(
                    x=crane_summary.index,
                    y=crane_summary[che_id],
                    name=legend_name,
                    marker_color=crane_colors[i % len(crane_colors)]
                ))

            fig_crane_perf.update_layout(
                barmode='stack',
                title_text=f"Moves per Crane by {granularity}",
                xaxis_title=xaxis_title,
                yaxis_title=f"Total Moves per {granularity}",
                legend_title_text="Crane -> [Total Moves]",
                xaxis=dict(tickformat=tick_format)
            )
            st.session_state.report_figs['Crane Performance Chart'] = fig_crane_perf
            st.plotly_chart(fig_crane_perf, use_container_width=True)

        else:
            st.warning("No data available for the selected date range.")


def render_overview_tab(moves_df_time_filtered):
    """Renders the Overview tab with high-level metrics and charts."""
    st.header("Overall Performance At a Glance")
    if moves_df_time_filtered.empty:
        st.warning("No data available for the selected time range.")
    else:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Moves", f"{len(moves_df_time_filtered):,}")
        avg_duration = moves_df_time_filtered.get(
            'OverallDurationMinutes', pd.Series(dtype='float')).mean()
        col2.metric("Avg. Move Duration (min)",
                    f"{avg_duration:.2f}" if pd.notna(avg_duration) else "N/A")
        total_teu = moves_df_time_filtered.get(
            'TEU', pd.Series(dtype='float')).sum()
        col3.metric("Total TEU", f"{total_teu:,.0f}" if pd.notna(
            total_teu) else "N/A")
        st.markdown("---")
        st.subheader("Visualizations")
        if 'CompletionTimestamp' in moves_df_time_filtered.columns and not moves_df_time_filtered['CompletionTimestamp'].dropna().empty:
            moves_over_time = moves_df_time_filtered.set_index(
                'CompletionTimestamp').resample('h').size().reset_index(name='Move Count')
            fig_time_overview = px.line(moves_over_time, x='CompletionTimestamp',
                                        y='Move Count', title='MPH - Moves Per Hour', markers=True)
            fig_time_overview.update_traces(line_color='#74A4BC')
            st.session_state.report_figs['Moves Over Time Chart'] = fig_time_overview
            st.plotly_chart(fig_time_overview, use_container_width=True)
        fig_pie_overview = px.pie(
            moves_df_time_filtered, names='MoveType', title='Move Type Proportions', hole=0.3)
        fig_pie_overview.update_traces(
            textposition='inside', textinfo='percent+label')
        st.session_state.report_figs['Move Type Proportions Chart'] = fig_pie_overview
        st.plotly_chart(fig_pie_overview, use_container_width=True)


def render_train_performance_tab(train_df, moves_df_master, reseats_df_master, start_datetime, end_datetime):
    """Renders the Train Performance tab with train-specific metrics and charts."""
    st.header("Train-Specific Performance")
    if train_df.empty:
        st.warning("Train data ('Trains' table) not available for this analysis.")
    else:
        available_trains = sorted(train_df['TrainVisitID'].unique())
        departed_trains_df = train_df[train_df['DepartureTimestamp'].between(
            start_datetime, end_datetime)]
        default_selection = list(departed_trains_df['TrainVisitID'].unique())
        col1_train, col2_train = st.columns([3, 1])
        with col1_train:
            select_all = st.checkbox("Select All Trains")
        if select_all:
            selected_trains_default = available_trains
        else:
            selected_trains_default = default_selection
        selected_trains = st.multiselect(
            "Select Train Visits", options=available_trains, default=selected_trains_default)
        if not selected_trains:
            st.info("Please select at least one train visit to see the analysis.")
        else:
            train_details_filtered_df = train_df[train_df['TrainVisitID'].isin(
                selected_trains)].copy()
            train_moves_df = moves_df_master[moves_df_master['TrainVisitID'].isin(
                selected_trains)].copy()
            st.subheader("Container & TEU Summary")
            loads_df = train_moves_df[train_moves_df['LoadDischargeStatus'] == 'Loaded']
            discharges_df = train_moves_df[train_moves_df['LoadDischargeStatus'] == 'Discharged']

            reseats_df = reseats_df_master[reseats_df_master['TrainVisitID'].isin(
                selected_trains)]

            loaded_containers = len(loads_df)
            discharged_containers = len(discharges_df)
            reseat_containers = len(reseats_df)

            reseat_performance = (
                reseat_containers / loaded_containers) * 100 if loaded_containers > 0 else 0

            t_col1, t_col2, t_col3 = st.columns(3)
            t_col1.metric("Loaded Containers", f"{loaded_containers:,}")
            t_col2.metric("Discharged Containers",
                          f"{discharged_containers:,}")
            t_col3.metric("Reseat Moves", f"{reseat_containers:,}")
            t_col4, t_col5, t_col6 = st.columns(3)
            t_col4.metric("Loaded TEU", f"{loads_df['TEU'].sum():,.2f}")
            t_col5.metric("Discharged TEU",
                          f"{discharges_df['TEU'].sum():,.2f}")
            t_col6.metric("Reseat Performance", f"{reseat_performance:.2f}%",
                          help="Reseats as a percentage of total load moves. Lower is better.")
            st.markdown("---")
            st.subheader("TMPH - Train Moves Per Hour (Load/Discharge Phases)")
            all_phase_moves = []
            try:
                for index, row in train_details_filtered_df.iterrows():
                    if pd.notna(row['FirstDischargeAssignedTS']) and pd.notna(row['LastDischargeCompletionTS']):
                        discharge_moves = train_moves_df[(train_moves_df['CompletionTimestamp'] >= row['FirstDischargeAssignedTS']) & (
                            train_moves_df['CompletionTimestamp'] <= row['LastDischargeCompletionTS']) & (train_moves_df['LoadDischargeStatus'] == 'Discharged')].copy()
                        if not discharge_moves.empty:
                            discharge_moves['Phase'] = 'Discharge'
                            all_phase_moves.append(discharge_moves)
                    if pd.notna(row['FirstLoadAssignedTS']) and pd.notna(row['LastLoadCompletionTS']):
                        load_moves = train_moves_df[(train_moves_df['CompletionTimestamp'] >= row['FirstLoadAssignedTS']) & (
                            train_moves_df['CompletionTimestamp'] <= row['LastLoadCompletionTS']) & (train_moves_df['LoadDischargeStatus'] == 'Loaded')].copy()
                        if not load_moves.empty:
                            load_moves['Phase'] = 'Load'
                            all_phase_moves.append(load_moves)
            except Exception as e:
                # Assuming 'logger' is not available here, falling back to st.error
                st.error(f"An error occurred during TMPH calculation: {e}")
            if all_phase_moves:
                combined_phase_moves = pd.concat(all_phase_moves)
                if not combined_phase_moves.empty:
                    thirty_min_phase_moves = combined_phase_moves.set_index('CompletionTimestamp').groupby(
                        'Phase').resample('30min').size().reset_index(name='Move Count')
                    fig_ld_hourly = px.line(
                        thirty_min_phase_moves,
                        x='CompletionTimestamp',
                        y='Move Count',
                        color='Phase',
                        title='Train Moves Per 30 Mins (TMPH) by Phase',
                        color_discrete_map={
                            'Load': '#74A4BC', 'Discharge': '#B4D2B1'},
                        markers=True
                    )
                    st.session_state.report_figs['TMPH Chart'] = fig_ld_hourly
                    st.plotly_chart(fig_ld_hourly, use_container_width=True)
                else:
                    st.warning(
                        "No load or discharge moves found in the defined phases for the selected train(s).")
            else:
                st.warning(
                    "Not enough data with defined load/discharge phases to create TMPH graph for the selected train(s).")

            # Train Turnaround Gantt Chart
            st.markdown("---")
            st.subheader("Train Turnaround Gantt Chart")
            gantt_data = []
            for index, row in train_details_filtered_df.iterrows():
                if pd.notna(row['ArrivalTimestamp']) and pd.notna(row['DepartureTimestamp']):
                    gantt_data.append(dict(Task=f"{row['TrainVisitID']} - Initial Wait", Start=row['ArrivalTimestamp'],
                                      Finish=row['FirstDischargeAssignedTS'], Resource="Initial Wait"))
                    gantt_data.append(dict(Task=f"{row['TrainVisitID']} - Discharge", Start=row['FirstDischargeAssignedTS'],
                                      Finish=row['LastDischargeCompletionTS'], Resource="Discharge Phase"))
                    gantt_data.append(dict(Task=f"{row['TrainVisitID']} - Mid-Service Delay",
                                      Start=row['LastDischargeCompletionTS'], Finish=row['FirstLoadAssignedTS'], Resource="Mid-Service Delay"))
                    gantt_data.append(dict(Task=f"{row['TrainVisitID']} - Load", Start=row['FirstLoadAssignedTS'],
                                      Finish=row['LastLoadCompletionTS'], Resource="Load Phase"))
                    gantt_data.append(dict(Task=f"{row['TrainVisitID']} - Final Wait",
                                      Start=row['LastLoadCompletionTS'], Finish=row['DepartureTimestamp'], Resource="Final Wait"))

            if gantt_data:
                gantt_df = pd.DataFrame(gantt_data)
                fig_gantt = px.timeline(gantt_df, x_start="Start", x_end="Finish",
                                        y="Task", color="Resource", title="Train Turnaround Phases")
                fig_gantt.update_yaxes(categoryorder="total ascending")
                st.session_state.report_figs['Train Turnaround Gantt Chart'] = fig_gantt
                st.plotly_chart(fig_gantt, use_container_width=True)
            else:
                st.warning("Not enough data to create Gantt chart.")

            st.markdown("---")
            st.subheader("Crane Productivity (Gross Crane Rate)")
            gcr_filter = st.radio(
                "Select Productivity View",
                ("Overall", "Load", "Discharge"),
                horizontal=True,
                label_visibility="collapsed"
            )
            gcr_moves_df = pd.DataFrame()
            if gcr_filter == "Overall":
                gcr_moves_df = train_moves_df[train_moves_df['LoadDischargeStatus'].isin(
                    ['Loaded', 'Discharged'])]
            elif gcr_filter == "Load":
                gcr_moves_df = train_moves_df[train_moves_df['LoadDischargeStatus'] == 'Loaded']
            elif gcr_filter == "Discharge":
                gcr_moves_df = train_moves_df[train_moves_df['LoadDischargeStatus']
                                              == 'Discharged']
            gcr_moves_df = gcr_moves_df.dropna(
                subset=['CHEID', 'AssignedTimestamp', 'CompletionTimestamp']).copy()
            if not gcr_moves_df.empty:
                gcr_moves_df['DurationMinutes'] = (
                    gcr_moves_df['CompletionTimestamp'] - gcr_moves_df['AssignedTimestamp']).dt.total_seconds() / 60
                che_op_times = gcr_moves_df.groupby('CHEID').agg(
                    move_count=('ContainerMoveID', 'count'),
                    total_duration_minutes=('DurationMinutes', 'sum')
                ).reset_index()
                che_op_times['duration_hours'] = che_op_times['total_duration_minutes'] / 60
                che_op_times.loc[che_op_times['duration_hours']
                                 == 0, 'duration_hours'] = 1
                che_op_times['GCR'] = (
                    che_op_times['move_count'] / che_op_times['duration_hours']).round(2)
                st.dataframe(che_op_times[['CHEID', 'move_count', 'duration_hours', 'GCR']].sort_values(
                    by='GCR', ascending=False))
                fig_gcr = px.bar(
                    che_op_times,
                    x='CHEID',
                    y='GCR',
                    title=f'{gcr_filter} GCR (Moves per Hour of Active Work)',
                    color='GCR',
                    text='GCR',
                    color_continuous_scale=px.colors.sequential.Blues
                )
                fig_gcr.update_traces(textposition='outside')
                st.session_state.report_figs['GCR Chart'] = fig_gcr
                st.plotly_chart(fig_gcr, use_container_width=True)
            else:
                st.warning(
                    f"No {gcr_filter.lower()} moves with valid timestamps for this train.")
