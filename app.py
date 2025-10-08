"""
Main application script for the Moorebank KPI Dashboard.

This script initializes the Streamlit application, loads data from an SQLite database,
and orchestrates the rendering of various UI components (tabs) defined in separate modules.
"""

import datetime
import logging
import os
import sqlite3
import sys

import pandas as pd
import pytz
import streamlit as st

# Ensure the current directory is in the Python path for local imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    # Import tab rendering functions from local modules
    from performance_tabs import (render_overview_tab,
                                  render_terminal_performance_tab,
                                  render_train_performance_tab)
    from analysis_tabs import (render_data_explorer_tab,
                               render_efficiency_analysis_tab,
                               render_move_analysis_tab, render_yard_analysis_tab)
except ImportError:
    st.error(
        "Failed to import necessary modules (`performance_tabs.py`, `analysis_tabs.py`). "
        f"Please ensure these files are in the same directory as your main script "
        f"('{os.path.basename(__file__)}')."
    )
    st.stop()


# --- App Configuration ---
APP_CONFIG = {
    "db_path": r'C:\Users\arcotka\Documents\Logs\Output\Analysis_Reports\eis_kpi_data.db',
    "sql_path": 'eis_kpi_data.db.sql',
    "log_dir": r'C:\Users\arcotka\Documents\Logs\Output\Analysis_Reports'
}

# --- Core Logic Functions ---


def setup_logging():
    """Initializes file-based logging for the dashboard session."""
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        try:
            os.makedirs(APP_CONFIG["log_dir"], exist_ok=True)
            log_filename = (
                f"dashboard_session_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            )
            log_file_path = os.path.join(APP_CONFIG["log_dir"], log_filename)

            file_handler = logging.FileHandler(log_file_path, mode='w')
            log_formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(name)s - %(message)s')
            file_handler.setFormatter(log_formatter)

            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(log_formatter)

            logger.setLevel(logging.DEBUG)
            logger.addHandler(file_handler)
            logger.addHandler(stream_handler)

            logger.info(
                "Dashboard session started. Logging to: %s", log_file_path)
            st.session_state.logger_configured = True

        except (IOError, OSError) as e:
            st.error(f"Failed to configure file logging: {e}")
            logger.error(
                "Could not write to log file, logging to console only.", exc_info=True)
    return logger

# --- Kaleido Engine Configuration Block (DISABLED) ---
# try:
#     logger.info("Initializing Kaleido engine...")
#     import plotly.io as pio
#     pio.kaleido.scope.chromium_args = ("--headless", "--no-sandbox",
#                                        "--disable-dev-shm-usage", "--disable-gpu")
#     logger.debug("Performing a test image export to validate Kaleido engine.")
#     pio.to_image(go.Figure(), format="png")
#     logger.info("Kaleido engine initialized successfully.")
# except Exception as e:
#     logger.error("Failed to initialize Kaleido engine: %s", e, exc_info=True)
#     st.error(f"Failed to initialize PDF report engine (Kaleido): {e}.")


@st.cache_resource
def get_db_connection(db_path):
    """Creates and caches a connection to the SQLite database."""
    try:
        return sqlite3.connect(db_path, check_same_thread=False)
    except sqlite3.OperationalError as e:
        st.error(
            f"Error connecting to database: {e}. Please ensure it is a valid SQLite file.")
        return None


@st.cache_data(ttl=600)
def load_all_data(_conn, _logger):
    """Loads all tables from the database and performs initial data processing."""
    if _conn is None:
        return {}
    data = {}
    try:
        cursor = _conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        available_tables = [table[0] for table in cursor.fetchall()]
        _logger.info("Discovered Tables in Database: %s", available_tables)
        for table_name in available_tables:
            query = f'SELECT * FROM "{table_name}"'
            data[table_name] = pd.read_sql_query(query, _conn)
            _logger.info("Successfully loaded table '%s' with %d rows.",
                         table_name, len(data[table_name]))

    except (sqlite3.Error, pd.errors.DatabaseError) as e:
        _logger.error(
            "An unexpected error occurred while loading data: %s", e, exc_info=True)
        st.error(f"An error occurred while loading data: {e}")
        return {}

    aest_tz = pytz.timezone('Australia/Sydney')

    for df_name, df in data.items():
        ts_cols = [c for c in df.columns if 'timestamp' in c.lower()
                   or 'ts' in c.lower()]
        for col in ts_cols:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if pd.api.types.is_datetime64_any_dtype(df[col]):
                        if df[col].dt.tz is None:
                            df[col] = df[col].dt.tz_localize(
                                'UTC').dt.tz_convert(aest_tz)
                        else:
                            df[col] = df[col].dt.tz_convert(aest_tz)
                except Exception as e:  # Broad exception for various parsing errors
                    _logger.warning("Could not convert column %s in table %s to datetime: %s",
                                    col, df_name, e)

        if df_name.lower() == 'moves':
            if 'CompletionTimestamp' in df.columns and 'AssignedTimestamp' in df.columns:
                df['ReferenceTimestamp'] = df['CompletionTimestamp'].fillna(
                    df['AssignedTimestamp'])
            for col in df.columns:
                if 'durationseconds' in col.lower():
                    new_col_name = col.replace('Seconds', 'Minutes')
                    df[new_col_name] = df[col] / 60
        data[df_name] = df

    return data


def initialize_database(db_path, sql_path, logger):
    """Initializes the database from a SQL script if it doesn't exist."""
    if not os.path.exists(db_path):
        logger.info("Database not found at %s. Creating from %s...",
                    db_path, sql_path)
        if not os.path.exists(sql_path):
            st.error(
                f"SQL file not found at {sql_path}. Cannot initialize database.")
            st.stop()
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            with open(sql_path, 'r', encoding='utf-8') as sql_file:
                sql_script = sql_file.read()
            cursor.executescript(sql_script)
            conn.commit()
            conn.close()
            logger.info("Database created and populated successfully.")
        except (IOError, sqlite3.Error) as e:
            logger.error(
                "Failed to create database from SQL file: %s", e, exc_info=True)
            st.error(f"Failed to initialize database from {sql_path}: {e}")
            st.stop()

# --- Main Application Flow ---


def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(
        page_title="Moorebank KPI Dashboard",
        page_icon="🚆",
        layout="wide",
    )

    logger = setup_logging()

    initialize_database(APP_CONFIG["db_path"], APP_CONFIG["sql_path"], logger)

    if not os.path.exists(APP_CONFIG["db_path"]):
        st.error(
            f"Database file not found at specified path: {APP_CONFIG['db_path']}")
        st.info(
            "Please ensure the database file exists and script has permission to read it.")
        st.stop()

    conn = get_db_connection(APP_CONFIG["db_path"])
    all_data = load_all_data(conn, logger)

    if not all_data or 'Moves' not in all_data or all_data['Moves'].empty:
        st.error(
            "The 'Moves' table is empty or could not be loaded from the database.")
        st.warning(
            "Please ensure the database is valid and contains required tables.")
        st.stop()

    st.title("Moorebank KPI Dashboard")

    moves_df_master = all_data.get('Moves', pd.DataFrame())
    train_df_master = all_data.get('Trains', pd.DataFrame())
    reseats_df_master = all_data.get('Reseats', pd.DataFrame())

    # --- Global Sidebar Filters ---
    st.sidebar.header("Global Filters")
    aest_tz = pytz.timezone('Australia/Sydney')

    if st.sidebar.button("Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    if st.sidebar.checkbox("Auto-Refresh (every 10 minutes)"):
        st.session_state.auto_refresh = True
        # Note: A manual refresh is still required.

    if 'ReferenceTimestamp' in moves_df_master.columns and not \
       moves_df_master['ReferenceTimestamp'].dropna().empty:
        min_date = moves_df_master['ReferenceTimestamp'].min().date()
        max_date = moves_df_master['ReferenceTimestamp'].max().date()
        from_date = st.sidebar.date_input("From Date", min_date, min_value=min_date,
                                          max_value=max_date)
        from_time = st.sidebar.time_input("From Time", datetime.time(0, 0))
        to_date = st.sidebar.date_input("To Date", max_date, min_value=min_date,
                                        max_value=max_date)
        to_time = st.sidebar.time_input("To Time", datetime.time(23, 59))
        start_datetime = aest_tz.localize(
            datetime.datetime.combine(from_date, from_time))
        end_datetime = aest_tz.localize(
            datetime.datetime.combine(to_date, to_time))
        moves_df_time_filtered = moves_df_master[
            (moves_df_master['ReferenceTimestamp'] >= start_datetime) &
            (moves_df_master['ReferenceTimestamp'] <= end_datetime)
        ]
        train_df = train_df_master
    else:
        st.sidebar.warning("No timestamp data available to filter on.")
        moves_df_time_filtered = moves_df_master.copy()
        train_df = train_df_master.copy()

    if 'report_figs' not in st.session_state:
        st.session_state.report_figs = {}

    # --- Tab Creation and Rendering ---
    tab_list = ["Terminal Performance", "Overview", "Train Performance", "Move Analysis",
                "Efficiency Analysis", "Yard Analysis", "Data Explorer"]
    (term_perf_tab, overview_tab, train_perf_tab, move_analysis_tab, efficiency_tab,
     yard_analysis_tab, data_explorer_tab) = st.tabs(tab_list)

    with term_perf_tab:
        render_terminal_performance_tab(moves_df_master, aest_tz)

    with overview_tab:
        render_overview_tab(moves_df_time_filtered)

    with train_perf_tab:
        render_train_performance_tab(train_df, moves_df_master, reseats_df_master,
                                     start_datetime, end_datetime)

    with move_analysis_tab:
        render_move_analysis_tab(moves_df_master, aest_tz)

    with efficiency_tab:
        render_efficiency_analysis_tab(moves_df_time_filtered)

    with yard_analysis_tab:
        render_yard_analysis_tab(moves_df_master)

    with data_explorer_tab:
        render_data_explorer_tab(all_data)


if __name__ == "__main__":
    main()
