import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output

# Optional Mapbox token. If absent, the dashboard falls back to OpenStreetMap.
mapbox_token = os.environ.get("MAPBOX_TOKEN")
if mapbox_token:
    px.set_mapbox_access_token(mapbox_token)

# -----------------------------------------------------------------------------
# 1. CONSTANTS & DATA PREPARATION
# -----------------------------------------------------------------------------
ICD10_DESCRIPTIONS = {
    'C34': 'C34 - Malignant neoplasm of bronchus & lung',
    'E11': 'E11 - Type 2 diabetes mellitus',
    'I21': 'I21 - Acute myocardial infarction',
    'I63': 'I63 - Cerebral infarction (Stroke)',
    'J06': 'J06 - Acute upper respiratory infections',
    'J18': 'J18 - Pneumonia, organism unspecified',
    'J44': 'J44 - Chronic obstructive pulmonary disease (COPD)',
    'J45': 'J45 - Asthma',
    'K35': 'K35 - Acute appendicitis',
    'M54': 'M54 - Dorsalgia (Back pain)',
    'O80': 'O80 - Single spontaneous delivery',
    'R07': 'R07 - Pain in throat and chest',
    'S62': 'S62 - Fracture at wrist and hand level',
    'S93': 'S93 - Dislocation/sprain of ankle and foot'
}

LHD_COORDINATES = {
    'Western Sydney LHD': (-33.805, 150.985, 93.4),
    'South Western Sydney LHD': (-33.925, 150.915, 90.8),
    'Hunter New England LHD': (-32.928, 151.781, 88.5),
    'Nepean Blue Mountains LHD': (-33.750, 150.690, 86.2),
    'Sydney LHD': (-33.890, 151.180, 84.1),
    'Illawarra Shoalhaven LHD': (-34.425, 150.893, 81.6),
    'Northern Sydney LHD': (-33.820, 151.190, 78.9),
    'Murrumbidgee LHD': (-35.118, 147.369, 75.4),
    'Far West LHD': (-31.956, 141.453, 71.8)
}

AGE_ORDER = [
    '0-14 years', '15-24 years', '25-34 years', '35-44 years',
    '45-54 years', '55-64 years', '65-74 years', '75+ years'
]

def load_and_prepare_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(base_dir, "hospital_mockup_data.csv")

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            "hospital_mockup_data.csv was not found. "
            "Place the dataset in the same folder as app.py."
        )

    df = pd.read_csv(filepath)
    df['Triage_Date_Time'] = pd.to_datetime(df['Triage_Date_Time'])
    df['Admission_Date_Time'] = pd.to_datetime(df['Admission_Date_Time'])
    df['Discharge_Date_Time'] = pd.to_datetime(df['Discharge_Date_Time'])
    df['Triage_Date'] = df['Triage_Date_Time'].dt.date
    df['YearQuarter'] = df['Triage_Date_Time'].dt.to_period('Q').astype(str)
    df['YearMonth'] = df['Triage_Date_Time'].dt.to_period('M').astype(str)
    df['ICD10_Label'] = df['ICD-10_Code'].map(lambda x: ICD10_DESCRIPTIONS.get(x, x))
    return df

df = load_and_prepare_data()

min_d = df['Triage_Date'].min()
max_d = df['Triage_Date'].max()
lhd_options = sorted(df['Local_Health_District'].unique())
tier_options = sorted(df['Hospital_Tier'].unique())
adm_options = sorted(df['Admission_Type'].unique())
icd_options = sorted(df['ICD10_Label'].unique())

full_ipd_per_lhd = df[df['Admission_Type'] == 'IPD'].groupby('Local_Health_District')['Length_of_Stay_Days'].sum()
total_days_full = (max_d - min_d).days + 1

# -----------------------------------------------------------------------------
# 2. INITIALIZE DASH APP & DARK THEME CSS
# -----------------------------------------------------------------------------
app = Dash(__name__, title="NSW Statewide Hospital Utilisation Dashboard | Group 105")
server = app.server

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
<style>
            body {
                background-color: #0e1117;
                color: #ffffff;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                margin: 0;
                padding: 18px 24px;
            }
            .kpi-card {
                background-color: #1a202c;
                border: 1px solid #2d3748;
                border-radius: 8px;
                padding: 16px 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.25);
                flex: 1;
            }
            .kpi-title { color: #a0aec0; font-size: 0.9rem; font-weight: 600; margin-bottom: 4px; }
            .kpi-value { color: #ffffff; font-size: 1.65rem; font-weight: 700; margin-bottom: 4px; }
            .kpi-delta-warn { color: #f87171; font-size: 0.85rem; font-weight: 600; }
            .kpi-delta-ok { color: #60a5fa; font-size: 0.85rem; font-weight: 600; }
            .kpi-delta-good { color: #34d399; font-size: 0.85rem; font-weight: 600; }
            .panel-box {
                background-color: #161b22;
                border: 1px solid #2d3748;
                border-radius: 8px;
                padding: 14px 16px;
            }

            /* FORCE BLACK TEXT INSIDE ELEMENT A FILTER BOXES, DATE PICKERS & POPUP MENUS */
            #filter-date, #filter-date *,
            #filter-lhd, #filter-lhd *,
            #filter-tier, #filter-tier *,
            #filter-adm, #filter-adm *,
            #filter-icd, #filter-icd *,
            .dash-dropdown, .dash-dropdown *,
            .dash-dropdown-content, .dash-dropdown-content *,
            .dash-dropdown-option, .dash-dropdown-option *,
            .dash-datepicker, .dash-datepicker *,
            .dash-datepicker-content, .dash-datepicker-content *,
            .DateInput_input, .SingleDatePickerInput, .DateRangePickerInput,
            [role="listbox"], [role="listbox"] *,
            [role="option"], [role="option"] *,
            [role="dialog"], [role="dialog"] *,
            input, select, button {
                color: #000000 !important;
                -webkit-text-fill-color: #000000 !important;
            }

            /* Keep placeholder text slightly muted dark grey so it is readable on white */
            input::placeholder {
                color: #475569 !important;
                -webkit-text-fill-color: #475569 !important;
                opacity: 1 !important;
            }
        /* Keep Quarterly / Monthly radio selector text in Element D pure white */
            #radio-granularity,
            #radio-granularity *,
            #radio-granularity label,
            #radio-granularity span {
                color: #ffffff !important;
                -webkit-text-fill-color: #ffffff !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# -----------------------------------------------------------------------------
# 3. DASHBOARD LAYOUT (ELEMENTS A - G)
# -----------------------------------------------------------------------------
app.layout = html.Div([
    # Header
    html.H2("🏥 STATEWIDE HOSPITAL UTILISATION DASHBOARD", style={'margin': '0 0 4px 0', 'fontWeight': '700'}),
    html.Div(
        "Portfolio prototype using mock/synthetic data | Group 105 | Weekly retrospective batch-feed concept | Illustrative 85% capacity benchmark",
        style={'color': '#94a3b8', 'fontSize': '0.88rem', 'marginBottom': '14px'}
    ),

    # ELEMENT A: Global Filter Panel
    html.Details(open=True, style={'backgroundColor': '#161b22', 'border': '1px solid #2d3748', 'borderRadius': '8px', 'padding': '12px 16px', 'marginBottom': '16px'}, children=[
        html.Summary("🎛️ ELEMENT A: Global Filter Panel (Click to Collapse/Expand)", style={'fontWeight': '600', 'cursor': 'pointer', 'color': '#e2e8f0', 'marginBottom': '10px'}),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1.2fr 1.6fr 1.1fr 1.1fr 1.5fr', 'gap': '12px', 'alignItems': 'start'}, children=[
            html.Div([
                html.Label("Date From / To", style={'fontSize': '0.82rem', 'color': '#cbd5e1', 'display': 'block', 'marginBottom': '4px'}),
                dcc.DatePickerRange(
                    id='filter-date',
                    min_date_allowed=min_d,
                    max_date_allowed=max_d,
                    start_date=min_d,
                    end_date=max_d,
                    display_format='YYYY-MM-DD'
                )
            ]),
            html.Div([
                html.Label("LHD Region", style={'fontSize': '0.82rem', 'color': '#cbd5e1', 'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='filter-lhd',
                    options=[{'label': x, 'value': x} for x in lhd_options],
                    value=lhd_options,
                    multi=True,
                    placeholder="Select LHD Regions..."
                )
            ]),
            html.Div([
                html.Label("Hospital Tier", style={'fontSize': '0.82rem', 'color': '#cbd5e1', 'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='filter-tier',
                    options=[{'label': x, 'value': x} for x in tier_options],
                    value=tier_options,
                    multi=True,
                    placeholder="Select Tiers..."
                )
            ]),
            html.Div([
                html.Label("Admission Type", style={'fontSize': '0.82rem', 'color': '#cbd5e1', 'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='filter-adm',
                    options=[{'label': x, 'value': x} for x in adm_options],
                    value=adm_options,
                    multi=True,
                    placeholder="Select Admission Type..."
                )
            ]),
            html.Div([
                html.Label("ICD-10 Diagnostic Code (All if empty)", style={'fontSize': '0.82rem', 'color': '#cbd5e1', 'display': 'block', 'marginBottom': '4px'}),
                dcc.Dropdown(
                    id='filter-icd',
                    options=[{'label': x, 'value': x} for x in icd_options],
                    value=[],
                    multi=True,
                    placeholder="All ICD-10 Diagnostic Codes..."
                )
            ]),
        ])
    ]),

    # ELEMENT B: KPI Cards Container
    html.Div(id='kpi-container', style={'display': 'flex', 'gap': '16px', 'marginBottom': '18px'}),

    # MIDDLE ROW: Elements C, D, F
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1.05fr 1.1fr 1.05fr', 'gap': '16px', 'marginBottom': '18px'}, children=[
        # Element C
        html.Div(className='panel-box', children=[
            html.H4("Bed Occupancy Rate by LHD", style={'margin': '0 0 4px 0', 'fontSize': '1.02rem'}),
            html.Div("Cross-hatched bars (//) & ⚠️ indicate districts breaching the 85% Infrastructure Utilization Benchmark.", style={'color': '#94a3b8', 'fontSize': '0.78rem', 'marginBottom': '6px'}),
            dcc.Graph(id='graph-c', config={'displayModeBar': False})
        ]),

        # Element D
        html.Div(className='panel-box', children=[
            html.Div(style={'display': 'flex', 'justifyContent': 'space-between', 'alignItems': 'center', 'marginBottom': '4px'}, children=[
                html.H4("NSW Hospital Activity Trend", style={'margin': '0', 'fontSize': '1.02rem'}),
                dcc.RadioItems(
                    id='radio-granularity',
                    options=[{'label': html.Span(' Quarterly', style={'color': '#ffffff', 'paddingLeft': '4px'}), 'value': 'Quarterly'},
        {'label': html.Span(' Monthly', style={'color': '#ffffff', 'paddingLeft': '4px'}), 'value': 'Monthly'}],
                    value='Quarterly',
                    inline=True,
                    style={'fontSize': '0.82rem', 'color': '#ffffff'},
                    labelStyle={'marginLeft': '10px', 'cursor': 'pointer'}
                )
            ]),
            html.Div("Solid line = IPD Admissions | Dashed line = OPD Presentations", style={'color': '#94a3b8', 'fontSize': '0.78rem', 'marginBottom': '6px'}),
            dcc.Graph(id='graph-d', config={'displayModeBar': False})
        ]),

        # Element F
        html.Div(className='panel-box', children=[
            html.H4("IPD Average Length of Stay", style={'margin': '0 0 4px 0', 'fontSize': '1.02rem'}),
            html.Div(id='caption-f', style={'color': '#94a3b8', 'fontSize': '0.78rem', 'marginBottom': '6px'}),
            dcc.Graph(id='graph-f', config={'displayModeBar': False})
        ]),
    ]),

    # BOTTOM ROW: Elements E & G
    html.Div(style={'display': 'grid', 'gridTemplateColumns': '1.8fr 1.2fr', 'gap': '16px'}, children=[
        # Element E inside app.layout
html.Div(className='panel-box', children=[
    html.H4("Regional NSW Bottleneck & Spatial Map", style={'margin': '0 0 4px 0', 'fontSize': '1.02rem'}),
    html.Div(
        "Scroll to zoom, click-and-drag to pan (Outer halo & ⚠️ badge = >85% Benchmark).",
        style={'color': '#94a3b8', 'fontSize': '0.78rem', 'marginBottom': '6px'}
    ),
    dcc.Graph(
        id='graph-e',
        config={
            'scrollZoom': True,          # <-- Enables mouse-wheel / trackpad pinch zooming
            'displayModeBar': True,      # <-- Shows the zoom (+ / - / reset) buttons on hover
            'displaylogo': False,
            'modeBarButtonsToRemove': ['select2d', 'lasso2d']
        }
    )
]),

        # Element G
        html.Div(className='panel-box', children=[
            html.H4("NSW Patient Age Cohort Split", style={'margin': '0 0 4px 0', 'fontSize': '1.02rem'}),
            html.Div("Clustered horizontal comparison of IPD Admissions (solid) vs. OPD Presentations (striped //) across MECE age brackets.", style={'color': '#94a3b8', 'fontSize': '0.78rem', 'marginBottom': '6px'}),
            dcc.Graph(id='graph-g', config={'displayModeBar': False})
        ])
    ])
])

# -----------------------------------------------------------------------------
# 4. INTERACTIVE CALLBACK
# -----------------------------------------------------------------------------
@app.callback(
    Output('kpi-container', 'children'),
    Output('graph-c', 'figure'),
    Output('graph-d', 'figure'),
    Output('caption-f', 'children'),
    Output('graph-f', 'figure'),
    Output('graph-e', 'figure'),
    Output('graph-g', 'figure'),
    Input('filter-date', 'start_date'),
    Input('filter-date', 'end_date'),
    Input('filter-lhd', 'value'),
    Input('filter-tier', 'value'),
    Input('filter-adm', 'value'),
    Input('filter-icd', 'value'),
    Input('radio-granularity', 'value')
)
def update_dashboard(start_date, end_date, selected_lhds, selected_tiers, selected_adm, selected_icds, time_granularity):
    filtered_df = df.copy()

    if start_date and end_date:
        s_d = pd.to_datetime(start_date).date()
        e_d = pd.to_datetime(end_date).date()
        filtered_df = filtered_df[(filtered_df['Triage_Date'] >= s_d) & (filtered_df['Triage_Date'] <= e_d)]
    if selected_lhds:
        filtered_df = filtered_df[filtered_df['Local_Health_District'].isin(selected_lhds)]
    if selected_tiers:
        filtered_df = filtered_df[filtered_df['Hospital_Tier'].isin(selected_tiers)]
    if selected_adm:
        filtered_df = filtered_df[filtered_df['Admission_Type'].isin(selected_adm)]
    if selected_icds:
        filtered_df = filtered_df[filtered_df['ICD10_Label'].isin(selected_icds)]

    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)',
            annotations=[dict(text="No records match filter criteria", showarrow=False, font=dict(size=14, color="white"))]
        )
        warn_card = [html.Div("⚠️ No records match the current filter selection. Please broaden your filter criteria.", className='kpi-card')]
        return warn_card, empty_fig, empty_fig, "No IPD data available", empty_fig, empty_fig, empty_fig

    # --- KPI & LHD Calculations ---
    total_encounters = len(filtered_df)
    ipd_sub = filtered_df[filtered_df['Admission_Type'] == 'IPD']
    opd_sub = filtered_df[filtered_df['Admission_Type'] == 'OPD']
    ipd_count, opd_count = len(ipd_sub), len(opd_sub)
    alos_val = ipd_sub['Length_of_Stay_Days'].mean() if ipd_count > 0 else 0.0
    alos_median = ipd_sub['Length_of_Stay_Days'].median() if ipd_count > 0 else 0.0

    selected_days = max((filtered_df['Triage_Date'].max() - filtered_df['Triage_Date'].min()).days + 1, 1)

    lhd_summary = filtered_df.groupby('Local_Health_District').agg(
        Encounters=('Patient_ID', 'count'),
        IPD_Volume=('Admission_Type', lambda x: (x == 'IPD').sum()),
        OPD_Volume=('Admission_Type', lambda x: (x == 'OPD').sum()),
        IPD_Bed_Days=('Length_of_Stay_Days', 'sum'),
        Baseline_Beds=('Baseline_Bed_Count', 'max'),
        Hospital_Tier=('Hospital_Tier', 'first')
    ).reset_index()

    def compute_calibrated_bor(row):
        lhd_name = row['Local_Health_District']
        base_target_bor = LHD_COORDINATES.get(lhd_name, (0, 0, 82.0))[2]
        full_daily_bed_days = full_ipd_per_lhd.get(lhd_name, 1.0) / total_days_full
        filtered_daily_bed_days = row['IPD_Bed_Days'] / selected_days
        intensity_ratio = filtered_daily_bed_days / max(full_daily_bed_days, 0.01)
        if selected_icds:
            scale_factor = len(ICD10_DESCRIPTIONS) / max(len(selected_icds), 1)
            intensity_ratio = intensity_ratio * scale_factor
        return round(min(max(base_target_bor * intensity_ratio, 15.0), 100.0), 1)

    lhd_summary['Occupancy_Rate'] = lhd_summary.apply(compute_calibrated_bor, axis=1)
    lhd_summary['Lat'] = lhd_summary['Local_Health_District'].map(lambda x: LHD_COORDINATES[x][0])
    lhd_summary['Lon'] = lhd_summary['Local_Health_District'].map(lambda x: LHD_COORDINATES[x][1])
    lhd_summary['Status'] = np.where(
        lhd_summary['Occupancy_Rate'] >= 85.0,
        '⚠️ Exceeds 85% Benchmark',
        '✓ Within Benchmark (<85%)'
    )

    overall_bor = round(lhd_summary['Occupancy_Rate'].mean(), 1)
    bor_breach = overall_bor >= 85.0
    badge = "⚠️ EXCEEDS 85% BENCHMARK" if bor_breach else "✓ WITHIN 85% BENCHMARK"
    delta_cls = "kpi-delta-warn" if bor_breach else "kpi-delta-ok"

    # --- ELEMENT B: KPI Cards ---
    kpi_cards = [
        html.Div(className="kpi-card", children=[
            html.Div("KPI: Infrastructure Utilization (BOR)", className="kpi-title"),
            html.Div([f"{overall_bor:.1f}%  ", html.Span(f"[{badge}]", style={'fontSize': '0.95rem'})], className="kpi-value"),
            html.Div(f"Selected-period view | Static Baseline Beds: {lhd_summary['Baseline_Beds'].sum():,}", className=delta_cls)
        ]),
        html.Div(className="kpi-card", children=[
            html.Div("KPI: Cumulative Patient Encounters", className="kpi-title"),
            html.Div([f"{total_encounters:,}  ", html.Span(f"({ipd_count:,} IPD | {opd_count:,} OPD)", style={'fontSize': '0.95rem', 'color': '#cbd5e1'})], className="kpi-value"),
            html.Div(f"Admission mix: {ipd_count/max(total_encounters,1)*100:.1f}% IPD | {opd_count/max(total_encounters,1)*100:.1f}% OPD", className="kpi-delta-ok")
        ]),
        html.Div(className="kpi-card", children=[
            html.Div("KPI: IPD Average Length of Stay (ALOS)", className="kpi-title"),
            html.Div([f"{alos_val:.2f} Days  ", html.Span(f"(Median: {alos_median:.2f}d)", style={'fontSize': '0.95rem', 'color': '#cbd5e1'})], className="kpi-value"),
            html.Div("Calculated from IPD encounters in the current filter selection", className="kpi-delta-good")
        ])
    ]

    # --- ELEMENT C: Bed Occupancy Rate by LHD ---
    lhd_sorted = lhd_summary.sort_values('Occupancy_Rate', ascending=True)
    fig_c = px.bar(
        lhd_sorted, x='Occupancy_Rate', y='Local_Health_District', color='Status', pattern_shape='Status',
        orientation='h',
        text=lhd_sorted.apply(lambda r: f" {r['Occupancy_Rate']:.1f}% ⚠️" if r['Occupancy_Rate'] >= 85.0 else f" {r['Occupancy_Rate']:.1f}%", axis=1),
        color_discrete_map={'⚠️ Exceeds 85% Benchmark': '#ef4444', '✓ Within Benchmark (<85%)': '#3b82f6'},
        pattern_shape_map={'⚠️ Exceeds 85% Benchmark': '/', '✓ Within Benchmark (<85%)': ''},
        custom_data=['Hospital_Tier', 'Baseline_Beds', 'IPD_Volume'],
        template='plotly_dark'
    )
    fig_c.add_vline(x=85.0, line_dash="dash", line_color="#facc15", line_width=2, annotation_text="85% Benchmark", annotation_position="bottom right", annotation_font_color="#facc15")
    fig_c.update_traces(textposition='outside')
    fig_c.update_layout(
        height=360, xaxis_range=[0, 114], xaxis_title="Infrastructure Utilization Rate (%)", yaxis_title="",
        legend_title_text="", legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=20, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)'
    )

    # --- ELEMENT D: NSW Hospital Activity Trend ---
    time_col = 'YearQuarter' if time_granularity == "Quarterly" else 'YearMonth'
    trend_df = filtered_df.groupby([time_col, 'Admission_Type']).size().reset_index(name='Volume')
    if time_granularity == "Quarterly" and '2026Q3' in trend_df[time_col].values and len(trend_df[time_col].unique()) > 2:
        trend_df = trend_df[trend_df[time_col] != '2026Q3']

    fig_d = px.line(
        trend_df, x=time_col, y='Volume', color='Admission_Type', symbol='Admission_Type',
        markers=True, color_discrete_map={'IPD': '#3b82f6', 'OPD': '#f59e0b'},
        line_dash='Admission_Type', line_dash_map={'IPD': 'solid', 'OPD': 'dash'},
        template='plotly_dark'
    )
    fig_d.update_layout(
        height=360, xaxis_title="Period", yaxis_title="Patient Encounters", legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)'
    )

    # --- ELEMENT F: IPD Average Length of Stay ---
    if not ipd_sub.empty:
        alos_trend = ipd_sub.groupby('YearQuarter')['Length_of_Stay_Days'].mean().reset_index()
        min_alos, max_alos = alos_trend['Length_of_Stay_Days'].min(), alos_trend['Length_of_Stay_Days'].max()
        latest_alos = alos_trend['Length_of_Stay_Days'].iloc[-1]
        caption_f_text = f"Operating Band: {min_alos:.2f} – {max_alos:.2f} Days | Latest Quarter: {latest_alos:.2f} Days"
        fig_f = go.Figure()
        fig_f.add_trace(go.Scatter(
            x=alos_trend['YearQuarter'], y=alos_trend['Length_of_Stay_Days'],
            mode='lines+markers', fill='tozeroy',
            line=dict(color='#38bdf8', width=2.5), fillcolor='rgba(56, 189, 248, 0.2)', name='IPD ALOS (Days)'
        ))
        fig_f.update_layout(
            template='plotly_dark', height=360,
            yaxis_range=[max(0, min_alos - 0.6), max_alos + 0.5],
            xaxis_title="Quarter", yaxis_title="Average Length of Stay (Days)",
            margin=dict(l=10, r=10, t=30, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)'
        )
    else:
        caption_f_text = "Select 'IPD' in Admission Type to view Inpatient Length of Stay trends."
        fig_f = go.Figure().update_layout(
            template='plotly_dark', height=360, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)'
        )

    # --- ELEMENT E: Regional NSW Bottleneck & Spatial Map ---
    # If mapbox token is missing, use open-street-map vector style instead of darkmatter style
    map_style = "carto-darkmatter" if mapbox_token else "open-street-map"

    fig_map = px.scatter_mapbox(
        lhd_summary, lat='Lat', lon='Lon', size='Encounters', color='Status',
        hover_name='Local_Health_District',
        hover_data={'Hospital_Tier': True, 'Occupancy_Rate': ':.1f', 'Baseline_Beds': True, 'IPD_Volume': ':,', 'OPD_Volume': ':,', 'Lat': False, 'Lon': False},
        color_discrete_map={'⚠️ Exceeds 85% Benchmark': '#ef4444', '✓ Within Benchmark (<85%)': '#3b82f6'},
        size_max=28, zoom=5.2, center=dict(lat=-33.3, lon=148.5), mapbox_style=map_style,
        template='plotly_dark'
    )
    breach_df = lhd_summary[lhd_summary['Occupancy_Rate'] >= 85.0]
    fig_map.add_trace(
        go.Scattermapbox(
            lat=breach_df['Lat'], lon=breach_df['Lon'], mode='markers+text',
            marker=dict(size=34, color='rgba(255, 255, 255, 0.25)', allowoverlap=True),
            text=breach_df['Occupancy_Rate'].apply(lambda x: f"⚠️ {x:.1f}%"),
            textposition="top center", textfont=dict(size=11, color="#facc15"),
            name="⚠️ >85% Alert Ring", hoverinfo='skip'
        )
    )
    fig_map.update_layout(
        height=410, legend_title_text="Infrastructure Status",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=0, r=0, t=25, b=0), paper_bgcolor='rgba(0,0,0,0)'
    )

    # --- ELEMENT G: NSW Patient Age Cohort Split ---
    cohort_df = filtered_df.groupby(['Age_Cohort', 'Admission_Type']).size().reset_index(name='Patient_Volume')
    cohort_df['Age_Cohort'] = pd.Categorical(cohort_df['Age_Cohort'], categories=AGE_ORDER, ordered=True)
    cohort_df = cohort_df.sort_values('Age_Cohort')
    fig_g = px.bar(
        cohort_df, x='Patient_Volume', y='Age_Cohort', color='Admission_Type',
        barmode='group', orientation='h',
        color_discrete_map={'IPD': '#3b82f6', 'OPD': '#f59e0b'},
        pattern_shape='Admission_Type', pattern_shape_map={'IPD': '', 'OPD': '/'},
        template='plotly_dark'
    )
    fig_g.update_layout(
        height=410, legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=25, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(26,32,44,0.6)'
    )

    return kpi_cards, fig_c, fig_d, caption_f_text, fig_f, fig_map, fig_g

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
