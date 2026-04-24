#!/usr/bin/env python3

'''
Interactive RCP viewer using Dash + Plotly.
Run with: python rcp_dash_app.py
Then open http://localhost:8050 in your browser.
'''

import sys
import os
import base64
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go

from mlperf_logging.rcp_checker.rcp_checker import (
    RCP_Checker, read_submission_file, submission_runs,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUPPORTED_VERSIONS = [
    '1.0.0', '1.1.0', '2.0.0', '2.1.0', '3.0.0', '3.1.0',
    '4.0.0', '4.1.0', '5.0.0', '5.1.0', '6.0.0',
]


def available_benchmarks(usage, version):
    '''Return benchmarks that have an RCP json file for a given usage+version.'''
    rcp_dir = os.path.join(os.path.dirname(__file__), '..', f'{usage}_{version}')
    rcp_dir = os.path.normpath(rcp_dir)
    if not os.path.isdir(rcp_dir):
        return []
    benchmarks = []
    for fname in sorted(os.listdir(rcp_dir)):
        if fname.startswith('rcps_') and fname.endswith('.json'):
            benchmarks.append(fname[len('rcps_'):-len('.json')])
    return benchmarks


def load_rcp_data(usage, version, benchmark, pruned):
    '''Return (bs_list, mean_list, stdev_list, min_list) sorted by BS.'''
    try:
        checker = RCP_Checker(usage, version, benchmark, verbose=False)
    except Exception:
        return [], [], [], []

    rcp_pass_arg = 'pruned_rcps' if pruned else 'full_rcps'
    data = checker._get_rcp_data(rcp_pass_arg)

    records = sorted(data.values(), key=lambda r: r['BS'])
    bs_list    = [r['BS']         for r in records]
    mean_list  = [r['RCP Mean']   for r in records]
    stdev_list = [r['RCP Stdev']  for r in records]
    min_list   = [r['Min Epochs'] for r in records]
    return bs_list, mean_list, stdev_list, min_list


def parse_submission_file(decoded_bytes, filename, usage, version):
    '''
    Write decoded bytes to a temp file, parse with read_submission_file,
    and return a dict with benchmark, bs, epochs, and scaling_factor.
    Returns an error string on failure.
    '''
    suffix = os.path.splitext(filename)[1] or '.txt'
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(decoded_bytes)
        tmp_path = tmp.name

    try:
        use_train_samples = False
        not_converged, subm_epochs, bs, benchmark = read_submission_file(
            tmp_path, version, use_train_samples,
        )
    finally:
        os.unlink(tmp_path)

    if benchmark is None:
        return 'Could not detect benchmark from file.'
    if bs == -1:
        return 'Could not detect global_batch_size from file.'
    if not_converged:
        return f'Run did not converge (benchmark={benchmark}, BS={bs}).'

    # Check this benchmark exists for the given usage+version
    if benchmark not in available_benchmarks(usage, version):
        return (
            f'Benchmark "{benchmark}" has no RCP data for {usage} v{version}. '
            'Try a different version.'
        )

    # Get the RCP record (exact or interpolated) for this BS
    rcp_pass_arg = 'pruned_rcps'
    try:
        checker = RCP_Checker(usage, version, benchmark, verbose=False)
    except Exception as exc:
        return f'Failed to load RCP checker: {exc}'

    rcp_record = checker._find_rcp(bs, rcp_pass_arg)
    if rcp_record is None:
        rcp_min = checker._find_top_min_rcp(bs, rcp_pass_arg)
        rcp_max = checker._find_bottom_max_rcp(bs, rcp_pass_arg)
        if rcp_min is not None and rcp_max is not None:
            _, rcp_record = checker._create_interp_rcp(bs, rcp_min, rcp_max)
        elif rcp_min is None and rcp_max is not None:
            rcp_record = checker._find_min_rcp(rcp_pass_arg)
        else:
            rcp_record = None

    scaling_factor = None
    if rcp_record is not None:
        norm = rcp_record['RCP Mean'] / subm_epochs
        scaling_factor = round(float(max(1.0, norm)), 6)

    return {
        'benchmark': benchmark,
        'bs': bs,
        'epochs': round(float(subm_epochs), 6),
        'scaling_factor': scaling_factor,
        'filename': filename,
    }


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

app = dash.Dash(__name__, title='MLPerf RCP Viewer')

app.layout = html.Div(
    style={'fontFamily': 'sans-serif', 'maxWidth': '1100px', 'margin': '0 auto', 'padding': '24px'},
    children=[
        html.H2('MLPerf RCP Interactive Viewer'),

        # RCP controls row
        html.Div(
            style={'display': 'flex', 'gap': '24px', 'flexWrap': 'wrap', 'marginBottom': '20px'},
            children=[
                html.Div([
                    html.Label('Usage', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='usage-dropdown',
                        options=[
                            {'label': 'Training', 'value': 'training'},
                            {'label': 'HPC',      'value': 'hpc'},
                        ],
                        value='training',
                        clearable=False,
                        style={'width': '160px'},
                    ),
                ]),
                html.Div([
                    html.Label('Version', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='version-dropdown',
                        options=[{'label': v, 'value': v} for v in SUPPORTED_VERSIONS],
                        value='6.0.0',
                        clearable=False,
                        style={'width': '140px'},
                    ),
                ]),
                html.Div([
                    html.Label('Benchmark', style={'fontWeight': 'bold'}),
                    dcc.Dropdown(
                        id='benchmark-dropdown',
                        options=[],
                        value=None,
                        clearable=False,
                        style={'width': '220px'},
                    ),
                ]),
                html.Div([
                    html.Label('RCP type', style={'fontWeight': 'bold'}),
                    dcc.RadioItems(
                        id='pruned-radio',
                        options=[
                            {'label': 'Pruned', 'value': True},
                            {'label': 'Full',   'value': False},
                        ],
                        value=True,
                        labelStyle={'display': 'inline-block', 'marginRight': '12px'},
                    ),
                ], style={'paddingTop': '4px'}),
            ],
        ),

        # Result file upload
        html.Div(
            style={'marginBottom': '20px'},
            children=[
                html.Label('Result file', style={'fontWeight': 'bold', 'display': 'block', 'marginBottom': '6px'}),
                dcc.Upload(
                    id='result-upload',
                    children=html.Div([
                        'Drag & drop or ',
                        html.A('select a result file', style={'cursor': 'pointer', 'color': '#4a90d9'}),
                    ]),
                    style={
                        'width': '100%',
                        'padding': '14px',
                        'borderWidth': '2px',
                        'borderStyle': 'dashed',
                        'borderRadius': '6px',
                        'borderColor': '#aaa',
                        'textAlign': 'center',
                        'backgroundColor': '#fafafa',
                        'cursor': 'pointer',
                    },
                    multiple=False,
                ),
                html.Div(id='upload-status', style={'marginTop': '8px', 'fontSize': '0.9em'}),
            ],
        ),

        # Hidden store for parsed submission data
        dcc.Store(id='submission-store', data=None),

        dcc.Graph(id='rcp-graph', style={'height': '520px'}),

        # Data table
        html.Div(id='rcp-table-container', style={'marginTop': '16px'}),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

@app.callback(
    Output('benchmark-dropdown', 'options'),
    Output('benchmark-dropdown', 'value'),
    Input('usage-dropdown', 'value'),
    Input('version-dropdown', 'value'),
    State('submission-store', 'data'),
)
def update_benchmark_options(usage, version, submission):
    benchmarks = available_benchmarks(usage, version)
    options = [{'label': b, 'value': b} for b in benchmarks]
    # Keep the benchmark from the loaded submission file when available
    if submission and isinstance(submission, dict):
        bmark = submission.get('benchmark')
        value = bmark if bmark in benchmarks else (benchmarks[0] if benchmarks else None)
    else:
        value = benchmarks[0] if benchmarks else None
    return options, value


@app.callback(
    Output('submission-store', 'data'),
    Output('upload-status', 'children'),
    Output('benchmark-dropdown', 'value', allow_duplicate=True),
    Input('result-upload', 'contents'),
    State('result-upload', 'filename'),
    State('usage-dropdown', 'value'),
    State('version-dropdown', 'value'),
    State('benchmark-dropdown', 'options'),
    prevent_initial_call=True,
)
def handle_upload(contents, filename, usage, version, bmark_options):
    if contents is None:
        return None, '', dash.no_update

    _header, encoded = contents.split(',', 1)
    decoded = base64.b64decode(encoded)

    result = parse_submission_file(decoded, filename, usage, version)

    if isinstance(result, str):
        # Error message
        status = html.Span(f'Error: {result}', style={'color': 'red'})
        return None, status, dash.no_update

    bmark = result['benchmark']
    valid_benchmarks = [o['value'] for o in bmark_options]
    bmark_value = bmark if bmark in valid_benchmarks else dash.no_update

    sf = result['scaling_factor']
    sf_text = f'  |  Scaling factor: {sf:.4f}' if sf is not None else ''
    status = html.Span(
        f'✓ Loaded {filename}  |  Benchmark: {bmark}  |  BS: {result["bs"]}'
        f'  |  Epochs: {result["epochs"]}{sf_text}',
        style={'color': 'green'},
    )
    return result, status, bmark_value


@app.callback(
    Output('rcp-graph', 'figure'),
    Output('rcp-table-container', 'children'),
    Input('usage-dropdown', 'value'),
    Input('version-dropdown', 'value'),
    Input('benchmark-dropdown', 'value'),
    Input('pruned-radio', 'value'),
    Input('submission-store', 'data'),
)
def update_graph(usage, version, benchmark, pruned, submission):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        title='No data — select a valid usage / version / benchmark combination.',
        xaxis_title='Batch Size',
        yaxis_title='Epochs to Converge',
    )

    if not benchmark:
        return empty_fig, html.Div()

    bs_list, mean_list, stdev_list, min_list = load_rcp_data(usage, version, benchmark, pruned)

    if not bs_list:
        return empty_fig, html.Div('Could not load RCP data for this selection.')

    rcp_label = 'Pruned' if pruned else 'Full'

    mean_upper = [m + s for m, s in zip(mean_list, stdev_list)]
    mean_lower = [m - s for m, s in zip(mean_list, stdev_list)]

    fig = go.Figure()

    # Filled stdev band
    fig.add_trace(go.Scatter(
        x=bs_list + bs_list[::-1],
        y=mean_upper + mean_lower[::-1],
        fill='toself',
        fillcolor='rgba(99,110,250,0.15)',
        line=dict(color='rgba(0,0,0,0)'),
        hoverinfo='skip',
        showlegend=True,
        name='Mean ± Stdev',
    ))

    # Mean + stdev upper boundary
    fig.add_trace(go.Scatter(
        x=bs_list,
        y=mean_upper,
        mode='lines+markers',
        name='Mean + Stdev',
        marker=dict(size=6, symbol='triangle-up'),
        line=dict(width=1.5, dash='dot', color='rgba(99,110,250,0.7)'),
        hovertemplate='BS: %{x}<br>Mean+Stdev: %{y:.4f}<extra></extra>',
    ))

    # Mean - stdev lower boundary
    fig.add_trace(go.Scatter(
        x=bs_list,
        y=mean_lower,
        mode='lines+markers',
        name='Mean − Stdev',
        marker=dict(size=6, symbol='triangle-down'),
        line=dict(width=1.5, dash='dot', color='rgba(99,110,250,0.7)'),
        hovertemplate='BS: %{x}<br>Mean-Stdev: %{y:.4f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=bs_list,
        y=mean_list,
        mode='lines+markers',
        name='RCP Mean',
        marker=dict(size=8),
        line=dict(width=2),
        hovertemplate='BS: %{x}<br>Mean: %{y:.4f}<extra></extra>',
    ))

    fig.add_trace(go.Scatter(
        x=bs_list,
        y=min_list,
        mode='lines+markers',
        name='Min Epochs',
        marker=dict(size=8, symbol='diamond'),
        line=dict(width=2, dash='dash'),
        hovertemplate='BS: %{x}<br>Min: %{y:.4f}<extra></extra>',
    ))

    # Submission result point
    if submission and isinstance(submission, dict) and submission.get('benchmark') == benchmark:
        sub_bs     = submission['bs']
        sub_epochs = submission['epochs']
        sub_sf     = submission['scaling_factor']
        sub_file   = submission.get('filename', '')

        sf_line = f'<br>Scaling factor: {sub_sf:.4f}' if sub_sf is not None else '<br>Scaling factor: N/A'
        hover_text = (
            f'<b>Submission result</b><br>'
            f'File: {sub_file}<br>'
            f'BS: {sub_bs}<br>'
            f'Epochs: {sub_epochs:.4f}'
            f'{sf_line}'
        )

        fig.add_trace(go.Scatter(
            x=[sub_bs],
            y=[sub_epochs],
            mode='markers',
            name='Submission',
            marker=dict(
                size=14,
                symbol='star',
                color='crimson',
                line=dict(width=1.5, color='darkred'),
            ),
            hovertemplate=hover_text + '<extra></extra>',
        ))

    fig.update_layout(
        title=f'{benchmark}  |  {usage} v{version}  |  {rcp_label} RCPs',
        xaxis_title='Batch Size (BS)',
        yaxis_title='Epochs to Converge',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
        yaxis=dict(showgrid=True, gridcolor='#e5e5e5'),
    )

    # Data table
    header_style = {
        'backgroundColor': '#f0f0f0',
        'fontWeight': 'bold',
        'padding': '6px 12px',
        'border': '1px solid #ccc',
        'textAlign': 'right',
    }
    cell_style = {
        'padding': '5px 12px',
        'border': '1px solid #ddd',
        'textAlign': 'right',
        'fontFamily': 'monospace',
    }

    header_row = html.Tr([
        html.Th('Batch Size', style={**header_style, 'textAlign': 'left'}),
        html.Th('RCP Mean',   style=header_style),
        html.Th('RCP Stdev',  style=header_style),
        html.Th('Min Epochs', style=header_style),
    ])

    data_rows = [
        html.Tr([
            html.Td(bs,             style={**cell_style, 'textAlign': 'left'}),
            html.Td(f'{mean:.4f}',  style=cell_style),
            html.Td(f'{stdev:.4f}', style=cell_style),
            html.Td(f'{mn:.4f}',    style=cell_style),
        ])
        for bs, mean, stdev, mn in zip(bs_list, mean_list, stdev_list, min_list)
    ]

    table = html.Table(
        [html.Thead(header_row), html.Tbody(data_rows)],
        style={'borderCollapse': 'collapse', 'width': '100%'},
    )

    return fig, html.Div([html.H4('RCP Data Table'), table])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Interactive RCP viewer (Dash)')
    parser.add_argument('--host', default='127.0.0.1', help='host to bind to')
    parser.add_argument('--port', type=int, default=8050, help='port to listen on')
    parser.add_argument('--debug', action='store_true', help='enable Dash debug mode')
    args = parser.parse_args()

    print(f'Starting RCP viewer at http://{args.host}:{args.port}/')
    app.run(host=args.host, port=args.port, debug=args.debug)
