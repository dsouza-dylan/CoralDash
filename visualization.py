# visualization.py

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import List

class CoralVisualizationEngine:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def temperature_timeline(self, reef_names: List[str] = None) -> go.Figure:
        df = self.data[self.data['reef_name'].isin(reef_names)] if reef_names else self.data
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=["Sea Surface Temperature (SST)", "Degree Heating Weeks (DHW)"],
                            vertical_spacing=0.1)

        for reef in df['reef_name'].unique():
            reef_data = df[df['reef_name'] == reef]
            fig.add_trace(go.Scatter(
                x=reef_data['date'], y=reef_data['sst'], mode='lines', name=f"{reef} - SST"
            ), row=1, col=1)

            fig.add_trace(go.Scatter(
                x=reef_data['date'], y=reef_data['dhw'], mode='lines', name=f"{reef} - DHW"
            ), row=2, col=1)

        fig.update_layout(height=600, title="SST & DHW Over Time")
        fig.update_yaxes(title_text="°C", row=1, col=1)
        fig.update_yaxes(title_text="°C-weeks", row=2, col=1)
        return fig

    def coral_bleaching_scatter(self) -> go.Figure:
        fig = px.scatter(
            self.data, x='dhw', y='bleaching_severity',
            color='reef_name', size='coral_cover',
            hover_data=['date'], title='Bleaching Severity vs. DHW'
        )
        fig.add_hline(y=30, line_dash='dot', line_color='orange',
                      annotation_text="Moderate Bleaching")
        fig.add_vline(x=4, line_dash='dot', line_color='red',
                      annotation_text="DHW Threshold")
        return fig

    def coral_cover_trend(self, reef_name: str) -> go.Figure:
        reef_data = self.data[self.data['reef_name'] == reef_name]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=reef_data['date'], y=reef_data['coral_cover'],
            mode='lines+markers', name='Coral Cover (%)',
            line=dict(color='green')
        ))
        fig.update_layout(
            title=f'Coral Cover Trend — {reef_name}',
            xaxis_title='Date',
            yaxis_title='Coral Cover (%)',
            height=400
        )
        return fig
