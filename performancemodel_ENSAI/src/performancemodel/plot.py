import plotly.graph_objects as go
import numpy as np






def plot_trajectory(trajectory):
    """
    trajectory : liste de listes [[v1, v2], [v1, v2], ...]
    """
    values = np.array(trajectory)
    x = list(range(len(trajectory)))

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=x, y=values[:, 0],
        mode='lines+markers',
        name='Indicator 1',
        line=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=x, y=values[:, 1],
        mode='lines+markers',
        name='Indicator 2',
        line=dict(color='red')
    ))

    fig.update_layout(
        title='Degradation Trajectory',
        xaxis_title='Time Step',
        yaxis_title='Indicator Value',
        legend=dict(x=0.01, y=0.99),
        template='plotly_white'
    )

    fig.show()
