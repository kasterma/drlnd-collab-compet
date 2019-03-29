import pandas as pd

from bokeh.io import output_file, show
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource, PrintfTickFormatter
from bokeh.plotting import figure
from bokeh.transform import transform

df = pd.DataFrame(
    [[10, 0, 1], [1, 10, 0], [1, 1, 9]],
    columns=['A', 'B', 'C'],
    index=['A', 'B', 'C'])
df.index.name = 'Treatment'
df.columns.name = 'Prediction'

df = df.stack().rename('value').reset_index()

# here the plot :
output_file("myPlot.html")

# You can use your own palette here
colors = ['#d7191c', '#fdae61', '#ffffbf', '#a6d96a', '#1a9641']

# Had a specific mapper to map color with value
mapper = LinearColorMapper(
    palette=colors, low=df.value.min(), high=df.value.max())
# Define a figure
p = figure(
    plot_width=800,
    plot_height=300,
    title="My plot",
    x_range=list(df.Treatment.drop_duplicates()),
    y_range=list(df.Prediction.drop_duplicates()),
    toolbar_location=None,
    tools="",
    x_axis_location="above")
# Create rectangle for heatmap
p.rect(
    x="Treatment",
    y="Prediction",
    width=1,
    height=1,
    source=ColumnDataSource(df),
    line_color=None,
    fill_color=transform('value', mapper))
# Add legend
color_bar = ColorBar(
    color_mapper=mapper,
    location=(0, 0),
    ticker=BasicTicker(desired_num_ticks=len(colors)))

p.add_layout(color_bar, 'right')

show(p)

