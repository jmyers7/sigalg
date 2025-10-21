from IPython.display import HTML


def print_html_df(
    df, drop_index=True, width="80px", font="monospace", align="right", title=None
):
    """
    Pretty-print a pandas DataFrame as styled HTML.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to render.
    drop_index : bool, default True
        Whether to hide the DataFrame index in the output.
    width : str, default '80px'
        Width for each column (any valid CSS unit works).
    font : str, default 'monospace'
        Font family to use for table text.
    align : str, default 'right'
        Horizontal alignment for text in cells and headers.
    title : str, optional
        Optional title to display above the table.

    Returns
    -------
    IPython.display.HTML
        A rendered HTML object suitable for display().
    """
    styled = df.style.set_properties(
        **{"text-align": align, "font-family": font, "width": width}
    ).set_table_styles(
        [{"selector": "th", "props": [("text-align", align), ("width", width)]}]
    )

    if drop_index:
        styled = styled.hide(axis="index")

    if title is not None:
        styled = styled.set_caption(title)

    return HTML(styled.to_html())
