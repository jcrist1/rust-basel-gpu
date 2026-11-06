import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import polars as pl

    return (pl,)


@app.cell
def _(pl):
    pl.read_parquet("sample_data.parquet")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
