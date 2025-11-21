from IPython.core.magic import register_cell_magic


def load_ipython_extension(ipython):
    """This is called when the module is loaded in Jupyter with %load_ext."""

    @register_cell_magic
    def ignore(line, cell):
        """Ignore the contents of this cell."""
        pass
