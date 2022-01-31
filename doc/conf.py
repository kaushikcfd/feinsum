from urllib.request import urlopen

_conf_url = (
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py")
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

copyright = "2022, feinsum Contributors"
author = "feinsum Contributors"

ver_dic = {}
exec(compile(open("../src/VERSION.py").read(), "../src/VERSION.py",
    "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
release = ver_dic["VERSION_TEXT"]

extensions += ["sphinx.ext.autodoc", "sphinx_autodoc_typehints"]

intersphinx_mapping = {
    "https://docs.python.org/3/": None,
    "https://numpy.org/doc/stable/": None,
    "https://documen.tician.de/pyopencl/": None,
    "https://documen.tician.de/pytools/": None,
    "https://documen.tician.de/pymbolic/": None,
    "https://documen.tician.de/loopy/": None,
    "https://documen.tician.de/islpy/": None,
    "https://optimized-einsum.readthedocs.io/en/stable/": None,
}

nitpick_ignore_regex = [
    ["py:class", r"numpy.(u?)int[\d]+"],
    ["py:class", r"numpy.typing.(.+)"],
    ["py:class", r"pyrsistent.typing.(.+)"],
]
