from urllib.request import urlopen

# {{{ Copied from https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py

from os.path import basename as _basename, dirname as _dirname

# NOTE: these are only imported for type checking
from docutils.nodes import Text, TextElement
from sphinx.addnodes import pending_xref
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment


html_show_sourcelink = True

project = _basename(_dirname(_dirname(__file__)))

autoclass_content = "class"

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

sphinxconfig_missing_reference_aliases: dict[str, str] = {}


def linkcode_resolve(
        domain: str,
        info: dict[str, str],
        linkcode_url: str | None = None,
        ) -> str | None:
    import inspect
    import os
    import sys

    if domain != "py" or not info["module"]:
        return None

    submodname = info["module"]
    topmodname = submodname.split(".")[0]
    fullname = info["fullname"]

    topmod = sys.modules.get(topmodname)
    submod = sys.modules.get(submodname)
    if submod is None:
        return None

    obj = submod
    for part in fullname.split("."):
        try:
            obj = getattr(obj, part)
        except Exception:
            return None

    try:
        modpath = os.path.dirname(os.path.dirname(inspect.getsourcefile(topmod)))
        filepath = os.path.relpath(inspect.getsourcefile(obj), modpath)
    except Exception:
        return None

    if filepath is None:
        return None

    try:
        source, lineno = inspect.getsourcelines(obj)
    except OSError:
        return None
    else:
        linestart, linestop = lineno, lineno + len(source) - 1

    if linkcode_url is None:
        linkcode_url = (
            f"https://github.com/kaushikcfd/{project}/blob/"
            + "main"
            + "/src/{filepath}#L{linestart}-L{linestop}"  # noqa: RUF027
        )

    return linkcode_url.format(
        filepath=filepath, linestart=linestart, linestop=linestop
    )


def process_autodoc_missing_reference(
        app: Sphinx,
        env: BuildEnvironment,
        node: pending_xref,
        contnode: Text | TextElement
        ) -> TextElement | None:
    """Fix missing references due to string annotations.

    Users of this function should add it to the ``missing-reference`` event as

    .. code:: python

        def setup(app: Sphinx) -> None:
            app.connect("missing-reference", process_autodoc_missing_reference)

    This uses an alias dictionary called ``sphinxconfig_missing_reference_aliases``
    that maps each unknown type to an reference type and actual type full name.
    For example, say that the ``numpy.float64`` reference is not recognized
    properly. We know that this object is in the ``numpy`` module as an attribute.
    Then, we write:

    .. code:: python

        autodoc_missing_reference_aliases: dict[str, str] = {
            "numpy.float64": "py:attr:numpy.float64",
        }
    """
    from sphinx.util import logging
    logger = logging.getLogger("sphinxconfig")

    # check if this is a known alias
    target = node["reftarget"]
    if target not in sphinxconfig_missing_reference_aliases:
        return None

    # parse alias
    fullname = sphinxconfig_missing_reference_aliases[target]
    parts = fullname.split(":")
    if len(parts) == 1:
        domain = "py"
        reftype = "obj"
        reftarget, = parts
    elif len(parts) == 2:
        domain = "py"
        reftype, reftarget = parts
    elif len(parts) == 3:
        domain, reftype, reftarget = parts
    else:
        logger.error("Could not parse alias for '%s': '%s'.", target, fullname)
        return None

    if domain != "py":
        logger.error("Domain not supported for '%s': '%s'.", target, domain)
        return None

    # parse module and object from reftarget
    parts = reftarget.split(".")
    if len(parts) == 1:
        inventory = module = objname = parts[0]
    elif len(parts) >= 2:
        inventory = parts[0]
        module = ".".join(parts[:-1])
        objname = parts[-1]
    else:
        logger.error("Could not parse reftarget for '%s': '%s'.", target, reftarget)
        return None

    # rewrite attributes
    node.attributes["refdomain"] = domain
    node.attributes["reftype"] = reftype
    node.attributes[f"{domain}:module"] = module

    if isinstance(contnode, Text):
        # NOTE: if the format is "short", we insist on using just the object name.
        # This also makes sure that the text matches the reftarget, in case the
        # object was renamed somehow (e.g. from `import Object as _Object`).
        if app.config.autodoc_typehints_format == "short":
            contnode = Text(objname)
        else:
            contnode = Text(reftarget)

    # resolve reference
    from sphinx.ext import intersphinx

    if intersphinx.inventory_exists(env, inventory):
        node.attributes["reftarget"] = reftarget

        new_node = intersphinx.resolve_reference_in_inventory(
            env, inventory, node, contnode)
    else:
        domain = env.get_domain(domain)
        new_node = domain.resolve_xref(
            env, node["refdoc"], app.builder, reftype, objname,
            node, contnode)

    return new_node


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

__all__ = (
    "autoclass_content",
    "copybutton_prompt_is_regexp",
    "copybutton_prompt_text",
    "extensions",
    "html_show_sourcelink",
    "html_theme",
    "linkcode_resolve",
    "project",
)

# }}}


html_theme = "sphinx_book_theme"
html_theme_options = {
}
html_static_path = ["static"]
html_logo = "static/logo.svg"
copyright = "2025, feinsum Contributors"
author = "feinsum Contributors"
release = "2025.3"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pyopencl": ("https://documen.tician.de/pyopencl/", None),
    "pytools": ("https://documen.tician.de/pytools/", None),
    "pymbolic": ("https://documen.tician.de/pymbolic/", None),
    "loopy": ("https://documen.tician.de/loopy/", None),
    "islpy": ("https://documen.tician.de/islpy/", None),
    "opt_einsum": ("https://optimized-einsum.readthedocs.io/en/stable/", None),
}

nitpick_ignore_regex = [
    ["py:class", r"numpy.(u?)int[\d]+"],
    ["py:class", r"numpy.(_?)typing.(.+)"],
    # As of 2022-09-15, it doesn't look like there's sphinx documentation
    # available.
    ["py:class", r"immutables\.(.+)"],
    ["py:class", r"bidict\.(.+)"],
]

# vim: fdm=marker
