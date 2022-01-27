#!/usr/bin/env python3
"""This script allows using Pylint with YAML-based config files.

The usage of this script is identical to Pylint, except that this script accepts
an additional argument "--yaml-rcfile=", which specifies a path to a YAML file
from which command line options are derived. The "--yaml-rcfile=" argument may
be given multiple times.

The YAML config file format is a list of "arg"/"val" entries. Multiple or
omitted values are allowed. Repeated arguments are allowed. An example is as
follows:

    ---
    - arg: errors-only
    - arg: ignore
      val:
        - dir1
        - dir2
    - arg: ignore
        val: dir3

This example is equivalent to invoking pylint with the options

    pylint --errors-only --ignore=dir1,dir2 --ignore=dir3

"""

import sys
import logging
import shlex

import pylint.lint
import yaml

logger = logging.getLogger(__name__)


def generate_args_from_yaml(input_yaml):
    """Generate a list of strings suitable for use as Pylint args, from YAML.

    Arguments:
        input_yaml: YAML data, as an input file or bytes

    """

    parsed_data = yaml.safe_load(input_yaml)

    for entry in parsed_data:
        arg = entry["arg"]
        val = entry.get("val")

        if val is not None:
            if isinstance(val, list):
                val = ",".join(str(item) for item in val)

            yield "--%s=%s" % (arg, val)
        else:
            yield "--%s" % arg


YAML_RCFILE_PREFIX = "--yaml-rcfile="


def main():
    """Process command line args and run Pylint."""
    args = []

    for arg in sys.argv[1:]:
        if arg.startswith(YAML_RCFILE_PREFIX):
            config_path = arg[len(YAML_RCFILE_PREFIX):]
            with open(config_path, "r") as config_file:
                args.extend(generate_args_from_yaml(config_file))
        else:
            args.append(arg)

    logger.info(" ".join(shlex.quote(arg) for arg in ["pylint"] + args))
    pylint.lint.Run(args)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
