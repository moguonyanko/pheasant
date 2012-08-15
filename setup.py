#!/usr/bin/python3
# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
	name = "pheasant",
	packages = ["pheasant","pheasant.const","pheasant.test"],
	version = "0.1.1",
	description = "Python library for work mathmatics",
	author = "moguonyanko",
	author_email = "moguo_50000_nyanko@yahoo.co.jp",
	url = "",
	download_url = "",
	keywords = ["mathmatics", "statistics"],
	classifiers = [
		"Programming Language :: Python",
		"Programming Language :: Python :: 3.1",
		"Development Status :: 1 - Planning",
		"Environment :: Other Environment",
		"Intended Audience :: Developers",
		"License :: OSI Approved :: BSD License",
		"Operating System :: OS Independent",
		"Topic :: Software Development :: Libraries",
		"Topic :: Scientific/Engineering :: Mathematics"
	],
	long_description = """\
Practical math library
-------------------------------------

This version requires Python 3 or later; a Python 2 version is not available.
"""
)



