#!/bin/bash

#
# Copyright 2020 Benjamin Worpitz
# SPDX-License-Identifier: MPL-2.0
#

set +xv
source ./script/setup_utilities.sh

travis_retry sudo apt-get -y --quiet install --no-install-recommends doxygen graphviz
