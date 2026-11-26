#!/usr/bin/env bash
set -eo pipefail
unset ASCEND_TOOLKIT_LATEST_HOME TOOLCHAIN_HOME CMAKE_PREFIX_PATH SOC_VERSION
source /usr/local/Ascend/cann-9.2.1/set_env.sh
exec "$@"
