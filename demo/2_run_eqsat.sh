#!/bin/sh
cur_dir=$(pwd)
cp resmlp.relay $GLENSIDE_HOME/models
cd $GLENSIDE_HOME
/usr/bin/time -f "Time elapsed: %e" sh -c '"$0" "test" "test_resmlp" "$@" >/dev/null 2>&1' cargo
cp ./models/resmlp-dump.json $cur_dir/resmlp-rewritten.json
echo "[INFO] Model after EqSat: resmlp-rewritten.json"
