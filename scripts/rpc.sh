#! /usr/bin/env bash
seq -w 0 9 | parallel -j 10 julia --project=@. --color=yes ./scripts/rpc.jl -m "ae" -d 8 -c {} -r 160
