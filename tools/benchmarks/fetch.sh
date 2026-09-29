#!/usr/bin/env bash
# Rebuilds the min-cost flow benchmark suite from
#   Péter Kovács, "Minimum-Cost Flow Algorithms: an Experimental Evaluation",
#   Optimization Methods and Software 30(1):94-127, 2015.
#
# The generated families are recreated from Kovács's published generator
# parameters with the original DIMACS generators (NETGEN, GOTO, GRIDGEN),
# which are downloaded and built locally. The GRIDGRAPH families (GRID-*)
# are skipped: that generator is only available as Fortran.
#
# Usage: tools/benchmarks/fetch.sh [--real]
#   --real   also download the ROAD and VISION instances (about 1 GB)
#
# Environment:
#   DATA       output directory (default: bench-data in the repository root)
#   MAX_ARCS   skip generated instances with more arcs than this (default 4000000)
#   FAMILIES   space-separated generated families to build (default: all;
#              set it empty to only fetch the --real instances)
#   SEEDS      instance letters per size to build, e.g. "a b" (default: "a")

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA="${DATA:-$ROOT/bench-data}"
MAX_ARCS="${MAX_ARCS:-4000000}"
FAMILIES="${FAMILIES-netgen_8 netgen_sr netgen_lo_8 netgen_lo_sr netgen_deg gridgen_8 gridgen_sr gridgen_deg goto_8 goto_sr}"
SEEDS="${SEEDS:-a}"
REAL=0
[[ "${1:-}" == "--real" ]] && REAL=1

DIMACS=http://archive.dimacs.rutgers.edu/pub/netflow/generators/network
KOVACS_FOLDER=1zvKpxD3hzbnfDDKx5oEjzS0i-Rfb9g6A

# Google Drive file ids of the parameter archives in Kovács's folder.
declare -A PARAM_IDS=(
  [goto_8]=1P0aAKeTc7REuWAGb7VAX15tIGqQm8gTX
  [goto_sr]=1ugig6-MQ_4fFuzF2DosrJK7QU4XKc0Eg
  [gridgen_8]=1pt1J5tGq-SHr_6x9z0ihzuoTjtBHUzOW
  [gridgen_deg]=1Zrnyu9-m6RT7RZKtBbCbUxUVxzSB_bRc
  [gridgen_sr]=1bytEvv_5rMZpQrCAyuJQFBvF9Oua8XxZ
  [netgen_8]=1osa5fiB2_Qx6UiPAYI5tTFJLGWDYWSU4
  [netgen_deg]=1QMlxiy1b6Z-WLE6MbKNe6ekcIA1wXQM2
  [netgen_lo_8]=16q2WvP3wZG7e6WbxKSGMglfdghyc8GII
  [netgen_lo_sr]=1kvyC-O8YiJlWIbmzGXlX7dK4i7OEp0Pa
  [netgen_sr]=1GGz6FwPTXcfvHq8kUP8HbZDnbTYOka6_
)

drive_download() {
  curl -fsSL "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t" -o "$2"
}

# Lists a public Drive folder as "name<TAB>file id" lines.
drive_list() {
  curl -fsSL "https://drive.google.com/embeddedfolderview?id=$1" | python3 -c '
import html, re, sys
page = sys.stdin.read()
entry = r"<a href=\"https://drive.google.com/(?:file/d|drive/folders)/([^/\"?]+)[^\"]*\"[^>]*>.*?<div class=\"flip-entry-title\">([^<]+)</div>"
for m in re.finditer(entry, page, re.S):
    print(html.unescape(m.group(2)) + "\t" + m.group(1))
'
}

build_generators() {
  local gen="$DATA/generators"
  mkdir -p "$gen/netgen"
  if [[ ! -x "$gen/netgen/netgen" ]]; then
    for f in netgen.c netgen.h index.c random.c; do
      curl -fsSL "$DIMACS/netgen/$f" -o "$gen/netgen/$f"
    done
    # NETGEN's own random() clashes with the C library's, and its built-in
    # size limits are far below the benchmark sizes.
    sed -i -E 's/\brandom\b/ng_random/g' "$gen"/netgen/*.c "$gen"/netgen/*.h
    sed -i -E 's/^#define MAXNODES .*/#define MAXNODES 4200000/; s/^#define MAXARCS .*/#define MAXARCS 34000000/' \
      "$gen/netgen/netgen.h"
    cc -O2 -DDIMACS -w -std=gnu89 -o "$gen/netgen/netgen" \
      "$gen/netgen/netgen.c" "$gen/netgen/index.c" "$gen/netgen/random.c"
  fi
  if [[ ! -x "$gen/goto" ]]; then
    curl -fsSL "$DIMACS/grid-on-torus/goto.c" -o "$gen/goto.c"
    cc -O2 -w -std=gnu89 -o "$gen/goto" "$gen/goto.c" -lm
  fi
  if [[ ! -x "$gen/gridgen" ]]; then
    curl -fsSL "$DIMACS/gridgen/gridgen.c" -o "$gen/gridgen.c"
    cc -O2 -w -std=gnu89 -o "$gen/gridgen" "$gen/gridgen.c" -lm
  fi
}

# Prints the arc count a parameter line will produce (exact for NETGEN and
# GOTO, approximate for GRIDGEN).
arc_count() {
  local family=$1; shift
  local p=($*)
  case $family in
    netgen*) echo "${p[5]}" ;;
    goto*) echo "${p[1]}" ;;
    gridgen*) echo $(( p[2] * p[6] )) ;;
  esac
}

generator_for() {
  case $1 in
    netgen*) echo "$DATA/generators/netgen/netgen" ;;
    goto*) echo "$DATA/generators/goto" ;;
    gridgen*) echo "$DATA/generators/gridgen" ;;
  esac
}

generate_family() {
  local family=$1
  local params="$DATA/params/$family.zip"
  local out="$DATA/instances/$family"
  mkdir -p "$DATA/params" "$out"
  [[ -f "$params" ]] || drive_download "${PARAM_IDS[$family]}" "$params"

  local gen
  gen="$(generator_for "$family")"
  local made=0 skipped=0
  while read -r entry; do
    local name="${entry%.param}"
    local letter="${name%.min}"
    letter="${letter: -1}"
    [[ " $SEEDS " == *" $letter "* ]] || continue
    [[ -s "$out/$name" ]] && { made=$((made + 1)); continue; }
    local line
    line="$(unzip -p "$params" "$entry")"
    if (( $(arc_count "$family" $line) > MAX_ARCS )); then
      skipped=$((skipped + 1))
      continue
    fi
    # GRIDGEN exits with a garbage status, so judge by the output: the arc
    # lines must match the count declared in the problem line.
    echo "$line" | "$gen" > "$out/$name.tmp" || true
    if ! awk '/^p/ { m = $4 } /^a/ { a++ } END { exit !(m > 0 && a == m) }' "$out/$name.tmp"; then
      echo "generator failed on $family/$name" >&2
      rm -f "$out/$name.tmp"
      continue
    fi
    mv "$out/$name.tmp" "$out/$name"
    made=$((made + 1))
  done < <(unzip -Z1 "$params" | sort)
  echo "$family: $made instances ($skipped over MAX_ARCS=$MAX_ARCS skipped)"
}

fetch_real() {
  local folders
  folders="$(drive_list "$KOVACS_FOLDER")"
  for family in road vision; do
    local id
    id="$(awk -F'\t' -v f="$family" '$1 == f { print $2 }' <<< "$folders")"
    mkdir -p "$DATA/instances/$family"
    while IFS=$'\t' read -r name file_id; do
      local target="$DATA/instances/$family/${name%.gz}"
      [[ -s "$target" ]] && continue
      drive_download "$file_id" "$target.gz"
      gunzip -f "$target.gz"
    done < <(drive_list "$id")
    echo "$family: $(ls "$DATA/instances/$family" | wc -l) instances"
  done
}

mkdir -p "$DATA"
build_generators
for family in $FAMILIES; do
  generate_family "$family"
done
if (( REAL )); then
  fetch_real
fi
