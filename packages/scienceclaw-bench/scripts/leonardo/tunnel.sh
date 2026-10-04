#!/bin/bash
# Forward every running self-hosted vLLM server (from $L/sc-serve/endpoints/<jobid>) to local ports 18001, 18002, ...
# (EPDIR=<subdir> / PORT_BASE=<n> select another endpoint group) and print the matching `llm.endpoints` YAML list.
# Incremental: a forward whose server is still listed and answers /health keeps its port (so running clients are not
# disturbed); forwards of vanished servers are cancelled, new servers get the lowest free port, broken forwards are redone.
# The forwards live in a dedicated persistent ControlMaster (~/.ssh/cm-sc-leo, opened here if missing; do not share
# ~/.ssh/cm-scienceclaw-leo, which other sessions may take over) and are managed with ssh -O forward/cancel.
L=/leonardo_scratch/large/userexternal/rqian000
SOCK="$HOME/.ssh/cm-sc-leo"; STATE="${TMPDIR:-/tmp}/sc-tunnel-${EPDIR:-endpoints}.state"; BASE=${PORT_BASE:-18000}
ssh -S "$SOCK" -O check leonardo >/dev/null 2>&1 || ssh -o BatchMode=yes -o ControlMaster=yes -o ControlPath="$SOCK" -o ControlPersist=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -fN leonardo
if [ -S "$SOCK" ]; then MUX=1; SSH=(ssh -S "$SOCK" -o BatchMode=yes); else MUX=0; SSH=(ssh -o BatchMode=yes); fi
remote=$("${SSH[@]}" leonardo "for f in $L/sc-serve/${EPDIR:-endpoints}/[0-9]*; do case \$f in *.starting) ;; *) [ -f \$f ] && awk '{print \$1}' \$f;; esac; done" 2>/dev/null | sort -u)
[ -f "$STATE" ] || : > "$STATE"
new_state=$(mktemp)
cancel() { [ $MUX = 1 ] && "${SSH[@]}" -O cancel -L "$1" leonardo 2>/dev/null; }
# existing forwards: keep when still listed and healthy, otherwise cancel
while read -r spec; do
  [ -z "$spec" ] && continue
  port=$(echo "$spec" | cut -d: -f2); target=$(echo "$spec" | cut -d: -f3-)
  if echo "$remote" | grep -qxF "$target" && [ "$(curl -s -m 10 -o /dev/null -w '%{http_code}' http://127.0.0.1:$port/health)" = 200 ]; then
    echo "$spec" >> "$new_state"
  else
    cancel "$spec"
  fi
done < "$STATE"
# new servers: lowest free port above BASE
for target in $remote; do
  grep -q ":$target\$" "$new_state" && continue
  port=$((BASE+1)); while grep -q "^127.0.0.1:$port:" "$new_state"; do port=$((port+1)); done
  spec="127.0.0.1:$port:$target"
  if [ $MUX = 1 ]; then "${SSH[@]}" -O forward -L "$spec" leonardo; else "${SSH[@]}" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -fN -L "$spec" leonardo; fi
  [ $? = 0 ] && echo "$spec" >> "$new_state"
done
sort -t: -k2 -n "$new_state" > "$STATE"; rm -f "$new_state"
out=(); while IFS=: read -r _ port _; do [ -n "$port" ] && out+=("http://127.0.0.1:$port/v1"); done < "$STATE"
printf 'endpoints: ['; (IFS=,; printf '%s' "${out[*]}"); printf ']\n'
