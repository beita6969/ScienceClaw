#!/bin/bash
# usage: latest_pdf.sh outname
D="/Users/admin/.claude/projects/-Users-admin-Library-Application-Support-Claude-scratch-workspaces-b286c85c-fb50-4909-a7a4-692fbcc9e2e6-67544b4d-5c42-4571-a504-64db11cd447d-scratch-2026-09-28-749df2/14e47173-fba0-4284-9ce9-74847ea151c4/tool-results"
f=$(ls -t "$D"/*.pdf | head -1)
/private/tmp/claude-501/sc-scratch/sota/venv_g2/bin/python /private/tmp/claude-501/sc-scratch/sota/g2_work/pdf2txt.py "$f" /private/tmp/claude-501/sc-scratch/sota/g2_work/$1.txt
