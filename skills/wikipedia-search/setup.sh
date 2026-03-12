#!/bin/bash
# Setup script for wikipedia-search skill

SKILL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKILLS_DST="${HOME}/.claude/skills/wikipedia-search"

echo "Setting up wikipedia-search skill..."

# Make the script executable
chmod +x "$SKILL_DIR/scripts/wiki.py"
echo "✓ Made wiki.py executable"

# Create symbolic link
mkdir -p "$(dirname "$SKILLS_DST")"
ln -sf "$SKILL_DIR" "$SKILLS_DST"
echo "✓ Created symbolic link"

# Verify setup
echo ""
echo "Verification:"
ls -la "$SKILL_DIR/scripts/wiki.py"
echo ""
ls -la "$(dirname "$SKILLS_DST")" | grep wikipedia

echo ""
echo "Setup complete! Test with:"
echo "python3 $SKILLS_DST/scripts/wiki.py --help"
