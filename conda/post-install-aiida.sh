#!/bin/bash
# Post-install script for frictionsim2d-aiida conda package
# Copies startup scripts and hooks to $PREFIX locations

set -e

# Ensure PREFIX is set
if [ -z "$PREFIX" ]; then
    echo "ERROR: \$PREFIX not set" >&2
    exit 1
fi

# Determine source directory (where setup.py or source is located)
SRCDIR="${SRC_DIR:-.}"

# Create libexec directory
mkdir -p "$PREFIX/libexec/frictionsim2d"

# Copy startup script
if [ -f "$SRCDIR/src/aiida/start_aiida.sh" ]; then
    cp "$SRCDIR/src/aiida/start_aiida.sh" "$PREFIX/libexec/frictionsim2d/"
    chmod +x "$PREFIX/libexec/frictionsim2d/start_aiida.sh"
else
    echo "WARNING: start_aiida.sh not found at $SRCDIR/src/aiida/start_aiida.sh" >&2
fi

# Create symlink in bin for easy access
ln -sf "$PREFIX/libexec/frictionsim2d/start_aiida.sh" "$PREFIX/bin/frictionsim2d-start-aiida"

# Copy hook scripts and installer
if [ -d "$SRCDIR/src/aiida/hooks" ]; then
    mkdir -p "$PREFIX/libexec/frictionsim2d/hooks"
    cp "$SRCDIR/src/aiida/hooks/activate_services.sh" "$PREFIX/libexec/frictionsim2d/hooks/" || true
    cp "$SRCDIR/src/aiida/hooks/deactivate_services.sh" "$PREFIX/libexec/frictionsim2d/hooks/" || true
    chmod +x "$PREFIX/libexec/frictionsim2d/hooks"/*.sh || true
else
    echo "WARNING: hooks directory not found at $SRCDIR/src/aiida/hooks" >&2
fi

if [ -f "$SRCDIR/src/aiida/install_conda_hooks.sh" ]; then
    mkdir -p "$PREFIX/libexec/frictionsim2d/hooks"
    cp "$SRCDIR/src/aiida/install_conda_hooks.sh" "$PREFIX/libexec/frictionsim2d/hooks/"
    chmod +x "$PREFIX/libexec/frictionsim2d/hooks/install_conda_hooks.sh"
else
    echo "WARNING: install_conda_hooks.sh not found at $SRCDIR/src/aiida/install_conda_hooks.sh" >&2
fi

# Create symlink for hook installer
ln -sf "$PREFIX/libexec/frictionsim2d/hooks/install_conda_hooks.sh" "$PREFIX/bin/frictionsim2d-install-hooks"

echo ""
echo "FrictionSim2D AiiDA installation complete!"
echo ""
echo "Next steps:"
echo "  1. Bootstrap AiiDA profile and start services:"
echo "     frictionsim2d-start-aiida"
echo ""
echo "  2. (Optional) Install conda hooks for automatic service lifecycle:"
echo "     frictionsim2d-install-hooks"
echo ""
