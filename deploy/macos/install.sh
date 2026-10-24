#!/bin/sh
# Run the live lab as a low-priority background job on a Mac (a per-user LaunchAgent).
# It starts at login, restarts if it stops, and yields the CPU to anything you're doing.
#
#   deploy/macos/install.sh [UPLOAD_DEST] [WORKERS] [SLOW_WORKERS]
#
# UPLOAD_DEST is an rsync destination such as user@host:/opt/battleship-lab/public/
# (key-based SSH required). Leave it out to keep the snapshot local. SLOW_WORKERS is how
# many of the WORKERS go to the lane = "slow" conditions (default: a quarter).
#
# Stop:      launchctl bootout gui/$(id -u)/com.kalebhoffer.battleship-lab
# Start:     launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.kalebhoffer.battleship-lab.plist
# Logs:      tail -f lab-state/lab.log
# Uninstall: stop it, then delete the plist above.
set -eu

LABEL=com.kalebhoffer.battleship-lab
REPO=$(cd "$(dirname "$0")/../.." && pwd)
UPLOAD=${1:-}
WORKERS=${2:-8}
SLOW=${3:-}
UV=$(command -v uv)
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"

SLOW_ARGS=""
if [ -n "$SLOW" ]; then
  SLOW_ARGS="<string>--slow-workers</string><string>$SLOW</string>"
fi
UPLOAD_ARGS=""
if [ -n "$UPLOAD" ]; then
  UPLOAD_ARGS="<string>--upload</string><string>$UPLOAD</string>"
fi

mkdir -p "$REPO/lab-state" "$HOME/Library/LaunchAgents"
cat > "$PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$UV</string><string>run</string><string>--frozen</string>
    <string>battleship-sim</string><string>lab</string>
    <string>--workers</string><string>$WORKERS</string>
    $SLOW_ARGS
    $UPLOAD_ARGS
  </array>
  <key>WorkingDirectory</key><string>$REPO</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>60</integer>
  <key>Nice</key><integer>15</integer>
  <key>LowPriorityIO</key><true/>
  <key>StandardOutPath</key><string>$REPO/lab-state/lab.log</string>
  <key>StandardErrorPath</key><string>$REPO/lab-state/lab.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
# bootout returns before the old job is gone; bootstrap fails if it's still loaded.
while launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; do sleep 1; done
launchctl bootstrap "gui/$(id -u)" "$PLIST"
echo "Installed $PLIST ($WORKERS workers${SLOW:+, $SLOW slow}). Logs: $REPO/lab-state/lab.log"
