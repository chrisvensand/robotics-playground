#!/bin/bash
echo "STABLE_GIT_SHA $(git rev-parse HEAD)"
git diff --quiet && echo "STABLE_GIT_STATUS clean" || echo "STABLE_GIT_STATUS dirty"
VERSION=$(git describe --tags --abbrev=7 2>/dev/null | sed 's/^v//')
echo "STABLE_VERSION ${VERSION:-$(git rev-parse --short=7 HEAD)}"
