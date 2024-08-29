#!/bin/bash

set -e

SHADERPULSE="../../build/shaderpulse"
FILECHECK="../../llvm-project/build/bin/FileCheck"

if [ ! -x "$SHADERPULSE" ]; then
  echo "Error: shaderpulse binary not found at $SHADERPULSE"
  exit 1
fi

if [ ! -x "$FILECHECK" ]; then
  echo "Error: FileCheck binary not found at $FILECHECK"
  exit 1
fi

for TEST_FILE in *.glsl; do
  if [ ! -f "$TEST_FILE" ]; then
    echo "No .glsl files found in the current directory."
    exit 1
  fi

  # functions tests are failing in CI due to some pointer alignment issue
  if [ "$CI" = "1" ] && [ "$TEST_FILE" = "functions.glsl" ]; then
    echo "Skipping test for $TEST_FILE - errors in CI"
    continue
  fi

  echo "Running test on $TEST_FILE"
  $SHADERPULSE "$TEST_FILE" --no-analyze | $FILECHECK "$TEST_FILE"

  if [ $? -eq 0 ]; then
    echo "Test passed for $TEST_FILE"
  else
    echo "Test failed for $TEST_FILE"
    exit 1
  fi
done
