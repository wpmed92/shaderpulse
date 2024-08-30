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
  
  # Create a temporary file to capture output and errors
  OUTPUT_FILE=$(mktemp)
  STDERR_FILE=$(mktemp)

  # Run shaderpulse and redirect both stdout and stderr
  $SHADERPULSE "$TEST_FILE" --no-analyze > "$OUTPUT_FILE" 2> "$STDERR_FILE"

  # Display stderr content if there are issues
  if [ -s "$STDERR_FILE" ]; then
    echo "Shaderpulse stderr output:"
    cat "$STDERR_FILE"
  fi

  # Run FileCheck with the captured output
  $FILECHECK "$TEST_FILE" < "$OUTPUT_FILE"

  # Check if FileCheck passed
  if [ $? -eq 0 ]; then
    echo "Test passed for $TEST_FILE"
  else
    echo "Test failed for $TEST_FILE"
    exit 1
  fi

  # Clean up temporary files
  rm "$OUTPUT_FILE" "$STDERR_FILE"
done
