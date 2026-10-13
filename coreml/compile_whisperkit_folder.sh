#!/bin/sh
# Compile WhisperKit .mlpackage files into .mlmodelc next to them (macOS + Xcode only).
# Use this for the folder you bundle in the app as a folder reference: Xcode only
# auto-compiles .mlpackage files added directly to a target, not ones inside a
# folder reference, and WhisperKit loads compiled .mlmodelc fastest.
#
#   sh coreml/compile_whisperkit_folder.sh coreml/build/openai_whisper-small_6bit
set -e
DIR="${1:?usage: $0 <whisperkit model folder>}"
for pkg in "$DIR"/*.mlpackage; do
    echo "Compiling $pkg"
    xcrun coremlcompiler compile "$pkg" "$DIR"
done
echo "Done. You can delete the .mlpackage files from $DIR before bundling."
