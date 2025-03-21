#!/bin/bash

# Loop through each submodule path
for submodule_path in $(grep path .gitmodules | awk -F = '{print $2}' | sed 's/^[ \t]*//'); do
  echo "Cleaning up submodule at $submodule_path..."

  # Remove the submodule directory if it exists
  rm -rf "$submodule_path"

  # Remove the submodule entry from the index
  git rm --cached "$submodule_path"

  # Deinitialize any existing submodule information
  git submodule deinit -f "$submodule_path"

  # Remove submodule metadata if present
  rm -rf ".git/modules/$submodule_path"

  # Remove submodule info from .gitmodules
  sed -i "/$submodule_path/,+2d" .gitmodules
  
  # Re-add the submodule
  git submodule add $(grep -A 1 "$submodule_path" .gitmodules | grep url | awk -F = '{print $2}' | sed 's/^[ \t]*//') "$submodule_path"

  # Initialize and update the submodule
  git submodule update --init --recursive

  echo "Re-added submodule at $submodule_path"
done

# Commit the changes
git add . && git commit -m "Re-added all submodules"
