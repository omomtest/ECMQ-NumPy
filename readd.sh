#!/bin/bash

# Loop through each submodule path in .gitmodules
grep path .gitmodules | awk -F = '{print $2}' | sed 's/^[ \t]*//' | while read submodule_path; do
    echo "Processing submodule at $submodule_path..."

    # Remove the submodule directory if it exists
    if [ -d "$submodule_path" ]; then
        rm -rf "$submodule_path"
    fi

    # Remove the submodule entry from the index
    git rm --cached "$submodule_path"

    # Deinitialize the submodule
    git submodule deinit -f "$submodule_path"

    # Remove submodule metadata if it exists
    rm -rf ".git/modules/$submodule_path"

    # Remove submodule information from .gitmodules
    sed -i "/$submodule_path/,+2d" .gitmodules

    # Extract the URL from .gitmodules using the submodule path
    submodule_url=$(grep -A 1 "$submodule_path" .gitmodules | grep url | awk -F = '{print $2}' | sed 's/^[ \t]*//')

    # Re-add the submodule
    git submodule add "$submodule_url" "$submodule_path"

    # Initialize and update the submodule
    git submodule update --init --recursive "$submodule_path"

    echo "Re-added submodule at $submodule_path"
done

# Commit the changes
git add . && git commit -m "Re-added all submodules"
