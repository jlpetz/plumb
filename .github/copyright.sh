#!/bin/bash

# Checks that every Rust source file starts with the project's copyright header, as
# fearless_simd's .github/copyright.sh does. Files that can't carry it can be ignored with an
# additional glob argument, e.g. -g "!src/special_file.rs".

command -v rg >/dev/null 2>&1 || { echo "copyright.sh needs ripgrep (rg)" >&2; exit 2; }

output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the plumb Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT$\n\n" --files-without-match --multiline -g "*.rs" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nPlease add the following header:\n"
	echo "// Copyright $(date +%Y) the plumb Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

echo "All files have correct copyright headers."
exit 0
