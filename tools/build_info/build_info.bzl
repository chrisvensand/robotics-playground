"""Starlark rule for generating a build info header from workspace status."""

def _build_info_header_impl(ctx):
    out = ctx.actions.declare_file("build_info.h")
    ctx.actions.run_shell(
        inputs = [ctx.info_file],
        outputs = [out],
        command = """\
SHA=$(grep '^STABLE_GIT_SHA ' {stable} 2>/dev/null | awk '{{print $2}}')
VERSION=$(grep '^STABLE_VERSION ' {stable} 2>/dev/null | awk '{{print $2}}')
if [ -z "$SHA" ]; then SHA="unknown"; fi
if [ -z "$VERSION" ]; then VERSION="dev"; fi
printf '#pragma once\\n#define BUILD_GIT_SHA "%s"\\n#define BUILD_VERSION "%s"\\n' "$SHA" "$VERSION" > {out}
""".format(stable = ctx.info_file.path, out = out.path),
    )
    return [DefaultInfo(files = depset([out]))]

build_info_header = rule(
    implementation = _build_info_header_impl,
    attrs = {},
)
