from __future__ import annotations

import os
import shlex
import tempfile
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

from .schema import GitAuthConfig


class GitAuthError(RuntimeError):
    pass


@dataclass(frozen=True)
class ResolvedGitAuth:
    username: str
    password: str
    source_env: str


class GitAuthManager:
    def __init__(self, config: GitAuthConfig, *, helper_root: Optional[Path] = None) -> None:
        self.config = config
        self.helper_root = helper_root

    def resolve(self, *, required: bool = False) -> Optional[ResolvedGitAuth]:
        env_names = [str(self.config.token_env).strip()]
        fallback = self.config.fallback_token_env
        if fallback:
            env_names.append(str(fallback).strip())
        for env_name in env_names:
            if not env_name:
                continue
            token = str(os.environ.get(env_name, "")).strip()
            if token:
                return ResolvedGitAuth(
                    username=str(self.config.username).strip(),
                    password=token,
                    source_env=env_name,
                )
        if required:
            names = ", ".join(name for name in env_names if name)
            raise GitAuthError(f"Missing Git token in environment. Expected one of: {names}")
        return None

    @contextmanager
    def local_git_env(self, *, enabled: bool) -> Iterator[dict[str, str]]:
        if not enabled:
            yield {}
            return
        auth = self.resolve(required=True)
        helper_parent = self.helper_root or (Path.cwd() / "tmp_boa_git_auth_helpers")
        helper_parent.mkdir(parents=True, exist_ok=True)
        suffix = ".cmd" if os.name == "nt" else ".sh"
        helper = helper_parent / f"askpass_{uuid.uuid4().hex}{suffix}"
        try:
            if os.name == "nt":
                helper.write_text(
                    "\n".join(
                        [
                            "@echo off",
                            "set prompt=%~1",
                            "echo %prompt% | findstr /I \"username\" >nul",
                            "if not errorlevel 1 (",
                            "  echo %BOA_GIT_USERNAME%",
                            "  exit /b 0",
                            ")",
                            "echo %BOA_GIT_PASSWORD%",
                        ]
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                helper.write_text(
                    "#!/usr/bin/env sh\n"
                    "case \"$1\" in\n"
                    "  *sername*) printf '%s\\n' \"$BOA_GIT_USERNAME\" ;;\n"
                    "  *) printf '%s\\n' \"$BOA_GIT_PASSWORD\" ;;\n"
                    "esac\n",
                    encoding="utf-8",
                )
                helper.chmod(0o700)
            yield {
                "BOA_GIT_USERNAME": auth.username,
                "BOA_GIT_PASSWORD": auth.password,
                "GIT_ASKPASS": str(helper),
                "GIT_TERMINAL_PROMPT": "0",
                "GCM_INTERACTIVE": "Never",
            }
        finally:
            try:
                helper.unlink()
            except OSError:
                pass

    def build_remote_fetch_setup(self, *, enabled: bool) -> list[str]:
        if not enabled:
            return []
        auth = self.resolve(required=True)
        return [
            'git_auth_dir="$(mktemp -d)"',
            "cleanup_git_auth() {",
            '  rm -rf "$git_auth_dir"',
            "  unset BOA_GIT_USERNAME BOA_GIT_PASSWORD GIT_ASKPASS GIT_TERMINAL_PROMPT GCM_INTERACTIVE",
            "}",
            "trap cleanup_git_auth EXIT",
            'cat > "$git_auth_dir/askpass.sh" <<\'BOA_GIT_ASKPASS\'',
            "#!/usr/bin/env sh",
            'case "$1" in',
            '  *sername*) printf \"%s\\n\" "$BOA_GIT_USERNAME" ;;',
            '  *) printf \"%s\\n\" "$BOA_GIT_PASSWORD" ;;',
            "esac",
            "BOA_GIT_ASKPASS",
            'chmod 700 "$git_auth_dir/askpass.sh"',
            f"export BOA_GIT_USERNAME={shlex.quote(auth.username)}",
            f"export BOA_GIT_PASSWORD={shlex.quote(auth.password)}",
            'export GIT_ASKPASS="$git_auth_dir/askpass.sh"',
            'export GIT_TERMINAL_PROMPT=0',
            'export GCM_INTERACTIVE=Never',
        ]
