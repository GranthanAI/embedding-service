import runpy
from unittest import mock


def _run_main(monkeypatch, env: str | None):
    if env is None:
        monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    else:
        monkeypatch.setenv("FORWARDED_ALLOW_IPS", env)
    with mock.patch("uvicorn.run") as run:
        runpy.run_module("app.main", run_name="__main__")
    return run.call_args.kwargs


def test_proxy_headers_default_to_loopback_only(monkeypatch):
    kwargs = _run_main(monkeypatch, None)
    assert kwargs["proxy_headers"] is True
    assert kwargs["forwarded_allow_ips"] == "127.0.0.1"


def test_forwarded_allow_ips_comes_from_env(monkeypatch):
    assert _run_main(monkeypatch, "10.0.0.0/8")["forwarded_allow_ips"] == "10.0.0.0/8"
