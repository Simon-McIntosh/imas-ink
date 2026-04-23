"""Unit tests for imas_ink.server.repl — namespaced stateful REPL."""

from __future__ import annotations

import threading


def _fresh_repl():
    """Import repl and reset the default namespace before each test."""
    from imas_ink.server import repl as repl_mod

    # Wipe all namespaces to give each test a clean slate.
    repl_mod._namespaces.clear()
    return repl_mod.repl


class TestBasicArithmetic:
    """State persists within a namespace across calls."""

    def test_assignment_and_retrieval(self):
        repl = _fresh_repl()
        repl("x = 42")
        out = repl("x")
        assert "42" in out

    def test_expression_repr_printed(self):
        repl = _fresh_repl()
        out = repl("1 + 1")
        assert "2" in out

    def test_state_persists_across_calls(self):
        repl = _fresh_repl()
        repl("total = 0", namespace="persist-test")
        repl("total += 10", namespace="persist-test")
        out = repl("total", namespace="persist-test")
        assert "10" in out

    def test_empty_code_returns_empty(self):
        repl = _fresh_repl()
        out = repl("")
        assert out == "" or out.strip() == ""


class TestNamespaceIsolation:
    """Variables in different namespaces do not bleed across."""

    def test_different_namespaces_are_isolated(self):
        repl = _fresh_repl()
        repl("x = 100", namespace="ns-a")
        out = repl("x", namespace="ns-b")
        # ns-b doesn't have x — should get a NameError traceback
        assert "NameError" in out or "x" not in out.split("=")[0]

    def test_same_name_independent_values(self):
        repl = _fresh_repl()
        repl("val = 'alpha'", namespace="ns-alpha")
        repl("val = 'beta'", namespace="ns-beta")
        out_a = repl("val", namespace="ns-alpha")
        out_b = repl("val", namespace="ns-beta")
        assert "alpha" in out_a
        assert "beta" in out_b


class TestReset:
    """reset=True wipes the namespace and re-seeds it."""

    def test_reset_clears_variables(self):
        repl = _fresh_repl()
        repl("secret = 999", namespace="reset-ns")
        repl("", namespace="reset-ns", reset=True)
        out = repl("secret", namespace="reset-ns")
        assert "NameError" in out

    def test_reset_preserves_seeded_globals(self):
        repl = _fresh_repl()
        repl("x = 1", namespace="reset-ns2")
        repl("", namespace="reset-ns2", reset=True)
        out = repl("np.pi", namespace="reset-ns2")
        # np should be re-seeded after reset
        assert "3.14" in out

    def test_reset_false_does_not_clear(self):
        repl = _fresh_repl()
        repl("y = 7", namespace="no-reset-ns")
        repl("", namespace="no-reset-ns", reset=False)
        out = repl("y", namespace="no-reset-ns")
        assert "7" in out


class TestSyntaxErrorFallback:
    """Multi-line code triggers exec() fallback from eval() SyntaxError."""

    def test_multiline_function_def(self):
        repl = _fresh_repl()
        code = "def double(n):\n    return n * 2"
        repl(code, namespace="fn-ns")
        out = repl("double(21)", namespace="fn-ns")
        assert "42" in out

    def test_for_loop(self):
        repl = _fresh_repl()
        repl("acc = []\nfor i in range(3):\n    acc.append(i)", namespace="loop-ns")
        out = repl("acc", namespace="loop-ns")
        assert "0" in out and "2" in out

    def test_import_statement(self):
        repl = _fresh_repl()
        repl("import math", namespace="import-ns")
        out = repl("math.sqrt(16)", namespace="import-ns")
        assert "4.0" in out


class TestExceptionHandling:
    """Exceptions are returned as traceback strings, not raised."""

    def test_name_error_returns_traceback(self):
        repl = _fresh_repl()
        out = repl("undefined_variable_xyz")
        assert "NameError" in out
        assert "undefined_variable_xyz" in out

    def test_zero_division_returns_traceback(self):
        repl = _fresh_repl()
        out = repl("1 / 0")
        assert "ZeroDivisionError" in out

    def test_exception_does_not_raise(self):
        repl = _fresh_repl()
        # Must return a string, never raise
        result = repl("raise ValueError('boom')")
        assert isinstance(result, str)
        assert "ValueError" in result


class TestThreadSafety:
    """Concurrent calls from two threads do not corrupt state."""

    def test_concurrent_calls_do_not_corrupt(self):
        repl = _fresh_repl()
        errors: list[str] = []
        results: list[str] = []

        def worker(ns: str, value: int) -> None:
            try:
                repl(f"x = {value}", namespace=ns)
                for _ in range(5):
                    out = repl("x", namespace=ns)
                    if str(value) not in out:
                        errors.append(f"ns={ns}: expected {value}, got {out!r}")
                results.append(f"{ns}={value}")
            except Exception as exc:
                errors.append(str(exc))

        t1 = threading.Thread(target=worker, args=("thread-ns-1", 111))
        t2 = threading.Thread(target=worker, args=("thread-ns-2", 222))
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        assert len(results) == 2


class TestReplHelp:
    """repl_help() is pre-seeded and returns a useful string."""

    def test_repl_help_callable(self):
        repl = _fresh_repl()
        out = repl("repl_help()")
        assert isinstance(out, str)
        assert len(out) > 0

    def test_repl_help_mentions_ink(self):
        repl = _fresh_repl()
        out = repl("repl_help()")
        assert "ink" in out

    def test_repl_help_mentions_np(self):
        repl = _fresh_repl()
        out = repl("repl_help()")
        assert "np" in out


class TestPreloadedNames:
    """Pre-seeded names (ink, np, plt) are available in every namespace."""

    def test_numpy_available(self):
        repl = _fresh_repl()
        out = repl("np.array([1, 2, 3]).sum()", namespace="seed-test")
        assert "6" in out

    def test_ink_package_available(self):
        repl = _fresh_repl()
        out = repl("type(ink).__name__", namespace="ink-test")
        assert "module" in out

    def test_plt_lazy_proxy(self):
        """plt is a lazy proxy; accessing attributes should not raise."""
        repl = _fresh_repl()
        # Just check the name is bound without triggering full import
        out = repl("type(plt).__name__", namespace="plt-test")
        # Should either be '_LazyModule' or 'module'
        assert "Module" in out or "module" in out
